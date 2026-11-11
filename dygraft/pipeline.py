"""Portable orchestration of the frozen sampling and selection protocol."""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('adult', 'bank', 'churn', 'default', 'insurance', 'magic', 'shoppers')
QUALITY = ('--bins 16 --pair-bins 4 --max-pairs 24 --target-pair-quota 8 '
           '--pair-weight 1.5 --coverage-weight .30').split()


def configuration(dataset):
    return json.loads((ROOT / f'configs/datasets/{dataset}.json').read_text())


def run(module, arguments, output, *, env=None):
    command = [sys.executable, '-u', '-m', f'dygraft.{module}', *map(str, arguments)]
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(ROOT) + os.pathsep + environment.get('PYTHONPATH', '')
    if env:
        environment.update(env)
    with (output / 'commands.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(command) + '\n')
    started = time.perf_counter()
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    return time.perf_counter() - started


def sample(dataset, seed, checkpoint, output, arm, rows=None, model_config=None):
    config = configuration(dataset)
    settings = dict(config['sampling'])
    native = arm != 'dygraft'
    if native:
        settings.update(projection_strength=0, distribution_transport_mode='none',
                        categorical_guidance_mode='none')
    settings.update(repo=ROOT, dataset=dataset, seed=seed,
                    graph=ROOT / f'configs/graphs/{dataset}.json',
                    config=model_config or ROOT / f'configs/models/{dataset}.json',
                    checkpoint=checkpoint,
                    samples=rows or (config['output_rows'] if arm == 'backbone' else config['candidate_rows']),
                    latency_pool='native' if native else 'routed', output=output / 'pool')
    # Read the actual parser definitions, so Boolean options retain their
    # original store_true semantics rather than becoming string arguments.
    tree = ast.parse((ROOT / 'dygraft/sampling.py').read_text())
    arguments = []
    covered = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'add_argument' and node.args
                and isinstance(node.args[0], ast.Constant)):
            continue
        option = node.args[0].value
        if not isinstance(option, str) or not option.startswith('--'):
            continue
        key = option[2:].replace('-', '_')
        if key not in settings or settings[key] is None:
            continue
        covered.add(key)
        value = settings[key]
        if isinstance(value, bool):
            if value:
                arguments.append(option)
        else:
            arguments.append(f'{option}={value}')
    missing = set(settings) - covered - {k for k, v in settings.items() if v is None}
    if missing:
        raise ValueError(f'Unknown sampling settings: {sorted(missing)}')
    run('sampling', arguments, output)
    filename = 'unconditional.csv' if native else 'dense_factor.csv'
    return output / 'pool/samples' / filename


def select(dataset, seed, pool, output, screening='on', stage='full'):
    config = configuration(dataset)
    base = ['--repo', ROOT, '--dataset', dataset, '--pool', pool, '--seed', seed]
    common = [*base, '--backbone-pool', pool, '--count', config['output_rows']]
    timings = {}
    timings['anchor_seconds'] = run('anchoring', [
        *common, *QUALITY, *('--mode rake_marginal_pair_select --rake-iterations 5 '
                            '--rake-damping .5 --temperature .5 --tail-mix .10 --draws 128').split(),
        '--output', output / 'anchor',
    ], output)
    final = output / 'anchor/selected.csv'
    if stage == 'anchor':
        return final, timings
    timings['support_seconds'] = run('screened_support', [
        *common, *QUALITY, '--anchor-index', output / 'anchor/selected_indices.csv',
        *('--swap-fraction .10 --quality-relative-tolerance .08 --neighbors 32 '
          '--addition-shortlist 8 --removal-shortlist 16').split(),
        '--output', output / 'support_trace',
    ], output, env={'DYGRAFT_SCREEN_SUPPORT': '1' if screening == 'on' else '0'})
    timings['calibration_seconds'] = run('calibration', [
        *base, '--repair-dir', output / 'support_trace', '--folds', 5,
        '--checkpoint-stride', 8, '--output', output / 'policies',
    ], output)
    final = output / 'policies/cv_joint_1se/selected.csv'
    if stage == 'support':
        return final, timings
    timings['refinement_seconds'] = run('cached_refinement', [
        *common, *QUALITY, '--graph', ROOT / f'configs/graphs/{dataset}.json',
        '--include-edge-ids', config['sampling']['include_edge_ids'], '--include-roles', 'actuate',
        '--anchor-index', output / 'policies/cv_joint_1se/selected_indices.csv',
        *('--selection-priority quality --swap-temperature .75 --uniform-mix .10 '
          '--coverage-boost 2 --hard-boost 2 --swap-fraction .0025 --quality-relative-tolerance .015 '
          '--required-graph-gain .05 --graph-activation-threshold .001 --geometric-coverage-weight 20 '
          '--geometric-coverage-k 16 --geometric-coverage-tolerance 0 --integrated-coverage-weight 20 '
          '--integrated-coverage-tolerance .005 --integrated-coverage-shortlist 8 '
          '--integrated-coverage-folds 2 --draws 256').split(), '--output', output / 'refine',
    ], output)
    return output / 'refine/selected.csv', timings


def evaluate(dataset, seed, samples, output):
    samples = [f'{label}={Path(path).resolve()}' for label, path in (value.split('=', 1) for value in samples)]
    output.mkdir(parents=True, exist_ok=False)
    common = ['--repo', ROOT, '--dataset', dataset, '--seed', seed,
              '--graph', ROOT / f'configs/graphs/{dataset}_evaluation.json']
    arms = [v for sample in samples for v in ('--sample', sample)]
    evaluator = 'evaluation_mixed' if dataset in {'bank', 'churn', 'insurance'} else 'evaluation'
    run(evaluator, [*common, *arms, '--max-rows', 4096, '--output', output / 'quality'], output)
    run('query_fidelity', [*common, *arms, '--max-queries', 120, '--output', output / 'sql'], output)


def execute(args):
    config = configuration(args.dataset)
    seed = args.seed if args.seed is not None else config['main_seeds'][0]
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    checkpoint = (getattr(args, 'checkpoint', None) or ROOT / config['training']['checkpoint']).resolve()
    if args.command == 'run' and not checkpoint.is_file():
        raise FileNotFoundError(f'{checkpoint}: extract the checkpoint asset or train a backbone first')
    output.mkdir(parents=True)
    started = time.perf_counter()
    timings = {}
    if args.command == 'run':
        pool = sample(args.dataset, seed, checkpoint, output, args.arm,
                      model_config=args.model_config.resolve() if args.model_config else None)
    else:
        pool = args.pool.resolve()
    if args.command == 'run' and args.arm == 'backbone':
        final = pool
    else:
        final, timings = select(args.dataset, seed, pool, output, args.screening, args.stage)
    shutil.copyfile(final, output / 'synthetic.csv')
    with (output / 'synthetic.csv').open(newline='') as stream:
        count = sum(1 for _ in csv.reader(stream)) - 1
    if count != config['output_rows']:
        raise RuntimeError(f'Expected {config["output_rows"]} output rows, found {count}')
    timings['command_seconds'] = time.perf_counter() - started
    (output / 'run.json').write_text(json.dumps({
        'dataset': args.dataset, 'seed': seed, 'output_rows': count,
        'screening': args.screening, 'stage': args.stage,
        'arm': getattr(args, 'arm', 'existing_pool'), **timings,
    }, indent=2) + '\n')
    if args.evaluate:
        evaluate(args.dataset, seed, [f'{getattr(args, "arm", "DyGRAFT")}={output / "synthetic.csv"}'], output / 'evaluation')
    print(f'Synthetic table: {output / "synthetic.csv"}', flush=True)


def train(args):
    config = configuration(args.dataset)
    training = config['training']
    if training['protocol'] == 'official':
        command = [sys.executable, 'main.py', '--dataname', args.dataset, '--mode', 'train',
                   '--no_wandb', '--deterministic', '--exp_name', 'dygraft']
        subprocess.run(command, cwd=ROOT, check=True)
    else:
        output = args.output or ROOT / f'checkpoints/{args.dataset}'
        if output.exists():
            raise FileExistsError(output)
        output.mkdir(parents=True)
        run('training', [
            '--repo', ROOT, '--dataset', args.dataset,
            '--base-config', ROOT / f'configs/models/{args.dataset}.json',
            '--epochs', training['epochs'], '--num-timesteps', training['timesteps'],
            '--seed', training['seed'], '--save-every', 2000, '--output', output,
            '--regression-target-mode', config['sampling']['regression_target_mode'],
        ], output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('run', 'select'):
        sub = commands.add_parser(name, help='Generate and select a table' if name == 'run' else 'Select from a fixed candidate pool on CPU')
        sub.add_argument('--dataset', choices=DATASETS, required=True)
        sub.add_argument('--seed', type=int)
        sub.add_argument('--output', type=Path, required=True)
        sub.add_argument('--screening', choices=('on', 'off'), default='on')
        sub.add_argument('--stage', choices=('anchor', 'support', 'full'), default='full')
        sub.add_argument('--evaluate', action='store_true')
        if name == 'run':
            sub.add_argument('--checkpoint', type=Path)
            sub.add_argument('--model-config', type=Path)
            sub.add_argument('--arm', choices=('dygraft', 'backbone', 'backbone-selector'), default='dygraft')
        else:
            sub.add_argument('--pool', type=Path, required=True)
        sub.set_defaults(function=execute)
    sub = commands.add_parser('evaluate', help='Evaluate CSV tables with the paper metrics')
    sub.add_argument('--dataset', choices=DATASETS, required=True)
    sub.add_argument('--seed', type=int, required=True)
    sub.add_argument('--sample', action='append', required=True, help='METHOD=CSV_PATH (repeatable)')
    sub.add_argument('--output', type=Path, required=True)
    sub.set_defaults(function=lambda args: evaluate(args.dataset, args.seed, args.sample, args.output.resolve()))
    sub = commands.add_parser('prepare', help='Download and prepare a public benchmark')
    sub.add_argument('--dataset', choices=(*DATASETS, 'all'), required=True)
    sub.add_argument('--source', type=Path, help='Optional local raw CSV/XLS instead of downloading')
    def prepare(args):
        from .data import prepare as prepare_data
        if args.dataset == 'all' and args.source:
            parser.error('--source requires one dataset')
        for name in DATASETS if args.dataset == 'all' else [args.dataset]:
            prepare_data(name, args.source)
    sub.set_defaults(function=prepare)
    sub = commands.add_parser('train', help='Train a backbone with the dataset-specific training loop')
    sub.add_argument('--dataset', choices=DATASETS, required=True)
    sub.add_argument('--output', type=Path, help='Output directory for the frozen-loop trainer')
    sub.set_defaults(function=train)
    args = parser.parse_args()
    args.function(args)
