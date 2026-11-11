"""Prepare the seven public datasets using the original split implementation."""
from __future__ import annotations

import io
import json
from pathlib import Path
import subprocess
import sys
from urllib.request import urlopen
from zipfile import ZipFile

from .pipeline import ROOT

UCI = {
    'adult': ('2/adult.zip', ['adult.data', 'adult.test']),
    'default': ('350/default+of+credit+card+clients.zip', ['default of credit card clients.xls']),
    'magic': ('159/magic+gamma+telescope.zip', ['magic04.data']),
    'shoppers': ('468/online+shoppers+purchasing+intention+dataset.zip', ['online_shoppers_intention.csv']),
}
SOURCES = {
    'bank': 'https://raw.githubusercontent.com/muellermarkus/cdtd/main/data/raw_data/bank/bank-additional-full.csv',
    'churn': 'https://raw.githubusercontent.com/muellermarkus/cdtd/main/data/raw_data/churn/Customer%20Churn.csv',
    'insurance': 'https://raw.githubusercontent.com/SJTU-DMTai/LCATS/main/data/insurance/insurance.csv',
}


def prepare(dataset, source=None):
    import pandas as pd

    directory = ROOT / 'data' / dataset
    if directory.exists():
        raise FileExistsError(f'Refusing to overwrite {directory}')
    if source and dataset == 'adult':
        raise ValueError('Adult uses both adult.data and adult.test; use the download or frozen splits.')
    directory.mkdir(parents=True)
    info = json.loads((ROOT / f'data/Info/{dataset}.json').read_text())
    if dataset in UCI:
        if source:
            (ROOT / info['data_path']).write_bytes(Path(source).read_bytes())
        else:
            archive, filenames = UCI[dataset]
            with urlopen('https://archive.ics.uci.edu/static/public/' + archive, timeout=120) as response:
                data = response.read()
            with ZipFile(io.BytesIO(data)) as zipfile:
                for name in filenames:
                    member = next(p for p in zipfile.namelist() if Path(p).name == name)
                    (directory / name).write_bytes(zipfile.read(member))
    else:
        if source:
            data = Path(source).read_bytes()
        else:
            with urlopen(SOURCES[dataset], timeout=120) as response:
                data = response.read()
        frame = pd.read_csv(io.BytesIO(data), sep=';' if dataset == 'bank' else ',')
        frame.columns = [' '.join(str(column).strip().split()) for column in frame.columns]
        for column in frame.select_dtypes(include='object').columns:
            frame[column] = frame[column].astype(str).str.strip()
        # This is the extension campaign's original column order and cleanup.
        schemas = {
            'bank': (
                ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate',
                 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed'],
                ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact',
                 'month', 'day_of_week', 'poutcome'], 'y'),
            'churn': (
                ['Call Failure', 'Subscription Length', 'Charge Amount', 'Seconds of Use',
                 'Frequency of use', 'Frequency of SMS', 'Distinct Called Numbers', 'Age', 'Customer Value'],
                ['Complains', 'Age Group', 'Tariff Plan', 'Status'], 'Churn'),
            'insurance': (['age', 'bmi', 'children'], ['sex', 'smoker', 'region'], 'charges'),
        }
        numeric, categorical, target = schemas[dataset]
        frame = frame[numeric + categorical + [target]].dropna().reset_index(drop=True)
        if dataset == 'bank':
            frame['y'] = frame['y'].map({'no': 0, 'yes': 1})
        for column in numeric:
            frame[column] = pd.to_numeric(frame[column], errors='raise')
        frame.to_csv(directory / f'{dataset}.csv', index=False)
    subprocess.run([sys.executable, str(ROOT / 'process_dataset.py'), '--dataname', dataset], cwd=ROOT, check=True)
