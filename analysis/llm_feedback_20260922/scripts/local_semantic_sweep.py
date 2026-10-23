#!/usr/bin/env python3
"""Evaluate cached local instruction models on synthetic diffusion previews."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from llm_feedback_audit import SCHEMA
from semantic_llm_sweep import SEMANTIC_RUBRIC, STATIC_RULES


RELATIONS = [
    "education_fd", "numeric_domain", "spouse_consistency",
    "age_education_plausibility", "missingness_coupling", "none",
]

RELATION_RUBRIC = {
    "education_fd": "education and education.num must denote the same attained education level",
    "numeric_domain": "capital gain/loss cannot be negative and hours per week must be positive",
    "spouse_consistency": "household spouse labels must agree with sex and marital-status coding",
    "age_education_plausibility": "very advanced attained education is impossible at a very young age",
    "missingness_coupling": "the '?' sentinel for workclass and occupation must occur together",
}

SYSTEM = (
    "You are a conservative tabular-data verifier. Follow the requested output contract literally. "
    "Return only a JSON object; never write Python, prose, Markdown, or a rule implementation. "
    "Rarity is not invalidity and uncertainty should be an abstention."
)


def prompt_for(case: dict, condition: str) -> str:
    payload = {
        "schema": SCHEMA,
        "diffusion_t": case["t"],
        "synthetic_record": case["record"],
    }
    if condition == "semantic_rubric":
        payload["semantic_rubric"] = SEMANTIC_RUBRIC
    elif condition == "static_rules":
        payload["static_validated_rules"] = STATIC_RULES
    elif condition == "temporal_pair":
        payload["previous_synthetic_record"] = case.get("previous_record")
    elif condition in {"focused_label", "focused_rubric", "focused_temporal"}:
        relation = case["intended_relation"]
        payload["candidate_relation_to_verify"] = relation
        if condition in {"focused_rubric", "focused_temporal"}:
            payload["relation_description"] = RELATION_RUBRIC.get(
                relation,
                "No specific candidate relation is active; do not invent a violation.",
            )
        if condition == "focused_temporal":
            payload["previous_synthetic_record"] = case.get("previous_record")
    contract = (
        "Return only one JSON object with keys status, relation, confidence. "
        "status must be violated, satisfied, or abstain. relation must be one of "
        + ", ".join(RELATIONS)
        + ". Use violated only for a concrete inconsistency. Rare is not invalid. "
        "If valid use satisfied and relation none; if uncertain abstain. "
        "Example output: {\"status\":\"satisfied\",\"relation\":\"none\",\"confidence\":0.9}."
    )
    if condition in {"focused_label", "focused_rubric", "focused_temporal"}:
        contract += (
            " Verify only candidate_relation_to_verify. If it is violated, relation must equal that candidate; "
            "otherwise relation must be none. Do not report a different relation."
        )
    return contract + "\nINPUT:\n" + json.dumps(payload, ensure_ascii=False)


def parse_object(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--condition", choices=[
        "typed_contract", "semantic_rubric", "static_rules", "temporal_pair",
        "focused_label", "focused_rubric", "focused_temporal",
    ], required=True)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--load-in-8bit", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = [json.loads(line) for line in args.cases.read_text().splitlines() if line.strip()]
    if args.max_cases:
        cases = cases[: args.max_cases]

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {
        "local_files_only": True,
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
    }
    if args.load_in_8bit:
        model_kwargs.update(
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map="auto",
        )
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).to("cuda").eval()

    rows, raw_rows = [], []
    input_tokens = output_tokens = 0
    started = time.perf_counter()
    for start in range(0, len(cases), args.batch_size):
        batch = cases[start : start + args.batch_size]
        chats = [[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt_for(case, args.condition)},
        ] for case in batch]
        rendered = []
        for chat in chats:
            try:
                rendered.append(tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=False))
            except TypeError:
                rendered.append(tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))
        encoded = tokenizer(rendered, return_tensors="pt", padding=True, truncation=True, max_length=4096).to("cuda")
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_width = encoded["input_ids"].shape[1]
        texts = tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=True)
        input_tokens += int(encoded["attention_mask"].sum().item())
        output_tokens += int((generated[:, prompt_width:] != tokenizer.pad_token_id).sum().item())
        for case, text in zip(batch, texts):
            decision = parse_object(text)
            status = str(decision.get("status", "missing")).lower()
            relation = str(decision.get("relation", "none")).lower()
            predicted = relation if status == "violated" and relation in RELATIONS[:-1] else ""
            raw_rows.append({"case_id": case["case_id"], "raw_output": text, "parsed": decision})
            rows.append({
                "case_id": case["case_id"], "semantic_key": case["semantic_key"], "pair_id": case["pair_id"],
                "t": case["t"], "control_type": case["control_type"], "intended_relation": case["intended_relation"],
                "gold": "|".join(sorted(case.get("gold_relations", []))), "repeat": 0,
                "condition": args.condition, "model": args.model_tag, "status": status, "predicted": predicted,
                "confidence": decision.get("confidence"), "columns": "", "rationale_code": "", "action": "",
                "missing_decision": not bool(decision), "batch_error": "",
            })

    details = pd.DataFrame(rows)
    details.to_csv(args.output / "decisions.csv", index=False)
    with (args.output / "responses.jsonl").open("w") as handle:
        for item in raw_rows:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    tp = fp = fn = 0
    for row in rows:
        gold = set(filter(None, row["gold"].split("|")))
        pred = {row["predicted"]} if row["predicted"] else set()
        tp += len(gold & pred); fp += len(pred - gold); fn += len(gold - pred)
    elapsed = time.perf_counter() - started
    summary = {
        "status": "executed", "condition": args.condition, "model": args.model_tag,
        "cases": len(cases), "repeats": 1, "successful_calls": len(cases), "failed_calls": 0,
        "tp": tp, "fp": fp, "fn": fn,
        "micro_precision": tp / (tp + fp) if tp + fp else None,
        "micro_recall": tp / (tp + fn) if tp + fn else None,
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens, "api_latency_seconds": elapsed,
        "privacy_boundary": "Fully local inference over synthetic previews; no API and no real rows.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
