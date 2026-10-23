#!/usr/bin/env python3
"""Deterministic type repair and property-based audit of compiled contracts."""
from __future__ import annotations
import argparse, itertools, json
from pathlib import Path
import numpy as np
import pandas as pd
from request_condition_common import evaluate_dsl


def typed_copy(dsl: dict, vocabulary: dict[str,list[str]]) -> dict:
    fixed=json.loads(json.dumps(dsl))
    for part in ("antecedent_all","requirement_all","exception_all"):
        for predicate in fixed[part]:
            if predicate["column"] not in vocabulary: continue
            values=vocabulary[predicate["column"]]
            if predicate["value_type"]=="number":
                raw=predicate.get("number_value")
                text=str(int(raw)) if float(raw).is_integer() else str(raw)
                if text in values:
                    predicate.update(value_type="string",string_value=text,number_value=0,string_list=[])
    repair=fixed["repair"]
    if repair["column"] in vocabulary and repair["value_type"]=="number":
        raw=repair.get("number_value"); text=str(int(raw)) if float(raw).is_integer() else str(raw)
        if text in vocabulary[repair["column"]]:
            repair.update(value_type="string",string_value=text,number_value=0)
    return fixed


def candidates(column: str, oracle: dict, predicted: dict, vocabulary: dict, observed_rows: list[dict]) -> list[object]:
    if column in vocabulary:
        return list(vocabulary[column])+["__other__"]
    values={0.0}
    for dsl in (oracle,predicted):
        for part in ("antecedent_all","requirement_all","exception_all"):
            for predicate in dsl.get(part,[]):
                if predicate.get("column")==column and predicate.get("value_type")=="number":
                    value=float(predicate.get("number_value",0)); values.update((value-1,value,value+1))
    for row in observed_rows:
        if column in row:
            try: values.add(float(row[column]))
            except (TypeError,ValueError): pass
    return sorted(values)


def equivalence(oracle: dict, predicted: dict, vocabulary: dict, rows: list[dict], limit: int=2000) -> tuple[float,int]:
    columns=sorted({p["column"] for d in (oracle,predicted) for part in ("antecedent_all","requirement_all","exception_all") for p in d.get(part,[])})
    grids=[candidates(column,oracle,predicted,vocabulary,rows) for column in columns]
    total=int(np.prod([len(values) for values in grids])) if grids else 1
    equal=count=0
    for values in itertools.islice(itertools.product(*grids),limit):
        row=dict(zip(columns,values)); count+=1
        try: gold=evaluate_dsl(row,oracle)
        except Exception: gold="oracle_error"
        try: guess=evaluate_dsl(row,predicted)
        except Exception: guess="execution_error"
        equal+=gold==guess
    return equal/max(count,1),min(total,limit)


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--runs",type=Path,required=True); parser.add_argument("--bank-root",type=Path,required=True); parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    runs=pd.read_csv(args.runs); request_rows=[]; run_rows=[]
    for run in runs.itertuples(index=False):
        run_dir=Path(run.run_dir); dataset="default" if "schema_default" in run_dir.name else "shoppers"
        bank=args.bank_root/f"schema_transfer_bank_{dataset}_20260928"
        requests={r["request_id"]:r for r in (json.loads(line) for line in (bank/"requests.jsonl").read_text().splitlines() if line)}
        cases=[json.loads(line) for line in (bank/"cases.jsonl").read_text().splitlines() if line]
        vocab=json.loads((bank/"categorical_vocabulary.json").read_text()); compiled=pd.read_csv(run_dir/"compiled_requests.csv")
        raw_correct=typed_correct=raw_equiv=typed_equiv=0; total_cases=total_requests=0
        for item in compiled.itertuples(index=False):
            if not isinstance(item.predicted_dsl,str) or not item.predicted_dsl: continue
            predicted=json.loads(item.predicted_dsl); fixed=typed_copy(predicted,vocab); oracle=requests[item.request_id]["oracle_dsl"]
            request_cases=[case for case in cases if case["request_id"]==item.request_id]; observed=[case["record"] for case in request_cases]
            raw_property,grid_size=equivalence(oracle,predicted,vocab,observed); fixed_property,_=equivalence(oracle,fixed,vocab,observed)
            raw_case=sum(evaluate_dsl(case["record"],predicted)==case["gold_status"] for case in request_cases)/len(request_cases)
            fixed_case=sum(evaluate_dsl(case["record"],fixed)==case["gold_status"] for case in request_cases)/len(request_cases)
            request_rows.append({"job_id":run.job_id,"dataset":dataset,"model":run.model,"strategy":run.compile_strategy,"request_id":item.request_id,"grid_size":grid_size,"raw_case_accuracy":raw_case,"typed_case_accuracy":fixed_case,"raw_property_agreement":raw_property,"typed_property_agreement":fixed_property,"type_repair_changed":json.dumps(predicted,sort_keys=True)!=json.dumps(fixed,sort_keys=True)})
            raw_correct+=raw_case*len(request_cases); typed_correct+=fixed_case*len(request_cases); total_cases+=len(request_cases); raw_equiv+=raw_property; typed_equiv+=fixed_property; total_requests+=1
        run_rows.append({"job_id":run.job_id,"dataset":dataset,"model":run.model,"strategy":run.compile_strategy,"requests":total_requests,"raw_case_accuracy":raw_correct/max(total_cases,1),"typed_case_accuracy":typed_correct/max(total_cases,1),"raw_property_agreement":raw_equiv/max(total_requests,1),"typed_property_agreement":typed_equiv/max(total_requests,1)})
    request_frame=pd.DataFrame(request_rows); run_frame=pd.DataFrame(run_rows); request_frame.to_csv(args.output/"request_audit.csv",index=False); run_frame.to_csv(args.output/"run_audit.csv",index=False)
    summary=run_frame.groupby(["dataset","model","strategy"]).agg(runs=("job_id","size"),raw_case=("raw_case_accuracy","mean"),typed_case=("typed_case_accuracy","mean"),raw_property=("raw_property_agreement","mean"),typed_property=("typed_property_agreement","mean"),worst_typed_property=("typed_property_agreement","min")).reset_index(); summary.to_csv(args.output/"summary.csv",index=False)
    lines=["# Kiểm tra kiểu dữ liệu và phản ví dụ cho bộ biên dịch","","Mỗi contract được kiểm tra trên lưới giá trị sinh tự động, ngoài bốn hàng đánh giá ban đầu.","","| Bảng | Mô hình | Cách nhắc | Số lần | Đúng trên ca gốc | Sau kiểm tra kiểu | Đồng thuận trên lưới | Sau kiểm tra kiểu | Tệ nhất sau sửa |","| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in summary.itertuples(index=False): lines.append(f"| {r.dataset} | {r.model} | {r.strategy} | {r.runs} | {r.raw_case:.4f} | {r.typed_case:.4f} | {r.raw_property:.4f} | {r.typed_property:.4f} | {r.worst_typed_property:.4f} |")
    (args.output/"RESULTS.md").write_text("\n".join(lines)+"\n"); print(summary.to_string(index=False),flush=True)

if __name__=="__main__": main()
