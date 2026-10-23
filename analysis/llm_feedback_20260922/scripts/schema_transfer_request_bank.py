#!/usr/bin/env python3
"""Fresh-schema compiler diagnostic for Default and Online Shoppers.

Rows are hand-constructed synthetic policy examples, never real dataset rows.
This is a schema-transfer stress test, not an external natural-policy corpus.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from request_condition_common import evaluate_dsl, make_predicate as pred, make_repair as repair


SCHEMAS = {
    "default": {
        "LIMIT_BAL":"numeric credit limit", "AGE":"numeric age", "BILL_AMT1":"numeric bill amount",
        "SEX":"categorical code", "EDUCATION":"categorical education code", "MARRIAGE":"categorical marital code",
        "PAY_0":"categorical repayment-status code",
    },
    "shoppers": {
        "PageValues":"numeric page value", "ExitRates":"numeric exit rate", "Administrative_Duration":"numeric duration",
        "Month":"month category", "VisitorType":"visitor category", "Weekend":"boolean category represented as text",
        "TrafficType":"categorical traffic code", "OperatingSystems":"categorical operating-system code",
        "Browser":"categorical browser code", "Region":"categorical region code",
    },
}

VOCABULARIES = {
    "default": {"SEX":["1","2"],"EDUCATION":["0","1","2","3"],"MARRIAGE":["0","1","2"],"PAY_0":["-2","0","1","2","3"]},
    "shoppers": {"Month":["Feb","May","Nov","Dec"],"VisitorType":["Returning_Visitor","New_Visitor"],"Weekend":["False","True"],
                 "TrafficType":["1","2","13"],"OperatingSystems":["1","2","3"],"Browser":["1","2","4"],"Region":["1","3","9"]},
}


def contracts(dataset: str) -> list[dict]:
    if dataset == "default":
        return [
            {
                "name":"married_limit", "a":"MARRIAGE equals 1", "r":"LIMIT_BAL must be at least 100000",
                "e":"EDUCATION equals 3", "f":"set LIMIT_BAL to 100000",
                "dsl":{"antecedent_all":[pred("MARRIAGE","eq",string_value="1")],"requirement_all":[pred("LIMIT_BAL","gte",number_value=100000)],"exception_all":[pred("EDUCATION","eq",string_value="3")],"repair":repair("LIMIT_BAL",number_value=100000)},
                "rows":{"satisfied":{"MARRIAGE":"1","EDUCATION":"2","LIMIT_BAL":150000},"violated":{"MARRIAGE":"1","EDUCATION":"2","LIMIT_BAL":50000},"not_applicable":{"MARRIAGE":"2","EDUCATION":"2","LIMIT_BAL":50000},"excepted":{"MARRIAGE":"1","EDUCATION":"3","LIMIT_BAL":50000}},
            },
            {
                "name":"education_age", "a":"EDUCATION is one of 1 or 2", "r":"AGE must not be below 21",
                "e":"PAY_0 equals -2", "f":"set AGE to 21",
                "dsl":{"antecedent_all":[pred("EDUCATION","in",string_list=["1","2"])],"requirement_all":[pred("AGE","gte",number_value=21)],"exception_all":[pred("PAY_0","eq",string_value="-2")],"repair":repair("AGE",number_value=21)},
                "rows":{"satisfied":{"EDUCATION":"1","PAY_0":"0","AGE":30},"violated":{"EDUCATION":"2","PAY_0":"0","AGE":18},"not_applicable":{"EDUCATION":"3","PAY_0":"0","AGE":18},"excepted":{"EDUCATION":"1","PAY_0":"-2","AGE":18}},
            },
            {
                "name":"delay_bill_cap", "a":"PAY_0 is either 2 or 3", "r":"BILL_AMT1 must be no greater than 200000",
                "e":"MARRIAGE equals 0", "f":"set BILL_AMT1 to 200000",
                "dsl":{"antecedent_all":[pred("PAY_0","in",string_list=["2","3"])],"requirement_all":[pred("BILL_AMT1","lte",number_value=200000)],"exception_all":[pred("MARRIAGE","eq",string_value="0")],"repair":repair("BILL_AMT1",number_value=200000)},
                "rows":{"satisfied":{"PAY_0":"2","MARRIAGE":"1","BILL_AMT1":150000},"violated":{"PAY_0":"3","MARRIAGE":"1","BILL_AMT1":300000},"not_applicable":{"PAY_0":"0","MARRIAGE":"1","BILL_AMT1":300000},"excepted":{"PAY_0":"2","MARRIAGE":"0","BILL_AMT1":300000}},
            },
            {
                "name":"joint_category", "a":"SEX equals 2 and MARRIAGE equals 2", "r":"EDUCATION must not equal 0",
                "e":"AGE is at least 65", "f":"set EDUCATION to 2",
                "dsl":{"antecedent_all":[pred("SEX","eq",string_value="2"),pred("MARRIAGE","eq",string_value="2")],"requirement_all":[pred("EDUCATION","neq",string_value="0")],"exception_all":[pred("AGE","gte",number_value=65)],"repair":repair("EDUCATION",string_value="2")},
                "rows":{"satisfied":{"SEX":"2","MARRIAGE":"2","EDUCATION":"2","AGE":40},"violated":{"SEX":"2","MARRIAGE":"2","EDUCATION":"0","AGE":40},"not_applicable":{"SEX":"1","MARRIAGE":"2","EDUCATION":"0","AGE":40},"excepted":{"SEX":"2","MARRIAGE":"2","EDUCATION":"0","AGE":70}},
            },
        ]
    return [
        {
            "name":"new_visitor_value", "a":"VisitorType equals New_Visitor", "r":"PageValues must be at least 5",
            "e":"Weekend equals True", "f":"set PageValues to 5",
            "dsl":{"antecedent_all":[pred("VisitorType","eq",string_value="New_Visitor")],"requirement_all":[pred("PageValues","gte",number_value=5)],"exception_all":[pred("Weekend","eq",string_value="True")],"repair":repair("PageValues",number_value=5)},
            "rows":{"satisfied":{"VisitorType":"New_Visitor","Weekend":"False","PageValues":10},"violated":{"VisitorType":"New_Visitor","Weekend":"False","PageValues":0},"not_applicable":{"VisitorType":"Returning_Visitor","Weekend":"False","PageValues":0},"excepted":{"VisitorType":"New_Visitor","Weekend":"True","PageValues":0}},
        },
        {
            "name":"holiday_exit", "a":"Month belongs to November or December, encoded as Nov or Dec", "r":"ExitRates must be at most 0.6",
            "e":"TrafficType equals 13", "f":"set ExitRates to 0.6",
            "dsl":{"antecedent_all":[pred("Month","in",string_list=["Nov","Dec"])],"requirement_all":[pred("ExitRates","lte",number_value=.6)],"exception_all":[pred("TrafficType","eq",string_value="13")],"repair":repair("ExitRates",number_value=.6)},
            "rows":{"satisfied":{"Month":"Nov","TrafficType":"2","ExitRates":.2},"violated":{"Month":"Dec","TrafficType":"2","ExitRates":.8},"not_applicable":{"Month":"May","TrafficType":"2","ExitRates":.8},"excepted":{"Month":"Nov","TrafficType":"13","ExitRates":.8}},
        },
        {
            "name":"platform_browser", "a":"OperatingSystems is 1 or 2", "r":"Browser must be one of 1 or 2",
            "e":"Region equals 9", "f":"set Browser to 2",
            "dsl":{"antecedent_all":[pred("OperatingSystems","in",string_list=["1","2"])],"requirement_all":[pred("Browser","in",string_list=["1","2"])],"exception_all":[pred("Region","eq",string_value="9")],"repair":repair("Browser",string_value="2")},
            "rows":{"satisfied":{"OperatingSystems":"1","Browser":"2","Region":"1"},"violated":{"OperatingSystems":"2","Browser":"4","Region":"1"},"not_applicable":{"OperatingSystems":"3","Browser":"4","Region":"1"},"excepted":{"OperatingSystems":"1","Browser":"4","Region":"9"}},
        },
        {
            "name":"weekend_admin", "a":"Weekend equals True and VisitorType equals Returning_Visitor", "r":"Administrative_Duration must be at least 10",
            "e":"Month equals Feb", "f":"set Administrative_Duration to 10",
            "dsl":{"antecedent_all":[pred("Weekend","eq",string_value="True"),pred("VisitorType","eq",string_value="Returning_Visitor")],"requirement_all":[pred("Administrative_Duration","gte",number_value=10)],"exception_all":[pred("Month","eq",string_value="Feb")],"repair":repair("Administrative_Duration",number_value=10)},
            "rows":{"satisfied":{"Weekend":"True","VisitorType":"Returning_Visitor","Month":"May","Administrative_Duration":20},"violated":{"Weekend":"True","VisitorType":"Returning_Visitor","Month":"May","Administrative_Duration":0},"not_applicable":{"Weekend":"False","VisitorType":"Returning_Visitor","Month":"May","Administrative_Duration":0},"excepted":{"Weekend":"True","VisitorType":"Returning_Visitor","Month":"Feb","Administrative_Duration":0}},
        },
    ]


def surfaces(item: dict) -> list[str]:
    return [
        f"Applicability: {item['a']}. Mandatory outcome: {item['r']}. Exclusion: {item['e']}. Correction for an actual breach: {item['f']}.",
        f"Ignore an otherwise covered row when {item['e']}. For all remaining rows, whenever {item['a']}, ensure that {item['r']}; if not, {item['f']}.",
        f"Policy card — scope [{item['a']}]; compliant state [{item['r']}]; explicit waiver [{item['e']}]; authorized remediation [{item['f']}].",
    ]


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--dataset",choices=["default","shoppers"],required=True); parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    requests=[]; cases=[]
    for item in contracts(args.dataset):
        for surface_id,text in enumerate(surfaces(item)):
            request_id=f"{args.dataset}_{item['name']}_s{surface_id}"
            requests.append({"request_id":request_id,"family":item["name"],"surface_id":surface_id,"text":text,"oracle_dsl":item["dsl"]})
            for status,row in item["rows"].items():
                observed=evaluate_dsl(row,item["dsl"])
                if observed!=status: raise AssertionError((item["name"],status,observed))
                cases.append({"case_id":f"fresh_{len(cases):04d}","row_id":f"{item['name']}_{status}","pair_id":None,"trajectory_id":"hand_synthetic",
                              "t":0.0,"dataset":args.dataset,"record":row,"request_id":request_id,"request_family":item["name"],
                              "request_text":text,"design":f"fresh_surface_{surface_id}","gold_status":status,
                              "gold_action":"repair_candidate" if status=="violated" else "no_action","privacy":"synthetic_preview_only"})
    for name,rows in (("requests.jsonl",requests),("cases.jsonl",cases)):
        with (args.output/name).open("w") as handle:
            for row in rows: handle.write(json.dumps(row)+"\n")
    (args.output/"schema.json").write_text(json.dumps(SCHEMAS[args.dataset],indent=2))
    (args.output/"categorical_vocabulary.json").write_text(json.dumps(VOCABULARIES[args.dataset],indent=2))
    metadata={"status":"executed","dataset":args.dataset,"contracts":len(contracts(args.dataset)),"requests":len(requests),"cases":len(cases),
              "scope":"fresh-schema controlled diagnostic; not external natural-policy corpus","privacy":"hand-constructed synthetic rows only"}
    (args.output/"run_metadata.json").write_text(json.dumps(metadata,indent=2)); print(json.dumps(metadata,indent=2))

if __name__=="__main__": main()
