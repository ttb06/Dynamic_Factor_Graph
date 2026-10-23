#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
def main():
 p=argparse.ArgumentParser(); p.add_argument("--input-root",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
 paths=sorted(a.input_root.glob("cross_active_*_seed*/active_set_metrics.csv"));
 if not paths: raise FileNotFoundError("no cross active-set metrics")
 d=pd.concat([pd.read_csv(path).assign(source=str(path.parent)) for path in paths],ignore_index=True)
 # Retry jobs reuse the same seed and paired noise.  Keep the newest run rather
 # than treating it as an independent replication.
 d=d.sort_values("source").drop_duplicates(["dataset","guidance","seed","policy_id","method"],keep="last")
 d.to_csv(a.output/"all_metrics.csv",index=False)
 s=d.groupby(["dataset","guidance","method"]).agg(seeds=("seed","nunique"),policies=("policy_id","nunique"),violation=("conditional_violation_rate","mean"),reduction=("violation_reduction","mean"),inactive_drift=("inactive_target_drift","mean"),active_drift=("active_target_drift","mean"),exception_drift=("exception_target_drift","mean"),applicability_flip=("applicability_flip_rate","mean"),numeric_ks=("numeric_ks_to_base","mean"),categorical_tv=("categorical_tv_to_base","mean"),edit_rate=("edit_rate","mean")).reset_index(); s.to_csv(a.output/"summary.csv",index=False)
 lines=["# Tập tác động theo từng hàng trên hai bảng dữ liệu","","Yêu cầu là chính sách sinh có kiểm soát, không được xem là quy luật tự nhiên của dữ liệu.","","Bảng tổng hợp có thể chứa số seed khác nhau giữa các method. Cột `Seed` được ghi rõ; mọi kết luận so sánh method mới ở phần sau chỉ dùng cùng seed, cùng hệ số và cùng policy.","","| Bảng | Hệ số | Cách tác động | Seed | Vi phạm | Mức giảm | Lệch hàng ngoài phạm vi | Lệch hàng ngoại lệ | Đổi phạm vi | KS | TV | Tỷ lệ sửa |","| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
 for r in s.itertuples(index=False): lines.append(f"| {r.dataset} | {r.guidance:g} | {r.method} | {r.seeds:d} | {r.violation:.4f} | {r.reduction:.4f} | {r.inactive_drift:.4f} | {r.exception_drift:.4f} | {r.applicability_flip:.4f} | {r.numeric_ks:.4f} | {r.categorical_tv:.4f} | {r.edit_rate:.4f} |")

 # The four-state screen was intentionally run on one seed.  Restrict its
 # comparisons to exactly the same seed/guidance/policy instead of comparing
 # it with the three-seed means above.
 four="preview_violation_selective"
 keys=d.loc[d.method.eq(four),["dataset","guidance","seed"]].drop_duplicates()
 matched=d.merge(keys,on=["dataset","guidance","seed"],how="inner")
 wanted=["no_actuation","preview_selective",four,"preview_violation_then_certifier","selective_postrepair"]
 matched=matched[matched.method.isin(wanted)]
 if not matched.empty:
  ms=matched.groupby(["dataset","guidance","method"]).agg(seeds=("seed","nunique"),policies=("policy_id","nunique"),violation=("conditional_violation_rate","mean"),reduction=("violation_reduction","mean"),applicability_flip=("applicability_flip_rate","mean"),numeric_ks=("numeric_ks_to_base","mean"),categorical_tv=("categorical_tv_to_base","mean"),edit_rate=("edit_rate","mean")).reset_index()
  ms.to_csv(a.output/"matched_four_state_summary.csv",index=False)
  lines += ["","## So sánh matched cho four-state screen","","| Bảng | Hệ số | Cách tác động | Seed | Policy | Vi phạm | Mức giảm | Đổi phạm vi | KS | TV | Tỷ lệ sửa |","| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
  for r in ms.itertuples(index=False): lines.append(f"| {r.dataset} | {r.guidance:g} | {r.method} | {r.seeds:d} | {r.policies:d} | {r.violation:.4f} | {r.reduction:.4f} | {r.applicability_flip:.4f} | {r.numeric_ks:.4f} | {r.categorical_tv:.4f} | {r.edit_rate:.4f} |")
 (a.output/"RESULTS.md").write_text("\n".join(lines)+"\n"); print(f"loaded {len(paths)} runs")
if __name__=="__main__": main()
