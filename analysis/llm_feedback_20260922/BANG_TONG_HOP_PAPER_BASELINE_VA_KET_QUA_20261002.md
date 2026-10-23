# Bảng tổng hợp paper-reported baselines, kết quả tự chạy và kết quả của dự án

Ngày đóng băng: 02/10/2026

## 1. Quy ước

Ba loại evidence được tách hoàn toàn:

- **Paper-reported:** số do bài báo công bố, theo dataset, split và metric của bài đó.
- **Official self-run:** chạy code tác giả trên server hiện tại. Đây là kết quả tự chạy nhưng không mặc nhiên là reproduction đầy đủ nhiều seed.
- **Our result:** analysis hoặc method/proxy chạy trên TabDiff và dynamic-graph protocol của dự án.

Không xếp hạng trực tiếp các dòng có protocol khác nhau. Ví dụ, CuTS báo XGB accuracy tại 100% constraint satisfaction; Harpoon báo violation, alpha-score và utility trên tail constraints; SAGE báo TSTR F1/MAPE; kết quả của mình còn đo exception preservation và conditional distortion.

## 2. SAGE — số do paper báo cáo

Nguồn: [SAGE, ACL 2026](https://aclanthology.org/2026.acl-long.174/), Table 1 và Table 3. SAGE là autoregressive LLM generator với value-conditioned pseudo-feature graph; chưa được tự chạy trên server trong đợt này.

### 2.1 Downstream utility với Random Forest

F1 cao hơn tốt hơn; Housing dùng MAPE, thấp hơn tốt hơn.

| Dataset | Metric | TabSyn | GReaT | GRADE | SPADA | SAGE-FS | SAGE-LC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Adult | F1 | 0.59 | 0.69 | 0.63 | 0.75 | 0.75 | **0.76** |
| HELOC | F1 | 0.70 | 0.70 | **0.71** | 0.69 | **0.71** | **0.71** |
| Iris | F1 | **1.00** | 0.35 | **1.00** | **1.00** | **1.00** | **1.00** |
| Diabetes | F1 | 0.12 | 0.41 | **0.42** | 0.37 | 0.41 | 0.40 |
| MIC | F1 | 0.95 | **0.96** | 0.95 | **0.96** | 0.95 | **0.96** |
| Housing | MAPE ↓ | **0.23** | 0.26 | **0.23** | 0.25 | 0.25 | 0.40 |

Diễn giải: SAGE có gain mạnh so với GReaT trên một số dataset, nhưng không thống trị TabSyn/GRADE/SPADA trên mọi task. Đặc biệt, Logit Correction kém trên Housing MAPE.

### 2.2 Constraint violation

| Dataset/rule | TVAE | CTGAN | GReaT | TabSyn | SPADA | SAGE-FS | SAGE-LC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Adult education consistency | 4.21% | 34.41% | **0.00%** | 2.32% | 3.59% | 1.32% | 16.47% |
| Housing geographic boundary | 15.55% | 34.48% | 3.61% | 10.70% | 5.26% | 4.73% | **2.65%** |

Kết quả này rất quan trọng cho positioning: SAGE dependency guidance không tương đương constraint enforcement. Feature Selector và Logit Correction đổi thứ hạng theo loại relation; SAGE-LC tốt nhất trên spatial boundary nhưng rất kém trên Adult semantic consistency.

## 3. Harpoon — số do paper báo cáo

Nguồn: [Harpoon, ICLR 2026](https://arxiv.org/abs/2602.07875), Table 3. Các số trong `external/harpoon/experiments/general_constraints_updated_utility.csv` khớp với bảng paper tới sai khác làm tròn; đây là artifact tác giả cung cấp, không phải self-run mới.

### 3.1 Violation rate

| Constraint | Dataset | DiffPuter | GReaT | Harpoon |
| --- | --- | ---: | ---: | ---: |
| Range | Adult | 73.86% | 83.58% | **3.06%** |
| Range | Default | 88.53% | 95.37% | **7.78%** |
| Range | Shoppers | 76.69% | 96.53% | **20.55%** |
| Category | Adult | **0.00%** | **0.00%** | 2.18% |
| Category | Default | **0.00%** | **0.00%** | 0.91% |
| Category | Shoppers | **0.00%** | **0.00%** | **0.00%** |
| Range AND Category | Adult | 78.88% | 80.22% | **3.37%** |
| Range AND Category | Default | 91.30% | 94.79% | **2.98%** |
| Range AND Category | Shoppers | 78.68% | 95.23% | **12.05%** |
| Range OR Category | Adult | 63.22% | 67.14% | **3.04%** |
| Range OR Category | Default | 23.03% | 24.04% | **4.23%** |
| Range OR Category | Shoppers | 50.56% | 65.76% | **20.20%** |

Harpoon rất mạnh cho differentiable range/composition constraints, nhưng hard substitution của DiffPuter/GReaT thắng violation trên simple categorical constraints. Đây là baseline trực tiếp cho mọi claim về diffusion-time guidance.

### 3.2 Harpoon fidelity/utility do paper báo cáo

| Constraint | Dataset | Alpha-score | Utility |
| --- | --- | ---: | ---: |
| Range | Adult | 0.92 | 0.95 |
| Range | Default | 0.79 | 0.85 |
| Range | Shoppers | 0.60 | 0.89 |
| Category | Adult | 0.86 | 0.88 |
| Category | Default | 0.92 | 0.83 |
| Category | Shoppers | 0.71 | 0.88 |
| AND | Adult | 0.82 | 0.91 |
| AND | Default | 0.87 | 0.81 |
| AND | Shoppers | 0.81 | 0.85 |
| OR | Adult | 0.87 | 0.93 |
| OR | Default | 0.65 | 0.81 |
| OR | Shoppers | 0.47 | 0.89 |

## 4. CuTS — số do paper báo cáo

Nguồn: [CuTS, ICML 2024](https://arxiv.org/abs/2307.03577), Table 2 và Table 9.

### 4.1 Adult logical constraints tại 100% satisfaction

Metric là XGB accuracy; mọi method trong bảng đã đạt 100% CSR bằng rejection sampling hoặc fine-tuning + rejection sampling.

| Method | I1 | I2 | I3 | RC1 | RC2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TVAE | 82.1 | 82.1 | 82.1 | 81.1 | 81.3 |
| CTGAN | 83.4 | 83.0 | 83.4 | 82.5 | 82.5 |
| CuTS + rejection sampling | **85.1** | **85.1** | **85.1** | 82.9 | 84.5 |
| CuTS + fine-tuning + rejection sampling | **85.1** | 85.0 | **85.1** | **84.7** | **84.8** |

### 4.2 German logical constraints tại 100% satisfaction

| Method | I1 | I2 | I3 | RC1 | RC2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TVAE | 72.0 | 71.3 | 72.2 | 72.1 | 70.5 |
| CTGAN | 63.4 | 63.8 | 64.4 | 64.0 | 63.5 |
| CuTS + rejection sampling | **72.4** | **73.3** | **73.6** | 72.8 | 71.2 |
| CuTS + fine-tuning + rejection sampling | 70.2 | 73.0 | 72.6 | **73.7** | **72.6** |

CuTS cho thấy final validity 100% không phân biệt được method; utility/fidelity sau enforcement mới là metric quyết định.

## 5. Official code tự chạy trên server

### 5.1 CuTS official trên Adult

Một training/sample seed; năm repeated sampling runs dùng cùng trained model. Tên file program của code (`implication2/4/5`) không được tự động đồng nhất với ký hiệu I1/I2/I3 trong paper, nên không ghép hàng bằng suy đoán.

| Program | Base satisfaction | CuTS satisfaction | Base TV | CuTS TV | Base BAC | CuTS BAC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| implication2 | 0.9280 | **1.0000** | 0.0001 | 0.0001 | 0.7735 | 0.7753 |
| implication4 | 0.9851 | **1.0000** | 0.0001 | 0.0003 | 0.7691 | 0.7742 |
| implication5 | 0.5894 | **1.0000** | 0.0001 | 0.0002 | 0.7730 | 0.7756 |
| line_constraint1 | 0.3256 | **1.0000** | 0.0001 | 0.0021 | 0.7756 | 0.7742 |
| line_constraint2 | 0.4030 | **1.0000** | 0.0001 | 0.0002 | 0.7691 | 0.7722 |

Repeated sampling giữ satisfaction `1.0000 ± 0.0000` cho cả năm program. Đây không phải năm independent training seeds.

### 5.2 LCATS official trên Insurance

Official phases 1–5 đã hoàn tất. Independent evaluator không được dùng bởi LCATS để chọn candidate.

| Output | Any violation | Numeric KS | Categorical TV | Correlation MAE | Propensity AUC | TSTR NRMSE | TSTR R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Real-train ceiling | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5674 | 0.4921 | 0.7578 |
| TabSyn original | 0.0822 | 0.0359 | 0.0233 | 0.0141 | 0.5071 | 0.5280 | 0.7212 |
| LCATS aligned | **0.0017** | **0.0318** | 0.0233 | 0.0150 | 0.5070 | **0.5268** | **0.7224** |

LCATS giảm gần hết violation, nhưng còn hai conditional violations. Exact row edit rate là 1.0 và cell edit rate 0.2884 do nhiều numeric values dịch chuyển nhỏ; normalized numeric L1 chỉ 0.00486, categorical edit rate bằng 0.

### 5.3 Baseline đang chạy, chưa đưa số vào bảng

| Job | Nội dung | Trạng thái tại thời điểm đóng băng |
| --- | --- | --- |
| `1052788` | CuTS German baseline | Đang chạy |
| `1052789` | CuTS German constrained | Chờ baseline |
| `1052794` | CuTS German audit | Chờ constrained run |
| `1052791` | Harpoon official environment | Đang cài dependencies |
| `1052792` | Harpoon backbone Adult | Chờ setup |
| `1052793` | Harpoon/RePaint 4 constraint types | Chờ training |
| `1052795` | Harpoon official audit | Chờ sampling |

Các số Harpoon ở Section 3 là paper/repository-reported, chưa phải kết quả rerun mới.

## 6. TabDiff baselines của mình trên public CuTS Adult programs

| Program | Method | Satisfaction | Numeric KS | Categorical TV | Edit rate | Passes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| implication2 | unconditional | 0.9939 | 0.0204 | 0.0201 | 0.0000 | 1 |
| implication2 | final repair | 1.0000 | 0.0204 | 0.0194 | 0.0061 | 1 |
| implication2 | best-of-8 + fallback | 1.0000 | 0.0209 | 0.0153 | 0.0000 | 8 |
| implication4 | unconditional | 0.9934 | 0.0223 | 0.0154 | 0.0000 | 1 |
| implication4 | final repair | 1.0000 | 0.0223 | 0.0154 | 0.0066 | 1 |
| implication4 | best-of-8 + fallback | 1.0000 | 0.0178 | 0.0170 | 0.0000 | 8 |
| implication5 | unconditional | 0.9448 | 0.0201 | 0.0143 | 0.0000 | 1 |
| implication5 | final repair | 1.0000 | 0.0201 | 0.0179 | 0.0552 | 1 |
| implication5 | best-of-8 + fallback | 1.0000 | 0.0216 | 0.0245 | 0.0000 | 8 |
| line_constraint1 | unconditional | 0.3279 | 0.0218 | 0.0155 | 0.0000 | 1 |
| line_constraint1 | final repair | 1.0000 | 0.0218 | 0.0890 | 0.6721 | 1 |
| line_constraint1 | best-of-8 + fallback | 1.0000 | 0.0514 | 0.2002 | 0.0371 | 8 |
| line_constraint2 | unconditional | 0.4365 | 0.0249 | 0.0176 | 0.0000 | 1 |
| line_constraint2 | final repair | 1.0000 | 0.0942 | 0.0176 | 0.5635 | 1 |
| line_constraint2 | best-of-8 + fallback | 1.0000 | 0.1146 | 0.0804 | 0.0095 | 8 |

Implication2/4 gần bão hòa. Line constraints cho thấy validity 1.0 có thể đi cùng distribution shift lớn.

## 7. Kết quả dynamic graph của dự án

### 7.1 Graph readability theo diffusion trajectory

Probe dùng 2,048 synthetic rows, 50 reverse steps, 12 edges, một seed. Label là membership ở final generated row, chưa phải semantic ground truth bên ngoài.

| Normalized timestep | Edge precision | Edge recall | Edge accuracy |
| --- | ---: | ---: | ---: |
| `(0.75, 1.00]` | 0.6688 | 0.2416 | 0.9272 |
| `(0.50, 0.75]` | 0.8162 | 0.6976 | 0.9657 |
| `(0.25, 0.50]` | 0.9216 | 0.9311 | 0.9895 |
| `(0.10, 0.25]` | 0.9874 | 0.9926 | 0.9986 |
| `[0, 0.10]` | 0.9999 | 1.0000 | 1.0000 |

Đây là evidence trực tiếp cho gap trajectory: active graph từ early preview có recall rất thấp và chỉ trở nên đáng tin ở mid/late trajectory.

### 7.2 End-to-end controlled subgroup policies

Các method có `+certifier` đạt final satisfaction 1.0. `Exceptions changed` đo collateral damage lên valid exceptions.

| Dataset | Method | Satisfaction | Exceptions changed | Target KS |
| --- | --- | ---: | ---: | ---: |
| Default | Unconditional | 0.7024 | 0.0000 | 0.3256 |
| Default | Static final repair | 1.0000 | **0.9744** | 0.3129 |
| Default | Subgroup final repair | 1.0000 | 0.0000 | 0.3129 |
| Default | Best-of-8 graph + certify | 1.0000 | 0.0000 | **0.1586** |
| Default | Mid lookahead + certify | 1.0000 | 0.0000 | 0.2335 |
| Shoppers | Unconditional | 0.4470 | 0.0000 | 0.5642 |
| Shoppers | Static final repair | 1.0000 | **0.6491** | 0.5642 |
| Shoppers | Subgroup final repair | 1.0000 | 0.0000 | 0.5642 |
| Shoppers | Best-of-8 graph + certify | 1.0000 | 0.0000 | 0.2685 |
| Shoppers | Mid numgraph lookahead + certify | 1.0000 | 0.0000 | **0.2259** |

Kết quả support dynamic row/subgroup activation: static repair phá phần lớn exception trong controlled policies. Tuy nhiên best-of-8 thắng mid lookahead trên Default target KS; mid lookahead thắng trên Shoppers. Không có universal winner.

### 7.3 Incremental value của intervention stage

Chỉ giữ outputs có active satisfaction ít nhất 0.999 và không thay đổi exception. Score là within-policy normalized rank, thấp hơn tốt hơn.

| Objective | Policies | Mid gain so với best non-mid | Mid win fraction |
| --- | ---: | ---: | ---: |
| Balanced | 30 | +0.0118 | 0.6000 |
| Intervention-conservative | 30 | +0.0370 | 0.7667 |
| Joint fidelity | 30 | **+0.0550** | 0.6667 |
| Target fidelity | 30 | **−0.1004** | 0.2667 |

Kết luận: mid-denoising có giá trị cho joint structure và preservation, nhưng bị final repair/resampling vượt rõ khi objective chỉ là target marginal.

### 7.4 Solver routing

| Objective | Best fixed score | Oracle score | Oracle gain | Cross-relation router gain |
| --- | ---: | ---: | ---: | ---: |
| Balanced | 0.4574 | 0.3586 | +0.0988 | +0.0118 |
| Intervention-conservative | 0.4741 | 0.4012 | +0.0729 | −0.0016 |
| Joint fidelity | 0.4218 | 0.3213 | +0.1005 | −0.0030 |
| Target fidelity | 0.4343 | 0.3000 | +0.1343 | +0.0557 |

Oracle cho thấy solver heterogeneity có thật; learned router hiện tại chưa đủ mạnh để làm contribution chính.

### 7.5 Dynamic graph dưới distribution shift

| Dataset | Policy | Safe retention | Unsafe exposure | Weighted violation |
| --- | --- | ---: | ---: | ---: |
| Beijing | Frozen | 1.0000 | 1.0000 | 0.0042 |
| Beijing | Last-window | 0.8866 | 1.0000 | 0.0043 |
| News | Frozen | 1.0000 | 1.0000 | 0.0148 |
| News | Last-window | 1.0000 | **0.4713** | **0.0094** |
| Shoppers | Frozen | 1.0000 | 1.0000 | 0.0171 |
| Shoppers | Last-window | 0.9383 | **0.4783** | **0.0072** |

Dynamic monitoring tốt trên News/Shoppers nhưng không dự đoán được transient shift ở Beijing. Đây là evidence cho graph lifecycle, chưa phải generation-under-shift result.

## 8. Kết luận từ toàn bộ bảng

1. **SAGE xác nhận row/value-conditioned graph có ích**, nhưng paper không giải quyết uncertain graph topology dọc diffusion trajectory. Đây vẫn là gap hợp lý của dự án.
2. **Harpoon là baseline khó nhất cho inference-time guidance.** Nó rất mạnh trên range/AND/OR, nhưng không đạt zero violation và không xử lý explicit exception preservation.
3. **CuTS và repair chứng minh final validity dễ đạt** trên executable constraints. Contribution của mình phải nằm ở conditional/joint fidelity, exception preservation, stage và cost, không phải chỉ satisfaction.
4. **LCATS official gần loại hết violation với fidelity trade-off nhỏ**, nên trajectory method phải chứng minh incremental benefit so với post-repair trên constraint thực sự cần joint correction.
5. **Kết quả của mình support feasibility nhưng chưa state of the art.** Evidence mạnh nhất hiện tại là trajectory readability curve, exception damage của static repair và objective-dependent value của mid-denoising. Weak point là learned router và thiếu official Harpoon rerun hoàn tất.

## 9. Artifact nguồn

- SAGE paper: <https://aclanthology.org/2026.acl-long.174/>
- Harpoon paper: <https://arxiv.org/abs/2602.07875>
- CuTS paper: <https://arxiv.org/abs/2307.03577>
- CuTS Adult official self-run: `outputs/cuts_official_result_audit_1045524/RESULTS.md`
- CuTS repeated sampling: `outputs/cuts_official_resample_audit_1045748/RESULTS.md`
- LCATS official audit: `outputs/lcats_official_output_audit_1052790/RESULTS.md`
- TabDiff public CuTS programs: `outputs/cuts_common_tabdiff_1051083/RESULTS.md`
- Trajectory readability: `outputs/persistence_graph_gate_1051548/RESULTS.md`
- Dynamic solver grid: `outputs/dynamic_graph_solver_default_1051370_0/` và `outputs/dynamic_graph_solver_shoppers_1051370_1/`
- Stage analysis: `outputs/intervention_stage_value_1052276/RESULTS.md`
- Solver router: `outputs/dynamic_solver_router_1051529/RESULTS.md` và `outputs/cross_relation_solver_router_1052275/RESULTS.md`
- Sequential monitoring: `outputs/temporal_graph_change_1052277/RESULTS.md`
