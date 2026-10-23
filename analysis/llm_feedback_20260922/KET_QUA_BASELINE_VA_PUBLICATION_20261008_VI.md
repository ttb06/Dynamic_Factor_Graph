# Tổng hợp kết quả, gain, baseline và thông tin publication

**Cập nhật:** 08-10-2026

## 1. Quy ước nguồn số

| Nhãn | Ý nghĩa | Có dùng để kết luận thắng/thua trực tiếp? |
| --- | --- | --- |
| **Our local — common protocol** | Chạy trong repo với cùng backbone, split, constraint catalog, seed/noise hoặc paired control | Có |
| **Official self-run** | Chạy code chính thức của tác giả trên server, nhưng có thể khác dataset/task của method mình | Chỉ khi protocol khớp; hiện chủ yếu dùng làm reference |
| **Paper-reported** | Số lấy nguyên từ paper/proceedings | Không so trực tiếp với local nếu khác split, constraint hoặc metric |
| **Proxy** | Tự implement cơ chế gần giống paper | Chỉ gọi là proxy; không gọi là reproduction |

Mọi gain phần trăm dưới đây dùng công thức:

\[
\text{relative reduction}=\frac{\text{baseline error}-\text{method error}}
{\text{baseline error}}\times100\%.
\]

---

## 2. Kết quả chính của dynamic-middle graph

Nguồn: **Our local — common protocol**. Positive improvement là `frozen − dynamic`.

| Dataset | Dynamic error ↓ | Frozen error ↓ | Relative reduction | Seed wins | Hierarchical 95% CI của improvement | Kết luận |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Shoppers | **0.001596** | 0.003116 | **48.8%** | 3/3 | **[0.000994, 0.002057]** | Bằng chứng mạnh |
| Default | 0.007940 | 0.009903 | 19.8% | 2/3 | [-0.003091, 0.007210] | Chưa chắc chắn |
| Diabetes | **0.001077** | 0.003543 | **69.6%** | 3/3 | **[0.001270, 0.003956]** | Bằng chứng mạnh |

### Overall heldout error

| Dataset | Dynamic ↓ | Frozen ↓ | Relative reduction | Seed wins |
| --- | ---: | ---: | ---: | ---: |
| Shoppers | **0.019686** | 0.020613 | **4.5%** | 3/3 |
| Default | 0.018586 | 0.019239 | 3.4% | 2/3 |
| Diabetes | **0.011112** | 0.012311 | **9.7%** | 3/3 |

Diễn giải:

- Shoppers và Diabetes support core claim rằng evolving row–trajectory state chứa thông tin ngoài frozen graph.
- Default chỉ là mixed/negative dataset; không dùng để claim robustness.
- Đây là uncapped comparison. Action efficiency phải được đọc riêng ở phần sau.

---

## 3. Controls quan trọng và gain

### 3.1. Exact action budget

Nguồn: **Our local — common protocol**.

| Dataset | Dynamic | Frozen | Action budget | Gain của dynamic | Kết luận |
| --- | ---: | ---: | ---: | ---: | --- |
| Shoppers | **0.002498** | 0.003116 | 33 / 33 | **19.8% reduction** | Dynamic thắng 3/3 |
| Default | 0.017417 | **0.009903** | 73.3 / 73.3 | **75.9% error increase** | Dynamic thua 3/3 |
| Diabetes | 0.005213 | **0.004740** | 20.0 / 19.3 | **10.0% error increase** | Dynamic thua 3/3 |

Không được claim dynamic graph action-efficient phổ quát. Shoppers là positive case; Default và Diabetes là falsification evidence.

### 3.2. Correct row assignment so với shuffled rows

Shoppers, đúng 40 actions:

| Method | Actuation error ↓ | Overall error ↓ | KS ↓ |
| --- | ---: | ---: | ---: |
| Correct row-dynamic | **0.003307** | **0.020994** | **0.026137** |
| Shuffled row-dynamic | 0.004326 | 0.021497 | 0.026265 |

Correct row assignment giảm actuation error **23.5%**, thắng 3/3 seeds. Đây là control sạch nhất chứng minh row-specific graph state có ích.

### 3.3. Timing trên Diabetes, cap 20 actions

| Timing | Actuation error ↓ | Relative gain so với early |
| --- | ---: | ---: |
| Early | 0.006152 | — |
| Middle | 0.005213 | **15.3%** |
| Late | **0.003478** | **43.5%** |

Middle tốt hơn early nhưng late tốt hơn middle **33.3%**. Vì vậy claim đúng là “middle là vùng sớm nhất hữu ích”, không phải “middle luôn tối ưu”.

### 3.4. Transition-survival certifier

Nguồn: **Our local — calibration seed và hai evaluation seeds tách riêng**.

| Dataset/state | Immediate precision | Wait-5 precision | Precision gain | Genuine-transition recall |
| --- | ---: | ---: | ---: | ---: |
| Default applicability | 0.7006 | 0.8313 | **+0.1307** | 0.9140 |
| Default violation | 0.5574 | 0.7156 | **+0.1582** | 0.9007 |
| Diabetes applicability | 0.6966 | 0.8247 | **+0.1281** | 0.8951 |
| Diabetes violation | 0.5184 | 0.7904 | **+0.2720** | 0.9230 |
| Shoppers applicability | **0.9655** | 0.9776 | +0.0121 | 0.9955 |
| Shoppers violation | 0.6005 | **0.9078** | **+0.3073** | 0.8980 |

Temporal survival là signal tốt, đặc biệt cho violation transitions. Tuy nhiên chỉ Shoppers applicability vượt precision 0.95 ngay lập tức; chưa được gọi là high-confidence certifier hoàn chỉnh.

### 3.5. Online transition policy trên ba seeds mới

Nguồn: **Our local — common protocol**. Năm policies chạy tuần tự trong cùng task/GPU; unconditional sample hash khớp 9/9 dataset-seed groups.

| Dataset | Immediate ↓ | Wait-3 ↓ | Dynamic + violation wait-5 ↓ | Topology + violation wait-5 ↓ | Frozen + wait-5 ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shoppers | **0.001819** | 0.002568 | 0.002578 | 0.003138 | 0.003486 |
| Default | 0.009669 | 0.005267 | **0.004497** | 0.007163 | 0.012903 |
| Diabetes | **0.000855** | 0.001247 | 0.001674 | 0.002064 | 0.002885 |

Pre-registered policy từ offline audit là `dynamic+violation-wait-5` cho Shoppers và `topology+violation-wait-5` cho Default/Diabetes:

- Shoppers: thua immediate 0/3 seeds nhưng thắng frozen 3/3;
- Default: thắng immediate 2/3 và frozen 3/3, nhưng chỉ thắng wait-3 ở 1/3;
- Diabetes: thua immediate và wait-3 ở 3/3, nhưng thắng frozen 3/3.

Kết luận: **offline transition precision không phải surrogate đầy đủ cho end-to-end generation quality**. Chờ lâu loại false transitions nhưng cũng bỏ lỡ thời điểm action còn tác động được trajectory. Điểm dương đáng chú ý là trên Default, chỉ chờ violation (`dynamic_wait5`) giảm error **53.5%** so với immediate và **14.6%** so với wait-3. Policy này không tổng quát sang Shoppers/Diabetes nên chỉ tạo hypothesis mới, chưa phải main method.

---

## 4. Fidelity và utility của method

Nguồn: **Our local — ba sampling seeds, cùng evaluator**.

| Dataset | Method | Shape ↓ | Trend ↓ | α-Precision ↑ | β-Recall ↑ | Detection ↑ | MLE AUC ↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Shoppers | TabDiff | 1.3933 | 1.8358 | 98.9079 | **49.9300** | 0.9906 | 0.9043 |
| Shoppers | Dynamic-stable | **1.3778** | 1.7967 | **98.9158** | 49.9051 | 0.9912 | **0.9108** |
| Shoppers | Final repair | 1.3792 | **1.7911** | 98.8208 | 49.9737 | **0.9924** | 0.9062 |
| Default | TabDiff | 1.7829 | 1.9035 | 97.7084 | **52.6055** | 0.9488 | 0.7395 |
| Default | Dynamic-current | **1.7792** | 1.9033 | **97.7111** | 52.5993 | 0.9478 | 0.7445 |
| Default | Final repair | 1.7800 | 1.9035 | 97.7084 | 52.6055 | **0.9543** | **0.7451** |

Các gain nhỏ nhưng không thấy fidelity collapse:

- Shoppers dynamic giảm Shape error khoảng **1.1%**, Trend error khoảng **2.1%**, MLE AUC tăng `+0.0065` tuyệt đối so với TabDiff.
- Default dynamic giảm Shape error khoảng **0.2%**, MLE AUC tăng `+0.0050` tuyệt đối.
- Final repair vẫn cạnh tranh hoặc tốt hơn ở một số quality/utility metrics.

---

## 5. General generator baselines trên Shoppers

Nguồn: **Our local — cùng repository split, sample count, graph catalog và metric implementation**. Đây là quality reference; không phải bảng causal dynamic-vs-frozen chính.

| Method | Shape error ↓ | Trend error ↓ | Detection ↑ | MLE AUC ↑ | Actuator violation ↓ | Overall heldout error ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TabDiff | 1.7462 | 1.5848 | 0.9879 | 0.8988 | 0.0070 | 0.02177 |
| DynamicGraph-MidLookahead | 1.7394 | 1.5086 | 0.9810 | **0.9003** | 0.0059 | 0.02093 |
| DynamicGraph-ReverseX0 | **1.7326** | 1.5117 | 0.9825 | 0.8977 | 0.0052 | 0.02075 |
| PostRepair | 1.7353 | **1.5009** | **0.9982** | 0.8995 | **0.0036** | **0.02046** |
| CTGAN-SDV | 14.2642 | 15.4354 | 0.7497 | 0.8013 | 0.3823 | 0.30628 |
| TVAE-SDV | 16.6945 | 9.4195 | 0.1801 | 0.8773 | 0.2758 | 0.18525 |

So với TabDiff trong bảng này:

- ReverseX0 giảm actuator violation khoảng **26.4%** và overall heldout error khoảng **4.7%**.
- MidLookahead giảm Trend error khoảng **4.8%**.
- PostRepair vẫn có actuator violation thấp hơn ReverseX0 khoảng **30.1%**.
- CTGAN/TVAE thấp hơn rõ rệt về quality và semantic validity, nhưng không phải constraint-aware competitors mạnh nhất.

---

## 6. Official self-run baselines

### 6.1. LCATS official code trên Insurance

**Publication:** KDD 2026, tháng 08/2026.  
**Nguồn số:** official code tự chạy và independent local evaluator; **không phải số lấy từ paper**.

| Output | Any violation ↓ | Numeric KS ↓ | TV ↓ | TSTR R² ↑ | Row edit | Cell edit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TabSyn original | 0.0822 | 0.0359 | 0.0233 | 0.7212 | — | — |
| LCATS aligned | **0.0017** | **0.0318** | 0.0233 | **0.7224** | 1.0000 | 0.2884 |

Gain của LCATS local:

- any-violation giảm **97.9%**;
- numerical KS giảm **11.4%**;
- TSTR R² tăng `+0.0012` tuyệt đối;
- đổi ít nhất một cell ở toàn bộ rows và đổi 28.84% cells.

Đây là strong post-repair reference, nhưng khác dataset, generator và rules nên không được nói method mình beat LCATS.

### 6.2. CuTS official code trên German

**Publication:** ICML 2024, 21–27/07/2024.  
**Nguồn số:** official code tự chạy; **không phải chỉ chép từ paper**.

- Ba implication programs và hai line-constraint programs đều đạt satisfaction `1.0000`.
- Base satisfaction nằm từ `0.0988` đến `0.8380` cho implication, và `0.2800–0.3775` cho line constraints.
- TV tăng từ khoảng `0.0009–0.0010` lên `0.0011–0.0027` tùy program.
- Balanced accuracy có thể tăng hoặc giảm tùy constraint.

Ý nghĩa: exact final validity không đủ làm contribution; phải báo utility và distribution distortion.

---

## 7. Paper-reported baselines và thời điểm publication

Các số trong phần này **lấy từ paper**, không phải kết quả tự chạy và không so trực tiếp với bảng local.

### 7.1. TabSyn

**Venue/time:** ICLR 2024.  
**Trạng thái:** main conference paper.  
**Paper-reported:** trên sáu datasets và năm metrics, TabSyn báo giảm error khoảng **86%** cho column-wise distribution và **67%** cho pair-wise correlation so với baseline cạnh tranh nhất.

Vai trò: diffusion generator mạnh trong VAE latent space; chưa phải constraint-guidance baseline trực tiếp.

### 7.2. TabDiff

**Venue/time:** ICLR 2025.  
**Trạng thái:** main conference paper; backbone chính của dự án.  
**Paper-reported:** paper báo tốt hơn các baselines trên trung bình tám metrics, với gain tối đa **22.5%** trên pair-wise correlation estimation.

Vai trò: generator baseline chính. Kết quả dynamic graph phải được báo như gain trên TabDiff, không thay thế quality claim của TabDiff paper.

### 7.3. CuTS

**Venue/time:** ICML 2024, tháng 07/2024.  
**Trạng thái:** main conference paper.  
**Paper-reported:** tại 100% constraint satisfaction:

- Adult XGB accuracy: CuTS variants khoảng `82.9–85.1`, tùy program;
- German XGB accuracy: khoảng `70.2–73.7`;
- paper cũng báo ở cùng fairness level, Adult downstream accuracy cao hơn prior work **2.3%**.

Vai trò: training/fine-tuning-time declarative constraint baseline. Không có row–trajectory dynamic graph.

### 7.4. Harpoon

**Venue/time:** ICLR 2026; arXiv ban đầu tháng 02/2026.  
**Trạng thái:** main conference paper.  
**Paper-reported violation rates:** 

| Constraint | Adult | Default | Shoppers |
| --- | ---: | ---: | ---: |
| Range | 3.06% | 7.78% | 20.55% |
| Category | 2.18% | 0.91% | **0.00%** |
| Range AND Category | 3.37% | 2.98% | 12.05% |
| Range OR Category | 3.04% | 4.23% | 20.20% |

Paper-reported α-score nằm khoảng `0.47–0.92`, utility khoảng `0.81–0.95` tùy dataset/constraint.

Vai trò: baseline trực tiếp nhất cho inference-time differentiable guidance. Local official reproduction hiện chưa hoàn tất do CUDA/PyTorch environment; các số trên là **paper/artifact-reported**, không phải local rerun.

### 7.5. SAGE

**Venue/time:** ACL 2026 main conference, tháng 07/2026.  
**Trạng thái:** long paper.  
**Paper-reported utility:** 

| Dataset | Metric | TabSyn | GReaT | GRADE | SPADA | SAGE-FS | SAGE-LC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Adult | F1 | 0.59 | 0.69 | 0.63 | 0.75 | 0.75 | **0.76** |
| HELOC | F1 | 0.70 | 0.70 | **0.71** | 0.69 | **0.71** | **0.71** |
| Diabetes | F1 | 0.12 | 0.41 | **0.42** | 0.37 | 0.41 | 0.40 |
| Housing | MAPE ↓ | **0.23** | 0.26 | **0.23** | 0.25 | 0.25 | 0.40 |

Paper-reported policy violations:

| Dataset/rule | TabSyn | GReaT | SPADA | SAGE-FS | SAGE-LC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Adult education consistency | 2.32% | **0.00%** | 3.59% | 1.32% | 16.47% |
| Housing geographic boundary | 10.70% | 3.61% | 5.26% | 4.73% | **2.65%** |

SAGE không thống trị mọi dataset/rule. Nó là novelty threat cho value-conditioned, row-dependent dependencies, nhưng không model graph thay đổi theo reverse-diffusion timestep.

### 7.6. LCATS

**Venue/time:** KDD 2026, tháng 08/2026.  
**Trạng thái:** conference paper; DOI `10.1145/3770855.3818181`.  
**Số trong báo cáo này:** dùng official local rerun ở Section 6.1, không suy diễn thêm paper result chưa kiểm tra.

Vai trò: LLM constraint discovery + TabSyn + post-generation diagnosis/repair. Đây là baseline gần nhất cho graph acquisition và repair, không phải middle-step trajectory guidance.

### 7.7. PromptLoop

**Venue/time:** CVPR 2026, tháng 06/2026.  
**Trạng thái:** main conference paper.  
**Không đưa số vào bảng comparison:** khác modality và metric. Paper chứng minh intermediate latent feedback + iterative prompt refinement đã tồn tại trong image diffusion.

Vai trò: novelty boundary, không phải tabular quantitative baseline.

---

## 8. Bảng publication tóm tắt

| Work | Venue | Publication time | Loại bằng chứng dùng trong report |
| --- | --- | --- | --- |
| TabSyn | ICLR 2024 | 2024 | Paper-reported |
| CuTS | ICML 2024 | 21–27/07/2024 | Paper-reported + official self-run |
| TabDiff | ICLR 2025 | 2025 | Paper-reported + official backbone local |
| Harpoon | ICLR 2026 | 2026; arXiv 02/2026 | Paper/artifact-reported; local setup chưa thành công |
| PromptLoop | CVPR 2026 | 06/2026 | Positioning only |
| SAGE | ACL 2026 main | 07/2026 | Paper-reported |
| LCATS | KDD 2026 | 08/2026 | Official self-run; paper metadata |
| Dự án này | Chưa submit | 10/2026 experiments | Our local common protocol |

---

## 9. Kết luận so sánh hiện tại

### Method mình đã beat được gì

- Frozen-state graph trên Shoppers và Diabetes trong uncapped middle trajectory.
- Frozen-state graph trên Shoppers khi action budget bằng nhau.
- Shuffled row assignment trên Shoppers với exact action budget.
- Unconditional TabDiff về semantic error trong các local Shoppers settings, trong khi quality gần như giữ nguyên.
- Early intervention về validity trên cả ba dataset.

### Chưa beat hoặc không được phép claim beat

- Final repair: vẫn tốt hơn trên nhiều Shoppers/Diabetes settings.
- Harpoon: chưa có faithful same-task local reproduction.
- LCATS: official local run khác Insurance task và dùng TabSyn.
- CuTS: official run khác German/Adult declarative programs.
- SAGE: khác generator, datasets, dependency protocol và metrics.
- SOTA tabular generation nói chung: chưa có second backbone/checkpoint và common benchmark đầy đủ.

### Claim kết quả hợp lý nhất

> Dynamic row–trajectory graph giảm semantic calibration error so với frozen graph trên hai datasets mà không gây fidelity collapse rõ ràng. Lợi ích không phổ quát dưới matched action budget, và strong post-repair vẫn cạnh tranh. Transition survival là tín hiệu hứa hẹn để lọc graph-state updates, nhưng online gain đang chờ confirmation.

Sau online confirmation, câu cuối được cập nhật: transition survival cải thiện state-estimation precision nhưng **không tạo online gain phổ quát**; nó chỉ giúp rõ trên Default. Main method chưa nên đóng băng quanh wait-5 certifier.

---

## 10. Artifacts

- Main bootstrap: `outputs/bootstrap_dynamic_topology_1060488/RESULTS.md`
- Diabetes controls: `outputs/diabetes_controls_aggregate_1060481/RESULTS.md`
- Transition survival: `outputs/transition_survival_aggregate_1060883/RESULTS.md`
- Online transition policy: `outputs/transition_policy_aggregate_1060902/RESULTS.md`
- Shoppers local baselines: `outputs/shoppers_baseline_metric_suite_1057841/RESULTS.md`
- LCATS official audit: `outputs/lcats_official_output_audit_1052790/RESULTS.md`
- CuTS official German audit: `outputs/cuts_official_german_audit_1052794/RESULTS.md`
- Paper/local frozen comparison: `BANG_TONG_HOP_PAPER_BASELINE_VA_KET_QUA_20261002.md`
