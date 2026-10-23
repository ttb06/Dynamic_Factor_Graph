# Tổng hợp kết quả, đối chiếu SOTA và mức sẵn sàng của dự án

**Ngày chốt:** 07-10-2026  
**Backbone đang dùng:** TabDiff, ICLR 2025  
**Hướng chính:** trajectory-conditioned dynamic constraint-state graph cho mixed-type tabular diffusion

## 1. Kết luận điều hành

1. **Đã đủ bằng chứng để bắt đầu phát triển method sâu hơn và viết dần problem/method/evaluation.** Ba kết quả nhất quán là: predicted-clean state ở early steps chưa đáng tin; row–edge alignment có ý nghĩa nhân quả; cập nhật trạng thái edge `satisfied ↔ violated` trong middle trajectory có thể hơn frozen graph.
2. **Chưa đủ để tuyên bố SOTA hoặc chuẩn bị bản nộp hoàn chỉnh.** Trên Shoppers, post-repair vẫn mạnh hơn dynamic guidance ở constraint metric. Trên Default, dynamic guidance có thể hơn final repair, nhưng lợi ích dynamic so với frozen-state nhỏ và đổi chiều khi sampler giảm từ 50 xuống 30 steps.
3. **Contribution nên nằm ở graph-state lifecycle, không nằm ở hard constraint enforcement.** Harpoon và CTDF đã đưa constraint vào reverse diffusion; SAGE đã có value-dependent graph theo row; LCATS đã có LLM discovery và post-repair. Khoảng trống còn lại là edge state thay đổi theo predicted-clean trajectory, kèm selective action trong middle steps.
4. **LLM nên xây và kiểm chứng graph trước sampling, không nên được gọi trực tiếp ở mọi step cho executable rules.** API probe trước đây cho thấy compile-once thành typed DSL đạt execution equivalence `1.0` với khoảng `1.85k` tokens, trong khi direct row judgement dùng khoảng `37.5k` tokens và kém ổn định hơn.

## 2. Phát biểu bài toán hiện tại

Cho một verified typed graph (G), mỗi synthetic row (i), edge (e) và reverse timestep (t) có trạng thái:

\[
s_{i,e,t}\in\{\text{unreadable},\text{inactive},\text{satisfied},\text{violated},\text{uncertain}\}.
\]

Method cần ước lượng (s_{i,e,t}) từ predicted-clean row \(\hat{x}_{0,i}(t)\), chỉ cho phép edge phát action trong vùng trajectory đủ đọc được, rồi sửa direction của denoising thay vì chỉ sửa final table. Static graph giữ nguyên specification; phần **dynamic** là active typed state và action eligibility của từng row–edge thay đổi dọc trajectory.

## 3. Bảng SOTA do paper công bố trên Shoppers

Nguồn là bảng evaluation của TabDiff trên đúng dataset Shoppers. Đây là **số paper-reported**, không trộn trực tiếp với số local vì split, checkpoint, sample size và phiên bản metric có thể khác.

| Method | Shape error ↓ | Trend error ↓ | α-Precision ↑ | β-Recall ↑ | Detection ↑ | MLE AUC ↑ | DCR, 50 lý tưởng |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CTGAN | 21.15 | 13.08 | 76.97 | 31.80 | 0.7488 | 0.875 | – |
| TVAE | 24.51 | 18.67 | 58.19 | 19.78 | 0.2962 | 0.871 | – |
| GOGGLE | 22.33 | 23.90 | 86.95 | 9.79 | 0.1418 | 0.658 | – |
| GReaT | 14.51 | 45.16 | 78.88 | 44.90 | 0.4285 | 0.902 | – |
| STaSy | 9.37 | 8.49 | 89.65 | 37.24 | 0.5482 | 0.914 | 51.53 |
| CoDi | 31.84 | 17.78 | 94.95 | 20.82 | 0.2784 | 0.865 | 51.06 |
| TabDDPM | 2.72 | 6.61 | 88.55 | 47.79 | 0.8349 | 0.918 | 63.23 |
| TabSyn | **1.01** | 2.13 | 98.36 | 49.10 | 0.9662 | 0.914 | 52.90 |
| TabDiff | 1.28 | **1.74** | **99.11** | **49.75** | **0.9843** | **0.921** | **50.24** |

Điểm rút ra: TabDiff là backbone hợp lý vì chất lượng tổng quát đã ở mức mạnh. Method mới cần giữ các metric này trong khi cải thiện semantic validity; không cần đánh đổi fidelity để thắng một constraint score.

## 4. Bảng local cùng protocol trên Shoppers

Tất cả hàng dùng cùng split, 4,096 samples, catalog và metric implementation. CTGAN/TVAE là implementation có sẵn từ SDV. `α-Precision` và `β-Recall` còn thiếu do xung đột phiên bản SynthCity/Torch; không điền bằng metric thay thế.

| Method | Shape ↓ | Trend ↓ | Detection ↑ | MLE AUC ↑ | DCR (%) | Actuate violation ↓ | Heldout violation error ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CTGAN-SDV | 14.264 | 15.435 | 0.750 | 0.801 | 89.72 | 0.3823 | 0.3063 |
| TVAE-SDV | 16.694 | 9.419 | 0.180 | 0.877 | 88.43 | 0.2758 | 0.1852 |
| TabDiff | 1.746 | 1.585 | **0.988** | 0.899 | 91.14 | 0.0070 | 0.0218 |
| DynamicGraph-MidLookahead | 1.739 | 1.509 | 0.981 | **0.900** | 91.16 | 0.0059 | 0.0209 |
| DynamicGraph-ReverseX0 | **1.733** | 1.512 | 0.983 | 0.898 | 91.14 | 0.0052 | 0.0207 |
| PostRepair | 1.735 | **1.501** | **0.998** | 0.900 | 91.19 | **0.0036** | **0.0205** |

Diễn giải:

- DynamicGraph giữ được Shape/Trend/MLE của TabDiff và cải thiện constraint metrics nhẹ.
- PostRepair vẫn mạnh nhất trên Shoppers. Đây là đối chứng chưa bị method vượt qua.
- DCR local xa 50 ở mọi method, kể cả TabDiff; vì protocol local chưa khớp paper nên chỉ dùng so sánh nội bộ.
- Đây chưa phải bảng chứng minh SOTA. Nó chứng minh method không phá chất lượng backbone trong pilot.

## 5. Bảng chính cho cơ chế dynamic constraint-state, 8,192 rows và 50 steps

Mỗi cặp dynamic/frozen dùng cùng seed và paired unconditional samples. `Actuate heldout error` đo khoảng cách giữa violation rate synthetic và real heldout trên các edge được phép tạo action; thấp hơn tốt hơn và an toàn hơn việc tối ưu violation về 0.

| Dataset | Method | Actuate violation ↓ | Actuate heldout error ↓ | Overall heldout error ↓ | Numeric KS ↓ | Actions |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Shoppers | Unconditional | 0.00877 | 0.00452 | 0.02185 | 0.02583 | 0 |
| Shoppers | Frozen state | 0.00666 | 0.00240 | 0.02090 | **0.02573** | 113 |
| Shoppers | **Dynamic state** | 0.00563 | 0.00138 | 0.02040 | 0.02569 | 134 |
| Shoppers | Final repair | **0.00360** | **0.00066** | **0.02018** | 0.02578 | 124 |
| Default | Unconditional | 0.02806 | 0.01895 | 0.02310 | 0.02934 | 0 |
| Default | Frozen state | 0.01485 | 0.00574 | 0.01892 | **0.02932** | 45 |
| Default | **Dynamic state** | **0.01463** | **0.00553** | **0.01879** | 0.02933 | 45 |
| Default | Final repair | 0.01801 | 0.00891 | 0.01937 | 0.02937 | 17 |

Kết luận:

- Shoppers: dynamic state hơn frozen-state rõ, nhưng chưa hơn final repair.
- Default: dynamic state hơn frozen-state nhỏ, và hơn final repair trên cả constraint error lẫn overall error.
- Numeric KS gần như không đổi, nên gain không đến từ biến dạng marginal lớn.

## 6. Bảng transition subgroup

Chỉ đánh giá những rows có ít nhất một edge đổi `satisfied ↔ violated` trong middle trajectory.

| Dataset | Transition rows | Base violation events | Dynamic | Frozen-state | Final repair | Dynamic net removed | Frozen net removed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Shoppers | 103 | 99 | 38 | 56 | **30** | **61** | 43 |
| Default | 63 | 56 | **16** | 17 | 26 | **40** | 39 |

Đây là evidence sát nhất với novelty:

- Trên Shoppers, cập nhật state loại thêm 18 violation events so với frozen-state.
- Trên Default, dynamic chỉ hơn một event nhưng cả hai middle methods tốt hơn final repair 10 events.
- Applicability-flip subgroup trước đó không tạo khác biệt dynamic/frozen; do đó novelty phải là **constraint status evolution**, không chỉ active/inactive antecedent.

## 7. Robustness với 30 reverse steps

| Dataset | Method | Actuate violation ↓ | Actuate heldout error ↓ | Overall heldout error ↓ | Actions |
| --- | --- | ---: | ---: | ---: | ---: |
| Shoppers | Unconditional | 0.00836 | 0.00410 | 0.02576 | 0 |
| Shoppers | Frozen state | 0.00583 | 0.00157 | 0.02466 | 113 |
| Shoppers | **Dynamic state** | **0.00519** | **0.00093** | **0.02436** | 125 |
| Shoppers | Final repair | 0.00360 | 0.00066 | 0.02434 | 108 |
| Default | Unconditional | 0.03112 | 0.02202 | 0.02522 | 0 |
| Default | **Frozen state** | **0.01126** | **0.00312** | **0.01804** | 65 |
| Default | Dynamic state | 0.01316 | 0.00405 | 0.01821 | 67 |
| Default | Final repair | 0.01440 | 0.00530 | 0.01870 | 33 |

Robustness chưa hoàn toàn đạt: dynamic vẫn thắng trên Shoppers nhưng thua frozen-state trên Default. Đây là lý do cần state-confidence/persistence mechanism và temporal-window ablation vừa submit.

## 8. Đối chiếu các work constraint/dependency gần nhất

| Work | Vị trí constraint | Graph/condition có đổi theo row | Có đổi theo trajectory | Kết quả paper/artifact liên quan | Threat đối với claim |
| --- | --- | --- | --- | --- | --- |
| TabDiff, ICLR 2025 | Backbone mixed-type diffusion | Không | Feature-wise noise schedule, không semantic graph | SOTA mạnh trên 7 datasets/8 metrics | Method phải giữ fidelity của backbone |
| TabSyn, ICLR 2024 | Diffusion trong VAE latent | Không | Không | Paper báo giảm trung bình shape/trend error so với baseline trước đó | Backbone thay thế cần đánh giá sau |
| Harpoon, ICLR 2026 | Differentiable guidance trong reverse diffusion | Objective cố định, gradient phụ thuộc state | Có state-dependent gradient | Official code/paper đánh giá range/category/composed constraints trên Adult/Default/Shoppers | Không được claim chung chung “constraint in denoising” |
| CTDF, ICAIF 2025 | Feasibility operator ở từng reverse step | Hard logical/numeric condition | Operator chạy mỗi step | Paper báo zero constraint violations trên finance tasks | Hard per-step enforcement không mới |
| SAGE, ACL 2026 | Sparse dependency guidance cho LLM generator | Có value-aware pseudo-feature graph | Không có diffusion trajectory | Paper báo F1 tăng 10% so với LLM baselines và giảm policy violations | Row-dynamic dependency graph không mới |
| LCATS, KDD 2026 | LLM discovery trước sinh, diagnosis/repair sau sinh | Diagnostic bucket theo output | Không | Official code có executable discovery + TabSyn + weighted MILP repair | LLM graph discovery và post-repair không mới |
| Dự án này | Typed edge-state graph trong middle reverse diffusion | Có | **Có: unreadable/inactive/satisfied/violated/uncertain** | Pilot cross-dataset ở Sections 5–7 | Claim chỉ đứng vững nếu hơn frozen-state và cạnh tranh post-repair |

Các con số giữa hàng không thể so trực tiếp vì work giải các constraint và dataset khác nhau. Bảng này dùng để định vị novelty; bảng local cùng protocol mới dùng để kết luận thắng/thua.

## 9. Trạng thái các hypothesis

| Hypothesis | Trạng thái | Bằng chứng |
| --- | --- | --- |
| Early predicted-clean row đủ để semantic guidance | **Bị bác bỏ** | Violation recall early: Shoppers `0.0145`, Adult `0.111`, Default `0.035` |
| Middle/late-middle state đủ đọc | **Được ủng hộ** | Recall tăng lên khoảng `0.61–0.90+`; Default late-middle `0.967` |
| Dynamic applicability là nguồn gain chính | **Bị bác bỏ** | Trên rows đổi applicability, dynamic và frozen bằng nhau |
| Dynamic satisfied/violated state hơn frozen-state | **Được ủng hộ có điều kiện** | 50-step thắng trên hai dataset; 30-step chỉ thắng Shoppers |
| Row-specific edge assignment có ý nghĩa | **Được ủng hộ mạnh** | Shuffled-row làm heldout error và drift xấu rõ, 583 actions |
| Edge-specific cutoff cần thiết | **Chưa được ủng hộ** | Common cutoff cạnh tranh hoặc tốt hơn edge-specific |
| Refresh graph ở mọi step cần thiết | **Bị bác bỏ** | Periodic k=2/k=4 gần bằng every-step trên Shoppers |
| Mid-sampling luôn hơn final repair | **Bị bác bỏ** | Shoppers final repair vẫn tốt nhất; Default mid-sampling tốt hơn final |
| Direct LLM judgement ở từng step đáng chi phí | **Bị bác bỏ trong executable setting** | NL→DSL→executor `1.0` execution equivalence với ~`1.85k` tokens; direct judgement kém hơn với ~`37.5k` tokens |

## 10. Đã đủ để phát triển method và viết bài chưa?

### Đủ để làm ngay

- Viết Introduction quanh failure của static graph state trong iterative generation.
- Formalize graph-state estimator và middle-step event-triggered action.
- Giữ graph construction như verified support pipeline, không đặt nó làm novelty duy nhất.
- Làm ablation theo bốn trục: state update, temporal region, refresh frequency, action trigger.
- Viết trước phần problem formulation, related work boundary và experimental protocol.

### Chưa đủ để tuyên bố SOTA

- Chưa có official Harpoon/CTDF/LCATS chạy cùng constraints và cùng TabDiff samples.
- Chỉ có hai dataset tạo actuation evidence hữu ích; Adult gần saturated.
- Chưa có 3 seeds và confidence intervals cho bảng cuối.
- Chưa có local α-Precision/β-Recall; utility mới chỉ có Shoppers MLE trong baseline suite.
- Chưa đánh giá privacy/membership risk sau intervention.
- Dynamic-vs-frozen effect trên Default chưa robust với số reverse steps.
- Independent verifier hiện vẫn cùng họ executable relations với graph catalog; cần một withheld relation set hoặc human-audited semantic set để tránh circular evaluation.

Đánh giá tổng thể: **đây đã là một research project hợp lý để phát triển tiếp, nhưng chưa phải một SOTA result**. Mức hiện tại tương ứng mechanism paper pilot: claim khoa học đã xuất hiện, còn benchmark breadth và đối chứng mạnh chưa hoàn chỉnh.

## 11. Method nên phát triển ở mức nào

Giữ method gọn với bốn component:

1. **Verified graph construction:** LLM/external knowledge đề xuất edge; train aggregates kiểm chứng support, directionality và exception rate; typed compiler tạo executable graph.
2. **Trajectory state estimator:** từ \(\hat{x}_0(t)\) ước lượng readability, applicability, satisfaction và uncertainty cho từng row–edge.
3. **Middle-step state policy:** refresh sparse checkpoints; chỉ phát action khi state đủ tin cậy hoặc transition đã ổn định.
4. **Typed action layer:** chọn minimal feasible edit, branch score hoặc abstain; inject corrected \(\hat{x}_0\) trở lại reverse direction.

Không nên thêm nhiều module agentic hoặc learned controller trước khi state-gate và temporal-window jobs trả lời được signal có ổn định hay không.

## 12. Jobs mới và hypothesis quyết định

| Job | Dataset/jobs | Hypothesis | Outcome cần để giữ claim |
| --- | --- | --- | --- |
| `1060241[0-5]` | Shoppers/Default × `current`, `stable`, `transition` | Confidence/persistence của violation state giảm action sai | `stable` hoặc `transition` giảm heldout error/drift mà không mất nhiều validity |
| `1060242[0-5]` | Shoppers/Default × early, middle, late; dynamic vs frozen-state | Dynamic graph có lợi nhất trong middle region | Dynamic–frozen gap dương ở middle, nhỏ/âm ở early và late |

Nếu cả hai hypothesis thất bại, nên hạ claim xuống temporal selective guidance. Nếu ít nhất một state gate tạo gain nhất quán trên hai dataset, có thể formalize method thành confidence-aware dynamic constraint graph.

## 13. Thực nghiệm bắt buộc tiếp theo

Ưu tiên theo information gain:

1. Đọc `1060241–1060242`; chọn duy nhất một state policy.
2. Chạy official Harpoon general-constraint suite trên Adult/Default/Shoppers để có một reproduced external baseline; không so trực tiếp với graph semantics, nhưng dùng để kiểm tra constraint/fidelity trade-off.
3. Tạo matched hard-constraint subset mà Harpoon/CTDF-style, final repair và dynamic graph đều thực thi được; cùng seed, action budget và verifier.
4. Thêm một dataset có process constraints và natural violation rate đủ lớn; Adult hiện không phù hợp cho main actuation table.
5. Sau khi method đóng băng, chạy 3 seeds và bootstrap confidence intervals trên Shoppers, Default và dataset mới.
6. Hoàn thiện α-Precision, β-Recall, downstream utility và privacy metrics.

## 14. Ước lượng thời gian

| Hạng mục | Thời gian full-time ước lượng |
| --- | ---: |
| Chốt state policy và temporal policy | 3–7 ngày |
| Matched Harpoon/CTDF-style/post-repair baseline | 1–2 tuần |
| Dataset thứ ba và verified relation set | 1–2 tuần |
| Full seeds, utility, fidelity, privacy, cost | 1–2 tuần |
| Viết, figure, error analysis và chỉnh reviewer threats | 2–3 tuần, làm song song |

Tổng hợp lý: khoảng **6–9 tuần full-time** để có bản paper draft có sức thuyết phục; thêm 2–4 tuần nếu official baseline hoặc dataset mới cần retrain/preprocessing lớn.

## 15. Artifacts

- Baseline local: `outputs/shoppers_baseline_metric_suite_1057841/`
- Dynamic/frozen 50-step: `outputs/constraint_state_pair_*_1059037_*/`
- Transition subgroup: `outputs/constraint_state_audit_*_1059038_*/`
- Robustness 30-step: `outputs/native_step_state_*_1059041_*/`
- Report tiến trình đầy đủ: `ACTIVE_GRAPH_THEO_THOI_GIAN_REPORT_20261006.md`
- Baseline detail: `SHOPPERS_BASELINES_AND_METRICS_20261007.md`
