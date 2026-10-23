# Báo cáo nhánh graph-conditioned mới và quyết định phương pháp

Ngày cập nhật: 2026-10-14

## 1. Phạm vi và câu hỏi quyết định

Pipeline `dynamic graph message guidance` trước ngày 2026-10-12 được đóng băng thành baseline. Nhánh này không tiếp tục tăng guidance weight hoặc thêm gate vào action layer cũ. Các phép thử mới trả lời ba câu hỏi tách biệt:

1. Có thể tạo graph đủ phong phú và ổn định chỉ từ train data, không cần LLM hay không?
2. Một learned graph-conditioned denoiser có hơn TabDiff và calibrated final repair hay không?
3. Nếu residual guidance thất bại, conditional resampling bằng chính TabDiff có tạo được validity gain mà ít phá fidelity hơn final repair hay không?

Claim boundary vẫn giữ chặt: constraint satisfaction trên graph tự khám phá không thể là metric duy nhất. Mọi method phải được đo bằng held-out graph error, Shape, Trend, precision/recall support, MLE utility và independent evaluation graph.

## 2. Bootstrap-Stable Conditional Graph

### 2.1. Protocol

`discover_bootstrap_conditional_hypergraph.py` thực hiện hoàn toàn trên train split:

1. Mine typed empirical implications.
2. Đánh giá lại candidate trên 32 bootstrap samples.
3. Loại edge support thấp, lift thấp, upper violation quantile cao hoặc rate variance lớn.
4. Gộp các rule có truth mask tương đương, tránh đếm `x=0` và `x<=0` như hai phát hiện khác nhau khi chúng giống nhau trên train.
5. Chỉ thêm conjunctive hyperedge khi conjunction có conditional gain so với parent tốt nhất.
6. Chọn graph bằng evidence, stability, node coverage và degree penalty; không ép graph thành DAG.

### 2.2. Kết quả discovery

| Dataset | Source bank | Stable pairwise | Redundant removed | Eligible conjunctions | Final graph | Nodes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Default | 105 | 52 | 6 | 6 | 16 edges, 2 hyperedges | 17 |
| Shoppers | 80 | 40 | 12 | 6 | 16 edges, 2 hyperedges | 15 |
| Diabetes | 22 stable pairwise | 22 | 6 | 0 | 16 pairwise | 11 |

Kết luận: graph generation không dùng LLM đã vượt mức ba rule thủ công và tạo được coverage/topology có ý nghĩa trên Default và Shoppers. Diabetes không có higher-order edge vượt threshold; pipeline đúng khi abstain thay vì ép graph phức tạp giả tạo. Tuy vậy đây vẫn là empirical dependency graph, không tự động đồng nghĩa với semantic truth.

Artifacts chính:

- `outputs/data_candidate_bank_{dataset}_1070390_*`
- `outputs/bootstrap_conditional_graph_{dataset}_1070443_*`

## 3. Learned graph-conditioned residual denoiser

### 3.1. Kiến trúc đã kiểm tra

Two-pass TabDiff gồm:

1. TabDiff dự đoán clean estimate \(\hat{x}_0(t)\).
2. Soft graph executor tính applicability, satisfaction, violation mass và uncertainty cho từng row/edge.
3. Edge encoder và topology propagation aggregate messages qua các edge dùng chung column.
4. Target-specific heads sinh bounded residual chỉ cho consequent columns.
5. Middle soft window giới hạn intervention vào vùng trajectory đã đọc được nhưng vẫn còn khả năng actuation.

Loss gồm clean reconstruction, applicability, relation satisfaction, train-rate calibration và proximity/KL tới backbone. Ba chế độ đã chạy: one-step với frozen backbone, truncated trajectory-unrolled training, và joint end-to-end fine-tuning.

### 3.2. One-step result

#### Shoppers

| Method | Seed | Held-out graph error ↓ | Shape error ↓ | Trend error ↓ | MLE AUC ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| TabDiff | 20261420 | 0.02015 | 1.6117 | 1.9840 | 0.8941 |
| Calibrated final repair | 20261420 | **0.00856** | **1.6087** | **1.9313** | 0.8889 |
| One-step graph adapter | 20261420 | 0.03531 | 2.2465 | 3.3925 | 0.8932 |
| TabDiff | 20261421 | 0.02101 | **1.6000** | **2.0884** | **0.9057** |
| Calibrated final repair | 20261421 | **0.00866** | 1.6432 | 2.2003 | 0.9005 |
| One-step graph adapter | 20261421 | 0.03539 | 1.7368 | 2.9291 | 0.9025 |

One-step adapter bị bác bỏ trên Shoppers. No-topology ít xấu hơn topology-aware model, nên graph propagation không phải nguồn gain.

#### Diabetes

| Method | Seed | Held-out error ↓ | Shape error ↓ | Trend error ↓ | MLE AUC ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| TabDiff | 20261420 | 0.00488 | **1.9434** | **6.2426** | **0.6538** |
| Final repair | 20261420 | **0.00084** | 1.9507 | 7.1734 | 0.6503 |
| One-step adapter | 20261420 | 0.00312 | 2.0399 | 7.7501 | 0.6511 |
| TabDiff | 20261421 | 0.01572 | **1.8848** | **7.5746** | 0.6502 |
| Final repair | 20261421 | **0.00266** | 1.8854 | 7.6986 | **0.6521** |
| One-step adapter | 20261421 | 0.00963 | 2.0904 | 8.4887 | 0.6506 |

Adapter giảm held-out error so với TabDiff 36.1% và 38.7%, đồng thời giảm số final-repair actions 25→14 và 40→26. Tuy nhiên nó vẫn thua final repair rõ rệt và làm Shape/Trend xấu hơn. Đây là mechanism signal, không phải main result.

### 3.3. Trajectory-unrolled result

| Variant, Shoppers | TabDiff held-out | Final repair | Graph-conditioned | Shape | Trend |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rollout-4, seed 20261420 | 0.02015 | 0.00856 | 0.07033 | 5.4469 | 10.8522 |
| Rollout-4, seed 20261421 | 0.02101 | 0.00866 | 0.06615 | 4.9407 | 9.2073 |
| Rollout-4, no topology | 0.02015 | 0.00856 | 0.03541 | 3.8576 | 5.1204 |

BSCG + rollout cũng thất bại: topology đạt held-out error 0.06762, no-topology 0.04242, so với TabDiff 0.02693 và final repair 0.00988 trong run đó. Diabetes rollout gặp NaN trong differentiable categorical/numeric rollout; không có bằng chứng từ Shoppers để biện minh cho việc tiếp tục sửa numerical stability.

### 3.4. Joint end-to-end result

Các cấu hình head LR \(10^{-6}\), \(3\times10^{-6}\), full LR \(5\times10^{-7}\) và seed confirmation đều thất bại trên Shoppers:

- graph held-out error khoảng 0.058–0.074, trong khi TabDiff khoảng 0.019–0.021 và final repair khoảng 0.009–0.011;
- Shape error khoảng 4.7–5.2, trong khi TabDiff khoảng 1.55–1.75;
- Trend error khoảng 8.2–10, trong khi TabDiff khoảng 2.

Kết luận phương pháp: learned residual adapter, topology propagation, truncated unrolling và joint fine-tuning đều bị loại khỏi main method. Không tiếp tục hyperparameter tuning vì failure nhất quán và lớn hơn sai số seed.

## 4. Phương pháp mới đang chạy: Dynamic Graph Conditional Repainting

### 4.1. Motivation

Failure chung của guidance cũ là action đẩy state theo một correction vector không nhất thiết nằm trên data manifold. Dynamic Graph Conditional Repainting thay action bằng conditional generation của chính TabDiff:

1. Lấy predicted-clean preview tại middle timestep.
2. Evaluate graph theo từng row; chỉ các hard/near-hard edge có calibration rate dưới threshold mới được phép mở action.
3. Reopen consequent columns của violated active edges; giữ context columns cố định.
4. TabDiff conditional-impute các columns này qua một reverse process mới.
5. Sinh bốn stochastic proposals; base trajectory luôn là proposal 0 để method có thể abstain.
6. Chọn row-wise bằng graph energy + train-only kNN density + categorical rarity + normalized edit cost.

Đối chứng `trigger=final` dùng cùng graph, proposal budget và selector nhưng graph được đọc trên final row. Vì vậy middle-vs-final là matched comparison, không lẫn với khác biệt compute/action family.

### 4.2. Giả thuyết

- H-R1: diffusion inpainting đạt constraint gain thấp hơn hoặc bằng deterministic repair, nhưng giữ Shape/Trend/utility tốt hơn vì proposal quay lại learned manifold.
- H-R2: middle graph state có ích nếu nó phát hiện đúng row/target trước khi final state khóa vào mode xấu; khi preview quá nhiễu, final-trigger sẽ thắng.
- H-R3: BSCG graph không dùng LLM đủ để kích hoạt cơ chế trên nhiều dataset; actionable graph cũ là graph-source ablation.
- H-R4: nếu repaint + calibrated repair cần ít final actions hơn repair-only và có fidelity tốt hơn, middle intervention có incremental value ngay cả khi repair vẫn cần để certify output.

### 4.2.1. Positioning và claim dự kiến

Nếu kết quả được xác nhận, contribution không nên viết là “đưa constraint vào diffusion” vì Harpoon/CTDF đã bao phủ vùng đó. Claim chính xác hơn là:

> Một constraint graph được khám phá và hiệu chỉnh trên train data được tái đánh giá theo từng row và reverse state; active graph không trực tiếp chiếu record vào feasible set mà quyết định conditional repaint mask, proposal scope và thời điểm tái sinh. Middle-state repaint có thể cải thiện đồng thời independent semantic error và data fidelity so với matched final repair/final inpainting.

Ranh giới với các work sát nhất:

| Work family | Constraint/state | Actuation | Khác biệt cần chứng minh |
| --- | --- | --- | --- |
| Harpoon-style guidance | Constraint specification cố định; gradient thay đổi theo state | Differentiable loss trong sampling | Graph applicability và target mask thay đổi theo row/time; không yêu cầu differentiable constraint |
| CTDF/feasibility projection | Hard affine/logical constraints | Feasibility operator mỗi reverse step | Stochastic conditional repaint trên learned manifold; có abstention và empirical hardness |
| LCATS/Zhao repair | Graph/rule đã verify trên final samples | Deterministic post-generation repair | Active graph được đọc từ intermediate clean estimate và mở lại subspace trước final output |
| SCTab-Diff | Semantic prior cố định condition denoiser | Learned prior conditioning | Per-row/per-time active subgraph quyết định proposal scope, không chỉ cung cấp global prior |
| Image repaint/lookahead | Preview và local resampling | Rollback/branching | Typed mixed-table graph, categorical/numeric target masks và held-out distribution calibration |

LLM không còn là thành phần bắt buộc của main mechanism. Nó có thể là một graph-source ablation hoặc bổ sung external semantics; BSCG là đường chính nếu independent evaluation tiếp tục tốt. Điều này làm claim gọn hơn và tránh phụ thuộc API, nhưng semantic audit vẫn cần để không biến stable association thành hard knowledge.

### 4.3. Protocol thực nghiệm đã hoàn thành

Tất cả các bảng dưới đây dùng cùng TabDiff checkpoint, cùng seed `20261440`, 2,048 rows và independent evaluation graph `actionable_union_eval`; graph dùng để guidance là BSCG được khám phá chỉ từ train split. Shape/Trend là phần trăm sai số nên thấp hơn tốt hơn; MLE AUC cao hơn tốt hơn. `Actionable error` chỉ tính các relation đủ điều kiện actuation, còn `held-out graph error` tính toàn catalog độc lập.

| Jobs | Nội dung |
| --- | --- |
| `1070752`, `1070838`, `1070875` | Fixed middle/final trigger trên Shoppers, Default, Diabetes; independent metrics |
| `1070780→1070781`, `1070822→1070830` | Proposal budget, timestep, rounds, density/edit/rarity weight trên Shoppers |
| `1070793→1070843` | Matched final-state repaint với cùng graph/action family |
| `1070844` | Paired row-bootstrap cho Shoppers |
| `1070835→1070836` | Transfer density selector sang Default |
| `1070867→1070868` | Multi-state trajectory proposal pool: latest, persistent và union activation |
| `1071179→1071180` | Routed temporal graph: graph chọn cả columns và proposal timestep; đang chạy |

### 4.4. Fixed-checkpoint repaint

#### Shoppers

| Method | Held-out error ↓ | Actionable error ↓ | Shape error ↓ | Trend error ↓ | MLE AUC ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| TabDiff | 0.020420 | 0.015349 | 2.0633 | 2.7094 | 0.8929 |
| Calibrated final repair | 0.010846 | 0.007681 | 2.0202 | 2.5408 | 0.8947 |
| Mid-0.25 + repair, 4 proposals | 0.010304 | 0.007094 | 2.0120 | 2.5141 | **0.8976** |
| **Mid-0.25 + repair, 8 proposals, density=1** | 0.010229 | 0.006977 | **2.0093** | 2.5130 | 0.8959 |
| Mid-0.25 + repair, density=2 | 0.010171 | 0.006917 | 2.0147 | 2.5115 | 0.8937 |
| Matched final-state repaint + repair | **0.009872** | **0.006578** | 2.0174 | **2.5123** | 0.8935 |

Density=1 strict-dominate deterministic final repair trên năm cột trong bảng: held-out error giảm 5.69%, actionable error giảm 9.17%, Shape giảm 0.54%, Trend giảm 1.09%, AUC tăng 0.00116. Density=2 đạt semantic gain lớn nhất, gần 10% actionable, nhưng AUC thấp hơn final repair 0.0010. Chỉ tăng số branches không giúp; gain đến từ train-density-aware selection.

Matched final-state repaint vẫn tốt hơn middle-state repaint về hai graph errors. Middle tốt hơn Shape và MLE AUC, nên đây là Pareto trade-off chứ không phải bằng chứng middle luôn hơn final. Paired bootstrap của cấu hình 4-proposal cho gain 5.0% so với repair-only nhưng CI 95% `[-0.001262, 0.002186]`, `P(gain>0)=0.713`; chưa có ý nghĩa thống kê. Bootstrap này chưa áp dụng cho density=1 và không bao gồm seed/graph uncertainty.

#### Default

| Method | Held-out error ↓ | Actionable error ↓ | Shape error ↓ | Trend error ↓ | MLE AUC ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| TabDiff | 0.024583 | 0.020195 | 1.8965 | 2.8518 | 0.7455 |
| Calibrated final repair | 0.010663 | 0.007008 | **1.7784** | 2.6258 | **0.7450** |
| Mid-0.25 repaint + repair | 0.009942 | 0.005847 | 1.7737 | 2.6258 | 0.7364 |
| **Mid-0.40 repaint + repair** | **0.009499** | **0.005133** | 1.7817 | **2.6213** | 0.7390 |
| Matched final-state repaint + repair | 0.010631 | 0.006956 | 1.7708 | 2.6258 | 0.7438 |

Middle-0.40 giảm actionable error 26.75% và held-out error 10.92% so với final repair; matched final-state gần như không cải thiện. Đây là evidence mạnh nhất hiện tại rằng intermediate state chứa signal hữu ích. Tuy nhiên AUC giảm 0.0059, nên chưa thể gọi là strict improvement. Density-weight transfer không tăng thêm trên Default: density 1 và 2 cùng đạt held-out `0.010026`, actionable `0.005982`, yếu hơn cấu hình 4-proposal ở trên.

#### Diabetes

Final repair đã đưa actionable error từ `0.021449` xuống `0.002573`. Mid-0.25 + repair đạt `0.002580`, final repaint đạt `0.002579`; khác biệt thực tế bằng không và hơi xấu hơn. Shape/Trend cải thiện rất nhỏ nhưng AUC giảm. Dataset này đã saturated bởi deterministic repair, nên bác bỏ H-R2/H-R4 cho Diabetes và không tiếp tục tuning.

### 4.5. Multi-state trajectory graph

Nhánh này quan sát hai predicted-clean states của cùng trajectory tại (t≈0.40) và (t≈0.25). Tổng proposal budget vẫn là tám: bốn proposal từ mỗi state. Ba activation policies trả lời ba câu hỏi khác nhau:

- `latest`: lấy active graph ở 0.25 nhưng cho selector chọn proposal từ cả hai state;
- `persistent`: chỉ action nếu edge bị vi phạm ở cả hai state;
- `union`: action nếu edge bị vi phạm ở ít nhất một state.

#### Default

| Method + final certification | Held-out error ↓ | Gain | Actionable error ↓ | Gain | Shape ↓ | Trend ↓ | MLE AUC ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Final repair | 0.010663 | — | 0.007008 | — | 1.7784 | 2.6258 | 0.7450 |
| Trajectory latest | 0.010280 | 3.6% | 0.006392 | 8.8% | 1.7733 | 2.6258 | 0.7402 |
| **Trajectory persistent** | **0.009701** | **9.0%** | **0.005458** | **22.1%** | **1.7727** | 2.6258 | **0.7439** |
| Trajectory union | 0.009362 | 12.2% | 0.004913 | 29.9% | 1.7906 | 2.6258 | 0.7411 |

`Persistent` là cấu hình cân bằng: Shape tốt hơn repair-only, Trend bằng nhau, AUC chỉ giảm 0.00103 trong khi actionable error giảm 22.1%. `Union` tối đa hóa validity nhưng trả giá khoảng 0.69% tương đối ở Shape và 0.00388 AUC. Đây là evidence cho một semantic–fidelity frontier có thể điều khiển, không phải một cấu hình thắng mọi metric.

#### Shoppers

`Persistent` chỉ giảm held-out error 2.44% và actionable error 1.51% so với repair-only; Shape/Trend tốt hơn nhẹ và AUC gần như bằng nhau. Fixed middle-0.25 density selector vẫn tốt hơn. Vì vậy giả thuyết “multi-state luôn hơn fixed middle” bị bác bỏ. Diễn giải phù hợp là temporal persistence có ích khi quan hệ ổn định theo chuỗi trạng thái như Default, nhưng không cần thiết cho các relation local count–duration của Shoppers.

### 4.6. Routed temporal action: ablation bị loại

`Routed` không gửi mọi row qua mọi checkpoint. Với mỗi row-target, active graph chỉ mở proposal ở timestep nơi edge đó thực sự active; edge persistent có hai nguồn proposal, edge transient chỉ có một. Đây là test trực tiếp cho việc graph quyết định `(row, relation, target columns, intervention time)`.

| Dataset | Method | Held-out error ↓ | Actionable error ↓ | Shape ↓ | Trend ↓ | MLE AUC ↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Shoppers | Final repair | 0.010846 | 0.007681 | 2.0202 | 2.5408 | 0.8947 |
| Shoppers | Routed + repair | 0.010564 | 0.007120 | 2.0147 | 2.5117 | 0.8940 |
| Default | Final repair | **0.010663** | **0.007008** | **1.7784** | 2.6258 | 0.7450 |
| Default | Routed + repair | 0.010813 | 0.007251 | 1.7863 | 2.6258 | **0.7513** |

Routed có gain nhỏ trên Shoppers và fidelity/utility gain nhưng semantic loss trên Default; nó không hơn `persistent`. Nó giảm proposal rows của Shoppers từ 680 xuống 484 và Default xuống 488, nhưng chia thành 13/29 nhóm nhỏ làm runtime tăng lên khoảng 72/242 giây. Kết luận: row-wise timing route hợp lý về mặt biểu diễn nhưng implementation phân mảnh quá nhiều, không giữ làm main method. Nó chỉ là ablation cho thấy tối ưu số proposal rows không đồng nghĩa tối ưu wall time.

### 4.7. Baseline evidence và phạm vi so sánh

Đã có ba lớp baseline, nhưng không được gộp số một cách thiếu kiểm soát:

1. **Official Harpoon rerun:** public code/checkpoint, 200 reverse steps và năm trials trên Adult, Default, Shoppers. Đây là bằng chứng baseline chạy được, nhưng backbone, preprocessing, request-conditioned subset và metric scale khác pipeline hiện tại; chỉ đặt ở bảng riêng.
2. **Matched Harpoon-style proxy:** port graph-energy guidance vào cùng TabDiff và natural graph. Trên ba paired seeds cũ, held-out error là `0.0179`, so với `0.0120` của dynamic graph baseline và `0.0161` của final repair; Shape/Trend/AUC chênh nhỏ. Đây là matched proxy, không gọi là official reproduction.
3. **CDD-inspired projection:** trên exact Harpoon category request, full/delayed projection đạt 0% request violation; delayed schedule có conditional AUC `0.7325`, final repair `0.7142`. Kết quả này xác nhận projection rất mạnh khi constraint cứng đã biết, nhưng không trả lời bài toán active graph thay đổi theo row/time.

Main table mới chỉ được so sánh bằng local rerun cùng split, checkpoint, sample count và evaluator. Số official Harpoon/SCTab-Diff từ protocol khác phải để ở bảng tham khảo, không dùng để tính phần trăm gain của method.

### 4.8. Frozen three-seed confirmation: temporal-persistent không đủ làm main

Sau screen seed `20261440`, policy `persistent`, hai checkpoints `.40,.25`, tám proposals và density weight 1.0 được freeze trước khi chạy ba seed mới `20261441–20261443`, 4,096 rows/seed.

| Dataset | Metric | Final repair | Persistent + repair | Favorable delta | Wins |
| --- | --- | ---: | ---: | ---: | ---: |
| Default | Shape error | 1.710130 | 1.708095 | +0.002035 ± 0.003668 | 2/3 |
| Default | Trend error | 2.037270 | 2.037270 | 0 | 0/3 |
| Default | MLE AUC | 0.727069 | 0.728513 | +0.001444 ± 0.004405 | 2/3 |
| Default | Held-out error | 0.010363 | 0.010035 | +0.000327 ± 0.000572 | 2/3 |
| Default | Actionable error | 0.006762 | 0.006235 | +0.000527 ± 0.000921 | 2/3 |
| Shoppers | Shape error | 1.558273 | 1.553442 | +0.004831 ± 0.010589 | 2/3 |
| Shoppers | Trend error | 1.861118 | 1.843778 | **+0.017340 ± 0.012413** | **3/3** |
| Shoppers | MLE AUC | 0.896501 | 0.898551 | +0.002049 ± 0.005243 | 1/3 |
| Shoppers | Held-out error | 0.009717 | 0.009698 | +0.000019 ± 0.000558 | 2/3 |
| Shoppers | Actionable error | **0.005739** | 0.005776 | **−0.000037 ± 0.000535** | 2/3 |

Default actionable relative gain trung bình khoảng 7.8%, thấp hơn nhiều so với 22.1% ở screen seed và không thắng cả ba seed. Shoppers semantic gain trung bình bằng không; chỉ Trend cải thiện nhất quán. Do đó temporal-persistent không đủ làm main result. Nó được giữ như một conservative ablation cho thấy multi-state activation có thể giữ hoặc cải thiện distribution metrics, nhưng không chứng minh validity gain tổng quát.

Artifacts: `outputs/trajectory_persistent_confirm_aggregate_1071190/{RESULTS.md,paired_gains.csv,per_seed_metrics.csv}`.

### 4.9. Conditional-density action layer: candidate mới

Global kNN của repaint baseline chỉ hỏi proposal có nằm ở vùng train dày hay không; nó không hỏi target value có phù hợp với các cột đang giữ cố định của chính row đó hay không. `ConditionalProposalScore` tìm 64 train neighbors theo các non-target numeric columns, rồi chấm numeric target bằng local robust support và categorical target bằng smoothed neighbor frequency. Selector vẫn không dùng held-out data và base row vẫn là abstention candidate.

Single-seed screen, 2,048 rows:

| Dataset | Method + repair | Held-out error ↓ | Actionable error ↓ | Shape ↓ | Trend ↓ | MLE AUC ↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Shoppers | Final repair | 0.010846 | 0.007681 | 2.0202 | 2.5408 | **0.8947** |
| Shoppers | Conditional density, weight 1 | **0.010038** | **0.006799** | **2.0120** | **2.5110** | 0.8937 |
| Default | Final repair | 0.010663 | 0.007008 | **1.7784** | 2.6258 | 0.7450 |
| Default | Conditional density, weight 1 | **0.009418** | **0.005003** | 1.7805 | **2.6197** | **0.7480** |

So với final repair, actionable gain là 11.5% trên Shoppers và 28.6% trên Default; held-out gain là 7.5% và 11.7%. Trên Default, method cũng tốt hơn fixed-middle cũ về mọi cột chính. Trên Shoppers, Shape/Trend tốt hơn nhưng AUC giảm 0.0011 và detection giảm, nên chưa phải strict dominance.

Điểm method đáng giữ không phải thêm một weight, mà là định nghĩa action selection có hệ thống: trong nhiều valid corrections, chọn correction có conditional support cao nhất dưới context không bị sửa.

Ba seed mới `20261444–46`, 4,096 rows/seed, weight 1.0 được freeze cho cả hai dataset:

| Dataset | Metric | Final repair | Conditional density | Favorable delta | Wins |
| --- | --- | ---: | ---: | ---: | ---: |
| Default | Shape error | 1.867987 | 1.864706 | +0.003281 ± 0.007781 | 2/3 |
| Default | Trend error | **2.100226** | 2.102708 | −0.002482 ± 0.002405 | 0/3 |
| Default | MLE AUC | **0.729711** | 0.727516 | −0.002195 ± 0.006094 | 1/3 |
| Default | Held-out error | 0.009140 | **0.008474** | **+0.000666 ± 0.000479** | **3/3** |
| Default | Actionable error | 0.006976 | **0.005903** | **+0.001073 ± 0.000772** | **3/3** |
| Shoppers | Shape error | 1.620010 | 1.613643 | +0.006366 ± 0.027930 | 1/3 |
| Shoppers | Trend error | 1.960198 | **1.949970** | **+0.010228 ± 0.006639** | **3/3** |
| Shoppers | MLE AUC | 0.895676 | **0.902724** | **+0.007048 ± 0.003109** | **3/3** |
| Shoppers | Held-out error | 0.009818 | **0.009355** | +0.000462 ± 0.000711 | 2/3 |
| Shoppers | Actionable error | 0.005432 | **0.005015** | +0.000417 ± 0.000579 | 2/3 |

Relative gain trung bình là 15.4% actionable và 7.3% held-out trên Default; 7.7% actionable và 4.7% held-out trên Shoppers. Default semantic gain thắng cả ba seed nhưng AUC/Trend giảm nhẹ; Shoppers Trend/AUC thắng cả ba seed và semantic error thắng hai seed. Đây là kết quả ổn định nhất của nhánh mới, nhưng vẫn là Pareto improvement chứ không strict dominance trên mọi metric. Với chỉ ba seed, mean±std và win count chưa đủ để claim statistical significance.

Paired row-bootstrap 5,000 draws/seed cho full held-out error vẫn có CI 95% chứa 0 ở cả sáu runs. Default có observed relative gains `4.53%, 13.62%, 3.92%` với `P(gain>0)=0.610, 0.865, 0.608`; Shoppers là `10.50%, −2.93%, 6.09%`. Vì vậy kết luận đúng là consistent positive mean và sign consistency trên Default, chưa phải significance trong từng seed. Bootstrap này cũng chưa tính graph-discovery uncertainty.

Artifacts: `outputs/conditional_density_confirm_aggregate_1071211/{RESULTS.md,paired_gains.csv,per_seed_metrics.csv}`.
Row-bootstrap: `outputs/conditional_density_bootstrap_{dataset}_s{seed}_1071229_*`.

### 4.10. External constraint suite: Harpoon requests

Để tránh chỉ đánh giá trên graph tự khám phá, conditional-density method được chạy trên hai exact Default requests lấy từ public Harpoon protocol, 4,096 rows và cùng seed `20261376` với matched baselines.

#### Numeric range request: `AGE ≥ 50`

| Method | Violation % ↓ | Shape ↓ | Trend ↓ | Alpha ↑ | Beta ↑ | Conditional AUC ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TabDiff | 92.041 | 11.9868 | 10.8384 | 78.513 | 16.456 | 0.6304 |
| Final repair | **0** | 11.4574 | 10.8384 | 93.023 | 43.445 | 0.6241 |
| **Conditional density + repair** | **0** | **10.3212** | 10.8384 | **93.226** | **49.071** | **0.6607** |
| Harpoon-style strong proxy | **0** | **10.0792** | **7.8797** | 93.203 | 46.667 | **0.6715** |

Conditional repaint hơn final repair rõ rệt vì range constraint có nhiều valid corrections; chọn AGE value theo local context tốt hơn ép tất cả row về boundary 50. Tuy nhiên nó chưa beat Harpoon-style strong proxy về Shape, Trend hoặc AUC. Vì vậy external result hỗ trợ action-selection hypothesis nhưng không hỗ trợ SOTA claim.

#### Singleton categorical request: `EDUCATION = 3`

| Method | Violation % ↓ | Shape ↓ | Trend ↓ | Beta ↑ | Conditional AUC ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Final repair | **0** | 6.6216 | 4.5816 | 47.003 | 0.7038 |
| Conditional density + repair | **0** | 6.6216 | 4.5816 | 47.003 | 0.7038 |
| CDD full | **0** | **4.7070** | **3.5252** | 46.909 | 0.7054 |
| CDD delayed | **0** | 5.5590 | 3.7373 | **48.414** | **0.7325** |

Conditional repaint + repair đúng bằng final repair: chỉ có một valid category nên không tồn tại ambiguity để selector giải quyết. CDD projection thắng rõ. Đây là negative boundary nên nêu thẳng: method nhắm vào context-dependent choice giữa nhiều feasible actions, không nhắm thay thế exact categorical projection.

Artifacts: `outputs/conditional_density_harpoon_eval_{range,category}_1071225_*`.

### 4.11. Matched activation/action ablation: state nào làm việc gì?

Batch ngày 14/10/2026 giữ nguyên backbone, graph, seeds `20261444–46`, 4,096 rows/seed, tám proposals và independent evaluation graph. Hai quyết định được tách riêng:

- **activation state:** middle preview hay final row quyết định row-edge nào active;
- **proposal state:** correction được conditional diffusion sinh từ middle state hay final state.

| Dataset | Variant + repair | Shape ↓ | Trend ↓ | MLE AUC ↑ | Held-out error ↓ | Actionable error ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Default | Final repair | 1.867987 | 2.100226 | 0.729711 | 0.009140 | 0.006976 |
| Default | Middle / global | 1.861972 | 2.100226 | 0.731605 | 0.008473 | 0.005900 |
| Default | Final / global | 1.857815 | 2.100226 | 0.732926 | 0.008788 | 0.006408 |
| **Default** | **Middle activate, final propose / global** | **1.860615** | **2.100226** | **0.733804** | **0.008396** | **0.005777** |
| Default | Middle / conditional | 1.864706 | 2.102708 | 0.727516 | 0.008474 | 0.005903 |
| Default | Final / conditional | 1.858154 | 2.100226 | 0.735037 | 0.008801 | 0.006430 |
| Default | Middle activate, final propose / conditional | 1.862650 | 2.102708 | 0.730656 | 0.008398 | 0.005779 |
| Shoppers | Final repair | 1.620010 | 1.960198 | 0.895676 | 0.009818 | 0.005432 |
| Shoppers | Middle / global | 1.614480 | 1.952261 | 0.900104 | 0.009518 | 0.005171 |
| **Shoppers** | **Final / global** | 1.617492 | 1.956069 | **0.900772** | **0.008829** | **0.004443** |
| Shoppers | Middle activate, final propose / global | 1.614780 | 1.952209 | 0.898696 | 0.009471 | 0.005213 |
| Shoppers | Middle / conditional | 1.613643 | 1.949970 | 0.902724 | 0.009355 | 0.005015 |
| Shoppers | Final / conditional | 1.615216 | 1.955703 | 0.897360 | 0.008768 | 0.004296 |
| Shoppers | Middle activate, final propose / conditional | 1.613475 | 1.949594 | 0.902623 | 0.009407 | 0.005066 |

Kết luận nhân quả:

1. Conditional target score không phải nguồn gain chính. Ở middle, nó gần như bằng global selector trên Default và chỉ thêm gain nhỏ trên Shoppers. Proposal search + graph certification là thành phần chính.
2. Middle không luôn hơn final. Middle activation tốt hơn cho semantic error của Default; final activation tốt hơn rõ trên Shoppers.
3. Tách activation khỏi proposal state có ích trên Default: middle activate + final propose giảm actionable error thêm khoảng 2.1% tương đối so với middle propose, đồng thời cải thiện Shape và AUC.
4. Final conditional resampling là đối chứng rất mạnh. Trên Shoppers nó đạt semantic error thấp nhất; vì vậy final repair yếu không còn là lời giải thích cho gain của method.

Artifact tái chạy: `outputs/decoupled_activation_aggregate_20261014/{RESULTS.md,summary.csv,gains_vs_final_repair.csv,per_seed_metrics.csv}`.

### 4.12. Vì sao timing khác nhau theo dataset?

Active graph transition được audit trực tiếp trên ba paired seeds bằng cùng row trajectories. Jaccard tính trên tập row-edge active/violated ở middle và final.

| Dataset | Active Jaccard ↑ | Rows đổi active set | Violated Jaccard ↑ | Rows đổi violated set | Middle-only violated pairs | Final-only violated pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Default | 0.8521 | 8.18% | **0.2201** | 3.75% | **153.0** | 36.0 |
| Shoppers | **0.9847** | 2.25% | **0.8452** | 0.60% | 4.7 | **20.3** |

Default có nhiều precursor violations chỉ xuất hiện ở middle; nếu đợi final, active graph bỏ lỡ chúng. Shoppers có graph gần như ổn định và nhiều final-only violations hơn, nên defer tới final tránh đánh giá preview không cần thiết. Đây là evidence định lượng cho hypothesis “timing phải relation/state-dependent”, thay vì chọn một timestep toàn cục.

Artifact: `outputs/active_graph_transition_audit_20261014/{RESULTS.md,summary.csv,per_seed.csv}`.

### 4.13. Calibrated temporal edge router: main candidate đã freeze

Router được học từ một unlabeled calibration trajectory seed `20261440`, không dùng held-out rows hoặc metric. Với candidate states `.40`, `.25`, `final`, mỗi edge dùng:

`precursor surplus_e(t) = # violated only at t − # violated only at final`.

Nếu surplus lớn nhất dương, edge lưu activation từ state đó nhưng correction proposal được sinh từ final state sạch hơn; nếu không, edge defer hoàn toàn tới final. Graph certification và global train-density selector sau đó chọn proposal hoặc abstain. Calibration tự chọn `.40` cho 7/7 Default edges và `final` cho 7/7 Shoppers edges.

| Dataset | Method | Shape ↓ | Trend ↓ | MLE AUC ↑ | Held-out error ↓ | Actionable error ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Default | TabDiff | 1.904991 | 2.141658 | 0.731184 | 0.018517 | 0.018228 |
| Default | Final repair | 1.867987 | 2.100226 | 0.729711 | 0.009140 | 0.006976 |
| **Default** | **Temporal router + repair** | **1.860615** | **2.100226** | **0.733804** | **0.008396** | **0.005777** |
| Shoppers | TabDiff | 1.618349 | **1.951804** | 0.897537 | 0.017646 | 0.012819 |
| Shoppers | Final repair | 1.620010 | 1.960198 | 0.895676 | 0.009818 | 0.005432 |
| **Shoppers** | **Temporal router + repair** | **1.617492** | 1.956069 | **0.900772** | **0.008829** | **0.004443** |

So với final repair, router giảm held-out/actionable error `8.14%/17.19%` trên Default và `10.07%/18.20%` trên Shoppers; cả hai semantic metrics thắng `3/3` seeds ở cả hai dataset. So với TabDiff, actionable error giảm `68.3%` và `65.3%`. Common fidelity không bị đổi lấy semantic gain: mean Shape và AUC đều tốt hơn TabDiff/final repair ở cả hai dataset; Default Trend tốt hơn TabDiff và bằng repair; Shoppers Trend chỉ kém TabDiff `0.0043` nhưng tốt hơn repair.

Đây là main candidate hiện tại vì nó dùng một policy cố định và được calibration tự chọn, thay vì chọn method tốt nhất sau khi nhìn held-out result. Jobs: sampler `1079827`, evaluator `1079828`. Code: `scripts/calibrated_temporal_router_repaint.py`.

Aggregate artifact: `outputs/calibrated_temporal_router_aggregate_20261014/{RESULTS.md,paired_gains.csv,per_seed_metrics.csv}`.

Chi phí hiện chưa tối ưu: trên A40, mean sampling/action runtime là khoảng `49.7 s` cho Shoppers và `158.2 s` cho Default, so với base TabDiff `1.28 s/1.36 s`. Router tạo trung bình `1,224/1,320` proposal rows và thực sự đổi `105/124` rows. Vì implementation gọi `sample_impute` theo từng target group, runtime 39–116× là giới hạn lớn; paper phải report và batching proposal groups là optimization bắt buộc trước claim practical efficiency.

Phạm vi claim phải chính xác: đây là **two-pass lookahead/conditional resampling**. Middle preview quyết định active graph và target columns, nhưng action được sinh bằng conditional diffusion pass riêng từ final state; chưa phải một single-pass update rồi rejoin chính reverse trajectory ban đầu.

## 5. Trạng thái dự án sau batch này

Đã đủ bằng chứng để khóa problem, main mechanism và bắt đầu viết method/experiment section; chưa đủ để claim SOTA tổng quát:

- Graph discovery không dùng LLM tạo 16-edge graph trên ba dataset và đủ để chạy end-to-end, nhưng cần independent semantic audit trước claim knowledge quality.
- Residual learned injection, topology propagation và joint end-to-end fine-tuning bị bác bỏ rõ ràng.
- Calibrated temporal edge routing là main candidate: một calibration-only policy tự chọn middle activation cho Default và final activation cho Shoppers, giảm actionable error 17–18% so với final repair và thắng 3/3 seeds mà không hy sinh mean Shape/AUC.
- Context-conditional selector không còn là contribution chính; global density đủ mạnh cho temporal router. Đây là negative ablation giúp method gọn hơn.
- External Harpoon test xác nhận gain trên ambiguous numeric range nhưng không có gain trên singleton categorical request; Harpoon-style strong/CDD vẫn thắng ở setting tương ứng.
- Middle-state benefit có bằng chứng structure-dependent: graph churn cao/precursor surplus dương trên Default, nhưng graph ổn định và late-emerging trên Shoppers. Router biến sự không đồng nhất này thành decision rule thay vì dataset-specific manual choice.
- Naive multi-state proposal pooling không ổn định và không được giữ. Temporal router tốt hơn vì calibration chọn một activation state theo từng edge thay vì cho selector một pool lớn dễ overfit score proxy.
- Không có bằng chứng hiện tại cho SOTA tổng quát. Tuy nhiên result chính không còn là single-seed signal hay metric-only tuning: router được học từ seed calibration tách biệt và xác nhận trên ba sampling seeds mới.

Code tái chạy:

- `scripts/discover_bootstrap_conditional_hypergraph.py`
- `scripts/graph_conditioned_residual_denoiser.py`
- `scripts/dynamic_graph_conditional_repaint.py`
- `scripts/trajectory_graph_conditional_repaint.py`
- `scripts/conditional_density_graph_repaint.py`
- `scripts/aggregate_persistent_trajectory_confirm.py`
- `scripts/paired_graph_error_bootstrap.py`
- `scripts/calibrated_temporal_router_repaint.py`
- `scripts/audit_active_graph_transition.py`
- `scripts/aggregate_decoupled_activation.py`
- `slurm/564_dynamic_graph_repaint_screen.sbatch`
- `slurm/565_dynamic_graph_repaint_eval.sbatch`
- `slurm/567_shoppers_repaint_refine.sbatch` đến `slurm/591_conditional_density_paired_bootstrap.sbatch`
- `slurm/592_global_density_matched.sbatch` đến `slurm/606_calibrated_temporal_router_eval.sbatch`

## 6. Disk cleanup

Để phục hồi Ceph quota, đã xóa:

- khoảng 3.1 GB model/adapter checkpoints trung gian của các nhánh condition adapter, semantic-prior proxy và end-to-end đã có kết luận/evaluation;
- khoảng 8.8 GB `~/.cache/pip` có thể tải lại.

Không xóa backbone checkpoints, datasets, sample CSV, metric tables, metadata hay logs dùng để tái kiểm tra kết luận. Các checkpoint trung gian đã xóa không thể phục hồi trực tiếp nhưng có thể tái tạo từ script/config/seed còn giữ.

## 7. Các giới hạn còn chặn claim paper

1. **Graph truth:** BSCG tìm stable dependency, không chứng minh semantic necessity. Một edge có thể là association hoặc artifact của preprocessing. Cần human/external audit và một constraint set không tham gia discovery.
2. **Evaluation independence:** `actionable_union_eval` khác graph guidance nhưng vẫn được xây trong cùng project/dataset. Nó giảm self-evaluation bias, chưa tương đương benchmark độc lập do cộng đồng định nghĩa.
3. **Calibration uncertainty:** main router đã được xác nhận trên ba seeds × 4,096 rows, nhưng route policy mới được fit bằng một calibration seed 20261440 × 2,048 rows. Phải kiểm tra route agreement qua calibration seeds/sample sizes.
4. **Baseline status:** calibrated final repair, Harpoon-style guidance và graph variants là local matched proxies, không được gọi là official reproduction nếu code/settings paper không public hoặc không chạy nguyên bản.
5. **Runtime:** repaint tạo nhiều partial diffusion trajectories. Phải báo equivalent denoiser passes, wall time, trigger rows và tổng số repaint + repair actions; không chỉ báo final repairs.
6. **Magnitude và tính không đồng nhất:** frozen router đạt actionable gain 17.2–18.2% so với final repair trên hai dataset, nhưng Diabetes trước đó saturated và không có incremental gain. Không được claim universal improvement trên mọi bảng.
7. **Ý nghĩa thống kê:** router thắng semantic metrics 3/3 seeds nhưng mới có ba seeds. Cần mean±std/CI giữa seeds; paired row-bootstrap không thay thế route-calibration và seed uncertainty.
8. **Middle-state causality:** matched comparison hiện đã có và bác bỏ claim “middle luôn tốt hơn”. Claim mới là calibration chọn timing theo edge transition; không được viết thành universal middle-step superiority.
9. **Metric multiplicity:** actionable error là metric sát bài toán nhưng do project định nghĩa. Shape, Trend, AlphaPrecision/BetaRecall, detection, DCR và MLE phải cùng xuất hiện; không được chọn riêng metric có gain lớn.
10. **Action ambiguity:** conditional selector chỉ có lý do tồn tại khi một violation có nhiều feasible corrections. Ablation mới cho thấy nó không phải nguồn gain chính trên natural graph. Exact singleton categorical constraints nên dùng projection/final repair.
11. **Edge-wise heterogeneity chưa được chứng minh trong một dataset:** router hỗ trợ per-edge routes, nhưng calibration hiện chọn cùng route cho cả 7 eligible edges của mỗi dataset. Evidence hiện là cross-dataset timing adaptation; cần một graph/dataset có mixed routes để claim edge-level scheduling mạnh.
12. **Two-pass cost và terminology:** current router dùng một unconditional preview/base pass cộng conditional resampling cho triggered rows. Phải báo denoiser evaluations và không gọi nó là single-pass in-loop update.

Các bước có giá trị thông tin cao nhất sau screen:

1. Freeze router hiện tại; không tune thêm weight/timestep trên Default/Shoppers.
2. Tìm một dataset/graph có cả precursor, stable và late-emerging edges để kiểm tra mixed edge routes trong cùng run. Đây là experiment novelty quan trọng nhất.
3. Xây manually/publicly verified constraint subset không tham gia discovery; chạy matched Harpoon-style, CTDF-style projection, final repair và temporal router trên subset đó.
4. Kiểm tra robustness của route calibration qua 3 calibration seeds và sample sizes 512/1,024/2,048; báo route agreement và performance variance.
5. Báo compute theo denoiser evaluations, proposal rows, wall time và final repair events; two-pass router đắt hơn repair-only.
6. Single-pass rejoin là follow-up method, không phải điều kiện để viết current paper: chỉ giữ nếu vượt two-pass router ở matched compute. Các online projection/rollback cũ đã thất bại nên không lặp lại naive variant.

## 8. Kết luận hypothesis batch matched ngày 14/10/2026

Batch này không tìm thêm hyperparameter; nó tách ba giả thuyết nhân quả bằng cùng backbone, graph, ba seed `20261444–46`, 4,096 rows và evaluator độc lập.

### H-M1 — Conditional action selection tạo gain thật, không chỉ do proposal search

So sánh fixed-middle `global density` và `conditional density` với cùng checkpoint, tám branches và cùng random seed. Proposal pool là matched; khác biệt duy nhất là score target value theo context các cột không bị sửa. Giả thuyết được ủng hộ nếu conditional selector giảm held-out/actionable error nhất quán mà Shape/Trend/MLE không suy giảm đáng kể.

Jobs: sampling `1079208`, evaluation `1079209`.

**Kết luận:** không được ủng hộ trên Default và chỉ có gain nhỏ trên Shoppers. Conditional selector không phải contribution chính; global graph+density selector được giữ cho main router.

### H-M2 — Middle state có ích ngoài action selector

So sánh conditional selector ở middle state với chính selector đó ở final state. Final-state variant là conditional post-generation resampling, mạnh hơn minimal repair và do đó là đối chứng sát hơn. Giả thuyết middle injection được ủng hộ nếu nó thắng final conditional resampling trên semantic error và ít nhất giữ common fidelity/utility trong vùng sai số nhỏ.

Jobs: sampling `1079210`, evaluation `1079211`.

**Kết luận:** hypothesis “middle luôn hơn final” bị bác bỏ. Middle tốt hơn semantic error trên Default nhưng final tốt hơn trên Shoppers; common fidelity tạo frontier ngược chiều.

### H-M3 — Multi-state active graph chỉ hữu ích khi action được condition đúng

Ghép conditional selector vào proposal pool tại hai trạng thái `.40,.25`, thử `persistent` và `union` trên một seed mới. Đây là screen có điều kiện: chỉ xác nhận nhiều seed nếu vượt fixed-middle conditional selector; nếu không, kết luận fixed middle đủ mạnh và multi-state chỉ là ablation về trajectory.

Jobs: sampling `1079212`, evaluation `1079213`.

**Kết luận:** bị bác bỏ như universal mechanism. Default `union` có signal, Shoppers không; pool nhiều states làm proxy selector khó calibration hơn.

### H-M4 — Dynamic action-time selection thay vì cố định middle hoặc final

Kết quả sơ bộ của H-M2 cho thấy Default ưu tiên semantic validity ở middle, còn Shoppers có semantic error thấp hơn khi conditional resampling ở final; đồng thời common fidelity có xu hướng ngược lại. Vì vậy batch mới tạo một proposal pool gồm state `.40`, `.25` và final, rồi dùng cùng conditional selector để chọn cả correction lẫn source timestep theo từng row. Đây là test trực tiếp hơn cho dynamic graph theo trajectory: thời điểm action không còn là hyperparameter chung cho cả dataset.

Screen thử `persistent` và `union`, ba branches/state (chín proposals, gần fixed baseline tám proposals), một seed mới. Chỉ xác nhận nhiều seed nếu time-pool vượt cả fixed-middle và fixed-final trên Pareto frontier.

Jobs: sampling `1079432`, evaluation `1079433`.

**Kết luận:** time-pool chọn proposal từ cả ba states nhưng không generalize: Shoppers `union` giảm actionable error 13.3%, Default lại xấu đi. Dynamic routing phải quyết định state trước proposal selection thay vì bỏ mọi state vào một pool.

### H-M5 — Decouple activation state và proposal state

Middle preview chỉ quyết định active row-edge/target columns; correction luôn được sinh từ final state sạch hơn. Ba-seed matched test (`1079600–1079601`) cho actionable gain 17.2% trên Default và 6.8% trên Shoppers so với final repair, đồng thời common metrics gần như giữ hoặc tốt hơn. H-M5 được ủng hộ trên Default và là non-dominated point trên Shoppers.

### H-M6 — Calibrated temporal edge routing

Precursor-surplus calibration chọn `.40` hoặc `.25` chỉ khi middle-only violations nhiều hơn final-only; nếu không, defer edge tới final. Router (`1079827–1079828`) tự chọn `.40` cho Default, `final` cho Shoppers và đạt 17.2–18.2% actionable gain, 8.1–10.1% held-out gain, đều 3/3 seeds. Đây là hypothesis được giữ làm main.

Mốc quyết định đã đạt ở mức preliminary project: contribution được đóng khung thành **calibration-driven temporal active graph + decoupled late action generation**. Không claim context-conditional scoring là mới/chính, không claim middle luôn tốt, và chưa claim single-pass in-loop guidance hay SOTA tổng quát.

## 9. Nhánh mới: row-edge graph thực sự thay đổi trong từng reverse trajectory

### 9.1. Vấn đề của temporal router hiện tại

Router ở mục 4.13 có API per-edge nhưng calibration hiện chọn cùng một route cho mọi eligible edge trong từng dataset: `0.40` cho 7/7 edge của Default và `final` cho 7/7 edge của Shoppers. Vì vậy bằng chứng hiện tại mới cho thấy **timing khác nhau giữa dataset**, chưa chứng minh được mixed routing trong cùng graph. Hơn nữa, router quan sát một state rồi sinh correction bằng một conditional diffusion pass riêng; nó chưa thay đổi guidance liên tục trong trajectory gốc.

### 9.2. Cơ chế mới đã implement

Nhánh mới là single-pass `row-edge temporal active graph guidance`. Primal-dual là một biến thể đã thử, không còn là lựa chọn mặc định:

1. Một calibration trajectory độc lập học readability cutoff riêng cho từng edge bằng precision/recall của applicability và violation so với final state.
2. Sau cutoff, mỗi cặp `(row, edge)` duy trì một active state riêng. Row-wise EMA đo volatility của topology; hysteresis yêu cầu edge không ổn định lặp lại cùng thay đổi trước khi commit, trong khi edge ổn định có thể bật/tắt ngay.
3. Action gate mặc định chỉ mở khi một violation đã tồn tại qua nhiều preview liên tiếp. Biến thể mạnh hơn duy trì thêm dual variable cho mỗi `(row, edge)`:

   `lambda <- clip(decay * lambda + lr * (violation - calibrated_rate * active), 0, lambda_max)`.

   Lambda tăng nếu violation tồn tại, giảm khi row đã thỏa hoặc edge deactive. Do đó strength không còn là một scalar chung cho dataset, edge hay timestep.
4. Action layer sửa predicted-clean estimate `x_hat_0`; numeric action dùng typed feasible target và categorical action dùng feasible value. Update sau đó rejoin ngay reverse trajectory hiện tại. Ở biến thể primal-dual, action chỉ mở khi `lambda` vượt threshold và numeric strength tỷ lệ với `1 + lambda`.
5. Final repair vẫn chạy như residual certifier. Gain phải được đo trên output sau cùng so với cùng final repair, không chỉ trên output online trước certification.

Đây là cơ chế mới so với repaint main candidate: graph được đánh giá, bật/tắt và thay đổi action intensity ở từng reverse step; không có second diffusion pass. LLM không nằm trong loop này. LLM chỉ có thể cung cấp candidate graph trước sampling; executor, controller và action đều local/deterministic.

### 9.3. Nguồn ý tưởng từ các domain diffusion gần

- `Limited Guidance Interval` (NeurIPS 2024) cho thấy guidance ở high-noise có thể gây hại và phần hữu ích tập trung ở một interval giữa trajectory. Bản tabular dùng per-edge readability thay cho một interval chung.
- `DynaGuide` (NeurIPS 2025) dùng external dynamics signal ngay trong action denoising và cho thấy direct trajectory steering khác với best-of-N selection. Bản tabular thay dynamics model bằng typed active graph và row-edge state.
- `Constrained Diffusers` (NeurIPS 2025) dùng projected, primal-dual và augmented-Lagrangian updates trong reverse sampling. Bản tabular dùng dual multiplier riêng cho từng row-edge thay vì một constraint set luôn active.
- `CoDiG` (CoRL 2025) đưa barrier constraints vào denoising dưới dynamic environment. Bản tabular dùng calibrated empirical residual và dynamic applicability thay barrier cố định.
- `UnGuide` (UAI 2026) điều chỉnh guidance theo trajectory variance. Bản tabular dùng row-edge topology volatility, không dùng một prompt-level/global variance.
- `Progressive Guidance` (NeurIPS 2023) tăng dần condition-specific gradients khi state sạch hơn. Bản tabular đạt hiệu ứng này qua readability gate và dual accumulation thay một hand-designed global ramp.

Các nguồn trên là inspiration, không phải tabular baseline trực tiếp. Baseline trực tiếp vẫn phải gồm TabDiff, final repair, static/fixed-window online guidance, Harpoon-style guidance và CTDF/CDD-style projection.

### 9.4. Kết quả single-pass dynamic graph

**H-T1 — Per-edge readability tạo mixed temporal graph.** BSCG edge phải có cutoff khác nhau trong cùng dataset và row-edge active set phải có cả additions lẫn removals sau cutoff. Nếu mọi cutoff/route vẫn đồng nhất, claim per-edge temporal graph chưa được support.

Kết quả probe support mixed timing trên cả hai dataset. Default có 7 eligible actuator edges với bảy cutoff khác nhau `0.082, 0.102, 0.143, 0.163, 0.245, 0.449, 0.592`; Shoppers có các cutoff `0.102, 0.163, 0.224, 0.224, 0.306, 0.327, 0.429`. Với cùng Default trajectory, mean violation precision/recall tăng từ `0.300/0.662` ở interval `(0.25,0.50]` lên `0.854/0.914` ở `(0.10,0.25]` và `0.996/0.994` ở `[0,0.10]`. Online traces ghi nhận `106` additions và `122` removals trên Default, `76` additions và `73` removals trên Shoppers. Do đó graph không chỉ có onset khác nhau theo edge mà topology của từng row còn bật và tắt trong cùng trajectory. H-T1 được ủng hộ.

**H-T2 — Dynamic topology có giá trị ngoài limited interval.** Với cùng graph, seed, action family và final certifier, `row_ema_hysteresis` phải tốt hơn `static_window + frozen_state` trên actionable/held-out error mà không làm Shape, Trend hoặc MLE giảm rõ.

Trên Default, dynamic+repair giảm independent actionable error từ `0.007035` của final repair xuống `0.005898` (**16.2%**) và held-out error từ `0.009833` xuống `0.009133` (**7.1%**). So với static-window+repair, hai gain tương ứng là `11.5%` và `7.5%`. MLE AUC tăng `0.7255 -> 0.7314`; Shape error tăng nhẹ `1.9172 -> 1.9264`, Trend error tăng `1.8321 -> 1.8450`. Residual final repair bằng không: online trajectory đã xử lý toàn bộ row mà final certifier còn muốn sửa.

Trên Shoppers, dynamic+repair kém final repair về actionable error (`0.004352` so với `0.004027`) và held-out error (`0.009746` so với `0.009466`), dù Trend tốt nhẹ và MLE AUC tăng `0.9145 -> 0.9180`. Vì vậy H-T2 chỉ được ủng hộ trên Default, không phải universal. Bằng chứng này dẫn tới controller phân cấp: calibration phải được quyền **defer** edge ổn định về final certifier thay vì ép online guidance.

**H-T3 — Primal-dual intensity tốt hơn binary dynamic gate.** Thêm dual state phải giảm residual final-repair events hoặc independent actionable error so với dynamic hysteresis có fixed strength. Nếu chỉ tăng action count/drift, dual controller bị bác bỏ.

H-T3 bị bác bỏ. Trên Default, primal-dual và binary dynamic gần như bằng nhau (`0.005904` so với `0.005898` actionable error). Trên Shoppers, primal-dual tăng số action `102 -> 118`, làm Trend error xấu từ `2.3669` lên `2.5373` và held-out error tăng `0.009746 -> 0.009862`. Dual accumulation không được giữ trong main method.

**H-T4 — Single-pass có trade-off compute tốt hơn two-pass router.** Single-pass dynamic phải báo denoiser calls gần TabDiff 1x và wall time thấp hơn repaint đáng kể. Nếu accuracy thấp hơn nhiều, two-pass router vẫn là main result và single-pass chỉ là efficiency ablation.

Single-pass dynamic dùng đúng một reverse trajectory và mất `6.76 s` trên Default, `4.28 s` trên Shoppers cho 4,096 rows trong implementation hiện tại. Two-pass conditional repaint router ở mục 4.13 mất trung bình `158.2 s` và `49.7 s`. Đây là giảm lớn về wall time, nhưng so sánh này chưa loại hoàn toàn overhead khác nhau giữa hai implementation; paper nên report cả denoiser evaluations và GPU time.

| Dataset | Method | Shape error ↓ | Trend error ↓ | MLE AUC ↑ | Actionable error ↓ | Held-out error ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Default | TabDiff | 1.9131 | 1.8321 | 0.7196 | 0.021880 | 0.019048 |
| Default | Final repair | 1.9172 | 1.8321 | 0.7255 | 0.007035 | 0.009833 |
| Default | Static window + repair | 1.9212 | 1.8330 | 0.7262 | 0.006665 | 0.009871 |
| Default | **Dynamic row-edge + repair** | 1.9264 | 1.8450 | **0.7314** | **0.005898** | **0.009133** |
| Default | Primal-dual + repair | 1.9264 | 1.8450 | 0.7241 | 0.005904 | 0.009137 |
| Shoppers | TabDiff | **1.3769** | **2.2975** | 0.9153 | 0.011201 | 0.013829 |
| Shoppers | Final repair | 1.3837 | 2.3757 | 0.9145 | 0.004027 | 0.009466 |
| Shoppers | Static window + repair | 1.3821 | 2.3809 | 0.9142 | **0.003940** | **0.009442** |
| Shoppers | Dynamic row-edge + repair | 1.3839 | 2.3669 | **0.9180** | 0.004352 | 0.009746 |
| Shoppers | Primal-dual + repair | 1.3879 | 2.5373 | 0.9159 | 0.004352 | 0.009862 |

Đây là một seed mới (`20261447`), 4,096 rows, cùng checkpoint, sample seed và independent evaluation graph. Chưa được dùng để claim statistical significance.

### 9.5. Hypothesis tiếp theo đang chạy

**H-T5 — Calibration-gated temporal authority.** Chỉ edge có số violation ở readable middle state lớn hơn final state mới nhận online authority; những edge khác được defer tới final certifier. Tiêu chí dùng calibration trajectory, không xem test output. Dự đoán: giữ phần lớn gain Default nhưng tự suy biến về final repair trên Shoppers.

Per-edge gate quá bảo thủ: nó chỉ giữ 2/7 cạnh Default và actionable gain so với final repair còn `2.6%`; trên Shoppers nó defer đúng toàn bộ. Biến thể graph-level precursor được giữ: tổng positive precursor bằng `2` trên Default nên cấp quyền cho graph, bằng `0` trên Shoppers nên abstain. Đây là outer safety controller; sau khi được cấp quyền, topology vẫn thay đổi theo từng row-edge và timestep.

**H-T6 — Gain đến từ đúng row-state, không chỉ từ action budget.** So sánh dynamic topology với `shuffled topology`, trong đó support của mỗi edge được giữ nhưng active mask bị dịch sang row khác. Nếu shuffled ngang dynamic, claim row-conditioned graph không đứng vững.

H-T6 được ủng hộ. Shuffled topology có `298` flips và `60` online actions nhưng output sau repair đúng bằng final repair (`0.007035` actionable, `0.009833` full held-out), đồng thời numeric drift cao hơn. Đúng row-state với progressive gate đạt `0.005652` và `0.009027`.

**H-T7 — Event-triggered và progressive gate.** Event-triggered chỉ action khi edge vừa chuyển từ satisfied sang violated; progressive gate yêu cầu violation lặp nhiều lần ở middle rồi giảm dần bằng chứng cần thiết khi `x_hat_0` sạch hơn. Đây là hai cách biến action policy thành hàm của trajectory, không chỉ là lịch theo timestep.

Progressive gate được giữ. Trên Default seed `20261447`, nó giảm actionable error `19.7%` và full held-out error `8.2%` so với final repair; final certifier không còn residual action. Event-triggered chỉ dùng `30` online actions nhưng cần thêm `46` final actions và gain còn `4.7%` actionable, nên chỉ phù hợp làm efficiency ablation.

Xác nhận policy đã đóng băng trên hai seed mới cho Default cho actionable gain `22.4%` và `16.3%`; full held-out gain `8.4%` và `7.3%`. Gộp ba seed, actionable error giảm `0.006323 -> 0.005097` (**19.38%, 3/3 wins**), full held-out error giảm `0.011192 -> 0.010293` (**8.03%, 3/3 wins**). Shape và Trend trung bình tốt nhẹ; MLE AUC giảm trung bình `0.00172`, là trade-off cần report. Trên Shoppers, controller abstain và khớp final repair ở cả ba seed.

Kết quả tổng hợp tái lập được nằm tại `outputs/progressive_graph_controller_aggregate_20261015/RESULTS.md`.

Batch đã submit:

| Jobs | Nội dung | Trạng thái lúc submit |
| --- | --- | --- |
| `1080086[0-1]` | Dense BSCG trajectory probe, Default/Shoppers, calibration seed `20261440` | complete |
| `1080093[0-5]` | Static window, row-wise dynamic hysteresis, row-edge primal-dual; hai dataset, seed `20261447` | complete |
| `1080094[0-5]` | Shape, Trend, MLE và independent graph evaluation cho sáu runs | complete |
| `1080956[0-1]`, `1080957[0-1]` | Hierarchical temporal authority, Default/Shoppers | running/dependency |
| `1080959[0-1]` | Event-triggered và progressive gate trên Default | complete |
| `1080959[2]`, `1080960` | Shuffled control đầu tiên không matched-budget | canceled/replaced |
| `1080966[0-1]`, `1080967[0-1]` | Progressive graph-level precursor controller, Default/Shoppers | complete |
| `1080969[0-1]` | Common-metric evaluation cho event/progressive gates | complete |
| `1080973[2]`, `1080974[2]` | Matched-budget shuffled-topology control | complete |
| `1080977[0-3]`, `1080978[0-3]` | Hai seed xác nhận mới, Default/Shoppers | complete |
| `1080985`, `1080986`, `1080987` | Frozen pipeline trên Diabetes: probe → controller → evaluation | queued/dependency |

Code mới:

- `scripts/typed_graph_trajectory_probe.py`: held-out violation được tính trực tiếp từ held-out split, không còn yêu cầu graph mang sẵn field `heldout`.
- `scripts/rowwise_reverse_action_guidance.py`: thêm `row_ema_hysteresis`, `state_gate=dual`, `numeric_guidance_mode=primal_dual` và actuation-rate filter.
- `slurm/607_bscg_dynamic_trajectory_probe.sbatch`
- `slurm/608_rowedge_primal_dual_screen.sbatch`
- `slurm/609_rowedge_primal_dual_eval.sbatch`
- `slurm/610_hierarchical_temporal_controller.sbatch`
- `slurm/611_hierarchical_temporal_controller_eval.sbatch`
- `slurm/612_temporal_gate_controls.sbatch`
- `slurm/613_temporal_gate_controls_eval.sbatch`
- `slurm/614_progressive_graph_controller.sbatch`
- `slurm/615_progressive_graph_controller_eval.sbatch`
- `slurm/616_progressive_graph_confirmation.sbatch`
- `slurm/617_progressive_graph_confirmation_eval.sbatch`
- `slurm/618_diabetes_dynamic_probe.sbatch`
- `slurm/619_diabetes_progressive_controller.sbatch`
- `slurm/620_diabetes_progressive_eval.sbatch`

Hiện kết luận đúng nhất là: **mixed per-edge readability và row-level topology change đã được chứng minh; progressive online actuation cho gain lặp lại trên Default, nhưng lợi ích phụ thuộc dataset và phải được calibration-gated.** Không claim dynamic luôn hơn final repair. Graph-level precursor biến negative result trên Shoppers thành abstention có nguyên tắc, nhưng vẫn cần dataset thứ ba để xác nhận selector tổng quát.
