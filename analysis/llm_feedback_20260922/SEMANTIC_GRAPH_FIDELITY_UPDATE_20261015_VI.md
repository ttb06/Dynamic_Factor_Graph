# Cập nhật: mở rộng semantic graph và cải thiện fidelity/utility

Ngày cập nhật: 2026-10-15.

## 1. Cạnh đang thỏa mãn hiện được dùng như thế nào?

Trong controller đã chạy trước đây, một cạnh đang `active` và `satisfied` có ba tác động gián tiếp:

1. nó vẫn thuộc active topology của bản ghi tại timestep đó;
2. violation streak của cạnh được đưa về 0;
3. dual state giảm dần.

Tuy nhiên, cấu hình chính dùng `hardness_adaptive`, không dùng dual state để chọn action. Vì vậy một cạnh satisfied **không tạo correction dương và không bảo vệ quan hệ đang đúng khỏi action của cạnh khác**. Trên thực tế nó chủ yếu dẫn đến quyết định “không hành động”. Các sweep `graph_weight` cũ cũng chưa kiểm tra đúng giả thuyết này, vì score lúc đó chỉ nhìn các cạnh `score_only`, trong khi graph chính gần như chỉ có cạnh `actuate`.

Biến thể mới `preserve_satisfied_actuate` sửa đúng sự bất đối xứng đó. Ở mỗi timestep, chỉ những actuator edge đã readable mới có quyền làm guardrail. Nếu một candidate action biến một cạnh đang satisfied thành violated, score của action bị phạt; cạnh satisfied không được tự khởi tạo edit. Đây là positive state evidence ở action-selection layer, không phải thêm hard projection.

## 2. Constraint graph hiện tại có thật sự phức tạp không?

### Graph LLM chất lượng cao đã có trên Default

Nguồn: `outputs/default_hq_graph_1060998` và audit mới `outputs/default_hq_graph_complex_audit_20261015`.

| Thuộc tính | Kết quả |
| --- | ---: |
| Candidate do multi-role LLM đề xuất | 58 |
| Cạnh LLM giữ lại sau critic, judge và calibration | 28 |
| Hard / near-hard / soft | 0 / 3 / 25 |
| Cạnh complex theo cấu trúc | 23/28 (82.1%) |
| Cạnh có arity từ 3 cột | 23 |
| Cạnh có conjunction | 4 |
| Cạnh có exception group | 22 |
| Contextual column comparison | 0 |
| Cạnh ổn định trên held-out với shift ≤ 0.05 | 23/23 complex |
| Chi phí protocol cũ | 20 API calls, 177,176 tokens |

Kết luận: graph không nhỏ nếu chỉ nhìn arity, nhưng semantic diversity còn thấp. Phần lớn relation là các biến thể của ba pattern:

- statement balance bằng/không dương dẫn đến payment bằng 0;
- trạng thái `PAY_k=-2` tiếp tục sang tháng lân cận;
- delinquency hoặc vượt credit limit liên quan bill/payment dương.

Nhiều exception chỉ mô tả prepayment, refund hoặc posting-time effect; chúng làm cạnh dài hơn nhưng chưa tạo một family mới. Đặc biệt, 25/28 cạnh là soft và không có contextual column comparison. Vì vậy chưa nên đưa toàn bộ 28 cạnh vào actuator graph.

### Protocol mới đang chạy

Job `1082396` dùng UnoRouter và chỉ gửi schema, glossary và aggregate train statistics. Nó dùng:

- `gpt-5.6-luna` cho proposal, external/domain role, adversarial critic và semantic judge;
- budget tối đa 180,000 tokens;
- profile `connectivity_aware`, có nhìn topology của BSCG hiện tại;
- hai proposal rounds, counterexample partition, calibration partition và frozen held-out audit.

Mục tiêu không phải tăng số cạnh tùy ý, mà tìm process chain hoặc accounting/temporal bridge nối các endpoint family đang rời rạc. Job `1082397` sẽ audit arity, conjunction, exception, action ambiguity và held-out shift sau khi graph được đóng băng. Lần submit đầu `1082384` thất bại sau một request vì UnoRouter trả response shape khác parser OpenAI cũ; usage không báo token. Parser đã được sửa để nhận cả wrapper và Chat-Completions-compatible content, dependency hỏng đã bị hủy, rồi job được submit lại bằng Luna để giảm latency/cost.

## 3. Dataset nào có tiềm năng?

| Dataset | Tiềm năng semantic graph | Rủi ro chính | Ưu tiên |
| --- | --- | --- | --- |
| Default | Cao nhất hiện tại: có chuỗi thời gian bill–payment–repayment, nhiều endpoint và state thay đổi theo trajectory | Codebook `PAY_k`, adjustment, refund và posting delay làm nhiều quan hệ chỉ là probabilistic | 1 |
| Shoppers | Có relation dễ giải thích giữa page count–duration, BounceRates–ExitRates và session state | Nhiều relation hard đã gần saturated; controller chính từng abstain vì không có middle-before-final surplus | 2, dùng để kiểm tra generalization hơn là tối ưu graph |
| Diabetes | Có co-occurrence giữa medication, diagnosis và utilization | Nhiều quan hệ phản ánh practice pattern, không phải semantic invariant; dễ tạo constraint sai | 3 |
| Adult | Có vài codebook/missingness dependency | Rủi ro stereotype và sensitive relation cao; hard semantic edges ít | Không dùng làm dataset chính |
| News/Magic | Có public numeric constraints hoặc distributional structure | Ít metadata domain để LLM tạo dynamic semantic graph đáng tin | Baseline phụ |

Default vẫn là dataset hợp lý nhất để phát triển mechanism. Shoppers nên là dataset thứ hai sau khi semantic proposal được đóng băng, nhằm kiểm tra phương pháp có biết abstain hoặc chuyển sang relation khác hay không.

## 4. Vì sao fidelity/utility hiện chưa hơn TabDiff rõ rệt?

Kết quả confirmation của progressive controller trên Default, 3 unseen seeds:

| Metric | Final repair | Progressive controller | Thay đổi có lợi |
| --- | ---: | ---: | ---: |
| Shape error ↓ | 2.041838 | 2.038613 | 0.16% |
| Trend error ↓ | 1.971839 | 1.961033 | 0.55% |
| XGBoost AUC ↑ | 0.731765 | 0.730041 | -0.24% |
| Actionable held-out error ↓ | 0.006323 | 0.005097 | 19.38% |
| Full held-out error ↓ | 0.011192 | 0.010293 | 8.03% |

Semantic calibration gain ổn định, nhưng standard fidelity gần như hòa và utility giảm nhẹ. Nguyên nhân hợp lý nhất hiện tại là action layer chọn một edit chỉ bằng target-edge validity, global numeric kNN density và edit size. Nó chưa dùng:

- các cạnh đang đúng như guardrail;
- local conditional manifold của chính bản ghi;
- utility objective trực tiếp.

Không nên thêm target-label utility vào sampler vì dễ thành task-specific tuning và làm yếu claim generator-general. Thay vào đó, hai thay đổi mới chỉ dùng train predictor columns và được đánh giá bằng XGBoost AUC độc lập.

## 5. Hai hypothesis fidelity/utility đang được kiểm tra

### H-F1: satisfied-edge guardrail

Một action sửa cạnh A có thể phá cạnh B vốn đang đúng. Nếu action selector phạt collateral violation trên mọi readable satisfied actuator, fidelity và utility có thể tăng mà không giảm semantic calibration.

Điều kiện ủng hộ: so với progressive controller cũ, Shape/Trend hoặc AUC tăng; held-out semantic error không xấu hơn đáng kể; số action không tăng bất thường.

Một audit bổ sung trên 4,096 output rows của run chính cho thấy controller sửa ít nhất một violation ở 54 rows, nhưng không có row nào kết thúc với một cạnh trước đó satisfied bị biến thành violated. Graph triển khai có 7 edges, 12 nodes nhưng 5 connected components; bốn component chỉ chứa một edge. Do đó guardrail có thể vẫn ngăn transient collateral damage ở giữa trajectory, nhưng xác suất tạo gain lớn trên graph hiện tại không cao. Kết quả này cũng giải thích tại sao `graph_weight` cũ không có tác dụng và làm rõ nhu cầu tìm semantic bridge đã được kiểm chứng.

### H-F2: conditional-density action selection

Khi một violation có nhiều action hợp lệ, chọn endpoint phổ biến toàn cục chưa đủ. Candidate mới được chấm thêm theo phân phối của edited cells trong các train neighbours giống bản ghi ở các cột không bị sửa. Đây là train-only local plausibility, không dùng held-out và không thay đổi graph activation.

Nhánh repainting trước đây đã cho bằng chứng ban đầu trên Shoppers, 3 seeds:

- Trend error tốt hơn final repair ở 3/3 seeds;
- XGBoost AUC tăng trung bình 0.00705 ở 3/3 seeds;
- held-out constraint error tốt hơn nhẹ ở 2/3 seeds.

Trên Default, conditional-density cũ không ổn định về AUC. Vì vậy lần này nó được thử trực tiếp bên trong progressive trajectory controller, riêng lẻ và kết hợp satisfied-edge guardrail.

## 6. Jobs đã submit

| Job | Nội dung | Trạng thái khi submit |
| --- | --- | --- |
| 1082396 | LLM connectivity-aware semantic graph trên Default, UnoRouter, 180k token cap | Queued |
| 1082397 | Audit structural/semantic readiness của graph 1082396 | Dependency |
| 1082386[0-3] | Guardrail và conditional-density action selector, một matched seed | Queued |
| 1082387[0-3] | Standard fidelity/utility evaluation của 1082386 | Dependency |

Bốn variant của job 1082386:

1. satisfied guardrail, weight 1;
2. conditional density, weight 1;
3. kết hợp cả hai, mỗi weight 1;
4. kết hợp guardrail weight 1 và conditional density weight 2.

Tất cả dùng cùng TabDiff checkpoint, graph, seed, sample count, trajectory schedule, final repair và independent evaluation graph. Đây là screen một seed để chọn mechanism; chỉ variant thắng mới cần confirmation nhiều seed.

## 7. Tiêu chí quyết định

Một graph LLM mới chỉ đáng đưa vào sampler nếu relation:

1. có semantic rationale không phụ thuộc vào correlation của target hoặc sensitive fields;
2. executable và có directionality rõ;
3. vượt discovery, counterexample và calibration checks;
4. held-out shift nhỏ sau khi graph đã freeze;
5. có action authority rõ: `actuate`, `score_only` hoặc chỉ `diagnostic`.

Một mechanism mới chỉ đáng giữ nếu cải thiện ít nhất một standard metric quan trọng (Shape, Trend hoặc AUC) và không đổi gain semantic thành distributional damage. Constraint satisfaction tự định nghĩa không được dùng làm tiêu chí thắng duy nhất.
