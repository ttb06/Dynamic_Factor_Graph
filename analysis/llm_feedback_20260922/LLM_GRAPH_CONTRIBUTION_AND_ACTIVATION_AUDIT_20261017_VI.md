# Kiểm tra contribution và cơ chế active của LLM graph

**Ngày kiểm tra:** 2026-10-17  
**Dataset:** Online News Popularity (`news`)  
**Backbone:** TabDiff checkpoint epoch 8000, 4096 mẫu, cùng seed `20261604`

## 1. Kết luận ngắn

Thông tin liên quan đến LLM **đã đi vào sampler và làm thay đổi trajectory thật**, nhưng bằng chứng hiện tại chưa cho thấy nó đủ mạnh để trở thành nguồn gain chính:

- Statistical compiler tìm lại 9/14 constraint trong catalog độc lập mà không cần API.
- LLM đề xuất thêm 3 quan hệ từ ngữ nghĩa tên cột; 2/3 khớp constraint công khai, làm recall tăng từ 9/14 = 64.3% lên 11/14 = 78.6%. Quan hệ sai còn lại vi phạm trên khoảng 96.3% train/calibration rows và đã bị deterministic verifier loại bỏ.
- Hai soft relation được cả hai LLM reviewer giữ lại tạo 10 conditional factors, với validation \(R^2\) từ 0.489 đến 0.855. Chúng active ở 23/50 bước giữa và thật sự đổi 45.4–53.0% số row-factor opportunities.
- Mặc dù can thiệp rất nhiều, gain so với cùng route không có soft factor chỉ nhỏ và không đồng nhất: Trend tăng tối đa 0.230 điểm, Alpha tăng tối đa 0.206 điểm, nhưng Shape giảm và utility không cải thiện ổn định.

Do đó, kết luận hiện tại là: **LLM có giá trị ở khâu mở rộng coverage và semantic filtering; chưa có bằng chứng rằng global soft messages từ các cạnh đó đủ tốt để trực tiếp tạo gain lớn.** Nút thắt hiện tại là activation quá rộng, không phải cạnh không được sử dụng.

## 2. Contribution của LLM trong graph discovery

| Thành phần | Số constraint catalog tìm lại | Recall trên 14 constraint | Vai trò |
|---|---:|---:|---|
| Typed statistical compiler | 9 | 64.3% | Tìm order, simplex và functional relation từ schema + train split |
| LLM semantic proposal bổ sung | +2 | +14.3 điểm % | Tìm hai directional bounds chưa được statistical compiler đề xuất |
| Pipeline sau verification | 11 | 78.6% | Chỉ 11 cạnh có independent authority được phép actuate |

Hai constraint hợp lệ do LLM đề xuất:

1. `num_self_hrefs <= num_hrefs`;
2. `n_non_stop_unique_tokens <= n_non_stop_words`.

Một đề xuất sai là `n_non_stop_unique_tokens <= n_unique_tokens`. Nó nghe hợp lý nếu hai đại lượng dùng cùng mẫu số, nhưng dữ liệu vi phạm khoảng 96%; pipeline vì vậy không cho cạnh này đi vào actuator. Ví dụ này chứng minh LLM có thể mở rộng semantic coverage, nhưng không thể tự làm authority.

Chi phí run v2 là 3 API calls, 24,603 tokens: một call sinh typed proposal và hai call semantic review. Raw rows không được gửi ra API.

## 3. Soft graph có thật sự active không?

Hai polarity components được cả hai reviewer giữ lại là quan hệ **statistical do LLM phân loại**, không phải kiến thức mới do LLM tự sinh:

- negative polarity component: 5 cột;
- positive polarity component: 5 cột.

Chúng sinh ra 10 ExtraTrees conditional factors. Validation \(R^2\) nằm trong khoảng 0.489–0.855, nên signal dự đoán có thật và không phải random. Với cửa sổ `log-sigma quantile` từ 0.15 đến 0.65, factor active ở 23/50 reverse steps.

| Variant | Strength | Active steps | Row-factor opportunities | Ô thật sự đổi | Tỷ lệ đổi |
|---|---:|---:|---:|---:|---:|
| Route only | 0 | 23 | 0 | 0 | 0% |
| Consensus soft | 0.02 | 23 | 942,080 | 427,852 | 45.4% |
| Consensus soft | 0.05 | 23 | 942,080 | 470,904 | 50.0% |
| Consensus soft | 0.10 | 23 | 942,080 | 499,188 | 53.0% |

Vấn đề: các soft components không có row-level applicability predicate. Vì vậy gate cũ coi toàn bộ 4096 rows là candidate tại mọi timestep trong cửa sổ. Rank transport chỉ khiến một phần ô đổi, nhưng bản thân edge activation vẫn quá dày.

## 4. Gain có tương xứng với mức can thiệp không?

Bảng dưới là chênh lệch so với `route_only` trong cùng run, cùng seed. Đơn vị Shape/Trend/Alpha/Beta là điểm phần trăm; detection, \(R^2\), NRMSE giữ thang gốc.

| Soft factor | ΔShape ↑ | ΔTrend ↑ | ΔAlpha ↑ | ΔBeta ↑ | ΔDetection ↑ | ΔR² ↑ | ΔNRMSE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| strength 0.02 | -0.024 | +0.120 | +0.125 | +0.101 | +0.0017 | -0.0090 | +0.0044 |
| strength 0.05 | -0.040 | +0.230 | +0.169 | -0.052 | +0.0022 | -0.0057 | +0.0028 |
| strength 0.10 | -0.090 | +0.111 | +0.206 | +0.039 | -0.0064 | +0.0082 | -0.0040 |

Diễn giải:

- Signal đủ mạnh để đổi sample, nhưng chưa đủ chính xác để gain tăng theo strength.
- Strength lớn hơn làm số update và tổng movement tăng mạnh, trong khi Trend không tăng đơn điệu.
- Soft factor có thể giúp một số joint metric nhưng làm giảm marginal fidelity; đây là trade-off thật, không nên che bằng constraint satisfaction metric.
- Hai directional bounds do LLM đề xuất gần như đã được TabDiff thỏa mãn tự nhiên (violation khoảng 0.069% và 0.0015%), nên direct projection của chúng khó tạo gain fidelity lớn. Giá trị chính của chúng là coverage/certification.

## 5. Cơ chế active nên dùng

Thay vì `active = time_window AND applicability`, cơ chế mới dùng:

\[
A_{e,i,t} = R_{e,t}\;\land\;P_e(\hat{x}_{0,i,t})\;\land\;Q_e
\;\land\;N_{e,i,t}\;\land\;U_{e,i,t}\;\land\;S_{e,i,t}.
\]

Trong đó:

- \(R_{e,t}\): feature/edge đã đủ readable ở timestep hiện tại;
- \(P_e\): applicability predicate của edge đúng với row;
- \(Q_e\): factor có cross-validated reliability đủ cao;
- \(N_{e,i,t}\): row thật sự cần message, đo bằng normalized conditional residual;
- \(U_{e,i,t}\): ensemble uncertainty đủ thấp;
- \(S_{e,i,t}\): semantic target ổn định giữa hai preview liên tiếp.

Implementation mới đã thêm hai gate trực tiếp:

1. `factor_residual_gate`: chỉ active nếu
   \(|x_j-f_j(x_{N(j)})| / s_{residual}\) vượt threshold;
2. `factor_uncertainty_gate`: chỉ active nếu độ lệch chuẩn dự đoán giữa các trees, chuẩn hóa bởi \(s_{residual}\), thấp hơn threshold.

Time readability và validation \(R^2\) vẫn được giữ. Consecutive-state stability hiện đã có cho documented projection; bước kế tiếp là áp dụng tương tự cho soft-factor target nếu residual + uncertainty gate có tín hiệu tốt.

## 6. Thực nghiệm đang chạy

| Job | Nội dung | Câu hỏi quyết định |
|---|---|---|
| `1091631[0-5]` | Residual gate {0.25, 0.5, 1.0}; uncertainty gate {off, 1.0, 0.75}; consensus và Luna graph | Có giảm update thừa mà giữ/tăng Trend, utility và Shape không? |
| `1091632[0-2]` | LLM-only và order+LLM actuation, strength 0.25/1.0 | Hai cạnh LLM-verified có thật sự active và thêm gì ngoài statistical graph? |
| `1091598[0-5]` | Soft graph chỉ làm action scorer/veto, không gửi message | Dùng LLM graph để chọn action có an toàn hơn dùng nó làm direct prior không? |

## 7. Tiêu chí giữ hay bỏ LLM branch

Giữ LLM như một contribution chính chỉ khi ít nhất một trong hai kết quả xuất hiện:

1. LLM tăng coverage trên independent catalog/cross-dataset rõ rệt mà statistical compiler không đạt được; hoặc
2. LLM-filtered graph + state-selective gate cải thiện joint fidelity/utility có hệ thống so với cùng graph không LLM và unfiltered graph.

Nếu không đạt, thiết kế hợp lý hơn là để LLM làm optional semantic proposer/source-retrieval module; contribution chính vẫn là dynamic graph activation theo trajectory với graph có independent authority.

## 8. Positive messages trên các cạnh LLM đã thỏa mãn

Một inequality như `num_self_hrefs <= num_hrefs` chỉ xác định feasible region; nó không chỉ ra một row hợp lệ nên nằm ở đâu bên trong miền đó. Vì vậy không nên tự tạo một target tùy ý cho satisfied rows. Implementation mới phân tách hai nguồn thông tin:

- LLM cung cấp semantic topology: hai cột nào có quan hệ và chiều quan hệ;
- train split cung cấp conditional potential trên hai endpoint của cạnh, có cross-validation và uncertainty calibration.

Với mỗi cạnh LLM đã verification, một ExtraTrees factor được học theo cả hai chiều. Ở mỗi middle timestep, factor dự đoán conditional ordering của target từ node lân cận. Copula rank transport sau đó tái phân bổ đúng tập giá trị target hiện có theo prediction đó. Vì chỉ hoán vị các giá trị hiện có, empirical marginal của từng cột được giữ nguyên; thông tin được đưa vào chủ yếu qua joint structure.

Positive message không yêu cầu row đang violation. Một satisfied row vẫn được cập nhật nếu conditional residual đủ lớn và model uncertainty đủ thấp. Projection layer vẫn được tách riêng và chỉ xử lý violation.

Trace mới ghi riêng:

- `factor_satisfied_candidate_row_factors`;
- `factor_violating_candidate_row_factors`;
- `factor_satisfied_active_row_factors`;
- `factor_violating_active_row_factors`;
- `factor_satisfied_cell_events`;
- `factor_violating_cell_events`.

Job `1091637[0-5]` so sánh route-only, positive factor strength 0.02/0.05, state-selective positive factor, strength 0.10 và hybrid positive-message + violation-projection. Factor chỉ được fit trên family `llm_directional_bound`, trong khi documented order graph được giữ cố định để tránh gán gain của statistical edges cho LLM.
