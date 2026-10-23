# Cải thiện graph guidance và thí nghiệm theo khoảng nhiễu

Ngày cập nhật: 2026-10-16

## 1. Kết luận ngắn

Guidance chính hiện tại **không phải classifier-free guidance (CFG)**. CFG chạy denoiser có điều kiện và không điều kiện rồi ngoại suy hiệu của condition. Phương pháp hiện tại chạy TabDiff một lần, decode dự đoán clean state `x_hat_0`, dùng dynamic graph biến đổi một số giá trị hoặc logit, rồi dùng clean state đã hiệu chỉnh trong bước Euler tiếp theo.

Hai cơ chế giống nhau ở mức khái quát: cả hai thay đổi vector field của reverse diffusion và có thể được bật/tắt theo noise level. Vì vậy, kết luận của Kynkäänniemi et al., *Applying Guidance in a Limited Interval Improves Sample and Distribution Quality in Diffusion Models* (NeurIPS 2024) là một giả thuyết rất phù hợp để chuyển sang tabular, nhưng khoảng sigma tuyệt đối của image diffusion không được sao chép trực tiếp.

## 2. Vì sao không dùng nguyên khoảng sigma của paper

Paper báo các khoảng tối ưu khác nhau ngay giữa các image backbone:

- EDM2-XXL, FID: `(0.19, 1.61]`, 6/32 bước, weight 2.0.
- EDM2-XXL, FD-DINOv2: `(0.60, 5.00]`, weight 2.9.
- DiT-XL/2, FID: `(0.34, 1.02]`, 75/250 bước, weight 2.5.
- SDXL trong phân tích định tính dùng `(0.28, 5.42]`.

TabDiff chuẩn hóa dữ liệu và học một noise schedule riêng cho từng numerical feature. Vì thế `sigma=1` trong image space không có cùng ý nghĩa với `sigma=1` của từng cột tabular. Thí nghiệm mới dùng **quantile của log-sigma học được**:

- `0`: clean end;
- `1`: noisy end;
- gate guidance bằng một khoảng `[q_lo, q_hi]` tại đúng step boundary.

Đây là phép chuyển đổi tương đối theo chính schedule của backbone, tránh chọn interval tùy ý bằng chỉ số timestep.

## 3. Graph guidance được mở rộng như thế nào

### 3.1 Hai loại graph signal

1. **Violation projection**: cạnh đang vi phạm tạo typed correction lên endpoint. Numerical inequality được chiếu cân bằng theo scale hai cột; categorical relation tác động lên clean logits.
2. **Dense satisfied-edge message**: cạnh đã thỏa vẫn truyền thông tin. Một conditional factor được fit chỉ trên train giữa các node kề nhau. Reliability là out-of-fold validation R2. Message chỉ active khi applicability của ít nhất một incident edge đúng ở clean preview hiện tại.

Như vậy satisfied edge không còn chỉ là veto bảo vệ một action. Nó có thể cải thiện joint structure trong khi chưa có violation rời rạc.

### 3.2 Hai transport cho satisfied edges

- `residual`: kéo endpoint về conditional prediction, giới hạn displacement bằng cross-validated residual scale, bỏ batch-mean shift và khớp lại mean/std.
- `copula`: sắp xếp lại assignment theo conditional score nhưng giữ nguyên chính xác empirical marginal của batch tại step đó. Đây là cơ chế ưu tiên vì nó tách cải thiện dependence khỏi marginal distortion.

Graph propagation chạy nhiều pass trong cùng step, nên thông tin có thể đi qua một đường nhiều cạnh. Reliability làm hệ số giảm message của factor yếu.

### 3.3 Cửa sổ theo noise

Hai lớp được tách khi ablate:

- dense message chỉ chạy trong noise interval;
- typed certifier/projection có thể chạy riêng hoặc chỉ ở clean end.

Thí nghiệm interval đầu tiên tắt online projection để không nhầm hiệu ứng của “khi nào truyền message” với hiệu ứng repair. Sau sampling, cùng một deterministic certifier được áp cho cả TabDiff và phương pháp để tạo bảng so sánh với final repair.

## 4. Sinh constraint/graph: kiến trúc đề xuất

Không nên yêu cầu LLM làm mọi loại constraint.

### Tầng A — deterministic schema compiler

- valid category vocabulary;
- integer columns có trong metadata;
- nonnegative/range nếu codebook định nghĩa;
- missing-value representation;
- dữ liệu count/probability nếu có metadata xác nhận.

TabDiff hiện đã dùng `int_columns` khi inverse transform, nên integer typing phần lớn đã có trong preprocessing; cần đánh giá trước khi claim thêm gain.

### Tầng B — externally documented inter-column semantics

Ví dụ: subset count, `min <= mean <= max`, probability/simplex, process implication. Đây là nhóm có thể nhận hard/near-hard authority sau typecheck và counterexample validation.

### Tầng C — conditional process hyperedges

LLM dùng codebook, aggregate evidence và value-aware retrieval để đề xuất applicability, requirements, directionality và plausible counterexample. Candidate được compile về typed DSL, kiểm tra trên discovery/counterexample/calibration splits và audit heldout.

### Tầng D — soft external/causal hypotheses

Không dùng trực tiếp để overwrite. Chỉ làm conditional potential hoặc action-scoring guardrail. Nếu không có authoritative mechanism, causal plausibility và train correlation không đủ để gọi là validity constraint.

Profile mới `semantic_programmatic` dùng few-shot để dạy sự phân biệt bốn vai trò trên. Few-shot có cả positive và negative example; không chứa heldout rows hoặc public expert constraint list.

## 5. Hypothesis đang chạy

### H-A: satisfied edges có cải thiện fidelity không?

Giữ backbone, seed và sample count cố định. So sánh:

- TabDiff;
- final typed projection;
- projection-only trong trajectory;
- dense copula message;
- dense residual message;
- dense message + online projection;
- dense message + cùng final certifier.

Grid strength/pass được giữ nhỏ để xác định mechanism, không phải hyperparameter sweep không kiểm soát.

### H-B: guidance interval có lợi hơn full guidance không?

Cùng graph, seed, strength, 3 passes và không online projection:

- full `[0,1]`, binary;
- middle `[0.20,0.60]`, binary;
- middle `[0.20,0.60]`, smooth middle peak;
- narrow middle `[0.30,0.55]`, binary;
- clean-end `[0,0.25]`, binary;
- noisy-end `[0.65,1]`, binary.

Paper NeurIPS 2024 thấy binary interval tốt hơn smooth weighting trong image. Nếu tabular tái hiện ordering `middle > full, clean-only, noisy-only`, đó là evidence mạnh cho contribution trajectory-specific. Nếu không, không nên dùng paper này làm motivation định lượng.

### H-C: few-shot semantic program discovery có tạo graph tốt hơn không?

Đánh giá không chỉ số cạnh. Cần báo:

- số candidate theo authority/type;
- semantic acceptance qua independent judge;
- heldout violation stability;
- discovery recall trên News public constraints (expert list không được đưa vào prompt);
- tỷ lệ trivial/redundant/unsupported bị loại;
- endpoint diversity và component connectivity;
- số cạnh actuator so với score-only.

## 6. Metrics quyết định

Main fidelity/utility:

- Shape, Trend, α-precision, β-recall;
- detection score;
- downstream R2 và NRMSE;
- marginal KS theo cột và pairwise/dependency error nếu evaluator hỗ trợ;
- duplicate rate và rare-bin/category coverage.

Graph-specific nhưng không được đứng một mình:

- public/external constraint violation rate;
- heldout graph error;
- edit rate, standardized movement, unique rows changed;
- factor events theo noise interval;
- conditional error trên các edge/subgroup được pre-registered.

Kết quả chỉ đáng đưa vào main nếu cải thiện joint/utility hoặc distributional coverage ở cùng hoặc tốt hơn validity. Chỉ đạt zero violation trên constraint do ta tự đề xuất không đủ.

## 7. Tình trạng jobs

- `1085336`: 6 cấu hình dense numerical factor message/projection trên News, 4096 samples.
- `1085337`: 6 cấu hình noise interval theo log-sigma quantile, 4096 samples.
- `1085296`: few-shot `semantic_programmatic` graph discovery trên News, budget tối đa 180k tokens qua UnoRouter.
- `1085436`: 6 cấu hình limited-noise interval cho trực tiếp typed projection/action guidance; đây là test sát nhất với câu hỏi chuyển phương pháp NeurIPS 2024 sang mechanism hiện tại.
- `1085444`: 3 cấu hình edge-local noise clock; mỗi cạnh dùng noise quantile của chính các endpoint thay vì bật toàn graph theo một clock chung.

Lần submit `1085281/1085282` phát hiện adapter final-repair cũ chỉ hỗ trợ categorical equality. Các sample trajectory đã chạy nhưng final aggregation dừng. Adapter đã được thay bằng repeated numerical typed projection; jobs trên là lần chạy sửa lỗi.

### Kết quả kiểm tra sớm: projection-only

Một cấu hình projection-only hoàn tất trước khi grid đầy đủ chạy lại:

| Method | Shape ↑ | Trend ↑ | α-precision ↑ | β-recall ↑ | Detection ↑ | R2 ↑ | NRMSE ↓ | Any violation ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TabDiff | 96.6574 | 91.1556 | 97.4611 | 45.1253 | 0.8925 | -0.0055 | 1.0027 | 25.4883% |
| Final repair | 94.8482 | 91.7890 | 97.4676 | 45.1188 | 0.9000 | -0.0035 | 1.0017 | 0% |
| Trajectory projection | 96.6375 | **91.8450** | 97.4627 | 45.0423 | 0.8867 | **0.0156** | **0.9922** | 0.0732% |
| Trajectory + final certifier | 94.8486 | **91.8450** | 97.4627 | 45.0423 | 0.8870 | **0.0163** | **0.9918** | 0% |

Đây là single seed và chưa phải main result. Tuy nhiên, nó cho tín hiệu đúng hướng: online projection gần như xoá violation, tăng Trend khoảng `+0.69` điểm và đưa R2 từ âm sang dương trong khi Shape chỉ giảm `0.02` điểm. Final repair đạt zero violation nhưng làm Shape giảm khoảng `1.81` điểm. Việc thêm final certifier sau trajectory lại gây đúng mức giảm Shape của final repair; do đó nên report trajectory projection riêng thay vì mặc định ép zero violation ở cuối.

## 8. Tiêu chí quyết định sau jobs

- Nếu middle interval thắng full/noisy/clean trên Trend hoặc dependency/β-recall mà Shape và utility không giảm đáng kể: giữ limited-interval dynamic graph message làm mechanism chính.
- Nếu copula message thắng residual và projection-only: claim hợp lý là satisfied-edge dependency propagation với marginal-preserving transport, không phải “sửa nhiều constraint hơn”.
- Nếu final repair bằng hoặc tốt hơn trajectory trên mọi metric: mechanism hiện chưa đủ; dùng final repair làm baseline và không claim trajectory benefit.
- Nếu few-shot graph tăng số cạnh nhưng heldout stability/semantic acceptance giảm: không dùng “nhiều cạnh” làm mục tiêu; giữ graph authority phân tầng.
- Nếu LLM graph không tốt hơn documented + statistical graph: hạ LLM thành optional graph proposal module, không để toàn bài phụ thuộc LLM.

## 9. Nguồn phương pháp gần nhất

- Kynkäänniemi et al., NeurIPS 2024: limited CFG interval; chuyển ý tưởng timing, không sao chép CFG operator.
- Harpoon: differentiable constraint loss trong sampling; baseline cho rule-gradient guidance.
- CDD, NeurIPS 2025: feasibility layer cho constrained discrete diffusion; gợi ý typed categorical projection.
- SCTab-Diff: static LLM prior condition denoising; đối chứng cho knowledge condition cố định.
- LCATS: discovery/verification và post-generation repair; đối chứng cho graph discovery và final correction.
