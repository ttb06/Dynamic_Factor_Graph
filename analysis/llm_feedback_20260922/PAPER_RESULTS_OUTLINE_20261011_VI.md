# Báo cáo kết quả và khung bài báo: Dynamic Constraint Graph Guidance

Ngày chốt vòng thực nghiệm: 2026-10-11  
Backbone: TabDiff, 50 reverse steps  
Quy mô xác nhận: 4.096 synthetic rows × 3 paired seeds cho mỗi dataset

## 1. Kết luận điều hành

Dự án hiện đã có thể bắt đầu viết bài. Kết quả đủ ổn định để đóng khung contribution chính, nhưng chưa nên gọi là SOTA về chất lượng synthetic tabular nói chung.

Contribution có bằng chứng tốt nhất là:

> Một constraint graph đã được kiểm chứng không nên được áp dụng nguyên trạng trong toàn bộ reverse trajectory. Với mỗi row và timestep, method xây active subgraph từ predicted-clean state, chỉ can thiệp vào excess constraint error đã được hiệu chỉnh theo dữ liệu thật, rồi chọn typed action có chi phí thấp. Cơ chế này khớp constraint profile trên held-out tốt hơn fixed guidance, frozen topology và final repair trong các setting mà quan hệ đã đủ readable trước bước cuối.

Ba kết luận thực nghiệm chính:

1. **Default, graph 8 cạnh:** dynamic trajectory đạt held-out profile error `0.02250 ± 0.00079`, tốt hơn final repair `18.3%`, Harpoon-style proxy `19.4%`, frozen topology `6.4%`, thắng frozen `3/3` seed.
2. **Shoppers, graph 27 cạnh:** hệ đầy đủ dynamic trajectory + residual repair đạt `0.01189 ± 0.00281`, tốt hơn final repair `5.8%` và frozen counterpart `8.8%`, thắng cả hai trên `3/3` seed. Riêng trajectory-only tốt hơn frozen trajectory khoảng `33.5%`.
3. **Diabetes:** graph 17 cạnh xác nhận dynamic trajectory tốt hơn frozen trajectory `29.3%`, nhưng sau residual repair chỉ hơn final repair `1.1%`. Graph mở rộng 22 cạnh không làm khoảng cách với final repair lớn hơn: full method chỉ hơn `0.28%`. Vì vậy Diabetes là boundary result: dynamic topology có ích trước hậu xử lý, nhưng final repair đã gần ceiling.

Kết quả tốt đã qua confirmation được đưa vào main result. Một kết quả chỉ được ghi là negative khi không lặp lại trên confirmation seeds, đòi hỏi mức distortion không chấp nhận được, hoặc giả thuyết cơ chế bị dữ liệu bác bỏ.

## 2. Bài toán nghiên cứu đã chốt

### 2.1 Input và output

Input gồm:

- một tabular diffusion backbone đã huấn luyện;
- schema và metadata;
- tập train được chia thành discovery, counterexample và calibration partitions;
- một verified constraint graph chứa hard, near-hard và probabilistic relations;
- với mỗi relation: applicability predicate, requirement, exception, support, target violation rate, uncertainty, provenance và usage role.

Output là synthetic table trong đó:

- relation profile gần held-out real data;
- không ép mọi soft relation về zero violation;
- statistical fidelity, coverage và downstream utility được giữ;
- số lần can thiệp, runtime và distortion được đo rõ.

### 2.2 Dynamic có nghĩa gì

Dynamic không chỉ có nghĩa là “rule phụ thuộc row”. Active graph còn thay đổi theo reverse state:

\[
G_i(t)=\{e\in G:\operatorname{Readable}_e(t)\land
\operatorname{Applicable}_e(\hat x_{0,t}^{(i)})\land
\operatorname{Stable}_e(i,t)\land
\operatorname{Authorized}_e\}.
\]

Một edge có thể:

- chưa readable ở early steps;
- được active khi antecedent xuất hiện;
- bị deactive khi categorical state flip;
- chỉ được dùng để score hoặc monitor, không được actuation;
- được abstain nếu violation mass chưa vượt target rate;
- được active trở lại khi violation bền qua nhiều steps.

Đây là điểm phân biệt với graph cố định, final repair và fixed differentiable guidance.

## 3. Method hiện tại

### 3.1 Verified graph discovery

Candidate relations đến từ ba nguồn:

1. LLM external knowledge và schema reasoning;
2. train-only statistical discovery;
3. predefined domain/schema rules.

Candidate được compile thành typed executable predicates. Pipeline sau đó chạy:

1. type checking;
2. discovery-partition support/violation audit;
3. counterexample search;
4. calibration-partition audit;
5. semantic critic;
6. role assignment: `actuate`, `score_only`, `monitor`;
7. held-out evaluation chỉ dùng để báo cáo.

LLM hiện nằm ở knowledge proposal/semantic verification, không được gọi cho từng row ở từng reverse step. Các predicate đã compile được thực thi cục bộ. Thiết kế này phù hợp với bằng chứng trước đó: compile một lần rồi deterministic execution chính xác và rẻ hơn repeated direct LLM judgment.

### 3.2 Trajectory probing và readability

Tại mỗi reverse step, denoiser tạo predicted-clean estimate `x̂₀,t`. Mỗi edge có readability schedule riêng, được ước lượng bằng:

- agreement với clean/final state;
- categorical flip rate;
- numerical error;
- stability của applicability và satisfaction;
- verifier precision/recall theo timestep.

Early preview không được xem là đáng tin mặc định. Action chỉ mở khi edge đạt cutoff và state đủ bền.

### 3.3 Active graph controller

Controller thực hiện theo row và timestep:

1. decode `x̂₀,t`;
2. đánh giá applicability, satisfaction và exception;
3. cập nhật dynamic topology;
4. ước lượng excess violation so với target rate;
5. chỉ cấp quyền actuation cho active excess;
6. giữ `score_only` edges làm guardrail khi chọn action;
7. abstain nếu không có action cải thiện objective.

Soft relation dùng target rate từ train partitions; mục tiêu không phải satisfaction bằng 1. Hard relation có thể dùng target bằng 0 nếu verification đủ mạnh.

### 3.4 Action layer

Với một violated edge, action layer không luôn có một đáp án duy nhất. Nó:

- khóa applicability columns để tránh “sửa” bằng cách xóa antecedent;
- sinh typed candidate actions trên requirement columns;
- đánh giá residual graph error, edit cost và train-density;
- xử lý connected components bằng local hoặc component beam search;
- dùng feedback-adaptive guidance strength khi cần;
- giới hạn số action trên row-edge và cooldown;
- có thể chạy residual final repair sau trajectory, nhưng dòng này luôn được tách khỏi trajectory-only.

### 3.5 Graph đang dùng

| Dataset | Nodes | Edges | Actuate | Score only | Monitor | Quan hệ chính |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Default | 9 | 8 | 8 | 0 | 0 | distributional và temporal BILL/PAY relations |
| Shoppers | 15 | 27 | 12 | 7 | 8 | process, duration/count, rate, temporal và distributional |
| Diabetes, bản cũ | 6 | 17 | 4 | 13 | 0 | medication logic và empirical dependencies |
| Diabetes, mở rộng | 14 | 22 | 6 | 15 | 1 | medication, admission, diagnosis, count và duration relations |

Việc mở rộng graph không được đánh giá bằng số cạnh đơn thuần. Edge phải có support, verifier executability, directionality và role rõ ràng.

## 4. Protocol thực nghiệm

### 4.1 Baselines matched

- **TabDiff:** sampling không guidance.
- **Final repair:** cùng graph, target, action candidates và certifier; chỉ sửa sau sampling.
- **Frozen trajectory:** đọc initial active state rồi giữ topology cố định trong active window.
- **Dynamic trajectory:** refresh applicability/state theo `x̂₀,t`.
- **Dynamic + residual repair:** full method; final pass chỉ xử lý residual còn lại.
- **Harpoon-style proxy:** fixed differentiable/logit energy trên cùng TabDiff backbone, graph, seed và sample count. Đây là local proxy, không phải official reproduction.

### 4.2 Tách tuning và confirmation

- Hyperparameter được chọn bằng calibration split và development seed.
- Held-out split không dùng để chọn configuration.
- Main result chỉ dùng three-seed confirmation.
- Development result được phép mô tả để giải thích quá trình, không được thay main number.

### 4.3 Công thức metric chính

Với edge `e`, gọi `r_e^syn` là violation rate trên synthetic rows mà edge applicable, và `r_e^test` là rate do independent verifier đo trên real held-out:

\[
E_{profile}=\frac{1}{|E|}\sum_e |r_e^{syn}-r_e^{test}|.
\]

Đây là **held-out constraint-profile error**. Metric này phù hợp hơn raw satisfaction vì nhiều edge là probabilistic: ép violation về zero có thể làm sai real distribution.

## 5. Main result đã xác nhận

### 5.1 Default: graph 8 cạnh

| Method | Held-out profile error ↓ | Calibration error ↓ | Macro violation | Numeric KS ↓ | Cat. TV ↓ | Interventions | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TabDiff | 0.05452 ± 0.00995 | 0.03991 | 0.17545 | 0.03086 | **0.01763** | 0 | 0 |
| Final repair | 0.02755 ± 0.00086 | **0.01338** | 0.14563 | 0.03086 | 0.01897 | 180.7 | 0 |
| Harpoon-style proxy | 0.02792 ± 0.00421 | 0.02122 | **0.13746** | **0.03078** | 0.02075 | 7,130.3 editable | 3.68 s |
| Frozen trajectory | 0.02402 ± 0.00103 | 0.02051 | 0.13779 | 0.03084 | 0.02006 | 197.3 | 77.95 s |
| **Dynamic trajectory** | **0.02250 ± 0.00079** | 0.01852 | 0.13849 | 0.03083 | 0.02013 | 195.3 | 72.81 s |

Diễn giải:

- dynamic giảm profile error `58.7%` so với TabDiff;
- giảm `18.3%` so với final repair;
- giảm `19.4%` so với Harpoon-style proxy;
- giảm `6.4%` so với frozen topology và thắng `3/3` seed;
- Harpoon có raw macro violation thấp hơn nhưng profile error cao hơn: đây là over-enforcement, không phải semantic fidelity tốt hơn.

### 5.2 Shoppers: graph 27 cạnh

| Method | Held-out profile error ↓ | Calibration error ↓ | KS ↓ | TV ↓ | Affected drift ↓ | Actions | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TabDiff | 0.01978 ± 0.00269 | 0.01830 | **0.02510** | 0.02241 | 0 | 0 | 0 |
| Final repair | 0.01262 ± 0.00275 | 0.01033 | 0.02516 | 0.02212 | **0.02996** | 164.7 | 0 |
| Frozen trajectory | 0.01908 ± 0.00273 | 0.01736 | 0.02520 | 0.02215 | 0.04006 | 291.0 | 72.64 s |
| Dynamic trajectory | 0.01268 ± 0.00299 | 0.01035 | 0.02530 | 0.02200 | 0.03495 | 613.3 | 32.11 s |
| Frozen + residual repair | 0.01304 ± 0.00284 | 0.01073 | 0.02520 | 0.02212 | 0.03794 | 442.3 | 72.64 s |
| **Dynamic + residual repair** | **0.01189 ± 0.00281** | **0.00972** | 0.02530 | **0.02196** | 0.03470 | 639.0 | 32.11 s |

Diễn giải:

- full method giảm profile error `39.9%` so với TabDiff;
- giảm `5.8%` so với final repair, thắng `3/3` seed;
- giảm `8.8%` so với frozen + residual repair, thắng `3/3` seed;
- trajectory-only giảm khoảng `33.5%` so với frozen trajectory. Đây là evidence trực tiếp cho dynamic middle-step activation;
- KS/TV gần như không đổi; affected drift tăng nhẹ so với final repair nhưng nằm rất xa các aggressive configurations đã bị loại.

### 5.3 Diabetes: graph 17 cạnh, boundary result

| Method | Held-out profile error ↓ | Calibration error ↓ | KS ↓ | TV ↓ | Affected drift ↓ | Actions | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TabDiff | 0.01391 ± 0.00209 | 0.01354 | 0.02018 | **0.01992** | 0 | 0 | 0 |
| Final repair | 0.00630 ± 0.00243 | **0.00593** | 0.02018 | 0.02008 | 0 | 31.0 | 0 |
| Frozen trajectory | 0.00885 ± 0.00163 | 0.00849 | 0.02014 | 0.01997 | 0.06231 | 28.3 | 370.48 s |
| **Dynamic trajectory** | **0.00626 ± 0.00242** | 0.00611 | **0.02007** | 0.02044 | 0.03020 | 158.0 | **114.42 s** |
| Frozen + residual repair | 0.00631 ± 0.00243 | 0.00594 | 0.02014 | 0.02006 | 0.04410 | 40.0 | 370.48 s |
| Dynamic + residual repair | 0.00623 ± 0.00244 | 0.00607 | **0.02007** | 0.02044 | 0.03014 | 158.3 | **114.42 s** |

Diễn giải:

- dynamic trajectory giảm profile error `29.3%` so với frozen trajectory và thắng `3/3` seed;
- runtime giảm khoảng `3.2×` so với frozen component search;
- sau residual repair, dynamic chỉ tốt hơn final repair `1.1%`, thắng `2/3` seed. Đây không phải gain đủ mạnh để dùng làm headline;
- result này vẫn quan trọng vì chứng minh topology có tác dụng trước post-processing, nhưng edge set cũ quá tập trung vào medication variables.

### 5.4 Diabetes: graph mở rộng 22 cạnh

| Method | Held-out profile error ↓ | Calibration error ↓ | Macro violation | KS ↓ | TV ↓ | Affected drift ↓ | Actions | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TabDiff | 0.01391 ± 0.00209 | 0.01354 | 0.02065 | 0.02018 | **0.01992** | 0 | 0 | 0 |
| Final repair | 0.00630 ± 0.00243 | 0.00593 | **0.01304** | 0.02018 | 0.02005 | 0 | 32.0 | 0 |
| Dynamic trajectory | 0.00631 ± 0.00245 | 0.00594 | 0.01310 | **0.02018** | 0.02013 | 0.02826 | 151.7 | 119.16 s |
| **Dynamic + residual repair** | **0.00628 ± 0.00247** | **0.00590** | 0.01306 | **0.02018** | 0.02013 | 0.02821 | 152.0 | 119.16 s |

Full method giảm `54.9%` so với TabDiff và thắng final repair `2/3` seed, nhưng relative gain trên mean chỉ `0.28%`. Việc tăng từ 17 lên 22 cạnh và từ 6 lên 14 nodes mở rộng phạm vi knowledge graph, song không tạo gain đủ lớn trên metric tổng hợp. Đây là evidence rằng bottleneck không còn nằm ở số cạnh; final repair đã giải gần hết actionable error của graph này.

## 6. Standard quality và utility

### 6.1 Default, ba seeds

| Method | Shape error ↓ | Trend error ↓ | Alpha precision ↑ | Beta recall ↑ | Detection ↑ | XGBoost AUC ↑ | DCR, 50 ideal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TabDiff | 2.0030 | 2.2268 | 98.4531 | **52.5347** | 0.9553 | 0.7227 | 52.2705 |
| Final repair | **1.9847** | **2.2017** | 98.5738 | 52.4729 | 0.9523 | 0.7241 | **52.0101** |
| Harpoon-style proxy | 2.0310 | 2.3122 | **98.5917** | 52.3161 | 0.9535 | 0.7238 | 52.1240 |
| Dynamic trajectory | 2.0305 | 2.2843 | 98.5426 | 52.3958 | **0.9574** | **0.7265** | 52.2217 |

Shape/Trend của final repair tốt nhất; dynamic tốt nhất về detection và downstream AUC. Chênh lệch nhỏ, nên kết luận đúng là semantic gain không đi cùng quality collapse, không phải dynamic đạt SOTA trên mọi fidelity metric.

### 6.2 Shoppers, ba seeds

| Method | Shape error ↓ | Trend error ↓ | Alpha precision ↑ | Beta recall ↑ | Detection ↑ | XGBoost AUC ↑ | DCR, 50 ideal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TabDiff | **1.5925** | **1.7499** | **98.4801** | 48.7099 | 0.9923 | 0.8984 | **49.9674** |
| Final repair | 1.6158 | 1.8192 | 98.1060 | 48.8509 | 0.9938 | **0.8995** | 49.8942 |
| Frozen + residual | 1.5986 | 1.8235 | 98.0561 | 48.9063 | 0.9906 | 0.8973 | 49.9186 |
| Dynamic + residual | 1.6019 | 1.8242 | 98.0219 | **48.9589** | **0.9956** | 0.8945 | 49.8617 |

Dynamic đạt semantic profile tốt nhất và detection/beta tốt, nhưng AUC thấp hơn final khoảng `0.005`. Đây là trade-off nhỏ cần báo cáo, không được bỏ khỏi main quality table.

### 6.3 Diabetes mở rộng

| Method | Shape error ↓ | Trend error ↓ | Alpha precision ↑ | Beta recall ↑ | Detection ↑ | XGBoost AUC ↑ | DCR, 50 ideal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TabDiff | **1.8570** | 8.5872 | 91.5137 | 53.7495 | 0.9300 | **0.6535** | 52.0752 |
| Final repair | 1.8618 | **8.2127** | 91.3856 | **53.9681** | **0.9351** | 0.6482 | **52.0671** |
| Dynamic trajectory | 1.8682 | 8.5696 | **91.7388** | 53.7174 | 0.9254 | 0.6503 | 52.1322 |
| Dynamic + residual | 1.8684 | 8.7238 | 91.7361 | 53.7174 | 0.9263 | 0.6500 | 52.1322 |

Các khác biệt nhỏ và không có một method thắng đồng loạt. Dynamic giữ Alpha precision tốt nhất và AUC cao hơn final repair nhẹ, nhưng thấp hơn TabDiff về AUC/detection. Kết luận phù hợp là không có collapse lớn, đồng thời không có quality gain để dùng làm headline trên Diabetes.

## 7. Metric nào nên là chính

### 7.1 Primary metrics

1. **Held-out constraint-profile error:** metric trực tiếp cho research question; independent verifier, không dùng guidance score làm evaluator.
2. **Paired gain so với frozen topology:** định lượng contribution của việc graph thay đổi theo trajectory, tránh nhầm với lợi ích chỉ do có graph.
3. **Paired gain so với final repair:** trả lời liệu intervention trong trajectory có giá trị hơn hậu xử lý hay không.
4. **Shape/Trend + Alpha precision/Beta recall:** statistical fidelity và coverage theo họ metric phổ biến của tabular synthesis.
5. **Downstream utility:** XGBoost AUC trên real held-out. Nếu dataset/task hỗ trợ thêm F1, báo cả hai nhưng không chọn metric tốt hơn sau khi thấy kết quả.

Nên trình bày primary result dưới dạng Pareto: profile error cùng utility/fidelity, thay vì một scalar tổng hợp do ta tự đặt trọng số.

### 7.2 Metrics báo thêm

- calibration constraint-profile error;
- macro/micro raw violation;
- per-edge error và edge coverage;
- numerical KS, categorical TV;
- affected-row drift và density delta;
- action count, unique rows, runtime, peak memory;
- topology flips, active edges theo timestep;
- DCR privacy diagnostic;
- API calls/tokens của graph discovery, không tính vào sampling runtime nếu graph được compile offline.

### 7.3 Metric không nên dùng làm headline

- **Satisfaction hoặc raw violation đơn độc:** hard guidance có thể đạt zero violation bằng over-enforcement.
- **Calibration error đơn độc:** calibration split là nơi chọn configuration; held-out profile mới là kết quả chính.
- **Best single seed:** mọi main number phải là mean ± std trên fixed confirmation seeds.
- **LLM self-score:** chỉ dùng để proposal/diagnostic, không làm evaluator duy nhất.
- **Một quality composite tự thiết kế:** dễ bị phản biện vì trọng số chủ quan.

## 8. Hypotheses và negative results

### 8.1 Được ủng hộ

| Hypothesis | Kết luận | Bằng chứng |
| --- | --- | --- |
| Dynamic topology có giá trị hơn frozen topology | Ủng hộ | Default `6.4%`, Shoppers trajectory-only `33.5%`, Diabetes-17 `29.3%` |
| Rate-aware guidance tốt hơn ép hard satisfaction | Ủng hộ | Default dynamic profile error tốt hơn Harpoon proxy dù raw violation cao hơn |
| Mid/late trajectory intervention có thể hơn final repair | Ủng hộ có điều kiện | Default `18.3%`; Shoppers full method `5.8%`; không mạnh trên Diabetes-17 |
| Graph lớn hơn vẫn điều khiển được | Ủng hộ sơ bộ | Shoppers 27 edges; Diabetes 22-edge confirmation |
| Compile LLM knowledge thành executable graph tốt hơn gọi LLM liên tục | Ủng hộ trong probe | NL→DSL execution đạt 1.0, khoảng 1.85k tokens; direct judgment kém ổn định và khoảng 37.5k tokens |

### 8.2 Negative results cần giữ trong paper

1. **Early states không đủ đáng tin.** Precision/recall thấp ở đầu trajectory; arbitrary early intervention dễ tạo feedback sai.
2. **Middle window không phổ quát.** Diabetes hard categorical edges cũ chỉ readable gần `t≈0.082`; cửa sổ `[0.15, 0.60]` tạo zero action.
3. **Tăng guidance weight không tạo breakthrough ổn định.** Weight lớn giảm raw violation nhưng profile error hoặc distortion tăng.
4. **Aggressive Shoppers configurations bị loại.** Một số development runs có error thấp hơn nhưng affected drift khoảng `0.18–0.24 SD` và density shift lớn; không dùng làm main result.
5. **Point target từ một split không bền.** Partition shift khiến target sai; robust multi-partition estimate tốt hơn trên Default.
6. **Graph fine-tuning chưa tăng held-out accuracy.** Nó giảm action count/runtime nhưng không thắng continued-training control; hai trên ba edge còn readable muộn hơn.
7. **Graph mở rộng không tự động tốt hơn.** Default 12-edge từng kém local controller khoảng `4%`; edge quality và target calibration quan trọng hơn số lượng.
8. **Repeated direct LLM judgment chưa đáng tin.** Một audit cũ bị label leakage qua `case_id`; kết quả đó bị loại hoàn toàn khỏi evidence. Probe sạch cho thấy compile→DSL ổn định hơn direct row judgment.
9. **Development gain có thể co lại khi confirmation.** Diabetes-17 development từng hơn final khoảng `22.8%`, confirmation chỉ còn `1.1%`; main table dùng confirmation number.
10. **Wall-clock overhead còn cao.** Default dynamic khoảng `72.8 s` so với Harpoon proxy `3.68 s`; method sparse về action nhưng chưa tối ưu runtime.

Các kết quả tốt đã xác nhận không nằm trong danh sách negative. Ví dụ Shoppers full method thắng final/frozen trên `3/3` seed nên được giữ trong main table.

## 9. Claim nên duy trì

### 9.1 Claim chính

> Row- and timestep-specific activation of a verified constraint graph improves held-out semantic relation-profile fidelity over frozen activation and matched post-generation correction, while preserving standard data quality within a small trade-off range.

### 9.2 Claim phương pháp

> The key object is not a fixed constraint set but a trajectory-indexed active subgraph with calibrated authority, abstention and typed actions.

### 9.3 Claim về LLM

> Foundation models are used to propose and semantically organize candidate knowledge; data-grounded verification and deterministic execution prevent the LLM from being the final authority.

### 9.4 Claim cần tránh

- LLM feedback trong diffusion là hoàn toàn mới;
- per-step constraint enforcement là mới;
- dynamic luôn hơn final repair trên mọi dataset;
- method đạt SOTA về synthetic tabular fidelity nói chung;
- raw satisfaction càng cao càng tốt;
- mọi edge trong graph đều do LLM khám phá;
- Harpoon proxy là official reproduction.

## 10. Main result cần duy trì khi chạy thêm

Không thay đổi các thành phần sau nếu không khai báo một protocol mới:

1. frozen train/test split và evaluation graph;
2. 4.096 rows, 50 reverse steps;
3. confirmation seeds đã chốt;
4. cùng backbone checkpoint giữa method và matched baselines;
5. graph/target/action budget giống nhau giữa dynamic, frozen và final repair;
6. held-out verifier không được dùng cho tuning;
7. báo cả trajectory-only và trajectory + residual repair;
8. common quality evaluator giống nhau cho mọi sample;
9. mean ± standard deviation và paired wins;
10. phân biệt official paper number, official rerun và local proxy.

Nếu phát triển architecture mới, nó nên được thêm như một dòng mới, không thay thế protocol hiện tại. Bảng hiện tại là anchor để tránh moving target.

## 11. Paper outline đề xuất

### 1. Introduction

- Tabular constraints gồm hard, near-hard và probabilistic relations.
- Fixed constraint program có hai lỗi: act quá sớm khi state chưa readable, hoặc over-enforce soft relations.
- Final repair không tác động được trajectory và có thể chọn một mode sửa cục bộ.
- Đề xuất verified dynamic constraint graph theo row và timestep.
- Tóm tắt ba contributions và main numbers.

### 2. Related Work

- tabular diffusion: TabSyn, TabDiff;
- fixed differentiable constraint guidance: Harpoon, CTDF, constrained discrete diffusion;
- LLM/static prior: LLM-TabLogic, SCTab-Diff;
- discovery và post-repair: LCATS, Zhao et al., EVS;
- value-dependent generation: SAGE;
- intermediate-preview feedback ngoài tabular: PromptLoop, PPAD, Free²Guide, TTGen.

Boundary cần nói rõ: SCTab-Diff và Harpoon cũng condition/update theo current state; novelty không phải “state-dependent loss” chung chung. Contribution nằm ở verified graph whose active topology and authority change per row/step, với soft-rate targets và abstention.

### 3. Problem Formulation

- verified graph schema;
- hard/soft relation profiles;
- active graph `G_i(t)`;
- objective: profile fidelity + distribution fidelity + intervention cost;
- threat model: noisy preview, wrong constraint, exception, conflicting actions.

### 4. Method

4.1 Knowledge proposal and data-grounded verification  
4.2 Trajectory readability estimation  
4.3 Dynamic active graph construction  
4.4 Rate-aware authority and abstention  
4.5 Typed action selection and residual repair

### 5. Experimental Setup

- datasets, graph statistics và train/test protocol;
- TabDiff backbone;
- baselines;
- primary/secondary metrics;
- hyperparameter selection và seeds;
- compute/API cost.

### 6. Main Results

- Table 1: dataset/graph statistics;
- Table 2: profile error cho TabDiff, final, frozen, Harpoon proxy, dynamic;
- Table 3: standard fidelity/utility;
- Figure 1: trajectory readability;
- Figure 2: one-row active graph evolution;
- Figure 3: semantic gain–distortion/runtime Pareto.

### 7. Analysis and Ablation

- dynamic vs frozen;
- trajectory-only vs residual repair;
- hard vs calibrated target;
- active window/readability gate;
- graph size/role assignment;
- action planner và guidance strength;
- LLM proposal vs statistical/predefined sources.

### 8. Negative Results and Limitations

- Diabetes late readability;
- end-to-end fine-tuning result;
- graph quality sensitivity;
- runtime;
- no official mixed-type Harpoon reproduction;
- three-seed uncertainty và dataset coverage.

### 9. Conclusion

Không kết luận “LLM luôn cần thiết”. Kết luận rằng verified semantic knowledge có ích nhất khi authority được hiệu chỉnh theo row và trajectory, còn LLM là một source/organizer trong discovery pipeline.

## 12. Việc còn lại trước khi khóa submission

Ưu tiên cao:

1. khóa Diabetes expanded confirmation và common metrics;
2. thêm mixed-type Harpoon/CTDF-style proxy có operator coverage rõ trên Shoppers hoặc Diabetes;
3. chạy ít nhất một backbone hoặc dataset bổ sung nếu thời gian cho phép;
4. tạo figure trajectory/active graph và semantic–fidelity Pareto;
5. profile runtime, cache graph masks và batch action scoring;
6. chạy statistical test/paired bootstrap trên rows hoặc nhiều sampling seeds hơn cho bảng cuối.

Ưu tiên vừa:

- downstream F1 bên cạnh AUC;
- subgroup fidelity và rare-mode coverage;
- per-edge confidence intervals;
- API token/cost table cho graph discovery;
- graph source ablation: statistical only, LLM only, combined.

Không nên ưu tiên lúc này:

- gọi LLM ở mọi reverse step;
- thêm nhiều engineering modules không gắn với hypothesis;
- tăng số edge mà không qua counterexample/role audit;
- sweep lớn trên held-out metric.

## 13. Artifacts

- Main aggregation script: `scripts/aggregate_paper_tables_20261011.py`
- Frozen paper tables: `outputs/paper_tables_20261011/` (`main_results_*` và `common_quality_*`)
- Default main: `outputs/graph_rate_8edge_confirmation_20261010/RESULTS.md`
- Default Harpoon: `outputs/harpoon_default_8edge_confirm_s*_1067613_*/`
- Shoppers confirmation: `outputs/shoppers_promote_confirm_s*_1067743_*/`
- Shoppers common metrics: `outputs/shoppers_promoted_common_s*_1068460_*/`
- Diabetes-17 confirmation: `outputs/diabetes_promote_confirm_s*_1067739_*/`
- Diabetes-22 confirmation: `outputs/diabetes_expanded_confirm_s*_1068457_*/`
- Diabetes common metrics: `outputs/diabetes_expanded_common_s*_1068463_*/`
- Slurm configs: `slurm/505_diabetes_expanded_confirm.sbatch`, `slurm/506_shoppers_promoted_common_quality.sbatch`, `slurm/507_diabetes_expanded_common_quality.sbatch`

## 14. Provenance labels

- **Đã chạy:** mọi số trong các bảng Default, Shoppers, Diabetes và common metrics có output local tương ứng.
- **Local proxy:** Harpoon-style trên Default; cùng mechanism family nhưng không phải official code reproduction.
- **Paper-reported:** không trộn vào ranking trong báo cáo này.
- **Suy luận:** novelty boundary, paper framing và danh sách ưu tiên.
- **Chưa thử đầy đủ:** mixed-type Harpoon trên Shoppers/Diabetes, backbone khác, end-to-end graph training ổn định trên cả ba datasets.
