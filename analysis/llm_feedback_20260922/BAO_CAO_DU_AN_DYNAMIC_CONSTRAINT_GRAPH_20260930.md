# Báo cáo dự án: Dynamic Constraint Graph cho synthetic tabular diffusion

Ngày cập nhật: 30/09/2026

## 1. Tóm tắt điều hành

Dự án bắt đầu từ ý tưởng dùng LLM quan sát dự đoán bản ghi sạch `x_hat_0(x_t)` tại một số reverse diffusion steps, phát hiện lỗi ngữ nghĩa và đưa feedback trở lại sampler. Các phép thử đã thu hẹp đáng kể hypothesis ban đầu.

Kết quả hiện tại không ủng hộ một claim rộng rằng LLM feedback giữa denoising luôn tốt hơn repair. Với constraint số học đơn giản, direct projection, diffusion imputation và best-of-8 là các baseline rất mạnh. Trên Shoppers, best-of-8 tốt hơn mid-denoising conditioning ở năm trong sáu fidelity metrics của single-threshold benchmark. Với conjunction chỉ gồm lower bounds, direct clamp tiếp tục thắng rõ.

Tín hiệu tích cực nhất xuất hiện ở constraint có nhiều cách sửa hợp lệ. Với cùng violation quan sát được `count=0, duration>0`, nguyên nhân thật có thể là count bị hỏng hoặc duration bị hỏng. Dynamic graph chọn endpoint cần resample theo từng bản ghi làm giảm counterfactual pair error khoảng 31--39% so với fixed-route TabDiff imputation, lặp lại qua hai seed và một compute ablation. Independent certifier đưa mọi phương pháp về validity 1.0 nhưng không xóa chênh lệch reconstruction error. Điều này cho thấy validity đơn thuần không đo được việc sửa đúng hay sai ngữ nghĩa.

Hướng nghiên cứu được đề xuất hiện nay là một **verified dynamic constraint graph** làm tầng solver routing. Graph biểu diễn constraint nào áp dụng, bản ghi đang satisfied/violated/not-applicable/exception, endpoint và descendant nào được phép thay đổi, và actuator nào nên dùng. LLM chủ yếu compile natural-language policy và semantic exception thành typed graph; deterministic verifier kiểm tra graph; runtime controller chọn projection, best-of-N, imputation, conditional denoising hoặc abstention. LLM không nên thay rule engine ở các constraint thực thi được.

Đây chưa phải kết quả SOTA và chưa đủ cho submission. Nó đã tạo được một research question rõ, một negative result quan trọng, một method hypothesis có bằng chứng ban đầu, cùng pipeline và strong baselines để tiếp tục.

## 2. Bài toán nghiên cứu

### 2.1 Đầu vào

- Một pretrained tabular diffusion generator cho bảng mixed-type.
- Schema, kiểu dữ liệu, metadata và một runtime semantic request hoặc constraint program.
- Có thể có subgroup condition, exception clause và nhiều endpoint có thể sửa.
- Ở thời điểm chạy API bên ngoài, chỉ schema/metadata và synthetic rows được phép rời server; raw real rows không được gửi.

### 2.2 Đầu ra mong muốn

Sinh bảng synthetic thỏa semantic request nhưng vẫn bảo toàn:

- marginal và joint fidelity;
- subgroup-conditional fidelity;
- rare-mode coverage và valid exceptions;
- diversity và downstream utility;
- chi phí sampling/API hợp lý.

### 2.3 Câu hỏi trung tâm đã sửa

Thay vì hỏi chung “LLM feedback có cải thiện denoising không?”, câu hỏi phù hợp với evidence hiện tại là:

> Khi một semantic violation có applicability phụ thuộc bản ghi và nhiều repair routes hợp lệ, dynamic constraint graph có chọn đúng active constraint, endpoint, descendant scope và solver để giảm semantic/counterfactual distortion hơn static guidance, projection, best-of-N và post-repair hay không?

Câu hỏi phụ cho foundation model là:

> Foundation model có compile unseen natural-language policies, subgroup conditions và exceptions thành typed graph có thể kiểm chứng, hoặc chọn semantic route trên held-out schemas, tốt hơn parser/statistical router hay không?

## 3. Motivation

### 3.1 Validity không đồng nghĩa với sửa đúng

Với implication `count=0 -> duration=0`, khi quan sát `count=0, duration>0`, cả hai thao tác `duration=0` và `count>0` đều có thể tạo row hợp lệ. Nếu count mới là cột bị hỏng, luôn clamp duration tạo một row hợp lệ nhưng xa clean counterfactual. Đây là failure mode của projection và deterministic repair khi constraint không xác định duy nhất repair route.

### 3.2 Constraint thực tế không phải lúc nào cũng global và hard

Cross-dataset audit có 788 candidates nhưng chỉ 80 empirical zero-exception constraints; 95 near-hard, 334 probabilistic, 85 demographic/sensitive chỉ phù hợp audit, 57 liên quan target và 132 thiếu support. Ví dụ:

- Shoppers: `Administrative=0 -> Administrative_Duration=0` gần exact, nhưng chiều đảo có 2--3% exceptions.
- Adult: `Own-child -> age<30` có khoảng 19.7% exceptions và không an toàn để hard-enforce.
- Cùng dạng dependency có thể là invariant, probabilistic relation, subgroup exception hoặc spurious correlation.

Do đó graph phải biểu diễn direction, applicability, hardness, uncertainty và abstention.

### 3.3 Intermediate semantics và khả năng điều khiển không đồng bộ

Trajectory probes cho thấy condition adapter điều khiển tốt khi kích hoạt sớm, trong khi preview-derived semantic scope đáng tin hơn ở middle/late steps. Late activation không phục hồi được control dù tăng guidance strength. Full-trajectory oracle cải thiện satisfaction từ 0.8195 lên 0.9407 và target KS từ 0.2084 xuống 0.0896, nhưng dynamic preview gate gây thay đổi 48.4% exception rows. Đây là temporal alignment gap có cơ sở thực nghiệm.

### 3.4 Giá trị data-management

Hướng này gắn với integrity constraints, policy changes, exception preservation, explainable repair, provenance của quyết định và quality guarantees. Nó rộng hơn rare generation: trọng tâm là semantic control dưới heterogeneous regimes và ambiguous correction.

## 4. Ranh giới với related work

| Hướng | Cơ chế chính | Ranh giới đối với dự án |
| --- | --- | --- |
| TabSyn | VAE đưa mixed types vào latent liên tục rồi diffusion | Backbone/official baseline thích hợp; không xử lý dynamic semantic routing |
| TabDiff | Diffusion trực tiếp numeric và categorical với feature-wise schedules | Backbone chính hiện tại; official sampling và imputation được dùng trực tiếp |
| Harpoon | Differentiable constraint loss guidance trong sampling | Đã bao phủ per-step differentiable guidance; proxy hiện có chưa phải official reproduction |
| CTDF | Feasibility operator tại từng reverse step cho mixed numerical/categorical constraints | Đã bao phủ per-step hard constraint enforcement; CTDF-style hiện tại chỉ là numeric proxy |
| SCTab-Diff | LLM-derived static prior condition reverse denoising | “LLM prior trong denoising” không còn đủ mới |
| LCATS | Constraint discovery/verification, TabSyn generation, diagnosis và weighted post-repair | Baseline trực tiếp cho post-generation correction; official pipeline đang được dựng |
| Zhao et al. | LLM agents sinh executable rules, counterexample validation và deterministic repair | Compile/validate/repair bằng LLM không đủ mới nếu đứng riêng |
| EVS | LLM noisy oracle, constraint verification và utility-preserving selection | LLM extraction/filtering không đủ mới |
| PromptLoop/PPAD/TTGen | Model nhìn preview/clean estimate rồi refine prompt, rollback hoặc chọn branch | LLM-in-the-loop nói chung đã tồn tại ở image/video; novelty phải đến từ mixed-type graph, applicability, exceptions và repair ambiguity |
| CuTS | Official customizable synthesizer được fine-tune theo constraint program | Official baseline đã chạy trên Adult; cần common-task adaptation để so trực tiếp |

Không claim first per-step constraints, first LLM prior, first graph conditioning hoặc first LLM repair.

## 5. Datasets và phạm vi thực nghiệm

### 5.1 Datasets chính

- Adult: mixed categorical/numeric; dùng cho rule audit, LLM verifier, compiler và CuTS official baseline. Sensitive attributes chỉ dùng audit, không hard-enforce.
- Default: financial mixed-type table; pretrained TabDiff checkpoint, payment/bill policies và subgroup compositions.
- Online Shoppers: mixed-type behavioral table; pretrained TabDiff checkpoint, count-duration relations và controlled runtime policies.

### 5.2 Datasets mở rộng

Cross-dataset graph audit đã gồm Adult, Default, Shoppers, Diabetes, Beijing, News và các prepared tables liên quan. Chúng hiện chủ yếu cung cấp statistical constraint candidates; chưa phải tất cả đều có matched generator checkpoint.

### 5.3 Split discipline

- Rule/candidate discovery chỉ dùng train.
- Heldout dùng cho replication và evaluation.
- Controlled corruption giữ clean counterfactual chỉ để đánh giá và oracle upper bound.
- Threshold audit phân biệt exact training threshold với heldout threshold.

## 6. Method hiện tại

### 6.1 Typed constraint graph

Mỗi constraint node/edge cần chứa:

- typed predicate và direction;
- applicability condition và subgroup;
- exception clause;
- hardness/confidence;
- allowed endpoint và descendant set;
- independent verifier;
- candidate solver và expected cost.

Active set cho row `i` tại step `t` có dạng:

`C_i(t) = Retrieve(request_i) ∩ Applicable(x_i,t) \ Exception(x_i,t)`.

### 6.2 Four-state controller

Mỗi row được phân thành:

1. applicable và violated;
2. applicable và satisfied;
3. not applicable;
4. valid exception.

Chỉ trạng thái đầu được actuation. Các trạng thái còn lại phải được bảo toàn và được đo collateral edit riêng.

### 6.3 Two-pass lookahead và rollback

- Pass 1 sinh preview/`x_hat_0` để xác định active graph.
- Pass 2 rollback với cùng initial noise.
- Anchor antecedent/subgroup context.
- Chỉ resample target hoặc graph-selected descendants.
- Cuối cùng chạy independent certifier.

Thiết kế này xử lý temporal alignment gap giữa semantic readability và controllability.

### 6.4 Solver routing

- Constraint đơn giản, convex hoặc clampable: projection/direct repair.
- Feasible mass cao: rejection hoặc best-of-N.
- Missing/corrupted endpoint: diffusion imputation.
- Coupled relation hoặc feasible mass thấp: conditional denoising/rollback.
- Ambiguous endpoint: row-specific route selector rồi route-specific imputation.
- Verifier không chắc chắn hoặc rule probabilistic: abstain/soft guidance.

### 6.5 Vai trò LLM

LLM được giữ ở các phần có semantic value:

- natural language -> typed predicates/exception clauses;
- đề xuất candidate graph trên unseen schema;
- semantic route selection khi executable statistics không đủ;
- giải thích quyết định.

Parser, statistics và deterministic executor phải là baseline bắt buộc. Kết quả request-condition probe cho thấy compile-once thành DSL rồi execute đạt accuracy 1.0, swap-pair 1.0, khoảng 1.85K tokens; direct repeated judgement chỉ đạt 0.864/0.867 với GPT-4.1-mini và 0.65/0.367 với GPT-5.4-mini, tiêu thụ khoảng 37.5K tokens. Vì vậy repeated row-level LLM judgement không phải default method.

## 7. Metrics

### 7.1 Constraint và safety

- independent violation/satisfaction rate;
- applicability, exception và non-applicable preservation;
- antecedent/subgroup flip rate;
- wrong-column edit rate;
- repair-route accuracy;
- residual certifier edit rate.

### 7.2 Fidelity

- numeric KS và Wasserstein distance;
- categorical total variation;
- target marginal fidelity;
- subgroup-conditional numeric/categorical fidelity;
- correlation/joint dependency error;
- active-row normalized displacement;
- rare-mode coverage và diversity.

### 7.3 Ambiguous repair

- normalized counterfactual pair error;
- P90 pair error;
- route accuracy;
- validity trước và sau certification.

Pair error chuẩn hóa hai endpoint theo train standard deviation; metric này phân biệt repair hợp lệ nhưng sai endpoint.

### 7.4 Utility và cost

- downstream utility toàn bảng và theo subgroup;
- runtime/GPU hours;
- số diffusion branches/steps;
- API calls, tokens và latency.

## 8. Kết quả chính đã chạy

### 8.1 Rule và benchmark audit

Natural hard-rule violation của full checkpoints chỉ khoảng 1.6--1.8%, nên benchmark near-hard bị saturated. Cross-dataset audit tìm thấy phần lớn candidate là probabilistic hoặc thiếu support; điều này phản bác việc biến mọi discovered dependency thành hard constraint.

### 8.2 Four-state active set

Oracle violated-row selection giảm applicability flip từ 0.0171 xuống 0.0029 trên Default và từ 0.0182 xuống 0.0116 trên Shoppers so với applicability-only guidance. Guidance + certifier giảm residual edit rate 42.6% trên Default và 71.2% trên Shoppers so với post-repair, nhưng post-repair vẫn tốt hơn về một số fidelity metrics. Kết luận: active-set selection có ích, nhưng single-target repair quá dễ.

### 8.3 Descendant scope frontier

Trên Default, numeric graph conditioning đạt drift 0.0136, target KS 0.1253, numeric KS 0.0748, categorical TV 0.0351 và correlation error 0.0595. Trên Shoppers, method tạo tradeoff ổn định: tốt hơn empirical repair về displacement/joint metrics nhưng kém target marginal. Sparse top-k3 scope giảm drift trên Shoppers từ 0.8072 xuống 0.7204 nhưng target KS tăng từ 0.2872 lên 0.3030. Không có universal winner.

### 8.4 Matched-scope imputation

Default: conditioning giảm drift và correlation error nhưng matched imputation tốt hơn target KS/Wasserstein. Shoppers: conditioning tốt hơn target KS, Wasserstein, numeric KS và categorical TV, nhưng drift cao hơn 0.3127 SD. Đây là Pareto tradeoff, không phải dominance.

### 8.5 Best-of-8 falsification

Trên Shoppers, best-of-8 đạt drift/target-KS/Wasserstein/numeric-KS/correlation error `0.7307/0.2413/0.8808/0.1482/0.1474`; top-k3 conditioning là `0.7622/0.3130/0.9523/0.1632/0.1719`. Conditioning chỉ tốt hơn categorical TV. Vì vậy mid-denoising condition chưa vượt constraint-aware branch selection trên single threshold.

### 8.6 Conjunction screen

Default: clamp đạt drift 0.0016 và numeric KS 0.1265, tốt hơn conditional denoising 0.0603/0.1844 và best-of-8 0.0487/0.1904. Shoppers cho cùng kết luận. Conjunction của clampable lower bounds không tạo method gap.

### 8.7 Ambiguous relational actuation

| Run | Method sau certification | Route accuracy | Validity | Pair error |
| --- | --- | ---: | ---: | ---: |
| seed 20260930, 3 rounds | context router | 0.7428 | 1.0000 | 0.9586 |
| | fixed duration | 0.5000 | 1.0000 | 1.3959 |
| | fixed count | 0.5000 | 1.0000 | 1.5563 |
| | oracle route | 1.0000 | 1.0000 | 0.4593 |
| seed 20261001, 3 rounds | context router | 0.7406 | 1.0000 | 0.9303 |
| | fixed duration | 0.5000 | 1.0000 | 1.3740 |
| | fixed count | 0.5000 | 1.0000 | 1.5245 |
| | oracle route | 1.0000 | 1.0000 | 0.4407 |
| seed 20260930, 1 round | context router | 0.7428 | 1.0000 | 0.9450 |
| | fixed duration | 0.5000 | 1.0000 | 1.3959 |
| | fixed count | 0.5000 | 1.0000 | 1.4599 |
| | oracle route | 1.0000 | 1.0000 | 0.4865 |

Đây là result tích cực mạnh nhất hiện nay. Tuy nhiên router supervised trên controlled corruption và mới có hai relations trên Shoppers.

Job `1050898` kiểm tra hai cách giảm artificial supervision. Natural-state router chỉ học phân biệt clean zero-zero và clean positive-positive states, loại các endpoint đang test khỏi feature set; nó đạt route accuracy `0.7157` và pair error `1.0599` sau certification. Leave-one-relation-out router học controlled route từ các count-duration relations khác, đạt `0.6873/1.0887`. Cả hai tốt hơn fixed-duration (`1.3832`) và fixed-count (`1.5816`), nhưng còn kém relation-specific supervised router (`0.7468/0.9608`). Tín hiệu routing vì vậy không hoàn toàn phụ thuộc nhãn corruption của đúng relation, song chưa phải cross-table transfer.

### 8.8 LLM và compiler

Direct LLM judgement nhạy với prompt/model và tốn chi phí. NL-to-DSL-to-executor đạt perfect execution equivalence trên request-conditioned bank với ít token hơn khoảng 20 lần. LLM subgroup selector sau safety verification đạt F1 0.957 nhưng accept-all-FDR-significant sau cùng safety filter đạt F1 0.964. Vì vậy LLM chưa vượt statistical selector ở candidate graph hiện tại.

## 9. Baseline matrix

| Baseline | Trạng thái | Mức độ có thể dùng để claim |
| --- | --- | --- |
| TabDiff unconditional/sampling/imputation | Official repository và checkpoint; đã chạy | Direct baseline |
| TabDiff best-of-8 matched scope | Official sampler, selection logic của dự án | Strong baseline nhưng phải ghi selection wrapper |
| CuTS | Official repository; đã chạy Adult | Official baseline; chưa common task với dynamic graph |
| LCATS + TabSyn | Official repository; setup/pipeline đã submit | Khi chạy xong là official baseline trên Insurance |
| TabSyn riêng | Nằm trong LCATS phase 2 | Official generator baseline khi pipeline hoàn tất |
| CTDF-style | Numeric projection proxy đã chạy | Không được gọi là CTDF reproduction |
| Harpoon-style | Differentiable guidance proxy đã chạy | Không được gọi là Harpoon reproduction |
| Clamp/projection/post-repair | Implementation nội bộ | Strong algorithmic baseline |
| Empirical joint repair | Implementation nội bộ, dùng train bank | Data-assisted baseline |
| Natural-state/cross-relation routers | Đang chạy | Ablation nội bộ, không phải prior work |

CuTS official trên Adult đạt constraint score 1.0 ở năm chương trình, với TV khoảng 0.0001--0.0021 và BAC khoảng 0.769--0.776. Thời gian quan sát: baseline 4 phút 30 giây; constrained fine-tuning 1 giờ 57 phút. Kết quả này chưa được so trực tiếp với method mới do task/dataset khác nhau.

## 10. Hypotheses và trạng thái

| Hypothesis | Trạng thái |
| --- | --- |
| Intermediate `x_hat_0` luôn đủ đọc để feedback | Bị thu hẹp; scope đáng tin muộn hơn control window |
| Static prior không đủ cho subgroup/exception | Được ủng hộ về safety, chưa đủ dominance về fidelity |
| Mid-denoising feedback hơn post-repair | Bị bác bỏ trên simple thresholds/clampable conjunctions |
| LLM hơn rule engine | Chưa được ủng hộ; DSL executor/statistical selector mạnh hơn |
| Descendant graph tạo fidelity frontier | Được ủng hộ trên Default và Shoppers, không universal dominance |
| Row-specific endpoint routing có ích | Được ủng hộ trong controlled Shoppers benchmark |
| Foundation model transfer qua datasets | Chưa thử đúng nghĩa; adapters hiện vẫn table-specific |

## 11. Contribution có thể hình thành

### 11.1 Contribution chính có tiềm năng

Một benchmark và method cho ambiguous semantic actuation:

- cùng violation nhưng nhiều valid repair routes;
- đánh giá route recovery và counterfactual error thay vì chỉ validity;
- dynamic graph chọn endpoint, descendant scope và solver;
- independent certification cung cấp final guarantee.

### 11.2 Contribution phụ

- Four-state active-set formulation bảo toàn satisfied/not-applicable/exception rows.
- Phân tích temporal gap giữa semantic readability và controllability.
- Compiler architecture: LLM compile once, typed executor chạy nhiều lần.
- Baseline taxonomy và matched-budget evaluation cho projection, repair, best-of-N, imputation và conditioning.

### 11.3 Claim chưa được phép dùng

- SOTA synthetic tabular generation;
- first LLM-in-denoising;
- first constraint-guided tabular diffusion;
- cross-dataset foundation model;
- vượt LCATS/CuTS/CTDF trước common benchmark.

## 12. Hạn chế hiện tại

1. Ambiguous benchmark dùng controlled corruption và hidden route labels.
2. Positive route result mới tập trung ở Shoppers và hai relations.
3. Router hiện là supervised logistic model, chưa phải LLM hoặc trajectory-aware controller.
4. Chưa có common dataset/task với LCATS và CuTS.
5. CTDF/Harpoon mới là proxy, không phải official reproduction.
6. Chưa có downstream utility và privacy audit đầy đủ cho ambiguous route method.
7. Free-generation setting không có clean counterfactual; cần surrogate evaluation hoặc semi-synthetic benchmark.
8. Foundation transfer cần held-out tables/schemas, không chỉ held-out thresholds.

## 13. Hướng phát triển tiếp theo

### Ưu tiên 1: làm benchmark đủ mạnh

- Thêm ordering, ratio, temporal/accounting và mixed categorical-numeric relations.
- Mỗi violation có ít nhất hai valid repair routes.
- Có clean/semi-synthetic counterfactual và natural free-generation evaluation.
- Bao gồm easy projection cases để chứng minh solver router biết chọn phương pháp đơn giản khi phù hợp.

### Ưu tiên 2: strong prior-work baselines trên common task

- Chạy LCATS/TabSyn official trên Insurance.
- Chuyển dynamic graph method sang Insurance hoặc chuyển LCATS constraint repair sang một prepared common table nếu code cho phép.
- Dùng CuTS official constraint programs làm public benchmark, không chỉ tự tạo policies.
- Tìm hoặc tích hợp CTDF official code trước claim camera-ready; giữ proxy riêng biệt.

### Ưu tiên 3: loại bỏ artificial route supervision

- Natural-state density router.
- Leave-one-relation-out và leave-one-table-out route prediction.
- Best-of-N candidate repair + likelihood/utility scoring.
- Calibrated abstention khi route posterior thấp.

### Ưu tiên 4: trajectory-aware routing

- Đo route stability của `x_hat_0` theo timestep.
- So final-row router với middle-step previews và two-pass lookahead.
- Chỉ claim intermediate feedback nếu trajectory state tăng route accuracy hoặc giảm compute/distortion hơn final-state selection.

### Ưu tiên 5: foundation model

- Compile unseen policy và exception thành typed graph trên held-out schemas.
- So với template parser, retrieval, shallow classifier và statistical graph.
- Chỉ gửi synthetic rows/schema ra API.
- Đo accuracy, calibration, contradiction, token cost và abstention.

## 14. Jobs mới

- `1050896`: dựng environment cho LCATS official.
- `1050897`: chạy LCATS official trên Insurance sau khi setup thành công; bao gồm TabSyn generation và LCATS post-repair.
- `1050898`: natural-state và leave-one-relation-out router ablations trong ambiguous TabDiff actuation; đã hoàn tất thành công.

Các job baseline official có thể chạy lâu; kết quả sẽ được đọc từ log/output ở vòng tiếp theo. Không dùng trạng thái submitted làm experimental evidence.

## 15. Tiêu chí quyết định tiếp tục

Tiếp tục hướng dynamic graph nếu ít nhất một row-specific router không dùng artificial corruption labels giảm counterfactual error so với fixed repair, best-of-N và density-aware post-repair trên từ ba datasets trở lên, đồng thời không giảm utility/diversity đáng kể.

Giữ LLM trong main method chỉ khi nó cải thiện held-out semantic route/graph compilation so với deterministic/statistical baselines sau calibration. Nếu không, LLM chuyển thành optional interface và contribution chính là verified dynamic solver routing.

Không tiếp tục claim mid-denoising feedback nếu final-state branch selection hoặc official post-repair baseline đạt cùng frontier với chi phí thấp hơn. Khi đó kết quả trajectory trở thành analysis contribution, còn method tập trung vào dynamic graph và solver selection.

## 16. Phân tích hypothesis và baseline sau vòng official-baseline setup

### 16.1 Hypothesis A — Dynamic solver routing cho ambiguous constraints

Đây là hypothesis mạnh nhất hiện nay. Evidence hiện có cho thấy supervised, natural-state và cross-relation routers đều giảm pair error so với fixed endpoint repair. Khoảng cách oracle còn lớn, nên method chưa bão hòa.

Falsification test bắt buộc:

- density-aware candidate selection hoặc best-of-N đạt ngang router;
- gain biến mất khi chuyển khỏi controlled corruption;
- route accuracy tăng nhưng downstream utility/joint fidelity không tăng;
- graph chỉ học dataset-specific correlations và không transfer qua relation/table.

Nếu vượt các test này, contribution có thể là dynamic graph chọn endpoint/solver, không phụ thuộc LLM cụ thể.

### 16.2 Hypothesis B — Intermediate diffusion state có thông tin bổ sung

Hypothesis này chỉ còn hợp lệ dưới dạng incremental claim:

> Với cùng final preview và candidate routes, trajectory features hoặc middle-step disagreement dự báo đúng endpoint/solver tốt hơn final-row context và cho phép dừng sampling sớm.

Không đủ nếu intermediate controller chỉ bằng final-state router. Baselines bắt buộc gồm final clean estimate, final synthetic row, two-route lookahead score và trajectory-free context router. Metrics là route AUROC/accuracy, calibration, pair error và diffusion evaluations per valid row.

### 16.3 Hypothesis C — Foundation model cho held-out semantic graph

Foundation model có cơ hội ở cross-schema semantics, không phải executable arithmetic. Evaluation phải hold out toàn bộ tables hoặc relation families. Baselines gồm template parser, embedding retrieval, statistical graph, shallow classifier và oracle DSL. LLM phải cải thiện execution-equivalent graph/route decisions sau type checking; semantic rationale không được dùng làm metric chính.

### 16.4 Hypothesis D — Một condition layer dùng chung qua datasets

Đây là hướng impact cao nhưng rủi ro lớn. Hiện mỗi table có adapter riêng nên chưa có foundation claim. Một shared schema-aware condition encoder có thể điều khiển nhiều pretrained diffusion backbones hoặc một multi-table backbone. Test nhỏ hợp lý là giữ encoder chung, chỉ cho phép table-specific input/output projection, rồi hold out một relation family hoặc một table. Chưa nên đầu tư full training trước khi Hypothesis A sống qua common-task baselines.

### 16.5 Thứ tự ưu tiên baseline

1. Same generator, same constraints, same sample budget: unconditional, projection, repair, best-of-N và imputation.
2. Official prior-work code trên public programs: CuTS Adult; LCATS/TabSyn Insurance.
3. Common evaluator: constraint score, fidelity và utility phải được tính bằng cùng code khi đưa vào bảng claim.
4. Proxy baselines chỉ dùng mechanism analysis, không dùng cho câu “vượt prior work”.

### 16.6 Common-task experiments mới

Job `1051083` chạy official Adult TabDiff checkpoint trên đúng năm public CuTS programs: `implication2`, `implication4`, `implication5`, `line_constraint1`, `line_constraint2`. Nó đo unconditional, best-of-8 + fallback và final repair. Đây là bước xác định difficulty/common-task floor; dynamic graph chưa được đưa vào nên chưa phải comparison cuối.

LCATS environment jobs `1050896` và `1051081` thất bại ở dependency resolution của official `synthcity` evaluation stack, không phải method code. Job `1051084` chỉ dựng dependencies cho phases 1--5; `1051085` sẽ chạy official constraint discovery, TabSyn generation, diagnosis và aligned repair. Phase 6 evaluation được hoãn và phải ghi rõ.
