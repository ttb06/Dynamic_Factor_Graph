# Báo cáo định hướng: Dynamic Constraint Graph cho tabular diffusion

Ngày cập nhật: 01/10/2026

## 1. Kết luận định hướng

Dự án nên tiếp tục bám sát **dynamic constraint graph**, nhưng graph cần được định nghĩa rộng hơn một tập rule bật/tắt hoặc một router chọn cột sửa. Graph là tầng trung gian có thể kiểm chứng giữa nhiều nguồn constraint và generator, thay đổi theo dataset, row, diffusion state và distribution regime. Natural-language semantic request là một nguồn specification tùy chọn, không phải điều kiện bắt buộc của bài toán.

Một formulation đủ rộng gồm bốn graph liên kết:

1. **Specification graph**: constraint, direction, subgroup, exception và provenance từ user request, domain metadata hoặc data discovery.
2. **Evidence graph**: support, heldout stability, hardness, uncertainty và sensitive/target flags.
3. **Active graph**: node/edge nào áp dụng cho từng row tại timestep hiện tại.
4. **Action graph**: endpoint, descendants, solver, thời điểm và budget được phép can thiệp.

Pipeline tổng quát:

`optional request + schema/metadata/data evidence -> graph construction -> statistical verification -> row/timestep/regime activation -> solver selection -> diffusion/repair actuation -> independent certification`.

Ambiguous endpoint routing chỉ là một trường hợp kiểm tra Action graph, không phải toàn bộ bài toán. Subgroup activation, exception preservation, descendant scope, temporal reliability và solver routing vẫn là các phần chính.

## 2. Motivation

### 2.1 Static global constraints không biểu diễn được heterogeneity

Cùng một relation có thể là exact trong một subgroup, probabilistic ở subgroup khác và không áp dụng cho valid exceptions. Cross-dataset audit đã tìm 788 candidates nhưng chỉ 80 empirical zero-exception; 334 probabilistic và 132 thiếu support. Hard-enforce toàn bộ graph sẽ xóa legitimate modes.

### 2.2 Một constraint không xác định duy nhất hành động

Constraint chỉ nói trạng thái hợp lệ, không nói endpoint nào sai hoặc solver nào tối ưu. Projection, imputation, best-of-N và conditional denoising có frontier khác nhau. Dynamic graph phải biểu diễn allowed actions và uncertainty, không chỉ predicate.

### 2.3 Graph thay đổi theo diffusion state

Early states có controllability cao nhưng semantic readability thấp; late states ngược lại. Membership/applicability có thể flip qua timestep. Một graph cố định từ preview sớm dễ over-enforce; một graph chỉ tạo muộn có thể không còn đủ control.

### 2.4 Graph discovery dễ lặp lại schema

LCATS official phase 1 trên Insurance tạo 14 constraints, trong đó 12 là domain bounds/vocabulary, một order relation và một conditional relation. Một graph lớn không đồng nghĩa có semantic content mới. Cần đo novelty, support, redundancy và heldout stability của từng edge.

## 3. Research question tổng quát

> Một verified dynamic constraint graph có thể điều phối semantic control cho mixed-type tabular diffusion theo subgroup, row, timestep, distribution regime và user objective tốt hơn static priors, per-step projection, best-of-N và post-generation repair về validity, conditional fidelity, exception preservation và cost hay không?

Ba câu hỏi thành phần:

1. Graph nào nên tồn tại và edge nào đủ đáng tin để activate?
2. Khi nào và cho row nào edge được active?
3. Active edge nên điều khiển columns/solver/timestep nào?

### 3.1 Semantic request nằm ở đâu?

Project hiện bao phủ ba settings, nhưng chỉ setting thứ nhất bắt buộc có user request:

1. **Request-specified control**: user mô tả policy/constraint bằng natural language; foundation model hoặc parser compile nó thành typed graph.
2. **Data-discovered dynamic graph**: constraints được discover từ train data/metadata, kiểm chứng trên heldout data và update khi subgroup hoặc distribution regime thay đổi.
3. **Hybrid control**: user chỉ nêu objective hoặc policy scope; graph kết hợp request đó với verified domain/data edges để quyết định applicability, topology, timing và solver.

Các kết quả mới nhất về temporal graph lifecycle, sequential monitoring, solver frontier và intervention stage chủ yếu thuộc setting 2 và 3. Chúng không sử dụng natural-language request ở từng diffusion step. Các controlled policies đóng vai trò executable task specification, không phải bằng chứng rằng hệ thống đã giải quyết open-ended semantic request generation.

## 4. Method architecture đề xuất

### 4.1 Typed graph schema

Mỗi edge nên chứa:

- typed predicate và direction;
- source/target columns;
- applicability/subgroup condition;
- exception clause;
- hard/soft/probabilistic semantics;
- train support và heldout confidence interval;
- temporal stability;
- allowed endpoints và descendants;
- candidate solvers;
- verifier và certification rule;
- provenance: user, metadata, statistics hoặc foundation model.

### 4.2 Four-state activation

Mỗi row-edge pair thuộc một trong bốn trạng thái:

- applicable + violated;
- applicable + satisfied;
- not applicable;
- valid exception.

Chỉ applicable + violated được actuation mặc định. Các trạng thái khác được đo preservation riêng.

### 4.3 Temporal graph

Tại timestep `t`, active set có thể viết:

`C_i(t) = Retrieve(G_spec, optional_request, objective) ∩ Applicable(x_hat_i(t)) \ Exception(x_hat_i(t))`.

Nếu không có user request, `Retrieve` lấy candidate edges từ verified specification/evidence graph theo dataset và objective. Vì vậy dynamic activation vẫn được định nghĩa đầy đủ trong setting data-discovered.

Controller phải có uncertainty threshold, hysteresis hoặc persistence requirement để tránh edge flip do noisy previews. Two-pass lookahead/rollback là một implementation, không phải formulation duy nhất.

### 4.4 Action graph và solver router

Action graph quyết định:

- target-only, sparse descendants hoặc full subgraph;
- projection, repair, imputation, best-of-N hoặc conditional denoising;
- activation time và guidance strength;
- abstain, rollback hoặc residual certification.

Một policy hợp lý có thể chọn projection cho simple convex constraint, best-of-N khi feasible mass cao, imputation cho missing endpoint, và conditional resampling cho coupled/low-feasible-mass constraints.

### 4.5 Vai trò foundation model

Foundation model nên hỗ trợ:

- compile natural-language request thành typed graph;
- đề xuất candidate subgroup/exception/endpoint;
- align semantic relations qua schemas khác nhau;
- chọn semantic route khi executable evidence chưa đủ.

Deterministic type checker, statistical validation và independent verifier vẫn là bắt buộc. Repeated LLM judgement trên mọi row không phải thiết kế mặc định.

## 5. Hypothesis map

### H-G1 — Verified graph construction

LLM + statistics tạo graph có semantic coverage cao hơn metadata/rule mining đơn thuần, trong khi deterministic validation giữ false activation thấp.

Thực nghiệm:

- hold out table/relation families;
- so metadata-only, statistical mining, template parser, LLM-only, verified LLM và oracle;
- đo executable precision/recall, heldout replication, redundancy và sensitive-edge activation.

Điều kiện bác bỏ: LLM chỉ diễn đạt lại domain bounds hoặc không vượt statistical graph.

### H-G2 — Dynamic row/subgroup activation

Four-state activation giảm collateral edits và exception loss so với global/static graph mà không làm giảm validity.

Evidence hiện có: applicability flip giảm mạnh trên Default/Shoppers; residual repair burden giảm 42.6% và 71.2%. Tuy nhiên fidelity chưa luôn hơn post-repair.

Thực nghiệm tiếp:

- balanced violated/satisfied/not-applicable/exception bank;
- same-row/different-request pairs;
- natural subgroup exceptions và temporal distribution shift;
- decision tree/gating network/oracle active set baselines.

### H-G3 — Dynamic topology và descendant scope

Graph-selected sparse descendants tạo frontier tốt hơn target-only và full-record actuation.

Evidence: sparse top-k3 trên Shoppers giảm drift và numeric KS so với full numeric graph nhưng đổi lại target KS; không có universal winner.

Thực nghiệm tiếp:

- learn descendants from train-only conditional dependencies;
- compare causal/statistical/LLM proposed scopes;
- test edge direction, cycle handling và overlapping constraints;
- report Pareto hypervolume thay vì chọn một metric.

### H-G4 — Temporal graph reliability

Persistence/calibrated uncertainty hoặc lookahead/rollback giúp tận dụng early controllability mà không activate sai graph từ noisy previews.

Evidence: late condition yếu; early dynamic preview thay đổi nhiều exception; frozen oracle mạnh. Job `1051548` cho thấy edge-scope precision/recall tăng từ 0.669/0.242 ở đầu trajectory lên 0.922/0.931 ở giữa. Job `1052276` cho thấy mid-denoising có incremental gain cho intervention-conservative và joint-fidelity profiles trên cả Default và Shoppers, nhưng thua mạnh final/resampling khi ưu tiên target marginal fidelity. Vì vậy timing phải phụ thuộc objective, không có lịch can thiệp cố định tốt nhất.

Thực nghiệm tiếp:

- edge membership accuracy/flip rate theo timestep;
- persistence length, confidence threshold và hysteresis ablations;
- one-pass dynamic, two-pass rollback, final-only và oracle schedules;
- cost trên mỗi valid, preserved sample.

### H-G5 — Dynamic solver selection

Không solver nào tối ưu cho mọi graph. Router dựa trên graph features mở rộng Pareto frontier so với luôn dùng projection, repair, best-of-N hoặc conditioning.

Graph features có thể gồm constraint type, arity, feasible mass, violation margin, route ambiguity, subgroup support, preview uncertainty và estimated edit cost.

Policy grid `1051370` và post-hoc router `1051529` đã hoàn tất. Oracle per-policy routing cải thiện normalized rank so với fixed solver từ 0.073 đến 0.134 tùy metric profile. Leave-one-relation-family-out random-forest router `1052275` cải thiện modest ở balanced (`+0.012`) và target-fidelity (`+0.056`), nhưng kém fixed solver ở intervention-conservative (`-0.002`) và joint-fidelity (`-0.003`). Exact oracle choice chỉ 0.17--0.33. Kết quả hiện chứng minh solver heterogeneity và một ít transferable signal, chưa chứng minh learned router đủ mạnh.

Điều kiện bác bỏ: một strong baseline duy nhất chi phối router trên mọi policy sau matched budget.

### H-G6 — Ambiguous action graph

Row-specific endpoint/route selection giảm counterfactual distortion so với fixed repair, kể cả khi mọi method có final validity bằng nhau.

Evidence hiện có mạnh nhưng controlled: context router giảm pair error 31--39%; natural-state và cross-relation routers vẫn tốt hơn fixed routes.

Đây là component hypothesis của H-G5, không nên trở thành toàn bộ framing.

### H-G7 — Cross-table graph transfer

Typed graph encoder hoặc foundation model chuyển được semantic edge roles và solver priors qua schemas/datasets, sau local calibration nhỏ.

Thực nghiệm:

- leave-one-table-out graph compilation;
- shared relation vocabulary và schema descriptions;
- local verifier calibration nhưng không train lại semantic encoder;
- compare lexical retrieval, shallow classifier, foundation model và oracle mapping.

### H-G8 — Graph evolution under policy/distribution shift

Edge confidence và solver choice cần cập nhật khi subgroup rates hoặc temporal regimes thay đổi. Dynamic graph với abstention/change detection an toàn hơn graph discover một lần.

Evidence mới từ leakage-free temporal discovery `1051556`: graph được discover trên 50% đầu, update trên 20% giữa và đánh giá trên 30% cuối. Sau dedup theo ordered column pair, frozen graph giữ late-stability precision 1.000 trên Beijing, 0.685 trên News và 0.919 trên Shoppers. Sequential controller `1052277` giảm unsafe exposure từ 1.00 xuống 0.471 trên News với safe retention 1.00, và xuống 0.478 trên Shoppers với safe retention 0.938. Nó không cải thiện Beijing do transient window shifts không dự đoán được từ window trước. Đây là evidence cho graph lifecycle, đồng thời đặt giới hạn rõ cho reactive monitoring.

Thực nghiệm:

- early-only discovery, late-only evaluation;
- edge confidence decay/change-point detection;
- update graph vs freeze graph;
- measure false activation, exception loss và retraining cost.

## 6. Kết quả thực nghiệm tổng hợp

### 6.1 Những kết quả ủng hộ dynamic graph

- Four-state active set giảm collateral activation và residual repair burden.
- Descendant scope tạo frontier có thứ tự trên hai datasets.
- Lookahead + context anchoring đưa antecedent/subgroup flip và exception edits về gần 0 trong controlled policies.
- Ambiguous action routing giảm pair error rõ rệt dù final validity giống nhau.
- Threshold difficulty quyết định vị trí trên fidelity--intervention frontier; không có universal solver.

### 6.2 Những phản chứng quan trọng

- Best-of-8 đánh bại mid-denoising conditioning trên phần lớn metrics của Shoppers single-threshold policies.
- Direct clamp thắng trên conjunction chỉ gồm lower bounds.
- LLM direct judgement kém compile-once DSL + deterministic executor cả accuracy và token cost.
- LLM subgroup selector chưa vượt safety-filtered statistical selection.
- Near-hard natural constraints có violation chỉ 1--2%, không đủ room cho headline effect.

### 6.3 Common CuTS programs

TabDiff common-task job `1051083` đã hoàn tất.

- `implication2/4`: unconditional đã khoảng 0.994 satisfaction; benchmark bão hòa.
- `implication5`: unconditional 0.945; final repair và best-of-8 đạt 1.0 với fidelity tradeoff nhỏ.
- `sex=Female`: unconditional 0.328; final repair đạt 1.0 nhưng categorical TV tăng từ 0.0155 lên 0.0890; best-of-8 TV 0.2002.
- `35<age<55`: unconditional 0.437; final repair đạt 1.0 nhưng numeric KS tăng từ 0.0249 lên 0.0942; best-of-8 tăng lên 0.1146.

Kết luận: public global row constraints tạo distribution shift mạnh nhưng không kiểm tra dynamic applicability. Chúng hữu ích làm common baseline và solver-selection sanity test, không đủ thay benchmark dynamic graph.

### 6.4 LCATS official feasibility

LCATS phase 1 đã chạy official code và phát hiện 14 Insurance constraints. Graph audit `1051372` cho thấy 12 domain bounds, chỉ 2 relational/conditional edges; cả 14 có zero train và zero test violations. Điều này xác nhận graph chủ yếu mô tả support/schema invariants và benchmark tự nhiên bị bão hòa. Job `1052248` đã hoàn tất official phases 2--5 sau hai compatibility fixes không đổi thuật toán. TabSyn được train/sampled, uncertainty trajectory được thu, LCATS diagnostics/reasoning chạy xong và phase 5 tạo ba aligned candidates. Đây mới là xác nhận baseline chạy end-to-end; fidelity/utility và independent violation audit vẫn phải được tổng hợp trước khi dùng trong bảng so sánh.

### 6.5 Multi-solver frontier và khả năng routing

Grid `1051370` so sánh post-repair, empirical repair, TabDiff imputation, best-of-8 và nhiều phạm vi conditional denoising trên 30 policies của Default/Shoppers. Không có universal winner:

- Default: best-of-8 hoặc numeric-graph conditioning thường nằm trên frontier cân bằng; empirical marginal repair giảm một số target-distance nhưng gây displacement lớn.
- Shoppers: best-of-8 có drift thấp hơn; conditional variants giữ một số joint/categorical metrics tốt hơn; empirical repair có target marginal tốt nhưng drift rất lớn.
- Mỗi policy trung bình có 11,27/12 phương pháp không bị dominate hoàn toàn, chứng tỏ metric conflict mạnh và cũng cảnh báo benchmark còn noisy.

Router `1051529` cho oracle gain 0.073--0.134 normalized rank, nhưng shallow graph-feature router không tái tạo được oracle. H-G5 vì vậy còn mở; bước tiếp theo phải học cost/performance predictor từ nhiều task hơn, không được báo oracle routing như method result.

### 6.6 Leakage-free graph evolution

Job `1051556` loại leakage của temporal stress cũ bằng cách discover predicate/edge chỉ trên early 50%, cho middle 20% làm update và giữ late 30% để đánh giá.

- Beijing: 22 deduplicated ordered-column edges, 22/22 late-stable.
- News: 587 edges, chỉ 402 late-stable; frozen precision 0.685 với 185 unsafe activations. Wilson update chỉ tăng precision lên 0.700 và còn 172 unsafe activations.
- Shoppers: 37 edges, 34 late-stable; point update giữ 30 edges và đạt precision 1.000, đổi lại recall 0.882.

Con số trước dedup lớn hơn nhưng không dùng làm headline vì nhiều threshold predicates lặp ordered column pairs. Kết quả support graph lifecycle/change detection, đồng thời phản bác update bằng confidence threshold như một method đủ mạnh.

### 6.7 Persistence gate trên diffusion trajectory

Job `1051548` dùng 2.048 synthetic rows, 50 steps và 12 subgroup edges trên Default. Edge-scope readability tăng từ precision/recall 0.669/0.242 ở `t>0.75`, lên 0.922/0.931 ở `0.25<t<=0.50` và gần 1 ở `t<=0.10`. Yêu cầu edge active liên tiếp ba previews tăng early precision từ 0.669 lên 0.703 và giảm false activation 0.0155 xuống 0.0123, nhưng recall giảm từ 0.242 xuống 0.204. Ở mid trajectory, precision tăng 0.922 lên 0.939 trong khi recall giảm 0.931 xuống 0.902.

Kết luận: persistence là một precision--recall knob hợp lệ nhưng gain nhỏ; chưa đủ làm contribution riêng, và chưa có downstream actuation result. Bằng chứng mạnh hơn vẫn là timing-dependent readability và nhu cầu calibrated activation.

### 6.8 Sequential graph monitoring dưới distribution shift

Job `1052277` chia phần sau discovery thành năm monitoring windows; quyết định ở mỗi window chỉ dùng dữ liệu từ các window trước.

- News: last-window controller giảm unsafe exposure `1.000 → 0.471`, unsafe active windows `418 → 197`, weighted violation `0.0148 → 0.0094`, trong khi safe retention vẫn `1.000`.
- Shoppers: unsafe exposure `1.000 → 0.478`, unsafe windows `23 → 11`, weighted violation `0.0171 → 0.0072`; safe retention giảm nhẹ xuống `0.938`.
- Beijing: reactive controllers không giảm 13 transient unsafe windows và còn làm giảm safe retention. Các shifts ngắn hoặc đảo chiều không thể được dự đoán chỉ từ window trước.

Kết quả này ủng hộ một method contribution về graph lifecycle/controller. Tuy nhiên nó hiện là graph audit trên real aggregates, chưa phải generation experiment; bước kế tiếp phải nối active-edge update với sampler hoặc abstention policy và đo fidelity/utility.

### 6.9 Giá trị tăng thêm của mid-denoising

Job `1052276` lọc các output có active satisfaction ít nhất 0.999 và không thay đổi exception, rồi so best mid-denoising, best final repair và best conditional resampling trên cùng 30 policies.

- Balanced: mid-denoising thắng best non-mid ở 60% policies; mean normalized-rank gain `+0.012`.
- Intervention-conservative: thắng 76.7%; gain `+0.037`.
- Joint-fidelity: thắng 66.7%; gain `+0.055`.
- Target-fidelity: chỉ thắng 26.7%; gain `-0.100`, tức final repair/resampling tốt hơn rõ.

Theo dataset, joint-fidelity gain gần như giống nhau trên Default (`+0.054`) và Shoppers (`+0.056`). Intervention-conservative gain là `+0.052` và `+0.022`. Target-fidelity đều âm (`-0.128`, `-0.073`). Đây là bằng chứng định lượng đầu tiên rằng can thiệp giữa trajectory có ích cho preservation/joint structure, nhưng không phải cho mọi objective.

## 7. Baseline matrix cho project rộng

### Graph construction

- schema/domain constraints;
- statistical/FDR graph;
- decision tree hoặc RuleFit subgroup graph;
- LLM-only graph;
- verified LLM graph;
- LCATS official discovery;
- oracle graph.

### Activation

- global static graph;
- request-static graph;
- subgroup-static graph;
- one-pass dynamic preview;
- four-state frozen lookahead;
- persistence/calibrated dynamic graph;
- oracle active graph.

### Topology

- target-only;
- full record;
- full numeric graph;
- train-ranked sparse descendants;
- learned/LLM descendants;
- oracle descendants.

### Solver

- no action;
- direct projection;
- deterministic and empirical post-repair;
- official TabDiff imputation;
- rejection/best-of-N;
- CTDF-style and Harpoon-style proxies, labeled correctly;
- conditional denoising/lookahead;
- official CuTS and LCATS where common task permits.

## 8. Experiment matrix đang triển khai

| Trục | Datasets | Trạng thái |
| --- | --- | --- |
| Four-state activation | Default, Shoppers | Đã chạy |
| Sparse/full topology | Default, Shoppers | Đã chạy |
| Timing/lookahead | Default, Shoppers | Đã chạy một phần |
| Solver grid toàn policy | Default, Shoppers | `1051370` hoàn tất |
| Solver router từ graph features | Default, Shoppers | `1051529`, `1052275`; transferable gain còn nhỏ và phụ thuộc objective |
| Ambiguous endpoint | Shoppers | Đã chạy nhiều router |
| Public CuTS programs | Adult | Job `1051083` hoàn tất |
| Official LCATS discovery | Insurance | Phase 1 hoàn tất |
| LCATS graph redundancy | Insurance | Hoàn tất: 12/14 domain bounds; 14/14 zero heldout violations |
| Official LCATS generation/repair | Insurance | `1052248` hoàn tất end-to-end; metrics audit chưa tổng hợp |
| Cross-table graph transfer | 6--7 tables | Candidate audit có; generation test chưa đủ |
| Leakage-free graph evolution | Beijing, News, Shoppers | `1051556` hoàn tất |
| Persistence gate theo timestep | Default | `1051548` hoàn tất; gain nhỏ, chưa có actuation |
| Sequential graph monitoring | Beijing, News, Shoppers | `1052277` hoàn tất; tốt trên News/Shoppers, âm trên transient Beijing |
| Mid-denoising so với best non-mid | Default, Shoppers | `1052276` hoàn tất; tốt cho joint/preservation, kém target marginal |

## 9. Paper framing đề xuất

Tên tạm:

> Verified Dynamic Constraint Graphs for Semantic Control of Tabular Diffusion

Claim mục tiêu:

> A typed, verified graph that dynamically selects applicable constraints, allowed topology and actuation solver per row and diffusion state provides a safer and broader validity--fidelity--cost frontier than static graph guidance or a fixed post-processing operator.

Ba contribution chính:

1. Dynamic graph formulation bao phủ applicability, exceptions, topology, timing và solver.
2. Graph-aware controller với independent verification/certification.
3. Benchmark/evaluation protocol có four-state cases, policy shift, ambiguous actions và matched strong baselines.

LLM là graph compiler/semantic transfer component, không phải toàn bộ novelty.

## 10. Tiêu chí quyết định

Hướng đủ mạnh để tiếp tục thành paper nếu:

- solver router hoặc active graph mở rộng Pareto frontier trên ít nhất ba datasets;
- gain còn tồn tại trước best-of-N, strong repair và official prior-work baselines;
- graph bảo toàn valid exceptions và không né antecedent;
- ít nhất một cross-table hoặc held-out-relation result chứng minh graph không chỉ memorize table;
- intermediate-state component có incremental benefit rõ so với final-state controller.

Hiện oracle routing cho thấy room; cross-relation router chỉ có gain nhỏ ở hai trong bốn objectives, nên chưa được dùng làm headline method result. Bằng chứng mạnh hơn đang chuyển sang active graph lifecycle và objective-dependent intervention timing. Nếu solver routing tiếp tục không hơn một operator cố định, contribution nên đặt ở active graph + lifecycle + benchmark. Nếu LLM không hơn parser/statistics, giữ LLM như optional compiler/interface. Nếu cross-table transfer thất bại, không dùng foundation-model claim nhưng dynamic graph vẫn có thể là method contribution trong từng table.

## 11. Đánh giá trạng thái research project

### 11.1 Những claim hiện đã có evidence

1. Constraint applicability và graph topology thay đổi theo row, subgroup và diffusion timestep; preview sớm chưa đủ tin cậy.
2. Static empirical graph có thể drift mạnh theo thời gian; sequential monitoring giảm unsafe exposure trên hai trong ba datasets.
3. Mid-denoising không universally tốt hơn, nhưng có incremental value nhất quán cho joint fidelity và intervention preservation trên hai generation datasets.
4. Không có universal solver; objective quyết định repair, resampling hay mid-denoising nằm trên frontier nào.
5. Foundation model direct judgement và semantic selection hiện chưa vượt typed execution hoặc conservative statistics.

### 11.2 Những claim chưa đủ evidence

- Chưa có learned solver router mạnh: gain cross-relation chỉ dương ở hai trong bốn objectives.
- Chưa nối temporal graph monitoring với sampler trên một dataset có real distribution shift và checkpoint phù hợp.
- Chưa có leave-one-table-out foundation-model controller hoặc shared condition encoder.
- Official LCATS generation/repair chưa hoàn tất; CTDF/Harpoon vẫn chủ yếu là proxy implementations trong repo này.
- Multi-solver grid hiện là one-seed hypothesis screen; chưa đủ cho state-of-the-art claim.

Vì vậy project đã có một research question và một tập phản chứng đủ nghiêm túc, nhưng **chưa phải state of the art result**. Framing phù hợp nhất hiện tại là verified dynamic graph/controller, không phải “LLM feedback beats all baselines”.

### 11.3 Thứ tự thực nghiệm tiếp theo

1. **Generation under graph shift**: tạo stream/batches trên Default và Shoppers, freeze graph ở batch đầu, sau đó so frozen, last-window, EWMA, abstain và oracle activation trong sampler. Đo validity, exception preservation, fidelity và adaptation cost.
2. **Objective-conditioned controller**: input gồm graph features, preview uncertainty và user metric profile; output là stage/scope/solver. Đánh giá leave-one-relation-family-out và leave-one-dataset-out, không dùng oracle policy labels ở test.
3. **Thêm generation dataset thứ ba**: ưu tiên một table có timestamp/regime và mixed types; nếu phải train checkpoint mới, chỉ train một backbone trước khi chạy solver grid.
4. **Foundation-model transfer**: compile unseen schema/request thành typed graph, sau đó local verification. Baselines bắt buộc gồm template parser, embedding retrieval, statistical graph và oracle DSL.
5. **Official baselines**: hoàn tất LCATS; chỉ thêm CTDF/Harpoon official nếu có common task và environment chạy được, tránh gọi proxy là reproduction.

Một paper ICDE khả thi cần ít nhất: ba generation datasets, một official prior-work baseline end-to-end, một controller result vượt fixed strong baseline trên held-out relation/table, và ablation chứng minh intermediate diffusion state cung cấp thông tin hoặc control không thể thay bằng final-only processing cùng budget.
