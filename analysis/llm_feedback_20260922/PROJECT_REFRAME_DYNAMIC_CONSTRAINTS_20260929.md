# Project reframe: dynamic semantic constraints for tabular diffusion

## 1. Vì sao rare conditional generation không nên là headline

Nhận xét “hơi niche” là hợp lý. Rare cohort generation có ba vấn đề khi đứng một mình:

1. Motivation phụ thuộc use case: không phải hệ thống synthetic data nào cũng thường xuyên yêu cầu cohort rất hiếm.
2. Reviewer có thể xem đây chủ yếu là sample-efficiency problem; rejection sampling, conditional fine-tuning hoặc oversampling là baseline tự nhiên.
3. Khi support thấp, khó phân biệt khả năng control thật với extrapolation/hallucination. Một kết quả validity cao chưa chắc có semantic fidelity.

Các kết quả hiện có vẫn có giá trị nhưng nên chuyển vai trò:

- rare conditions là stress test cho sample cost và mode coverage;
- không dùng “sinh nhóm hiếm” làm contribution chính;
- rejection frontier và CuTS adaptation cost trở thành secondary evaluation.

## 2. Bài toán chính được đề xuất

> **Runtime semantic policy enforcement for synthetic tabular diffusion:** sau khi generator đã được train, người dùng đưa một policy/constraint mới. Policy có thể chỉ áp dụng cho một số hàng, có exception, liên quan nhiều cột và có nhiều cách sửa hợp lệ. Hệ thống phải tạo dữ liệu tuân thủ policy mà không làm sai các hàng ngoài phạm vi, xóa exception hợp lệ hoặc phá joint distribution.

Bài toán này rộng hơn rare generation vì xuất hiện trong:

- test-data generation với business rules thay đổi theo release/customer;
- simulation và what-if analysis với policy mới;
- data sharing khi integrity requirements khác nhau theo downstream consumer;
- schema evolution và cross-table deployment;
- synthetic benchmark construction cần kiểm soát semantics mà không retrain generator cho từng request.

Constraint không cần hiếm. Difficulty đến từ applicability, exception, mixed-type coupling, conflict và repair ambiguity.

## 3. Gap cụ thể

Existing methods giải quyết từng phần nhưng chưa khép kín:

- CTDF/projection: mạnh với hard executable constraints, nhưng constraint và feasibility operator phần lớn cố định;
- Harpoon: loss cố định, gradient thay đổi theo state nhưng không nhất thiết thay đổi active semantic structure;
- SCTab-Diff: prior được condition vào denoising nhưng prior không phải request-specific dynamic active graph;
- LCATS/Zhao et al.: discovery, validation và post-repair mạnh, nhưng repair diễn ra sau generation và cần chọn cách sửa;
- PromptLoop: feedback từ intermediate estimate đã có ở image diffusion, nhưng không xử lý typed predicates, applicability, exceptions và mixed-type branch ambiguity.

Gap cần bảo vệ:

> Chọn active constraint subgraph và actuation route theo từng row/timestep, có calibrated abstention và independent verification, cho runtime policies chưa thấy trong mixed-type tables.

## 4. Method target

### 4.1 Prior graph

LLM compile schema, metadata và user policy thành typed graph. Nodes biểu diễn predicate, column/value, applicability, exception và action. Edges gồm `requires`, `implies`, `conflicts`, `exception_to`, `depends_on`, `actuates`.

LLM output phải qua schema/type checker, counterexample tests và train-only validation. LLM không trực tiếp thay executor cho hard rules.

### 4.2 Dynamic active graph

Tại một số reverse steps:

`G_(i,t) = Select(G_prior, request_i, x_hat_0(i,t), trajectory_i, uncertainty_i)`.

Activation cần tách:

`readability × applicability × violation_probability × (1 - exception_probability)`.

Hành động gồm activate selected columns, adjust strength, defer, abstain, rollback hoặc call LLM khi có semantic conflict.

### 4.3 Independent verification

Verifier cuối không được chính là LLM/controller dùng để guidance. Residual certifier được phép sửa phần còn lại và phải báo edit burden, distortion và exception loss.

## 5. Evidence hiện có

### 5.1 Vì sao active set cần thay đổi theo hàng

Trong controlled policies trên Default/Shoppers, global condition và preview-selective giảm violation giống nhau, nhưng global condition gây drift lớn trên inactive và exception rows. Preview-selective đưa hai drift này về zero. Shuffled active set kém rõ rệt. Điều này chứng minh đúng active rows quan trọng; chưa chứng minh LLM cần thiết.

Post-repair vẫn rất mạnh trên one-target threshold rules: đạt zero violation với distortion thấp. Vì vậy những rule này không được dùng làm headline benchmark.

### 5.2 Dynamic graph so với fixed graph

Full three-seed grid `1048642` đã hoàn tất:

| Dataset/catalog | Mean Δ success | Mean Δ calls/valid | Mean Δ KS | Mean Δ TV | Strict dominance configs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Default mixed | +0.0749 | -2.404 | +0.0129 | +0.0010 | 5/72 |
| Shoppers mixed | +0.0363 | -2.137 | -0.0102 | +0.0011 | 15/72 |
| Default numeric | +0.0018 | -0.006 | -0.0011 | -0.0015 | 26/108, effect rất nhỏ |
| Shoppers numeric | +0.0183 | -0.087 | -0.0003 | -0.0008 | 45/108 |

Diễn giải:

- topology adaptation có signal rõ hơn trên mixed conditions;
- không có bằng chứng cho claim “dynamic graph luôn tốt hơn”;
- Default mixed đổi success lấy numeric fidelity ở nhiều configuration;
- Shoppers mixed có kết quả tốt hơn, gồm 15 strict-dominance configurations;
- numeric-only không phải target setting thuyết phục.

Đây là evidence để tập trung mixed/compositional semantics, không phải rare support.

### 5.3 Vai trò của LLM

Direct per-record judgement hiện kém compile-once:

- GPT-4.1-mini direct accuracy 0.864, khoảng 37.5k tokens;
- GPT-5.4-mini direct accuracy 0.650, khoảng 37.7k tokens;
- LLM → typed DSL → deterministic executor đạt 1.0 execution equivalence trên probe, khoảng 1.85k tokens.

Vì vậy kiến trúc mặc định là LLM-compiled prior graph + deterministic dynamic executor. Event-triggered LLM update chỉ được thêm cho ambiguous/non-executable cases.

## 6. Hypotheses ưu tiên

### H2.1 — Composition, không phải rarity, tạo lợi ích

Dynamic graph cải thiện success–fidelity–cost frontier so với fixed dependency graph trên held-out categorical/mixed compositions và ordinary held-out thresholds.

Falsification: gain chỉ xuất hiện ở quantile rarity sweep hoặc biến mất trên ordinary compositions.

### H2.2 — Guidance trước certifier phải giảm repair burden

Dynamic guidance + residual certifier phải giảm edit rate/numeric distortion hoặc cải thiện fidelity so với repair-only, khi cả hai đạt final validity 1.

Falsification: repair-only bằng hoặc tốt hơn trên toàn bộ metrics. Khi đó mid-denoising contribution không đủ.

### H2.3 — Oracle four-state graph là upper bound quyết định

Oracle graph biết `violated`, `satisfied`, `not applicable`, `valid exception` phải tốt hơn always-on graph và post-repair về exception/inactive preservation.

Falsification: oracle không mở rộng Pareto frontier. Nếu vậy không nên đầu tư LLM controller.

### H2.4 — LLM chỉ có giá trị trên semantic ambiguity

LLM graph update chỉ được đánh giá ở policies có external semantics, conflict hoặc multiple valid repairs. Hard executable rules dùng deterministic baseline.

Falsification: LLM không vượt typed parser/verifier hoặc không thu hồi đáng kể oracle gain trên ambiguous cases.

## 7. Focused experiments đã submit

Không thêm seed ở vòng screening này.

- `1049835[0-7]`: một seed, Default/Shoppers, fixed versus dynamic graph, constant/midpulse, catalog `mixed` gồm held-out thresholds và held-out compositions;
- `1049836–1049837`: unconditional generation + identical certifier, làm repair-only baseline;
- `1049838`: paired claim audit, gồm dynamic-versus-fixed và dynamic+certifier-versus-repair-only.

Output dự kiến: `outputs/dynamic_composition_claim_1049838/RESULTS.md`.

Resource cleanup: giữ một Diabetes backbone seed `1048479`; hủy replication `1048480` và các adapter/aggregate jobs `1048596–1048601`, `1048605`, `1048609–1048614`, `1048618`. Không xóa partial checkpoints hoặc embedding outputs. Việc này ưu tiên GPU cho hypothesis H2.1–H2.2 thay vì tiếp tục broad foundation ablation trước khi core claim được xác nhận.

### Kết quả H2.1–H2.2

Jobs `1049835–1049838` đã hoàn tất.

- Default non-rarity catalog gần như neutral; gain về success thường dưới 0.003.
- Shoppers categorical compositions có tín hiệu tốt: success `+0.014–0.016`, giảm calls/valid, đồng thời KS và TV tốt hơn.
- Shoppers mixed compositions tăng success `+0.006–0.009` nhưng KS xấu hơn khoảng `0.012–0.015`.
- Held-out single thresholds gần như không hưởng lợi.

Vì vậy H2.1 chỉ được ủng hộ hẹp cho categorical composition trên Shoppers; không đủ để claim dynamic topology nói chung.

Dynamic guidance + residual certifier so với repair-only, khi cả hai đạt final validity 1:

- mean edit rate giảm `0.3400`;
- numeric distortion giảm `0.1865`;
- numeric KS giảm `0.0294`;
- categorical TV giảm `0.0125`.

Tuy nhiên fixed graph + certifier đạt gần tương đương. Dynamic-versus-fixed sau certification chỉ thay đổi edit rate `-0.0029`, numeric distortion `+0.0001`, KS `-0.0003`, TV `-0.0007`. Do đó phần được support là **guidance-before-certification**, không phải topology heuristic hiện tại.

Topology heuristic được dừng tại đây. Jobs `1049886–1049888` kiểm tra oracle-like four-state active graph trên một seed: chỉ actuate rows `applicable + violated`, giữ nguyên `satisfied`, `not applicable`, `valid exception`, rồi dùng cùng residual certifier. Đây là upper-bound screen trước mọi LLM controller.

### Kết quả H2.3: four-state active set

So sánh dưới đây matched theo cùng seed, guidance strength và policy; không so method một seed với baseline ba seed.

| Dataset | Method | Violation | Applicability flip | KS | TV | Edit rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Default | applicability-only guidance | 0.1299 | 0.0171 | 0.0110 | 0.0055 | 0 |
| Default | violated-row guidance | 0.1299 | 0.0029 | 0.0066 | 0.0018 | 0 |
| Default | violated-row guidance + certifier | 0 | 0.0029 | 0.0105 | 0.0018 | 0.0549 |
| Default | post-repair only | 0 | 0 | 0.0068 | 0 | 0.0955 |
| Shoppers | applicability-only guidance | 0.0786 | 0.0182 | 0.0174 | 0.0073 | 0 |
| Shoppers | violated-row guidance | 0.0786 | 0.0116 | 0.0125 | 0.0050 | 0 |
| Shoppers | violated-row guidance + certifier | 0 | 0.0116 | 0.0146 | 0.0050 | 0.0223 |
| Shoppers | post-repair only | 0 | 0 | 0.0077 | 0 | 0.0773 |

Các kết luận được support:

- tách `satisfied` khỏi `violated` không đổi violation reduction trong screen này, nhưng giảm applicability flip `0.0142` trên Default và `0.0067` trên Shoppers; KS/TV cũng giảm;
- hybrid guidance + certifier giảm edit rate `42.6%` trên Default và `71.2%` trên Shoppers so với post-repair cùng seed;
- active-state selection vì vậy có giá trị cho intervention sparsity và residual repair burden.

Các kết luận chưa được support:

- hybrid chưa vượt post-repair về distribution fidelity: KS cao hơn `0.0037`/`0.0069` và TV cao hơn `0.0018`/`0.0050` trên Default/Shoppers;
- benchmark one-target threshold quá dễ cho post-repair vì chỉ cần clamp một consequent;
- kết quả không chứng minh dynamic topology hoặc LLM controller; active state ở đây là oracle, deterministic và chỉ chạy một seed.

Do đó H2.3 được ủng hộ một phần: oracle four-state selection giảm collateral intervention và repair burden, nhưng chưa mở rộng toàn bộ Pareto frontier trước repair-only. Thí nghiệm quyết định tiếp theo phải dùng constraint có nhiều repair routes hợp lệ hoặc non-local coupling; nếu vẫn chỉ có một target dễ clamp thì không tạo được setting công bằng cho mid-denoising guidance.

## 8. Claim ladder

Claim gần nhất đã có evidence:

> State-aware guidance trước independent certification làm giảm collateral intervention và residual repair burden trong runtime policy enforcement.

Claim này chưa phải kết quả SOTA và chưa đủ cho paper chính vì post-repair còn tốt hơn về fidelity trên constraint đơn giản.

Claim mục tiêu cần kiểm tra tiếp:

> Với mixed-type constraints có nhiều repair routes, trajectory-aware routing chọn cách tác động ít phá joint distribution hơn deterministic post-repair, trong khi independent certifier bảo đảm validity cuối.

Chỉ nâng lên semantic/LLM claim nếu H2.3 và H2.4 thành công:

> A verified LLM-compiled dynamic semantic graph selectively enforces unseen runtime policies while preserving non-applicable rows and valid exceptions.

Không claim:

- first constraint guidance in tabular diffusion;
- first LLM prior in denoising;
- dynamic graph universally improves all constraints;
- foundation/cross-table generalization trước khi có shared model và held-out table evaluation.

## 9. Quyết định sau H2.1–H2.3

- Dừng dynamic-topology heuristic hiện tại; nó không hơn fixed graph sau certification.
- Giữ four-state active set như cơ chế cốt lõi để tránh tác động lên `satisfied`, `not applicable` và `valid exception`.
- Không dùng rare generation hoặc one-target thresholds làm headline; chúng chỉ là stress test và sanity check.
- Thí nghiệm ưu tiên tiếp theo là ambiguous repair: cùng một violation có ít nhất hai cách sửa hợp lệ, đánh giá branch recovery, joint fidelity, collateral edits và repair cost.
- Chỉ gọi LLM cho compile-once policy graph và ambiguity resolution. Hard executable rule tiếp tục dùng parser/verifier deterministic.
- Chỉ nâng lên claim foundation/cross-dataset sau khi method thắng strong repair/projection baseline trên held-out policies và ít nhất ba bảng dữ liệu.

### Ambiguous-repair audit đang chạy

Job `1049914` kiểm tra ba implication count–duration trên Shoppers bằng controlled corruption. Mọi case quan sát đều có dạng `count=0, duration>0`, nhưng một nửa do duration bị corrupt và một nửa do count bị corrupt. Cả hai fixed repair routes đều đạt validity 1; metric quyết định là có sửa đúng cột và khôi phục gần clean counterfactual hay không.

Baselines gồm consequent-only repair, antecedent-only repair, minimum normalized edit, context router và oracle route. Router không thấy cột count đã bị corrupt, không dùng target `Revenue`, và duration ở hai class dùng cùng multiset để tránh magnitude leakage.

Smoke test trên hai relation đủ support cho thấy context router đạt route accuracy `0.785`, so với `0.5` của mọi fixed route, và giảm normalized pair error từ `1.350` của consequent-only xuống `0.914`; oracle là `0.490`. Đây chỉ là pipeline evidence, chưa phải kết quả chính. Official test chỉ có sáu ProductRelated zero-pairs nên relation này sẽ bị loại và ghi rõ insufficient support thay vì được dùng để tạo claim.

## 10. Subgroup-conditioned dynamic graph

Kết quả mới thu hẹp dynamic graph thành edge applicability theo subgroup thay vì heuristic topology. Corrected seven-dataset audit dùng train-only discovery, subgroup-base-rate interaction control và heldout replication. Default có `150/151` non-sensitive candidates replicate; Diabetes `14/20`; Beijing `3/3`; News `56/57`; Shoppers `0/2`. Adult candidates đều sensitive và chỉ dùng để audit.

LLM selector thuần không an toàn. Sau deterministic safety/direction verification, GPT-4.1-mini đạt precision `0.930`, recall `0.985`, F1 `0.957`; conservative statistics đạt `0.955/0.627/0.757`. Tuy nhiên baseline mạnh hơn là accept mọi FDR-significant edge sau cùng safety filter, đạt `0.931/1.000/0.964` và vẫn hơn GPT-4.1. GPT-5.4 mất nhiều recall hơn. Vì vậy LLM selection hiện không thêm giá trị ngoài statistical graph + deterministic safety; LLM chỉ nên tiếp tục ở unseen natural-language subgroup proposal hoặc external semantic ambiguity.

Formalization, novelty boundary, baselines và falsification tests nằm trong `SUBGROUP_DYNAMIC_GRAPH_PROPOSAL_20260929.md`.

### Online subgroup actuation

Job `1050330` là phép thử online đầu tiên cho subgroup graph trên cùng TabDiff checkpoint và paired initial noise. Kết quả không tạo claim dương: các quan hệ payment--bill tự nhiên đã đạt `98.99%` ngay ở unconditional, nên static graph, dynamic graph và oracle graph đều không cải thiện. Subgroup-aware certifier đạt `100%` với khoảng `1%` active rows cần sửa. Global repair cũng đạt `100%` nhưng sửa trung bình `88.6%` valid-exception rows. Vì vậy benchmark near-hard hiện chỉ đủ chứng minh nguy cơ over-enforcement, chưa đủ chứng minh lợi ích của mid-denoising.

Hướng method tiếp theo là tách graph khỏi condition model. Job `1050333` huấn luyện một structured adapter chỉ trên atomic payment-threshold conditions. Job phụ thuộc `1050334` dùng subgroup preview để bật condition theo từng row ở giữa reverse diffusion; subgroup composition hoàn toàn chưa xuất hiện trong training. So sánh gồm static graph, dynamic graph, oracle frozen graph, subgroup-aware repair và global repair trên cả quan hệ tự nhiên `>0` lẫn policy threshold `>=1000`. Đây là phép thử trực tiếp cho hypothesis: một condition foundation layer có thể tái sử dụng qua nhiều graph/policy thay vì train generator riêng cho từng subgroup.

Temporal stress `1050336` cho thấy `3/3` cạnh Beijing giữ được interaction, nhưng News chỉ giữ `28/39` cạnh evaluable (`71.8%`); Shoppers không đủ support. Candidate đã được discover trước temporal split nên đây chỉ là falsification screen. Kết quả đủ để bác bỏ cách diễn đạt rằng random-heldout replication đồng nghĩa với graph ổn định. Benchmark chính cần early-only discovery, late-only evaluation và cơ chế abstain khi edge weight không ổn định.

Jobs `1050334` và `1050337` cho thấy temporal alignment gap. Atomic condition adapter gần như không còn tác dụng khi chỉ bật sau `t<=0.5`, kể cả khi dùng oracle frozen subgroup. Khi oracle scope được biết từ đầu và condition chạy toàn trajectory, controlled-policy satisfaction tăng từ `0.8195` lên `0.9407`, target KS giảm từ `0.2084` xuống `0.0896`, không làm đổi exception. Nhưng dynamic preview gate chạy từ đầu chỉ đạt `0.8918` và làm thay đổi `48.4%` exception vì early scope chưa đáng tin. Đây là gap method rõ nhất hiện tại: controllability nằm sớm, semantic readability nằm muộn. Hai hướng hợp lý là lookahead/rollback để ước lượng scope trước khi resample, hoặc huấn luyện condition adapter với random activation time để đáp ứng tốt ở late steps. Job `1050338` đang kiểm tra early condition + residual certifier với final validity bằng nhau.

Job `1050338` đã xác nhận upper bound có ích trên controlled policy: khi cả hai đạt validity `1.0`, frozen-lookahead condition + certifier có target KS `0.0896`, tốt hơn subgroup repair `0.1999`, và không sửa target của exception rows. Tuy vậy antecedent flip còn `6.35%`; trên near-hard empirical rules, condition lại làm KS xấu hơn. Job `1050339` đang kiểm tra two-pass lookahead có context anchoring: pass đầu xác định active graph, pass hai rollback cùng noise, giữ trajectory của antecedent/subgroup và chỉ resample descendant được phép. Đây là method candidate cụ thể nhất hiện tại.

Jobs `1050339` và `1050675` xác nhận two-pass context anchoring đưa antecedent/subgroup flip và exception edits về `0`. Four-state graph chỉ resample `applicable + violated` rows, nhờ đó giảm collateral edits mạnh so với condition toàn subgroup. Strong empirical repair trong `1050673` thắng về target marginal, nhưng four-state lookahead tốt hơn về toàn bộ numeric cohort và target-context correlation; hai phương pháp nằm trên một fidelity--intervention frontier.

Ba sampling seed (`1050676`, `1050680`, aggregate `1050685`) xác nhận full-subgraph lookahead dùng displacement thấp hơn empirical repair `0.151` SD và giữ correlation tốt hơn `0.0151`, nhưng categorical TV xấu hơn `0.0097`. Job `1050684` cho thấy actuation scope tạo tradeoff có thứ tự: target-only bảo toàn categorical tốt nhất nhưng control yếu; BILL/PAY numeric subgraph ở giữa; full-record condition cho numeric/joint fidelity tốt nhất nhưng categorical drift lớn nhất. Vì vậy graph method nên dự đoán cả active edge và allowed descendant set. `1050686–1050687` đang xác nhận numeric-subgraph trên hai seed còn lại.

Aggregate `1050687` xác nhận numeric-subgraph trên ba seed: validity `1.0`, active drift `0.0136` SD, target KS `0.1253`, cohort numeric KS `0.0748`, categorical TV `0.0351`, correlation error `0.0595`. Nó tốt hơn clamp repair trên các fidelity metrics chính, và dùng thay đổi nhỏ hơn empirical repair rất nhiều, nhưng empirical repair vẫn tốt hơn rõ về target marginal. Do đó contribution hiện là controllable Pareto frontier thông qua graph-selected actuation scope. Job `1050689` đang chuyển cơ chế sang Shoppers với adapter đã train độc lập và ba unseen subgroup compositions.

Shoppers pilot `1050689` có unconditional satisfaction `0.368`, khó hơn Default. Numeric-subgraph + certifier đạt target KS `0.271`, cohort numeric KS `0.161`, categorical TV `0.106`, correlation error `0.201`, active drift `0.867`. Empirical repair tương ứng `0.199/0.176/0.120/0.228/1.117`; clamp repair `0.646/0.221/0.120/0.177/0.277`. Vì vậy cùng một non-dominated pattern xuất hiện ở dataset thứ hai.

Aggregate ba seed `1050691` xác nhận pattern này. Numeric-subgraph so với empirical repair giảm active drift `0.288` SD, cohort numeric KS `0.0212` và categorical TV `0.0097`, nhưng target KS tăng `0.0730` và Wasserstein tăng `0.2632`. So với clamp repair, numeric-subgraph giảm target KS `0.3658`, Wasserstein `0.3882` và cohort numeric KS `0.0651`, đổi lại active drift cao hơn `0.5618` SD. Kết quả nhất quán giữa Default và Shoppers: không có một method chi phối hoàn toàn; descendant scope quyết định tradeoff giữa mức can thiệp, target marginal và joint fidelity.

### Đối chứng diffusion-imputation

Jobs `1050693–1050694` thêm baseline mạnh hơn: giữ nguyên context, chỉ mask target, dùng chính TabDiff để impute ba vòng rồi certify. Xác nhận ba seed trong `1050714–1050715` giữ nguyên kết luận. Trên Default, numeric-subgraph so với target-only imputation giảm active drift `0.0112` SD, cohort numeric KS `0.0082` và correlation error `0.0333`; target KS hòa trong uncertainty, Wasserstein xấu hơn `0.0092` và categorical TV hòa. Trên Shoppers, numeric-subgraph giảm target KS `0.2298`, Wasserstein `0.2494`, cohort numeric KS `0.0515` và categorical TV `0.0097`, nhưng active drift tăng `0.4175` SD; correlation error hòa. Đây là evidence dương có ý nghĩa cho descendant graph, nhưng chưa phải matched-scope comparison vì baseline mới chỉ mask target.

### Hypotheses đang kiểm tra tiếp

**H2.5 — Lợi ích phụ thuộc độ khó policy.** Jobs `1050695–1050696` đã sweep 12 Default và 15 Shoppers controlled policies. Trên Default, numeric-subgraph thắng target-only imputation ở drift `12/12`, numeric KS `9/12`, categorical TV `10/12`, correlation `11/12`, nhưng chỉ thắng target KS `5/12` và Wasserstein `1/12`. Trên Shoppers, nó thắng imputation ở target KS `13/15`, Wasserstein `11/15`, numeric KS `13/15`, categorical TV `11/15`, nhưng drift chỉ `3/15`. Không policy nào bị một method strict-dominate trên cả sáu metrics. Khi violation tăng, lợi ích fidelity của graph so với imputation tăng mạnh trên Shoppers (Spearman `rho=0.94` cho target KS, `0.92` cho Wasserstein), đồng thời bất lợi displacement cũng tăng (`rho=-0.90`). Vì vậy hypothesis được sửa: difficulty không quyết định một universal winner; nó quyết định vị trí trên fidelity--intervention frontier.

**H2.6 — Dynamic graph hơn matched-scope diffusion-imputation khi descendant neighborhood thật sự cần thiết.** Target-only imputation đã bị vượt trên phần lớn joint-fidelity metrics. Phép thử khó hơn hiện đã submit: `1050740` mask cùng numeric descendant set trong TabDiff imputation, sau đó `1050741–1050742` aggregate ba seed. Nếu matched-scope imputation bằng hoặc hơn numeric-subgraph, contribution của in-denoising condition bị bác bỏ; phần còn lại chỉ là active-set và scope benchmark. Nếu numeric-subgraph vẫn hơn, claim method sẽ là graph-routed conditional resampling chứ không phải generic LLM feedback.

**H2.7 — Không dùng near-hard empirical relations làm headline.** Các quan hệ tự nhiên có violation khoảng `1–2%` đã bão hòa; method không có đủ room để tạo effect. Chúng chỉ còn vai trò safety test cho over-enforcement và exception preservation. Headline benchmark phải là unseen runtime policy có nhiều mức difficulty, heldout real reference và strong repair/imputation baselines.

### Claim hiện tại sau strong baselines

Claim được support tốt nhất là:

> Two-pass lookahead xây active graph bốn trạng thái từ một preview không điều kiện; rollback cùng noise, context anchoring và graph-selected descendant resampling tạo một fidelity--intervention frontier ổn định trên hai mixed-type datasets, đồng thời bảo toàn satisfied, not-applicable và valid-exception rows.

Vai trò LLM vẫn phải giới hạn ở compile unseen natural-language policy thành typed graph và đề xuất descendant scope. Evidence hiện tại không ủng hộ LLM direct judgement từng row hoặc LLM selection từ empirical candidate graph. Chưa thể gọi kết quả là SOTA cho tới khi có: kết quả ba seed với diffusion-imputation, policy difficulty curve, ít nhất một benchmark policy ngoài tập tự thiết kế, và baseline CTDF/Harpoon-style được matched actuation budget.

## 11. Strong-baseline falsification và hướng sửa

### Matched-scope imputation

Aggregate `1050741–1050742` cho thấy kết luận khác nhau theo dataset. Trên Default, matched numeric-scope imputation tốt hơn numeric-graph conditioning về target KS `0.0256`, Wasserstein `0.0248` và cohort numeric KS `0.0027`; conditioning tốt hơn về active drift `0.0328` SD và correlation error `0.0280`. Trên Shoppers, conditioning tốt hơn matched imputation về target KS `0.1743`, Wasserstein `0.1439`, numeric KS `0.0275` và categorical TV `0.0096`, nhưng active drift cao hơn `0.3127` SD. Do đó matched imputation chưa bác bỏ graph, nhưng vẫn để lại Pareto tradeoff.

### Sparse descendant graph

Jobs `1050814–1050816` dùng train-only Spearman ranking để chọn target + top-2/top-4 descendants. Trên Default, sparse scope gần như target-only và không cải thiện fidelity. Trên Shoppers, top-k3 giảm drift từ `0.8072` của full numeric graph xuống `0.7204`, đồng thời cohort numeric KS giảm từ `0.1632` xuống `0.1584`; target KS tăng từ `0.2872` lên `0.3030`. Sparse graph vì vậy mở rộng frontier nhưng không tạo universal winner. Policy-grid `1050817` xác nhận top-k3 thắng matched imputation trên `12/15` policies về target KS và `13/15` về numeric KS, nhưng chỉ `5/15` về drift.

### Best-of-8 là phản chứng quyết định cho single-threshold policies

Jobs `1050828–1050830` chọn branch hợp lệ gần anchor nhất trong tám diffusion imputations rồi certify. Trên Shoppers, best-of-8 numeric graph đạt drift `0.7307`, target KS `0.2413`, Wasserstein `0.8808`, numeric KS `0.1482` và correlation error `0.1474`. Nó tốt hơn top-k3 conditioning (`0.7622/0.3130/0.9523/0.1632/0.1719`) trên năm metrics này; conditioning chỉ tốt hơn categorical TV (`0.1111` so với `0.1195`). Trên Default, best-of-8 cũng thắng conditioning về target marginal, còn conditioning giữ lợi thế displacement/correlation ở một số scope.

Kết luận bắt buộc là: **mid-denoising conditioning chưa được support như method tốt hơn branch selection cho single numeric threshold**. Kết quả đẹp trước đó phụ thuộc baseline imputation chưa có constraint-aware branch selection. Single-threshold benchmark vẫn hữu ích cho active-set, exception preservation và scope analysis, nhưng không thể làm headline method benchmark.

### Generalization boundary

Audit training catalog cho thấy Default headline threshold `1000` đã xuất hiện trong atomic training; zero-shot ở đó chỉ là subgroup composition. Shoppers headline thresholds `5/150/1500` không xuất hiện trực tiếp trong training catalog. Held-out-only jobs `1050840–1050842` dùng Default thresholds `500/2000` và Shoppers thresholds `5/25/50/150/1500`; kết quả giữ cùng fidelity--drift tradeoff. Vì vậy có evidence cho threshold interpolation + unseen subgroup composition, nhưng chưa có cross-dataset foundation model: mỗi dataset vẫn dùng adapter riêng.

### Hypothesis được ưu tiên tiếp theo

**H2.8 — Constraint conjunction tạo khoảng trống thực cho conditioning.** Shoppers screen của job `1050853` cho kết quả hỗn hợp. Best-of-8 tìm được ít nhất một valid branch cho `78.8%` violated rows, trong khi một conditional trajectory chỉ đạt pre-certifier joint validity `57.1%`. Sau cùng certifier, conditioning có numeric drift `0.4726` và cohort numeric KS `0.3436`, tốt hơn best-of-8 `0.5852/0.3611`, nhưng categorical TV xấu hơn `0.0747` so với `0.0660`. Quan trọng hơn, direct clamp đạt drift `0.0454` và numeric KS `0.2425`, tốt nhất vì cả hai constraints chỉ là lower bounds có một thao tác hiển nhiên. Do đó H2.8 bị thu hẹp: conjunction chỉ có giá trị khi constraints là relational/non-local hoặc có nhiều repair routes; thêm nhiều threshold clampable không đủ tạo method gap.

**H2.9 — Solver routing thay vì một actuator cố định.** Dynamic constraint graph nên chứa applicability, satisfaction, descendant scope, feasible-mass estimate và operator choice. Với single constraints có feasible mass cao, dùng best-of-N/imputation; với conjunction hoặc non-local coupling có feasible mass thấp, dùng conditional denoising; luôn certify cuối. Đây hiện là method hypothesis mạnh hơn việc bắt buộc LLM feedback ở mọi reverse step.

**H2.10 — Vai trò foundation model.** Foundation model compile natural-language request thành typed predicates, exception clauses và candidate descendants; deterministic statistics kiểm chứng graph; runtime router chọn solver. LLM không nên thay verifier hoặc gọi lặp lại trên từng row nếu executable policy đã có. Chỉ giữ row-level LLM feedback cho semantic constraints không thể compile thành verifier đáng tin.

**H2.11 — Ambiguous relational actuation.** Benchmark tiếp theo nên chứa cùng một violation nhưng ít nhất hai cách sửa hợp lệ, ví dụ count--duration consistency, ordering/ratio constraints hoặc context-dependent choice of which endpoint to resample. Metric chính phải là repair-route recovery, joint counterfactual error và downstream utility, không chỉ final violation. Đây là nơi dynamic graph có thể thay topology/actuator theo row; clamp, projection và best-of-N không còn mặc nhiên giải quyết đúng causal endpoint. Existing ambiguous-repair audit đã cho context router route accuracy `0.785` so với `0.5` của fixed routes, cung cấp cơ sở thực nghiệm để triển khai diffusion actuation cho H2.11.

### Kết quả conjunction đầy đủ và thí nghiệm route-specific actuation

Default conjunction screen của `1050853` củng cố phản chứng từ Shoppers. `best-of-8` tìm được valid branch cho `98.92%` violated rows, còn conditional trajectory đạt pre-certifier validity `81.17%`. Sau certification, clamp có drift `0.0016` và cohort numeric KS `0.1265`, tốt hơn conditional denoising (`0.0603/0.1844`) và best-of-8 (`0.0487/0.1904`). Unconditional có fidelity tốt nhất nhưng validity chỉ `62.49%`. Như vậy conjunction của các cận số học vẫn là benchmark quá dễ cho projection; không nên dùng làm headline.

Audit ambiguous repair chính thức trên hai count--duration relations cho context router route accuracy `0.7345`, pair error `0.9996`; fixed consequent repair đạt `0.5/1.4003`, fixed antecedent `0.5/1.8332`, oracle `1.0/0.4776`. Tất cả đều đạt validity `1.0`, nên violation rate hoàn toàn không phân biệt được sửa đúng và sửa sai. Đây là bằng chứng trực tiếp rằng benchmark cần đo route recovery/counterfactual error.

Job `1050858` thay deterministic endpoint repair bằng TabDiff imputation, với cùng controlled cases và bốn route: fixed-duration, fixed-count, learned context-router và oracle. Trước certifier, context router đạt route accuracy `0.7428`, validity `0.9977`, pair error `0.9594`; fixed-duration là `0.5/1.0/1.3959`, fixed-count là `0.5/0.9828/1.5480`, oracle là `1.0/0.9967/0.4613`. Sau certifier mọi method đạt validity `1.0`, trong khi pair error gần như giữ nguyên (`0.9586`, `1.3959`, `1.5563`, `0.4593`). Đây là evidence dương rõ cho row-specific topology/actuator selection và đồng thời cho thấy independent certification không thể thay thế route selection. Tuy nhiên đây vẫn là controlled corruption trên hai relations và router supervised, chưa chứng minh LLM, intermediate feedback hoặc generation tự do.

Seed xác nhận `1050860` lặp lại kết quả: context-router/fixed-duration/fixed-count/oracle pair error lần lượt là `0.9303/1.3740/1.5245/0.4407`. Ablation một resampling round `1050861` vẫn cho `0.9450/1.3959/1.4599/0.4865` sau certification. Do đó gain không phụ thuộc seed đầu tiên hoặc ba vòng resampling. So với fixed-duration mạnh hơn, router giảm pair error khoảng `31–36%`; so với fixed-count giảm khoảng `35–39%`.
