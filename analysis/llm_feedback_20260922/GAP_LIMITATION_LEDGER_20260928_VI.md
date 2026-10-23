# Bản đồ khoảng trống, giới hạn và hướng nghiên cứu — 28/09/2026

## 1. Kết luận điều hành

Ý tưởng ban đầu — cho LLM đọc `x̂₀(x_t)` rồi phản hồi trong reverse diffusion — **không còn đủ làm claim trung tâm**. PromptLoop, PPAD, Free²Guide và TTGen đã bao phủ nhiều biến thể foundation model đọc preview rồi sửa prompt, chấm điểm hoặc chọn nhánh. Trong tabular diffusion, Harpoon và CTDF đã áp objective/feasibility operator trong denoising; SCTab-Diff đã condition reverse process bằng prior do LLM trích xuất; LCATS và Zhao et al. đã bao phủ discovery, validation và post-generation repair.

Kết quả hiện tại ủng hộ một bài toán hẹp nhưng mạnh hơn:

> **Open-contract conditional tabular diffusion:** biên dịch một yêu cầu ngôn ngữ tự nhiên chưa thấy thành typed contract có parameter, applicability và exception; dùng contract làm condition để sinh đúng *phân phối cohort*, không chỉ sửa cell cho qua rule; khi contract chỉ áp dụng cho một số hàng, ước lượng active set theo từng hàng và chỉ tác động lên các cột được phép.

Ba kết quả mới làm hướng này có cơ sở thành research project:

1. **Khả năng chuyển giao qua contract là thật và lặp lại trên ba dataset.** Request-ID không phản ứng với threshold/composition chưa thấy, trong khi structured condition phản ứng mạnh trên Adult, Default và Shoppers.
2. **Learned conditioning có vùng Pareto dưới ngân sách tương đương.** Trên Default held-out composition, structured CFG 2 hơn best-of-2 về satisfaction, numeric KS và categorical TV trong toàn bộ 9 cặp contract–seed. Trên Shoppers, satisfaction tăng mạnh nhưng categorical fidelity còn trade-off.
3. **Active-set đúng hàng giải quyết một failure mode thật.** Global-early guidance tăng validity nhưng làm lệch các hàng không áp dụng. Paired unconditional preview rồi selective rerun giữ gần toàn bộ lợi ích validity với inactive-row error gần baseline; shuffled gate làm mất phần lớn lợi ích.
4. **Phương án mạnh nhất hiện tại không cần LLM ở từng bước.** Structured condition tạo một “điểm khởi đầu phân phối” gần cohort đích, sau đó certifier xác định chỉ sửa phần dư. Ở ngân sách một lần gọi denoiser mỗi bước (CFG 1), phương án này đạt validity tuyệt đối và tốt hơn proxy feasibility-per-step hiện có về fidelity, edit rate và distortion trên cả Default lẫn Shoppers.

Kết luận hiện tại:

- **Đủ tốt để tiếp tục thành project hướng ICDE**, đặc biệt theo framing controllable data generation / conditional data synthesis / constraint-aware generation.
- **Chưa đủ để gọi là state of the art.** Baseline CTDF/Harpoon/SCTab-Diff/LCATS chưa được tái lập faithful trên cùng benchmark; contract còn được kiểm soát thủ công; chưa có utility, privacy và benchmark semantic độc lập đủ rộng.
- Claim khả thi nhất là **generalization và fidelity của contract-conditioned generation**, cộng với **selective active-set intervention**. Không nên claim “first LLM feedback during denoising”.

## 2. Phân rã bài toán để tránh claim chồng lấn

| Thành phần | Câu hỏi | Những gì related work đã làm | Phần còn mở trong project |
| --- | --- | --- | --- |
| Constraint specification | Người dùng muốn gì? | Harpoon/CTDF nhận objective đã executable; SCTab-Diff nhận prior; EVS/Zhao dùng LLM tìm rule | Request chưa thấy, có parameter/composition/exception, được compile thành typed contract |
| Applicability | Contract áp dụng cho hàng nào? | Static rule engine có thể evaluate row sạch; Harpoon response thay đổi theo state | Applicability bất định khi row còn nhiễu; exception và abstention theo hàng/timestep |
| Conditional generation | Sinh cohort phù hợp hay chỉ ép cell? | Projection/repair bảo đảm feasibility | Học `p(x | contract)` để các cột liên quan cùng dịch chuyển hợp lý |
| Actuation | Tác động vào diffusion ra sao? | Gradient, projection, remask, branch selection đã có | Target-scoped actuation; preview-rerun; calibrated active set; rollback có kiểm chứng |
| Verification | Biết kết quả đúng bằng gì? | Nhiều work dùng chính rule đã guidance | Independent verifier, subgroup/rare-mode coverage, utility, cost và semantic audit |

Điểm ranh giới quan trọng:

- **Specification cố định không có nghĩa response cố định.** Harpoon giữ objective cố định nhưng gradient thay đổi theo `x̂₀,t`; SCTab-Diff giữ prior cố định nhưng score vẫn phụ thuộc row context; CTDF áp feasibility map ở mỗi bước.
- Vì vậy, “row-conditioned” hoặc “timestep-dependent” nói chung không phải novelty.
- Phần có thể bảo vệ là **request-specific contract chưa thấy + calibrated applicability/exception + distribution-aware conditional generation + verified selective actuation**.

## 3. Bằng chứng thực nghiệm hiện có

### 3.1 Dataset và rule audit

Audit Adult, Default và Shoppers tạo 788 candidate relations: 80 empirical executable, 95 near-hard, 334 xác suất, 85 demographic/sensitive chỉ nên audit, 57 dính target, 132 thiếu support và 5 exception-rate shift.

Ví dụ quan trọng:

- Shoppers `Administrative == 0 → Administrative_Duration == 0` gần như exact theo hướng này.
- Chiều đảo `Administrative_Duration == 0 → Administrative == 0` có khoảng 2–3% exception.
- Adult `Own-child → age < 30` có khoảng 19,7% exception ở cả train và test; biến nó thành hard constraint sẽ xóa nhóm hợp lệ.

Kết luận: majority của relation khai phá từ dữ liệu **không được phép tự động biến thành hard semantics**. Đây là động lực cho hardness, confidence, exception và abstention, không phải bằng chứng LLM nên enforce correlation.

Artifact: `outputs/cross_dataset_constraint_20260924/`.

### 3.2 Adult hard-rule setting đã bão hòa

Full TabDiff checkpoint tự nhiên chỉ vi phạm hard rules khoảng 1,56–1,76%. Post-repair đạt 0%; direct typed projection còn khoảng 0,065–0,13%. Trong setting này LLM không cần thiết, dynamic feedback không hơn repair và validity gần bão hòa. Adult hard rules chỉ còn giá trị như sanity check và counterexample, không nên là main benchmark.

### 3.3 Open-contract transfer trên Adult

Typed adapter học atomic contracts và được test trên threshold/composition chưa thấy. Ba seed cho thấy:

- request-ID không zero-shot được contract mới;
- structured AST/typed vector chuyển giao được;
- unseen composition, typed CFG 2 đạt satisfaction khoảng 0,608 và cohort KS khoảng 0,110;
- best-of-2 khoảng 0,498 / 0,142;
- best-of-4 khoảng 0,712 / 0,093 nhưng dùng bốn trajectories;
- final projection đạt satisfaction 1,0 nhưng edit nhiều và categorical fidelity kém.

Đây là evidence đầu tiên cho framing “generate a cohort” thay vì “repair a row”.

### 3.4 Cross-dataset transfer: Default và Shoppers

Ba seed, cùng dataset checkpoint, cùng contract split:

| Dataset / split | Method | Satisfaction | Cohort numeric KS | Categorical TV |
| --- | --- | ---: | ---: | ---: |
| Default / unseen composition | request-ID CFG 2 | 0,281 | 0,298 | 0,150 |
| Default / unseen composition | structured CFG 2 | **0,543** | **0,151** | **0,042** |
| Default / unseen composition | best-of-2 | 0,333 | 0,284 | 0,110 |
| Default / unseen composition | best-of-4 | 0,531 | 0,199 | 0,076 |
| Default / unseen composition | final projection | 1,000 | 0,359 | 0,158 |
| Shoppers / unseen composition | request-ID CFG 2 | 0,223 | 0,265 | 0,082 |
| Shoppers / unseen composition | structured CFG 2 | **0,615** | **0,170** | 0,093 |
| Shoppers / unseen composition | best-of-2 | 0,186 | 0,291 | **0,086** |
| Shoppers / unseen composition | best-of-4 | 0,332 | 0,200 | **0,066** |
| Shoppers / unseen composition | final projection | 1,000 | 0,393 | 0,097 |
| Shoppers / unseen threshold | structured CFG 2 | **0,859** | 0,115 | 0,067 |
| Shoppers / unseen threshold | best-of-4 | 0,635 | **0,107** | **0,058** |

Paired bootstrap trên contract–seed:

- Default composition, structured CFG 2 so với best-of-2: satisfaction `+0,210` [0,152; 0,265], KS reduction `+0,133` [0,076; 0,184], TV reduction `+0,068` [0,051; 0,083]; 9/9 cặp là Pareto win.
- Shoppers threshold, structured CFG 2 so với best-of-2: satisfaction `+0,427` [0,323; 0,528], KS reduction `+0,061` [0,028; 0,092]; TV chưa rõ, interval cắt 0.
- So với best-of-4, kết quả phụ thuộc dataset/split. Không được claim learned conditioning dominate best-of-4 tổng quát.
- Final projection thắng validity tuyệt đối nhưng thua rõ về cohort fidelity; đây là trade-off, không phải learned conditioning thắng tuyệt đối.

Large-sample evaluation với 4.096 mẫu cho mỗi held-out contract giữ nguyên hiệu ứng: Default CFG 2 đạt `0,534 / 0,150 / 0,035` trên composition và `0,737 / 0,090 / 0,045` trên threshold; Shoppers đạt `0,615 / 0,165 / 0,086` và `0,862 / 0,108 / 0,058`. Sai khác so với run 512 mẫu nhỏ, nên tín hiệu không phải Monte Carlo artifact rõ ràng.

Artifacts: `outputs/cross_condition_aggregate_1044279/RESULTS.md`, `outputs/cross_large_eval_aggregate_1044339/RESULTS.md`, `outputs/cross_actuation_v2_aggregate_1044346/RESULTS.md`, `outputs/paired_claim_audit_1044347/RESULTS.md`.

Lưu ý: `target_distortion` trong actuation baseline v1 bị invalid do pandas aliasing; satisfaction/KS/TV/edit rate vẫn đúng. Bản v2 sau immutable pre-edit copy cho final projection normalized distortion khoảng 0,31–0,47 trên held-out contracts, xác nhận phép projection không phải edit bằng không.

### 3.5 Causal audit cho active set

Tất cả conditional deltas được giới hạn vào target column; initial noise được giữ cặp. Ở CFG 4, trung bình trên held-out contracts:

- không actuation: violation khoảng 0,660;
- global-early rồi hard gate: khoảng 0,583 nhưng inactive error khoảng 3,63;
- soft posterior, không floor: khoảng 0,603, inactive error khoảng 1,21;
- shuffled gate: khoảng 0,638, cho thấy prevalence/timing một mình không giải thích được gain;
- paired unconditional preview rồi selective rerun: khoảng 0,581, inactive error khoảng 1,05;
- baseline inactive error khoảng 0,97.

Paired bootstrap xác nhận global-early CFG 4 giảm violation `0,081` [0,054; 0,110] nhưng làm inactive error xấu thêm `2,856` [2,328; 3,388]. Soft posterior giảm violation `0,060` [0,040; 0,081], inactive error xấu thêm khoảng 0,265. Preview-rerun giảm violation khoảng `0,085` [0,064; 0,106] trong audit đầu với inactive drift nhỏ hơn global rất nhiều. Shuffled gate chỉ giảm khoảng `0,025`.

Calibrated active-set sweep mới cho thấy:

- CFG 4, held-out composition: no actuation `0,502`; preview-rerun `0,383`; prevalence top-k `0,410`; calibrated soft tốt nhất khoảng `0,428`;
- preview-rerun giữ inactive error `1,565`, gần no-actuation `1,546`, trong khi global-early ở sweep trước khoảng `4,85` cho riêng composition;
- exception split vẫn khó: các method chỉ cải thiện rất ít và đôi lúc exception error tăng.

Diễn giải đúng: có một attainable frontier nếu active set đúng hàng; posterior hiện tại chưa đủ calibrated; preview-rerun là phương pháp hai lượt chạy được, không phải ground-truth oracle; kết quả vẫn là controlled contract pilot.

Artifacts: `outputs/prob_gate_aggregate_1044263/RESULTS.md`, `outputs/calibrated_gate_aggregate_1044299/RESULTS.md`.

### 3.6 Foundation model: direct judgement hay compile-once

Benchmark mở rộng vừa chạy có 10 semantic contracts × 5 surface forms = 50 requests, 400 synthetic-only cases, cân bằng bốn trạng thái; opaque IDs và gold/oracle chỉ ở local.

- GPT-4.1-mini compile execution equivalence: 0,955 / 0,900 / 0,900; khoảng 11,8–12,3k tokens.
- GPT-5.4-mini compile: 0,915 / 0,800 / 0,880; khoảng 11,8k tokens.
- GPT-4.1-mini direct: khoảng 0,65–0,70 ở batch 32; 0,77–0,80 ở batch 8; 104–124k tokens.
- GPT-5.4-mini direct: khoảng 0,54–0,57 ở batch 32; 0,65–0,69 ở batch 8; 105–125k tokens.

Compile-once đáng tin hơn và rẻ hơn gần một bậc độ lớn so với repeated row judgement, nhưng chưa solved: seed variance còn lớn; schema-valid JSON không bảo đảm semantic equivalence. Hai surface khó nhất là exception-first và policy formula; basic compile accuracy theo surface chỉ khoảng `0,73–0,84`, trong khi canonical/ordinary prose khoảng `0,93–1,00`. Riêng trạng thái `excepted` chỉ đạt `0,760` với GPT-4.1-mini và `0,613` với GPT-5.4-mini. Đây là evidence trực tiếp rằng exception semantics, không phải JSON syntax, là bottleneck.

Hai follow-up diagnostic:

- self-consistency ba lần của GPT-4.1-mini: majority coverage `0,985`, selective accuracy `0,939`; unanimous coverage `0,865`, selective accuracy `1,000`. Cross-model unanimous coverage `0,725`, accuracy `1,000`;
- prompt yêu cầu tách `scope / exception / requirement / repair` trước khi emit DSL tăng mean execution equivalence lên khoảng `0,957` với GPT-4.1-mini và `0,975` với GPT-5.4-mini, gần như loại exception false-action, vẫn chỉ khoảng 12k tokens.

Kết quả cho thấy uncertainty có thể quan sát qua disagreement và semantic role decomposition giải quyết phần lớn lỗi. Tuy nhiên prompt được thiết kế sau khi nhìn failure mode, nên có nguy cơ benchmark overfitting; phải test trên held-out templates/policy sources trước khi dùng làm claim.

Artifact: `outputs/request_condition_aggregate_1044328/` và job breakdown `1044348`.

### 3.7 Mixed-type open-contract transfer

Adapter mới chỉ thấy các request nguyên tử trong training: numeric lower bound hoặc categorical equality. Evaluation giữ lại numeric parameter mới, categorical–categorical conjunction và numeric–categorical conjunction chưa thấy. Ba seed cho hai dataset:

| Dataset / split | Unconditional satisfaction | Structured CFG 1 | Structured CFG 2 | KS/TV tại CFG 1 |
| --- | ---: | ---: | ---: | ---: |
| Default / categorical composition | 0,223 | 0,318 | 0,393 | 0,058 / 0,112 |
| Default / mixed composition | 0,321 | 0,406 | 0,472 | 0,137 / 0,112 |
| Shoppers / categorical composition | 0,134 | 0,411 | 0,559 | 0,076 / 0,133 |
| Shoppers / mixed composition | 0,112 | 0,454 | 0,685 | 0,077 / 0,095 |

Request-ID control hầu như trùng unconditional và không đổi theo CFG, đúng như negative control mong đợi. Trên Shoppers CFG 1, structured condition đồng thời tăng satisfaction và giảm mạnh numeric KS/categorical TV so với unconditional (`0,129/0,217` xuống `0,076/0,133` cho categorical composition; `0,194/0,202` xuống `0,077/0,095` cho mixed composition). Default cũng có Pareto improvement ở CFG 1, nhưng tăng validity nhỏ hơn.

CFG 4 tiếp tục cho thấy over-guidance: satisfaction tăng nhưng fidelity thường xấu rõ. Điều này ủng hộ việc chọn guidance theo Pareto frontier, không tối đa hóa satisfaction trước certifier.

Giới hạn: categorical values đều đã xuất hiện trong các atomic training requests; zero-shot nằm ở **composition**, không phải unseen category semantics. Catalog chỉ có ba categorical và ba mixed compositions mỗi dataset. Phép chiếu categorical trực tiếp đặt giá trị đích, nên validity 1,0 và target-column TV giảm là hệ quả cơ học; contribution phải được đo bằng edit burden, non-target dependencies và utility.

Đối chứng same-certifier sau đó xác nhận structured warm start giảm correction burden ở CFG 1:

- Default categorical composition: edit `0,682` so với `0,777` từ unconditional; mixed composition `0,594` so với `0,679`;
- Shoppers categorical composition: `0,589` so với `0,867`; mixed composition `0,546` so với `0,888`;
- Shoppers mixed composition còn giảm numeric distortion `0,228 → 0,070`, KS `0,194 → 0,077` và TV `0,107 → 0,040` sau cùng một certifier.

Paired bootstrap theo contract–seed cho thấy claim mạnh nhất nằm ở Shoppers mixed composition: structured CFG 1 hơn request-ID `+0,343` satisfaction [0,232; 0,439], giảm KS `0,118` [0,085; 0,153] và TV `0,110` [0,069; 0,152], Pareto 9/9. Sau cùng certifier, hybrid giảm edit `0,342` [0,231; 0,439], numeric distortion `0,158` [0,087; 0,236], KS `0,117` [0,086; 0,150] và TV `0,066` [0,037; 0,101], Pareto 9/9.

Decomposition target/non-target bác bỏ lo ngại rằng mean fidelity chỉ được làm đẹp bởi target column. Trên Shoppers mixed composition, structured CFG 1 so với request-ID giảm target numeric KS `0,588 → 0,236`, non-target numeric KS `0,151 → 0,059`, target categorical TV `0,771 → 0,436` và non-target categorical TV `0,124 → 0,046`. Sau certifier, target categorical TV đương nhiên về 0, nhưng non-target TV vẫn là `0,046` so với `0,122` khi certify unconditional sample. Default mixed cũng cải thiện cả bốn nhóm, nhưng biên nhỏ hơn.

Mức giảm trên Default categorical/mixed nhỏ hơn Shoppers: Default mixed hybrid giảm edit và KS rõ, nhưng TV interval `[-0,002; 0,037]` cắt 0 và Pareto chỉ 3/9. Vì vậy không claim universal domination; đây là heterogeneity cần giải thích theo dependency strength/cohort support. Aggregate đầu `1044450` thất bại chỉ ở khâu đọc bốn cột metric mới được thêm sau khi worker đã khởi chạy; sáu worker đều hoàn tất và retry `1044465` tổng hợp đủ kết quả.

Artifacts: `outputs/mixed_condition_aggregate_1044417/RESULTS.md`, `outputs/mixed_condition_cert_aggregate_retry_1044465/RESULTS.md`, `outputs/mixed_condition_decomp_aggregate_1044463/summary.csv`, `outputs/mixed_type_claim_audit_1044467/RESULTS.md`.

### 3.8 Compiler khi đổi schema

Một diagnostic mới dùng Default và Shoppers, mỗi bảng bốn semantics × ba cách diễn đạt, 48 synthetic hand-constructed cases; không có real row. Đây là schema-transfer test chưa dùng khi chỉnh prompt Adult, nhưng **không phải natural-policy corpus bên ngoài**.

- Default, GPT-4.1-mini basic: `1,000` execution equivalence ở cả ba seed; decomposed trung bình `0,993`.
- Default, GPT-5.4-mini basic: `1,000 / 0,500 / 1,000`, cho thấy một catastrophic parse run; decomposed `1,000` cả ba seed. Audit cho thấy run lỗi đổi các categorical codes (`SEX`, `MARRIAGE`, `EDUCATION`, `PAY_0`) từ string trong vocabulary thành number trong DSL, làm deterministic executor đánh dấu nhiều row là not-applicable. Đây là lỗi grounding/type thật, không phải JSON syntax.
- Shoppers, GPT-4.1-mini basic: trung bình `0,972`; decomposed `1,000` cả ba seed.
- Shoppers, GPT-5.4-mini: execution equivalence `1,000` cho cả basic và decomposed, dù một decomposed run chỉ đạt structural exact `0,75` nhưng execution vẫn tương đương trên cases.

Kết luận đúng là role decomposition có giá trị như robustness guard, đặc biệt ngăn failure seed-level của GPT-5.4-mini trên Default. Không được kết luận compiler đã solved: bank nhỏ, language vẫn controlled, và execution equivalence chỉ kiểm tra 4 case/status mỗi semantic surface. Một parse có thể sai ngoài support nhưng tình cờ tương đương trên cases; cần property-based counterexamples hoặc SMT/table enumerator.

Artifact: `outputs/request_condition_aggregate_1044443/RESULTS.md`.

### 3.9 Kết quả bị loại khỏi evidence

Focused-label F1 khoảng 0,92 trước đây **không hợp lệ** vì API payload chứa `case_id` có tên relation và hậu tố `_clean/_corrupt`; controls còn nhận candidate relation=`none`. Kết quả này không được dùng trong claim nào. Benchmark sau dùng opaque IDs; request, label, family, pair design và oracle ở local.

## 4. Bản đồ related work và ranh giới novelty

| Work | Cơ chế liên quan | Điều không được claim | Khoảng trống còn lại |
| --- | --- | --- | --- |
| TabSyn, ICLR 2024 | VAE mixed-type → latent continuous → diffusion | Không claim latent tabular diffusion | Interface decode `x̂₀` và feedback quay lại latent; backbone phụ |
| TabDiff, ICLR 2025 | Numeric/categorical diffusion, feature-wise noise | Không claim mixed-type diffusion | Backbone chính; conditional adapter và active-set controller |
| Harpoon, ICLR 2026 | Differentiable inference loss trên predicted clean sample | Không claim first constraint guidance trong sampling | Non-differentiable request semantics, exception, selective activation |
| [CTDF, ICAIF 2025](https://arxiv.org/abs/2606.28674) | Euclidean/KL feasibility maps mỗi reverse step | Không claim first per-step mixed-type enforcement | Distribution-aware cohort generation và request-conditioned active set; không tìm thấy official code công khai nên proxy phải ghi rõ |
| SCTab-Diff, preprint 2026 | LLM prior condition reverse denoising; score phụ thuộc row | Không claim first LLM prior hay generic row-conditioned score | Unseen request-specific parameters/compositions/exceptions |
| LLM-TabLogic | LLM khám phá relation và representation trước diffusion | Không claim first LLM-assisted representation | Online user contract và executable interface |
| LCATS, KDD 2026 | Discovery/verification, diagnosis và repair | Không claim first LLM validation/repair | Conditional generation trước final output; same-compiler comparison |
| Zhao et al., preprint 2026 | Executable constraint, counterexample revision, repair | Không claim first NL/metadata → rule → repair | Request-specific policy, active set, uncertainty, fidelity |
| EVS, workshop 2026 | Noisy LLM oracle, extract/verify/select constraints | Không claim first LLM extraction/filtering | User-request semantics và trajectory intervention |
| PromptLoop, CVPR 2026 | MLLM đọc denoised estimate và cập nhật prompt | Không claim first intermediate FM feedback | Typed tabular contract, applicability, exception, verifier |
| PPAD, preprint 2025 | Preview/lookahead, rollback/resample | Không claim first preview rollback | Mixed-type active-set calibration và subgroup preservation |
| Free²Guide, ICCV 2025 | Black-box LVLM reward | Không claim first gradient-free FM reward | Executable contract và matched best-of-N |
| TTGen, workshop 2025 | Clean-latent prompt update và branch selection | Không claim first prompt update/branch score | Request/compiler/active-set problem |
| Constrained Discrete Diffusion, NeurIPS 2025 | Simplex/KL projection cho discrete states | Không claim first categorical projection | Mixed-table conditioning và exception-aware actuation |

## 5. Khoảng trống ưu tiên

### G1 — Từ feasibility sang conditional cohort generation

Projection/repair có thể đạt zero violation nhưng thường chỉ clamp target và không sinh correlated features của cohort. Best-of-N giữ sample tự nhiên nhưng tốn nhiều trajectories và yếu khi event hiếm. Claim tiềm năng là parameterized typed condition cho phép sinh từ `p(x | contract)` với fidelity cohort tốt hơn projection và hiệu quả hơn rejection/best-of-N trong một vùng ngân sách.

Evidence đã có trên ba dataset, mạnh nhất khi so CFG 2 với best-of-2 trên Default composition. Giới hạn chính: contracts còn là threshold/conjunction; final projection vẫn thắng validity tuyệt đối. Bác bỏ hướng này nếu faithful CTDF/Harpoon loại bỏ toàn bộ Pareto region.

CTDF-style numeric per-step proxy mới củng cố rằng đây là trade-off thật: Default composition per-step projection đạt satisfaction `1,000` nhưng KS/TV `0,307/0,144`, còn structured CFG 2 là `0,534/0,150/0,035`; Shoppers threshold là `0,948/0,185/0,064` so với `0,862/0,108/0,058`. Projection tối đa hóa feasibility; learned condition tạo cohort gần reference hơn. Proxy này chưa có categorical KL mapping và không được gọi là CTDF reproduction.

Harpoon-style squared-hinge gradient proxy trên cùng checkpoint/contract cũng tạo frontier tương tự. Ở eta `0,3`, Default composition đạt satisfaction `0,710` nhưng KS/TV `0,319/0,146` và distortion `0,599`; structured CFG 1 đạt satisfaction `0,406` với `0,214/0,063` trước certifier. Trên Shoppers composition, eta `0,3` đạt `0,493/0,277/0,076`, còn structured CFG 1 là `0,460/0,139/0,044`. Sau certifier, Harpoon eta `0,3` đôi lúc có edit rate thấp hơn hybrid (Default composition `0,290` so với `0,595`; Shoppers `0,507` so với `0,540`) nhưng distortion cao hơn rất lớn (`0,599/0,765` so với `0,194/0,108`) và fidelity kém hơn. Đây là trade-off count-versus-magnitude, không phải dominance. Proxy dùng loss và step size tự thiết kế, nên chưa thay thế faithful Harpoon.

Hybrid `structured condition → exact final projection` đạt satisfaction `1,000` mà gần như giữ nguyên fidelity của conditional sample. So với projection từ unconditional:

- Default composition: edit `0,466` thay vì `0,814`, distortion `0,132` thay vì `0,357`, KS/TV `0,147/0,035` thay vì `0,359/0,158`;
- Default threshold: edit `0,264` thay vì `0,580`, distortion `0,162` thay vì `0,470`;
- Shoppers composition: edit `0,385` thay vì `0,895`, distortion `0,070` thay vì `0,314`;
- Shoppers threshold: edit `0,138` thay vì `0,729`, distortion `0,049` thay vì `0,422`, KS/TV `0,108/0,058` thay vì `0,252/0,089`.

Đây hiện là method candidate mạnh nhất: conditional generation tạo distributional warm start, projection chỉ certify residual. Kết quả compute-matched CFG 1 đã hoàn tất:

- Default composition: hybrid đạt `1,000 / 0,211 / 0,063` (satisfaction/KS/TV), edit `0,595`, distortion `0,194`; CTDF-style certified proxy đạt `1,000 / 0,307 / 0,144`, distortion `0,469`;
- Default threshold: hybrid `1,000 / 0,105 / 0,046`, edit `0,372`, distortion `0,254`; proxy `1,000 / 0,176 / 0,095`, distortion `0,544`;
- Shoppers composition: hybrid `1,000 / 0,139 / 0,044`, edit `0,540`, distortion `0,108`; proxy sau raw-space certification `1,000 / 0,251 / 0,067`, residual edit `0,091`, distortion `0,522`;
- Shoppers threshold: hybrid `1,000 / 0,083 / 0,029`, edit `0,296`, distortion `0,128`; proxy certified `1,000 / 0,185 / 0,064`, residual edit `0,052`, distortion `0,541`.

Đây là bằng chứng mạnh cho **distributional warm start + residual certification**, nhưng vẫn chưa phải bằng chứng hơn CTDF thật: proxy chỉ clamp lower bound trong encoded numeric space, không triển khai Euclidean/KL feasibility operator đầy đủ. “Projection steps = 31” chỉ đếm 30 encoded clamps và một raw-space certification; nó không phản ánh thêm denoiser call.

### G2 — Active-set attribution dưới trajectory uncertainty

Global early guidance làm lệch inactive rows, còn late hard gate đến quá muộn. Row-aligned applicability evidence và calibration theo timestep là gap hẹp có thể bảo vệ. Shuffled control mất phần lớn gain; preview-rerun gần global validity nhưng bảo toàn inactive rows.

Giới hạn: preview-rerun tốn hai trajectories; antecedent hiện chủ yếu categorical và consequent numeric; exception split chưa cải thiện. Bác bỏ nếu post-repair/best-of-2 có frontier tốt hơn hoặc result không chuyển sang Default/Shoppers.

### G3 — Foundation model là compiler có uncertainty, không phải per-row judge

Direct judgement lặp lại tốn token, không ổn định và trộn reasoning với execution. Foundation model nên compile request thành typed AST; deterministic executor xử lý row; verifier/counterexample loop phát hiện parse sai; uncertainty quyết định abstain/ask/ensemble.

Giới hạn: controlled language trên một schema và EVS/Zhao đã có extraction/validation. Novelty không thể chỉ là “LLM compile rule”. Bác bỏ nếu grammar/parser hoặc smaller model đạt ngang trên realistic policies.

### G4 — Exception-aware probabilistic contracts

Majority relation thật là near-hard/probabilistic. Ép thành invariant làm mất rare valid modes và có thể mã hóa stereotype. Contract nên biểu diễn hardness, applicability, exception và allowed action; generator abstain hoặc optimize chance constraint thay vì hard projection.

Evidence hiện là audit, chưa có method hoàn chỉnh hay semantic ground truth. Đây là gap tốt nhưng rủi ro cao hơn G1/G2.

### G5 — Contract shift như benchmark mới

Benchmark synthetic tabular thường không đo generalization sang policy/contract chưa thấy. Có thể formalize split parameter, composition, exception, paraphrase và table/domain. Hạ tầng đã có signal rõ trên ba dataset, nhưng nếu contract vẫn do tác giả chọn và verifier trùng với request, reviewer có thể coi là toy benchmark.

## 6. Sổ giả thuyết

| Giả thuyết | Trạng thái | Bằng chứng / vấn đề |
| --- | --- | --- |
| H1: `x̂₀` đủ đọc ở một số timestep | Ủng hộ có điều kiện | Categorical/relations ổn muộn hơn numeric; preview quá sớm gây gate sai |
| H2: có semantic lỗi static prior không giải quyết | Chưa đủ | Controlled request shift có; natural semantic ground truth chưa có |
| H3: dynamic feedback hơn post-repair | Bác bỏ cho Adult hard rules | Repair đạt 0%; dynamic actuator cũ kém hơn |
| H4: LLM hơn executable verifier | Bác bỏ cho rule đã biết | LLM hợp lý ở compilation của request mới |
| H5: per-row LLM critique đáng tin/đáng tiền | Không ủng hộ | Direct judgement kém compile-once và tốn gần 10× tokens |
| H6: text–structured consistency | Chưa thử | Dataset hiện tại không có long text phù hợp |
| H7: typed condition generalize unseen parameter/composition | Ủng hộ trên ba dataset | Request-ID control không generalize |
| H8: learned condition có Pareto region | Ủng hộ bước đầu | Thắng best-of-2 rõ ở một số split; best-of-4/projection tạo trade-off |
| H9: row-aligned evidence có unique information | Ủng hộ | Shuffled gate mất phần lớn validity gain |
| H10: calibrated active set giảm row leakage | Ủng hộ một phần | Soft/top-k tốt hơn global; preview-rerun tốt nhất; exception khó |
| H11: compile-once tốt hơn direct judgement | Ủng hộ mạnh trên controlled language | Decomposed compile đạt khoảng 0,96–0,98; cần held-out policy source để loại prompt overfitting |
| H12: semantic contract tự nhiên/nonlocal cần FM | Chưa thử | DSL hiện chỉ conjunction of row predicates |
| H13: FM không cần ở mỗi inference step | Ủng hộ mạnh | Compile-once + typed condition tốt hơn repeated step calls |
| H14: active-set benefit chuyển qua dataset khác | Đang chạy/thiết kế | Adult support; cần Default/Shoppers applicability contracts |
| H15: effect giữ ở 4.096 samples/contract | Ủng hộ | Default/Shoppers large evaluation gần run 512 mẫu |
| H16: learned condition trước exact projection giảm correction burden | Ủng hộ mạnh ở CFG 1 và CFG 2 | Cùng validity 1,0 nhưng edit/distortion/fidelity tốt hơn unconditional projection trên cả hai dataset; CFG 1 tốt hơn CTDF-style numeric proxy hiện có trên metric đo được |
| H17: lợi thế chuyển sang mixed numeric–categorical contract | Ủng hộ trên hai dataset | CFG 1 tạo Pareto improvement trên categorical/mixed composition; request-ID không phản ứng; category values vẫn seen và catalog nhỏ |
| H18: compiler giữ độ chính xác khi đổi schema | Ủng hộ trên controlled diagnostic | 24 run, đa số 1,0; basic có catastrophic seed `0,5`, decomposed loại failure đó; chưa phải natural-policy/domain generalization |
| H19: differentiable guidance loại bỏ lợi ích của learned condition | Không được ủng hộ bởi proxy | Gradient tăng validity nhưng gây distortion/fidelity trade-off; faithful Harpoon vẫn cần để kết luận |

## 7. Phương pháp nên formalize

### 7.1 Contract representation và compiler

Contract `C` gồm schema-grounded predicates, mutable targets, immutable context, applicability, exception, hardness/chance level, action budget, confidence và provenance. Foundation model tạo một hoặc vài AST candidates; sau đó phải qua type checker, schema/category grounding, train-only counterexample search, contradiction check và calibration/abstention.

### 7.2 Contract-conditioned diffusion

Condition encoder nhận operator, column, value, conjunction và exception. Adapter/FiLM nhỏ condition TabDiff; condition dropout cho classifier-free guidance. Split bắt buộc: threshold, composition, exception, paraphrase và table/domain chưa thấy.

### 7.3 Active-set controller

`A_i(t) = Retrieve(C_i) ∩ Applicable(x̂₀,i,t) \ Exception(x̂₀,i,t)`.

So sánh no actuation, global condition, hard/soft posterior gate, calibrated posterior/top-k, shuffled evidence, paired preview-rerun, final repair, best-of-N và CTDF/Harpoon-style guidance. Actuator phải target-scoped để tránh antecedent evasion.

### 7.4 Verification transaction

Mỗi action cần proposal → type/range check → independent verifier → distribution/likelihood guard → accept/rollback/abstain. Với constraint executable, verifier không được chỉ là LLM. Với semantics không executable, cần annotation/human/tool ground truth độc lập.

## 8. Các giới hạn nghiêm trọng nhất

| Mức độ | Giới hạn | Vì sao reviewer sẽ phản biện | Cách khắc phục |
| --- | --- | --- | --- |
| Rất cao | Controlled contracts chưa phải natural semantics | Có thể coi là threshold toy task | Policy/ontology-derived requests, blind annotation, external sources |
| Rất cao | Chưa có faithful CTDF/Harpoon/SCTab-Diff/LCATS | Proxy không đủ cho SOTA | Port ít nhất CTDF + Harpoon-style; post-repair same compiler |
| Rất cao | Validity dùng chính executable contract | Metric có thể tautological | Held-out verifier, counterfactual tests, human/tool labels |
| Rất cao | “Semantic” hiện chủ yếu là policy do tác giả định nghĩa | Đúng DSL không đồng nghĩa đúng nghĩa trong domain | Policy/ontology ngoài, chuyên gia hoặc annotator mù, provenance rõ |
| Cao | Chỉ một backbone | Có thể là adapter-specific | Thêm TabSyn hoặc discrete/mixed backbone |
| Cao | Numeric threshold chiếm đa số | Chưa chứng minh mixed-type semantics | Categorical consequent, missingness, exception, aggregate |
| Cao | Adapter được train riêng cho từng bảng | Chưa phải foundation condition model chuyển bảng | Shared schema encoder, leave-one-table-out hoặc frozen FM encoder |
| Cao | Reference cohort lấy từ cùng empirical training table | Có thể thiên vị adapter và không đo generalization dữ liệu | Tách reference validation/test; bootstrap theo row và contract |
| Cao | Hybrid chắc chắn thắng validity vì có certifier | Validity 1,0 tự thân không phải đóng góp | Claim correction burden/fidelity/coverage; so same certifier cho mọi warm start |
| Cao | Fidelity trung bình có thể che lỗi đúng target column | Mean TV/KS pha loãng lỗi cột bị sửa | Báo target/non-target fidelity riêng, pairwise dependence và copula metrics |
| Cao | Compute matching còn xấp xỉ | “Passes” chưa chính xác | Log denoiser calls, wall time, GPU memory/energy, API cost |
| Cao | Utility/privacy chưa đủ | Thiếu tiêu chuẩn synthetic-data | TSTR/TRTS, subgroup utility, DCR, membership/attribute inference |
| Cao | Contract selection do tác giả | Cherry-picking risk | Pre-register catalog/splits; release candidates/rejection reasons |
| Trung bình | Contracts không phải iid | Bootstrap không phải population CI | Tăng families/datasets; hierarchical analysis |
| Trung bình | Preview-rerun 2× compute | Có thể thua repair về efficiency | Shared-prefix branching, cached latent, one-pass gate |
| Trung bình | Compiler variance | Một parse sai áp hàng loạt | Ensemble, equivalence tests, abstention, confirmation |
| Trung bình | Không có long text | Không cover AFT-Tab/TabDLM regime | Chỉ thêm khi dataset phù hợp |

## 9. Đánh giá state of the art và ICDE

### 9.1 Có phải state of the art chưa?

**Chưa.** Benchmark contract shift chưa chuẩn; chưa chạy faithful strongest baselines; evaluation còn controlled; thiếu utility/privacy và backbone replication. Tuy nhiên structured condition thắng request-ID/best-of-2 ở nhiều cặp và causal active-set controls tạo frontier rõ, đủ mạnh để tiếp tục.

### 9.2 Có form thành ICDE project được không?

**Có**, nếu đặt trọng tâm vào interface giữa natural-language data policy và generator, typed/auditable contracts, contract-shift benchmark, validity–fidelity–coverage–cost frontier và compiler/validator/generator separation. Framing “LLM feedback improves diffusion” yếu hơn nhiều so với “Open-contract conditional synthesis with verified selective activation”.

### 9.3 Ước lượng công việc

- 2–4 ngày: mixed-type aggregate, target/non-target metric audit và exact compute accounting;
- 1–2 tuần: faithful Harpoon và tốt nhất là CTDF hoặc một implementation bám sát toán học, cùng contract/backbone/certifier;
- 1–2 tuần: policy benchmark có nguồn độc lập, blinded semantic verification, exception/chance-constraint split;
- 1 tuần: TSTR/TRTS, subgroup/rare-mode coverage và privacy;
- 1–2 tuần: second backbone hoặc shared cross-table condition encoder;
- 1–2 tuần: ablation, hierarchical statistics, figures, writing và reproducibility.

Ước lượng: **4–6 tuần full-time** để có submission candidate; 6–8 tuần nếu tích hợp TabSyn và annotation ngoài.

## 10. Job ledger mới

| Jobs | Thí nghiệm | Trạng thái lúc viết | Quyết định |
| --- | --- | --- | --- |
| `1044251–1044263` | soft/global/shuffled/preview applicability | Hoàn tất | Row evidence có unique value |
| `1044267–1044279` | Default/Shoppers request-ID vs structured | Hoàn tất | Contract transfer qua dataset |
| `1044280–1044286` | best-of-N/final projection v1 | Hoàn tất; distortion invalid | Pareto baseline |
| `1044287–1044299` | calibrated soft/power/top-k/preview | Hoàn tất | Thu hẹp preview gap |
| `1044310–1044328` | 50-request language stress test | Hoàn tất | Compile-once vs direct |
| `1044332` | paired bootstrap v1 | Hoàn tất | Effect stability |
| `1044333–1044339` | 4.096 samples/held-out contract | Hoàn tất | Monte Carlo robustness được xác nhận |
| `1044340–1044346` | actuation baselines v2 | Hoàn tất | Distortion metric đã sửa |
| `1044347` | paired audit v2 | Hoàn tất | Claim table dùng baseline sạch |
| `1044348` | expanded API failure analysis | Thất bại do pandas column/method collision; output API gốc không ảnh hưởng |
| `1044356` | expanded API analysis retry | Hoàn tất | Paraphrase/status/batch/seed failures |
| `1044358` | compiler ensemble/abstention | Hoàn tất | Disagreement tạo selective accuracy–coverage frontier |
| `1044359–1044365` | decomposed compiler prompt, hai model × ba seed | Hoàn tất | Semantic role decomposition giải quyết phần lớn exception error |
| `1044366–1044372` | CTDF-style numeric per-step projection proxy | Hoàn tất | Feasibility và cohort fidelity tạo frontier rõ |
| `1044373–1044379` | structured condition → exact final projection | Thất bại sớm do thiếu numpy import; không có metric |
| `1044380–1044386` | hybrid retry sau one-line import fix | Hoàn tất | Validity 1,0 với correction burden giảm mạnh |
| `1044387–1044393` | conditional-only CFG 1 → final projection | Hoàn tất | Hybrid compute-matched tốt hơn proxy hiện có trên mọi metric đo được |
| `1044395–1044401` | CTDF-style proxy + exact raw-space certification | Hoàn tất | Loại bỏ bất công do inverse transform; kết luận fidelity frontier không đổi |
| `1044405–1044417` | mixed numeric/categorical contracts, hai representation × ba seed | Hoàn tất | Gain chuyển sang unseen mixed-type composition; request-ID không phản ứng |
| `1044419–1044443` | compiler schema-transfer, hai model × hai prompt × ba seed × hai bảng | Hoàn tất | Đa số rất tốt nhưng basic có catastrophic seed; decomposed tăng robustness |
| `1044444–1044449`, `1044465` | mixed-type same-certifier unconditional control + aggregate retry | Hoàn tất | Hybrid giảm edit/distortion rõ, mạnh nhất trên Shoppers mixed composition |
| `1044450` | same-certifier aggregate đầu | Thất bại analysis-only do optional metric columns mới | Worker outputs nguyên vẹn; retry `1044465` hoàn tất |
| `1044451–1044463` | target/non-target KS/TV decomposition | Hoàn tất | Gain tồn tại ở cả target và non-target; mean metric không che spillover xấu |
| `1044467` | paired mixed-type bootstrap | Hoàn tất | Shoppers mixed 9/9 Pareto; Default mixed TV chưa chắc chắn |
| `1044469–1044475` | Harpoon-style differentiable guidance proxy, eta sweep | Hoàn tất | Gradient validity gain đi kèm distortion lớn; tạo frontier chứ không dominate hybrid |

## 11. Thứ tự ưu tiên tiếp theo

1. Khóa bảng mixed-type ba seed và tách fidelity target/non-target.
2. Dùng cùng certifier cho unconditional, best-of-N, structured condition, Harpoon và CTDF; claim chính là correction burden/fidelity/cost chứ không phải zero violation.
3. Port faithful Harpoon; với CTDF, hoặc bám sát feasibility maps trong paper, hoặc giữ nhãn proxy và không dùng để claim state of the art.
4. Xây policy set có nguồn độc lập, pre-register catalog/split/rejection criteria và đánh giá mù.
5. Thêm exception/chance constraints trên Default/Shoppers; chạy preview-rerun và shuffled control.
6. Đo exact denoiser calls, runtime, GPU memory, API token/cost; thêm TSTR/subgroup utility/privacy.
7. Thêm second backbone hoặc shared cross-table encoder. Chỉ sau đó mới thử LLM gọi giữa trajectory nếu compiler + condition + certifier vẫn có failure không giải được.

## 12. Claim đề xuất cho bản thảo sớm

> A typed, parameterized contract interface enables a pretrained mixed-type diffusion model to generalize to unseen threshold values and compositions, creating condition-consistent cohorts with a favorable validity–fidelity–cost trade-off against matched-budget selection and final projection. For row-conditional contracts, row-aligned applicability evidence is necessary: global early conditioning improves validity but distorts inactive rows, whereas selective preview-rerun preserves them. A foundation model is most effective as a compile-once contract parser, not as a repeated per-row judge.

Không nên claim: first LLM feedback in diffusion; first constraint-guided tabular diffusion; first LLM-to-executable compiler; universal superiority over repair/projection/best-of-N; state of the art trước faithful baselines; hoặc semantic correctness từ empirical correlations.

## 13. Hình thức project nên chốt ở thời điểm hiện tại

### 13.1 Claim trung tâm khả thi nhất

> Với một policy mới có thể biên dịch thành typed contract, contract-conditioned diffusion học cách dịch chuyển toàn bộ cohort về phân phối phù hợp trước khi một certifier rẻ bảo đảm feasibility phần dư. Cách phân tách `compile → generate → certify` giảm lượng sửa và biến dạng so với certifying một mẫu unconditional, đồng thời tổng quát hóa qua parameter/composition chưa thấy tốt hơn request-ID conditioning.

Claim này có ba ưu điểm: không tranh “first guidance”; dùng được cả khi LLM không ở inference loop; và biến phép chiếu từ đối thủ thành thành phần chứng nhận chung. Nó cũng có điều kiện bác bỏ rõ: nếu cùng certifier trên faithful CTDF/Harpoon hoặc best-of-N đạt frontier tương đương, phần learned condition không đủ đóng góp.

### 13.2 Vai trò hợp lý của foundation model

Foundation model không nên được mô tả như oracle đánh giá từng row. Vai trò có bằng chứng hơn là:

1. dịch yêu cầu tự nhiên sang contract có kiểu;
2. grounding tên cột/category và tách scope, exception, requirement, allowed repair;
3. tạo nhiều parse khi bất định và abstain nếu không kiểm chứng được;
4. có thể mã hóa mô tả schema để condition encoder chuyển bảng trong phiên bản sau.

Executor, type checker, certifier và metric phải deterministic khi semantics đã executable. Direct per-row LLM chỉ còn hợp lý cho quan hệ thật sự không thể biểu diễn, và khi đó cần ground truth người/tool độc lập.

### 13.3 “Dynamic semantic constraints” phải được định nghĩa hẹp

Không dùng từ “dynamic” một cách chung chung. Project hiện có ba trục khác nhau:

- **query-dynamic:** policy/parameter/composition chỉ xuất hiện lúc yêu cầu sinh;
- **row-dynamic:** applicability và exception khác nhau theo row;
- **trajectory-uncertain:** khả năng xác định active set thay đổi theo độ ổn định của `x̂₀(t)`.

Kết quả mạnh nhất hiện ở query-dynamic; row-dynamic có causal pilot tốt nhưng mới trên Adult; trajectory-uncertain giải thích timing/gating nhưng chưa nên là claim chính. Nếu paper cố gom cả ba thành một phương pháp lớn, contribution sẽ bị loãng và baseline burden tăng quá cao.

### 13.4 Ba gói đề tài thu hẹp

| Gói | Trọng tâm | Bằng chứng hiện có | Điều còn thiếu quyết định |
| --- | --- | --- | --- |
| A — Open-contract cohort synthesis | Contract mới → conditional cohort → residual certification | Mạnh nhất; ba dataset numeric, large sample, hybrid CFG 1/2 | Mixed-type aggregate, faithful baselines, utility/privacy, policy nguồn độc lập |
| B — Exception-aware selective actuation | Active set theo row, bảo toàn inactive/rare modes | Causal controls và shuffled gate tốt | Chuyển qua Default/Shoppers, semantic exceptions, compute-efficient one-pass |
| C — Verified FM compiler | NL policy → typed contract với uncertainty/abstention | Compile tốt hơn direct; schema-transfer đang chạy | Natural policy corpus, grammar/smaller-model baseline, human semantic labels |

Khuyến nghị: dùng A làm paper chính; B là method extension hoặc section quan trọng nếu chuyển dataset thành công; C là interface/evaluation component, không đứng một mình vì Zhao/EVS đã rất gần.

## 14. Đánh giá nghiêm khắc về mức độ sẵn sàng

- **Có đủ tín hiệu để đầu tư thành project ICDE:** có hypothesis có thể bác bỏ, nhiều đối chứng cùng seed/backbone, failure mode rõ, và method candidate có frontier đẹp.
- **Chưa đủ để gọi state of the art:** chưa có faithful strongest baselines, natural semantic benchmark, second backbone, utility/privacy và cross-table model.
- **Mức sẵn sàng hiện tại:** exploratory evidence mạnh / submission evidence trung bình. Nếu khóa mixed-type, faithful baseline và external-policy evaluation trong 2–3 tuần tới mà frontier vẫn giữ, xác suất thành submission nghiêm túc tăng đáng kể.
- **Rủi ro lớn nhất không phải model performance mà là problem validity:** reviewer có thể nói đây chỉ là conditional sampling theo predicates do tác giả tự chọn. Do đó một policy benchmark có provenance và đánh giá semantic độc lập có giá trị thông tin cao hơn thêm nhiều seed trên cùng catalog.
