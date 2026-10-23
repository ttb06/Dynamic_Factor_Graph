# Báo cáo hướng nghiên cứu: LLM-derived dynamic constraint graph cho tabular diffusion

**Ngày cập nhật:** 07-10-2026  
**Backbone hiện tại:** TabDiff, checkpoint Shoppers epoch 8000  
**Trạng thái:** đã có pipeline knowledge graph, trajectory probe và active-edge guidance chạy end-to-end; claim về ưu thế so với post-repair chưa được xác lập.

## 1. Bài toán đang tập trung

Mục tiêu hiện tại không còn là “dùng LLM kiểm tra từng row” đơn thuần. Bài toán được phát biểu như sau:

> Từ schema, metadata, external knowledge và aggregate evidence trên train, xây dựng một typed constraint graph đã được kiểm chứng. Trong reverse diffusion, tại mỗi timestep và mỗi row, chỉ kích hoạt những edge có applicability đủ ổn định và chọn action thích hợp để điều khiển các bước còn lại, đồng thời giữ calibrated exception mass và fidelity của phân phối.

Một edge trong graph gồm:

- `applicability`: điều kiện để edge có hiệu lực;
- `requirements`: quan hệ phải được thỏa;
- `exceptions`: trạng thái làm edge tạm deactive;
- hardness: `hard`, `near_hard` hoặc `soft`;
- semantic scope: `domain_general`, `dataset_semantic` hoặc `dataset_empirical`;
- provenance: predefined rule, external knowledge, empirical proposal, critic, judge và refinement round.

Dynamic graph có ba tầng thay đổi:

1. **Theo row:** các row khác nhau có active set khác nhau do applicability và exception.
2. **Theo timestep:** một edge chỉ được sử dụng khi các cột liên quan trong \(\hat{x}_0(x_t)\) đã đủ readable và ổn định.
3. **Theo action:** cùng một violation có thể có nhiều cách sửa; action layer cần quyết định endpoint nào nên thay đổi thay vì luôn sửa consequent.

## 2. Motivation và research gap

### 2.1. Vì sao static constraint graph chưa đủ

Một static graph giả định tất cả edge có thể được áp dụng ở mọi state. Giả định này không đúng với diffusion:

- categorical state và numeric state hội tụ với tốc độ khác nhau;
- applicability có thể flip ở early steps;
- một near-hard edge không nên bị ép về zero violation;
- một violation có thể do nhiều endpoint khác nhau bị sai;
- can thiệp sớm vào một edge chưa readable gây false activation và distribution drift.

### 2.2. Khoảng trống so với related work

| Work | Đã giải quyết | Khoảng trống còn lại |
| --- | --- | --- |
| TabDiff | Mixed-type diffusion với feature-wise noise schedules | Không có semantic graph hay dynamic activation |
| TabSyn | Latent diffusion cho mixed-type table | Không giải quyết constraint discovery hoặc state-dependent graph |
| Harpoon | Differentiable constraint loss trong sampling | Constraint request cố định; gradient thay đổi theo state nhưng không có LLM-derived conditional graph và action ambiguity |
| CTDF | Training-free feasibility operator tại mỗi reverse step | Đã phủ per-step hard constraint enforcement; chưa giải quyết uncertain semantic edge, dynamic applicability và calibrated abstention |
| SCTab-Diff | LLM prior condition reverse denoising | Prior được xác lập trước sampling; không phải per-edge lifecycle theo readability và uncertainty |
| LCATS | Discover/verify constraint rồi post-generation diagnosis/repair | Graph được audit có 14 constraints nhưng 12 unary bounds, chỉ 2 multi-column và 1 conditional; không có trajectory-dependent active graph |
| SAGE | Value-dependent dependency khi LLM sinh bảng | Dependency thay đổi theo row/value, nhưng không nghiên cứu lifecycle của edge dọc diffusion trajectory |
| PromptLoop | Denoised preview và feedback loop cho image diffusion | Chứng minh pattern ngoài tabular; không xử lý typed mixed-column graph, exception mass và tabular repair ambiguity |

Do đó không nên claim “LLM feedback trong diffusion” hoặc “constraint tại mỗi reverse step” là mới. Claim tiềm năng hẹp hơn nhưng defendable hơn là:

> **Verified typed hypergraph với active set thay đổi theo row và denoising state, kèm uncertainty calibration và action routing cho mixed-type tabular diffusion.**

## 3. Pipeline hiện tại

### 3.1. Knowledge acquisition

Pipeline hiện có bốn nguồn:

1. predefined hypotheses có provenance;
2. external/domain proposal từ LLM;
3. empirical structure được mine chỉ trên discovery split;
4. directionality/exception proposal từ LLM.

Train được chia thành:

- 60% discovery;
- 20% counterexample;
- 20% blind calibration;
- test chỉ mở sau khi graph đã frozen.

LLM chỉ nhận schema, glossary, frequent categories và aggregate statistics. Raw real rows không được gửi ra API.

### 3.2. Propose–execute–refine loop

1. Các role đề xuất typed hyperedges độc lập.
2. Deterministic compiler kiểm tra column name, type, operator, target/sensitive leakage, tautology, support và vacuity.
3. Executor đo support/violation trên discovery và counterexample split.
4. Aggregate counterexamples được đưa vào refinement round; không gửi row.
5. Adversarial critic thực hiện `keep/revise/reject`.
6. Semantic judge gán scope và kiểm tra hardness/directionality.
7. Blind calibration quyết định `hard/near_hard/soft`.
8. Graph được frozen trước khi đánh giá heldout.

### 3.3. Active graph theo trajectory

Với mỗi edge \(e\), trajectory probe học cutoff \(\tau_e\): timestep sớm nhất mà active/violation state của \(\hat{x}_0(x_t)\) đạt precision và recall yêu cầu so với trạng thái cuối.

Tại timestep \(t\), edge chỉ active nếu:

- \(t \leq \tau_e\);
- applicability đúng ở state hiện tại;
- applicability tồn tại qua hai state liên tiếp;
- các cột liên quan không thay đổi quá ngưỡng stability;
- support đủ lớn;
- row không nằm trong exception group.

Với near-hard edge, hệ thống chỉ sửa phần violation vượt quá calibrated violation mass thay vì ép toàn bộ về zero.

### 3.4. Action layer hiện tại

- Literal numeric/categorical requirement: project target về giá trị hợp lệ.
- Column comparison: thử hai endpoint và chọn projected edit nhỏ hơn trong encoded space.
- Applicability column được bảo vệ khi có thể, nhằm tránh thỏa rule bằng cách làm antecedent false.
- Conflicting actions trên cùng endpoint bị chặn trong một timestep.

Đây mới là typed minimal-projection actuator. Nó chưa phải learned action policy và chưa tận dụng đầy đủ multi-column context.

## 4. Hypothesis và trạng thái bằng chứng

| Mã | Hypothesis | Trạng thái hiện tại | Bằng chứng |
| --- | --- | --- | --- |
| H1 | Edge state không readable ở early steps nhưng đủ ổn định ở middle/late steps | **Được ủng hộ** | Precision/recall tăng mạnh theo timestep; typed edges có cutoff 0.286–0.612 |
| H2 | Edge-specific schedule hợp lý hơn một global activation window | **Được ủng hộ** | Ba typed edges có cutoff khác nhau; process edges readable sớm hơn column comparison |
| H3 | State/persistence gate giảm can thiệp thừa so với static edge schedule | **Được ủng hộ** | Giảm 19.7% action events, 25.4% affected rows và 55.6% drift với violation tương đương |
| H4 | Calibration cần thiết cho near-hard edge | **Được ủng hộ** | Uncalibrated methods ép `BounceRates <= ExitRates` gần zero, lệch khỏi real violation mass |
| H5 | Mid-step guidance tốt hơn calibrated final repair | **Chưa được ủng hộ** | Calibrated final repair đang có heldout error và drift tốt hơn calibrated state guidance |
| H6 | Action routing cần context vì cùng violation có nhiều cách sửa | **Được ủng hộ trong controlled setting** | Context router đạt khoảng 0.73–0.74 route accuracy so với 0.50 của fixed action |
| H7 | LLM knowledge acquisition thêm active edges tốt hơn rule library | **Chưa đủ bằng chứng** | Ba active edges hiện đều bắt nguồn từ predefined seeds; LLM-discovered edges chủ yếu chỉ phù hợp verifier |
| H8 | LLM pipeline có thể sinh complex hyperedges ngoài predefined seeds | **Được ủng hộ về structural coverage** | 4/7 retained edges là complex, non-seed và heldout-stable; semantic/action quality vẫn cần audit |
| H9 | Sửa predicted clean sample tốt hơn project trực tiếp noisy state | **Bị bác bỏ trong run tự do hiện tại** | `x0` guidance có violation 0.0043 so với 0.0029 của direct noisy-state projection và lặp action nhiều hơn |
| H10 | Row-wise branch–score–select tốt hơn một fixed action | **Được ủng hộ trong cùng x0 mechanism, nhưng chưa thắng final repair** | Violation giảm 0.0043→0.0035, heldout error 0.0009→0.0001 và action 144→73; matched final repair vẫn đạt 0.0029 |
| H11 | Feedback lặp lại là nguyên nhân chính làm x0 guidance kém hiệu quả | **Bị bác bỏ** | One-shot giảm action nhưng làm violation xấu hơn (0.0049); cooldown phục hồi còn 0.0042 nhưng vẫn kém final repair 0.0029 |
| H12 | Selective rollback/resampling tốt hơn deterministic correction | **Bị bác bỏ với local rollback hiện tại** | One-shot/cooldown rollback có violation 0.0058/0.0060, đều kém x0 reward 0.0048/0.0042 và final reward 0.0029 |
| H13 | Multi-branch lookahead tạo giá trị ngoài một action đơn | **Được ủng hộ nội bộ, chưa thắng final repair** | Branch reward/selector 0.0048 tốt hơn directed 0.0059 và rollback 0.0062; best-of-2 đạt 0.0017 nhưng phá fidelity; final reward vẫn có trade-off tốt nhất |
| H14 | Lookahead ở middle stage có ích trên các edges đã readable | **Được ủng hộ yếu về effect, không về superiority** | t=0.286 giảm macro violation 0.0065→0.0055 với 20 rows, nhưng final repair vẫn đạt 0.0029 |

## 5. Kết quả thực nghiệm đã có

### 5.1. Pipeline LLM knowledge graph

Budgeted Shoppers run:

- 15 API calls;
- 110,513/165,000 tokens;
- model: `gpt-4.1-mini` cho empirical proposal/critic, `gpt-5.4` cho external proposal/judge;
- pipeline có checkpoint/resume và structured-output error handling.

| Graph | Proposed | Retained | Hard | Near-hard | Families | External/hybrid | Mean heldout shift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Judged single-pass | 14 | 1 | 0 | 1 | 1 | 1 | 0.0017 |
| Multi-stage LLM | 46 | 8 | 5 | 3 | 3 | 2 | 0.0013 |
| Empirical-only | 30 | 28 | 5 | 9 | 1 | 0 | 0.0117 |
| Unified | 76 | 36 | 10 | 12 | 4 | 2 | 0.0094 |

Không được diễn giải bảng này là semantic precision: single-pass compiler chỉ tạo được 1/14 edge structurally valid, nên comparison chưa đủ công bằng để claim multi-stage thắng strong single-pass.

### 5.2. Từ source graph đến active graph

| Stage | Số edge |
| --- | ---: |
| Source LLM graph | 8 |
| Verifier graph | 7 |
| Actuation-safe graph | 3 |

Các edge bị loại gồm duplicate, threshold bị subsume, reverse measurement action có thể fabricate count, và association ổn định nhưng không nên được coi là constraint.

### 5.3. Edge readability theo timestep

| Edge | Class | Cutoff | Final violations | Synthetic / heldout violation |
| --- | --- | ---: | ---: | ---: |
| `Administrative=0 -> Administrative_Duration=0` | Hard | 0.571 | 3 | 0.0031 / 0.0000 |
| `Informational=0 -> Informational_Duration=0` | Hard | 0.612 | 7 | 0.0043 / 0.0000 |
| `BounceRates <= ExitRates` | Near-hard | 0.286 | 45 | 0.0220 / 0.0154 |

Cutoff là reverse-time lớn nhất đạt precision ≥ 0.90 và recall ≥ 0.80. Hai process edges có thể bật ở nửa giữa sớm; column comparison chỉ nên bật muộn hơn.

### 5.4. Active-edge guidance

Job `1056090`, 4,096 samples, 50 steps, paired initial noise:

| Method | Macro violation ↓ | Heldout error ↓ | Numeric KS ↓ | Drift SD ↓ | Action events ↓ | Unique rows ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unconditional | 0.0091 | 0.0042 | **0.0259** | **0.0000** | **0** | **0** |
| Fixed middle | **0.0011** | 0.0046 | 0.0263 | 0.0028 | 377 | 120 |
| Edge schedule | 0.0013 | 0.0044 | 0.0266 | 0.0036 | 279 | 126 |
| State-active | 0.0013 | 0.0047 | 0.0262 | 0.0016 | 224 | 94 |
| Calibrated state-active | 0.0055 | 0.0006 | **0.0262** | 0.0006 | 29 | 29 |
| Final repair | **0.0000** | 0.0057 | 0.0262 | 0.0012 | 93 | 93 |
| Calibrated final repair | 0.0053 | **0.0004** | 0.0263 | **0.0003** | **28** | **28** |

Kết luận đúng của bảng:

- state gate tốt hơn static edge schedule về intervention efficiency;
- calibration quan trọng hơn việc đạt zero violation;
- middle-step guidance chưa thắng calibrated final repair trong graph có action quá đơn giản.

### 5.5. Action ambiguity

Controlled cases tạo cùng quan sát `count=0, duration>0` theo hai route: duration bị corrupt hoặc count bị corrupt. Cả hai fixed repairs đều tạo row hợp lệ, nhưng chỉ một repair phục hồi đúng clean record.

| Method | Route accuracy | Validity | Pair error | Wrong-column edit |
| --- | ---: | ---: | ---: | ---: |
| Fixed consequent | 0.5000 | 1.0000 | 1.4003 | 0.5000 |
| Minimal edit | 0.5000 | 1.0000 | 1.7158 | 0.5000 |
| Context router | 0.7345 | 1.0000 | 0.9996 | 0.2655 |
| Oracle | 1.0000 | 1.0000 | 0.4776 | 0.0000 |

Diffusion-imputation experiment cũng cho context router khoảng 0.7428 route accuracy và pair error 0.9586 sau certification, tốt hơn fixed route. Đây là motivation cụ thể cho action routing, nhưng hiện vẫn là controlled corruption chứ chưa phải unconditional generation.

Run mới `1056391` bổ sung hai baseline khó hơn:

| Method | Route accuracy | Validity sau certify | Pair error |
| --- | ---: | ---: | ---: |
| Fixed duration | 0.5000 | 1.0000 | 1.4037 |
| Fixed count | 0.5000 | 1.0000 | 1.5948 |
| Cross-relation router | 0.6891 | 1.0000 | 1.1054 |
| Natural-state router | 0.7084 | 1.0000 | 1.0604 |
| Context router | **0.7387** | 1.0000 | **0.9534** |
| Oracle | 1.0000 | 1.0000 | 0.4347 |

Cross-relation generalization thấp hơn within-relation context router nhưng vẫn tốt hơn fixed action. Khoảng cách lớn tới oracle cho thấy action selection là một bài toán thực, không được giải quyết chỉ bằng normalized minimal edit.

### 5.6. Complex LLM graph

Complex-profile run đã hoàn tất sau khi resume response đầu bị cắt:

- 16 API calls, tổng 138,596 tokens;
- 52 multi-role proposals, 7 edges được giữ;
- 4/7 retained edges thỏa định nghĩa complex;
- cả 4 complex edges đều không phải predefined seed và có heldout shift ≤ 0.05;
- 2 edges có arity ≥ 3, một conjunction, một multi-requirement edge và bốn edges có explicit exception;
- LCATS graph được audit có 14 constraints nhưng chỉ 2 multi-column và 1 explicit conditional edge.

| Edge | Class | Arity | Requirements | Exception | Shift | Ý nghĩa mechanism |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `r1_005` | Near-hard process | 4 | 3 | 1 | 0.0000 | Một violation có thể cần sửa ProductRelated, duration, BounceRates hoặc deactivate PageValues |
| `r1_009` | Near-hard process | 2 | 1 | 1 | 0.0000 | Count–duration consistency với explicit anomaly exception |
| `r1_016` | Soft process | 3 | 1 | 1 | 0.0129 | Conjunction trên VisitorType và ExitRates; chỉ dùng verifier, không actuation hiện tại |
| `r2_022` | Soft temporal | 2 | 1 | 1 | 0.0003 | Stable association `SpecialDay → Month`; không đủ semantic basis để guidance |

Kết quả này đạt structural objective nhưng chưa phải semantic precision. Đặc biệt `r1_005` chỉ có judge confidence 0.68 và cần manual/domain review trước paper run. Mechanism experiment chỉ giữ hard/near-hard edges có calibration support ≥100; soft temporal associations bị loại.

## 6. Jobs mới đã submit

| Job | Mục đích | Budget/setting | Trạng thái lúc submit |
| --- | --- | --- | --- |
| `1056389` + `1056392` | Complex LLM graph protocol trên Shoppers | 138,596 tokens, 2 refinement rounds | Completed |
| `1056393` | Audit complex graph và so với LCATS graph structure | 4 complex non-seed retained edges | Completed |
| `1056391` | Diffusion action routing với natural-state và cross-relation baselines | GPU, một seed | Completed |
| `1056394` | Curate mechanism graph | 5 edges được chọn, gồm 2 complex non-seed edges | Completed |
| `1056395` | Học edge-specific cutoff cho mechanism graph | Cả 5 edges readable, cutoff 0.245–0.571 | Completed |
| `1056396`, `1056397` | First row-wise reverse action attempt | Categorical indices được truyền nhầm vào one-hot preview helper | Failed sớm; không phải experimental result |
| `1056398`, `1056399` | Second action attempt | Phát hiện no-action decode–encode round-trip làm thay đổi toàn bộ numeric preview | Completed nhưng invalid; không dùng làm kết quả |
| `1056401`, `1056402` | Third action attempt | Logic sửa cell đã đúng nhưng sampler thủ công chưa tiêu thụ cùng random draws như official sampler | Completed nhưng invalid; không dùng làm kết quả |
| `1056410` | Row-wise reverse action, balanced reward | Exact no-action parity; 2,048 samples | Completed, valid |
| `1056411` | Row-wise reverse action, edit-conservative | Exact no-action parity; 2,048 samples | Completed, valid; aggregate trùng run balanced |
| `1056413` | One-shot trajectory action | Mỗi row-edge chỉ được can thiệp một lần; 4,096 samples | Completed, valid |
| `1056414` | Cooldown trajectory action | Tối đa ba action, cách nhau ít nhất bốn reverse steps; 4,096 samples | Completed, valid |
| `1056415` | Large-batch trajectory probe | 8,192 samples để giảm bất định của cutoff | Completed; phát hiện cutoff small-batch không ổn định |
| `1056416` | One-shot với cutoff large-batch | Kiểm tra độ nhạy với schedule; 4,096 samples | Completed, valid |
| `1056417` | Cooldown với cutoff large-batch | Kiểm tra kết hợp budget và robust schedule; 4,096 samples | Completed, valid |
| `1056418` | One-shot selective rollback | Re-noise/mask riêng các cột của persistent violation rồi để TabDiff tái tạo; 4,096 samples | Completed, valid; không vượt direct correction |
| `1056419` | Cooldown selective rollback | Tối đa ba rollback mỗi row-edge, cách nhau bốn step; 4,096 samples | Completed, valid; lặp rollback làm kết quả xấu hơn |
| `1056525` | Multi-branch lookahead smoke | Bốn branches từ cùng state tại t=0.184; 512 samples | Completed, exact parity |
| `1056526` | Multi-branch lookahead full | Cùng mechanism, 4,096 samples, khoảng 1.6 equivalent denoiser passes | Completed, valid; chưa có compute-matched best-of-2 |
| `1056527` | Lookahead + best-of-2 | Giữ nguyên method/seed, thêm two-pass best-of baseline | Completed, valid |
| `1056528` | Middle-stage lookahead | Fork tại t=0.286; unreadable hard edges tự bị loại bởi schedule | Completed, valid |

Complex proposal profile ưu tiên:

- multi-antecedent conditional dependencies;
- multiple requirements;
- contextual column comparisons;
- explicit exception/deactivation groups;
- process/accounting/temporal hyperedges có từ ba cột;
- edge có nhiều plausible repair endpoints.

Candidate vẫn phải qua typecheck, train-only execution, counterexample refinement, critic, semantic judge và blind calibration. Việc thêm nhiều atom không tự động được xem là tốt: prompt yêu cầu bỏ atom nếu nó không thực sự cần thiết.

### 6.1. Mechanism branch mới

Mechanism graph dự kiến có 5 edges: ba seed edges đã kiểm chứng và hai complex process edges `r1_005`, `r1_009`. Hai soft edges không được actuation.

Trajectory probe `1056395`:

| Edge | Cutoff | Final support | Final violations | Synthetic / heldout violation |
| --- | ---: | ---: | ---: | ---: |
| `seed_000` | 0.571 | 951 | 4 | 0.0042 / 0.0000 |
| `seed_001` | 0.245 | 1,609 | 13 | 0.0081 / 0.0000 |
| `seed_003` | 0.327 | 2,048 | 36 | 0.0176 / 0.0154 |
| `r1_009` | 0.388 | 254 | 0 | 0.0000 / 0.0000 |
| `r1_005` | 0.551 | 465 | 0 | 0.0000 / 0.0000 |

Cả hai complex edges đều có applicability readable từ middle steps. Tuy nhiên `seed_001` có cutoff 0.245 trong run này, khác run ba-edge trước đó vì số final violations tăng từ 7 lên 13: run trước thiếu support để đánh giá violation precision/recall và chỉ dùng applicability. Đây là dấu hiệu cutoff estimator có sampling uncertainty; bảng paper cần nhiều trajectory batches hoặc confidence interval cho \(\tau_e\).

Mỗi guided sampler dùng cùng graph, cutoff và initial noise nhưng khác injection operator:

1. **`xstate_directed_calibrated`:** baseline hiện tại, project trực tiếp noisy state sau reverse update.
2. **`x0_directed_calibrated`:** sửa predicted clean sample \(\hat{x}_0\), sau đó tính lại Euler direction \((x_t-\hat{x}_0')/\sigma_t\). Đây là cách đưa constraint vào denoiser direction thay vì overwrite state.
3. **`x0_reward_calibrated`:** với từng row/edge, tạo các action branch rồi chọn branch trước khi cập nhật reverse state.
4. **`final_directed_calibrated`:** cùng directed action nhưng chỉ dùng sau sampling.
5. **`final_reward_calibrated`:** cùng branch reward nhưng chỉ dùng sau sampling; đây là matched baseline quan trọng nhất.

Action candidates gồm:

- sửa tất cả violated requirements;
- đổi endpoint còn lại cho column comparison;
- deactivate một applicability atom;
- abstain.

Frozen black-box reward gồm:

- typed graph violation penalty;
- train-fitted kNN density trên numeric context;
- normalized edit cost;
- penalty nhỏ cho antecedent deactivation và abstention.

Đây là bản tabular của pattern branch–score–select từ video/image diffusion, nhưng không dùng image metric. Reward và action đều được tính riêng cho từng synthetic row. Không gọi LLM trong sampling loop; vai trò của LLM là sinh typed graph/action space trước sampling.

### 6.2. Kết quả row-wise action layer đã qua parity check

Hai control bắt buộc đã được thỏa trong jobs `1056410`–`1056411`:

- manual sampler không action tạo output giống official TabDiff với paired random stream;
- paired drift của control bằng đúng 0, vì vậy chênh lệch dưới đây đến từ action chứ không phải do viết lại sampler.

| Method | Macro violation ↓ | Heldout error ↓ | Calibration error ↓ | Numeric KS ↓ | Drift SD ↓ | Actions ↓ | Rows ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unconditional | 0.0060 | 0.0026 | 0.0031 | **0.0273** | **0.0000** | **0** | **0** |
| Manual no-action control | 0.0060 | 0.0026 | 0.0031 | **0.0273** | **0.0000** | **0** | **0** |
| Noisy-state directed | **0.0029** | 0.0005 | **0.0000** | 0.0275 | 0.0012 | 29 | 29 |
| Predicted-x0 directed | 0.0043 | 0.0009 | 0.0014 | 0.0275 | 0.0008 | 144 | 33 |
| Predicted-x0 branch reward | 0.0035 | **0.0001** | 0.0006 | 0.0275 | 0.0008 | 73 | 31 |
| Final directed repair | **0.0029** | 0.0005 | **0.0000** | 0.0276 | **0.0008** | **24** | **24** |
| Final branch reward | **0.0029** | 0.0005 | **0.0000** | 0.0276 | **0.0008** | **24** | **24** |

Diễn giải:

1. Branch reward có giá trị trong cùng predicted-x0 mechanism: nó dùng 73 thay vì 144 actions và giảm cả violation lẫn heldout error. Trong 73 action, 21 lần chọn endpoint thay thế thay vì luôn sửa consequent.
2. Sửa predicted x0 không tự động tốt hơn sửa noisy state. Action trên x0 có xu hướng bị denoiser kéo trở lại, tạo 131 lần sửa lặp chỉ trên edge `BounceRates <= ExitRates` ở fixed-action run.
3. Direct noisy-state guidance đạt đúng calibrated violation mass nhưng thay đổi 29 rows; final repair đạt cùng graph metrics với 24 rows và drift nhỏ hơn. Vì vậy **kết quả hiện tại chưa chứng minh middle-step injection có incremental value ngoài matched post-repair**.
4. Hai cấu hình reward (`density=1/edit=.5` và `density=.5/edit=1`) cho aggregate giống nhau. Action space hiện quá nhỏ để kết luận về cách đặt trọng số; không nên tuning thêm trọng số trước khi mở rộng mechanism.
5. Hai complex non-seed edges không có natural violation trong batch này. Chúng chứng minh graph có cấu trúc phức tạp và readable, nhưng chưa đóng góp vào actuation result.

One-shot/cooldown experiments kiểm tra một giải thích cụ thể cho thất bại của x0 injection: trajectory feedback có thể hữu ích nếu commit một action rồi để denoiser phối hợp các cột còn lại, thay vì ép lại cùng edge ở nhiều step. Kết quả bên dưới không ủng hộ giải thích này; hướng tiếp theo hợp lý là selective rollback/resampling hoặc training-time graph conditioning, không phải tăng feedback frequency.

### 6.3. Feedback budget và độ ổn định của activation cutoff

Jobs `1056413`–`1056417` dùng cùng 4,096 unconditional samples và paired random stream. So sánh trọng tâm:

| Schedule / budget | Method | Violation ↓ | Heldout error ↓ | Drift ↓ | Actions ↓ |
| --- | --- | ---: | ---: | ---: | ---: |
| Small-batch cutoff, one-shot | x0 reward | 0.0049 | 0.0015 | **0.0005** | **61** |
| Small-batch cutoff, cooldown | x0 reward | **0.0042** | **0.0008** | 0.0010 | 103 |
| Large-batch cutoff, one-shot | x0 reward | 0.0048 | 0.0014 | **0.0005** | **54** |
| Large-batch cutoff, cooldown | x0 reward | **0.0042** | **0.0008** | 0.0009 | 91 |
| Matched final action | final reward | **0.0029** | **0.0005** | 0.0011 | **50** |

Kết quả bác bỏ giải thích đơn giản rằng x0 mechanism thất bại chỉ do over-control. One-shot có drift thấp nhưng không giữ được correction; cooldown cải thiện validity bằng cách lặp action, song vẫn không vượt final action về validity, calibration hoặc số action. Large-batch schedule giảm 12 action ở cooldown mà giữ nguyên kết quả, nên activation timing giúp efficiency nhưng chưa tạo quality gain.

Cutoff cũng không nên được coi là một hằng số đã biết:

| Edge | 2,048 samples | 8,192 samples | Khoảng lệch |
| --- | ---: | ---: | ---: |
| `seed_000` | 0.571 | 0.184 | **0.388** |
| `seed_001` | 0.245 | 0.204 | 0.041 |
| `seed_003` | 0.327 | 0.286 | 0.041 |
| `r1_009` | 0.388 | 0.388 | 0.000 |
| `r1_005` | 0.551 | 0.551 | 0.000 |

`seed_000` chỉ có bốn final violations trong probe nhỏ nên violation state không đủ đánh giá; schedule khi đó dựa chủ yếu vào applicability và bật quá sớm. Với 8,192 samples có 17 violations, cutoff lùi về late trajectory. Đây không phải bằng chứng rằng mọi edge cần bật muộn, mà là bằng chứng cần confidence-aware cutoff: report số positive, confidence interval và abstain khi violation-readability chưa đủ support.

Một quan sát bổ sung về complex graph: `r1_005` không có final violation ở batch nhỏ nhưng vẫn gây 3–9 mid-trajectory actions. Phần lớn là transient violations tự biến mất. Complex edge vì vậy làm rõ lý do cần state persistence/certification, nhưng hiện chưa tạo natural setting nơi mid-step action thắng post-repair.

### 6.4. Selective rollback/resampling

Job `1056418` thay deterministic projection bằng một action gần với RePaint/PPAD hơn: khi một persistent violation được chứng nhận, chỉ các numeric columns liên quan được đưa về \(\hat{x}_0 + \sigma_{t-1}\epsilon\) và categorical columns được mask lại; các cột còn lại của row giữ nguyên. TabDiff sau đó tự hoàn thiện phần bị rollback.

| Method | Violation ↓ | Heldout error ↓ | Drift ↓ | Affected-row density delta ↓ | Actions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unconditional | 0.0065 | 0.0031 | **0.0000** | 0.0000 | 0 |
| One-shot x0 reward | 0.0048 | 0.0014 | 0.0005 | -0.7406 | 54 |
| One-shot selective rollback | 0.0058 | 0.0024 | 0.0005 | -0.3619 | 56 |
| Final reward | **0.0029** | **0.0005** | 0.0011 | **-1.5700** | **50** |

Naive selective rollback giữ global drift thấp nhưng không đủ mạnh để sửa semantic violation và còn cải thiện local density ít hơn direct x0 action. Cooldown rollback còn xấu hơn: violation 0.0060 với 93 actions, gần quay về unconditional 0.0065. Kết quả này không bác bỏ mọi rollback method; nó bác bỏ local-column re-noising không có lookahead selection. Nhánh resampling tiếp theo chỉ đáng làm dưới dạng multi-branch lookahead với final selector, không nên tiếp tục tăng rollback frequency.

### 6.5. Multi-branch lookahead

Job `1056526` tạo bốn branches từ đúng cùng state tại \(t=0.184\): no-action, directed x0 correction, reward-routed x0 correction và selective rollback. Mỗi branch chạy phần denoising còn lại bằng cùng random stream, sau đó selector chọn riêng cho từng row bằng frozen graph certifier, train-kNN density và edit cost. Chi phí lý thuyết khoảng 1.6 full denoiser passes/output vì chỉ fork 10/50 bước cuối.

| Method | Violation ↓ | Heldout error ↓ | Calibration error ↓ | KS ↓ | Drift ↓ | Actions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unconditional / branch-0 parity | 0.0065 | 0.0031 | 0.0036 | 0.0256 | **0.0000** | 0 |
| Directed branch | 0.0059 | 0.0025 | 0.0030 | **0.0255** | 0.0002 | 42 |
| Reward branch | **0.0048** | **0.0014** | **0.0019** | **0.0255** | 0.0008 | 43 |
| Rollback branch | 0.0062 | 0.0028 | 0.0033 | 0.0256 | **0.0002** | 43 |
| Mid-lookahead selector | **0.0048** | **0.0014** | **0.0019** | **0.0255** | 0.0008 | **42** |
| Best-of-2 full generation | **0.0017** | 0.0017 | 0.0012 | 0.1281 | 0.3700 | 2,016 selected rows |
| Final reward | **0.0029** | **0.0005** | **0.0000** | 0.0256 | 0.0011 | 50 |

Selector chọn 4,054 rows từ no-action, 21 directed, 14 reward và 7 rollback. Như vậy branch diversity có tín hiệu: selector không collapse hoàn toàn về một action, còn reward routing tốt rõ hơn directed và rollback. Best-of-2 dùng 2.0 full passes và giảm violation xuống 0.0017, nhưng chọn 2,016 rows từ một sample độc lập làm KS tăng 0.0256→0.1281 và paired drift lên 0.3700; đây là một ví dụ rõ về validity gain không đồng nghĩa với fidelity. Final repair vẫn cho heldout/calibration error tốt nhất với rất ít thay đổi, nên trajectory value chưa được chứng minh trên natural Shoppers setting.

Stage ablation `1056528` fork sớm hơn tại \(t=0.286\), chỉ cho phép `seed_003`, `r1_009` và `r1_005` actuation theo robust schedule. Selector dùng 20 rows, giảm overall violation 0.0065→0.0055 và drift chỉ 0.0004. Theo edge, `seed_003` giảm 95→90 violations và `r1_005` giảm 4→1; hai hard process edges giữ nguyên vì chưa readable. Đây là bằng chứng middle state có một effect đo được, nhưng effect nhỏ hơn late lookahead (0.0048) và final repair (0.0029). Do đó stage result hỗ trợ dynamic eligibility, chưa hỗ trợ claim middle-step superiority.

## 7. Hướng đi tiếp theo

### Nhánh A — Hoàn thiện complex knowledge graph

Điều kiện để đi tiếp sau job `1056389`:

1. Có ít nhất 3 non-seed complex edges sống qua semantic judge và calibration.
2. Có ít nhất một conjunction/contextual comparison hoặc exception-aware edge ổn định trên heldout.
3. Edge có semantic direction rõ, không chỉ là stable association.
4. Có action ambiguity thực sự: ít nhất hai plausible endpoints hoặc nhiều requirement cần phối hợp.

Nếu không đạt, không nên tăng thêm API calls trên cùng prompt. Cần bổ sung richer aggregate evidence hoặc dataset có process/accounting semantics rõ hơn.

### Nhánh B — Dynamic action routing

Nếu complex graph đạt điều kiện:

- xây candidate action set cho từng edge;
- dùng \(\hat{x}_0(x_t)\), edit cost và conditional likelihood để score action;
- chỉ commit action khi route confidence vượt threshold;
- cho phép abstain/rollback nếu active state hoặc preferred action flip;
- so sánh với fixed consequent repair, minimal final repair, calibrated final repair và oracle route.

Đây là nơi trajectory có khả năng tạo giá trị ngoài post-repair: state trung gian có thể cho biết endpoint nào đang hình thành ổn định trước khi row cuối trở thành một violation khó phân biệt nguyên nhân.

Kết quả mechanism hiện tại dẫn tới ba quyết định:

1. Giữ row-specific branch scoring: trong cùng x0 mechanism, nó giảm violation và giảm gần một nửa action so với fixed consequent.
2. Không tiếp tục tuning feedback frequency hoặc local rollback trên Shoppers: matched final repair vẫn tốt hơn, còn complex edges có quá ít natural violations để tạo phép thử phân biệt.
3. Phép thử tiếp theo phải tạo thông tin mới, theo một trong hai hướng:
   - multi-branch lookahead: từ cùng mid-state, chạy các action/rollback branches tới cuối rồi chọn bằng frozen certifier+density; so với compute-matched best-of-N và final branch selection;
   - graph-conditioned training/fine-tuning: đưa active-edge embedding và timestep reliability vào denoiser, sau đó kiểm tra heldout edges/cross-dataset transfer. Hướng này phù hợp nếu inference-time correction tiếp tục bị post-repair thống trị.

Để đánh giá middle-step claim trên natural generation, dataset tiếp theo phải có đủ support cho multi-column process violations. Không nên chọn thêm dataset chỉ vì phổ biến; cần audit trước để bảo đảm ít nhất vài trăm applicable rows và violation rate đủ đo nhưng không bão hòa.

### Nhánh C — Evaluation cần thiết trước claim paper

- Ít nhất 2–3 dataset có complex process/accounting relations.
- Tách calibration/test sạch; không dùng heldout-stability guard để chọn graph.
- Ba seeds cho bảng cuối, bootstrap confidence interval ở row level.
- Fidelity: numeric KS/Wasserstein, categorical TV, pairwise/joint distance và subgroup fidelity.
- Constraint metrics: per-edge support, violation, calibrated error, false activation và coverage.
- Intervention metrics: action events, unique affected rows, endpoint accuracy, rollback và runtime.
- Utility/privacy chỉ chạy sau khi mechanism vượt matched post-repair baseline.

## 8. Claim boundary hiện tại

Có thể claim ở mức exploratory:

1. Edge applicability/violation state có lifecycle đo được dọc tabular diffusion trajectory.
2. Temporal gate kết hợp row-specific state giảm can thiệp và drift so với can thiệp ở mọi bước; lợi ích riêng của edge-specific schedule chưa được chứng minh.
3. Calibrated activation tránh làm mất legitimate violation mass.
4. Action ambiguity tồn tại và context-aware routing tốt hơn fixed repair trong controlled cases.

Chưa thể claim:

- vượt SOTA synthetic tabular generation;
- LLM-discovered graph tốt hơn rule/empirical graph;
- mid-step injection tốt hơn calibrated post-repair;
- generalization cross-dataset;
- end-to-end action router tốt trên natural unconditional violations.

## 9. Artifacts chính

- Knowledge graph report: `KNOWLEDGE_GRAPH_PROTOCOL_20261005_VI.md`
- Budgeted graph: `outputs/shoppers_budgeted_graph_1056016/RESULTS.md`
- Typed graph trajectory: `outputs/typed_graph_trajectory_1056065/RESULTS.md`
- Dynamic guidance: `outputs/typed_dynamic_graph_guidance_1056090/RESULTS.md`
- Per-edge guidance metrics: `outputs/typed_dynamic_graph_guidance_1056090/edge_metrics.csv`
- Complex graph protocol: `scripts/knowledge_graph_protocol.py`
- Complex graph audit: `scripts/audit_complex_graph_protocol.py`
- Dynamic actuation: `scripts/typed_dynamic_graph_guidance.py`
- Row-wise mechanism graph: `scripts/curate_rowwise_mechanism_graph.py`
- Row-wise reverse action layer: `scripts/rowwise_reverse_action_guidance.py`
- Action ambiguity audit: `outputs/ambiguous_repair_1049914/RESULTS.md`
- Updated diffusion route experiment: `outputs/ambiguous_diffusion_route_1056391_seed20261006_r3/RESULTS.md`
- Complex graph result: `outputs/complex_graph_protocol_1056389/RESULTS.md`
- Complex graph audit: `outputs/complex_graph_audit_1056393/RESULTS.md`
- Parity-valid row-wise action result: `outputs/rowwise_reverse_action_1056410/RESULTS.md`
- Large-batch trajectory schedule: `outputs/rowwise_mechanism_trajectory_1056415/RESULTS.md`
- Full budget/cutoff/rollback ablation: `outputs/rowwise_mechanism_ablation_20261006_v2/RESULTS.md`
- Multi-branch lookahead with best-of-2: `outputs/rowwise_lookahead_1056527/RESULTS.md`

## 10. Audit mở rộng constraint catalog ngày 07-10-2026

### 10.1. Vì sao pipeline trước chỉ còn rất ít rules

Có hai thế hệ curation khác nhau, nên con số “3 rules” không mô tả toàn bộ pipeline hiện tại:

| Stage | Input | Output | Nguyên nhân giảm |
| --- | ---: | ---: | --- |
| Typed actuation cũ | 8 | 3 | confidence ≥ 0.90, heldout-shift ≤ 0.05, loại reverse measurement action và duplicate |
| Complex LLM protocol | 52 proposals | 7 | critic, semantic judge, calibration và executable DSL |
| Row-wise mechanism cũ | 7 | 5 nominal | chỉ giữ hard/near-hard, support ≥ 100, confidence ≥ 0.65 |
| Unified graph | 82 proposals | 34 | gồm 6 hard, 13 near-hard, 15 soft; trộn 27 empirical và 7 LLM edges |

Vấn đề implementation là row-wise curation vẫn đọc `llm_protocol_graph.json` gồm 7 edges, không đọc `unified_graph.json` gồm 34 edges. Tuy nhiên, đổi input rồi cho cả 34 edges quyền sửa row cũng không đúng: nhiều empirical edges chỉ là association hoặc chiều đảo của measurement relation. Ví dụ `Informational_Duration > 0 -> Informational > 0` hữu ích như certifier, nhưng sửa `Informational` dựa trên relation này có thể tự tạo page count.

Audit còn phát hiện `r1_009` có exception chứa trực tiếp trạng thái thất bại của requirement: requirement là `Informational_Duration > 0`, còn exception chứa `Informational_Duration == 0`. Vì vậy exception loại bỏ chính các violations trước khi đo, làm edge có vẻ đạt zero violation. Edge này đã bị loại khỏi action layer; con số 5 edges trước đây chỉ có 4 edges thực sự non-vacuous.

### 10.2. Thiết kế catalog mới

Thay vì hạ một threshold chung, mỗi edge được gán một `usage_role`:

- `actuate`: semantic hard/near-hard edge đã được review, có quyền đề xuất edit;
- `score_only`: empirical edge ổn định, chỉ dùng để rank lookahead branch và certify output;
- `monitor`: probabilistic relation, chỉ dùng để báo coverage/calibration, không tác động sampler.

Selection chỉ dùng discovery, counterexample và calibration partitions. Held-out không đi vào gate. Smoke audit trên frozen unified graph cho kết quả:

| Thành phần | Số cạnh |
| --- | ---: |
| Unified source | 34 |
| `actuate` | 4 |
| `score_only` | 8 |
| `monitor` | 5 |
| Tổng catalog | 17 |
| Cạnh tham gia branch decision | **12** |

Do đó graph được mở rộng từ 3 rules cũ lên 12 decision edges, nhưng chỉ 4 edges có quyền sửa cell. Đây là separation giữa “tri thức dùng đánh giá state” và “thẩm quyền actuation”, giúp dynamic graph rộng hơn mà không hard-code statistical associations thành deterministic repair.

### 10.3. Evaluation chung với baseline

Bảng baseline công bố và định nghĩa metric được tách riêng tại `SHOPPERS_BASELINES_AND_METRICS_20261007.md`. Suite local mới dùng cùng sample count và implementation cho CTGAN, TVAE, TabDiff, mid-lookahead, reverse-x0 guidance và post-repair. Ngoài các metric của TabDiff/TabSyn—Shape, Trend, α-Precision, β-Recall, Detection, MLE và DCR—suite bổ sung:

- violation theo từng role;
- sai số violation so với held-out và calibration;
- catalog edge coverage;
- active row-edge fraction.

Metric chính không nên là zero violation. Với near-hard relation, mục tiêu đúng là giảm **heldout violation error** trong khi giữ Trend, β-Recall và MLE.

### 10.4. Jobs đã submit

| Job | Nội dung | Dependency |
| --- | --- | --- |
| `1057680` | Curate role-separated catalog từ 34 edges | – |
| `1057681` | Trajectory readability cho 17-edge catalog, 8,192 samples | `1057680` |
| `1057682` | Mid-lookahead với 12 decision edges | `1057681` |
| `1057683` | Reverse-x0 action với 4 actuation edges | `1057681` |
| `1057684[0-1]` | CTGAN và TVAE từ SDV, 300 epochs | – |
| `1057685` | Unified baseline metric suite | method jobs và SDV baselines |

Kết quả sẽ xuất hiện lần lượt ở `outputs/expanded_constraint_catalog_1057680`, `outputs/expanded_catalog_trajectory_1057681`, `outputs/expanded_catalog_lookahead_1057682`, `outputs/expanded_catalog_reverse_action_1057683` và `outputs/shoppers_baseline_metric_suite_1057685`.

## 11. Kết quả expanded graph và điều chỉnh trọng tâm method

### 11.1. Expanded graph thực sự có lifecycle theo trajectory

Job `1057681` chạy 8,192 samples và cho thấy cả 17 edges đều có một cutoff readable, nhưng cutoff thay đổi mạnh từ 0.122 đến 0.551. Đây là kết quả quan trọng hơn việc tăng số rule đơn thuần: một thời điểm chung không phù hợp cho mọi edge.

| Khoảng timestep | Violation precision | Violation recall | Topology flip |
| --- | ---: | ---: | ---: |
| Early, `(0.75,1.0]` | 0.614 | **0.015** | 0.0028 |
| `(0.50,0.75]` | 0.565 | 0.167 | 0.0172 |
| Middle, `(0.25,0.50]` | 0.835 | 0.606 | 0.0030 |
| Late-middle, `(0.10,0.25]` | **0.975** | **0.903** | 0.0009 |
| Final, `[0,0.10]` | 0.997 | 0.982 | 0.0005 |

Early preview có precision trung bình không quá thấp nhưng recall chỉ 1.5%, nghĩa là hầu hết final violations chưa biểu hiện. Một static schedule gọi verifier hoặc LLM quá sớm sẽ chủ yếu quan sát false negatives. Middle state bắt đầu có tín hiệu, còn late-middle là vùng đầu tiên đủ đáng tin cho đa số edges.

Cutoff theo role cũng không đồng nhất:

| Role | Edges | Mean cutoff | Min–max cutoff | Final violations |
| --- | ---: | ---: | ---: | ---: |
| `actuate` | 4 | 0.306 | 0.184–0.551 | 252 |
| `score_only` | 8 | 0.278 | 0.122–0.531 | 851 |
| `monitor` | 5 | 0.290 | 0.143–0.429 | 885 |

Kết quả này chứng minh các edge có lifecycle quan sát khác nhau, nhưng tự nó chưa chứng minh edge-specific schedule tạo gain. Matched common-cutoff control ở mục 11.7 cho thấy trên Shoppers, một gate chung đơn giản hiện đạt kết quả tương đương.

### 11.2. Expanded graph guidance: gain có thật nhưng final repair vẫn mạnh hơn

Job `1057682` dùng 12 decision edges và fork tại `t=0.184`; job `1057683` dùng cùng catalog cho reverse-step correction. No-action parity vẫn chính xác.

| Method | Actuate violation ↓ | Actuate heldout error ↓ | Score-only violation ↓ | Numeric KS ↓ | Drift ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unconditional | 0.0070 | 0.0027 | 0.0589 | 0.0283 | **0.0000** |
| Mid-lookahead | 0.0059 | 0.0016 | 0.0575 | **0.0281** | 0.0007 |
| Reverse noisy-state | 0.0038 | 0.0007 | 0.0573 | 0.0282 | 0.0012 |
| Reverse predicted-x0 reward | 0.0052 | 0.0009 | 0.0575 | 0.0283 | 0.0007 |
| Final calibrated repair | **0.0036** | **0.0007** | **0.0573** | 0.0283 | 0.0010 |

Mid-lookahead giảm actuation violation khoảng 16% và actuation heldout error khoảng 41% với 34/4096 rows thay đổi. Tuy nhiên final repair giảm tương ứng khoảng 49% và 76% với 45 rows. Reverse noisy-state gần bằng final repair nhưng không tốt hơn một cách rõ ràng.

Score-only edges gần như không thay đổi vì action layer không có quyền trực tiếp sửa các statistical associations. Đây là hành vi đúng theo safety design, nhưng cũng cho thấy graph rộng hơn chủ yếu tăng khả năng diagnosis/certification chứ chưa tự động tăng effect size của guidance.

Best-of-2 tiếp tục là negative control quan trọng: raw violation giảm mạnh, nhưng numeric KS tăng 0.0283 → 0.1156 và drift tăng lên 0.342. Chọn sample chỉ theo constraint score phá fidelity, do đó raw satisfaction không thể là metric chính.

### 11.3. Trọng tâm method sau kết quả này

Method không nên tiếp tục mở rộng thành một hệ thống action routing nhiều tầng. Contribution nên giữ ở ba thành phần dễ mô tả và ablate:

1. Một verified typed constraint catalog được xây trước sampling.
2. Dynamic graph `G_i(t)` được tạo bằng row applicability và một temporal readability gate; edge-specific cutoff là biến thể ablation, chưa mặc định là thành phần bắt buộc.
3. Một intervention operator đơn giản để kiểm tra giá trị của dynamic activation, với post-repair và fixed schedule là đối chứng bắt buộc.

Action reward, rollback và multi-branch search chỉ nên là implementation variants trong ablation, không phải contribution trung tâm. Kết quả hiện tại hỗ trợ novelty của dynamic activation protocol, nhưng chưa hỗ trợ claim rằng mid-step intervention vượt post-repair hoặc cải thiện SOTA generation quality.

### 11.4. Ablation đang chạy

Metric job `1057685` thất bại do thiếu CUDA runtime library trong environment đánh giá, không phải do dữ liệu hay metric. Environment đã được bổ sung `nvidia-nvjitlink-cu12`; retry là `1057810`.

Hai ablation mới chỉ kiểm tra trực tiếp contribution:

| Job | So sánh | Câu hỏi |
| --- | --- | --- |
| `1057811[0-3]` | fixed edge schedule / edge-specific schedule × current-state / persistent-state gate | Gain đến từ temporal edge schedule hay chỉ từ row verifier? |
| `1057812[0-1]` | lookahead tại `t=0.30` và `t=0.45`, so với kết quả `t=0.184` | Middle state sớm có đủ tín hiệu để can thiệp hay dynamic graph nên abstain? |

Mỗi setting dùng một seed và cùng paired random stream. Không mở thêm model hay action architecture mới ở vòng này.

### 11.5. Core ablation: temporal schedule và state persistence

Job `1057811[0-3]` đã hoàn thành. Bảng dưới chỉ dùng predicted-x0 reward để giữ action operator cố định.

| Edge schedule | State gate | Actuate violation ↓ | Actuate heldout error ↓ | Actions ↓ | Drift ↓ |
| --- | --- | ---: | ---: | ---: | ---: |
| Fixed/all | Current state | 0.00525 | 0.00099 | **193** | **0.00203** |
| Fixed/all | Persistent state | 0.00496 | 0.00070 | 49 | 0.00057 |
| Edge-specific | Current state | **0.00482** | **0.00056** | **43** | **0.00051** |
| Edge-specific | Persistent state | **0.00482** | **0.00056** | **43** | **0.00051** |
| Final repair | – | 0.00360 | 0.00066 | 40 | 0.00099 |

Hai kết luận đủ rõ:

1. Dùng graph cố định và tin current preview gây can thiệp thừa: số actions tăng hơn 4 lần so với temporal schedule thưa.
2. Persistence giúp mạnh khi chưa có temporal calibration, nhưng gần như redundant sau khi dùng một temporal gate thưa. Matched common-cutoff control cho thấy chưa thể quy phần gain này riêng cho **per-edge temporal eligibility**; không cần một state machine phức tạp.

Noisy-state guidance với fixed/current gate đạt violation tương đương final repair nhưng cần 196 actions và drift 0.0119, trong khi temporal schedule thưa cần 47 actions và drift 0.00116. Temporal graph chủ yếu cải thiện intervention efficiency và fidelity preservation; matched control sau đó cho thấy schedule thưa có thể dùng common cutoff.

### 11.6. Early-middle timing và abstention

Jobs `1057812[0-1]` fork tại `t=0.306` và `t=0.449`. Cả hai setting tạo **zero actions** và selector chọn no-action cho toàn bộ 4,096 rows. Đây không phải failure của job: theo edge-specific cutoff, ba measurement/rate edges chưa readable, còn complex edge `r1_005` không có persistent natural violation trong batch.

Kết quả này củng cố cách diễn giải dynamic graph như một selective controller: ở early-middle trajectory, hành vi đúng có thể là abstain thay vì buộc phải inject feedback. Sweep cuối `1057820[0-2]` chỉ kiểm tra ba điểm quanh transition (`t=0.28`, `0.22`, `0.18`) để xác định khi graph bắt đầu có actionable state; sau đó không mở rộng thêm timing grid.

Sweep transition đã hoàn thành:

| Fork time | Active actuation edges | Branch actions | Mid-select violation ↓ | Mid-select heldout error ↓ | Drift ↓ |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0.449 | none | 0 | 0.00776 | 0.00351 | 0.00000 |
| 0.306 | none | 0 | 0.00674 | 0.00248 | 0.00000 |
| 0.286 | `seed_003` | 16 | 0.00637 | 0.00212 | 0.00020 |
| 0.224 | `seed_003` | 21 | 0.00650 | 0.00224 | 0.00023 |
| 0.184 | `seed_001`, `seed_003` | 31 | **0.00557** | **0.00131** | 0.00065 |
| Final repair | all applicable | 40 | **0.00360** | **0.00066** | 0.00099 |

Graph có transition rời rạc đúng theo learned edge cutoffs: nó abstain trước `t≈0.29`, bắt đầu với rate relation, rồi thêm measurement relation ở `t≈0.20`. Effect tăng khi nhiều edge trở nên readable; tuy vậy final repair vẫn tốt nhất về validity.

Để loại confound rằng edge-specific schedule chỉ tốt vì sparse hơn, job `1057823[0-1]` dùng một common cutoff `t=0.306`, bằng mean cutoff của bốn actuation edges và xấp xỉ matched edge-step budget. Đây là đối chứng cuối cho mechanism: edge-specific schedule phải vượt common schedule thì mới quy gain cho đúng edge timing, thay vì chỉ quy cho ít actions hơn.

### 11.7. Matched common-cutoff control: chưa chứng minh được lợi ích của lịch riêng từng cạnh

Job `1057823[0-1]` đã hoàn thành và thay đổi cách diễn giải kết quả. Khi dùng cùng action operator và khớp intervention budget, common cutoff gần như bằng edge-specific cutoff:

| Schedule | State gate | Actuate violation ↓ | Actuate heldout error ↓ | Score-only violation ↓ | KS ↓ | Drift ↓ | Actions |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed/all | Current | 0.00525 | 0.00099 | 0.05326 | 0.02631 | 0.00203 | 193 |
| Common cutoff | Current | **0.00482** | **0.00056** | 0.05269 | 0.02634 | 0.00054 | 44 |
| Common cutoff | Persistent | **0.00482** | **0.00056** | 0.05262 | 0.02634 | 0.00054 | 44 |
| Edge-specific | Current | **0.00482** | **0.00056** | 0.05262 | **0.02631** | **0.00051** | 43 |
| Edge-specific | Persistent | **0.00482** | **0.00056** | 0.05262 | **0.02631** | **0.00051** | 43 |

Do đó bằng chứng trên Shoppers hiện chỉ ủng hộ hai thành phần: (i) không nên can thiệp suốt toàn bộ trajectory, và (ii) activation phải phụ thuộc state của từng row. Nó **chưa ủng hộ** claim mạnh hơn rằng mỗi edge cần một lịch riêng. Khác biệt cutoff quan sát được ở trajectory là có thật, nhưng bốn cạnh được phép actuation có vùng chuyển tiếp đủ gần nhau để một common late-middle gate đạt cùng kết quả.

Điều này cũng loại bỏ nhu cầu phát triển thêm persistence controller hoặc action state machine: sau temporal gating, hai state gate cho kết quả giống nhau. Contribution method nên giữ đơn giản và chỉ nâng claim lên edge-specific dynamic graph nếu Adult/Default cho thấy common-cutoff control thất bại.

### 11.8. Baseline cục bộ và vị trí hiện tại của method

CTGAN và TVAE đã được chạy từ implementation SDV có sẵn với cùng 4,096 rows. Metric suite tính xong; job `1057825` chỉ lỗi sau cùng vì thiếu package `tabulate` lúc xuất Markdown. CSV vẫn hợp lệ và job `1057841` chạy lại phần xuất artifact sau khi bỏ dependency này.

| Method | Shape error ↓ | Trend error ↓ | Detection ↑ | MLE AUC ↑ | Actuate violation ↓ | Heldout violation error ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CTGAN (SDV) | 14.264 | 15.435 | 0.750 | 0.801 | 0.3823 | 0.3063 |
| TVAE (SDV) | 16.694 | 9.419 | 0.180 | 0.877 | 0.2758 | 0.1852 |
| TabDiff | 1.746 | 1.585 | 0.988 | 0.899 | 0.0070 | 0.0218 |
| DynamicGraph mid-lookahead | 1.739 | 1.509 | 0.981 | **0.900** | 0.0059 | 0.0209 |
| DynamicGraph reverse-x0 | **1.733** | 1.512 | 0.983 | 0.898 | 0.0052 | 0.0207 |
| Post-repair | 1.735 | **1.501** | **0.998** | 0.900 | **0.0036** | **0.0205** |

Các baseline cho thấy backbone TabDiff vốn đã mạnh hơn nhiều so với CTGAN/TVAE. Dynamic guidance giữ fidelity và cải thiện constraint metrics nhẹ, nhưng chưa vượt post-repair. Vì vậy đây chưa phải kết quả SOTA; giá trị khoa học hiện nằm ở việc xác định khi nào trajectory feedback đáng dùng, không nằm ở một bảng quality tổng hợp vượt mọi baseline.

### 11.9. Jobs cross-dataset đang chạy

| Job | Nội dung | Mục đích quyết định claim |
| --- | --- | --- |
| `1057835[0-1]` | Protocol tạo graph typed/complex trên Adult và Default, tối đa 180k tokens mỗi dataset | Tạo graph độc lập với Shoppers, có conjunction/column relation nếu evidence ủng hộ |
| `1057836[0-1]` | Curation thành `actuate`, `score_only`, `monitor` | Giữ separation giữa evidence và quyền sửa row |
| `1057837[0-1]` | Trajectory readability, 8,192 samples/dataset | Kiểm tra cutoff heterogeneity có lặp lại cross-dataset hay không |
| `1057841` | Xuất lại unified Shoppers baseline suite | Hoàn thiện artifact; không chạy lại training baseline |

Đây là vòng thử quyết định, không phải mở rộng engineering. Nếu common cutoff tiếp tục đủ trên Adult/Default, method nên được viết là **state-dependent temporally gated graph guidance** với một gate chung đơn giản. Nếu các nhóm cạnh numeric/categorical/process có vùng readable tách biệt và common cutoff tạo false actions, khi đó mới có đủ bằng chứng cho **edge-specific dynamic graph**.

## 12. Dynamic graph topology tại middle trajectory

### 12.1. Định nghĩa contribution được thu hẹp

Main novelty hiện được định nghĩa bằng active topology theo row và state:

\[
G_i(t)=\{e\in G:\;R_e(t)=1\;\land\;A_e(\hat{x}_{0,i}(t))=1\}.
\]

`R_e(t)` là readability gate đo từ trajectory; `A_e` là applicability của edge trên predicted clean row. Graph thay đổi khi một edge được thêm hoặc loại khỏi active set của một row. LLM được dùng để đề xuất và diễn giải graph trước sampling; execution trong sampler là typed và deterministic. Action operator không phải novelty trung tâm.

Các baseline topology dùng cùng catalog, action operator, seed và paired reverse process:

- `dynamic`: recompute row–edge topology ở từng middle step;
- `frozen-middle`: chụp topology khi edge bước vào readable region rồi giữ cố định;
- `periodic-k`: refresh mỗi 2 hoặc 4 reverse steps;
- `shuffled-row`: giữ edge support nhưng gán active edges sang row khác;
- `oracle-final`: static topology lấy từ paired final sample, chỉ là upper bound.

### 12.2. Shoppers: dynamic topology tốt hơn frozen-middle trên calibrated validity

| Topology | Actuate violation ↓ | Actuate heldout error ↓ | Overall heldout error ↓ | Actions | Graph flips |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unconditional | 0.00761 | 0.00335 | 0.02500 | 0 | – |
| Frozen-middle | 0.00553 | 0.00127 | 0.02404 | 37 | 0 |
| Dynamic every step | **0.00452** | **0.00026** | 0.02384 | 45 | 33 |
| Periodic, k=2 | 0.00465 | 0.00039 | **0.02381** | 44 | 27 |
| Periodic, k=4 | 0.00465 | 0.00039 | **0.02381** | 44 | 25 |
| Final repair | **0.00360** | 0.00066 | 0.02376 | 41 | – |

Dynamic graph không đạt raw violation thấp nhất, nhưng heldout error thấp hơn final repair vì nó giữ lại legitimate violation mass thay vì ép tất cả về zero. Refresh mỗi 4 steps gần như bằng refresh mỗi 2 steps, nên implementation không cần gọi LLM/verifier ở mọi step.

Shuffled-row là negative control mạnh: nó tạo 583 actions, tăng overall violation error `0.0250 → 0.0581`, numeric KS `0.0250 → 0.0333` và drift `0.00041 → 0.01578`. Điều này cho thấy graph semantics phải gắn đúng row; chỉ có cùng số edge hoặc cùng rule catalog là không đủ.

Middle-only và late-middle-only có cùng actuation violation `0.00427`, nhưng middle-only cho overall violation `0.06001`, tốt hơn late-middle `0.06020`. Đây mới là one-seed mechanism evidence; chưa đủ để claim middle luôn vượt late.

### 12.3. Adult và Default xác nhận lifecycle nhưng giới hạn claim edge-specific

Adult giữ 9 edges (`3 actuate`, `5 score_only`, `1 monitor`) và tất cả readable, cutoff từ `0.082` đến `0.490`. Ba actuation edges không có final natural violation, nên Adult hỗ trợ graph-lifecycle analysis nhưng không phải benchmark tốt cho actuation gain.

Default giữ 24 edges (`5 actuate`, `11 score_only`, `8 monitor`). Cutoff trải từ `0.122` đến `0.449`; riêng actuation edges từ `0.143` đến `0.449`. Violation recall tăng từ `0.035` ở early, lên `0.753` ở middle và `0.967` ở late-middle. Đây là bằng chứng cross-dataset rõ nhất rằng early semantic graph chưa đủ tin cậy.

| Default topology | Actuate violation ↓ | Actuate heldout error ↓ | Overall heldout error ↓ | Actions | Graph flips |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unconditional | 0.02223 | 0.01630 | 0.01951 | 0 | – |
| Dynamic | 0.01128 | 0.00535 | 0.01524 | 26 | 119 |
| Frozen-middle | 0.01131 | 0.00537 | **0.01474** | 23 | 0 |
| Periodic, k=4 | 0.01128 | 0.00535 | 0.01508 | 25 | 85 |
| Final repair | **0.01080** | **0.00487** | 0.01508 | 13 | – |

Dynamic và frozen gần như bằng nhau trên actuation edges của Default; frozen tốt hơn ở overall error do indirect changes trên score-only relations. Vì vậy dynamic refresh có ích trên Shoppers nhưng chưa universal.

Temporal schedule vẫn cần thiết. Trên Default, all-steps cần 101 actions nhưng actuation error là `0.00900`; edge schedule đạt `0.00535` với 26 actions. Tuy nhiên common cutoff đạt raw actuation violation `0.00783` và overall error `0.01556`, cạnh tranh với edge-specific schedule. Do đó claim chính là **middle state-dependent topology**, không phải edge-specific timing.

### 12.4. Flip-subgroup audit bác bỏ “dynamic applicability” như lời giải duy nhất

Paired audit chạy dynamic và frozen liên tiếp trên cùng GPU, sau đó chỉ giữ rows có ít nhất một applicability edge bật/tắt:

| Dataset | Rows đổi applicability | Base violation events | Dynamic | Frozen | Final repair |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shoppers | 176 | 5 | 5 | 5 | 4 |
| Default | 76 | 6 | 2 | 2 | 3 |

Dynamic và frozen không khác nhau trên chính nhóm đổi applicability. Điều này bác bỏ cách diễn giải rằng gain hiện tại chủ yếu đến từ antecedent/applicability topology flip. Trên Default, middle guidance tốt hơn final repair trên subgroup này, nhưng dynamic và frozen cùng tốt như nhau.

Do đó graph state cần được định nghĩa đầy đủ hơn bằng trạng thái edge `inactive / satisfied / violated`, thay vì chỉ active/inactive theo antecedent. Job `1059037[0-1]` đang so sánh dynamic edge state với `frozen-state`, trong đó cả applicability lẫn violation state đều bị đóng băng ở entry point. Job `1059038[0-1]` đánh giá riêng rows có constraint-state transition.

### 12.5. Oracle-final topology

Paired oracle audit `1058955[0-1]` đã hoàn thành:

- Trên Shoppers, dynamic middle topology và oracle-final topology cho kết quả giống hệt trong paired run: actuation violation `0.00505`, heldout error `0.00150`, 35 actions.
- Trên Default 4,096-row run, oracle-final cho raw actuation violation thấp hơn (`0.00623` so với `0.00981`), còn dynamic có heldout error thấp hơn. Tuy nhiên kết luận này không lặp lại ở 8,192-row confirmation: oracle đạt actuation error `0.00248`, tốt hơn dynamic `0.00498`. Do đó chưa có bằng chứng ổn định rằng oracle topology over-correct trên Default.

Shoppers cho thấy dynamic graph gần oracle-final, còn Default vẫn có topology-estimation gap. Mục tiêu hợp lý là chọn intervention topology làm output khớp real semantic violation mass, không chỉ tối thiểu hóa mọi violation.

### 12.6. Confirmation 8,192 rows và dynamic constraint-state

Job `1059035[0-1]` dùng common temporal gate để bỏ confound edge-specific schedule:

| Dataset | Method | Actuate violation ↓ | Actuate heldout error ↓ | Overall heldout error ↓ | Actions |
| --- | --- | ---: | ---: | ---: | ---: |
| Shoppers | Unconditional | 0.00810 | 0.00385 | 0.02061 | 0 |
| Shoppers | Frozen applicability | 0.00595 | 0.00169 | 0.01948 | 112 |
| Shoppers | Dynamic | **0.00532** | **0.00106** | 0.01909 | 124 |
| Shoppers | Oracle-final | **0.00532** | **0.00106** | **0.01908** | 125 |
| Shoppers | Final repair | 0.00360 | 0.00066 | 0.01889 | 114 |
| Default | Unconditional | 0.02385 | 0.01553 | 0.02255 | 0 |
| Default | Frozen applicability | 0.01378 | 0.00496 | 0.01870 | 36 |
| Default | Dynamic | 0.01380 | 0.00498 | 0.01862 | 36 |
| Default | Oracle-final | **0.00822** | **0.00248** | **0.01878** | 30 |
| Default | Final repair | 0.01459 | 0.00627 | 0.01902 | 17 |

Dynamic hơn frozen applicability rõ trên Shoppers nhưng gần như ngang trên Default. Tuy nhiên dynamic vượt final repair trên cả raw actuation violation lẫn heldout error ở Default. Oracle gap trên Default cho thấy còn dư địa cải thiện state estimator, không nhất thiết action operator.

Baseline `frozen-state` đóng băng cả applicability và satisfaction/violation label ở entry point. Với 8,192 rows:

| Dataset | State policy | Actuate violation ↓ | Actuate heldout error ↓ | Overall heldout error ↓ | Actions |
| --- | --- | ---: | ---: | ---: | ---: |
| Shoppers | Dynamic state | **0.00563** | **0.00138** | **0.02040** | 134 |
| Shoppers | Frozen state | 0.00666 | 0.00240 | 0.02090 | 113 |
| Default | Dynamic state | **0.01463** | **0.00553** | **0.01879** | 45 |
| Default | Frozen state | 0.01485 | 0.00574 | 0.01892 | 45 |

Đây là evidence cross-dataset rõ nhất hiện tại cho graph evolution: cập nhật `satisfied ↔ violated` state tốt hơn đóng băng graph state, mạnh trên Shoppers và nhỏ nhưng cùng chiều trên Default. Method nên biểu diễn mỗi edge bằng state có kiểu, không chỉ một cờ applicability.

### 12.7. Constraint-state transition subgroup

Job `1059038[0-1]` đã hoàn thành và xác nhận effect tập trung đúng hơn khi graph state được định nghĩa bằng `satisfied ↔ violated` transition:

| Dataset | Transition rows | Base violation events | Dynamic | Frozen-state | Final repair | Dynamic net removed | Frozen net removed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Shoppers | 103 | 99 | 38 | 56 | **30** | **61** | 43 |
| Default | 63 | 56 | **16** | 17 | 26 | **40** | 39 |

Trên Shoppers, dynamic state loại thêm 18 violation events so với frozen-state; final repair vẫn tốt hơn 8 events. Trên Default, dynamic chỉ hơn frozen-state một event nhưng cả hai middle-step methods vượt final repair rõ trên transition rows. Dynamic làm xấu 2 Default rows trong khi final repair không làm xấu row nào, nên cần giữ fidelity/false-action metrics song song.

Kết luận hiện tại:

1. Applicability topology flip đơn thuần không giải thích gain.
2. Typed edge-status transition là định nghĩa dynamic graph phù hợp hơn.
3. Middle graph update có giá trị rõ nhất trên rows mà semantic status chưa ổn định; không cần áp dụng cho toàn bộ rows.
4. Evidence mạnh trên Shoppers, cùng chiều nhưng nhỏ trên Default; cần thêm một dataset có process constraints và natural violations trước claim tổng quát.

Các phép thử này quyết định claim tiếp theo. Nếu dynamic chỉ hơn `frozen-state` trên rows có edge-status transition, main novelty có thể đặt ở **trajectory-conditioned constraint-state graph**. Nếu không, kết quả chỉ hỗ trợ temporal gating và row-specific applicability, chưa đủ hỗ trợ graph evolution như contribution chính.

### 12.8. Robustness với sampler 30 bước

Job `1059041[0-1]` lặp paired dynamic/frozen-state với 8,192 rows nhưng dùng đúng 30 reverse steps thay vì 50. Bảng dưới báo biến thể `x0_reward_calibrated` và cùng final repair trong mỗi paired run.

| Dataset | State policy | Actuate violation ↓ | Actuate heldout error ↓ | Overall heldout error ↓ | Numeric KS ↓ | Actions |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Shoppers | Unconditional | 0.00836 | 0.00410 | 0.02576 | 0.03204 | 0 |
| Shoppers | Dynamic state | **0.00519** | **0.00093** | **0.02436** | 0.03226 | 125 |
| Shoppers | Frozen state | 0.00583 | 0.00157 | 0.02466 | **0.03221** | 113 |
| Shoppers | Final repair | **0.00360** | 0.00066 | 0.02434 | 0.03236 | 108 |
| Default | Unconditional | 0.03112 | 0.02202 | 0.02522 | 0.03541 | 0 |
| Default | Dynamic state | 0.01316 | 0.00405 | 0.01821 | 0.03541 | 67 |
| Default | Frozen state | **0.01126** | **0.00312** | **0.01804** | **0.03540** | 65 |
| Default | Final repair | 0.01440 | 0.00530 | 0.01870 | 0.03542 | 33 |

Kết quả này vừa củng cố vừa giới hạn claim. Trên Shoppers, dynamic state vẫn hơn frozen-state ở 30 steps. Trên Default, thứ tự đảo lại: frozen-state hơn dynamic, dù cả hai đều hơn final repair. Vì vậy không thể claim cập nhật state ở mọi step luôn tốt. Cách diễn giải phù hợp hơn là **dynamic state update có lợi khi transition signal đủ chính xác; nếu state estimate dao động, cần confidence hoặc persistence gate**.

### 12.9. Hypothesis mới đã submit

Hai job array mới giữ một seed, 8,192 rows và chạy đồng thời trên Shoppers/Default:

| Job | Hypothesis | So sánh quyết định |
| --- | --- | --- |
| `1060241[0-5]` | Một edge chỉ nên phát action sau khi violation ổn định hai checkpoint, hoặc đúng tại thời điểm chuyển sang `violated` | `current` vs `stable` vs `transition` state gate, cùng dynamic graph |
| `1060242[0-5]` | Lợi ích của graph evolution tập trung ở middle region: early state quá nhiễu, late state gần final repair | early `[.55,.75]`, middle `[.25,.50]`, late `[.10,.25]`; trong mỗi vùng so dynamic với frozen-state |

Hai phép thử này không mở thêm nhánh bài toán. Chúng kiểm tra trực tiếp hai thành phần cần formalize cho method: **khi nào graph state đủ tin cậy** và **state transition nào được phép phát action**.

### 12.10. Kết quả state gate, temporal window và main 3-seed

`1060241[0-5]` cho thấy stable gate tốt hơn current nhỏ trên cả hai dataset trong seed đầu, còn transition-only chỉ tạo 4 Default actions và 60 Shoppers actions nên bỏ phần lớn violations. Transition-only bị loại khỏi main method.

`1060265[0-5]` chạy ba seeds mới. Dynamic state thắng frozen-state ở cả ba Shoppers seeds. Trung bình actuation heldout error là `0.000609` với dynamic-stable so với `0.001372` với frozen-state. Trên Default, dynamic-current đạt `0.003677`, dynamic-stable `0.003821`, frozen-state `0.004011`; effect dương theo mean nhưng nhỏ và có variance lớn.

Temporal window với stable gate củng cố middle-step claim:

- Shoppers middle: dynamic `0.003278`, frozen `0.005019` actuation error; overall `0.022361` so với `0.023452`.
- Default middle: dynamic `0.010543`, frozen `0.014352`; overall `0.018641` so với `0.020659`.
- Early: Shoppers bằng nhau; Default dynamic xấu hơn frozen.

Như vậy, evidence hiện hỗ trợ **middle trajectory dynamic state update**, không hỗ trợ early guidance và không yêu cầu LLM gọi ở mọi step. Kế hoạch chạy nhanh đến ICDE được tách ở `ICDE_SUBMISSION_SPRINT_20261007.md`.
