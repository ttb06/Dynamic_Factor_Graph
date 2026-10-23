# Research update — condition-guided tabular diffusion

**Ngày cập nhật:** 29/09/2026  
**Quy ước:** “đã chạy” nghĩa là đã có output; “đã submit” nghĩa là đang chạy hoặc chờ Slurm; “hypothesis” nghĩa là chưa kết luận cho tới khi aggregate hoàn tất.

## 1. Kết luận hiện tại

Dự án hiện không còn tập trung vào claim rộng “LLM feedback trong reverse diffusion”. Claim đó bị bao phủ một phần bởi PromptLoop, Harpoon, CTDF và SCTab-Diff. Bài toán đang được thu hẹp thành:

> Sinh dữ liệu bảng theo **unseen user condition**, đặc biệt cho rare cohort, bằng một generator có thể tái sử dụng qua nhiều condition và tiến tới nhiều schema; giảm chi phí so với rejection sampling hoặc per-condition fine-tuning mà vẫn giữ conditional fidelity.

LLM có vai trò phù hợp nhất ở control plane:

`natural-language request → typed constraint program → schema/type check → deterministic executor`

LLM chưa có bằng chứng cần thiết ở inner sampling loop. Direct per-row LLM judgement hiện vừa đắt vừa kém ổn định hơn compile-once rồi execute bằng code.

Kết quả đã có đủ để tiếp tục research project, nhưng **chưa đủ để gọi là state of the art** và chưa đủ cho một bài ICDE hoàn chỉnh. Điểm còn thiếu quan trọng nhất là một shared cross-table model và đánh giá trên nhiều dataset/condition có nguồn độc lập.

## 2. Gap có thể bảo vệ

### 2.1 Khoảng trống chính

Các hướng gần nhất giải quyết những phần khác nhau:

- CuTS hỗ trợ declarative constraints nhưng cần per-program fine-tuning trên từng table.
- CTSyn học cross-table generation và schema conditioning nhưng không nhằm thực thi unseen user condition.
- Harpoon và CTDF đưa differentiable/hard constraints vào denoising nhưng không giải quyết cross-schema zero/few-shot condition generalization.
- SCTab-Diff đưa LLM-derived prior vào reverse diffusion, nhưng prior chủ yếu cố định theo table.
- LCATS và Zhao cùng cộng sự phát hiện/kiểm chứng rule rồi post-repair, nhưng không chứng minh đang lấy mẫu từ conditional distribution.
- PromptLoop và các phương pháp image/video đã bao phủ ý tưởng foundation model đọc intermediate preview rồi feedback vào sampling.

Khoảng trống đáng theo đuổi là giao của bốn yêu cầu:

1. condition thay đổi ở inference time và có thể chưa thấy khi train;
2. condition có tham số, phép kết hợp, applicability và exception;
3. rare cohort làm rejection sampling trở nên đắt;
4. condition mechanism có thể tái sử dụng qua schema, thay vì fine-tune riêng cho mỗi table/program.

### 2.2 Điều không còn đủ mới

- LLM tạo static rule hoặc prior trước sampling;
- gradient/projection ở từng reverse step;
- LLM đọc `x_hat_0` rồi feedback nói chung;
- LLM compile metadata thành executable rule;
- đạt zero violation sau deterministic repair;
- ba adapter riêng trên ba table nhưng gọi là cross-table transfer.

### 2.3 Claim gần hạn và claim dài hạn

**Claim gần hạn có bằng chứng:** reusable condition adapter giảm sample complexity cho rare cohort, tạo một Pareto frontier giữa condition satisfaction và fidelity.

**Claim phương pháp đang kiểm tra:** temporal scope và dependency scope có thể giảm distortion do guidance mà vẫn giữ phần lớn sample-efficiency gain.

**Claim dài hạn chưa có bằng chứng:** shared schema-and-condition encoder có thể generalize đồng thời sang unseen table và unseen condition.

## 3. Bằng chứng thực nghiệm đã có

### 3.1 Rare-cohort generation, tám seed

So với unconditional generation + rejection sampling ở nhóm mixed-type hiếm nhất đang xét:

| Dataset | Speedup về số trajectory hợp lệ | Δ KS | Δ TV | Diễn giải |
| --- | ---: | ---: | ---: | --- |
| Default | `4.70 [4.37, 4.98]×` | `+0.011 [0.005, 0.018]` | `+0.029 [0.024, 0.034]` | nhanh hơn rõ nhưng fidelity kém hơn |
| Shoppers | `7.28 [6.97, 7.62]×` | `+0.003 [0.000, 0.007]` | `-0.015 [-0.017, -0.012]` | nhanh hơn; TV tốt hơn, KS gần hòa nhưng hơi kém |

Dấu dương của KS/TV là xấu hơn. Vì vậy kết quả hiện tại hỗ trợ sample-efficiency, không hỗ trợ claim fidelity tốt hơn một cách nhất quán.

Nguồn: `outputs/rejection_frontier_aggregate_1045512/` và `outputs/rare_support_statistical_audit_1045512_1045513/`.

### 3.2 CuTS official baseline

Trên năm Adult programs đã chọn, CuTS đạt validity `1.0` ở cả năm resample; marginal TV thay đổi rất nhỏ. Constrained fine-tuning mất gần hai giờ, trong khi baseline run khoảng vài phút. Năm resample dùng cùng một fine-tuned model, vì vậy chúng đo sampling variance, không phải training variance.

Hệ quả:

- không nên cạnh tranh bằng claim “đạt validity cao hơn CuTS”;
- cơ hội nằm ở amortized cost trên nhiều unseen conditions và nhiều tables;
- mọi bảng so sánh phải tách pretraining cost, per-condition adaptation cost và sampling cost.

Nguồn: `outputs/cuts_official_resample_audit_1045748/RESULTS.md`.

### 3.3 LLM direct judgement so với compile-once

Trên request-conditioned probe có 140 synthetic-only cases và 30 same-row/different-request pairs:

- GPT-4.1-mini direct: accuracy `0.8643`, swap-pair exact `0.8667`, khoảng `37.5k` tokens;
- GPT-5.4-mini direct: accuracy `0.65`, swap-pair exact `0.3667`, khoảng `37.7k` tokens;
- cả hai compile sang typed DSL rồi local execute: execution accuracy `1.0`, swap-pair exact `1.0`, exception false-action `0`, khoảng `1.85k` tokens.

Đây là bằng chứng mạnh chống lại thiết kế gọi LLM theo từng row/timestep trong setting hiện tại. Nó ủng hộ LLM compiler có type checking, không ủng hộ LLM làm verifier duy nhất.

Nguồn: `outputs/request_condition_aggregate_1031786/RESULTS.md`.

### 3.4 Static rule mining không thể tự động biến thành semantics

Audit trên Adult, Default và Shoppers sinh 788 candidate relations nhưng chỉ 80 quan hệ không có observed exception ở train; 334 là probabilistic và 132 thiếu support. Ví dụ `Own-child → age < 30` có khoảng 19.7% exception ở cả train và held-out. Shoppers cho thấy directionality rõ: `count=0 → duration=0` có thể exact, nhưng chiều ngược lại có exception.

Hệ quả: condition representation phải tách `applicability`, `satisfaction`, `exception` và `hardness`; mined dependency không được mặc định là hard constraint.

Nguồn: `outputs/cross_dataset_constraint_20260924/`.

### 3.5 Temporal guidance

216 runs trên hai dataset, hai condition catalogs và ba seed cho thấy không có một guidance schedule thắng đồng thời về success, cost, KS và TV.

- `step_release` cường độ cao giữ success tốt nhưng fidelity chưa ổn định;
- `midpulse` thường giảm TV với mức giảm success vừa phải;
- `linear_ramp` có vùng fidelity tốt nhưng sample cost cao.

Kết luận phù hợp là một Pareto frontier, không phải một schedule tối ưu phổ quát.

Nguồn: `outputs/temporal_contract_guidance_aggregate_1045740/RESULTS.md`.

### 3.6 Target-scoped guidance

Paired analysis ban đầu trên mixed conditions cho thấy chỉ áp conditional delta lên các cột xuất hiện trong condition cải thiện fidelity khá nhất quán nhưng giảm success.

Ví dụ so với all-feature guidance ở `peak=4`:

- Default `q975`, `midpulse`: success giảm `0.195`, KS giảm `0.109`, TV giảm `0.043`;
- Default `q975`, `step_release`: success giảm `0.228`, KS giảm `0.163`, TV giảm `0.083`;
- Shoppers `q975`, `midpulse`: success giảm `0.128`, KS giảm `0.025`, TV giảm `0.015`;
- Shoppers `q975`, `step_release`: success giảm `0.183`, KS giảm `0.038`, TV giảm `0.050`.

Đây là result có giá trị vì nó chẩn đoán đúng failure mode: all-feature guidance làm condition dễ đạt hơn bằng cách dịch chuyển quá nhiều cột; target-only guidance bảo vệ fidelity nhưng không đủ để tái tạo joint cohort.

36 numeric-condition reruns đã hoàn tất và lặp lại cùng pattern. Ví dụ Default `q950`, `step_release peak=4` giảm success `0.309` nhưng cải thiện KS `0.185` và TV `0.052`; Shoppers `q950` giảm success `0.193`, cải thiện KS `0.095` và TV `0.017`. Ở vài mức rất hiếm, TV gần hòa hoặc xấu hơn nhẹ dù KS tốt hơn. Vì vậy target-only không phải final method; nó là một endpoint hữu ích để kiểm tra all-to-target và dependency scope.

Các confidence interval sẽ được tính theo seed cluster trong aggregate mới; không xem các condition cùng seed là independent repetitions.

## 4. Method đang được kiểm tra

### 4.1 Reusable typed condition adapter

Backbone TabDiff được giữ cố định hoặc cập nhật rất ít. Typed condition vector đi qua một condition projection. Adapter học một lần trên một family of conditions rồi được đánh giá trên threshold và conjunction chưa thấy.

Ma trận hiện có:

- projection-only;
- low-learning-rate backbone update;
- full adapter update;
- fixed sparse thresholds;
- dense train-quantile curriculum;
- condition dropout.

Mục tiêu của ablation là biết generalization đến từ representation/curriculum hay chỉ do fine-tune backbone mạnh.

108 evaluation runs đã hoàn tất; statistical audit được submit ở job `1048595`. Kết quả paired sơ bộ đã đủ để bác bỏ một hypothesis quá mạnh: projection-only không phải lựa chọn tốt phổ quát. So với fixed-threshold projection-only:

- Shoppers, rare mixed, `dense_full` với guidance 1 tăng success `+0.0671 [0.0642, 0.0717]`, đồng thời giảm KS `0.0282 [0.0223, 0.0354]` và TV `0.0100 [0.0057, 0.0124]`;
- Adult, rare mixed, `dense_full` với guidance 1 tăng success `+0.0433 [0.0405, 0.0458]`, giảm KS `0.0244 [0.0118, 0.0345]` và TV `0.0278 [0.0181, 0.0449]`;
- Default không lặp lại dominance này: `dense_full` guidance 2 tăng success `+0.1981` nhưng làm KS xấu hơn `+0.0702` và TV xấu hơn `+0.0336`.

Diễn giải hiện tại: dense curriculum và backbone adaptation có thể cải thiện cả success lẫn fidelity khi sparse training conditions gây underfit, nhưng cũng có thể oversteer ở một table khác. Đây là bằng chứng cho adaptive regularization/table-aware calibration, không phải bằng chứng rằng full update luôn tốt hơn. Nó cũng làm yếu claim parameter-efficient transfer nếu method cuối vẫn phải cập nhật toàn backbone cho mỗi table.

### 4.2 All-to-target temporal scope

Reverse process bắt đầu bằng all-feature guidance để hình thành joint cohort, sau đó chuyển sang target-only guidance ở `t ∈ {0.5, 0.25, 0.1}` để phần cuối không tiếp tục kéo lệch các cột ngoài condition.

Hypothesis:

> So với all-feature guidance, method giữ phần lớn success nhưng cải thiện conditional fidelity; so với target-only guidance, method lấy lại success với distortion nhỏ hơn all-feature.

108 runs đã submit trên Default/Shoppers, numeric/mixed conditions, ba seed, ba schedules và ba switch times.

### 4.3 Dependency-scoped guidance

Binary scope có thể quá thô. Method mới cho conditional delta đi qua:

`columns(condition) ∪ top-k dependent columns`, với `k ∈ {1,2,4}`.

Dependency graph được tính chỉ từ training split bằng normalized mutual information sau khi discretize numeric features. Không dùng held-out data và không dùng target labels của evaluation để chọn scope. Danh sách cột được lưu lại cho từng condition để audit.

Supervised target và các cột có tên nhạy cảm/nhân khẩu học (`age`, `sex`, `gender`, `race`, `ethnic`, `marital`, `relationship`) bị loại khỏi automatic neighbor selection, trừ khi chính condition yêu cầu cột đó. Việc này tránh biến một association mạnh thành actuator không chủ ý lên label hoặc protected attribute.

Hypothesis:

> Dependency scope tạo điểm nằm giữa target-only và all-feature trên cả success và fidelity, đồng thời làm rõ condition propagation cần bao nhiêu context của joint distribution.

108 runs đã submit với cùng seed và schedule để paired comparison trực tiếp.

### 4.4 Hướng foundation/cross-table

Đây vẫn là phần quyết định tầm vóc của bài. Kiến trúc cuối nên tách:

- schema encoder dùng chung;
- typed condition encoder dùng chung;
- table-specific normalization/tokenization;
- shared diffusion backbone hoặc shared adapter/hypernetwork;
- deterministic verifier.

Evaluation tối thiểu phải có leave-one-table-out và leave-one-condition-family-out. Chỉ đạt tốt trên condition mới của một table chưa đủ cho claim foundation model.

Adult ablation đã có cho thấy frozen foundation embedding không tự động tốt hơn typed DSL. Ở held-out compositions với guidance 2, satisfaction của structured condition khoảng `0.608`, text embedding `0.534`, hybrid `0.555`; structured cũng có KS tốt hơn text/hybrid trong phép so sánh này. Vì vậy foundation encoder chỉ được giữ nếu nó giúp trên schema mới hoặc giảm adaptation cost, không phải vì tên model.

Jobs `1048607–1048618` kiểm tra lại trên Diabetes: raw text embedding, factorized hybrid (`foundation description + exact typed parameters`) và structured condition, cùng backbone seed và training budget. Đây vẫn là within-table ablation; kết quả dương chỉ cho phép quyết định có đáng xây shared cross-table encoder hay không, chưa cho phép claim cross-table transfer.

## 5. Jobs mới đã submit

| Jobs | Nội dung | Output dự kiến |
| --- | --- | --- |
| `1048218–1048325`, `1048326` | retry 108 adapter-ablation evaluations sau khi sửa condition batch size, rồi aggregate | `outputs/adapter_variant_frontier_aggregate_1048326/` |
| `1048327–1048362`, `1048363` | target-scoped numeric rerun với đúng adapter | `outputs/temporal_contract_guidance_aggregate_1048363/` |
| `1048364–1048471`, `1048472` | all-to-target temporal scope grid và aggregate | `outputs/temporal_contract_guidance_aggregate_1048472/` |
| `1048473` | paired scope comparison đầu tiên | `outputs/temporal_scope_comparison_1048473/` |
| `1048474` | chuẩn bị Magic, Diabetes, Beijing, News theo official TabDiff processing; đã hoàn tất | log `llmtd-dataset_1048474.*` |
| `1048475` | constraint audit trên bảy dataset | `outputs/extended_constraint_audit_1048475/` |
| `1048476–1048481` | ba-seed TabDiff backbone cho Magic và Diabetes | `outputs/full_train_{dataset}_{jobid}/` |
| `1048482–1048589`, `1048590` | dependency-scoped guidance grid và aggregate | `outputs/temporal_contract_guidance_aggregate_1048590/` |
| `1048591` | final paired comparison cho all, target, all-to-target và dependency scope | `outputs/temporal_scope_comparison_1048591/` |
| `1048592–1048595` | curation benchmark có provenance và statistical audit cho adapter variants | `outputs/constraint_benchmark_curation_*/`, `outputs/adapter_variant_statistical_audit_1048595/` |
| `1048596–1048605` | ba adapter variants × ba seed trên Diabetes và aggregate | `outputs/diabetes_adapter_variant_aggregate_1048605/` |
| `1048607–1048618` | foundation-condition embeddings và adapter ablation trên Diabetes | `outputs/diabetes_foundation_condition_aggregate_1048618/` |

Không train regression backbones cho Beijing/News ở lô này vì preprocessing helper hiện gắn target vào categorical block. Hai dataset vẫn được đưa vào rule/condition audit; regression support sẽ được thêm có kiểm thử riêng thay vì âm thầm dùng sai representation.

Dataset preparation đã hoàn tất sạch về exit code: Magic có 17,117 train/1,902 test rows; Diabetes 61,059/20,354; Beijing 37,581/4,176; News 35,679/3,965. Log chỉ có pandas `FutureWarning` về dtype khi upstream processor thay token thiếu, không có lỗi dữ liệu hoặc exception.

Seven-dataset audit đầu tiên (`1048475`) đã hoàn tất với 2,747 mined/manual candidates, nhưng số lượng này không thể dùng làm kích thước benchmark. Trong 1,073 exact/near-hard candidates của Diabetes, 862 có consequent base rate ít nhất 95%; News có 200/366 trường hợp tương tự. Sau filter base rate, lift, sensitive/target columns và deduplicate theo ordered column pair, số empirical stress candidates còn khoảng 179 cho Diabetes, 110 cho News, 15 cho Shoppers, 13 cho Beijing và 5 cho Adult. Chúng vẫn chưa phải semantic ground truth.

Lô `1048593–1048594`, với bản curation cuối `1048606`, bổ sung constraints có provenance từ UCI data dictionary: bốn medication-summary relations của Diabetes và chín derived/order-statistic relations của News, rồi chạy lại curation. Các relation này đã được kiểm tra độc lập trên train/held-out; phần lớn có zero observed exception, hai identity của News có violation dưới 0.015% ở train và zero ở held-out. Provenance làm chúng phù hợp làm benchmark hơn các correlation tự khai phá, nhưng vẫn cần independent review trước khi gọi là domain invariant.

Nguồn dataset/provenance: [UCI Diabetes 130-US Hospitals](https://archive.ics.uci.edu/dataset/296/diabetes), [UCI Online News Popularity](https://archive.ics.uci.edu/dataset/332/online+news+popularity), [UCI Beijing PM2.5](https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data).

## 6. Tiêu chí quyết định sau lô này

### Tiếp tục method theo từng table nếu

- dense curriculum hoặc projection-only adapter cải thiện unseen-condition fidelity ổn định qua dataset;
- all-to-target hoặc dependency scope chiếm thêm vùng Pareto, không chỉ đổi success lấy fidelity theo tỷ lệ tương đương;
- lợi ích sample-efficiency còn rõ khi tính đúng số denoiser calls;
- kết quả giữ dấu qua ít nhất Default, Shoppers và Diabetes.

### Chuyển trọng tâm sang benchmark/analysis nếu

- mọi scope/schedule chỉ di chuyển dọc cùng một trade-off curve;
- post-repair hoặc rejection sampling vẫn tốt hơn ở toàn bộ vùng support thực tế;
- adapter phải cập nhật phần lớn backbone cho mỗi table/condition;
- các gains biến mất khi condition family hoặc schema chưa thấy.

### Chỉ claim foundation/cross-table nếu

- cùng một shared parameter set chạy trên nhiều table;
- có held-out schemas và held-out condition families;
- so với table-specific TabDiff, CuTS fine-tuning, CTSyn/rejection và post-repair;
- reporting gồm quality, adaptation cost, sampling cost và failure/abstention.

## 7. Đánh giá khả năng thành bài ICDE

Project có thể thành bài ICDE nếu đóng góp cuối cùng gồm:

1. một benchmark được xây từ dataset/splits và constraint families đã có, cộng protocol unseen-table × unseen-condition × rarity;
2. một reusable condition mechanism có lợi ích rõ so với rejection, repair và per-condition fine-tuning;
3. một cross-table experiment thật, không chỉ nhiều model riêng;
4. error analysis về applicability, exception, rare-mode loss và cost.

Ước lượng từ trạng thái hiện tại:

- 1–2 tuần để chốt adapter/scope method và loại các hypothesis không hiệu quả;
- 2–3 tuần để mở rộng 5–8 dataset, thêm downstream utility và condition sources độc lập;
- 3–5 tuần để tích hợp shared cross-table backbone/adapter và chạy transfer protocol;
- 2–3 tuần cho official baselines, privacy/error analysis, ablation và viết bài.

Tổng hợp lý: khoảng **8–13 tuần** cho một submission đầy đủ nếu CTSyn hoặc backbone thay thế có thể chạy ổn định. Nếu cross-table transfer không có tín hiệu sau 2–3 tuần đầu, nên thu bài thành empirical study về rare conditional generation thay vì tiếp tục gắn nhãn foundation model.

## 8. Bổ sung 2026-09-29: dynamic constraint graph

### 8.1 Hypothesis đã thu hẹp

Ý tưởng này đáng tiếp tục, nhưng cần phân biệt hai khái niệm thường bị nhập làm một:

- **static graph với state-dependent gradient**: constraint/loss và cấu trúc graph cố định, còn gradient đương nhiên thay đổi theo `x_t` hoặc `x_hat_0`; Harpoon và các feasibility operator đã bao phủ phần lớn trường hợp này;
- **dynamic constraint graph**: active nodes, typed edges, applicability, confidence và action scope thay đổi theo từng bản ghi và timestep.

Định nghĩa mục tiêu là:

`G_(i,t) = Select(G_prior, request_i, x_hat_0(i,t), trajectory_i, uncertainty_i)`.

`G_prior` có thể được LLM compile một lần từ schema, metadata và user condition, sau đó được type-check và kiểm chứng trên train data. Ở mỗi step được chọn, sampler chỉ kích hoạt subgraph liên quan đến bản ghi hiện tại. Node nên tách rõ:

- condition/predicate node;
- column/value node;
- applicability và exception node;
- action node: guide cột nào, mức bao nhiêu, defer, abstain hoặc rollback.

Typed edge tối thiểu gồm `requires`, `implies`, `conflicts`, `exception_to`, `depends_on` và `actuates`. Điểm mới tiềm năng không phải graph representation tự thân, mà là **row- and timestep-specific active graph** cùng selective actuation trong mixed-type diffusion.

### 8.2 Vị trí so với related work

- PromptLoop cung cấp tiền lệ feedback từ intermediate estimate nhưng cập nhật prompt trong image generation, không giải quyết typed tabular predicates, asymmetric dependencies, valid exceptions hoặc rare-mode coverage.
- SCTab-Diff dùng prior cố định nhưng denoising vẫn phụ thuộc row context. Vì vậy không được claim chung rằng “graph phụ thuộc current row” là mới; cần chứng minh graph structure/active set và action thay đổi có ích.
- Harpoon có constraint request cố định và gradient thay đổi theo state. Dynamic graph chỉ khác nếu nó thêm/bỏ node hoặc edge, thay đổi applicability hoặc abstain, chứ không chỉ thay gradient magnitude.
- CTDF áp feasibility operator ở mỗi reverse step. Đây là baseline bắt buộc cho hard executable constraints.
- LCATS/Zhao et al. mạnh ở discovery/validation và post-generation repair. Dynamic graph cần thắng ở fidelity, rare exception preservation hoặc sample cost; chỉ đạt zero violation không đủ.

Vai trò hợp lý của LLM là compile semantic graph và xử lý event khó/ambiguous, không phải thay deterministic executor ở mọi step. Probe trước cho thấy compile-once đạt execution accuracy 1.0 với khoảng 1.85k token, trong khi direct LLM judgement chỉ đạt 0.65–0.864 với khoảng 37.5k token. Vì vậy phiên bản mặc định nên là **LLM-compiled graph + deterministic dynamic executor**, sau đó mới ablate event-triggered LLM graph update.

### 8.3 Probe cơ chế vừa triển khai

Probe đầu tiên chưa được gọi là full semantic graph method. Nó kiểm tra câu hỏi hẹp hơn: dynamic active subgraph có cải thiện success–fidelity frontier so với fixed dependency graph hay không.

- Graph prior: edges từ condition columns tới candidate columns, trọng số bằng discretized NMI trên training split.
- Safety: loại supervised target và protected/sensitive columns khỏi automatic candidates, trừ khi user condition trực tiếp yêu cầu chúng.
- Dynamic state: ở mỗi reverse step và từng synthetic row, rank candidate edges bằng conditional denoiser response trên `x_hat_0`; categorical entropy làm uncertainty signal.
- Actuation budget: luôn giữ condition columns và chọn đúng `k` dependency edges. So sánh với fixed top-k graph có cùng `k`, backbone, adapter, seed, noise, steps và hai denoiser calls.
- Diagnostics: selection rate theo edge và early/middle/late phase, cùng mean edge-change rate giữa hai step liên tiếp. Điều này kiểm tra graph có thực sự động thay vì chỉ mang tên dynamic.

Grid gồm Default và Shoppers, numeric/mixed condition catalogs, ba seeds, ba schedules (`constant`, `step_release`, `midpulse`), `k ∈ {1,2,4}` và candidate pool `{4,8}`: tổng cộng 216 runs. Pool bằng active budget là sanity control gần tương đương fixed graph; pool lớn hơn mới kiểm tra row/timestep adaptation.

Jobs:

- smoke `1048623`;
- dynamic graph array `1048642[0-215]`;
- global temporal aggregate `1048643`;
- paired comparison với all-feature guidance `1048644`;
- paired dynamic-versus-fixed graph comparison `1048645`.

Array đầu `1048624` đã bị hủy và thay thế: numeric tasks dùng nhầm mixed-condition adapter 16 chiều cho condition vector 6 chiều. Smoke mixed không bị ảnh hưởng. Grid mới chọn đúng structured adapter cho numeric catalog và dense mixed adapter cho mixed catalog. Output chính dự kiến: `outputs/dynamic_graph_comparison_1048645/RESULTS.md` và `paired_rows.csv`.

Smoke đã hoàn tất sạch trong 56 giây. Trên Default mixed conditions, `midpulse`, active `k=2`, candidate pool 8, dynamic graph so với fixed top-2 graph cùng seed cho:

- success tăng `+0.0493`, `+0.0697`, `+0.0907`, `+0.1175` lần lượt ở q750/q900/q950/q975;
- expected denoiser calls per valid sample giảm `0.87–4.41`;
- KS tăng nhẹ `0.0045–0.0290`; TV thay đổi từ `-0.0045` đến `+0.0042`.

Mean edge-change rate giữa hai reverse steps liên tiếp khoảng `0.118`, xác nhận active graph thực sự thay đổi. Đây là tín hiệu tăng condition satisfaction/sample efficiency nhưng chưa phải Pareto dominance vì numeric fidelity xấu hơn nhẹ. Cần full grid và confidence interval theo seed trước khi kết luận.

Để giải phóng một GPU cho smoke, đã hủy Magic backbones `1048476–1048478` và Diabetes seed thứ ba `1048481` cùng sáu adapter jobs phụ thuộc seed đó. Hai Diabetes seeds còn lại, embeddings và aggregates vẫn được giữ. Không xóa checkpoint hoặc partial output.

### 8.4 Điều kiện quyết định

Tiếp tục thành full dynamic semantic graph nếu:

- dynamic graph cải thiện success ở cùng fidelity, hoặc fidelity ở cùng success, qua nhiều condition/seed;
- edge selection thực sự thay đổi theo row/timestep và thay đổi đó tương quan với reduced distortion;
- lợi ích còn tồn tại khi so với post-repair, CTDF-style projection và fixed top-k graph;
- oracle applicability/exception graph bảo toàn rare valid modes tốt hơn hard graph.

Nếu probe chỉ di chuyển dọc đúng trade-off curve của fixed graph, response-driven topology chưa đủ. Bước tiếp theo khi đó không phải thêm LLM, mà là kiểm tra oracle graph với bốn trạng thái `violated`, `satisfied`, `not applicable`, `valid exception`. Nếu oracle cũng không thắng, nên dừng claim mid-denoising dynamic graph. Nếu oracle thắng nhưng learned/LLM selector không đạt, contribution có thể chuyển thành benchmark và calibrated graph selection.

Nếu probe dương, method đầy đủ nên phát triển theo ba tầng:

1. LLM compile natural-language request/schema thành typed graph một lần;
2. deterministic verifier cập nhật active predicate/exception nodes từ intermediate trajectory;
3. LLM chỉ được gọi lại khi verifier uncertainty cao, graph conflict hoặc unseen semantic relation xuất hiện.

Claim dự kiến khi đó là: **verified dynamic semantic graph for selective mixed-type diffusion guidance**, không phải “LLM feedback trong diffusion lần đầu tiên”.
