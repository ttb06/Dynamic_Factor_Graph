# Trạng thái thực nghiệm và baseline — 2026-10-17

## 1. Kết luận ngắn

- Middle-route/late-execute trên năm documented order programs cải thiện Trend ổn định qua ba seed, nhưng không trội late-only trên mọi metric. Mean gain so với TabDiff là `+0.505` điểm Trend, `+0.048` Alpha và `+0.573` Beta; Shape giảm `0.022` điểm.
- Mean regression gain lớn bị một seed heavy-tail chi phối. Median paired gain đáng tin hơn là `+0.127 R²` và `-0.0458 NRMSE`.
- Soft statistical graph được LLM filtering thật sự tác động lên trajectory. Variant Luna strength 0.05 tăng thêm `+0.279` Trend so với route-only, nhưng Shape giảm `0.033` và detection giảm `0.0034`; đây mới là single-seed evidence.
- Unfiltered graph có Trend gần Luna-filtered nhưng Shape và utility kém hơn. Điều này ủng hộ semantic filtering ở mức sơ bộ, chưa đủ thành claim chính.
- Trong public-constraint News setting, dynamic middle + certifier tốt hơn final repair về Trend và regression utility, nhưng final repair vẫn mạnh và tốt hơn ở một số fidelity/coverage metric.
- Official TabSyn, ForestDiffusion, Harpoon, LCATS, SAGE, SPADA và GReaT đều đã sinh output. Common evaluation của SAGE/SPADA/GReaT trước đây bị cancel; đã submit lại không dependency.
- Matched 4,096-row reevaluation của generator baselines đã hoàn thành. Default detector hội tụ; Shoppers vẫn chạm giới hạn 2,000 iterations nên cột detection Shoppers chưa được report.

## 2. Kết quả main mechanism, News, ba seed

| Method | Shape ↑ | Trend ↑ | Alpha ↑ | Beta ↑ | Detection ↑ | R² ↑ | NRMSE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| TabDiff | 96.690±0.108 | 90.215±0.045 | 97.082±0.664 | 43.605±0.623 | 0.8116±0.0057 | -1.288±1.423 | 1.463±0.473 |
| Late rank | 96.664±0.114 | 90.663±0.334 | 97.151±0.705 | **44.492±0.520** | **0.8103±0.0083** | -0.487±0.432 | 1.210±0.183 |
| Middle route q=.35 | **96.668±0.115** | **90.720±0.362** | 97.130±0.672 | 44.178±0.224 | 0.8094±0.0046 | **-0.485±0.426** | **1.210±0.180** |
| Middle route q=.24 | 96.666±0.111 | 90.668±0.337 | **97.159±0.714** | 44.437±0.390 | 0.8066±0.0055 | -0.493±0.445 | 1.212±0.187 |

Mean delta của route q=.35 so với TabDiff:

- Shape `-0.0225` điểm;
- Trend `+0.5046`;
- Alpha `+0.0481`;
- Beta `+0.5729`;
- Detection `-0.0023`;
- R² `+0.8031`, nhưng median paired delta chỉ `+0.1273`;
- NRMSE `-0.2528`, median paired delta `-0.0458`.

Route q=.35 chỉ hơn late rank `0.0568` Trend trung bình và kém `0.314` Beta. Vì vậy evidence hiện chứng minh middle-state routing hữu ích, nhưng chưa chứng minh nó Pareto-dominate một late-only rank mechanism.

## 3. LLM-filtered soft factors, News, single seed

| Variant | Shape ↑ | Trend ↑ | Alpha ↑ | Beta ↑ | Detection ↑ | R² ↑ | NRMSE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Route only | 96.5718 | 90.4366 | 96.3538 | 43.9258 | 0.7995 | -0.0293 | 1.0146 |
| Consensus soft, .02 | 96.5473 | 90.5568 | 96.4792 | 44.0267 | **0.8013** | -0.0383 | 1.0190 |
| Consensus soft, .05 | 96.5315 | 90.6671 | **96.5225** | 43.8737 | 0.8018 | -0.0351 | 1.0174 |
| Consensus soft, .10 | 96.4821 | 90.5472 | 96.5595 | 43.9648 | 0.7931 | **-0.0211** | **1.0105** |
| Luna soft, .05 | **96.5390** | **90.7152** | 96.5095 | 43.9925 | 0.7961 | -0.0252 | 1.0125 |
| Unfiltered soft, .05 | 96.4262 | 90.7089 | 96.4526 | **44.0723** | 0.7954 | -0.0326 | 1.0162 |

So với unfiltered graph, Luna-filtered graph tăng Shape `0.113`, Trend `0.006`, Alpha `0.057`, detection `0.0007`, R² `0.0074` và giảm NRMSE `0.0037`; đổi lại Beta giảm `0.080`. Đây là evidence có ích cho filtering, nhưng cần state-selective jobs và thêm seed trước khi report như main result.

## 4. Public hard constraints trên News, cùng TabDiff sample

| Method | Shape ↑ | Trend ↑ | Alpha ↑ | Beta ↑ | R² ↑ | NRMSE ↓ | Violation ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| TabDiff | **96.6554** | 89.1527 | 97.1327 | 42.8825 | -0.0254 | 1.0126 | 25.8789% |
| Final repair | 96.5951 | 90.2825 | 97.0486 | **43.0046** | -0.0162 | 1.0081 | **0%** |
| ConstrainedDGM final layer | 96.2996 | 90.5950 | **97.1344** | 42.8988 | -0.0293 | 1.0146 | **0%** |
| Full per-step projection + certifier | 96.5734 | **90.7079** | 97.0812 | 42.8809 | -0.0206 | 1.0102 | 0.0488% |
| Dynamic middle + certifier | 96.5864 | 90.6927 | 97.1024 | 42.8711 | **-0.0054** | **1.0027** | **0%** |

Dynamic middle + certifier so với final repair tăng Trend `0.410`, R² `0.0108` và giảm NRMSE `0.0054`; Shape giảm `0.0087`, Beta giảm `0.1335`. Đây là matched constraint-guidance comparison mạnh nhất hiện có.

### Soft graph làm action scorer, single seed

Job `1091598_0` đã hoàn tất. So với đúng unconditional sample, scorer tăng Trend từ `90.2665` lên `90.5869` (`+0.3204`) và Beta từ `42.9134` lên `44.0479` (`+1.1344`), nhưng giảm Shape `0.0335`, Alpha `0.0369`, R² `0.0086` và tăng NRMSE `0.0042`. Public-constraint violation không đổi (`26.2939%`), đúng với vai trò scorer/veto chứ không phải constraint repair.

So với route-only cùng seed, gain riêng của scorer nhỏ hơn: `+0.1503` Trend và `+0.1221` Beta, trong khi R² giảm `0.0121`. Kết luận tạm thời: scorer giúp coverage/dependence nhưng chưa đạt Pareto improvement; các slack variants đang chạy để kiểm tra trade-off.

## 5. Generator baselines đã hoàn thành

Các số dưới đây là job `1091649`: cùng local split, cùng evaluator, cùng graph và đúng 4,096 synthetic rows cho mọi method.

### Default

| Method | Shape error ↓ | Trend error ↓ | Alpha ↑ | Beta ↑ | XGB AUC ↑ |
|---|---:|---:|---:|---:|---:|
| TabDiff | 2.3851 | 2.5518 | 96.2240 | **53.3024** | **0.7175** |
| TabSyn official | **1.4810** | **1.6499** | **99.2198** | 47.7816 | 0.7127 |
| ForestDiffusion official | 8.9813 | 6.1003 | 94.4882 | 47.8939 | 0.6968 |
| CTGAN-SDV | 15.2233 | 17.8281 | 64.4922 | 21.3704 | 0.6695 |
| TVAE-SDV | 10.8799 | 6.5209 | 82.3844 | 25.9066 | **0.7244** |

### Shoppers

| Method | Shape error ↓ | Trend error ↓ | Alpha ↑ | Beta ↑ | XGB AUC ↑ |
|---|---:|---:|---:|---:|---:|
| TabDiff | **1.6044** | 2.5302 | 96.3738 | **50.3320** | 0.8978 |
| TabSyn official | 1.6652 | **1.9929** | **99.0039** | 46.3363 | 0.8817 |
| ForestDiffusion official | 14.5687 | 4.6032 | 91.0807 | 49.7168 | **0.8992** |
| CTGAN-SDV | 14.2642 | 15.4354 | 83.8363 | 31.6911 | 0.8013 |
| TVAE-SDV | 16.6945 | 9.4195 | 53.8737 | 18.1136 | 0.8773 |

TabSyn là generator baseline cạnh tranh nhất về marginal/joint fidelity trên Default và về Trend/Alpha trên Shoppers. TabDiff giữ coverage tốt hơn và tốt hơn TabSyn về Shape/utility trên Shoppers. ForestDiffusion không cạnh tranh về Shape/Trend trong các run này.

## 6. Constraint/LLM baselines đã hoàn thành

### Harpoon official

Official public code/checkpoint đã chạy năm trial, 200 reverse steps. Metric scale và request constraints là của Harpoon, không so trực tiếp với News graph. Ví dụ:

| Dataset/request | Violation field ↓ | Alpha ↑ | KS ↑ | Detection ↑ | XGB utility ↑ |
|---|---:|---:|---:|---:|---:|
| Adult/range | 28.0740 | 0.9220 | 0.9519 | 0.8684 | 0.9498 |
| Default/range | 4.8079 | 0.9003 | 0.9417 | 0.7988 | 0.7809 |
| Default/category | 17.6471 | 0.9478 | 0.9363 | 0.9284 | 0.7697 |
| Shoppers/category | 0.0000 | 0.5999 | 0.8974 | 0.8484 | 0.8741 |

Harpoon là official mechanism baseline nhưng cần matched proxy trên cùng catalog để xếp hạng trực tiếp.

### LCATS official, Insurance

| Method | Any violation ↓ | Numeric KS ↓ | Corr. MAE ↓ | TSTR R² ↑ | Row edit ↓ | Cell edit ↓ |
|---|---:|---:|---:|---:|---:|---:|
| TabSyn original | 0.9269 | 0.0370 | 0.0118 | **0.7239** | — | — |
| LCATS aligned | **0** | **0.0299** | **0.0115** | 0.7153 | 1.0000 | 0.2849 |

LCATS loại violation hoàn toàn và cải thiện KS, nhưng giảm R² `0.0086`, sửa mọi row và 28.49% cells. Ba LLM weight profiles cho cùng output vì graph release này chỉ gồm unary bounds/integer rules.

### SAGE, SPADA, GReaT

- SAGE GPT-2 và Qwen3-0.6B: đã train và sinh đủ hai variant, mỗi variant 4,096 rows.
- SPADA: đã sinh 4,096 rows từ 22-edge official demonstration graph được đổi tên cột cơ học.
- GReaT DistilGPT-2: đã train 10 epochs và sinh 4,096 rows.
- Common evaluator cũ bị cancel do dependency; job `1091644[0-5]` đã được submit lại độc lập.
- SAGE cần một lưu ý phương pháp quan trọng: protocol công bố khởi tạo khoảng 70% giá trị của mỗi row từ real training row trước khi model điền phần còn lại. Vì vậy cần đọc fidelity cùng DCR/copying audit, không xếp hạng chỉ bằng Shape/Trend.

## 7. Jobs hiện tại

| Job | Trạng thái tại thời điểm audit | Mục tiêu |
|---|---|---|
| `1091598_0` | running | Control soft graph action scorer, zero slack |
| `1091655[1-3]` | submitted, prioritized | Soft graph làm action scorer/veto; bỏ các unfiltered variants lặp lại |
| `1091656[0-5]` | submitted, prioritized | Residual/uncertainty state-selective gate |
| `1091657[0-2]` | submitted, prioritized | Contribution trực tiếp của verified LLM edges |
| `1091658[1-5]` | submitted, prioritized | Positive messages trên satisfied LLM edges; bỏ route-only control đã có |
| `1091669[0-2]` | submitted, prioritized | Tách weight của satisfied-state message khỏi violation correction: 0/.25/.50 |
| `1091644[0-2]` | task 0 completed, 1–2 finishing | Common evaluation cho ba SAGE variants; phần còn lại đã dừng để nhường compute cho method |
| `1091649[0-1]` | completed | Matched 4,096-row reevaluation của generator baselines |

Không có job nào trong nhóm hiện tại bị `DependencyNeverSatisfied`; chúng chỉ chờ priority/resources.

### Kiểm tra cơ chế sớm từ trace của lượt prioritized

Các trace đã được ghi trước khi common evaluator hoàn tất:

| Variant | Factor active fraction | Satisfied active pairs | Violating active pairs | Factor cell events | Standardized movement |
|---|---:|---:|---:|---:|---:|
| Residual gate 0.25 | 0.7487 | 670,911 | 34,395 | 361,656 | 9,645.53 |
| Residual gate 0.50 | 0.5540 | 494,180 | 27,765 | 261,372 | 7,920.16 |
| Residual gate 1.00 | 0.2639 | 232,107 | 16,499 | 116,080 | 4,621.92 |

Điều này xác nhận implementation state-selective hoạt động đúng: threshold thay đổi active graph theo state, và intervention budget giảm đơn điệu khi gate chặt hơn. Quan trọng hơn, khoảng 95–96% active row-factor pairs là các cạnh đang thỏa; graph hiện truyền positive messages chứ không chỉ sửa violation. Kết quả fidelity/utility tương ứng vẫn đang chạy nên chưa kết luận gate nào tốt nhất.

LLM-only verified edge variant cũng đã thực sự active: `1,092` projection events trong một trajectory, thay vì chỉ tồn tại trong graph catalog. Common metrics đang được tính.

Positive-message branch trên riêng verified LLM factors cũng đã chạy qua sampler:

| Variant | Active fraction | Satisfied active pairs | Violating active pairs | Cell events | Standardized movement |
|---|---:|---:|---:|---:|---:|
| LLM positive strength .02 | 1.0000 | 179,381 | 9,035 | 6,762 | 455.44 |
| LLM positive strength .05 | 1.0000 | 179,789 | 8,627 | 10,772 | 1,141.28 |
| LLM positive .05 + residual/uncertainty gate | 0.4828 | 85,342 | 5,629 | 5,777 | 648.33 |

Như vậy verified LLM relations không chỉ được dùng khi vi phạm: hơn 94% active pairs trong ba cấu hình là satisfied-state messages. Gate phụ thuộc state loại khoảng 52% candidate activations và giảm movement khoảng 43% so với strength .05 không gate. Đây mới là kiểm tra cơ chế; cần metric cuối để biết selective intervention có bảo toàn fidelity/utility tốt hơn không.

Vì satisfied pairs chiếm đa số lớn, implementation mới tách `satisfied_strength_scale` và `violating_strength_scale`. Job `1091669` giữ violation weight bằng 1 và quét positive weight `0/.25/.50`; cấu hình positive weight 1 đã có ở `1091658_3`. Ablation này trực tiếp kiểm tra positive messages có thêm giá trị hay chỉ tạo over-correction.

## 8. Kết luận cho paper

1. Có thể dùng public-constraint News table làm main mechanism table ngay, với final repair, ConstrainedDGM layer và full per-step projection.
2. Chưa nên claim SOTA generator fidelity: TabSyn tốt hơn TabDiff ở một số Shape/Trend metric và các pipeline không hoàn toàn cùng backbone.
3. Main contribution hợp lý vẫn là trajectory-dependent graph activation và trade-off fidelity/utility/intervention, không phải absolute Shape SOTA.
4. LLM branch hiện có evidence về graph coverage và filtering; chưa có evidence đủ mạnh rằng direct LLM factors thắng statistical graph. Positive-message và state-selective jobs là phép thử quyết định.
5. Detection của Default trong job `1091649` đã hội tụ và có thể dùng. Detection của Shoppers và các LLM baseline Adult đang chạm giới hạn `2,000` iterations; không report các giá trị đó cho tới khi có một lượt đánh giá detector hội tụ đồng nhất.
6. Hai lượt train baseline News (`1091652`, `1091653`) đã được chủ động huỷ trước khi chạy để dành GPU cho các phép thử method. Chúng vẫn là việc cần làm trước bản paper cuối, nhưng không nằm trên critical path của vòng hiện tại.

## 9. Kế hoạch đóng khung method trong 48 giờ

### Đánh giá hiện tại

Method đã đủ bằng chứng để đóng khung về mặt cơ chế, nhưng chưa đủ để claim SOTA về fidelity. Kết quả mạnh nhất hiện là dynamic middle guidance tăng Trend và regression utility so với final repair; kết quả ba seed cũng tăng Trend/Beta so với TabDiff. Tuy nhiên gain còn nhỏ, middle routing chưa trội late-only trên mọi metric, và contribution riêng của LLM chưa được chứng minh dứt khoát.

### Method sẽ đóng khung

1. Một verified typed graph được xây từ documented rules, train-fitted relations và optional LLM proposals.
2. Ở middle trajectory, clean preview được dùng để tính applicability, conditional residual, uncertainty và temporal stability cho từng row-edge pair.
3. Active graph thay đổi theo row và reverse state.
4. Action layer dùng distribution-preserving factor messages cho dependence đã thỏa, và typed projection cho hard violations.
5. Late certifier là safety layer tùy chọn, không phải nguồn novelty chính.

### 0–12 giờ: chọn duy nhất một cấu hình main

- Hoàn tất `1091655–1091669`.
- Xếp hạng trên cùng seed bằng Shape, Trend, Alpha, Beta, R², NRMSE và standardized movement.
- Chỉ giữ một trong ba họ: state-gated factors, LLM-positive gated factors, hoặc hybrid positive + projection.
- Không thêm action type hoặc graph learner mới.

Tiêu chí chọn: ưu tiên Pareto improvement; nếu không có, chọn cấu hình tăng Trend/Beta/utility trong khi Shape giảm không quá `0.05` điểm. Nếu hai cấu hình gần bằng nhau, chọn cấu hình có movement và active fraction thấp hơn.

### 12–30 giờ: xác nhận, không tuning rộng

- Chạy ba seed cho cấu hình thắng và bốn đối chứng bắt buộc: TabDiff, final repair, full per-step static projection, late-only guidance.
- Giữ graph, sample count và seed pairing giống nhau.
- Báo mean±std cùng paired delta; utility dùng median paired delta nếu có heavy-tail.
- Chạy một smoke transfer trên Shoppers và Default, mỗi dataset một seed. Mục tiêu là kiểm tra mechanism không chỉ work trên News, không dùng hai dataset này để tune lại threshold.

### 30–48 giờ: freeze

- Nếu dynamic method hơn late-only ở ít nhất một fidelity/dependence metric và không giảm utility đáng kể, freeze method hiện tại.
- Nếu quality tương đương nhưng giảm ít nhất 30% active messages/movement, đóng claim theo selective efficiency.
- Nếu cả quality lẫn intervention efficiency đều không hơn late-only, bỏ positive-message branch và freeze bản middle-route + typed projection đơn giản hơn.
- Sau thời điểm freeze, chỉ sửa lỗi hoặc chuẩn hóa evaluator; không đổi architecture.

## 10. Thực nghiệm làm dày sau khi freeze

1. **Main matched comparison:** TabDiff, final repair, best-of-N, ConstrainedDGM layer, full per-step static guidance, late-only, dynamic graph, Harpoon-style differentiable proxy.
2. **Cross-dataset:** 4–6 datasets, bao phủ numerical-heavy, mixed-type, nhiều categorical và các quan hệ có điều kiện.
3. **Temporal ablation:** early/middle/late/full trajectory; fixed schedule so với state-dependent activation.
4. **Graph-source ablation:** documented only, statistics only, verified LLM only, union, và unfiltered proposals.
5. **Component ablation:** applicability, residual gate, uncertainty gate, temporal stability, positive message, hard projection, certifier.
6. **Robustness:** thêm 5/10/20% cạnh sai hoặc giảm confidence để đo khả năng abstain và mức hỏng fidelity.
7. **Mechanism analysis:** activation heatmap theo timestep, fraction active, satisfied/violating messages, row movement, edge survival và examples theo trajectory.
8. **Efficiency:** runtime, memory, số graph evaluations, number of modified cells và quality gain trên mỗi intervention.
9. **Standard quality:** Shape/Trend, Alpha/Beta, KS/correlation, TSTR/TRTS utility, DCR/privacy; constraint satisfaction chỉ là metric phụ khi constraint không phải ground truth.
10. **LLM value audit:** coverage/novel relation types, acceptance rate sau verification, gain của verified LLM graph so với graph không LLM, token/runtime cost.

## 11. Finalist screen và portability sang TabSyn

### Single-seed paired deltas mới nhất trên News

| Variant | ΔShape ↑ | ΔTrend ↑ | ΔAlpha ↑ | ΔBeta ↑ | ΔR² ↑ | ΔNRMSE ↓ | ΔViolation điểm ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Luna graph action scorer | -0.0197 | +0.3443 | -0.0960 | +0.8301 | -0.0075 | +0.0037 | 0.0000 |
| State gate, residual 1.0 | -0.0277 | +0.2903 | +0.2423 | **+1.3135** | -0.0084 | +0.0041 | 0.0000 |
| Verified LLM edges only | **+0.0023** | **+0.6447** | **+0.0065** | +0.0602 | **-0.0019** | **+0.0009** | **-3.0762** |
| Order + LLM projection .25 | -0.0199 | +2.1993 | -0.1009 | +0.9554 | -0.1514 | +0.0526 | +0.0488 |
| LLM positive message .02 | -0.0006 | +0.1173 | -0.1286 | +1.0889 | -0.1374 | +0.0479 | -0.4883 |

Hai finalist ban đầu là verified LLM-edge-only và state gate residual 1.0. Positive-message-only và strong order-projection variants tăng một vài dependence metrics nhưng giảm utility quá nhiều nên không phù hợp làm main method.

### Xác nhận ba seed đã hoàn tất

| Finalist | ΔShape ↑ | ΔTrend ↑ | ΔAlpha ↑ | ΔBeta ↑ | ΔR² ↑ | ΔNRMSE ↓ | ΔViolation điểm ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Verified LLM edges only | -0.0136±0.0151 | **+1.5329±0.8243** | -0.0060±0.0108 | +0.0184±0.0377 | **+0.0059±0.0102** | **-0.0027±0.0046** | **-3.3447±0.8974** |
| State gate residual 1.0 | -0.0271±0.0008 | +0.4273±0.1638 | +0.0517±0.1663 | **+0.7780±0.4713** | -0.0071±0.0026 | +0.0035±0.0013 | +0.0244±0.0423 |

Verified LLM-edge dynamic routing hiện là main candidate: Trend tăng ở cả ba seed, mean utility tăng và constraint violations giảm rõ, trong khi Shape/Alpha/Beta gần như giữ nguyên. State-gated soft factors phù hợp làm ablation về coverage, không phải main quality result.

Một hybrid variant (`hybrid_s05_gate`) đạt single-seed ΔTrend `+0.8925`, ΔAlpha `+0.0244`, ΔBeta `+0.9050`, ΔR² `-0.0071`, Δviolation `-3.5889`; hai seed xác nhận được submit ở `1092101`.

### TabSyn có mạnh hơn TabDiff không?

Matched local evaluation cho thấy TabSyn mạnh hơn chủ yếu về marginal/joint fidelity, không phải trên mọi metric:

| Dataset | Backbone | Shape error ↓ | Trend error ↓ | Alpha ↑ | Beta ↑ | Utility ↑ |
|---|---|---:|---:|---:|---:|---:|
| Default | TabDiff | 2.3851 | 2.5518 | 96.2240 | **53.3024** | **0.7175** |
| Default | TabSyn | **1.4810** | **1.6499** | **99.2198** | 47.7816 | 0.7127 |
| Shoppers | TabDiff | **1.6044** | 2.5302 | 96.3738 | **50.3320** | **0.8978** |
| Shoppers | TabSyn | 1.6652 | **1.9929** | **99.0039** | 46.3363 | 0.8817 |

TabSyn tạo headline fidelity tốt hơn, nhưng có headroom nhỏ hơn và coverage/utility thấp hơn. Vì vậy port method sang TabSyn có giá trị nếu guidance tăng Trend/Beta/utility mà giữ Shape/Alpha, không chỉ để lấy absolute number cao.

### Portability experiment

Job `1091690[0-3]` chạy training-free dynamic graph guidance bên trong official TabSyn latent trajectory trên Shoppers. Clean latent được decode ở middle/late windows, ba independently checkable relations được project theo active row state, rồi correction được re-encode và giới hạn bằng latent trust radius. Sweep gồm middle strengths `.05/.10/.20` và late-only `.10`.

Initial TabSyn results cho thấy middle strength `.20` là điểm tốt nhất: Shape error `-0.0109`, Trend error `-0.0083`, Beta `+0.0049`, XGB AUC `+0.0023`, đổi lại Alpha `-0.0244`. Effect còn nhỏ vì chỉ khoảng 595 row-step events. Job `1092099` đang thử strength `.40/.80` và bỏ persistence gate để tăng intervention coverage; late-only cải thiện Shape/Beta nhưng làm AUC giảm nên không phải candidate chính.

Focused optimization đang chạy: `1092100` quét route time/strength quanh verified-edge winner trên một seed mới; chỉ cấu hình thắng mới được xác nhận thêm seed. Không có module mới được thêm.

### Add-on hay end-to-end

Main paper nên giữ method ở dạng training-free add-on trên ít nhất TabDiff và TabSyn. Điều này tách contribution của dynamic graph khỏi retraining và tạo matched comparison rõ. End-to-end graph-conditioned training có thể tăng số liệu nhưng làm scope và confounding tăng mạnh; không nên đưa vào critical path 48 giờ.

Sau khi freeze add-on, nhánh end-to-end hợp lý nhất là guidance distillation: tạo corrected latent target bằng chính action layer và fine-tune denoiser/adapter với auxiliary latent consistency loss. Không nên xây một graph neural network conditioning architecture mới ở giai đoạn này.

## 12. Cập nhật focused optimization: route time, strength và cross-backbone

### 12.1. Sweep route time/strength trên News

Job `1092100[0-5]` đã hoàn tất trên cùng seed `20261607`. Tất cả số dưới đây là paired delta so với unconditional sample được sinh trong chính run đó. Shape/Trend/Alpha/Beta/R² càng cao càng tốt; NRMSE và violation càng thấp càng tốt.

| Route quantile | Strength | ΔShape | ΔTrend | ΔAlpha | ΔBeta | ΔR² | ΔNRMSE | ΔViolation điểm |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| .20 | 1.00 | -0.0342 | +1.2323 | -0.0016 | +0.0391 | -0.0090 | +0.0042 | -5.8594 |
| .275 | 1.00 | -0.0342 | +1.2323 | -0.0016 | +0.0391 | -0.0183 | +0.0085 | -5.7617 |
| .425 | 1.00 | -0.0179 | +1.2323 | +0.0016 | +0.0407 | -0.0219 | +0.0101 | -2.2949 |
| .50 | 1.00 | **-0.0025** | +1.8236 | +0.0016 | -0.0163 | +0.0013 | -0.0007 | -0.6836 |
| .35 | .50 | -0.0209 | +2.3035 | -0.0033 | -0.0163 | +0.0002 | -0.0001 | 0.0000 |
| .35 | .75 | -0.0249 | **+2.3229** | -0.0016 | -0.0098 | **+0.0025** | **-0.0013** | 0.0000 |

Với baseline Trend score `90.5248`, route `.35/.75` tăng lên `92.8477`. Tương đương Trend error giảm từ `9.4752` xuống `7.1523`, tức **24.5% relative error reduction** trên screening seed. Soft projection chưa đẩy record qua hẳn biên bất đẳng thức nên violation count không đổi, nhưng nó cải thiện dependence fidelity và regression utility. Đây là bằng chứng rằng graph message có ích ngoài chức năng hard repair.

Trong quy ước này `q=1` là noisy start và `q=0` là clean end; do đó `.50` là earlier-middle còn `.35` là later-middle. Actuation sau khi route được áp ở late window `q∈[0,.15]`. Route `.20–.425` giảm violation nhiều hơn nhưng phần lớn làm utility xấu đi. Route `.50/1.0` và `.35/.75` là hai Pareto candidates: một điểm bảo toàn Shape tốt hơn, một điểm tăng Trend mạnh hơn. Job `1093404[0-3]` đã được submit để xác nhận hai cấu hình này trên seeds `20261605/20261606`; bốn task đã vào chạy và không có dependency. Đây là xác nhận hai middle-state targets, không được gọi nhầm là middle-vs-late ablation.

Trace cung cấp bằng chứng trực tiếp rằng active graph thay đổi theo diffusion state:

| Route state | Stable/active row-edge fraction | Late projection events |
|---:|---:|---:|
| .50 | 0.1469 | 266 |
| .425 | 0.4255 | 749 |
| .35 | 0.7750 | 1,757 |
| .275 | 0.9639 | 1,925 |
| .20 | 0.9946 | 1,953 |

Khi preview tiến gần clean end, nhiều row-edge signals vượt stability gate hơn. Tuy nhiên coverage gần 100% ở `.20/.275` đi kèm utility loss; graph dày hơn không đồng nghĩa tốt hơn. Route `.35` tạo coverage trung gian, còn strength `.75` thực hiện partial transport thay vì hard repair. Đây là mechanism evidence cho state-dependent graph activation, không chỉ là kết quả tuning timestep.

### 12.2. Hybrid graph qua ba seed

Hybrid gồm verified LLM directional edges, statistical-order projections, gated satisfied-edge messages và late certifier. Ba paired seeds đã hoàn tất:

| Metric | Mean paired delta ± std |
|---|---:|
| Shape | -0.0382 ± 0.0187 |
| Trend | **+1.7405 ± 0.9510** |
| Alpha | -0.0087 ± 0.0416 |
| Beta | **+0.6689 ± 0.3619** |
| Regression R² | +0.0010 ± 0.0181 |
| Regression NRMSE | -0.0004 ± 0.0088 |
| Any-violation rate | **-4.5410 ± 0.8503 điểm** |

Mean Trend error giảm khoảng **17.6%**. Hybrid mạnh hơn verified-LLM-only ở Beta và violation reduction, nhưng Shape giảm nhiều hơn và utility variance lớn hơn. Do đó hiện có hai cách trình bày hợp lệ:

1. `verified LLM edge only` là cấu hình bảo toàn chất lượng/utility an toàn nhất;
2. `hybrid` là cấu hình đa mục tiêu, tăng Trend/Beta và giảm violation mạnh hơn.

Không nên chọn sau khi nhìn riêng một seed. Kết quả xác nhận soft route sẽ quyết định main configuration.

### 12.3. Negative result quan trọng: broad statistical-order projection

Job `1092114[0-3]` giảm statistical-order weight xuống `.05–.15` và satisfied-message strength xuống `0–.02`. Trend tăng `+1.42` đến `+1.67`, Beta tăng `+0.48` đến `+0.88`, nhưng ΔR² nằm trong `[-0.93,-0.80]`. Như vậy bottleneck không phải đơn giản là weight quá lớn: broad order graph thay đổi các feature liên quan target theo cách phá downstream utility. Nhánh này bị loại khỏi main method.

Kết quả này củng cố nhu cầu active graph có chọn lọc theo row/state. Việc mọi edge dùng cùng route hoặc cùng projection window không an toàn, kể cả strength rất nhỏ.

### 12.4. TabSyn portability và giới hạn gain

| TabSyn variant, Shoppers | ΔShape error ↓ | ΔTrend error ↓ | ΔAlpha | ΔBeta | ΔAUC |
|---|---:|---:|---:|---:|---:|
| Middle .05, persistence | -0.0041 | -0.0016 | -0.0033 | 0.0000 | -0.0047 |
| Middle .10, persistence | -0.0054 | -0.0028 | -0.0049 | -0.0163 | -0.0078 |
| Middle .20, persistence | **-0.0109** | -0.0083 | -0.0244 | +0.0049 | **+0.0023** |
| Middle .40, persistence | -0.0095 | -0.0138 | -0.0293 | +0.0049 | -0.0007 |
| Middle .80, persistence | -0.0095 | **-0.0155** | -0.0342 | -0.0244 | -0.0098 |
| Middle .20, no persistence | -0.0081 | -0.0065 | -0.0016 | -0.0049 | -0.0059 |
| Middle .40, no persistence | -0.0091 | -0.0143 | -0.0260 | +0.0081 | -0.0082 |
| Late .10, persistence | -0.0148 | -0.0067 | -0.0293 | +0.0163 | -0.0049 |

Middle `.20` với persistence gate là điểm cân bằng duy nhất cải thiện đồng thời Shape error, Trend error và AUC. Tăng compute/intervention không tạo gain đơn điệu: `.80` và no-persistence đều làm AUC giảm. TabSyn vì vậy phù hợp làm portability evidence cho training-free add-on, chưa phải nơi tối ưu headline gain.

### 12.5. Quyết định sau vòng này

- Giữ main paper ở dạng **training-free add-on**: đã chạy thật trên cả TabDiff và official TabSyn trajectory.
- Main claim nên tập trung vào state-dependent route/activation và fidelity–utility trade-off, không claim absolute generator SOTA ở thời điểm này.
- Đóng broad statistical-order branch; không tiếp tục sweep weight.
- Chỉ xác nhận hai soft route candidates ở `1093404`, sau đó freeze inference mechanism.
- End-to-end training chỉ nên tiếp tục sau freeze dưới dạng distillation/adapter, và phải so với chính add-on teacher bằng paired metrics.

## 13. Xác nhận soft routing và tự động hóa interval

### 13.1. Ba-seed route confirmation

Job `1093404[0-3]` đã hoàn tất. Lưu ý `q=1` là noisy start và `q=0` là clean end, nên `.50` xảy ra sớm hơn `.35`.

| Route / strength | ΔShape | ΔTrend | ΔAlpha | ΔBeta | ΔR² | ΔNRMSE | ΔViolation điểm |
|---|---:|---:|---:|---:|---:|---:|---:|
| Later-middle `.35/.75` | -0.0264±0.0026 | **+2.2451±0.2747** | -0.0033±0.0059 | -0.0071±0.0124 | -0.0073±0.0097 | +0.0031±0.0042 | 0.0000±0.0000 |
| Earlier-middle `.50/1.0` | **-0.0022±0.0006** | +1.3549±1.1918 | 0.0000±0.0016 | -0.0109±0.0050 | -0.0047±0.0065 | +0.0023±0.0032 | -0.5778±0.1101 |

Route `.35/.75` tăng Trend ở cả ba seed (`+2.323`, `+2.473`, `+1.940`) và giảm mean Trend error khoảng **23.1%**. Utility giảm nhẹ trung bình, còn Shape/Alpha/Beta gần như giữ nguyên. Route `.50` bảo toàn Shape tốt hơn nhưng Trend gain có variance lớn và bằng zero ở một seed. Vì vậy `.35/.75` là quality-optimized route; `.35/1.0` trước đó vẫn là balanced route vì mean R² tăng và violation giảm qua ba seed.

### 13.2. Vì sao cần tự động hóa

NeurIPS 2024 Limited Interval Guidance dùng piecewise-constant guidance trong một khoảng noise, nhưng vẫn xem hai biên interval là hyperparameters. Nếu paper này chỉ chọn `.35` sau một test sweep, reviewer có thể xem gain là dataset-specific tuning.

Implementation mới thêm `--projection-route-mode auto_knee`:

1. Ở mỗi reverse state trước late actuation, tính stable row-edge fraction bằng thay đổi normalized rank của typed edge signal giữa hai clean previews liên tiếp.
2. Lưu graph snapshot gồm program signals và row-level stable masks ở mỗi state.
3. Khi bước vào actuation window, dựng monotone readability curve và chọn endpoint có độ dốc lớn nhất; đây là state nơi graph trở nên readable nhanh nhất.
4. Chỉ graph snapshot được chọn mới đi vào late action layer. Selector không đọc real test rows, Shape, Trend, utility hay constraint score cuối.

Job `1093489[0-1]` đã được submit để thử auto-selected route với strengths `.75/1.0` trên screening seed. Nếu selector chọn quanh `.35` và giữ gain, bước tiếp theo chỉ là xác nhận hai seed, không quét route time nữa.

Mechanism trace đã được ghi trước khi common evaluator hoàn tất: global `auto_knee` chọn `q=0.3703`, active row-edge fraction `0.6230`, trong khi fixed sweep tốt nhất nằm tại `.35`. Hai jobs dùng cùng seed nên đây mới là kiểm tra tính hợp lý của selector, chưa phải robustness qua seed; tuy nhiên selector không nhận `.35` hay metric cuối làm input.

Implementation tiếp theo `auto_edge_knee` chọn knee riêng cho từng typed edge, vì các relations có thể trở nên readable ở noise levels khác nhau. Job `1093501` kiểm tra per-edge selector ở strength `.75`. Nếu per-edge và global selector ngang nhau trên graph hai cạnh hiện tại, global mode đủ cho main table còn per-edge mode nên được giữ cho graph lớn/cross-dataset.

Trace sớm của `1093501` xác nhận edge-specific timing thực sự khác nhau: `num_self_hrefs ≤ num_hrefs` chọn `q=0.3703`, còn `n_non_stop_unique_tokens ≤ n_non_stop_words` chọn `q=0.3487`; combined active fraction là `0.6765`. Đây là ví dụ định lượng đầu tiên rằng một global interval không mô tả hết graph dynamics, dù hai edges đều là numerical directional bounds.

Strength vẫn tương ứng guidance scale, giống cách NeurIPS paper tách scale khỏi interval. Bước tự động hóa tiếp theo phù hợp là backtracking/trust-region: chọn step size lớn nhất làm giảm typed residual nhưng không làm train-fitted row energy xấu đi. Không nên học interval trực tiếp trên test fidelity metric.

## 14. Kết quả mới: automatic routing, contribution của LLM và mở rộng dataset

### 14.1. Automatic route trên screening seed

Selector `auto_knee` không nhận route quantile hay metric cuối. Nó quan sát stability curve của graph trên chính reverse trajectory, lưu các middle-state graph snapshots, chọn điểm tăng nhanh nhất, sau đó chỉ thực thi projection trong late window.

Trên seed `20261607`, selector chọn `q=0.3703`, gần route thủ công `.35` nhưng không được cung cấp giá trị `.35`. Paired delta so với TabDiff cùng noise/sample là:

| Method | ΔShape | ΔTrend | ΔAlpha | ΔBeta | ΔR² | ΔNRMSE | ΔViolation điểm |
|---|---:|---:|---:|---:|---:|---:|---:|
| Auto route, strength `.75` | -0.0093 | +2.3211 | 0.0000 | -0.0342 | +0.0021 | -0.0011 | 0.0000 |
| Auto route, strength `1.0` | -0.0093 | **+2.3410** | 0.0000 | -0.0244 | +0.0013 | -0.0007 | -2.2461 |
| Per-edge auto route, strength `.75` | -0.0093 | +2.3229 | 0.0000 | -0.0244 | +0.0016 | -0.0008 | 0.0000 |

Với baseline Trend score `90.5248`, strength `1.0` đạt `92.8658`; Trend error giảm từ `9.4752` xuống `7.1342`, tương đương **24.7% relative error reduction**. Shape chỉ giảm `0.0093` điểm và cả R²/NRMSE đều cải thiện nhẹ. Đây là điểm operating point tốt nhất hiện tại trên screening seed, nhưng chỉ được nâng thành main result sau xác nhận nhiều seed.

Per-edge selector chọn `q=0.3703` cho `num_self_hrefs ≤ num_hrefs` và `q=0.3487` cho `n_non_stop_unique_tokens ≤ n_non_stop_words`. Quality gần global selector trên graph chỉ có hai edge; do đó edge-specific timing hiện là mechanism evidence, chưa phải quality claim.

### 14.2. Matched comparison với final repair

Trên cùng 4,096 rows và seed, không thay generator sample:

| Method | Shape ↑ | Trend ↑ | Alpha ↑ | Beta ↑ | R² ↑ | NRMSE ↓ | Violation ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| TabDiff | 96.8020 | 90.5248 | 96.9408 | 44.2171 | 0.0089 | 0.9956 | 25.6104% |
| Final repair | 96.7751 | 92.7968 | 96.9392 | 44.2236 | 0.0121 | 0.9939 | 19.9951% |
| Soft middle-route | 96.7771 | **92.8477** | 96.9392 | 44.2074 | 0.0114 | 0.9943 | 25.6104% |
| Soft route + certifier | 96.7751 | **92.8477** | 96.9392 | **44.2318** | **0.0149** | **0.9925** | **19.9951%** |

So với final repair, soft route + certifier giữ cùng Shape/Alpha/violation, tăng Trend `0.0509`, Beta `0.0081`, R² `0.0027` và giảm NRMSE `0.0014`; detection giảm nhẹ khoảng `0.0037`. Kết quả này chưa phải chênh lệch lớn, nhưng nó chứng minh trajectory intervention không chỉ tái hiện post-processing: middle information cải thiện dependence trước khi cùng certifier được áp.

### 14.3. Phản ví dụ của raw peak và selector robust

Cross-seed trace phát hiện raw maximum-slope selector chưa robust:

- seed `20261605`: chọn `q=0.3918`, active row-edge fraction `0.5497`;
- seed `20261606`: một high-noise spike ở `q=0.9830` làm selector chọn quá sớm, active fraction chỉ `0.2109`;
- seed `20261607`: chọn `q=0.3703`, active fraction `0.6230`.

Do đó raw `auto_knee` không được dùng làm method cuối. Bản mới `auto_isotonic` fit một monotone readability curve bằng isotonic regression trước khi lấy maximum slope. Nó gộp các transient reversals ở high noise thay vì giữ spike bằng cumulative maximum. Offline replay trên đúng ba trace chọn lần lượt `q=0.3918`, `0.3487`, `0.3703`, đều nằm trong later-middle region và không đọc quality/utility metric.

Job `1093899[0-5]` đang xác nhận `auto_isotonic` trên ba paired seeds, hai strengths `.75/1.0`, đồng thời đánh giá TabDiff, final repair, auto route và auto route + certifier trong mỗi run. Đây là phép thử quyết định cho main inference method.

### 14.4. Contribution riêng của LLM trong graph discovery

Không nên gán toàn bộ sample-quality gain cho LLM. Pipeline hiện tách hai bài toán:

1. **Graph discovery:** LLM đề xuất relation từ schema/measurement semantics; statistics cung cấp candidate khác.
2. **Authority:** chỉ independent documentation cùng deterministic split verification mới cho phép edge đi vào actuator.
3. **Trajectory control:** typed graph đã freeze được route/actuate mà không gọi LLM trong sampling.

Trên catalog News gồm 14 documented programs bị giấu khỏi proposal stage:

- statistics-only tìm `9/14`, tức `64.3%` exact recall;
- Luna LLM đề xuất thêm ba quan hệ; hai quan hệ khớp catalog (`num_self_hrefs ≤ num_hrefs` và `n_non_stop_unique_tokens ≤ n_non_stop_words`), quan hệ thứ ba bị loại vì split violation trên 10%;
- union đạt `11/14`, tức `78.6%` recall: tăng **14.3 percentage points**, hay **22.2% relative recall** so với statistics-only;
- hai LLM edges được chấp nhận đều có independent documented match và zero violations trên discovery/counterexample/calibration/heldout; tổng chi phí run Luna là 3 API calls, 24,603 tokens, không gửi raw rows.

Cross-model run với Terra đã hoàn tất: 3 calls, 25,966 tokens, cũng đạt `11/14` recall và tìm lại đúng hai public edges trên. Terra đề xuất thêm hai sentiment-rate bounds có vẻ hợp lý về semantics, nhưng do không có independent catalog match nên chúng chỉ nhận role `verify_source`, không được actuation. Một subset hypothesis sai tiếp tục bị split audit loại vì violation trên 10%. Vì vậy phần LLM có cả evidence về reproducibility và evidence rằng authority gate thật sự ngăn semantic guess đi vào sampler.

Như vậy claim phù hợp là: **LLM mở rộng semantic coverage của typed graph, còn trajectory-aware controller tạo gain**. Không claim LLM là verifier cuối hoặc direct denoising policy.

### 14.5. Method hiện tại ở mức implementation

Với row `i`, typed edge `e` và reverse state `t`:

1. Denoiser dự đoán clean preview `x_hat_0(t)` rồi decode về schema gốc.
2. Mỗi edge tạo typed signal `s_{i,e}(t)`. Với inequality, signal là signed gap giữa hai cột; với functional/order programs, signal là target value mà edge đề xuất.
3. Rank stability giữa hai previews liên tiếp tạo row-edge mask `A_{i,e}(t)`. Graph readability là tỷ lệ row-edge pairs ổn định.
4. Isotonic change-point chọn middle snapshot `tau` toàn graph hoặc `tau_e` cho từng edge. Snapshot gồm signal và active mask, không chỉ một scalar timestep.
5. Ở late window, action layer chỉ xử lý row-edge pairs đã route và hiện vẫn vi phạm. Numerical inequalities dùng minimum standardized displacement; strength nhỏ hơn 1 là soft proximal step.
6. Corrected clean estimate thay denoised estimate trong update của sampler. LLM không nằm trong vòng lặp này và không cần gradient.
7. Optional certifier áp exact typed projection ở cuối để so công bằng với post-repair.

Các satisfied LLM inequality edges không bị đẩy sâu thêm vào interior trong main method. Positive-message branch đã được chạy nhưng làm utility giảm mạnh; vì vậy satisfied relations chỉ cung cấp stability/routing evidence. Train-fitted positive factor messages được giữ như ablation, không nằm trong main architecture.

### 14.6. Cross-dataset và cross-backbone đang chạy

- `1093898[0-1]`: official TabSyn trên Shoppers, automatic isotonic route từ persistent typed violation graph, strengths `.20/.40`.
- `1093900[0-1]`: TabDiff trên Shoppers với codebook relation `BounceRates ≤ ExitRates`, dùng cùng `auto_isotonic` router/action như News.
- Violation không phải primary metric trên Shoppers vì real heldout có khoảng `1.5%` exceptions cho Bounce–Exit. Quyết định dựa trên Shape/Trend, Alpha/Beta và XGBoost AUC; violation chỉ giúp giải thích action.

TabDiff–Shoppers đã hoàn tất trên screening seed:

| Strength | ΔShape error ↓ | ΔTrend error ↓ | ΔAlpha | ΔBeta | ΔDetection | ΔAUC | ΔActionable violation ↓ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `.75` | +0.0070 | -0.0317 | -0.0081 | +0.0553 | +0.0024 | -0.0024 | 0.0000 |
| `1.0` | +0.0070 | **-0.0337** | -0.0065 | +0.0553 | +0.0024 | **+0.0037** | -0.0001 |

Strength `1.0` là transfer tốt hơn: Trend, Beta, detection và AUC cùng tăng, trong khi Shape error xấu hơn `0.007` điểm. Relative Trend-error reduction là khoảng `1.53%`, nhỏ hơn nhiều so với News nhưng xác nhận mechanism không hoàn toàn dataset-specific. Selector chọn `q=0.2938` thay vì News `q≈0.37`, phù hợp với claim rằng route state phụ thuộc dataset/trajectory.

TabSyn persistent-violation router cho negative result rõ: nó chọn quá sớm tại `q≈0.806`, routed row-edge fraction chỉ `0.0159`, và không còn routed row nào vi phạm ở late window; action count bằng zero nên output giống hệt TabSyn baseline. Implementation đã được sửa để route bằng endpoint-rank stability giống News. Job `1093908[0-2]` đang quét stability threshold `.001/.005/.01` với strength cố định `.20`; đây là correction của state representation, không phải sweep tăng strength.

Nếu cả hai transfer giữ fidelity/utility gain, claim có thể mở rộng từ một News-specific observation sang backbone-agnostic state routing. Nếu TabSyn giữ gain nhưng TabDiff–Shoppers không, contribution phù hợp hơn là portability của interface `decode → typed graph → re-encode`, không phải universal improvement. Nếu cả hai không work, News vẫn là public-constraint main study và cross-dataset được report trung thực như limitation.

## 15. Đóng khung bài toán, method và giả thuyết sau vòng automatic routing

### 15.1. Bài toán hiện tại

Bài toán chính không còn là user-conditioned generation và cũng không phải gọi LLM để judge mọi diffusion step. Input của sampler là một **verified typed graph**. Mỗi edge chứa:

- endpoints là các cột hoặc một nhóm cột;
- một typed program có thể thực thi, chẳng hạn `less_equal`, `ordered_triplet`, `sum_equal` hoặc `abs_equal`;
- provenance: documentation, predefined specification, statistics hoặc LLM proposal;
- evidence trên các train-only splits và semantic authority;
- role `actuate`, `score` hoặc `retrieve`;
- typed signal và scale để đo stability trên clean preview;
- action operator được giới hạn theo kiểu dữ liệu.

Tại reverse state `t`, denoiser tạo clean preview `x_hat_0(t)`. Với từng row `i` và edge `e`, method tính signal `s_i,e(t)` cùng stability so với state liền trước. Vì vậy graph thực thi là `G_i(t)`, không phải một graph cố định dùng giống nhau cho mọi row và mọi timestep.

Question phù hợp cho paper là:

> Có thể dùng sự xuất hiện dần của relational structure trong clean-preview trajectory để tự động route các typed knowledge edges, rồi thực thi chúng với ít distortion hơn fixed per-step guidance và final repair hay không?

### 15.2. Pipeline ở mức implementation

**Phase A — knowledge acquisition và authority.** Candidate được tạo từ schema/statistics, external documentation và optional LLM. Candidate phải compile thành typed program, qua type check, counterexample split, calibration split và heldout audit. LLM không tự cấp quyền sửa dữ liệu; chỉ edge có independent semantic authority mới nhận role `actuate`.

**Phase B — state observation.** Ở mỗi state, decode `x_hat_0(t)` về schema gốc và tính typed signal. Với `a <= b`, signal là signed gap; với functional/order program, signal là target được relation đề xuất. Rank-based stability dùng thay raw residual vì scale noise khác nhau giữa feature và backbone.

**Phase C — automatic route.** Với mỗi edge hoặc toàn graph, fit monotone readability curve từ stable row-edge fraction và lấy endpoint của transition có slope lớn nhất. Với backbone tạo collapsed previews ở high noise, bỏ false-stability prefix bằng valley anchoring trước khi fit. Selector không đọc test data, Shape, Trend, utility hoặc final constraint score.

**Phase D — late action.** Snapshot ở middle state giữ row mask và typed target. Trong late window, action layer chỉ tác động các row-edge pairs đã route. Numerical inequality dùng minimum standardized displacement; functional programs dùng rank transport để giữ empirical marginal; categorical actions cần discrete projection riêng. Guidance strength nhỏ hơn 1 là soft proximal update.

**Phase E — optional certifier.** Exact typed projection ở final state tạo safety variant và matched comparison với post-repair. Main novelty nằm ở middle observation/route, không nằm ở certifier.

### 15.3. Contribution riêng của LLM

LLM hiện có một contribution đo được nhưng có giới hạn rõ:

- statistics-only tìm `9/14` documented News programs;
- Terra và Luna độc lập đều mở rộng lên `11/14`;
- gain là `+14.3` điểm recall tuyệt đối, tương đương `+22.2%` relative recall;
- cả hai model tìm lại đúng hai directional relations từ external specification; các semantic guesses không đủ authority bị giữ ở `verify_source` hoặc bị reject;
- cost khoảng 25k tokens, ba calls cho một graph; sampling không gọi API.

Do đó claim đúng là **LLM-assisted semantic graph completion under deterministic verification**. Không claim LLM feedback trực tiếp là nguyên nhân duy nhất của fidelity gain. Giá trị của LLM phải được tách bằng graph-source ablation, còn giá trị của dynamic graph được tách bằng fixed/final/full-step routing ablation.

### 15.4. Trạng thái các giả thuyết

| Giả thuyết | Trạng thái | Bằng chứng hiện tại |
|---|---|---|
| Clean preview có relational readability transition | Ủng hộ | Active row-edge fraction thay đổi mạnh theo trajectory; route `.35` tốt hơn các state gần-clean có coverage gần 100% |
| Middle state nên trực tiếp sửa mạnh | Bác bỏ dạng rộng | Early/middle hard projection thường phá utility; bản work tốt route ở middle nhưng actuation mềm ở late |
| Một fixed interval dùng cho mọi edge/backbone là đủ | Không ủng hộ | Hai News edges chọn `q=.3703/.3487`; Shoppers chọn `q=.2938`; TabSyn có false-stability prefix |
| LLM làm direct judge mỗi step là cần thiết | Bác bỏ trong setting này | Typed executor chính xác và rẻ hơn; LLM phù hợp với offline proposal/completion |
| LLM bổ sung tri thức ngoài statistics | Ủng hộ một phần | Withheld documented recall tăng `9/14 → 11/14`, reproducible qua hai model |
| Satisfied-edge positive messages cải thiện quality | Không ủng hộ | Chúng thực sự active nhưng broad/positive branches làm utility giảm; không nằm trong main method |
| Graph càng dày càng tốt | Bác bỏ | Broad statistical graph tăng Trend/Beta nhưng gây regression utility collapse |
| Dynamic route transfer qua dataset/backbone | Ủng hộ sơ bộ | TabDiff–Shoppers tăng Trend/Beta/AUC ở strength 1; TabSyn fixed-middle có một Pareto point nhưng automatic route còn đang sửa |

### 15.5. Các contribution có thể viết thành paper

1. **Verified multi-source typed graph.** LLM, statistics và documentation cùng tạo candidate, nhưng provenance/authority quyết định edge có được actuation hay không.
2. **Row-edge-time graph.** Edge activation được định nghĩa trên clean preview theo từng row và state, thay vì dùng một global static graph.
3. **Decoupled observation and action.** Middle state dùng để route semantic structure; late state dùng để thực thi correction. Đây là khác biệt chính với per-step projection và final repair.
4. **Data-free automatic state selection.** Isotonic change-point/per-edge routing chọn state từ trajectory readability, không tune bằng final quality metric.
5. **Typed, distribution-aware action.** Minimum standardized displacement, rank transport và optional certifier tách constraint satisfaction khỏi unnecessary marginal distortion.

Không nên claim generic per-step constraint enforcement, vì Harpoon/CTDF/Constrained Discrete Diffusion đã bao phủ phần đó. Không nên claim graph discovery bằng LLM tự thân là mới. Claim cần nằm ở verified knowledge được biến thành row-edge-time routing và ở matched fidelity/utility trade-off.

### 15.6. Batch thực nghiệm mới

- `1093899[0-5]`: ba seed, automatic isotonic route trên News, strengths `.75/1.0`, có matched final repair và certifier.
- `1093900[0-1]`: TabDiff–Shoppers transfer; đã hoàn tất và cho positive nhưng nhỏ hơn News.
- `1093908[0-2]`: TabSyn rank-stability thresholds. Trend/actionable violations cải thiện nhỏ; kết quả phụ thuộc threshold và AUC chưa ổn định.
- `1093921[0-2]`: corrected TabSyn valley-anchored selector; loại state đầu tiên có undefined stability trước khi fit.
- `1093915[0-5]`: graph-source ablation trên cùng automatic per-edge controller: documented/statistical graph so với union có verified LLM edges, ba paired seeds.

`1093911` là diagnostic run không được dùng làm result: selector cũ nhầm stability bằng zero ở state đầu tiên là valley thật. Lỗi đã được sửa trước `1093921`.

### 15.7. Hướng methodology tiếp theo, theo thứ tự ưu tiên

1. **Freeze automatic router** nếu `1093899` giữ khoảng gain của fixed `.35`; nếu không, fixed-route result chỉ được dùng như oracle interval upper bound.
2. **Chốt graph-source contribution** bằng `1093915`. Nếu union không hơn documented graph, giữ LLM ở discovery recall table chứ không đặt trong tên method.
3. **Hoàn tất cross-backbone selector** bằng `1093921`. Nếu TabSyn vẫn chỉ có gain rất nhỏ, report portability chứ không claim universal improvement.
4. **Thêm mixed-type action** sau khi numerical mechanism đã freeze: categorical logit masking/projection theo ý tưởng constrained discrete diffusion, dùng cùng routed graph snapshot.
5. **End-to-end distillation là extension**, không thay main experiment: distill corrected clean targets vào lightweight adapter, rồi so với training-free teacher trên cùng graph/seed.

Tiêu chí freeze nên dựa trên standard metrics: Shape/Trend, Alpha/Beta, TSTR utility, detection và DCR. Constraint satisfaction chỉ là primary metric khi relation đến từ public specification/documentation; các relation do statistics/LLM đề xuất chỉ được dùng để giải thích mechanism cho tới khi có independent authority.

## 16. Kết quả ba seed của automatic route

Sampler của `1093899` đã hoàn tất; common evaluator còn tính logistic detection. Các standard metrics không phụ thuộc detector đã được recompute trực tiếp từ output CSV bằng cùng evaluator.

Automatic isotonic selector chọn `q=0.3703` ở cả ba seed `20261605–20261607`. Đây là kết quả mạnh hơn raw knee: raw knee từng chọn sai `q=.9830` ở một seed, còn monotone pooling loại được spike đó mà không đọc final metric.

Paired delta so với TabDiff:

| Variant | ΔShape | ΔTrend | ΔAlpha | ΔBeta | ΔR² | ΔNRMSE |
|---|---:|---:|---:|---:|---:|---:|
| Auto isotonic, strength `.75` | -0.0243±0.0072 | +1.3935±0.8268 | -0.0054±0.0068 | +0.0027±0.0191 | +0.0081±0.0318 | -0.0040±0.0149 |
| Auto isotonic, strength `1.0` | **-0.0190±0.0111** | **+1.7560±0.9516** | -0.0049±0.0071 | -0.0217±0.0156 | +0.0034±0.0048 | -0.0016±0.0022 |
| Auto `1.0` + certifier | -0.0290±0.0027 | **+1.7575±0.9529** | -0.0060±0.0090 | -0.0157±0.0174 | **+0.0066±0.0035** | **-0.0031±0.0016** |

Strength `1.0` là automatic configuration tốt hơn: Trend tăng ở cả ba seed, Shape chỉ giảm `0.019` điểm và mean utility cùng cải thiện. Nó giảm any-public-constraint violation từ `25.72%` xuống `22.53%`, tức `-3.19` điểm, trước khi certifier chạy.

So với matched final repair, auto `1.0` + certifier:

- Trend `+0.0063±0.0667` điểm: thực chất ngang nhau;
- Shape `+0.0005±0.0005`: ngang nhau;
- R² `+0.0033±0.0022` và NRMSE `-0.0016±0.0010`: utility tốt hơn nhẹ;
- Beta `-0.0288±0.0155`: kém nhẹ;
- cùng final violation rate `20.58%` theo catalog năm public inequalities;
- số final repair events giảm từ trung bình `646.0` xuống `406.3`, tương đương **37.1%**.

Diễn giải đúng là: automatic trajectory routing tạo phần lớn relational correction trước final state, giữ standard fidelity gần final repair, cải thiện regression utility nhẹ và giảm khoảng 37% residual repair workload. Nó **không** Pareto-dominate final repair trên mọi fidelity/coverage metric. Headline `+1.76` Trend so với TabDiff là tốt, nhưng không được dùng để che việc final repair cũng đạt gần cùng Trend.

Kết quả này đủ để giữ automatic routing làm main method, với contribution là state selection và reduced residual correction, chưa đủ để claim SOTA generator fidelity. Detection từ Slurm cần được thêm sau khi hoàn tất; do logistic solver đang chạm 2,000 iterations, metric đó phải được gắn convergence note.

## 17. Vòng tối ưu fidelity sau khi freeze automatic route

### 17.1. Partial marginal restoration không giải quyết utility trade-off

Job `1093926[0-11]` áp một paired marginal restoration có strength `.10/.25/.50/.75`
cho cả automatic guidance và matched final repair trên ba News seeds. Phép biến đổi chỉ
dùng unconditional sample cùng noise làm empirical marginal reference; nó không đọc
heldout/test metric. So với final repair được xử lý bằng đúng cùng restoration, automatic
method giữ lợi thế Shape khoảng `+0.0104` và Trend khoảng `+0.0042–0.0047` điểm, nhưng
R² giảm khoảng `0.0028` và NRMSE tăng khoảng `0.00131` ở **cả ba seed**. Do đó nhánh
này không tạo Pareto improvement và bị loại khỏi main method.

Job `1093931[0-5]` kiểm tra cùng ý tưởng trên Shoppers. Ở weak guidance, restoration
`.10` tốt hơn restored final repair về Shape error `-0.00345`, Trend error `-0.00674`
và Alpha `+0.0138`, nhưng AUC giảm `0.00362`. Ở moderate guidance, các gain Shape
`-0.0147`, Trend `-0.0199`, Alpha `+0.0619` đi cùng Beta `-0.0789` và AUC
`-0.00669`. Kết luận: ép marginal sau sinh có thể làm bảng fidelity nhìn đẹp hơn ở
một vài cột, nhưng xóa signal phục vụ downstream utility. Không dùng kết quả này làm
headline.

### 17.2. Limited-interval actuation: tách observation, action và release

Job `1093945[0-5]` giữ automatic isotonic observation ở `q=0.3703`, nhưng chỉ
actuate trong một interval sáu steps rồi trả quyền điều khiển hoàn toàn cho score model:

| Interval theo log-sigma quantile | Steps thực sự actuation | Projection events |
|---|---:|---:|
| `[.05,.20]` | 6 | 1,078 |
| `[.10,.25]` | 6 | 1,076 |
| `[.15,.30]` | 6 | 1,071 |

Mỗi interval chạy strengths `.75/1.0`, cùng seed và cùng action budget gần như chính
xác. Đây là timing experiment chứ không phải so sánh có nhiều compute hơn. Giả thuyết
là guidance quá sát clean endpoint gây distribution drift; release sớm có thể cho denoiser
khôi phục learned distribution. Common evaluator đang chạy. Nếu không hơn continuous-late
guidance, limited interval chỉ được giữ làm timing ablation.

### 17.3. Manifold-safe rollback

Script `paired_manifold_rollback.py` hiện thực một controller độc lập với test metric.
Hai paired branches dùng cùng initial noise và đều đã certified: matched final repair và
middle-guided + certifier. Mixed-type train rows được quantile/one-hot embed, nén bằng PCA,
rồi fit mini-batch prototypes. Theo từng row, candidate trajectory chỉ được giữ khi
normalized train-support distance không xấu hơn final-repair branch; ngược lại rollback.

Smoke test trên interval `[.10,.25]`, strength `.75` cho thấy hai branches khác nhau ở
180/4,096 rows; strict controller chọn trajectory branch cho 102 rows (`56.7%` trong
tập changed, `2.49%` toàn bảng). Job `1093951[0-11]` đang đánh giá sáu trajectory
candidates với strict threshold và relative slack `.002`. Đây là PPAD-style rollback
được chuyển sang tabular, nhưng score là train-only manifold support thay vì LLM self-score.
Nó chỉ được giữ nếu thắng final repair trên standard fidelity/utility khi final validity
được giữ ngang nhau.

### 17.4. Kiểm tra under-guidance

Các graph-source và positive-message sweeps đã cho thấy thêm nhiều edge thường làm utility
xấu đi. Vì vậy job `1093958[0-3]` không thay graph hay route; nó chỉ quét projection
strength `1.25/1.5/2/3` trên automatic isotonic route đã chọn. Mục tiêu là falsify giả
thuyết rằng gain nhỏ chỉ do strength `1.0` quá yếu. Chỉ operating point hơn final repair
trên paired standard metrics mới được confirm thêm seed; strength lớn tạo Trend gain nhưng
làm Shape/utility xấu sẽ được report như high-guidance negative result.

### 17.5. Chuyển controller sang released TabDiff sampler

Một giới hạn quan trọng của toàn bộ mechanism screen tới đây là custom trajectory loop dùng
deterministic first-order Euler để paired comparison đơn giản, trong khi released TabDiff
sampler mặc định dùng EDM stochastic churn và Heun second-order correction. Training budget
8.000 epochs thực ra khớp config release; khác biệt sampler mới là confound lớn cần loại trước
khi nói tới SOTA fidelity.

`dense_numeric_factor_messages.py` đã được mở rộng với hai cờ
`--stochastic-sampler --second-order-correction`. Predictor sử dụng đúng churned state,
`sigma_hat` và categorical noise của released sampler; graph sửa predicted-clean estimate
ở predictor; unguided Heun corrector sau đó kéo update trở lại learned manifold. Thiết kế
này cũng tạo một interpretation gọn: **middle route → bounded graph predictor →
manifold-restoring corrector**. Job `1093959` là zero-action parity test 128 rows; chỉ khi
custom loop trùng released sampler trong tolerance thì mới chạy main quality sweep trên
official sampler. Đây là critical path mới cho mục tiêu fidelity, quan trọng hơn việc thêm
postprocessing hoặc tăng graph density.

Zero-action parity job `1093959` đã pass: trên 128 rows, chênh lệch numerical lớn nhất
bằng `0` và CSV trùng hoàn toàn. Job `1093963` vì vậy kiểm tra predictor guidance trong
stochastic-Heun sampler; `1093965` thêm graph action vào cả predictor và corrector để tránh
corrector vô tình triệt tiêu guidance. Tuy nhiên các job này vẫn dùng checkpoint pilot cũ,
nên chỉ là **solver ablation trên legacy representation**, chưa phải official-protocol result.

### 17.6. Audit representation của News và critical correction

Audit sâu hơn phát hiện một confound còn lớn hơn sampler. Script pilot cũ nối target hồi quy
`shares` vào categorical block, tạo cấu hình:

- `d_numerical = 45`;
- categorical cardinalities `[1422, 7, 8]`.

TabDiff release thực tế nối target hồi quy vào đầu numerical block:

- `d_numerical = 46`;
- categorical cardinalities `[7, 8]`.

Ngoài ra pilot cũ cố định `30` diffusion timesteps, trong khi release config dùng `50`.
Do mọi method/ablation cũ dùng cùng checkpoint, paired deltas của chúng vẫn trả lời câu hỏi
cơ chế. Nhưng absolute score không được dùng để claim vượt official TabDiff hoặc SOTA.

Code hiện giữ `categorical` làm default để không phá reproducibility của các job đang chạy,
đồng thời thêm `--regression-target-mode numerical --num-timesteps 50`. Ở chế độ mới,
preprocessing mirror đúng convention và tham số transform của implementation phát hành.
Hai lần khởi động `1093972/1093973` dừng trước training do lần lượt thiếu dependency và
binary C++ không tương thích; job `1093974` đã chạy ổn trong environment `grdm` với config
được xác nhận `46` numerical, categories `[7,8]`, `50` timesteps. Sau checkpoint này, bảng
main phải được chạy lại theo thứ tự: released TabDiff sampler → matched final repair →
predictor-only graph → Heun-consistent graph, cùng seed/noise/evaluator.

Action-layer mapping cũng đã được sửa cho representation mới: model coordinate `0` là
target `shares`, còn feature numerical đầu tiên bắt đầu ở coordinate `1`. Graph mask chỉ
sửa feature nodes nhưng re-encode đủ 46 numerical coordinates, giữ target slot bất biến.
Smoke test trả `encoded_shape=[256,46]`, target slot `shares`, first-feature slot `1`.

Kết luận tạm thời: chưa có cơ sở tuyên bố SOTA. Giá trị chắc nhất hiện nay là automatic
state routing và gain paired trên legacy backbone; bước có giá trị thông tin cao nhất là
re-establish toàn bộ effect trên representation và sampler đúng chuẩn phát hành.

### 17.7. Pareto gain đầu tiên khi guidance khớp stochastic-Heun solver

Quick evaluator bỏ riêng detection classifier nhưng giữ nguyên SDMetrics Shape/Trend,
AlphaPrecision/BetaRecall và XGBoost TSTR cho kết quả sau trên seed `20261609`. Đây vẫn là
legacy target representation; phần solver đã pass exact zero-action parity với release.

| Method | Shape ↑ | Trend ↑ | Alpha ↑ | Beta ↑ | R² ↑ | NRMSE ↓ |
|---|---:|---:|---:|---:|---:|---:|
| TabDiff sampler | 96.9259 | 88.5519 | 95.1533 | 40.8105 | -0.00782 | 1.00390 |
| Matched final repair | 96.8870 | 90.0062 | 95.1630 | 40.7943 | -0.00799 | 1.00399 |
| Predictor guidance `2.0` + certifier | **96.9801** | 90.3559 | 94.5999 | 42.4528 | **-0.00200** | **1.00100** |
| Heun two-stage `1.0` + certifier | 96.9786 | **90.4233** | 94.5983 | **42.4626** | -0.00364 | 1.00182 |

So với matched final repair, predictor `2.0` tăng Shape `+0.0931`, Trend `+0.3497`,
R² `+0.0060`, Beta `+1.6585` và giảm NRMSE `0.0030`; Heun two-stage tăng Shape
`+0.0916`, Trend `+0.4171`, R² `+0.0043`, Beta `+1.6683` và giảm NRMSE `0.0022`.
Cả hai giảm Alpha khoảng `0.56`, nên chưa Pareto-dominate trên toàn bộ sáu metric. Cách
diễn giải hiện tại là guidance cải thiện pairwise fidelity, coverage và downstream utility,
đổi lại precision-support giảm nhẹ.

Điểm quan trọng về mechanism: deterministic Euler strength sweep `1.25–3.0` không tạo
gain này; sau certifier nó gần như trùng final repair. Gain một seed chỉ xuất hiện khi graph
predictor được đặt trong stochastic-Heun trajectory. Điều này tạo hypothesis về
**solver-consistent dynamic actuation**, nhưng chưa phải evidence đủ mạnh.

Job `1093983[0-3]` đã xác nhận hai operating points trên hai seed mới và **không lặp lại
Pareto gain**:

| Seed | Variant so với final repair | ΔShape | ΔTrend | ΔR² | ΔNRMSE |
|---:|---|---:|---:|---:|---:|
| 20261609 | Predictor `2.0` | +0.0931 | +0.3497 | +0.0060 | -0.0030 |
| 20261610 | Predictor `2.0` | -0.0043 | -0.0214 | -0.0053 | +0.0022 |
| 20261611 | Predictor `2.0` | -0.0027 | +0.0014 | -0.0195 | +0.0097 |
| 20261609 | Heun `1.0` | +0.0916 | +0.4171 | +0.0043 | -0.0022 |
| 20261610 | Heun `1.0` | +0.0008 | -0.0165 | -0.0144 | +0.0059 |
| 20261611 | Heun `1.0` | -0.0012 | +0.0007 | +0.0050 | -0.0025 |

Mean ba seed của predictor là ΔShape `+0.0287`, ΔTrend `+0.1099`, ΔR² `-0.0063`,
ΔNRMSE `+0.0029`; Heun là `+0.0304`, `+0.1338`, `-0.0017`, `+0.0004`. Vì utility
mean không cải thiện, bảng một seed phía trên chỉ là mechanism-positive example, không
được dùng làm headline. Trace cho thấy route vẫn hoạt động ở mọi seed (`q≈.305–.349`,
routed rows `42–57%`, khoảng `1.35k` predictor events hoặc `3.0k` two-stage events), nên
negative confirmation không phải do graph bị inactive; effect thực sự phụ thuộc sample.

`1093982[0-3]` đang kiểm tra train-only manifold rollback để lấy lại Alpha mà không đọc
test metric. Full evaluator gồm detection được chuyển sang CPU `1093981[0-6]`. Kết luận:
solver-consistent implementation đáng giữ làm ablation/method component, nhưng chưa đủ để
claim SOTA. Critical decision vẫn là liệu effect có xuất hiện trên checkpoint đúng protocol
`1093974`; nếu không, không tiếp tục tune legacy representation.

## 18. Correct-protocol News: dynamic online mechanism không dùng LLM

Từ vòng này, vai trò của LLM được giới hạn dứt khoát ở graph discovery offline:
đề xuất quan hệ từ schema/metadata, chuyển thành typed program, rồi kiểm chứng bằng train
data. Reverse sampler không gọi API và không dùng LLM để đọc intermediate row. Active
edge được quyết định từ typed applicability, trạng thái predicted-clean hiện tại, độ ổn
định liên tiếp theo trajectory, feature-specific noise và thống kê chỉ fit trên train.

Checkpoint `1093974` đúng release protocol của News (`46` numerical dimensions, target
`shares` ở slot 0, hai categorical features, `50` diffusion steps). Job
`1093996[0-11]` là one-seed mechanism screen, 4.096 samples/configuration, released
stochastic churn + Heun, và common evaluator. Corrector-only bị lỗi local-variable ở lần
đầu; code đã sửa và rerun `1094354_2` hoàn tất. Lỗi này không ảnh hưởng 11 run còn lại.

### 18.1. Stage và schedule của directional edges

Trên cùng directional subgraph, TabDiff có Shape `97.3136`, Trend `92.0893`,
Alpha `95.7449`, Beta `44.4010`, R² `-11.4970`, NRMSE `3.5351`; matched final repair
có Trend `94.8373`, R² `-11.5322`. Kết quả dynamic trước certifier:

| Cơ chế | Shape | Trend | Alpha | Beta | R² | NRMSE | Violation công khai |
|---|---:|---:|---:|---:|---:|---:|---:|
| Predictor-only, late | 97.3141 | 94.8700 | 95.7465 | 44.4255 | -11.7743 | 3.5741 | 18.8477 |
| Predictor + corrector, late | 97.3141 | **94.8928** | 95.7465 | 44.3962 | -11.5326 | 3.5401 | 18.8477 |
| Corrector-only, late | 97.3141 | 92.1126 | 95.7449 | 44.3945 | -11.6489 | 3.5565 | 21.1426 |
| Predictor + corrector, middle | 97.3138 | 92.1303 | 95.7530 | 44.4076 | -11.4971 | 3.5351 | 21.0449 |

Kết luận có thể giữ: (i) corrector-only quá muộn để thay đổi final distribution đáng kể;
(ii) two-stage action giữ utility gần final repair hơn predictor-only; (iii) directional
gain phần lớn trùng với final repair, vì vậy chưa phải contribution chính. Edge-specific
noise coordinate và per-edge route cho cùng kết quả với global late route ở graph chỉ có
hai directional edges, nên không giữ như hai module độc lập.

### 18.2. Nhiều loại edge và action geometry

Các configuration full/algebraic chạy trên GPU khác directional runs; cùng seed nhưng
released stochastic sampler không bitwise-identical giữa hai GPU architecture. Mọi delta
dưới đây vì thế chỉ so trong từng run với baseline/final repair tạo trên chính hardware đó,
không so absolute row giữa hai nhóm.

| Cơ chế | ΔShape so final | ΔTrend so final | ΔAlpha so final | ΔBeta so final | ΔR² so final | ΔNRMSE so final |
|---|---:|---:|---:|---:|---:|---:|
| Algebraic per-edge | -0.5836 | +4.6617 | -2.0185 | -9.6696 | -0.2821 | +0.0538 |
| Full weighted per-edge | -0.6038 | +3.6911 | -2.6826 | -9.9479 | -0.3531 | +0.0667 |
| Full + density veto | -0.0026 | +3.5271 | +1.1537 | -4.1813 | -0.3065 | +0.0580 |
| Full + persistence + trust region | **+1.0146** | **+1.9001** | **+2.0242** | -0.6022 | **+0.2491** | **-0.0481** |
| Order edges + rank transport | -0.0269 | -0.1192 | -0.3539 | -1.1133 | **+5.0965** | **-1.1545** |

Hai tín hiệu đáng phát triển:

1. Persistence + sigma-normalized trust region loại được action không ổn định và thắng
   naive final repair ở năm trên sáu metric. Tuy nhiên so với untouched TabDiff, fidelity
   gần như ngang và utility chỉ cải thiện nhẹ; gain lớn so final một phần vì full-graph
   final repair làm hỏng distribution.
2. Rank transport trên order edges cho utility gain lớn nhất: R² từ `-5.8350` của TabDiff
   và `-6.7534` của final repair lên `-1.6569`; NRMSE từ `2.6144/2.7844` xuống `1.6300`.
   Đổi lại Shape/Trend giảm nhẹ. Absolute R² vẫn âm và sample size mới là 4.096, nên đây
   là promising mechanism signal, chưa phải SOTA result.

Algebraic direct projection và density veto bị loại khỏi main path: Trend tăng nhưng
coverage/utility giảm mạnh. Manifold rollback trước đó cũng đã bị loại vì không tạo Pareto
gain. Main online hypothesis hiện được thu hẹp thành: **trajectory-stable active edges +
noise-normalized bounded action + marginal-preserving transport**, sau đó Heun corrector
khôi phục learned score dynamics.

### 18.3. Positive-information screen

Job `1094355[0-11]` kiểm tra liệu graph có ích ngoài sửa violation hay không. Các edge
đang thỏa truyền conditional messages từ ExtraTrees potentials fit/cross-validate chỉ
trên train. Truyền dày không phù hợp: strength `.25` tạo khoảng `1.20M` cell-events và
làm Alpha giảm `6.95` điểm so với TabDiff. Tách satisfied-only cũng không cứu được effect:
Trend tăng `0.21`, nhưng Shape giảm `1.09`, Alpha giảm `2.06` và R² giảm `0.43`.

Selective configuration tốt nhất dùng normalized residual gate `1.0` và uncertainty gate
`1.0`, chỉ active `21.4%` candidate row-factors. So với TabDiff, nó tăng R² `0.685`, giảm
NRMSE `0.248` và tăng Beta `0.667`, nhưng giảm Shape `0.455`, Trend `0.028`, Alpha
`1.716`. So với matched final repair, nó tăng R² `1.801` và Beta `0.579`, nhưng giảm
Trend `1.656` và Alpha `0.571`. Kết luận: positive factor messages có thể là optional
utility-oriented component, nhưng không được giữ làm main method vì chưa tạo Pareto gain
và runtime của uncertainty gate cao hơn.

### 18.4. Valley routing và hybrid action: operating point đầu tiên thắng TabDiff toàn diện

Trace của full graph giải thích một phần các negative results: standard isotonic router
chọn một số sum/order edges tại noise quantile `q≈0.98`. Preview khi đó có thể ổn định vì
đang collapse, không phải vì semantic target đã readable. Job `1094367[0-11]` thêm
valley anchor trước khi tìm isotonic recovery, rồi so direct, rank và hybrid transport.

Kết quả nổi bật ở `hybrid_order_rank_valley_middle_trust`:

| Method | Shape ↑ | Trend ↑ | Alpha ↑ | Beta ↑ | R² ↑ | NRMSE ↓ | Public violation ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| TabDiff | 97.3440 | 92.7900 | 95.3289 | 42.8695 | -2.8755 | 1.9686 | 22.2168 |
| Matched final repair | 96.3616 | 92.5983 | 94.2196 | 43.3740 | -1.9019 | 1.7035 | 18.8965 |
| Dynamic hybrid action | **97.4007** | **92.8005** | **96.4683** | 43.0876 | **-1.0020** | **1.4149** | 22.0703 |

Dynamic method Pareto-dominate TabDiff trên cả sáu common metrics, đồng thời giảm nhẹ
public violation. Gain lần lượt là `+0.0566` Shape, `+0.0106` Trend, `+1.1393` Alpha,
`+0.2181` Beta, `+1.8735` R² và `-0.5537` NRMSE. So với matched final repair, method
tăng Shape `1.0391`, Trend `0.2022`, Alpha `2.2486`, R² `0.8999`, giảm NRMSE `0.2886`;
chỉ Beta thấp hơn `0.2865`.

Mechanism của operating point này gồm: edge-specific valley routing; interval
`q∈[.10,.40]`; middle-peak strength; persistence hai states; rank transport riêng cho
order edges để bảo toàn marginal; direct proximal action cho functional/sum/directional
edges; sigma-normalized trust radius `.5`; action ở cả predictor và corrector của Heun.
Không có LLM/API online.

Ablation timing ban đầu cũng mạnh: full-rank standard isotonic làm Beta giảm `11.24` điểm,
trong khi valley-anchored counterpart giữ Beta tăng `0.49`. LLM-directional-only tăng Trend
`0.65` và giảm public violation `2.49` điểm nhưng không tăng utility; statistical graph
không có LLM tăng R² `0.55` nhưng Shape giảm nhẹ. Điều này gợi ý hai nguồn graph mang
vai trò khác nhau, nhưng chưa đủ để claim synergy vì configuration chưa matched hoàn toàn.
Job `1094379[0-11]` vì vậy chạy exact component ablation ở 8.192 samples: full main,
bỏ LLM edges, LLM-only, bỏ order/algebraic edges, bỏ valley/trust/persistence, direct-order,
rank-all và late interval. Đây là phép kiểm tra quyết định trước khi đóng khung method.

### 18.5. Exact component ablation ở 8.192 samples

Job `1094379[0-11]` hoàn tất. Operating point chính lặp lại Pareto improvement so với
TabDiff trên seed mới và gấp đôi số mẫu:

| Variant so với paired TabDiff | ΔShape | ΔTrend | ΔAlpha | ΔBeta | ΔR² | ΔNRMSE |
|---|---:|---:|---:|---:|---:|---:|
| Hybrid main | +0.0481 | +0.0322 | +1.2622 | +0.0423 | +0.9712 | -0.2961 |
| Rank cho mọi program | +0.0025 | +0.0220 | +0.1904 | +0.0741 | **+2.1800** | **-0.4555** |
| Bỏ LLM edges | +0.0481 | +0.0335 | +1.2622 | +0.0391 | +0.9537 | -0.2902 |
| LLM edges only | +0.0007 | -0.0011 | -0.0008 | -0.0008 | +0.1344 | -0.0380 |
| Bỏ algebraic edges | +0.0008 | +0.0651 | +0.0155 | -0.0090 | -0.0467 | +0.0130 |
| Bỏ valley anchor | +0.0268 | +0.1141 | +1.0321 | +0.0570 | -0.1227 | +0.0234 |
| Bỏ trust region | -0.5836 | -0.0399 | -1.5210 | -6.4583 | -1.2983 | +0.2384 |
| Bỏ persistence | -0.0123 | +0.1603 | -0.0879 | +0.0570 | +0.1633 | -0.0315 |
| Direct thay cho rank ở order edges | +0.0150 | +0.1065 | +0.7171 | -0.2165 | +0.0977 | -0.0188 |
| Late thay cho middle interval | +0.0016 | +0.0027 | +0.1074 | +0.0391 | +0.1152 | -0.0222 |

Kết luận component:

- algebraic/functional statistical edges tạo phần lớn utility signal;
- valley routing, persistence và trust region là thiết yếu để signal đó không phá support;
- rank transport là action geometry tốt hơn direct projection cho utility;
- middle interval tốt hơn late interval rõ rệt trên R²;
- LLM-derived directional edges không giải thích gain trên News: bỏ chúng gần như không
  đổi kết quả. Vì vậy paper không được claim LLM-driven online gain từ dataset này.

Hai candidate được giữ: `hybrid main` ưu tiên Shape/Alpha, và `rank-all` ưu tiên utility.
Job `1094391[0-3]` đã submit validation ở đúng full train size `35.679` rows, gồm hai
candidate, matched no-LLM và no-order ablation; evaluator lần này có cả detection metric.

### 18.6. Hiệu chỉnh kết luận sau full-scale validation

`1094391[0-3]` đã hoàn tất. Khi tăng từ 8.192 lên 35.679 synthetic rows, gain fidelity
và support của hybrid main vẫn lặp lại nhưng gain utility không còn. Đây là correction
quan trọng; bảng 8.192 rows không được dùng làm main claim về utility.

| Method | Shape ↑ | Trend ↑ | Alpha ↑ | Beta ↑ | Detection ↑ | R² ↑ | NRMSE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| TabDiff paired | 97.5581 | 91.2034 | 96.4823 | 40.8225 | 0.9408 | **-0.3184** | **1.1482** |
| Matched final repair | 96.6098 | 86.5601 | 95.0428 | **42.3319** | 0.9269 | -0.4181 | 1.1909 |
| Dynamic hybrid | **97.6080** | **91.2263** | **97.7327** | 40.9636 | **0.9424** | -0.4350 | 1.1979 |

Paired delta của dynamic hybrid so với TabDiff là `+0.0499` Shape, `+0.0229` Trend,
`+1.2504` Alpha, `+0.1411` Beta và `+0.0015` Detection; nhưng R² giảm `0.1167` và
NRMSE tăng `0.0497`. Rank-all cũng chỉ giữ gain rất nhỏ ở Shape/Trend/Alpha/Beta và
làm utility giảm. Vì vậy trạng thái bằng chứng hiện tại là:

- **được ủng hộ:** bounded middle-step action có thể tăng marginal/joint fidelity và
  support mà không cần LLM online;
- **chưa được ủng hộ:** một operating point duy nhất Pareto-dominate backbone trên cả
  fidelity và downstream utility ở full train size;
- **không được ủng hộ trên News:** LLM-derived directional edges là nguồn chính của
  gain. Bỏ LLM giữ gần nguyên fidelity gain; LLM-only chỉ tạo effect nhỏ;
- **không dùng làm claim:** gain R² lớn ở 4.096/8.192 rows, vì nó không bền theo sample
  size.

Kết quả này làm tăng độ cần thiết của cross-backbone validation: nếu cùng graph mechanism
chỉ tăng metric trên một checkpoint TabDiff còn yếu, claim sẽ không đủ mạnh. Bước kế tiếp
phải tách rõ baseline quality khỏi method quality.

## 19. TabSyn transfer và baseline reproduction

TabSyn gốc báo trung bình trên 20 synthetic tables. Các local TabSyn result trước đây mới
là một checkpoint/một sample nên không được coi là exact reproduction. Mốc paper gốc trên
News là Shape error `1.64`, Trend error `1.44`, regression RMSE `0.861`; trên Shoppers là
Shape error `1.43`, Trend error `2.07`, AUC `0.920`. TabDiff paper sau đó tự rerun các
baseline trong protocol của họ, nên số TabSyn trong TabDiff/SCTab-Diff table cũng không
được trộn trực tiếp với một local single run.

Batch mới:

- `1094466`: official TabSyn VAE + latent diffusion trên News, sample full train size và
  đánh giá bằng common local News evaluator trong cùng job;
- `1094467[0-3]`: tiếp tục train official TabSyn-Shoppers checkpoint với fixed budget
  `250/750` epochs và learning rate `1e-4/3e-4`; checkpoint được chọn bằng training loss,
  tuyệt đối không đọc held-out metric; mỗi checkpoint được so unmodified và dynamic graph;
- `1094468[0-3]`: port middle-interval solver sang untouched TabSyn checkpoint, so
  predictor-only và dual-stage predictor+corrector ở strength `.05/.10/.20`;
- `1094469`: chạy lại untouched official TabDiff entrypoint trên News và report 20 tables,
  dùng experiment directory riêng để không ghi đè checkpoint cũ.

TabSyn sampler đã được mở rộng để nhận checkpoint ngoài, action ở predictor/corrector/cả
hai, persistence và latent trust region. Đây là backbone-portability test; không đổi VAE,
không fine-tune decoder và không dùng API lúc sampling.

Điều kiện quyết định sau batch này:

1. Nếu official reproduction gần paper nhưng checkpoint `1093974` không gần, thay
   checkpoint chính trước khi tiếp tục methodology tuning.
2. Nếu continuation cải thiện cả baseline và dynamic result, report method trên checkpoint
   tốt nhất nhưng vẫn giữ untouched checkpoint làm control; không chọn checkpoint bằng test.
3. Nếu TabSyn baseline mạnh nhưng dynamic gain biến mất hoặc âm, claim phải thu hẹp thành
   TabDiff-specific mechanism, không claim backbone-agnostic.
4. Nếu dual-stage middle action lặp lại fidelity gain trên TabSyn, có thể giữ formulation
   add-on/training-free và dùng end-to-end training chỉ như extension, không phải điều kiện
   để method hoạt động.

### 19.1. TabSyn middle-solver screen trên Shoppers

`1094468[0-3]` đã hoàn tất trên untouched official TabSyn checkpoint, 9.864 samples và
cùng generation seed. Các số dưới đây là paired gain so với TabSyn; Shape/Trend được đổi
thành hướng higher-is-better bằng cách lấy mức giảm error.

| Solver | ΔShape | ΔTrend | ΔAlpha | ΔBeta | ΔDetection | ΔAUC |
|---|---:|---:|---:|---:|---:|---:|
| Predictor `.10` | +0.00056 | +0.00104 | +0.00203 | -0.00135 | -0.00004 | -0.00291 |
| Predictor+corrector `.05` | +0.00169 | +0.00149 | +0.00203 | +0.00068 | +0.00004 | +0.00032 |
| Predictor+corrector `.10` | **+0.00563** | **+0.00353** | -0.00203 | -0.00068 | -0.00048 | **+0.00066** |
| Predictor+corrector `.20` | +0.00789 | +0.00591 | -0.00473 | -0.00135 | -0.00099 | -0.00435 |

Predictor `.10` và dual-stage `.05` có gần cùng tổng latent movement (`33.09` so với
`33.64`). Dual-stage `.05` tốt hơn predictor-only trên toàn bộ common metrics, nên effect
không chỉ do action budget lớn hơn. Tuy nhiên absolute gain còn rất nhỏ. Strength `.20`
tăng low-order fidelity nhiều hơn nhưng làm utility/detection giảm, xác nhận over-guidance.
Operating point tạm thời của TabSyn là dual-stage `.05–.10`, không phải `.20`.

Code mới `tabsyn_typed_program_guidance.py` đã port đủ 11 typed News programs sang latent
TabSyn: row-edge state được cập nhật ở mỗi predictor preview; persistence quyết định active
set; order programs dùng rank transport; các program còn lại dùng typed projection; delta
được re-encode và giới hạn bằng latent trust radius trước khi đi vào predictor/corrector.
Job `1094475[0-3]` sẽ chạy frontier strength sau khi News checkpoint xuất hiện, không gọi
API online. `1094483[0-4]` là five-sample protocol sanity check trước khi chi phí cho đủ
20 tables như paper.

### 19.2. Fine-tuning TabSyn checkpoint không giải quyết khoảng cách với paper

`1094467[0-3]` và matched untouched control `1094479` đã hoàn tất trên cùng seed, 9.864
rows. Hai budget `250/750` ở learning rate `1e-4` hội tụ vào đúng cùng checkpoint vì early
stopping tại epoch 161; đây không phải hai kết quả độc lập.

| Checkpoint | Shape error ↓ | Trend error ↓ | Alpha ↑ | Beta ↑ | Detection ↑ | AUC ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Untouched official | 1.8438 | **1.9797** | 99.3500 | 48.5347 | **0.9776** | 0.9020 |
| Continue, LR `1e-4` | **1.8152** | 2.1195 | 98.0360 | **48.5354** | 0.9681 | 0.8905 |
| Continue 250, LR `3e-4` | 2.0552 | 2.2438 | **99.3761** | 48.2191 | 0.9559 | 0.8921 |
| Continue 750, LR `3e-4` | 2.7570 | 2.7117 | 99.3152 | 48.0583 | 0.9271 | **0.9088** |

Continuation tạo trade-off chứ không có baseline tốt hơn toàn diện. LR `1e-4` cải thiện
Shape `0.0286` điểm nhưng làm Alpha giảm `1.314`, AUC giảm `0.0114`; LR `3e-4` dài hơn
tăng AUC `0.0068` nhưng phá Shape/Trend/Detection. Vì vậy không dùng fine-tuned checkpoint
làm main baseline và không gọi đây là reproduction improvement. Khoảng cách với paper có
thể đến từ averaging 20 tables, dependency version, split/evaluator hoặc training-seed;
`1094466/1094483` đang kiểm tra các yếu tố đó đúng protocol hơn.

Dynamic graph trên checkpoint fine-tuned thường giảm Shape/Trend error khoảng `.009–.015`
và có thể tăng AUC `.0025–.0027`, nhưng không cứu được degradation do checkpoint. Điều này
củng cố nguyên tắc: mechanism phải được đánh giá trên untouched/officially selected model,
không chọn backbone checkpoint bằng final test metrics.

### 19.3. TabDiff checkpoint frontier

`1094474[0-2]` dùng cùng seed, 8.192 samples và common evaluator:

| EMA epoch | Method | Shape | Trend | Alpha | Beta | R² | NRMSE |
|---:|---|---:|---:|---:|---:|---:|---:|
| 4000 | TabDiff | 97.5441 | 91.8258 | 94.9512 | 42.6286 | -0.6027 | 1.2660 |
| 4000 | Dynamic | **97.5690** | **91.8379** | **96.2077** | **42.7295** | **-0.5466** | **1.2436** |
| 6000 | TabDiff | 97.5187 | 91.7840 | 95.1262 | 42.9769 | -0.9187 | 1.3852 |
| 6000 | Dynamic | 97.5112 | 91.7824 | 95.4957 | 43.1689 | -0.8853 | 1.3731 |
| 8000 | TabDiff | 97.5129 | 91.7564 | 95.0775 | 43.1413 | -0.5623 | 1.2499 |
| 8000 | Dynamic | 97.5500 | 91.7862 | 96.0052 | 42.9989 | -0.5188 | 1.2324 |

Epoch 4000 là checkpoint duy nhất mà dynamic action cải thiện cả sáu standard metrics:
`+0.0249` Shape, `+0.0121` Trend, `+1.2565` Alpha, `+0.1009` Beta, `+0.0561` R² và
`-0.0224` NRMSE. Job `1094484` đang chạy đúng một full-scale confirmation. Tuy vậy cả ba
baseline vẫn có Trend `~91.8`, xa paper-reported TabDiff; do đó frontier không thay thế
official-entrypoint reproduction `1094469`.

## 20. Nhánh end-to-end mới: trajectory-teacher distillation

Các thử nghiệm end-to-end cũ không đủ mạnh để làm main method. Loss fuzzy chỉ ép
`applicability/satisfaction/rate` của graph, nhưng không nói rõ model phải thực hiện action
nào; graph-trained checkpoint thua pretrained dynamic ở held-out graph error trên trung
bình ba seed. Topology residual adapter còn làm Shape error từ `1.552` lên `5.142` và
Trend error từ `2.008` lên `9.966` trên Shoppers. Vì vậy không tiếp tục tăng weight của
hai loss cũ.

Implementation mới `finetune_trajectory_teacher_tabdiff.py` dùng một frozen TabDiff làm
teacher. Với mỗi real training row, script forward-noise row tới một timestep giữa, lấy
teacher `x-hat-0`, decode preview, cho verified typed graph tạo một corrected target, rồi
re-encode target về model space. Student vẫn học ordinary TabDiff mixed loss; graph target
chỉ là auxiliary loss. Các cột không bị graph action chạm tới có thể được neo về frozen
teacher để tránh distribution drift. Không có held-out row, test metric hoặc LLM call trong
training loop.

Frontier `1094488[0-9]` tách riêng các hypothesis sau, cùng checkpoint News epoch 4000,
cùng seed và 8.192-row evaluator:

| Task | Hypothesis |
|---:|---|
| 0 | Continued full-denoiser training, không graph; control cho training budget |
| 1 | Sparse action distillation, chỉ train output head và chỉ loss trên changed cells |
| 2 | Sparse distillation trên full denoiser với learning rate nhỏ hơn |
| 3 | Full-record target distillation; unchanged cells đóng vai trò implicit anchor |
| 4 | Delta-normalized loss để correction nhỏ không bị MSE nền lấn át |
| 5 | Hai noisy views, chỉ giữ action có hướng ổn định và thêm trajectory consistency |
| 6 | PCGrad: loại thành phần graph gradient xung đột với ordinary diffusion gradient |
| 7 | Rank-preserving teacher cho order programs thay vì direct projection |
| 8 | Strong sparse teacher để kiểm tra under-guidance ở target strength `.5` |
| 9 | Two-view stable loss trên cửa sổ rộng `.05–.65`, kiểm tra timing generalization |

Mỗi task đánh giá ba output: learned generator không guidance, learned generator + final
repair, và learned generator + dynamic middle-step sampler. Nhờ vậy có thể phân biệt gain
đã được hấp thụ vào model với gain do inference-time add-on. Các metric quyết định vẫn là
Shape, Trend, Alpha, Beta, R² và NRMSE; graph satisfaction chỉ dùng để giải thích cơ chế.
Smoke job độc lập `1094487` kiểm tra toàn bộ target/decode/backprop path; không dùng Slurm
dependency để tránh trạng thái `DependencyNeverSatisfied`.

### 20.1. Tín hiệu đầu tiên của frontier end-to-end

Smoke job `1094487` hoàn tất sạch trong 53 giây. Một epoch tạo 74.436 graph projection
events và 238.433 changed cells, nên action-target supervision không rỗng. Ba task đầu của
`1094488` đã hoàn tất:

| Generator, chưa thêm inference guidance | Shape ↑ | Trend ↑ | Alpha ↑ | Beta ↑ | R² ↑ | NRMSE ↓ |
|---|---:|---:|---:|---:|---:|---:|
| TabDiff epoch 4000 | **97.5441** | 91.8258 | 94.9512 | **42.6286** | -0.6027 | 1.2660 |
| Continued full denoiser, không graph | 97.5385 | 91.8263 | **95.0025** | 42.5578 | -0.5195 | 1.2327 |
| Sparse graph target, output head | 97.5468 | **91.8558** | 94.7877 | 42.5920 | -0.6185 | 1.2722 |
| Sparse graph target, full denoiser | 97.5385 | 91.8273 | 95.0009 | 42.5659 | **-0.4793** | **1.2163** |

Output-head-only distillation chưa tốt: Trend tăng nhẹ nhưng Alpha và utility giảm. Full
denoiser sparse target đáng theo dõi hơn: so với paired continued-training control, Shape,
Trend và Beta nhích lên, Alpha chỉ khác `-0.0016`, trong khi R² tăng thêm `0.0402` và NRMSE
giảm thêm `0.0164`. So với pretrained TabDiff, mức thay đổi là `+0.1234` R² và `-0.0497`
NRMSE với fidelity chỉ dao động khoảng `±0.06` điểm. Đây mới là 8.192 samples/một seed;
chưa dùng làm main claim cho tới khi strength frontier và full-size confirmation lặp lại.

Hai follow-up đã được submit không dependency: `1094506[0-3]` kiểm tra head-only ở
auxiliary weight `.5/1.0`, PCGrad và delta-normalized target; `1094510[0-3]` tập trung vào
candidate tốt hơn là full denoiser với weight `.10/.30`, PCGrad và two-view stability.
Frontier sau có ưu tiên khoa học cao hơn vì paired result ban đầu cho thấy graph signal chỉ
có ích khi được truyền qua toàn bộ denoiser, không chỉ output head.

### 20.2. Kết quả đầy đủ của loss/mechanism frontier

`1094488[0-9]` và `1094506[0-3]` hoàn tất. Các kết quả quan trọng nhất ở 8.192 samples:

| Method | Shape ↑ | Trend ↑ | Alpha ↑ | Beta ↑ | R² ↑ | NRMSE ↓ |
|---|---:|---:|---:|---:|---:|---:|
| TabDiff epoch 4000 | 97.5441 | 91.8258 | 94.9512 | 42.6286 | -0.6027 | 1.2660 |
| Continued full denoiser | 97.5385 | 91.8263 | 95.0025 | 42.5578 | -0.5195 | 1.2327 |
| Sparse target, full denoiser, weight `.03` | 97.5385 | 91.8273 | 95.0009 | 42.5659 | **-0.4793** | **1.2163** |
| Sparse target, full denoiser, weight `.10` | 97.5377 | 91.8286 | 94.9895 | 42.5618 | -0.5216 | 1.2335 |
| Broad two-view head + dynamic sampler | 97.5310 | **91.8765** | **96.0230** | 42.5773 | **-0.4791** | **1.2162** |
| Rank-order head + dynamic sampler | **97.5719** | 91.8740 | 96.0246 | **42.6904** | -0.5973 | 1.2639 |

Kết luận hiện tại:

- `weight=.03` trên full denoiser là sweet spot; tăng lên `.10` làm utility quay gần về
  continued-training control. Do đó effect không tăng đơn điệu theo constraint pressure.
- Output-head-only không hấp thụ action tốt. Các biến thể sparse/all/delta, PCGrad và
  weight `.5/1.0` đều giảm Alpha hoặc utility; tăng weight làm kết quả xấu hơn.
- PCGrad quan sát xung đột gradient thật (ví dụ 4/18 updates ngay epoch đầu), nhưng việc
  bỏ thành phần xung đột không phục hồi common metrics. Gradient conflict có thể report
  như chẩn đoán, không phải main technique.
- Two-view trên cửa sổ rộng tạo synergy với dynamic sampler: so với TabDiff gốc, Trend
  `+0.0507`, Alpha `+1.0718`, R² `+0.1236`, NRMSE `-0.0498`, nhưng Shape `-0.0131` và
  Beta `-0.0513`. Đây là candidate thiên utility, chưa Pareto-dominate.
- Rank-order dynamic là cấu hình duy nhất tăng cả sáu metric so với unguided TabDiff,
  nhưng utility gain rất nhỏ và nó vẫn kém dynamic sampler gốc ở utility. Vì vậy chỉ giữ
  làm transport ablation.
- Final repair thường tăng Beta/R² nhưng giảm mạnh Shape, Trend và Alpha; nó không phải
  lời giải cho mục tiêu fidelity toàn diện.

`1094510` gặp cluster-wide interruption (exit 120) cùng thời điểm nhiều job dài khác bị
dừng. Task full-weight `.10` đã train/sample xong và được đánh giá lại thủ công; ba task
còn lại không được coi là negative result khoa học.

Vòng xác nhận mới không dùng dependency:

- `1097903[0-9]`: hai paired seed cho continuation/sparse-full; full two-view ở middle và
  broad window; matched row-roll/sign-flip target controls; no-LLM và LLM-only graph;
- `1097904[0-1]`: full train-size 35.679 rows cho continued control và sparse-full
  checkpoint đã train;
- `1097905`: chỉ chạy lại evaluator trên full-scale baseline samples đã sinh xong trước
  interruption, không lãng phí GPU để resample.

### 20.3. Paired-seed correction: sparse distillation chưa ổn định

Hai paired seed đầu của `1097903` đã hoàn tất; mỗi cặp dùng cùng train seed và generation
seed cho continuation control và graph-trained checkpoint. Paired effect của sparse graph
target trên R² generator-alone qua seed 25/26/27 lần lượt là `+0.0402`, `-0.0109`,
`-0.0541`; trung bình `-0.0083`. Khi ghép dynamic sampler, paired effect là `-0.0163`,
`+0.0173`, `-0.0688`; trung bình `-0.0226`. Các fidelity delta trong mỗi cặp phần lớn rất
nhỏ, nhưng utility effect đổi dấu.

Do đó kết luận `sparse-full cải thiện utility` ở 20.1 bị hạ xuống thành single-seed signal,
không còn là candidate đã xác nhận. Kết quả cũng cho thấy absolute utility variance theo
generation seed rất lớn: continued control R² thay đổi từ `-0.519` ở seed 24 xuống
`-1.369/-1.230` ở seed 26/27. Job `1097917[0-7]` vì vậy giữ nguyên checkpoint seed 25 và
chỉ thay generation seed, nhằm tách sampling variance khỏi training variance.

Full-denoiser two-view loss cũng chưa giải quyết vấn đề. Middle-window checkpoint tăng R²
`0.034` nhưng dynamic composition làm R² giảm `0.081` so với TabDiff; broad-window
checkpoint tăng R² `0.052`, còn dynamic composition tăng `0.040` nhưng đánh đổi Shape,
Beta và vẫn kém continued-dynamic control. Không dùng các số này làm main claim.

Nhánh tiếp theo `1097925[0-3]` dùng benefit-filtered distillation: một graph action chỉ trở
thành target nếu nó giảm reconstruction error tới clean training `x0` ở cell hoặc row đó;
action có hại được neo về frozen TabDiff. Filter dùng duy nhất train row đã forward-noise,
không đọc test metric. Đây là sửa đổi phương pháp có nguyên nhân trực tiếp từ negative
result, thay vì tiếp tục tăng constraint weight.

### 20.4. Causal controls và full-scale correction

Các matched-target controls cho thấy không thể gán phần lớn single-seed utility gain cho
ngữ nghĩa đúng của graph. Trên cùng train/generation seed 25, row-roll và sign-flip targets
vẫn cải thiện R² so với continued-training control. Với generator chưa thêm online
guidance, R² lần lượt là `-0.5044` và `-0.5000`, so với `-0.5195` của continuation và
`-0.4793` của đúng graph target. Như vậy graph đúng chỉ hơn sign-flip khoảng `0.0207` R²;
phần còn lại phù hợp hơn với tác dụng regularization của auxiliary target. Cả hai control
đều không tạo Pareto gain rõ trên sáu common metrics.

Ablation nguồn cạnh còn rõ hơn:

| Train-time graph, seed 25 | Shape ↑ | Trend ↑ | Alpha ↑ | Beta ↑ | R² ↑ | NRMSE ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Full graph | 97.5385 | 91.8273 | 95.0009 | 42.5659 | **-0.4793** | **1.2163** |
| Không dùng cạnh LLM | 97.5380 | 91.8279 | 94.9992 | 42.5732 | -0.4929 | 1.2218 |
| Chỉ dùng cạnh LLM | 97.5272 | 91.8377 | 95.0001 | 42.6286 | -2.8384 | 1.9592 |

Việc bỏ cạnh LLM chỉ làm R² giảm `0.0136` so với full graph và gần như không đổi fidelity;
trái lại LLM-only graph làm utility sụp mạnh. Đây là bằng chứng hiện tại rằng LLM discovery
không nên là online/core mechanism. LLM có thể giữ vai trò đề xuất ứng viên offline, nhưng
cạnh phải qua data verification và một action-benefit gate; không claim rằng bản thân cạnh
LLM cải thiện fidelity/utility.

Đánh giá full train size 35.679 rows cũng sửa lại ấn tượng từ evaluator 8.192 rows:

| Checkpoint/method | Shape ↑ | Trend ↑ | Alpha ↑ | Beta ↑ | R² ↑ | NRMSE ↓ |
|---|---:|---:|---:|---:|---:|---:|
| TabDiff epoch 4000 | 97.6014 | 91.4573 | 95.7220 | **40.4783** | **-0.2151** | **1.1023** |
| TabDiff + dynamic sampler | 97.6365 | 91.4686 | 97.0155 | 40.5365 | -0.2216 | 1.1053 |
| Continued-training control | 97.6105 | 91.4823 | 95.7637 | 40.3619 | -0.3097 | 1.1444 |
| Continued + dynamic sampler | **97.6478** | 91.4967 | **97.0904** | 40.6159 | -0.1982 | 1.0946 |
| Sparse graph-distilled checkpoint | 97.6097 | 91.4834 | 95.7607 | 40.3634 | -0.3128 | 1.1458 |
| Sparse graph-distilled + dynamic | **97.6478** | **91.4978** | 97.0858 | 40.6020 | -0.2420 | 1.1144 |

Dynamic sampling trên continued checkpoint cải thiện Shape/Trend/Alpha và R² so với
TabDiff gốc, nhưng Beta giảm và mức utility gain chỉ `0.0169` R². Sparse graph
distillation không hơn continuation ở full scale; khi ghép sampler, nó còn kém continuation
dynamic `0.0438` R². Vì vậy end-to-end direct distillation hiện là negative ablation, chưa
phải main result. Main mechanism vẫn là training-free, row/state/timestep-dependent graph
activation trong middle reverse trajectory; end-to-end chỉ được giữ nếu benefit-filtered
target tạo paired gain mới.

Fixed-checkpoint resampling `1097917` tiếp tục đo treatment effect mà không trộn training
variance. Các seed 26 và 27 đã hoàn tất: sparse checkpoint kém continuation `-0.0497` R²
ở seed 26 nhưng hơn `+0.0400` ở seed 27; sau dynamic sampler, chênh lệch là `+0.0097` và
`+0.0530`. Effect còn nhạy với generation noise, nên mọi utility claim sau cùng phải dùng
paired seeds hoặc full-size sample, không chọn một sample thuận lợi.

### 20.5. Benefit filtering: chẩn đoán được lỗi nhưng chưa cải thiện generator

Fixed-checkpoint resampling đã hoàn tất bốn paired generation seeds 24/26/27/28. Sparse
direct distillation so với continuation có mean R² delta chỉ `+0.0029` ở generator-alone
và `-0.0036` khi ghép dynamic sampler. Đây là effect xấp xỉ bằng không; nhánh sparse
distillation được kết luận là negative result thay vì tiếp tục tune seed.

Benefit-filtered target audit cho thấy nguyên nhân trực tiếp: trong khoảng train-time
`t=.10–.40`, chỉ `40.65%` proposed cells của full graph làm frozen teacher `x-hat-0` gần
clean training row hơn; nếu giữ action chỉ khi tổng reconstruction benefit của cả row dương
thì còn `7.18%`. Broad two-view/stability filter chỉ giữ `12.12%`. Tuy nhiên bốn
benefit-filtered checkpoints vẫn không Pareto-dominate continuation:

| Generator-alone, seed 25 | Shape ↑ | Trend ↑ | Alpha ↑ | Beta ↑ | R² ↑ | NRMSE ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Continued-training control | 97.5385 | 91.8263 | 95.0025 | 42.5578 | -0.5195 | 1.2327 |
| Cell filter, weight `.03` | 97.5377 | 91.8271 | 95.0009 | 42.5488 | -0.5268 | 1.2356 |
| Cell filter, weight `.10` | 97.5364 | 91.8273 | 95.0025 | **42.5920** | **-0.5041** | **1.2264** |
| Row filter, weight `.03` | 97.5375 | **91.8277** | **95.0090** | 42.5749 | -0.5091 | 1.2284 |
| Broad two-view cell filter | 97.5031 | 91.9069 | 94.8829 | 42.6603 | -0.5655 | 1.2512 |

Benefit filtering loại đúng nhiều harmful targets nhưng chỉ tạo R² gain `0.010–0.015` ở
hai cấu hình và không vượt direct sparse result thuận lợi của một seed. Khi ghép sampler,
effect lại đổi dấu theo weight. Kết luận: biết action nào giúp tái tạo train `x0` không đồng
nghĩa checkpoint synthetic có distribution/utility tốt hơn; reconstruction benefit chỉ là
diagnostic/gating supervision, không phải surrogate cuối đáng tin cậy.

Full-size confirmation của broad two-view checkpoint có R² `-0.2010` và khi ghép dynamic
sampler là `-0.1710`, tốt hơn TabDiff full-size `-0.2151` và continued-dynamic `-0.1982`.
Tuy vậy Shape/Alpha/Beta đều kém continued-dynamic; đây là utility--fidelity trade-off,
không phải main Pareto result.

### 20.6. Edge-specific action benefit theo diffusion time

`audit_edge_action_benefit_by_time.py` thực hiện một phép đo mới trên 8.192 training rows,
9 timestep và toàn bộ 11 typed edges. Với mỗi forward-noised row, frozen TabDiff tạo
`x-hat-0`; từng edge được execute độc lập và so reconstruction error trước/sau với clean
train `x0`. Không dùng heldout/test row, final metric hoặc API.

Full graph có mean action benefit âm ở cả chín timestep; positive-cell rate chỉ khoảng
`39.5–45.6%`. Nguyên nhân chính là LDA simplex edge `typed_011`: nó tạo 38.220 changed
cells ngay tại `t=.03`, positive-cell rate `35.0%`, positive-row rate `1.5%` và mean
benefit `-0.0650` mỗi cell. Ngược lại, một số edge có vùng hữu ích riêng:

| Edge/family | Best train timestep | Changed cells | Positive-cell rate | Mean benefit/cell |
|---|---:|---:|---:|---:|
| `typed_004`, order | `.55` | 497 | 0.600 | +0.00722 |
| `typed_007`, order | `.55` | 498 | 0.614 | +0.01678 |
| `typed_008`, order | `.55` | 62 | 0.661 | +0.04295 |
| `typed_012`, conditional sum | `.55` | 14.198 | 0.536 | +0.00137 |
| `typed_020`, LLM/public bound | `.55` | 72 | 0.597 | +0.05164 |
| `typed_011`, LDA simplex | `.03` (least harmful) | 38.220 | 0.350 | -0.06499 |

Điểm này hỗ trợ đúng formulation dynamic graph: active set không nên chỉ là “edge đang bị
violate”, vì một invariant đúng vẫn có thể là action sai cho một noisy state. Edge identity,
row state và diffusion time đều cần đi vào action selection. Một train-only classifier dùng
`t`, action magnitude, current preview và feature noise đạt validation ROC-AUC `0.84–0.92`
cho các order edges `typed_005/007/008`; nó biến total action benefit âm thành dương trên
calibration split. Job `1097964[0-5]` đang kiểm tra classifier này như learned row-edge-time
gate trong sampler. Đây là experiment quyết định xem insight có chuyển thành sample-quality
gain hay chỉ có giá trị chẩn đoán.

Pure learned gate đã hoàn tất. Nó giảm 905.542 proposed row-edge actions xuống 125.809
actions (`13.9%`). Cấu hình predictor+corrector, strength `.5` tăng Alpha `+0.354`, Beta
`+0.041`, R² `+0.074` và giảm NRMSE `0.029` so với TabDiff, nhưng Shape `-0.0025` và
Trend `-0.0485`. Strength `.25` cải thiện năm trên sáu metric nhưng Beta giảm `0.022`;
nó vẫn kém main dynamic sampler ở phần lớn metric. Do đó classifier có khả năng chọn action
và cải thiện utility, nhưng gate toàn bộ graph quá bảo thủ đối với các edge vốn đã hoạt động
tốt dưới stability routing.

Hybrid experiment chỉ gate những edge mà train audit xác định có expected harm, còn các
edge khác giữ nguyên main dynamic mechanism. Kết quả 8.192 rows:

| Method | Shape ↑ | Trend ↑ | Alpha ↑ | Beta ↑ | R² ↑ | NRMSE ↓ |
|---|---:|---:|---:|---:|---:|---:|
| TabDiff epoch 4000 | 97.5441 | 91.8258 | 94.9512 | 42.6286 | -0.6027 | 1.2660 |
| Main dynamic graph | 97.5690 | 91.8379 | **96.2077** | 42.7295 | **-0.5466** | **1.2436** |
| Gate LDA simplex only | 97.5718 | 91.8377 | 96.1898 | 42.7498 | -0.5665 | 1.2516 |
| Gate LDA + conditional sum | **97.5713** | **91.8390** | 96.1760 | **42.7637** | -0.5568 | 1.2477 |

Hybrid algebraic gate cải thiện cả sáu metric so với TabDiff và đổi một phần utility/Alpha
để lấy Shape/Trend/Beta tốt hơn main dynamic. Đây là Pareto point mới có motivation rõ:
không thay action layer toàn cục, chỉ thêm learned abstention cho edge-family mà train-time
causal audit cho thấy projection thường làm xấu `x0`.

Full-size `1097979` xác nhận kết quả trên 35.679 samples:

| Full-size method | Shape ↑ | Trend ↑ | Alpha ↑ | Beta ↑ | R² ↑ | NRMSE ↓ |
|---|---:|---:|---:|---:|---:|---:|
| TabDiff epoch 4000 | 97.6014 | 91.4573 | 95.7220 | 40.4783 | -0.2151 | 1.1023 |
| Main dynamic graph | 97.6365 | 91.4686 | **97.0155** | 40.5365 | -0.2216 | 1.1053 |
| Hybrid algebraic benefit gate | **97.6387** | **91.4695** | 97.0013 | **40.5406** | **-0.2122** | **1.1010** |

Hybrid cải thiện cả sáu metric so với TabDiff. So với main dynamic, nó tăng Shape `0.0022`,
Trend `0.0008`, Beta `0.0041`, R² `0.0094`, giảm NRMSE `0.0043`, và chỉ giảm Alpha
`0.0142`. Absolute utility gain so với TabDiff vẫn nhỏ (`+0.0029` R²), nên không gọi là
SOTA hay breakthrough; nhưng gate đã loại regression utility của main dynamic full-size và
là operating point cân bằng nhất hiện tại. Component này đủ hợp lý để đưa vào method chính,
với pure gate/direct distillation giữ làm ablation âm.

Một independent train-noise calibration (`1097982`) xác nhận LDA simplex là harmful edge
ổn định: gate chọn zero hoặc gần-zero actions ở cả hai runs. Ngược lại, conditional-sum
`typed_012` không ổn định: một calibration thấy total action benefit âm và gate mạnh, run
còn lại thấy benefit dương và giữ `99.75%` proposals. Vì vậy bản robust được định nghĩa
trước kết quả test là chỉ gate edge có harm lặp lại qua calibration draws. `1097986` kiểm
tra lại algebraic gate với gate seed độc lập; `1097987` đánh giá robust LDA-only ở full
size. Nếu two-edge gain không lặp lại nhưng LDA-only lặp lại, paper phải dùng robust
cross-calibration criterion, không chọn edge từ một noise draw thuận lợi.

Hai follow-up đã hoàn tất và xác nhận concern này. Với gate seed độc lập, algebraic hybrid
ở 8.192 rows vẫn tăng Shape/Trend/Alpha/Beta nhưng R² `-0.60277` và NRMSE `1.26600`, gần
như bằng hoặc hơi kém TabDiff (`-0.60268`, `1.26597`). Robust LDA-only ở 35.679 rows đạt
Shape `97.6388`, Trend `91.4694`, Alpha `97.0052`, Beta `40.5437`, nhưng R² `-0.2171`
và NRMSE `1.1032`; nó giữ fidelity gain nhưng không phục hồi utility.

Vì vậy utility gain của two-edge gate trong một calibration (`-0.2122` R²) chưa lặp lại và
không được dùng làm main claim. Kết luận ổn định hơn là: cross-calibrated abstention loại
được harmful LDA actions và giữ low-order fidelity, còn utility recovery chưa đủ robust.
Main absolute full-size point hiện vẫn là continued TabDiff + original dynamic sampler
(Shape `97.6478`, Trend `91.4967`, Alpha `97.0904`, Beta `40.6159`, R² `-0.1982`, NRMSE
`1.0946`), cải thiện cả sáu metric so với cả paired continued control và original TabDiff.
Tuy nhiên paper phải report rõ checkpoint interaction và paired control; không được gán
gain do continued training cho graph.
