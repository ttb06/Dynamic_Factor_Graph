# Đánh giá lại graph guidance theo metric cuối

> Audit bổ sung ngày 2026-10-17 về contribution của LLM, mức độ edge thật sự active và state-selective gating: `LLM_GRAPH_CONTRIBUTION_AND_ACTIVATION_AUDIT_20261017_VI.md`.

Ngày cập nhật: 2026-10-17  
Backbone đang dùng cho nhánh chính: TabDiff checkpoint News epoch 8000

> **Đính chính metric detection (2026-10-17):** các giá trị detection khoảng 0.89--0.92 trong những bảng cũ được tạo bởi logistic detector chưa hội tụ, nên không được dùng làm bằng chứng. Đánh giá lại bằng `lbfgs`, `max_iter=2000` cho mean ba seed: TabDiff `0.81164`, late rank `0.81030`, middle route q=.35 `0.80938`, middle route q=.24 `0.80658`. Vì vậy hiện **không có detection gain ổn định**; claim chính chỉ dựa trên Shape/Trend/alpha/beta và utility. Các cột detection cũ phía dưới được giữ để truy vết lịch sử nhưng không được đưa vào paper.

## 1. Điều chỉnh quan trọng nhất

`Violation rate` chỉ có ý nghĩa đánh giá độc lập khi constraint đến từ một đặc tả bên ngoài, metadata chính thức, hoặc domain expert. Với cạnh do LLM hay statistical discovery đề xuất, violation không phải ground truth. Dữ liệu thật có thể vi phạm quan hệ đó vì relation chỉ là xác suất, vì thiếu biến, hoặc vì relation sai.

Do đó project được chia thành ba authority level:

| Nguồn graph | Vai trò trong generation | Có dùng violation làm metric chính không? |
|---|---|---|
| External/public/documented constraints | Có thể actuate sau typecheck và split audit | Có, nhưng vẫn là metric phụ bên cạnh fidelity/utility |
| Statistical relations ổn định trên train | Conditional factor, action scorer, context prior | Không |
| LLM external hypotheses | Proposal, retrieval, scorer hoặc diagnostic guardrail | Không, trừ khi được nguồn ngoài xác nhận |

Main table phải ưu tiên Shape, Trend, alpha-precision, beta-recall, detection, downstream utility và coverage. Constraint metric được tách sang bảng validity và ghi rõ nguồn authority.

## 2. Đánh giá năm concern

### 2.1 LLM sinh constraint còn mềm và đơn giản

Concern này đúng. Protocol Default gần nhất giữ 8/14 candidates, nhưng cả 8 đều là soft; audit xếp 5 cạnh vào `score_only`, 3 cạnh `reject_or_revise`, và không cạnh nào được auto-promote. Các mẫu lặp lại chủ yếu là zero-bill/payment hoặc persistence của `PAY=-2`. Chúng có vẻ hợp lý nhưng không đủ semantic necessity vì dataset thiếu refund, fee, write-off, posting time và mô tả đầy đủ các code âm.

Few-shot prompting hữu ích để cải thiện **định dạng và kỷ luật nhận thức**, không tự tạo ground truth. Profile mới đã tách:

1. schema/type facts do deterministic compiler xử lý;
2. documented functional relations;
3. state-conditioned process hypotheses;
4. causal/external hypotheses chỉ được làm soft score.

Không nên yêu cầu LLM suy ra causal constraint từ một observational table không có intervention/process metadata. Muốn causal relation thật, benchmark phải có temporal order, known mechanism hoặc structural equation. Nếu không, chỉ được gọi là directional process hypothesis.

Pipeline API trước đây còn có lỗi thực thi: response rỗng từ router với usage bằng 0 bị coi là schema failure vĩnh viễn. Code hiện retry loại lỗi transient này, nhưng vẫn fail-closed khi model đã dùng token và trả JSON sai.

Run sau khi sửa đã hoàn tất: 13 API calls, 141,590 tokens; multi-role protocol đề xuất 32 candidates nhưng chỉ giữ 5. Chỉ một cạnh được judge giữ hard là `num_self_hrefs <= num_hrefs`, tức relation đã có trong public graph. Bốn cạnh còn lại là soft: hai zero-positive/negative-rate implications, `n_non_stop_unique_tokens <= n_non_stop_words`, và zero self-reference summaries khi không có self link. LLM không tự tìm được các simplex, absolute transform hoặc ordered-summary hyperedge quan trọng hơn. Vì vậy tăng model/budget đơn thuần chưa giải quyết graph quality; typed DSL và metadata program compiler mới là phần thiếu chính.

### 2.2 Rule/statistical constraints quá đơn giản

Concern này cũng đúng nếu rule mining chỉ tìm pairwise implication theo threshold. Cách cải thiện hợp lý không phải chỉ tăng số cạnh, mà tăng **relation family**:

- typed domain: integer, nonnegative, category membership, range;
- directional inequality/subset;
- ordered summary hyperedge `min <= avg <= max`;
- affine/simplex relation `sum(x_j)=c`;
- deterministic transform `y=|x-c|`;
- conditional functional dependency;
- temporal/process relation khi metadata xác định thứ tự;
- statistical factor graph cho dependency mềm, có bootstrap/split stability.

Trên News, metadata UCI cho phép tạo chín functional hyperedges có cơ sở hơn LLM proposal:

- ba keyword-summary triplets;
- positive- và negative-polarity `min/avg/max`;
- simplex của năm LDA topics;
- positive/negative rate simplex khi có non-neutral token;
- absolute title polarity;
- absolute title subjectivity quanh mức trung tính 0.5.

Kiểm tra độc lập trên real splits cho thấy phần lớn là exact hoặc gần exact: ba keyword order có test violation từ 0 đến 0.025%; polarity order khoảng 0.10--0.23%; LDA sum và sentiment-rate sum gần machine precision; absolute polarity gần exact. Đây là graph phức tạp hơn và authority rõ hơn các statistical implication tự đề xuất. Nguồn metadata: [UCI Online News Popularity](https://archive.ics.uci.edu/dataset/332/online%2Bnews%2Bpopularity).

### 2.3 Có nên tăng computation và sửa nhiều sample hơn?

Có thể tăng computation, nhưng evidence hiện tại cho thấy **edit budget không phải bottleneck chính**.

- Các run mới đều dùng đủ 4,096 samples và guidance trên mọi row đủ điều kiện.
- Tăng strength của dense factor message làm Trend tăng nhưng utility/coverage giảm.
- Hard projection toàn trajectory giảm violation mạnh hơn clean-only, nhưng không cải thiện đồng đều metric cuối.
- Trust-region quá nhỏ gần như không tác động; tăng radius làm nhiều constraint được sửa hơn nhưng beta-recall/detection dao động xấu.

Vì vậy compute tăng nên dành cho branch/lookahead và distribution-aware action scoring, không phải sửa mọi detected violation. Một action chỉ được nhận nếu cải thiện independent train-fitted score và không làm hỏng các dependency khác. Best-of-N cũng phải có coverage correction; luôn chọn sample có density cao nhất sẽ mode-seek.

### 2.4 Mượn mechanism từ domain khác

Ba idea đã được port hoặc kiểm tra:

1. **Limited guidance interval** từ image diffusion: hữu ích về câu hỏi timing, nhưng kết quả News không tái hiện kết luận “middle interval tốt nhất” cho hard projection. Clean-end mới là vùng hiệu quả vì preview đủ ổn định và constraint không bị các step sau mở lại.
2. **Trust region/manifold-preserving guidance** từ DSG/Trust Sampling: ellipsoidal cap theo feature-specific TabDiff sigma đã implement. Kết quả đầu chưa Pareto-dominate projection thường.
3. **Feasibility/projection layer** từ constrained diffusion: phù hợp với external typed constraint, đặc biệt affine/simplex/order programs. Đây là nhánh mới đang chạy trên documented News hypergraph.

Bài học quan trọng: timing phụ thuộc operator. Semantic routing có thể cần middle state, nhưng exact projection thường cần late/clean state. Không nên ép mọi graph action phải xảy ra ở middle chỉ để giữ novelty.

### 2.5 Cạnh đã thỏa có truyền thông tin không?

Hiện đã có một implementation thật: mỗi node numerical nhận conditional message từ graph neighbors bằng ExtraTrees fit trên train; reliability lấy từ held-out nội bộ của train. Hai transport được thử:

- residual message;
- marginal-preserving copula reassignment.

Kết quả seed `20261601`:

| Method | Shape ↑ | Trend ↑ | Detection ↑ | R2 ↑ | Beta ↑ |
|---|---:|---:|---:|---:|---:|
| TabDiff | 96.657 | 91.156 | 0.8925 | -0.0055 | 45.125 |
| Copula message, 0.15 | 96.634 | **92.001** | 0.8853 | -0.0242 | 44.718 |
| Residual message, 0.15 | 96.336 | 91.883 | 0.8878 | -0.0029 | **45.156** |
| Projection-only | **96.638** | 91.845 | 0.8867 | **0.0156** | 45.042 |

Vậy satisfied-edge propagation hoạt động cơ học và tăng dependency score, nhưng direct message hiện chưa cải thiện toàn diện. Kết luận phù hợp hơn là dùng satisfied edges làm:

- guardrail/veto cho action gây collateral damage;
- conditional likelihood trong action ranking;
- context cho Graph Residual Adapter;
- weak energy guidance với trust/abstention.

Không nên để một satisfied soft edge tự overwrite row chỉ vì conditional predictor có giá trị khác.

## 3. Evidence về timing và trust

### 3.1 Projection window, seed 20261603

| Method | Shape ↑ | Trend ↑ | Detection ↑ | R2 ↑ | Beta ↑ | Public violation ↓ |
|---|---:|---:|---:|---:|---:|---:|
| TabDiff | 96.6034 | 90.5225 | 0.9213 | -0.00167 | 44.4499 | 25.586% |
| Full trajectory | 96.5533 | **93.3870** | 0.9176 | -0.00532 | 44.2627 | 0.049% |
| Middle only | 96.5750 | 90.6778 | 0.9238 | -0.00533 | 44.2448 | 22.217% |
| Clean end | **96.5655** | 93.3120 | **0.9233** | **-0.00182** | **44.3962** | 0.098% |
| Noisy end | 96.6019 | 90.5137 | 0.8983 | -0.00028 | 44.4840 | 25.415% |

Middle-only direct projection không được ủng hộ: sau khi guidance dừng, reverse process tái mở constraint. Clean-end đạt gần toàn bộ Trend gain của full trajectory và giữ common metrics tốt hơn. Main novelty vì vậy nên là **state-dependent graph routing/decision**, còn actuation có thể được trì hoãn tới vùng đáng tin cậy.

### 3.2 Trust-region projection

Clean-end radius 0.25 đến 2.0 gần như liên tục trade off số correction với common quality. Full radius 1 đạt Shape 96.5658, Trend 90.5781, R2 0.0042 nhưng beta-recall giảm 0.200 điểm và detection giảm 0.0195 so với TabDiff. Chưa có evidence để giữ trust region làm main component; nó là ablation an toàn, không phải breakthrough.

## 4. Method được đóng khung lại

### 4.1 Knowledge layer

1. `Schema compiler`: type/domain facts; phần lớn được preprocessing bảo đảm.
2. `Documented program compiler`: chuyển metadata/codebook thành typed functional hyperedges.
3. `Statistical factor discovery`: tạo soft dependency graph bằng value-aware MI, conditional prediction gain và split/bootstrap stability.
4. `LLM proposer`: bổ sung external/process hypothesis và compile về DSL; không cấp authority.
5. `Authority calibration`: external/documented, expert-verified, score-only, diagnostic, reject.

### 4.2 Dynamic state graph

Tại mỗi predicted-clean state, edge state không chỉ là violated/satisfied. Nó gồm:

- applicable/not applicable;
- readable/unreadable theo trajectory calibration;
- stable/transient;
- authority;
- residual hoặc factor score;
- candidate action set và uncertainty.

Graph có thể thay topology theo row và timestep. Middle state được dùng để nhận diện component, ambiguity và action route; late state được dùng cho exact execution khi cần.

### 4.3 Action layer

Action candidates gồm balanced projection, endpoint edit, affine projection, functional transform, abstain và branch/resample. Ranking ưu tiên common-quality proxy/conditional density và minimal displacement. Violation chỉ là hard objective cho external/documented constraint; LLM/statistical edge chỉ đóng góp score hoặc veto.

### 4.4 Family frontier mới

Một weight chung cho cả graph bị bác bỏ. Cùng seed và 4,096 samples, clean-end weak guidance cho thấy:

| Family | Strength | ΔShape | ΔTrend | ΔBeta | ΔDetection | ΔR² | ΔNRMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Tất cả | 0.10 | -0.723 | +0.096 | +1.079 | +0.0158 | -0.0114 | +0.0056 |
| Functional dependency | 0.25 | -0.067 | -0.019 | +0.295 | +0.0156 | +0.0005 | -0.0003 |
| Simplex | 0.25 | -0.795 | +0.015 | +0.894 | +0.0238 | +0.0018 | -0.0009 |
| Ordered summary | 0.25 | -0.060 | +0.166 | +0.207 | +0.0179 | -0.0005 | +0.0002 |

Điều này cho thấy graph signal có ích nhưng relation families có geometry rất khác. Ordered summary và deterministic transforms cho Pareto point tốt hơn; simplex cần weight thấp hoặc marginal-preserving transport.

Hai mechanism đang được thử trực tiếp để cải thiện bảng trên:

- `density-veto`: chỉ commit row correction nếu conditional energy fit trên train không xấu đi;
- `typed rank transport`: dùng functional program để reorder chính các giá trị hiện có, giữ empirical marginal chính xác thay vì ép equality.

### 4.5 Kết quả trực tiếp mới: rank transport trên ordered hyperedge

Các sweep mới xác nhận direct projection của nhiều family cùng lúc không phải hướng tốt nhất. Cấu hình `density-veto` không dùng simplex (`order=.25`, `functional=.25`) cải thiện Trend, beta-recall, detection, R² và NRMSE, nhưng còn giảm Shape 0.135 và alpha-precision 0.314 điểm. Giảm strength làm trade-off nhẹ hơn nhưng không vượt cấu hình rank transport.

Kết quả tốt nhất hiện tại dùng ba documented `min/avg/max` hyperedge của News, chỉ hoán vị các giá trị đang có của cột `avg` theo rank của midpoint `(min+max)/2` trong các late reverse states. Vì đây là phép hoán vị trong batch, marginal của cột được giữ đúng tại mỗi lần actuation; method chỉ thay đổi dependency/coupling.

| Method, seed 20261604 | Shape ↑ | Trend ↑ | Alpha ↑ | Beta ↑ | Detection ↑ | R² ↑ | NRMSE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| TabDiff | **96.5970** | 90.2665 | 96.3223 | 42.9134 | 0.8970 | -0.0328 | 1.0163 |
| Density-veto, no simplex | 96.4619 | 90.3880 | 96.0082 | 43.4212 | 0.9094 | -0.0282 | 1.0140 |
| Order rank, strength .10, late 25% | **96.6042** | **90.5429** | 96.3826 | 43.8151 | 0.9028 | -0.0335 | 1.0166 |
| Order rank, strength .25, late 15% | 96.5693 | 90.3991 | **96.3376** | **44.0951** | **0.9138** | **-0.0247** | **1.0123** |

Hai điểm mới là hai operating point không trội lẫn nhau. Strength .10 tăng cả Shape (`+0.0072`), Trend (`+0.2763`), alpha (`+0.0602`), beta (`+0.9017`) và detection (`+0.0058`), nhưng utility giảm rất nhẹ. Strength .25 ở late 15% tăng Trend (`+0.1326`), alpha (`+0.0152`), beta (`+1.1816`), detection (`+0.0168`), R² (`+0.00812`) và giảm NRMSE (`-0.00400`); chi phí là Shape `-0.0277`. Đây là các điểm gần Pareto tốt nhất hiện tại trên metric độc lập; phần dưới kiểm tra thêm hai seed và matched final-only baseline.

Hai seed xác nhận cho strength .25/late 15% đã hoàn tất. Paired delta so với TabDiff dùng đúng cùng latent seed:

| Seed | ΔShape | ΔTrend | ΔAlpha | ΔBeta | ΔDetection | ΔR² | ΔNRMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20261604 | -0.0277 | +0.1326 | +0.0152 | +1.1816 | +0.0168 | +0.0081 | -0.0040 |
| 20261605 | -0.0174 | +0.3567 | +0.0100 | +0.1790 | +0.0045 | +2.2786 | -0.7110 |
| 20261606 | -0.0346 | +0.8543 | +0.1827 | +1.2988 | +0.0157 | +0.1175 | -0.0422 |
| Mean | -0.0265 | **+0.4478** | **+0.0693** | **+0.8865** | **+0.0124** | +0.8014 | **-0.2524** |

Trend, alpha, beta, detection, R² và NRMSE đều cải thiện trên cả ba seed; Shape giảm nhỏ 0.017--0.035 điểm. Raw-share R² rất heavy-tail-sensitive: baseline seed 20261605 là -2.83, nên mean utility gain bị một seed chi phối. Vì vậy main table nên báo từng seed hoặc mean±std và bổ sung log-target/MAE; không chỉ báo mean R². Median paired delta là `+0.1175 R²` và `-0.0422 NRMSE`.

Matched final-only comparison cũng cho thấy lợi ích không chỉ đến từ operator. Với strength .10 và một final pass, trajectory tốt hơn final-only ở Shape, Trend (`+0.1442`), alpha, beta (`+0.5355`), detection, R² và NRMSE. Khi cho final-only 11 passes bằng số lần trajectory actuation, final-only đổi Trend gain lấy beta gain; nó không dominate trajectory. Với strength .15/một pass, trajectory tiếp tục hơn beta khoảng 0.547 điểm, detection khoảng 0.015 và R² khoảng 0.007, dù final-only hơn Trend 0.013 và giữ Shape chính xác. Do đó trajectory update tạo trade-off khác post-processing, nhưng chưa có quyền claim trajectory luôn trội trên mọi metric.

Trong lúc dựng matched final baseline, một dtype artifact đã được phát hiện: post-repair nâng mọi cột từ float32 lên float64, và sai số khoảng `1e-8` làm KS trên các cột nhiều ties giảm giả tạo. Code đã cast output về dtype gốc; bảng matched final cũ được đánh dấu không hợp lệ và không được report. Artifact này không ảnh hưởng trajectory outputs vì chúng được encode/decode lại về float32.

### 4.6 Middle routing, late execution

Scale-based stability ở late window bị bác bỏ vì quá lỏng: 99.8--100% row-edge active. Rank-based stability trong middle window tạo topology động thật, nhưng actuation ngay tại middle làm utility giảm. Ví dụ threshold .0005 cho active fraction tăng từ 16.3% ở state đầu tới gần 100% ở clean end; beta tăng 1.60 và detection tăng khoảng 0.020, nhưng R² giảm 0.012.

Biến thể tốt hơn tách hai pha:

1. tại một middle noise quantile, đóng băng row mask theo độ ổn định của target rank giữa hai state liên tiếp và lưu target của hyperedge;
2. không sửa state giữa; chỉ execute target đã route trong late 15% bằng marginal-preserving rank transport.

| Method, seed 20261604 | Routed row-edge | Shape ↑ | Trend ↑ | Alpha ↑ | Beta ↑ | Detection ↑ | R² ↑ | NRMSE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TabDiff | -- | **96.5970** | 90.2665 | 96.3223 | 42.9134 | 0.8970 | -0.0328 | 1.0163 |
| Late rank, no middle route | 100% | 96.5693 | 90.3991 | 96.3376 | 44.0951 | 0.9138 | -0.0247 | 1.0123 |
| Route q=.24, threshold .001 | 96.6% | 96.5729 | 90.4062 | 96.3343 | **44.1439** | 0.9140 | **-0.0237** | **1.0118** |
| Route q=.35, threshold .001 | 84.4% | 96.5718 | 90.4366 | **96.3538** | 43.9258 | **0.9191** | -0.0293 | 1.0146 |
| Route q=.43, threshold .001 | 40.6% | 96.5939 | **90.6716** | 96.3343 | 43.3219 | 0.9140 | -0.0370 | 1.0183 |

Route q=.24 cải thiện 6/7 metric so với late-only; alpha giảm 0.0033 điểm. Route q=.35 là operating point có dynamic sparsity rõ hơn: so với TabDiff, nó tăng Trend 0.1701, alpha 0.0315, beta 1.0124, detection 0.0221, R² 0.00346 và giảm NRMSE 0.00170, với Shape giảm 0.0252. Route q=.43 gần như giữ Shape (`-0.0031`) và tăng Trend 0.4051 nhưng làm utility giảm nhẹ. Kết quả này hỗ trợ cách định vị: middle state dùng để chọn topology/target; late state dùng để actuation.

Route q=.35 đã được xác nhận trên ba sampling seed. Routed row-edge fraction là 84.4%, 79.9% và 79.1%, tức topology thật sự phụ thuộc row/state thay vì mở toàn graph. Paired mean delta là `Shape -0.0225`, `Trend +0.5046`, `alpha +0.0481`, `beta +0.5729`, `detection +0.0175`, `R² +0.8031`, `NRMSE -0.2528`. Trend, beta, detection, R² và NRMSE cải thiện trên cả ba seed; alpha giảm nhẹ 0.0239 ở một seed và tăng ở hai seed. Do utility heavy tail, median paired delta (`R² +0.1273`, `NRMSE -0.0458`) đáng tin hơn mean. Đây hiện là candidate main configuration tốt nhất vì vừa có dynamic mechanism rõ, vừa có direct-metric gain lặp lại.

`Simplex rank transport` giữ Shape gần như nguyên vẹn (`ΔShape=-0.0029`) và tăng beta/detection, nhưng làm utility giảm. Hybrid direct-order/function + rank-simplex cũng chưa tốt bằng order-only. Kết quả này ủng hộ thiết kế operator riêng theo relation family thay vì một actuator chung cho toàn graph.

Nhánh tiếp theo đưa trajectory vào trực tiếp nhưng không tối ưu violation: một row-edge chỉ active khi semantic target của nó thay đổi ít hơn một ngưỡng train-standard-deviation giữa hai predicted-clean state liên tiếp. Sau khi active, action vẫn là marginal-preserving rank transport. Đây là state/row-dependent activation có thể kiểm tra bằng common metrics, thay vì gán novelty cho một middle window cố định.

## 5. Hypothesis mới có giá trị nhất

### H1 — Documented hypergraph cải thiện common fidelity

So sánh TabDiff, final projection, clean/full/middle trajectory trên chín documented functional hyperedges. Primary endpoint là Trend, Shape, alpha/beta, detection và downstream R2/NRMSE; program satisfaction là secondary sanity check.

### H2 — Middle decision, late execution

Freeze action/component được chọn từ middle preview, nhưng chỉ execute khi relevant endpoints đạt readability threshold. So sánh late recomputation, middle direct edit và static late projection. Đây là cách giữ contribution trajectory mà không mâu thuẫn với negative result của middle hard projection.

### H3 — Satisfied edge làm scorer tốt hơn actuator

Cùng candidate action set và compute, so sánh direct dense message với conditional-factor ranking/veto. Hypothesis chỉ được giữ nếu scorer cải thiện metric chung mà không giảm beta/coverage.

### H4 — LLM tăng semantic coverage, không trực tiếp tăng satisfaction

Đánh giá LLM bằng recovery của documented relation families, precision sau human/source verification, diversity, redundancy và compile success. Generation benefit chỉ được đo qua common metrics sau khi authority filtering; không báo violation của raw LLM graph như ground truth.

## 6. Jobs mới

- `1085722`: resume semantic-program LLM discovery sau khi sửa transient retry; giữ budget/cached calls, không dùng proposal làm hard constraint.
- `1085799[0-5]`: documented News hypergraph, 4,096 samples/configuration:
  - clean/full/middle exact projection;
  - clean half-strength;
  - middle residual hoặc copula satisfied-edge message kết hợp clean half projection.
- `1085850[0-5]`: family/strength frontier; đã hoàn tất và tạo bảng tại mục 4.4.
- `1091139[0-5]`: family-specific weighting kết hợp conditional-density veto.
- `1091168[0-5]`: typed marginal-preserving rank transport theo functional/order/simplex family.
- `1091189[0-5]`: refinement của family weights quanh density-veto; đã hoàn tất, không vượt order rank transport.
- `1091191[0-5]`: hybrid direct order/function với rank-simplex; đang hoàn tất, các task đầu chưa vượt order-only.
- `1091204[0-7]`: sweep strength, timing và edge-specific schedule cho order rank transport.
- `1091209[0-5]`: row/timestep stability-gated order rank transport; kiểm tra trực tiếp dynamic activation bằng metric chuẩn.
- `1091221[0-5]`: final-only order rank với cùng graph/operator, một pass và 11 passes để tách lợi ích trajectory khỏi post-processing.
- `1091228[0-3]`: hai sampling seed xác nhận cho hai operating point rank transport.
- `1091233[0-5]`: dynamic activation theo normalized rank stability; thay cho scale gate đã bị xác nhận là quá lỏng.
- `1091255[0-5]`: actuate trực tiếp bằng middle rank-stability gate; tăng beta/detection nhưng giảm utility, dùng làm negative timing ablation.
- `1091262[0-5]`: middle routing, late execution; hoàn tất và cho gain trực tiếp ở mục 4.6.
- `1091268[0-3]`: hai seed xác nhận cho hai route operating point q=.24 và q=.35.
- `1085625[0-4]`: trust-region projection đã hoàn tất; hiện là negative/ablation result.

## 7. Tiêu chí quyết định

Giữ một component trong main method chỉ khi:

1. cải thiện ít nhất một fidelity/utility metric chuẩn;
2. không gây giảm đáng kể coverage/beta hoặc detection;
3. tốt hơn matched final projection hay static factor tại cùng graph/checkpoint;
4. gain không phụ thuộc vào việc dùng chính proposed constraint làm evaluator;
5. effect còn tồn tại trên seed/dataset xác nhận.

Nếu documented hypergraph không cải thiện metric chung, kết luận đúng là constraint injection không tự động cải thiện tabular generation. Khi đó dynamic graph nên được định vị là controllability/validity method với quality-preserving guarantee, không phải SOTA unconditional synthesizer.

## 8. Nguồn phương pháp gần

- [SAGE, ACL 2026](https://aclanthology.org/2026.acl-long.174/): value-aware pseudo-features và sparse dependency context; gợi ý cho retrieval/statistical factor, không phải ground-truth constraints.
- [LCATS code](https://github.com/SJTU-DMTai/LCATS): LLM discovery, executable verification và weighted post-repair; typed syntax rộng nhưng official demo graph vẫn chủ yếu unary domain constraints.
- [Applying Guidance in a Limited Interval, NeurIPS 2024](https://arxiv.org/abs/2404.07724): nguồn của timing ablation; operator hiện tại không phải CFG.
- [DSG, ICML 2024](https://proceedings.mlr.press/v235/yang24h.html): nguồn cảm hứng cho trust-constrained guidance.
- [ConstrainedDGM, ICLR 2024](https://openreview.net/forum?id=G5g6tDg1ZE): nguồn năm public News inequalities dùng làm independent constraint benchmark.

## 9. Pipeline constraint có authority rõ ràng

Audit mới cho thấy lỗi của graph discovery trước không chỉ là “constraint đơn giản”, mà là trộn lẫn **empirical fit** và **semantic authority**. Trong graph empirical News cũ, 28/30 cạnh statistical được gán hard hoặc near-hard; 25/30 cạnh là các biến thể `zero -> zero`, thường mô tả cùng một latent zero-state. Chính sách mới bắt buộc:

1. statistical candidate chỉ được dùng để retrieve hoặc score, kể cả khi có zero observed violation;
2. LLM review phân loại ý nghĩa, direction và missing authority, nhưng không tự cấp quyền actuation;
3. chỉ program khớp independent documented catalog mới có `usage_role=actuate`;
4. held-out split chỉ được mở sau khi catalog đã freeze.

Typed statistical compiler mới tìm các family `ordered_triplet`, `sum_equal`, `abs_equal`, directional inequality và compressed zero-state component. Smoke test trên News đề xuất 20, giữ 17 sau discovery/counterexample/calibration, và tự tìm lại 9/9 program trong documented catalog. Chín program này được actuation; tám candidate còn lại ở hàng đợi retrieve/source verification. Zero implications được nén từ 25 cạnh pairwise xuống sáu component không trùng lặp.

Job `1091518[0-1]` đang kiểm tra semantic review bằng hai model. Mục tiêu không phải tối đa số rule LLM giữ, mà đo precision của ba quyết định: documented candidate cần tìm nguồn, soft semantic scorer, và statistical-only/reject. Nếu LLM không loại được relation thừa như `title_sentiment_polarity <= abs_title_sentiment_polarity`, vai trò LLM trong graph discovery cần thu hẹp thêm.

## 10. Kết quả semantic review và injection mới

Hai semantic-review run đã hoàn tất:

| Model | Candidates | Documented recall | Documented precision lower bound | Within-model reviewer agreement | Tokens |
|---|---:|---:|---:|---:|---:|
| GPT-5.6 Luna | 17 | 1.00 | 0.90 | 1.00 | 18,251 |
| GPT-5.6 Terra | 17 | 1.00 | 0.90 | 0.882 | 18,064 |

Cross-model decision agreement là `0.824`, nhưng hai model tạo **cùng actuation graph chín cạnh** và không có non-documented edge nào được actuate. Cả hai nhận ra self-reference `min <= avg <= max` là likely documented nhưng còn thiếu source match, đồng thời cùng reject quan hệ đúng nhưng thừa `title_sentiment_polarity <= abs_title_sentiment_polarity`. Khác biệt chủ yếu nằm ở zero-state components: Luna giữ năm soft scorer, Terra chỉ giữ hai. Vì vậy LLM hữu ích để semantic filtering, nhưng soft-decision variance còn cao và không nên trực tiếp điều khiển projection.

Full typed proposal loop dùng 25,418 tokens. Từ năm LLM proposals mới, hai cạnh sống qua pipeline cũ: `num_self_hrefs <= num_hrefs` vào source-verification queue và `global_rate_negative_words <= rate_negative_words` làm soft scorer. Audit sau đó phát hiện permutation-gap không phù hợp làm prefilter bắt buộc cho measurement-definition hypothesis: một subset bound vẫn đúng dù shuffle không phá được relation nếu marginal supports đã lồng nhau. Pipeline v2 giữ các hypothesis loại này để source review nhưng vẫn không cấp authority.

Independent catalog v2 hiện hợp nhất chín UCI functional programs và năm public News inequalities của ConstrainedDGM, tổng cộng 14 cạnh. Job `1091599` đang rerun typed proposal/review với catalog này để đo LLM recovery chính xác hơn.

Hai injection family mới đang chạy, luôn giữ documented/soft authority separation:

- `1091596[0-5]` — **soft prior injection**: soft edges tạo train-fitted conditional factor messages trong middle states; chỉ năm documented ordered programs được late rank actuation. Sweep consensus/Luna/unfiltered graph và strength `.02/.05/.10`.
- `1091598[0-5]` — **soft action scorer**: soft graph không sửa predicted-clean record; nó chỉ tính conditional energy để veto documented action làm joint distribution xấu đi. Đây là formulation sạch hơn nếu kết quả ngang hoặc tốt hơn soft prior.

Graph variants có lần lượt 5 documented ordered actuators cộng 0, 2, 5 hoặc 8 soft factors. LLM-filtering ablation vì thế dùng cùng backbone, noise, documented action và compute; chỉ tập soft relations dùng để inject/score thay đổi.

Không ưu tiên train graph-conditioned adapter mới trên News ở vòng này. Bằng chứng cũ trên Shoppers cho thấy residual adapter làm shape error tăng từ `1.612` lên `2.246` và trend error từ `1.984` lên `3.392`; bỏ topology vẫn là `2.073/2.740`, còn joint backbone training xấu hơn nhiều. Trước khi thiết kế lại objective/proximal guarantee, chạy lại kiến trúc đó trên News khó có information gain bằng hai injection family ở trên.
