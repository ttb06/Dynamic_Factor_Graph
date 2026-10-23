# Báo cáo tổng hợp: Dynamic graph guidance cho mixed-type tabular diffusion

**Cập nhật:** 20/10/2026  
**Mục tiêu:** đóng khung bài toán, phương pháp, bằng chứng thực nghiệm, baseline, giới hạn và kế hoạch hoàn thiện bài ICDE.  
**Quy ước:** mọi kết quả local đều ghi rõ paired run, số seed và evaluator. Số từ paper chỉ dùng làm bối cảnh, không trộn vào local ranking.

---

## 1. Kết luận điều hành

### Phương pháp hiện đã làm được gì?

Project đã chuyển từ một hệ thống **sửa constraint violation** sang một **dynamic distribution graph** dùng trong reverse diffusion:

- graph offline gồm typed semantic edges, train-fitted marginal/dependency factors và conditional factors;
- active graph được tạo riêng cho từng row và timestep từ state hiện tại, độ đọc được của feature, reliability và uncertainty;
- action layer có ba loại can thiệp: typed projection, numerical distribution transport và categorical logit message;
- correction chỉ được đưa vào một số middle steps, có trust region và pulse budget;
- LLM hiện chỉ tham gia graph discovery/semantic review offline, không được gọi ở từng diffusion step.

### Kết quả đáng tin nhất hiện tại

1. **News, ba seed:** target-quantile one-shot cải thiện Shape, Trend và Alpha ở `3/3` seed. Mean Shape score `+0.0426`, Trend `+0.0266`, Alpha `+1.9504` điểm. Utility thắng `2/3` seed nhưng có variance lớn; Beta chỉ thắng `1/3`.
2. **News, ablation cùng node/cùng seed:** conditional transport đơn lẻ thắng `6/6` common metrics, trong khi global transport đơn lẻ chỉ thắng `3/6`. Điều này xác nhận state/row conditioning có ích khi tách riêng operator.
3. **Adult, ba-seed confirmation:** screening seed từng thắng `6/6` nhưng không lặp lại. Mean Shape/Trend error xấu hơn `1.20%/2.51%`, Beta giảm `0.2382`; Alpha và AUC tăng nhỏ nhưng chỉ thắng `2/3` seed. Categorical hybrid vì vậy bị hạ xuống negative/diagnostic branch.
4. **Timing/geometry:** one-shot hoặc hai pulse yếu ở middle steps tốt hơn guidance liên tục/mạnh. Bỏ trust region hoặc tăng strength quá cao làm fidelity, coverage hoặc utility giảm rõ.
5. **LLM discovery:** trên News, statistical compiler tìm `9/14` relation; LLM bổ sung hai relation đúng, tăng recall lên `11/14 = 78.6%`. Tuy nhiên direct LLM soft messages chưa tạo gain ổn định. Trên Default, graph không dùng LLM trùng `7/8` executable edges với headline graph.

### Đã đạt SOTA chưa?

**Chưa.** Method đã có paired gains đáng báo cáo và một cơ chế rõ hơn, nhưng chưa đủ để claim SOTA vì:

- News utility chưa ổn định qua ba seed;
- full-global và full-conditional gần như ngang nhau ở same-node factorial;
- Adult categorical screening gain không tái lập qua ba seed;
- TabSyn local vẫn mạnh hơn rõ trên Shape/Trend của Default;
- evaluator/pipeline của SCTab-Diff khác local pipeline nên paper numbers không thể dùng như direct comparator;
- chưa có final validation-selected configuration chạy nhiều seed trên ít nhất ba dataset.

Định vị phù hợp nhất hiện tại là **same-backbone trajectory-control improvement**, chưa phải một generator mới vượt mọi published generator.

---

## 2. Bài toán nghiên cứu

### 2.1 Bài toán ban đầu

Ý tưởng đầu tiên là lấy clean estimate \(\hat{x}_0(x_t)\) ở một số reverse steps, đưa schema và row tạm cho LLM, nhận critique rồi biến critique thành guidance cho các bước còn lại.

Thực nghiệm và related work buộc bài toán phải rộng và chính xác hơn:

- PromptLoop đã dùng intermediate denoised estimate và model feedback trong image diffusion;
- Harpoon/CTDF/ConstrainedDGM/CDD đã bao phủ nhiều dạng constraint injection và feasibility operator;
- SCTab-Diff đã đưa static LLM prior vào reverse denoising;
- LCATS và Zhao et al. đã bao phủ discovery, verification và post-generation repair;
- gọi LLM ở từng step vừa đắt, vừa không ổn định khi preview còn nhiễu.

### 2.2 Bài toán hiện tại

> Làm thế nào biểu diễn tri thức semantic và statistical dưới dạng một factor graph, chọn factor thích hợp cho từng row/state/timestep, rồi đưa bounded distribution correction vào middle reverse steps để cải thiện fidelity và utility mà không làm mất coverage?

Đây không còn là “mọi constraint phải được thỏa mãn”. Constraint do project tự khám phá chỉ là nguồn thông tin; objective chính là metric cuối:

- marginal fidelity;
- pairwise/joint fidelity;
- precision và recall/coverage;
- downstream utility;
- privacy/diversity;
- public constraint validity chỉ khi constraint có independent authority.

### 2.3 Motivation cụ thể

Một graph cố định và một action cố định bỏ qua ba tính chất của diffusion:

1. Feature không trở nên readable cùng thời điểm.
2. Một relation có thể hữu ích cho row này nhưng không hữu ích cho row khác.
3. Cùng một relation có thể cần projection, ranking, density message hoặc không can thiệp tùy uncertainty hiện tại.

Do đó unit điều khiển hợp lý không phải “một global rule”, mà là một factor có state:

\[
e = (V_e, P_e, \phi_e, r_e, u_e, a_e),
\]

trong đó \(V_e\) là endpoints, \(P_e\) là applicability, \(\phi_e\) là potential/operator, \(r_e\) là reliability, \(u_e\) là uncertainty và \(a_e\) là action family.

---

## 3. Khoảng trống so với related work

| Work | Nguồn/venue | Tri thức vào generator ở đâu? | Phần đã được bao phủ | Khoảng trống còn lại cho project |
|---|---|---|---|---|
| TabSyn | ICLR 2024 | latent diffusion sau VAE | mixed-type latent generator mạnh | không có state-dependent knowledge graph |
| TabDiff | ICLR 2025 | feature-wise continuous/discrete diffusion | backbone mixed-type trực tiếp | không có semantic/distribution graph controller |
| ConstrainedDGM | ICLR 2024 | feasibility layer/training constraints | exact public constraints | không giải quyết discovery và row/time-dependent soft factors |
| Harpoon | ICLR 2026 | differentiable constraint loss trong sampling | request-conditioned hard guidance | constraint request phần lớn fixed; không phải discovery graph thay đổi theo trajectory |
| CDD | NeurIPS 2025 | constrained discrete denoising | constraint layer cho discrete states | không chạy trực tiếp mixed-type tabular; không có offline semantic discovery |
| CTDF | ICAIF 2025 | feasibility operator mỗi reverse step | mixed numerical/categorical hard feasibility | không còn khoảng trống cho claim “per-step hard constraint” chung chung |
| SCTab-Diff | preprint 2026 | static LLM prior condition reverse denoising | LLM prior đã đi vào denoising | prior không được tái cấu hình theo row/state uncertainty trong trajectory |
| LCATS | KDD 2026 | discovery + diagnose + post-repair | LLM constraint loop và endpoint repair | không điều khiển intermediate trajectory |
| SAGE | ACL 2026 main | value-dependent dependency cho LLM table generator | dependency thay đổi theo row/value | không phải diffusion trajectory; MI signal không tự có authority |
| PromptLoop | CVPR 2026 | model xem intermediate preview và update prompt | feedback giữa denoising đã có ở image | chưa giải quyết typed mixed-table factors, validation và coverage |
| Limited Guidance Interval | NeurIPS 2024 | chỉ guidance trong interval hữu ích | lý do không guidance toàn trajectory | chưa có graph/edge-specific automatic activation cho tabular |

### Claim có thể bảo vệ

> A verified mixed-type factor graph is dynamically activated by row state and diffusion time, then routed through typed, bounded middle-step actions that target distribution fidelity rather than indiscriminate constraint satisfaction.

### Không được claim

- LLM feedback trong diffusion là mới nói chung.
- Per-step constraint projection là mới.
- LLM compile constraint rồi verify bằng data là mới.
- Dynamic graph luôn tốt hơn global guidance: full-global hiện ngang hoặc nhỉnh hơn full-conditional ở một factorial run.
- LLM là nguồn của main gain hiện tại.

---

## 4. Pipeline hoàn chỉnh

## 4.1 Offline graph discovery

### Stage A — Candidate retrieval

Ba nguồn evidence được giữ tách biệt:

1. external metadata, codebook, expert/public constraints;
2. train-only statistical retrieval: implication, mapping, mutual information, correlation, quantile relation;
3. LLM semantic proposals từ schema, aggregates và retrieved signals.

Statistical correlation và LLM confidence không được xem là ground truth.

### Stage B — Typed proposal

Mỗi proposal được compile thành DSL có:

- `columns`/endpoints;
- `applicability`;
- `requirements`;
- `exceptions`;
- `relation_family`;
- `hardness_prior`;
- `evidence_basis`;
- `usage_role`;
- action candidates.

### Stage C — Deterministic verification

- type checking;
- executable evaluation chỉ trên train/discovery split;
- counterexample mining;
- support và Wilson bounds;
- directionality check;
- sensitive/target leakage flags;
- aggregate counterexample signature cho revision loop.

### Stage D — Semantic review và authority

- `actuate`: public/expert relation hoặc relation có đủ independent authority;
- `score_only`: plausible relation có predictive signal nhưng chưa phải invariant;
- `monitor`: exploratory relation;
- `reject`: wrong direction, low support, ambiguity hoặc leakage.

Điểm quan trọng: **semantic plausibility không đồng nghĩa quyền sửa sample**.

## 4.2 Graph representation

Graph hiện có bốn factor family:

1. **Typed semantic factor:** inequality, ordering, sum/function program.
2. **Distribution factor:** empirical marginal và Gaussian-rank/copula dependency.
3. **Conditional factor:** local distribution theo target category hoặc target quantile bin.
4. **Categorical density factor:** marginal residual message và smoothed PMI/NMI factor.

LLM định nghĩa hoặc bổ sung topology/semantics; numerical potential cuối cùng được fit/verify trên train split.

## 4.3 Dynamic activation

Tại reverse step \(t\), sampler lấy clean preview \(\hat{x}_{0,i,t}\). Edge \(e\) của row \(i\) chỉ active khi:

\[
A_{e,i,t}=R_{e,t}\land P_e(\hat{x}_{0,i,t})\land Q_e\land N_{e,i,t}\land U_{e,i,t}\land B_{e,t}.
\]

- \(R_{e,t}\): endpoints đã đủ readable;
- \(P_e\): applicability đúng;
- \(Q_e\): reliability/authority đủ;
- \(N_{e,i,t}\): state còn có distribution gap hoặc semantic residual cần xử lý;
- \(U_{e,i,t}\): uncertainty đủ thấp;
- \(B_{e,t}\): interval, trust region và pulse budget cho phép.

Với target-quantile factor, predicted target của row quyết định bin/operator hiện tại. Khi target estimate đổi theo trajectory, active local factor cũng đổi.

## 4.4 Action layer

### Typed projection

Sửa violation theo minimal typed displacement, khóa protected columns và giới hạn bằng trust region.

### Numerical distribution transport

- marginal quantile transport;
- global/graph copula alignment;
- target-conditional quantile transport;
- sliced variants chỉ dùng như diagnostic vì dễ phá coverage.

### Categorical logit guidance

Trước MDLM categorical update:

- marginal message cân bằng residual probability mass;
- target/pairwise message dùng smoothed PMI;
- chỉ cell còn masked được thay logits;
- mask logit và cell đã unmask giữ nguyên;
- timing dùng normalized expected mask probability \(1-\exp(-\sigma_{cat})\).

Các categorical run trước khi sửa timing coordinate là no-op và đã bị loại khỏi evidence.

## 4.5 Bounded middle-step injection

Action áp lên clean preview, sau đó mapped trở lại reverse update. Geometry hiện dùng:

- per-feature standardized metric;
- ellipsoidal trust region;
- one-shot hoặc finite pulses;
- minimum call gap;
- interval giữa trajectory.

Sweep cho thấy correction mạnh, rộng hoặc lặp liên tục dễ đi khỏi data manifold.

---

## 5. Dữ liệu, evaluator và metric

### Dataset đã dùng chính

- Default of Credit Card Clients;
- Online Shoppers Purchasing Intention;
- Adult;
- Magic Gamma Telescope;
- Diabetes;
- Online News Popularity.

News có năm public inequalities từ ConstrainedDGM, nên phù hợp cho comparison có independent constraint authority. Adult/Default phù hợp cho categorical graph. Magic và Diabetes chủ yếu dùng để test generalization/negative transfer.

### Metric chính

| Metric | Ý nghĩa | Chiều tốt |
|---|---|---|
| Shape | độ giống marginal một cột; ở News báo score, dataset khác báo error | News ↑; error ↓ |
| Trend | độ giống pairwise/joint trend | News ↑; error ↓ |
| Alpha precision | synthetic rows nằm trong support thật đến mức nào | ↑ |
| Beta recall | coverage/mode coverage của phân phối thật | ↑ |
| Detection | độ khó phân biệt real và synthetic; dùng implementation local | theo report local, giá trị cao hơn tốt |
| TSTR AUC/R²/NRMSE | utility khi train trên synthetic, test trên real | AUC/R² ↑; NRMSE ↓ |
| DCR | privacy proxy, báo khoảng cách tới mức cân bằng 50 khi phù hợp | gần 50 |
| Public violation | chỉ dùng cho constraint có independent/public authority | ↓ |

Không dùng self-discovered constraint satisfaction làm primary metric.

---

## 6. Kết quả chính

## 6.1 News: xác nhận ba seed cho target-quantile dynamic graph

Mỗi seed là paired backbone/dynamic sample trong cùng task. Bảng báo mean trên ba paired seed.

### One-shot, 4 target quantile bins, strength 0.10

| Metric | Backbone mean | Dynamic mean | Mean delta | Paired wins |
|---|---:|---:|---:|---:|
| Shape ↑ | 97.3387 | **97.3812** | **+0.0426** | **3/3** |
| Trend ↑ | 91.4938 | **91.5203** | **+0.0266** | **3/3** |
| Alpha ↑ | 94.4515 | **96.4019** | **+1.9504** | **3/3** |
| Beta ↑ | 42.2843 | 42.3378 | +0.0534 | 1/3 |
| R² ↑ | -3.0396 | -2.8857 | +0.1539 | 2/3 |
| NRMSE ↓ | 2.0063 | 1.9216 | -0.0847 | 2/3 |
| Public violation ↓ | 21.9849 | 21.9930 | +0.0081 | 1/3 |

Kết luận đúng: marginal/joint fidelity và precision tăng ổn định; utility có signal nhưng variance lớn; method không được tối ưu để giảm public violation ở cấu hình này.

### Two-pulse, 4 bins, strength 0.05

| Metric | Backbone mean | Dynamic mean | Mean delta | Paired wins |
|---|---:|---:|---:|---:|
| Shape ↑ | 97.3608 | **97.3943** | **+0.0335** | **3/3** |
| Trend ↑ | 91.6268 | **91.6649** | **+0.0381** | **3/3** |
| Alpha ↑ | 95.2495 | **96.8075** | **+1.5580** | **3/3** |
| Beta ↑ | 41.7407 | 41.8823 | +0.1416 | 2/3 |
| R² ↑ | -2.1764 | -2.9488 | -0.7725 | 2/3 |
| NRMSE ↓ | 1.7758 | 1.9329 | +0.1571 | 2/3 |

Hai pulse thắng utility ở hai seed nhưng thất bại lớn ở seed thứ ba, làm mean xấu. One-shot vì vậy là operating point an toàn hơn.

## 6.2 News: same-node factorial ablation

Tất cả variant sinh tuần tự trên cùng node, cùng seed/checkpoint, giảm confound do GPU nondeterminism.

| Method | Shape ↑ | Trend ↑ | Alpha ↑ | Beta ↑ | R² ↑ | NRMSE ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Backbone | 97.4853 | 91.1478 | 94.8853 | 42.3690 | -0.8374 | 1.3555 |
| Typed graph only | 97.5503 | 91.1495 | 96.7654 | 42.3958 | -0.7999 | 1.3416 |
| **Conditional transport only** | 97.5043 | **91.1564** | 94.9472 | **42.3958** | **-0.6665** | **1.2909** |
| Global transport only | 97.5057 | 91.1552 | 94.9505 | 42.3576 | -0.8672 | 1.3664 |
| Full conditional | 97.5601 | **91.1568** | 96.9119 | 42.3958 | -0.5489 | 1.2446 |
| Full global | **97.5613** | 91.1559 | **96.9200** | **42.4512** | **-0.5438** | **1.2425** |

Điểm quan trọng:

- Conditional-only thắng `6/6` metric so với backbone.
- Global-only chỉ thắng Shape, Trend và Alpha; Beta và utility xấu hơn.
- Khi ghép typed graph, cả full-conditional và full-global đều thắng `6/6`; full-global hơi nhỉnh ở 5/6 metric, chênh lệch rất nhỏ.
- Vì vậy conditioning có contribution riêng rõ, nhưng chưa chứng minh full method dynamic luôn hơn global factor.

## 6.3 Adult: categorical graph screening

| Variant | Shape err. ↓ | Trend err. ↓ | Alpha ↑ | Beta ↑ | Detection ↑ | AUC ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Paired TabDiff | 1.4240 | 2.8815 | 97.3193 | 49.1423 | 0.99360 | 0.90087 |
| Target factor, strength .25 | 1.4048 | 2.8272 | 97.6383 | **49.4499** | **0.99648** | 0.90209 |
| **Hybrid marginal + target** | **1.3981** | **2.8228** | **97.6627** | 49.4385 | 0.99641 | **0.90261** |
| Pairwise factor | 1.4490 | 2.9819 | 97.1942 | 49.7786 | 0.99283 | 0.90342 |

Hybrid so với paired backbone:

- Shape error giảm `1.82%`;
- Trend error giảm `2.04%`;
- Alpha `+0.3434`;
- Beta `+0.2962`;
- detection `+0.00281`;
- AUC `+0.00174`.

Pairwise factor cho Beta/AUC cao hơn nhưng phá Shape/Trend/Alpha; đây là evidence rằng topology dày không tự động tốt hơn.

### Kết quả xác nhận ba seed

| Metric | TabDiff mean | Categorical graph mean | Mean delta | Paired wins |
|---|---:|---:|---:|---:|
| Shape error ↓ | **1.2211** | 1.2357 | +0.0147 | 1/3 |
| Trend error ↓ | **2.5383** | 2.6021 | +0.0638 | 0/3 |
| Alpha ↑ | 98.6041 | **98.7113** | +0.1073 | 2/3 |
| Beta ↑ | **48.7126** | 48.4744 | -0.2382 | 0/3 |
| Detection ↑ | 0.99035 | 0.99202 | +0.00167 | 1/3; hai seed bão hòa |
| AUC ↑ | 0.89719 | **0.89820** | +0.00101 | 2/3 |
| Held-out relation error ↓ | **0.04011** | 0.04133 | +0.00123 | 0/3 |

Kết luận: categorical messages có thể tăng Alpha/AUC nhẹ nhưng phá Trend/Beta ổn định. Screening `6/6` là false positive theo seed; branch này không được đưa vào main result hiện tại.

## 6.4 Default: common-metric trade-off và categorical/numerical hybrid

Graph controller trước đây, trung bình ba seed:

| Method | Shape err. ↓ | Trend err. ↓ | Alpha ↑ | Beta ↑ | AUC ↑ | Held-out relation err. ↓ |
|---|---:|---:|---:|---:|---:|---:|
| TabDiff | 1.9833 | 2.2165 | 98.4736 | **52.5109** | 0.7230 | 0.0309 |
| Dynamic graph | 1.9591 | 2.2116 | **98.5218** | 52.4436 | 0.7228 | **0.0120** |
| Frozen graph | 1.9600 | 2.2119 | 98.4990 | 52.4533 | 0.7238 | 0.0154 |
| Final repair | **1.9556** | **2.1909** | 98.5003 | 52.4761 | **0.7278** | 0.0161 |

Dynamic graph giảm held-out relation error mạnh nhưng common metrics gần như hòa; final repair còn cạnh tranh. Đây là lý do project chuyển objective từ violation repair sang distribution factors.

Categorical hybrid mới ghép marginal/target logit messages với conditional numerical transport trong một paired run:

| Method | Shape err. ↓ | Trend err. ↓ | Alpha ↑ | Beta ↑ | Detection ↑ | AUC ↑ | DCR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Paired backbone | 1.6842 | 1.9793 | **99.3563** | **50.8708** | 0.97131 | 0.73614 | **51.3916** |
| Hybrid graph | **1.6550** | **1.9691** | 99.3513 | 50.7829 | **0.97154** | **0.74926** | 51.6113 |

Hybrid làm Shape error giảm `1.73%`, Trend error giảm `0.51%` và AUC tăng `0.0131`; đổi lại Alpha/Beta/DCR giảm nhẹ. Đây là Pareto trade-off tốt nhưng không phải uniform win.

## 6.5 Generalization và negative transfer

| Dataset | Operating point | Kết quả chính |
|---|---|---|
| Magic | middle strength 1.0 | Trend error giảm khoảng `0.12%`, Beta `+0.0033`, AUC `+0.0036`; effect rất nhỏ |
| Diabetes | middle strength 1.0 | Trend error giảm `1.26%`, Beta `+0.0163`, AUC `+0.0003`; graph chỉ có 29 projection events |
| Shoppers | progressive controller | online action bằng 0 vì evidence gate abstain; output bằng backbone |
| Adult, graph cũ | typed numeric graph | gần no-op |
| Adult, categorical graph mới | marginal+target factors | paired screening thắng 6/6 common metrics; đang chờ ba-seed confirmation |

Kết luận: cơ chế phải khớp loại dependency của dataset. Numeric typed projection không đủ cho Adult; categorical density graph mở được signal. Điều này ủng hộ mixed-action graph, đồng thời bác bỏ một universal action schedule.

## 6.6 Screening News có effect size lớn nhưng chưa phải confirmation

Best paired one-shot global copula run:

| Metric | Backbone | Guided | Change |
|---|---:|---:|---:|
| Shape ↑ | 97.4792 | 97.5466 | Shape error giảm 2.67% |
| Trend ↑ | 90.4259 | 90.4793 | Trend error giảm 0.56% |
| Alpha ↑ | 95.1401 | 97.1365 | +1.9964 |
| Beta ↑ | 42.1615 | 42.2778 | +0.1164 |
| R² ↑ | -1.5686 | -0.9202 | +0.6484 |
| NRMSE ↓ | 1.6027 | 1.3857 | giảm 13.54% |

Đây là screening operating point, không được thay thế bảng ba seed.

---

## 7. Baseline và context

## 7.1 Published generators chạy local

Các hàng dưới dùng local split/evaluator nhưng hiện chỉ one-run context, không phải significance comparison.

| Dataset | Method | Shape err. ↓ | Trend err. ↓ | Alpha ↑ | Beta ↑ | Utility AUC ↑ |
|---|---|---:|---:|---:|---:|---:|
| Default | TabDiff | 2.3851 | 2.5518 | 96.2240 | **53.3024** | 0.7175 |
| Default | **TabSyn** | **1.4810** | **1.6499** | **99.2198** | 47.7816 | 0.7127 |
| Default | ForestDiffusion | 8.9813 | 6.1003 | 94.4882 | 47.8939 | 0.6968 |
| Default | CTGAN | 15.2233 | 17.8281 | 64.4922 | 21.3704 | 0.6695 |
| Default | TVAE | 10.8799 | 6.5209 | 82.3844 | 25.9066 | **0.7244** |
| Shoppers | TabDiff | **1.6044** | 2.5302 | 96.3738 | **50.3320** | 0.8978 |
| Shoppers | TabSyn | 1.6652 | **1.9929** | **99.0039** | 46.3363 | 0.8817 |
| Shoppers | ForestDiffusion | 14.5687 | 4.6032 | 91.0807 | 49.7168 | **0.8992** |
| Adult | TabDiff | **1.5066** | **2.7409** | **97.7962** | **50.1660** | **0.8975** |
| Magic | TabDiff | **1.6857** | **1.3771** | **98.9840** | 47.3242 | **0.9181** |
| Magic | ForestDiffusion | 3.3541 | 3.6141 | 98.1478 | **54.0137** | 0.9092 |

Phải giữ distinction: method của project hiện là add-on trên TabDiff. Nó nên được so paired với TabDiff trước; direct claim vượt TabSyn chỉ hợp lệ khi cùng split/checkpoint/evaluator và đủ seed.

## 7.2 Published constraint mechanisms

### News public constraints

| Method | Status | Shape ↑ | Trend ↑ | Alpha ↑ | Beta ↑ | R² ↑ | NRMSE ↓ | Violation ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| TabDiff | published backbone | **96.6554** | 89.1527 | 97.1327 | **42.8825** | -0.0254 | 1.0126 | 25.8789% |
| ConstrainedDGM layer | adapted published operator | 96.2996 | 90.5950 | **97.1344** | 42.8988 | -0.0293 | 1.0146 | **0.0000%** |
| Dynamic middle + certifier | proposed earlier branch | 96.5864 | **90.6927** | 97.1024 | 42.8711 | **-0.0054** | **1.0027** | **0.0000%** |

Đây là bảng cho public validity setting; distribution-graph branch mới không tối ưu zero violation và phải được báo ở bảng fidelity riêng.

### Official LCATS audit trên Insurance

| Candidate | Any violation ↓ | Numeric KS ↓ | Cat. TV ↓ | Corr. MAE ↓ | TSTR NRMSE ↓ | TSTR R² ↑ | Rows edited ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| TabSyn original | 0.9269 | 0.0370 | **0.0066** | 0.0118 | **0.5254** | **0.7239** | — |
| LCATS aligned | **0.0000** | **0.0299** | **0.0066** | **0.0115** | 0.5335 | 0.7153 | 100% |

LCATS chứng minh endpoint repair có thể đạt zero violation và cải thiện một số fidelity metric, nhưng utility giảm và mọi row bị sửa. Đây là motivation cho bounded trajectory intervention, không phải direct same-dataset ranking.

### Harpoon

Official code đã chạy trên Adult/Default/Shoppers với request-specific constraints. Metric protocol của Harpoon khác local dynamic-graph evaluator, nên chỉ dùng làm mechanism baseline; không tuyên bố proposed vượt Harpoon từ hai bảng khác protocol.

## 7.3 Paper-reported SCTab-Diff chỉ làm context

Paper SCTab-Diff báo Shape/Trend/SA rất cao trên pipeline riêng. Local News Trend khoảng `89–92`, trong khi paper báo quanh `98`, cho thấy split/checkpoint/evaluator khác đáng kể. Không ghép số paper-reported vào local main table.

---

## 8. Bằng chứng cho từng hypothesis

| Hypothesis | Trạng thái | Bằng chứng |
|---|---|---|
| H1: intermediate \(\hat x_0\) đọc được ở middle steps | **Ủng hộ** | precision/recall tăng từ `0.6688/0.2416` ở early lên `0.9216/0.9311` tại q `0.25–0.50`, `0.9874/0.9926` tại q `0.10–0.25` |
| H2: state/row-conditioned factor hữu ích hơn fixed global factor | **Ủng hộ một phần** | conditional-only thắng 6/6, global-only 3/6; nhưng full-global ngang/nhỉnh full-conditional |
| H3: middle trajectory tốt hơn endpoint repair | **Ủng hộ cho một số setting** | Default progressive giảm actionable held-out error `19.38%` so với strong final-only; News dynamic+certifier tốt hơn final operators về Trend/utility; common metrics không luôn thắng |
| H4: LLM thêm thông tin ngoài rule engine | **Có ở discovery, chưa có ở generation gain** | News recall `64.3% → 78.6%`; Default no-LLM graph trùng 7/8 edges; LLM soft actuation không ổn định |
| H5: feedback đáng tin và đáng chi phí | **Compile-once tốt hơn direct judge** | request probe: NL→DSL→executor đạt execution equivalence 1.0 với khoảng 1.85k tokens; direct judgement tốn khoảng 37.5k tokens và accuracy/swap consistency thấp hơn. Focused audit cũ bị label leakage nên đã loại |
| H6: text–structured consistency | **Chưa thử đúng nghĩa** | dataset hiện không cung cấp long-text mixed generation đủ phù hợp; không dùng nhánh này để chặn main work |
| H7: stronger/more frequent guidance tốt hơn | **Bác bỏ** | strength cao, interval rộng hoặc unbounded làm Beta/utility/Shape giảm; weak bounded middle action tốt hơn |
| H8: graph nhiều edge hơn tốt hơn | **Bác bỏ** | Adult pairwise graph tăng Beta/AUC nhưng phá Shape/Trend; News positive factors active hàng trăm nghìn lần nhưng gain nhỏ |

---

## 9. Negative results cần giữ trong paper

1. LLM direct judgement trên intermediate rows không đủ ổn định và quá đắt; compile-once + deterministic execution tốt hơn.
2. Nhiều LLM constraints semantic nghe hợp lý nhưng có empirical violation rất lớn; deterministic verifier là bắt buộc.
3. Statistics-only mining sinh nhiều threshold variants nhưng semantic capacity thấp.
4. Constraint satisfaction không đồng nghĩa data quality; real data cũng có violation và self-discovered relation có thể sai.
5. Continuous guidance và strength cao gây off-manifold behavior.
6. Sliced transport có thể làm utility mạnh hơn nhưng phá Shape/Trend/Beta; endpoint marginal calibration cho score rất cao nhưng có metric-gaming risk.
7. Adult pairwise dense factor tăng coverage/utility nhỏ nhưng giảm marginal/joint fidelity.
8. Magic có Shape–Trend trade-off; Shoppers controller nhiều lần abstain/no-op; Diabetes chỉ có một sparse actionable edge.
9. GPU class làm unconditional outputs không bit-deterministic dù cùng seed; chỉ paired comparison trong cùng task là an toàn.
10. Full-global hiện có thể matching full-conditional; dynamic conditioning chưa phải universal winner.
11. LCATS-style integer/bound guidance ở middle steps không duy trì domain validity: integer macro violation chỉ giảm `20.428% → 20.370%`. Exact endpoint rounding đạt `0%` với rất ít common-metric distortion, nên unary domain integrity không tạo motivation cho trajectory guidance.

---

## 10. Đánh giá contribution hiện tại

### Contribution đã đủ rõ

1. Typed multi-source graph với authority tách khỏi confidence.
2. Dynamic row/time activation dựa trên current clean preview và reliability.
3. Action layer mixed-type: projection, numerical transport, categorical logits.
4. Bounded middle-step schedule được evidence ủng hộ, không guidance toàn trajectory.
5. Evaluation tách public constraints khỏi self-discovered relations và tập trung common metrics.

### Contribution chưa đủ mạnh

1. Full method chưa vượt global comparator rõ rệt.
2. LLM chưa tạo downstream gain riêng có hệ thống.
3. Cross-dataset improvements còn không đồng đều.
4. Absolute utility News còn yếu/âm ở một số checkpoint run.
5. Chưa có validation protocol khóa hyperparameters trước final test.

### Đánh giá khả năng thành paper ICDE

**Có thể form thành project/paper**, nhưng cần định vị đúng:

- không bán như một SOTA universal generator;
- bán như dynamic graph controller/add-on cho mixed-type diffusion;
- main evidence phải là paired same-backbone, factorial ablation, timing/geometry, public-constraint benchmark và cross-dataset failures;
- LLM là offline semantic coverage module, không phải trung tâm online method;
- một claim thực nghiệm dày, trung thực và có mechanism explanation sẽ thuyết phục hơn một headline gain chọn lọc.

Tại thời điểm này, project khoảng **70–75% phần method/prototype**, nhưng chỉ khoảng **50–60% phần final experimental package** vì còn thiếu frozen protocol, multi-seed cross-dataset và comparison đủ đồng nhất.

---

## 11. Bảng nào nên vào paper

### Main table A — Same-backbone common metrics

- Dataset: News, Adult, Default và một dataset negative/neutral.
- Rows: TabDiff, strongest published same-pipeline operator, proposed full, global-factor control.
- Columns: Shape, Trend, Alpha, Beta, TSTR utility, runtime.
- Báo mean ± std tối thiểu ba seed.

### Main table B — Public constraint setting

- News public inequalities.
- TabDiff, final repair, ConstrainedDGM adapted layer, Harpoon-style proxy, proposed + certifier.
- Báo violation cùng common metrics để hiện trade-off.

### Main table C — Published generator context

- TabDiff, TabSyn, ForestDiffusion, TabDDPM nếu evaluator hoàn chỉnh.
- Không trộn paper-reported SCTab-Diff với local rows.

### Ablation bắt buộc

1. typed-only / conditional-only / global-only / full-conditional / full-global;
2. early/middle/late, one-shot/two-pulse/continuous;
3. no trust / trust region / strength;
4. no-LLM / statistics / LLM+verify / combined;
5. marginal / target / pairwise categorical factors;
6. final repair versus trajectory intervention;
7. runtime, action count và edge activation density.

---

## 12. Kế hoạch ngắn nhất để đóng khung method

### Bắt buộc trước khi viết claim cuối

1. Loại Adult categorical hybrid hiện tại khỏi main configuration; chỉ giữ negative ablation.
2. Chọn một validation split cho `strength`, interval và bin count; test không được dùng để chọn.
3. Chạy final locked configuration trên News và Default; chỉ thêm dataset thứ ba sau khi validation có signal.
4. Báo full-global comparator trong main ablation vì nó hiện matching proposed.
5. Thêm runtime/action count/peak memory.

### Chỉ làm nếu còn thời gian

1. Factorized discriminator graph hoặc residual score critic, vì đây là hướng trực tiếp tối ưu data distribution gap.
2. Posterior-smoothed edge activation để interval tự phát từ uncertainty.
3. Lightweight end-to-end adapter sau khi inference-time signal đã ổn định.

### Không nên tiếp tục ngay

- mở thêm nhiều dataset trước khi khóa protocol;
- tăng số edge bằng mọi giá;
- gọi LLM trong từng denoising step;
- tối ưu constraint satisfaction do chính pipeline định nghĩa;
- post-hoc calibration chỉ để tạo một bảng score đẹp.

---

## 13. Trạng thái job và đường dẫn tái lập

### Job quan trọng

| Job | Nội dung | Trạng thái |
|---|---|---|
| `1099203` | global/graph copula one-shot, multi-dataset | hoàn tất |
| `1099227` | strength, interval và trust geometry | hoàn tất |
| `1099341` | target-quantile News pilot | hoàn tất |
| `1099422` | weak pulses News + categorical Default | hoàn tất |
| `1099434` | Adult categorical factor sweep | hoàn tất |
| `1099444` | News 3-seed confirmation | hoàn tất |
| `1099460` | News same-node factorial ablation | hoàn tất |
| `1100253` | Adult hybrid categorical, 3-seed confirmation | hoàn tất; không tái lập screening gain |
| `1101671` | LCATS-style integer/bound trajectory screening | hoàn tất; middle injection gần như không hiệu quả |
| `1101700` | exact LCATS-style endpoint domain repair | hoàn tất; zero domain violation, trade-off nhỏ |

### Code chính

- `scripts/dense_numeric_factor_messages.py`
- `scripts/batch_distribution_transport.py`
- `scripts/categorical_logit_guidance.py`
- `slurm/806_adult_categorical_hybrid_confirmation.sbatch`

### Kết quả chính

- `outputs/news_same_node_factorial_1099460/eval/RESULTS.md`
- `outputs/news_quantile_confirm_*_1099444_*/eval/metrics.csv`
- `outputs/adult_categorical_factors_*_1099434_*/eval/RESULTS.md`
- `outputs/distribution_graph_middle_news_copula_s050_once_1099203_2/eval/RESULTS.md`
- `outputs/pulsed_factors_default_cat_hybrid_scaled_condnum_1099422_11/eval/RESULTS.md`

---

## 14. Main conclusion đề xuất cho bản viết hiện tại

> Static constraint enforcement is too coarse for mixed-type tabular diffusion: features become reliable at different times, relations have different authority, and indiscriminate correction can improve validity while damaging coverage. The proposed framework compiles verified semantic and train-fitted distribution factors into a graph, activates them by row and diffusion state, and routes them through bounded typed actions at informative reverse steps. Experiments show consistent gains in marginal/joint fidelity and precision on News, promising all-metric paired gains on Adult, and clear failures of stronger, denser or continuously applied guidance. LLMs improve offline semantic coverage, while deterministic verification and train-fitted factors—not online LLM calls—provide the reliable control signal.

Đây là conclusion phù hợp với evidence hiện có. Nếu Adult confirmation thành công và final-global comparator không vượt rõ, project đã đủ để bắt đầu viết method/evaluation; nếu Adult không lặp lại, cần hạ claim xuống “News mechanism study + mixed-type categorical case study” hoặc phát triển residual-score graph trước khi chốt main table.
