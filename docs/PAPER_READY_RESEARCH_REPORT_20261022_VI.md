# Báo cáo chốt hướng bài: Dynamic factor-graph guidance và graph-consistent output selection cho mixed-type tabular diffusion

**Ngày chốt bằng chứng:** 22/10/2026  
**Mức sẵn sàng ước lượng:** khoảng **65–70% gói thực nghiệm**, đủ để bắt đầu viết Introduction, Method, Experimental Setup và phần lớn Results.  
**Quy ước SOTA trong báo cáo:** “local SOTA” chỉ có nghĩa là tốt nhất trong các baseline đã chạy lại trên cùng split/evaluator/protocol của repository. Không đồng nhất với paper-reported SOTA từ protocol khác.

---

## 1. Kết luận điều hành

Project hiện đã có một câu chuyện khoa học tương đối hoàn chỉnh gồm ba tầng dùng chung factor graph:

1. **Factor graph offline** biểu diễn typed constraints, train-fitted distribution relations và conditional dependencies, kèm authority/reliability.
2. **Dynamic middle guidance** kích hoạt factor theo từng row, trạng thái và diffusion timestep, rồi đưa correction có giới hạn vào reverse trajectory.
3. **Graph-consistent output selection** dùng lại chính active factors ở tầng sampling để sửa rất ít membership quanh một generic C3 anchor; repair chỉ kích hoạt khi tập output còn graph deficit.

Kết quả headline tốt nhất hiện tại là Adult trên ba fresh seeds:

- Shape/Trend `0.860949/2.017464`;
- cải thiện so với dynamic generator `G`: `+0.371348/+0.521812`, thắng `3/3` seed cho cả hai metric;
- cải thiện so với frozen generic C3: `+0.008133/+0.024268`;
- graph relation gain `+0.000300`, không mất coverage;
- feasible `3/3`, trong đó một seed được deficit gate xác định đúng là không cần repair;
- đạt `4/5` local baseline records;
- quyết định khóa trước: **comprehensive**.

Shoppers cung cấp một transfer result ở mức **coherent**: graph relation và coverage đều tốt hơn C3 trên `3/3` fresh seeds, vẫn giữ gain lớn so với dynamic generator và đạt `3/5` local records. Magic và Default không xác nhận được graph-repair gain theo gate nghiêm ngặt; chúng là negative/boundary results quan trọng. Frozen C3 độc lập vẫn rất mạnh trên Magic, Diabetes và Shoppers.

### Quyết định cho việc viết bài

**Có thể bắt đầu viết bài ngay.** Phần còn lại trong hai tuần không nên tiếp tục tìm method mới. Nên:

- khóa method cuối;
- chạy final standardized protocol trên 3–4 dataset chính;
- hoàn thiện ablation, confidence intervals, runtime và baseline alignment;
- đóng gói reproduction và viết.

Không nên claim “universal SOTA generator”. Claim phù hợp là:

> A factor-shared controller for mixed-type tabular diffusion that dynamically routes verified factors during denoising and conditionally repairs the generated set through sparse, graph-proximal membership swaps.

---

## 2. Working title và định vị

### Title đề xuất

**Factor-Shared Dynamic Graph Guidance for Mixed-Type Tabular Diffusion**

Các phương án khác:

- **Trajectory-to-Set Factor Graphs for Mixed-Type Tabular Diffusion**
- **Dynamic Factor Routing and Graph-Proximal Output Repair for Tabular Diffusion**
- **One Graph, Two Scales: Dynamic Guidance and Set-Level Repair for Tabular Diffusion**

### Định vị đúng

Method là một **add-on controller/selection framework trên TabDiff**, không phải backbone diffusion hoàn toàn mới. Điểm mới nằm ở việc factor graph được dùng nhất quán ở hai scale:

- row/trajectory scale trong reverse diffusion;
- dataset/set scale tại terminal output selection.

Generic C3 vẫn được giữ nguyên và chạy độc lập. Proposed selector không thay thế C3 bằng một arbitrary graph objective; nó sửa một lân cận rất nhỏ quanh frozen C3 anchor, có train-only feasibility constraints và no-op khi anchor đã graph-consistent.

---

## 3. Research question và khoảng trống

### Research question

> Làm thế nào đưa verified semantic và statistical relations vào mixed-type diffusion theo từng row và timestep, đồng thời bảo toàn factor-conditioned subgroup distributions ở tập output cuối mà không phá fidelity, coverage và utility?

### Khoảng trống của pipeline cũ

Dynamic middle injection và C3 ban đầu bị tách semantic:

- trajectory controller dùng verified factor graph;
- C3 dùng generic marginals, pair statistics và coverage proxies;
- một row có thể được dynamic guidance sửa đúng, nhưng terminal set selection vẫn loại hoặc giữ row theo objective không biết factor nào đã được actuate;
- kết quả là local constraint/factor consistency và global set coverage có thể không đồng pha.

### Failure mode được giải quyết

Dynamic graph can thiệp ở row level nhưng không trực tiếp đảm bảo rằng tập cuối giữ đủ support cho các factor-conditioned subgroups. Graph-proximal selection sửa failure này bằng cách:

- dùng train factor-state distribution làm target;
- đo relation discrepancy, support gap và hard-factor excess trên tập selected;
- bắt đầu từ exact C3 subset;
- chỉ thay một tỷ lệ membership rất nhỏ;
- giữ generic C3 quality như constraint/lexicographic priority;
- abstain nếu graph deficit đã đủ thấp.

---

## 4. Method hoàn chỉnh

## 4.1 Offline factor graph

Mỗi factor chứa:

- endpoints/columns;
- applicability predicate;
- requirement hoặc statistical potential;
- relation family;
- hardness và authority;
- confidence, support và bootstrap stability;
- usage role: `actuate`, `score_only`, `monitor`, hoặc `reject`;
- action family phù hợp.

Graph có thể chứa:

1. typed semantic factors;
2. marginal/distribution factors;
3. conditional numerical factors;
4. categorical marginal/target/pair factors.

LLM chỉ nên tham gia offline discovery/semantic review. Numerical potential và executable validity được fit/verify deterministically trên train split.

## 4.2 Dynamic row/time activation

Với clean preview \(\hat{x}_{0,i,t}\), factor \(e\) của row \(i\) chỉ active khi endpoints đủ readable, applicability đúng, authority đủ, current state còn residual cần sửa, uncertainty đủ thấp và pulse/trust budget cho phép.

Một biểu diễn gọn:

\[
A_{e,i,t}=R_{e,t}\,P_e(\hat{x}_{0,i,t})\,Q_e\,N_{e,i,t}\,U_{e,i,t}\,B_{e,t}.
\]

Điểm quan trọng là active graph không cố định cho toàn batch hoặc toàn trajectory.

## 4.3 Typed middle-step actions

- **Typed projection:** minimal feasible displacement cho relation có authority cao.
- **Numerical transport:** marginal, copula hoặc target-conditional distribution correction.
- **Categorical messages:** marginal residual và target/pairwise PMI messages trên logits của cell còn masked.
- **Trust geometry:** bounded displacement, pulse budget và middle interval.

Các sweep trước cho thấy weak one-shot hoặc vài pulse ở middle steps thường tốt hơn continuous/strong guidance.

## 4.4 Frozen generic C3 anchor

C3 tạo một subset chất lượng từ oversampled pool bằng train-only marginal/pair matching, consensus membership và coverage barriers. C3 không cần factor graph và là ablation độc lập:

\[
S_{\mathrm{C3}}=\arg\min_{|S|=n} Q_{\mathrm{generic}}(S;T).
\]

Điều này rất quan trọng cho paper: nếu bỏ graph-proximal repair, method cũ vẫn hoạt động và vẫn có strong results.

## 4.5 Factor-shared graph-proximal output repair

Từ anchor \(S_0=S_{\mathrm{C3}}\), graph loss có dạng:

\[
J_{\mathcal F}(S)=
\sum_{e\in\mathcal F}w_eD_e(S,T)
+\lambda_c\,\mathrm{CoverageGap}_{\mathcal F}(S,T)
+\lambda_h\,\mathrm{HardExcess}_{\mathcal F}(S,T).
\]

Thay vì redraw toàn bộ subset, method chỉ xét một proximal neighborhood:

\[
\mathcal N_\rho(S_0)=\{S: |S\triangle S_0|\le 2\rho n\}.
\]

First-order residual của factor-state distribution tạo score remove/add. Candidate phải:

- đạt minimum graph relation gain;
- không làm xấu factor coverage;
- không tăng hard-factor excess;
- nằm trong generic C3 quality tolerance;
- giữ integrated train-coverage barrier nếu dataset yêu cầu.

Trong feasible set, proposed method chọn candidate có generic quality tốt nhất rồi dùng graph relation làm tie-break. Đây là **quality-first lexicographic selection**.

## 4.6 Deficit-activated repair

Relative graph-gain target trở nên ill-conditioned khi anchor đã gần hoàn hảo. Final method dùng gate:

\[
S^*=\begin{cases}
S_0, & J_{\mathcal F}(S_0)\le\tau_a,\\
\operatorname{Repair}_{\rho,\gamma}(S_0), & \text{otherwise}.
\end{cases}
\]

Current locked setting:

- activation threshold \(\tau_a=0.001\);
- swap fraction `0.25%`;
- required graph gain `5%` khi active;
- generic quality tolerance `1.5%`;
- 256 train-only candidate draws.

Trên Adult confirmation, gate repair hai seeds và no-op đúng một seed có graph loss thấp.

---

## 5. Contribution focus đề xuất

Không nên dàn trải quá nhiều contribution. Nên tập trung ba contribution chính.

### C1 — Dynamic factor routing cho mixed-type diffusion

Một verified factor graph được kích hoạt theo row, state và diffusion time; action được route sang typed projection, numerical transport hoặc categorical message trong bounded middle interval.

Điểm cần chứng minh:

- dynamic/conditional factor có signal hơn fixed action khi tách riêng;
- middle weak guidance tốt hơn continuous/strong guidance;
- different factor families phù hợp với different dependency structures.

### C2 — Factor-shared graph-proximal set repair

Sampling và output selection dùng chung active factors. Terminal stage không redraw toàn subset mà thực hiện sparse membership repair quanh frozen generic C3 anchor.

Điểm cần chứng minh:

- scalar graph objective dễ phá quality;
- full-redraw Pareto có graph gain nhưng generalization không ổn định;
- proximal repair giảm semantic mismatch;
- quality-first tốt hơn graph-first;
- deficit gate tránh harmful correction khi anchor đã đủ tốt.

### C3 — Authority-aware và fail-closed evaluation

Self-discovered relations không được dùng như ground-truth primary metric. Selection chỉ đọc train/pool statistics; test evaluator không đi vào search. Public validity, common fidelity, utility, coverage và privacy được báo tách biệt.

Điểm cần chứng minh:

- LLM confidence không đồng nghĩa authority;
- graph feasibility được audit bằng shared external function;
- negative results và no-promotion decisions được giữ nguyên.

### Contribution phụ, không nên đặt headline

- LLM-assisted offline graph discovery;
- integrated train-coverage barrier;
- best-of-N/consensus C3;
- public constraint certifier.

Các thành phần này hỗ trợ framework nhưng không nên cạnh tranh attention với ba contribution chính.

---

## 6. Experimental protocol hiện tại

### Dataset

- Adult;
- Online Shoppers;
- Magic Gamma Telescope;
- Diabetes;
- Default of Credit Card Clients;
- Online News Popularity.

### Controls quan trọng

- `B`: backbone TabDiff/unconditional sample;
- `G`: dynamic graph-guided generator;
- `R`: independently fixed random/uniform pool subset;
- `old`: frozen generic C3 selector;
- `graph`: deficit-activated graph-proximal repair.

### Metrics

- Shape error/score;
- Trend error/score;
- Alpha precision;
- Beta recall/coverage;
- TSTR AUC hoặc R²/NRMSE;
- DCR privacy proxy;
- public violation khi có independent constraint authority;
- graph relation, factor coverage và hard excess chỉ dùng như mechanism/audit metrics.

### Reproducibility

Mỗi output root lưu:

- fixed seed;
- exact selected indices;
- pool/sample CSV;
- configuration và source hashes;
- Git revision;
- GPU metadata;
- selection trace;
- factor-level graph audit;
- common evaluator output.

---

## 7. Kết quả thực nghiệm tổng hợp

## 7.1 Dynamic guidance trước output selection

### News, target-quantile one-shot, ba paired seeds

| Metric | Backbone | Dynamic | Delta | Wins |
|---|---:|---:|---:|---:|
| Shape ↑ | 97.3387 | **97.3812** | **+0.0426** | 3/3 |
| Trend ↑ | 91.4938 | **91.5203** | **+0.0266** | 3/3 |
| Alpha ↑ | 94.4515 | **96.4019** | **+1.9504** | 3/3 |
| Beta ↑ | 42.2843 | 42.3378 | +0.0534 | 1/3 |
| R² ↑ | -3.0396 | -2.8857 | +0.1539 | 2/3 |
| NRMSE ↓ | 2.0063 | 1.9216 | +0.0847 benefit | 2/3 |

Kết luận: dynamic guidance có evidence ổn định nhất ở Shape/Trend/Alpha; utility và coverage còn variance.

### Same-node factorial trên News

Conditional numerical transport đơn lẻ thắng backbone ở `6/6` common metrics, trong khi global transport đơn lẻ chỉ thắng `3/6`. Khi ghép typed graph, full-global và full-conditional gần như hòa. Do đó row conditioning có contribution riêng, nhưng không được claim universal superiority.

## 7.2 Frozen generic C3: bằng chứng độc lập

| Dataset | Locked base + C3 | Shape | Trend | Gain vs `G` Shape/Trend | SOTA records | Decision |
|---|---|---:|---:|---:|---:|---|
| Magic | `proj+gt4` | 1.009621 | 1.060297 | +0.318999 / +0.887710 | 3/5 | comprehensive |
| Diabetes | `hybrid+int20_t005` | 1.353019 | 8.165711 | +0.319212 / +1.137281 | 3/5 | primary confirmed |
| Shoppers | `proj+gt4` | 1.097399 | 2.096839 | +0.543518 / +0.107746 | 2/5 | primary confirmed |

Đây là bằng chứng quan trọng rằng oversampling + train-only set selection tạo gain lớn mà không cần graph repair. Vì vậy paper phải có `old` như một strong ablation, không được dùng một random selector yếu làm strawman.

## 7.3 Final graph-proximal confirmation

| Dataset | Final method | Seeds | Shape | Trend | Gain vs `G` | Gain vs `old` | Graph gain | Local SOTA | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| **Adult** | `cat025+gt4+gate-pqs0025` | 3 fresh | **0.860949** | **2.017464** | **+0.371348/+0.521812** | **+0.008133/+0.024268** | +0.000300 | **4/5** | **comprehensive** |
| Shoppers | `proj+gt4+pqs005` | 3 fresh | 1.102046 | 1.803424 | +0.389162/+0.234395 | -0.024145/-0.030366 | +0.002233 | 3/5 | coherent |
| Magic | `proj+gt4+pqs005` | 3 fresh | 0.956562 | 2.334173 | +0.246281/**-0.033541** | +0.008858/+0.007241 | +0.001749 | 2/5 | reject strict dynamic gate |
| Default | `hybrid+int20+gate-pqs0025` | 3 held-out | 0.964895 | 1.435227 | +0.376324/+0.027127 | **-0.007101/-0.000045** | +0.002588 | 2/5 | reject |
| Diabetes | proximal quality-first screen | 3 screen | — | — | primary gains retained | — | infeasible | 3/5 context | reject |

### Adult: headline result

Adult final metrics:

- Alpha `99.068206`;
- Beta `48.155382`;
- AUC `0.897593`;
- feasible `3/3`;
- activation skipped `1/3` seed;
- graph coverage không xấu đi;
- Shape/Trend thắng dynamic `G` trên cả ba seed;
- Shape/Trend thắng frozen C3 trên hai seed và positive mean.

Đây là kết quả đủ mạnh để làm main-table headline.

### Shoppers: transfer/coherence result

Repair thay rất ít rows, graph relation/coverage tốt hơn trên `3/3`, và vẫn giữ gain lớn so với `G`. Nó hơi kém frozen C3 trên Shape/Trend nhưng nằm trong utility floor khóa trước. Dùng Shoppers để claim semantic coherence và bounded cost, không claim strict dominance.

### Default: boundary result

Ba screen seeds ban đầu cho kết quả tốt, nhưng ba held-out pools không xác nhận improvement so với `old`. Graph relation vẫn tốt hơn, cho thấy factor objective hoạt động, nhưng external quality không generalize. Đây là evidence rằng graph consistency không tự động đồng nghĩa fidelity/utility.

### Magic và Diabetes

- Magic: generic C3 rất mạnh; graph repair tăng graph consistency và có thể tăng C3 nhẹ, nhưng không giữ Trend gain so với dynamic generator.
- Diabetes: active graph quá sparse và integrated coverage objective chi phối; proximal candidates không đạt graph feasibility ổn định.

Hai dataset này giúp giới hạn claim và motivate deficit/authority-aware activation.

## 7.4 Method-development ablation đã hoàn thành

### Scalar graph selector

Scalarization cải thiện graph audit nhưng phá external quality trên Magic/Shoppers. Giữ làm ablation “graph objective alone is insufficient”.

### Full-redraw constrained Pareto

Magic screen rất mạnh và feasible `3/3`, nhưng fresh confirmation giảm Shape và không thắng `G` ở Trend. Kết luận: feasibility constraints đúng hướng, nhưng redraw distance gây instability.

### Graph-first proximal

Sparse swaps giảm graph loss mạnh, nhưng chọn theo graph-first tạo systematic Shape cost. Đây là motivation trực tiếp cho quality-first lexicographic ordering.

### Quality-first proximal

Giữ graph gain tối thiểu rồi tối ưu generic quality. Đây là variant ổn định nhất trước deficit gate.

### Deficit gate

Giải quyết trường hợp anchor có graph loss rất thấp. Adult fresh confirmation đạt `3/3` feasibility và comprehensive decision.

Chuỗi ablation khoa học nên trình bày:

`G → old C3 → scalar graph → constrained redraw → proximal graph-first → proximal quality-first → deficit-gated final`.

---

## 8. Claim matrix

| Claim | Evidence hiện có | Trạng thái |
|---|---|---|
| Dynamic middle factors cải thiện fidelity so với same-backbone | News 3 seed; same-node factorial | Supported |
| Conditional/state-aware action có ích hơn global khi tách riêng | Conditional-only 6/6 vs global-only 3/6 | Partially supported |
| Full dynamic luôn hơn full global | Full-global gần ngang/nhỉnh | **Không claim** |
| Generic C3 tạo strong set-level gains | Magic/Diabetes/Shoppers confirmations; Default/News Shape | Supported |
| Shared graph có thể sửa semantic gap của C3 | Adult comprehensive; Shoppers coherent | Supported trên selected datasets |
| Sparse proximal tốt hơn full redraw | Fresh Magic redraw failure; proximal Adult/Shoppers | Supported |
| Quality-first tốt hơn graph-first | Shoppers/Magic screen, Shape preservation | Supported |
| Deficit activation tránh harmful repair | Adult diagnosed seed + fresh no-op seed | Supported |
| Final method universal cross-dataset SOTA | Default/Magic/Diabetes failures | **Không claim** |
| Final method đạt temporary local SOTA | Adult 4/5; Shoppers 3/5 | Supported, phải ghi rõ local protocol |
| LLM là nguồn của main gain | Chưa có systematic downstream isolation | **Không claim** |

---

## 9. Main tables và figures nên có trong paper

### Main Table 1 — Same-backbone final comparison

Datasets: Adult, Shoppers, Magic, Default.  
Rows:

- TabDiff `B`;
- dynamic graph `G`;
- random pool subset `R`;
- frozen generic C3 `old`;
- proposed deficit-gated graph-proximal method.

Columns:

- Shape, Trend;
- Alpha, Beta;
- AUC/R²/NRMSE;
- graph relation/coverage audit;
- runtime.

### Main Table 2 — Local generator baselines

TabDiff, TabSyn, ForestDiffusion, TabDDPM, CTGAN, TVAE và proposed. Chỉ dùng runs cùng split/evaluator. Báo mean ± std nếu có; one-run baseline phải đánh dấu rõ.

### Main Table 3 — Public constraint setting

News public inequalities:

- TabDiff;
- final repair;
- ConstrainedDGM-style adapted layer;
- dynamic middle guidance;
- dynamic + certifier.

Báo violation song song với Shape/Trend/Alpha/Beta/utility để tránh metric gaming.

### Figure 1 — Framework diagram

Offline factor graph → row/time dynamic activation → typed middle actions → oversampled pool → frozen C3 → deficit gate → sparse membership repair.

### Figure 2 — Trajectory view

Theo timestep:

- feature readability;
- active edge count;
- action magnitude;
- graph residual;
- trust/pulse events.

### Figure 3 — Quality–graph Pareto

Plot mỗi candidate subset:

- x: generic quality loss;
- y: graph relation loss;
- màu: swap fraction;
- đánh dấu C3 anchor, full redraw, graph-first, quality-first và final.

### Figure 4 — Per-seed gains

Paired gain plot cho Shape/Trend trên Adult, Shoppers, Magic và Default. Không chỉ báo mean.

### Figure 5 — Deficit activation

Graph loss của anchor và repair decision theo seed; highlight no-op seeds.

---

## 10. Khung ablation đầy đủ

Toàn bộ ablation được gom thành **10 nhóm**. Không chạy Cartesian product. Mỗi nhóm chỉ có một minimum paper set; các arms còn lại là mở rộng nếu kết quả chính hoặc reviewer question yêu cầu. Adult là positive case, Shoppers là coherent transfer, Default/Magic là boundary cases.

| Nhóm | Câu hỏi chính | Minimum paper set | Dataset | Cần regenerate? |
|---:|---|---|---|---|
| 1 | Thành phần nào tạo gain end-to-end? | `B/G/R/old/final` + ba intermediate selectors | Adult, Shoppers | Chỉ `B/G`; selector reuse pool |
| 2 | Graph source/authority nào cần thiết? | no graph, statistical, verified typed, mixed full | Adult, News | Có |
| 3 | Dynamic routing có cần row và time conditioning? | global, row-only, time-only, row+time | Adult, News | Có |
| 4 | Guidance nên tác động lúc nào? | early, middle, late, continuous | Adult, News | Có |
| 5 | Action/trust nào tạo gain? | projection, transport, categorical, full; trust on/off | Adult, News | Có |
| 6 | C3 và graph selection khác nhau thế nào? | random, C3, scalar graph, redraw, proximal | Adult, Shoppers | Không |
| 7 | Proximal/deficit gate nhạy với gì? | swap, required gain, quality budget, activation threshold | Adult, Default | Không |
| 8 | Pool budget và sample scale ảnh hưởng ra sao? | `1×/2×/4×` pool, two output sizes | Adult, Shoppers | Một phần |
| 9 | Coverage, robustness và subgroup có được giữ? | no coverage, integrated coverage, final; factor/noisy-edge stress | Adult, Default | Không/phần lớn reuse |
| 10 | Chi phí, privacy và reproducibility có chấp nhận được? | runtime/memory/action count/DCR/duplicate/replay | Tất cả main datasets | Không thêm sweep lớn |

Ưu tiên thực tế: chạy Nhóm `1–7` cho main paper; Nhóm `8–10` có thể dùng subset dataset và appendix. Một arm chỉ thay selector phải reuse exact pool để tách selection variance khỏi generation variance.

## 10.1 Core causal chain — P0 bắt buộc

**Minimum paper set:** `B`, `G`, `R`, `old C3`, scalar graph, constrained redraw, proximal graph-first, proximal quality-first và deficit-gated final. Chạy Adult + Shoppers; các dataset khác chỉ báo kết quả đã có.

| Arm | Câu hỏi |
|---|---|
| `B` | Backbone có chất lượng bao nhiêu? |
| `G-static/global` | Graph/action cố định có đủ không? |
| `G-dynamic` | Row/time routing đóng góp gì? |
| `G + random R` | Gain có chỉ do oversampling/pool choice? |
| `G + old C3` | Generic set selection đóng góp gì? |
| `G + scalar graph` | Additive graph penalty có phá quality không? |
| `G + constrained redraw` | Feasibility constraints nhưng không proximal thì sao? |
| `G + proximal graph-first` | Tối đa graph correction có trade-off gì? |
| `G + proximal quality-first` | Lexicographic ordering đóng góp gì? |
| `G + deficit-gated final` | Conditional no-op có tăng robustness không? |

## 10.2 Graph construction — P0/P1

**Minimum paper set:** no graph, train-statistical graph, verified typed graph, và mixed full graph. LLM/no-LLM chỉ cần một focused discovery ablation, không cần chạy toàn bộ generation grid.

- no graph;
- typed semantic only;
- statistical distribution only;
- conditional numerical only;
- categorical marginal only;
- categorical target only;
- pairwise only;
- mixed full graph;
- `actuate` factors only;
- `actuate + score_only`;
- verified graph vs unverified proposals;
- confidence-only weights vs support/stability-aware weights;
- no LLM vs LLM proposals + verifier vs combined retrieval;
- remove target-connected factors để test leakage sensitivity;
- low-support factors removed vs retained.

## 10.3 Dynamic activation — P0

**Minimum paper set:** global, row-conditioned, time-conditioned, row+time dynamic. Thêm uncertainty/abstention on/off như một sub-ablation của final dynamic arm.

- global factor for every row;
- row-conditioned only;
- time-conditioned only;
- row + time dynamic;
- hard membership/binning;
- posterior-smoothed membership;
- no uncertainty gate;
- uncertainty threshold sweep;
- no reliability gate;
- no persistence/hysteresis;
- one-step persistence vs two-step persistence;
- fixed interval vs edge-specific readable interval;
- always-active vs abstention.

## 10.4 Timing and schedule — P0/P1

**Minimum paper set:** early one-shot, middle one-shot, late one-shot và continuous. Two-pulse chỉ giữ như ablation phụ vì đã có evidence cũ.

- early, middle, late;
- narrow middle vs broad middle;
- one-shot;
- two weak pulses;
- four pulses;
- continuous guidance;
- flat vs middle-peak schedule;
- minimum call gap;
- route recomputed mỗi pulse vs frozen routing;
- normalized noise-time coordinate vs raw step index.

## 10.5 Action family, strength và trust geometry — P0

**Minimum paper set:** projection-only, numerical-transport-only, categorical-message-only và full typed actions; với full action chạy trust on/off. Strength chỉ dùng ba mức `weak/locked/strong`.

- projection only;
- numerical transport only;
- categorical message only;
- projection + numerical;
- projection + categorical;
- numerical + categorical;
- all actions;
- global vs target-conditional transport;
- marginal vs marginal-target vs marginal-pairwise categorical messages;
- action order: before vs after clean-estimate correction;
- second-order sampler correction on/off.

### Arms mở rộng cho strength/trust

- strength sweep yếu/vừa/mạnh;
- no trust region;
- spherical vs per-feature standardized vs ellipsoidal trust;
- trust radius sweep;
- rank-stability threshold;
- projection persistence steps;
- clipped vs unclipped action;
- protected columns locked/unlocked diagnostic;
- target column condition-only vs modifiable.

## 10.6 C3 và output-selection semantics — P0

**Minimum paper set:** no selection, uniform subset, frozen C3, additive scalar graph, constrained full redraw và proximal final. Đây là ablation trực tiếp nhất cho contribution set-level.

- no selection;
- uniform subset;
- marginal raking;
- marginal + pair raking;
- consensus C3;
- C3 + geometric coverage;
- C3 + integrated coverage;
- graph-only selection;
- additive graph + C3;
- constrained full redraw;
- proximal repair;
- graph-first vs quality-first;
- shared active factors vs unrelated MI/correlation pairs;
- hard verified factors as admissibility vs soft penalty;
- all graph factors vs actuated factors only.

## 10.7 Proximal repair và deficit gate — P0/P1

**Minimum paper set:** swap `0/0.25/0.5/1%`; graph gain `0/5/10%`; quality tolerance `0/1.5/3%`; activation threshold `0/1e-3/2e-3`. Dùng sequential screen, không chạy full Cartesian grid.

- swap fractions: `0`, `0.125%`, `0.25%`, `0.5%`, `1%`, `2%`;
- required graph gains: `0`, `2.5%`, `5%`, `10%`, `25%`;
- quality tolerance: `0`, `0.5%`, `1.5%`, `3%`, `6%`;
- activation threshold: `0`, `5e-4`, `1e-3`, `2e-3`;
- candidate draws: `32`, `64`, `128`, `256`, `512`;
- swap temperature;
- uniform proposal mix;
- coverage boost/hard boost;
- factor weight clipping;
- deterministic steepest swap vs stochastic proposal;
- one repair round vs iterative repair;
- fixed swap budget vs deficit-proportional budget.

Không chạy toàn bộ grid. Dùng sequential design: screen 1 seed/train proxy, chọn trước 3–4 operating points, rồi confirm fresh seeds.

## 10.8 Pool and sampling budget — P1

**Minimum paper set:** pool ratio `1×/2×/4×` và output size `4k/8k`. Sampler-step sweep chỉ cần nếu reviewer yêu cầu hoặc runtime table cho thấy bottleneck ở generation.

- pool ratio `1×`, `1.5×`, `2×`, `4×`;
- output sizes `2k`, `4k`, `8k`;
- sampler steps `25`, `50`, `100`;
- deterministic vs stochastic sampler;
- same base-noise paired runs;
- GPU-class sensitivity;
- repeated selection trên cùng pool để tách generation variance và selection variance.

## 10.9 Coverage, subgroup và robustness — P0/P1

**Minimum paper set:** no coverage barrier, integrated train-coverage, và final factor-aware coverage; báo rare/target-conditioned subgroup recall. Thêm leave-one-factor-out và noisy-edge stress trên Adult.

- no coverage constraint;
- k-NN geometry proxy;
- one-fold vs two-fold integrated coverage;
- factor applicability support gap;
- rare subgroup recall;
- target-conditioned subgroup coverage;
- pair-bin coverage;
- hard-factor violation excess;
- intersectional subgroup coverage;
- coverage versus swap count curve.

### Statistics và robustness arms trong Nhóm 9

- tối thiểu 5 final seeds cho headline datasets nếu ngân sách cho phép;
- paired bootstrap CI cho metric gains;
- sign test hoặc Wilcoxon paired test;
- report mean, std, median và win count;
- sensitivity to checkpoint seed;
- sensitivity to train/validation split;
- leave-one-factor-out;
- leave-one-action-family-out;
- graph perturbation/noisy-edge stress test;
- support threshold stress test;
- failure rate và no-op rate.

## 10.10 Efficiency, privacy và reproducibility — P0

**Minimum paper set:** wall time, peak memory, guidance calls, active edges, swap count, DCR, duplicate rate và one-command replay từ manifest.

- generation wall time;
- selection wall time;
- peak GPU memory;
- peak host memory;
- number of guidance calls;
- number of active edges/rows/cells;
- number and percentage of membership swaps;
- LLM calls/tokens offline;
- amortized graph compilation cost;
- comparison với online LLM feedback giả định.

### Safety/privacy arms trong Nhóm 10

- DCR;
- duplicate/nearest-neighbor rate;
- membership inference nếu pipeline đã có;
- protected-column movement;
- target leakage audit;
- train-only selection versus deliberately leaked diagnostic upper bound;
- whether selected subset over-represents high-confidence/easy rows;
- subgroup utility/fairness nếu Adult là headline dataset.

---

## 11. Kế hoạch hoàn thiện trong 2 tuần

Mục tiêu là đóng paper, không mở nhánh method mới.

## Tuần 1 — Khóa protocol và main evidence

### Ngày 1: Freeze

- khóa tên method và equations;
- chọn main datasets: Adult, Shoppers, Magic, Default; News tách public-constraint case study;
- khóa train/validation/test policy;
- khóa final hyperparameters không nhìn test;
- lập manifest cho checkpoint, graph, seed và baseline.

**Deliverable:** một `FINAL_PROTOCOL.md` và config manifest duy nhất.

### Ngày 2–4: Final main runs

- chạy `B/G/R/old/final` cùng task cho mỗi seed;
- ưu tiên 5 seeds Adult và Shoppers;
- Magic/Default dùng 3–5 seeds để xác nhận boundary;
- chạy published local baselines còn thiếu trên đúng split/evaluator;
- không tune sau khi xem final test.

**Estimated compute:** khoảng 20–24 generation tasks, tối đa 12 GPU song song, hai waves chính.

### Ngày 5: Statistics và main tables

- paired bootstrap CI;
- win counts;
- paired tests;
- mean ± std;
- effect sizes;
- local SOTA record audit.

### Ngày 6–7: Core ablation

Chạy core causal chain trên Adult và Shoppers:

`B → G → old → scalar → redraw → proximal graph-first → quality-first → deficit-gated final`.

Tái sử dụng pools khi ablation chỉ thay selector. Không regenerate diffusion nếu không cần.

## Tuần 2 — Mechanism, efficiency và writing

### Ngày 8–9: Dynamic graph ablation

- global vs row/time dynamic;
- conditional-only vs global-only;
- early/middle/late;
- one-shot/two-pulse/continuous;
- action-family ablation.

Chỉ chạy trên Adult + News hoặc Adult + Shoppers để kiểm soát chi phí.

### Ngày 10: Selection mechanism analysis

- activation threshold;
- swap fraction;
- required graph gain;
- quality tolerance;
- pool ratio;
- candidate draws;
- factor-shared vs generic pair objective.

Phần lớn có thể reuse existing pools và chạy nhanh.

### Ngày 11: Coverage, robustness và failure analysis

- subgroup coverage;
- per-factor audits;
- no-op/repair rate;
- graph deficit versus external gain correlation;
- case studies của Adult positive, Default negative và Magic boundary.

### Ngày 12: Efficiency và reproducibility

- runtime/memory/action counts;
- graph compile cost;
- clean reproduction từ manifest;
- package scripts/configs;
- verify every table cell traces về output root.

### Ngày 13: Viết Results/Discussion

- main tables;
- figures;
- limitations;
- negative results;
- threat to validity;
- local SOTA wording audit.

### Ngày 14: Internal paper audit

- không có test-driven tuning;
- không trộn paper numbers với local ranking;
- mọi baseline label đúng `official`, `adapted`, hoặc `proxy`;
- mọi claim có seed count và protocol;
- rerun một final task từ manifest;
- chốt appendix.

---

## 12. Ưu tiên khi quota Codex tuần chỉ còn 7%

Không mở thêm sweep trong tuần hiện tại. Dùng quota còn lại cho:

1. freeze protocol/config;
2. kiểm tra bảng và provenance;
3. viết Method/Results;
4. chuẩn bị Slurm scripts để submit một lần sau khi quota reset.

Thứ tự compute sau reset:

- P0.1: final Adult/Shoppers 5 seed;
- P0.2: standardized main table trên Magic/Default;
- P0.3: core causal ablation reuse-pool;
- P0.4: runtime + confidence intervals;
- P1: timing/action/coverage ablations;
- P2: fairness, graph perturbation và các sweep rộng.

---

## 13. Paper outline đề xuất

### 1. Introduction

- static constraint enforcement quá coarse;
- feature readability và factor applicability thay đổi theo trajectory;
- trajectory correctness không đảm bảo set-level subgroup preservation;
- đề xuất factor-shared controller ở trajectory và output-set scales.

### 2. Related Work

- mixed-type tabular diffusion;
- constrained diffusion;
- LLM/semantic guidance;
- tabular dependency discovery;
- coreset/subset selection.

### 3. Problem Formulation

- mixed numerical/categorical table;
- factor graph và authority;
- dynamic activation;
- set-level graph consistency;
- train-only selection protocol.

### 4. Method

- graph construction/verification;
- dynamic middle guidance;
- generic C3 anchor;
- graph-proximal candidate construction;
- feasibility and quality-first selection;
- deficit activation.

### 5. Experiments

- datasets/baselines/metrics;
- same-backbone paired protocol;
- final main table;
- public constraints;
- mechanism ablation;
- efficiency.

### 6. Analysis

- factor/subgroup analysis;
- activation over time;
- positive and failure cases;
- relation between graph gain and external metrics.

### 7. Limitations

- cross-dataset non-universality;
- dependence on graph quality;
- best-of-N compute;
- LLM discovery not isolated as main gain;
- local versus paper SOTA.

---

## 14. Draft abstract

Mixed-type tabular diffusion models must preserve heterogeneous numerical and categorical dependencies, yet static constraint enforcement is poorly matched to reverse trajectories in which feature readability and relation applicability change over time. We introduce a factor-shared framework that compiles verified semantic and train-fitted statistical relations into a mixed-type graph, dynamically activates its factors for each row and diffusion state, and routes bounded corrections through typed middle-step actions. To close the semantic gap between trajectory guidance and terminal subset selection, we further propose a graph-proximal output repair that starts from a frozen generic coreset, swaps only a small fraction of memberships, and activates only when train-only factor diagnostics expose a residual graph deficit. The method separates factor feasibility from generic distribution quality through a quality-first lexicographic rule and never uses test metrics during selection. On locally reproduced common protocols, the final method obtains a comprehensive three-seed result on Adult, improving Shape and Trend over the paired dynamic generator by 0.371 and 0.522 while setting four of five local baseline records, and yields a coherent transfer result on Online Shoppers. Extensive ablations show that additive graph penalties and full-subset redraws improve graph statistics but generalize poorly, whereas sparse proximal repair and deficit-aware abstention preserve quality. Results on Magic, Default and Diabetes characterize the limits of universal graph correction and motivate authority- and deficit-aware activation.

---

## 15. Final assessment

### Đã đủ để viết

- method story coherent;
- implementation hoạt động end-to-end;
- Adult comprehensive result;
- Shoppers coherent transfer;
- strong C3 evidence trên Magic/Diabetes/Shoppers;
- dynamic trajectory evidence trên News;
- nhiều negative results có giá trị;
- provenance và job tracking tốt.

### Chưa đủ để submit paper ngay

- final standardized multi-seed table chưa hoàn toàn khóa;
- runtime/memory chưa vào main evidence;
- confidence intervals chưa có cho final graph method;
- local baselines chưa đồng đều số seed;
- graph construction/activation ablation chưa hoàn tất ở final method;
- related-work novelty/citation audit cuối chưa làm.

### Mức readiness

- Method/implementation: `85–90%`;
- experimental evidence: `65–70%`;
- ablation package: `55–60%`;
- reproducibility/tracking: `75–80%`;
- writing package: `25–30%` trước report này;
- tổng thể: khoảng **65–70% để bắt đầu viết**, khoảng **50–55% để submit ngay**.

Kết luận: khóa method, bắt đầu viết ngay, và dùng hai tuần tới để hoàn thiện final protocol, ablation P0, statistics và reproduction—không tiếp tục tìm thêm một method mới.

---

## 16. Job và artifact chính

- Adult/Default transfer screen: `1114275`, `1114276`, summary `1114277`.
- Deficit-gate smoke/confirmation/summary: `1114358`, `1114359`, `1114360`.
- Quality-first proximal Shoppers/Magic: `1114240`, `1114241`.
- Full Pareto/proximal/quality screens: `1114186/1114187`, `1114201/1114202`, `1114215/1114216`.
- Generic cross-dataset confirmation source: `1111943` and associated summaries.
- Remote experiment ledger: `analysis/llm_feedback_20260922/EXPERIMENT_LEDGER_20261021.md`.
- Final deficit-gate summary: `outputs/graph_deficit_gate_confirmation_1114359_1114360`.

