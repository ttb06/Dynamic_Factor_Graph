# Temporal Dynamic Constraint Graph for Mixed-Type Tabular Diffusion

**Cập nhật:** 15-10-2026  
**Trạng thái:** main mechanism đã có kết quả ba seed trên Default; Shoppers và Diabetes cho calibrated abstention; News đã hoàn tất một seed và đang xác nhận thêm hai seed; Adult và Magic đang train trên Slurm.  
**Backbone hiện tại:** TabDiff, ICLR 2025.  

## 1. Tóm tắt ngắn

Work hiện tại không còn là “gọi LLM ở một vài timestep để sửa row”. Bài toán đã được thu hẹp thành:

> Cho một diffusion generator đã huấn luyện và một graph các quan hệ có thể thực thi, có thể xác định **cạnh nào áp dụng cho row nào, tại timestep nào, với mức bằng chứng nào**, rồi chỉ can thiệp khi middle-state chứa thông tin có ích hơn final repair hay không?

Main method hiện tại gồm bốn quyết định động:

1. **Per-edge time:** mỗi cạnh có readability onset riêng.
2. **Per-row topology:** applicability được tính lại từ `x_hat_0` ở từng reverse step nên cạnh có thể bật hoặc tắt cho từng row.
3. **Per-state evidence:** số preview liên tiếp cần thiết để action giảm dần khi trajectory sạch hơn.
4. **Per-dataset/graph authority:** nếu calibration không tìm thấy middle precursor, method abstain và defer về final certifier.

Kết quả mạnh nhất hiện tại trên Default, ba seed chưa dùng khi calibration:

- actionable held-out violation error giảm `0.006323 -> 0.005097`, tương đương **19.38%**, thắng `3/3` seed;
- full held-out violation error giảm `0.011192 -> 0.010293`, tương đương **8.03%**, thắng `3/3` seed;
- Shape và Trend trung bình tốt nhẹ hơn final repair;
- MLE XGBoost AUC giảm nhẹ `0.00172`, cần report như trade-off.

Shoppers và Diabetes không có positive precursor nên controller abstain; output bằng final repair. Điều này bác bỏ claim “middle guidance luôn hữu ích”, nhưng hỗ trợ một claim chặt hơn: **temporal guidance phải selective và có khả năng từ chối can thiệp**.

## 2. Bài toán và motivation

### 2.1. Khoảng trống giữa distribution fidelity và row validity

Một generator có thể khớp tốt marginal hoặc pairwise distribution nhưng vẫn sinh row không thỏa quan hệ giữa các cột. Final repair có thể đạt validity cao, nhưng có ba hạn chế:

- khi một violation có nhiều cách sửa, boundary projection có thể chọn correction kém tự nhiên;
- sửa ở cuối không cho denoiser cơ hội tái điều chỉnh các cột liên quan;
- một rule chỉ áp dụng trong một trạng thái cụ thể của row; dùng static graph cho mọi row có thể làm hỏng legitimate modes.

Ngược lại, guidance ở mọi timestep cũng không hợp lý: `x_hat_0` ở high-noise state chưa đủ tin cậy, còn guidance quá muộn có thể tương đương post-repair. Kết quả trajectory probe của work xác nhận precision/recall của violation tăng mạnh về cuối và onset khác nhau đáng kể giữa các edge.

### 2.2. Research question hiện tại

> Một temporal active graph, trong đó topology và action evidence thay đổi theo predicted-clean state của từng row, có cải thiện constraint fidelity hơn static-window guidance và final repair, trong khi giữ distribution fidelity, utility và chi phí hợp lý không?

LLM không bắt buộc nằm trong sampling loop. Nó có thể là một nguồn đề xuất knowledge edge ở offline stage, nhưng edge phải được compile thành typed predicates và kiểm chứng bằng train data trước khi có quyền actuation.

## 3. Phạm vi claim

### 3.1. Không claim

- Không claim đưa constraint vào reverse diffusion nói chung là mới.
- Không claim middle guidance luôn tốt hơn final repair.
- Không claim row-conditioned dependency graph là mới; SAGE đã dùng value-aware dependency cho LLM tabular generation.
- Không claim LLM extraction hoặc LLM feedback loop nói chung là mới.
- Không claim zero violations là đủ để chứng minh data quality.

### 3.2. Claim đang được hỗ trợ

1. Mixed-type graph edges trở nên readable ở các timestep khác nhau trong cùng dataset.
2. Applicability graph thay đổi theo cả row và reverse state; additions và removals đều quan sát được.
3. Progressive state evidence tốt hơn fixed persistence trên Default.
4. Đúng row-edge assignment quan trọng: shuffled topology có action và graph flips nhưng không có gain.
5. Calibration-driven abstention tránh làm xấu Shoppers và Diabetes.

### 3.3. Claim chưa đủ bằng chứng

- superiority so với mọi constraint-guided tabular generator;
- cross-backbone generality ngoài TabDiff;
- outer authority selector tổng quát trên nhiều hơn ba dataset;
- lợi ích riêng của LLM so với statistical discovery;
- SOTA trên common fidelity/utility metrics.

## 4. Method tổng quát

### 4.1. Input

Method nhận:

- pretrained mixed-type diffusion model `p_theta`;
- schema và train split;
- candidate graph `G = (V, E)`;
- deterministic verifier và typed action operators cho mỗi edge.

Mỗi edge `e` chứa:

- `A_e(x)`: applicability predicates;
- `R_e(x)`: required relation;
- optional exception predicates;
- hardness và empirical calibration rate;
- provenance và authority role.

Với row `i` ở reverse state `t`, predicted-clean estimate là `x_hat_0^(i,t)`.

### 4.2. Knowledge graph construction

Graph construction là một loop tách biệt với sampling:

1. **Propose:** statistical mining, external published constraints, predefined rules, hoặc LLM/agent đề xuất.
2. **Compile:** chuyển thành typed executable predicates.
3. **Verify:** discovery/counterexample/calibration split chỉ lấy từ train.
4. **Bootstrap:** ước lượng stability, upper violation rate và support.
5. **Select:** giữ edge có support, lift và stability đủ tốt; loại target-related edge trong các benchmark mới.
6. **Audit:** giữ uncertainty và provenance; held-out không được dùng để chọn graph.

Implementation BSCG hiện tại cho phép pairwise edge và conjunctive hyperedge; cycles được giữ vì graph biểu diễn statistical compatibility, không claim causal DAG.

Authority được tách theo provenance:

- edge tự khám phá chỉ được actuation nếu train-calibration violation không quá `5%`;
- external hard specification có `external_authority=true`, nên empirical violation rate được dùng để đo distribution shift chứ không có quyền phủ quyết user request;
- soft/probabilistic edge vượt ngưỡng vẫn có thể làm score/monitor, nhưng không được cưỡng ép. Đây là lý do Magic không tự động biến các association 7–19% exception thành hard constraints.

LLM có thể bổ sung external semantic candidates ở bước 1. Tuy nhiên main trajectory mechanism chỉ nhận typed graph, vì vậy có thể so sánh trực tiếp `LLM graph`, `statistical graph`, `published constraint graph` và `union graph` mà không đổi sampler.

### 4.3. Edge readability calibration

Trên unguided calibration trajectories:

1. lưu `x_hat_0(t)` ở 50 reverse steps;
2. với mỗi edge, so applicability/violation ở step `t` với final sample;
3. tính precision và recall;
4. chọn earliest readable cutoff `tau_e` thỏa precision `>= 0.90` và recall `>= 0.80`.

Một edge không được đánh giá trước `tau_e`. Đây là per-edge generalization của một global guidance interval.

### 4.4. Graph-level temporal authority

Readability không đồng nghĩa với usefulness. Với mỗi edge:

`Delta_e = max_t [N_violation(e,t) - N_violation(e,final)]`

chỉ xét readable states. Graph precursor score là:

`Delta_G = sum_e max(0, Delta_e)`.

- Nếu `Delta_G > delta`, graph nhận online authority.
- Nếu không, sampler abstain và toàn bộ constraint handling được defer sang final certifier.

Đây là outer controller. Nó không thay thế row-level dynamics; nó chỉ quyết định online intervention có đáng chạy trên graph/dataset này hay không.

### 4.5. Dynamic row-edge topology

Tại mỗi reverse step sau `tau_e`:

`a_i,e,t = A_e(x_hat_0^(i,t))`

`v_i,e,t = a_i,e,t AND NOT R_e(x_hat_0^(i,t))`.

Do `x_hat_0` thay đổi, cùng một edge có thể:

- chưa applicable;
- trở thành applicable và satisfied;
- chuyển sang violated;
- deactive sau khi antecedent thay đổi.

Raw topology có thể flip do preview noise. Method giữ EMA volatility riêng cho từng `(row, edge)` và dùng hysteresis:

- stable row-edge có thể commit thay đổi ngay;
- volatile row-edge phải lặp lại cùng topology change ba lần.

Do đó graph state không chỉ là hàm của timestep; nó là hàm `G_i(t) = f(x_hat_0^(i,t), history_i,t)`.

### 4.6. Progressive evidence gate

Mỗi row-edge duy trì violation streak `s_i,e,t`. Required streak được anneal theo reverse progress:

`K(t) = ceil[1 + (K_max - 1) * remaining_noise_fraction(t)]`, với `K_max = 8`.

- ở phần đầu active window, cần tối đa tám preview liên tiếp;
- gần cuối, một preview đáng tin đã đủ.

Action gate mở khi:

- edge đã readable;
- stabilized topology đang active;
- violation streak `>= K(t)`;
- support đủ lớn;
- action budget/cooldown chưa hết;
- calibrated residual mass cho phép action.

Đây là thành phần tạo gain tốt nhất so với fixed `stable2` gate.

### 4.7. Typed action layer

Action layer không để LLM tự sửa raw row. Nó sinh một tập correction có kiểu:

- numeric inequality: project tới feasible target có margin;
- numeric equality: set/copy typed numeric value;
- categorical relation: chọn feasible category từ train-supported values;
- optional reward selection: graph satisfaction + train-kNN density - normalized edit cost.

Appplicability columns có thể được lock để tránh thỏa implication bằng cách phá antecedent. Với overlapping edges, component planner có thể joint-select correction, nhưng main result hiện dùng edge-local planner để giảm engineering confound.

### 4.8. Rejoin reverse trajectory

Correction được áp dụng lên `x_hat_0`; numerical/categorical masks chỉ ra cell nào đã đổi. Corrected clean estimate được ánh xạ trở lại state update hiện tại và tiếp tục cùng reverse trajectory. Action target được giữ trong hai steps nếu edge vẫn applicable, tránh denoiser xóa ngay correction.

Đây là single-pass in-trajectory actuation, không phải generate-final rồi chạy diffusion imputation pass thứ hai.

### 4.9. Final certifier

Sau reverse sampling, cùng typed verifier chạy lần cuối. Chỉ residual violations còn được sửa. Kết quả chính luôn so output sau final certifier với matched final-repair baseline, tránh lợi thế giả do một method có bước hậu xử lý còn method kia không có.

### 4.10. Pseudocode sát implementation

```text
G <- propose_compile_verify_bootstrap_select(train)
tau[e], trajectory_profile[e] <- unguided_probe(checkpoint, G, calibration_seed)
authority <- sum_e positive_middle_violation_surplus(e) > 0

x_T <- sample_prior()
for t = T ... 1:
    xhat0, categorical_logits <- TabDiff.denoise(x_t, t)
    row <- decode(xhat0, categorical_logits)

    for each authorized edge e with t <= tau[e]:
        raw_active[i,e] <- applicability_e(row_i)
        active[i,e] <- rowwise_EMA_hysteresis(raw_active[i,e])
        violated[i,e] <- active[i,e] and not requirement_e(row_i)
        streak[i,e] <- consecutive_violation_count(violated[i,e])
        gate[i,e] <- streak[i,e] >= annealed_threshold(t)
        gate[i,e] <- gate[i,e] and support_budget_cooldown_checks

        for each gated row i:
            candidate <- typed_action(row_i, e)
            candidate <- select_by(constraint_score, train_kNN_density, edit_cost)
            replace only changed cells in xhat0/categorical state
            hold chosen target for at most two steps while e remains active

    x_(t-1) <- same_TabDiff_update(x_t, corrected_xhat0, corrected_categories)

sample <- decode(x_0)
sample <- final_certifier(sample, G)  # residual violations only
```

Hai chi tiết implementation tránh confound quan trọng:

- decode rồi encode lại không phải identity dưới quantile transform, nên sampler giữ nguyên model prediction ở mọi cell không bị action;
- baseline và proposed dùng cùng initial seed, sample count, graph, verifier và final certifier. Khác biệt duy nhất cần đo là các correction đã đi qua reverse trajectory hay chỉ xuất hiện ở endpoint.

## 5. Nguồn ý tưởng và phần khác biệt

| Thành phần | Work gần nhất / ý tưởng mượn | Phần dùng trong work này | Khác biệt của work hiện tại |
| --- | --- | --- | --- |
| Mixed-type backbone | TabDiff, ICLR 2025 | Joint numerical/categorical reverse process và predicted-clean state | Không sửa backbone; thêm typed row-edge controller vào sampler |
| Latent mixed-type alternative | TabSyn, ICLR 2024 | Cho thấy mixed types có thể được đưa về continuous latent | Chưa dùng làm main backbone; là cross-backbone test tương lai |
| Limited interval | Limited Guidance Interval, NeurIPS 2024 | Không guidance ở high-noise và không mặc định guidance toàn trajectory | Học cutoff riêng cho từng edge bằng readability, không dùng một interval chung |
| Progressive strength/evidence | Progressive Guidance, NeurIPS 2023 | Guidance nên tăng độ quyết định khi state sạch hơn | Anneal số violation previews cần thiết, không anneal classifier gradient toàn cục |
| Intermediate feedback | PromptLoop, CVPR 2026 | Quan sát intermediate predicted-clean state rồi cập nhật phần sampling còn lại | Feedback là typed graph state per row-edge, không phải MLLM prompt refinement |
| Active external guidance | DynaGuide, NeurIPS 2025 | External signal can thiệp trực tiếp trong diffusion denoising | External model được thay bằng executable graph + empirical trajectory calibration |
| Dynamic barrier constraints | CoDiG, CoRL 2025 | Constraint-aware update trong dynamic environment | Constraint applicability tự thay đổi theo tabular row state; mixed categorical/numeric action |
| Constrained optimization | Constrained Diffusers, NeurIPS 2025 | Projection/primal-dual/augmented-Lagrangian là action families tự nhiên | Row-edge primal-dual đã thử và bị bác bỏ; giữ selective typed action |
| Discrete constraint layer | CDD, NeurIPS 2025 | Training-free constraint enforcement bên trong discrete denoising | Kết hợp categorical action với continuous numerical correction và dynamic topology |
| Manifold guidance | Harpoon, ICLR 2026 | Differentiable inference-time tabular constraint guidance | Harpoon-style gradient guidance là baseline; main method xử lý executable conditional graph và categorical action |
| Value-aware graph | SAGE, ACL 2026 | Dependency phụ thuộc giá trị thay vì dense/static feature graph | SAGE dùng row/value graph cho LLM token generation; work này thêm timestep/state lifecycle trong diffusion |
| LLM priors in denoising | SCTab-Diff, preprint 2026 | Static semantic prior được đưa vào reverse denoising | Không claim prior conditioning; focus vào graph state, timing, authority và abstention thay đổi theo trajectory |
| LLM discovery + repair | LCATS, KDD 2026; Zhao et al. 2026 | Propose/verify executable constraints và post-generation correction | Đặt compiler/repair làm upstream/baseline; contribution nằm ở temporal active graph inside sampling |
| Published hard constraints | ConstrainedDGM, ICLR 2024 | External News inequalities và final constraint layer | News branch kiểm tra temporal controller trên constraints không do work tự đề xuất |

Primary sources:

- TabDiff: https://proceedings.iclr.cc/paper_files/paper/2025/hash/5c882988ce5fac487974ee4f415b96a9-Abstract-Conference.html
- TabSyn: https://proceedings.iclr.cc/paper_files/paper/2024/hash/e9750610639c3e7a849cff746bf60dbd-Abstract-Conference.html
- Limited Guidance Interval: https://proceedings.neurips.cc/paper_files/paper/2024/hash/dd540e1c8d26687d56d296e64d35949f-Abstract-Conference.html
- Progressive Guidance: https://proceedings.neurips.cc/paper_files/paper/2023/hash/83ca9e252329e7b0704ead93893e6b1b-Abstract-Conference.html
- PromptLoop: https://openaccess.thecvf.com/content/CVPR2026/html/Lee_PromptLoop_Plug-and-Play_Prompt_Refinement_via_Latent_Feedback_for_Diffusion_Model_CVPR_2026_paper.html
- DynaGuide: https://proceedings.neurips.cc/paper_files/paper/2025/hash/3ec6c6fc9065aa57785eb05dffe7c3db-Abstract-Conference.html
- CoDiG: https://proceedings.mlr.press/v305/ma25a.html
- Constrained Diffusers: https://proceedings.neurips.cc/paper_files/paper/2025/hash/31fb284a0aaaad837d2930a610cd5e50-Abstract-Conference.html
- CDD: https://proceedings.neurips.cc/paper_files/paper/2025/hash/124c1cc9f7750f512ed81cdbf2fd5bf5-Abstract-Conference.html
- Harpoon: https://proceedings.iclr.cc/paper_files/paper/2026/hash/e785d64baa2cde06c69d9b7a14636ef2-Abstract-Conference.html
- SAGE: https://aclanthology.org/2026.acl-long.174/
- ConstrainedDGM: https://proceedings.iclr.cc/paper_files/paper/2024/hash/887932131fddf943e8fe3310b62c0147-Abstract-Conference.html

## 6. Implementation hiện tại

### 6.1. Graph discovery

- `scripts/mine_empirical_bank_no_llm.py`
  - 60/20/20 train-only discovery/counterexample/calibration;
  - thêm `--exclude-target` cho các benchmark mới;
  - không đọc test và không gọi API.
- `scripts/discover_bootstrap_conditional_hypergraph.py`
  - 32 bootstrap replicates;
  - lọc support, lift, upper violation rate và instability;
  - deduplicate edges bằng empirical applicability/violation masks;
  - chọn tối đa 16 edges và tối đa 6 conjunctive hyperedges.
- `scripts/build_news_public_typed_graph.py`
  - compile năm News inequalities từ ConstrainedDGM thành typed graph;
  - chỉ dùng train để ghi calibration violation rate.
- `scripts/build_adult_cuts_typed_graph.py`
  - compile CuTS Adult implication 2 và 4 thành bốn edge tương đương;
  - tách disjunctive antecedent thành nhiều edge, giữ conjunctive requirements trong cùng edge;
  - đánh dấu external authority tách biệt với empirical relation authority.

### 6.2. Trajectory calibration

- `scripts/typed_graph_trajectory_probe.py`
  - lưu dense 50-step trajectory;
  - đo applicability/violation precision, recall và flip rate;
  - tính held-out rates trực tiếp từ held-out split;
  - sinh `typed_readability_schedule.csv`, `edge_state_trajectory.csv`, `violation_streak_calibration.csv`.

### 6.3. Sampler/controller

- `scripts/rowwise_reverse_action_guidance.py`
  - `topology_mode=row_ema_hysteresis`;
  - `state_gate=annealed8`;
  - `authority_mode=graph_precursor`;
  - `schedule_mode=edge`;
  - `action_hold_steps=2`;
  - `max_actions_per_row_edge=2`;
  - `gate_ranking=trajectory_posterior`;
  - same-trajectory correction và residual final certifier.

### 6.4. Evaluation

- common fidelity: Shape error, Trend error, alpha precision, beta recall, detection;
- utility: XGBoost AUC cho classification, R2/NRMSE cho News regression;
- privacy diagnostic: DCR train-vs-held-out;
- graph validity: independent/actionable held-out violation error;
- mechanism: actions, residual actions, additions, removals, flips, runtime;
- News external validity: năm public ConstrainedDGM inequalities.

## 7. Kết quả hiện có

### 7.0. Bảng tổng hợp theo dataset

| Dataset | Graph | Middle authority | So với matched final repair | Kết luận hiện tại |
| --- | --- | --- | --- | --- |
| Default | 7 verified BSCG edges | Có | actionable error giảm **19.38%**, full error giảm **8.03%** qua 3/3 seed | Positive mechanism evidence |
| Shoppers | 7 verified BSCG edges | Không, `Delta_G=0` | Bằng final repair ở 3/3 seed | Correct abstention |
| Diabetes | 16 verified BSCG edges | Không, `Delta_G=0` | Bằng final repair; common metrics cũng giống hệt | Correct abstention / negative middle case |
| News | 5 public ConstrainedDGM inequalities | Có, `Delta_G=2` | Cùng 0% violation; Trend và NRMSE tốt hơn, các fidelity metrics khác mixed | External mechanism works, final advantage chưa ổn định |
| Adult | 6 target-free BSCG + 4 public CuTS edges | Chưa có kết quả | Đang chạy | Cross-dataset và external-logic test |
| Magic | 5 target-free BSCG edges | Expected abstention do safety filter | Đang chạy | Probabilistic-graph safety test |

`Middle authority` ở đây không phải nhãn được gán tay: nó được quyết định từ unguided calibration trajectory trước evaluation seed. “Bằng final repair” nghĩa là controller không action và trả cùng output, không phải hai sample độc lập tình cờ có metric gần nhau.

### 7.1. Default, ba seed xác nhận

| Metric | Final repair | Proposed | Relative change | Wins |
| --- | ---: | ---: | ---: | ---: |
| Shape error ↓ | 2.041838 | **2.038613** | 0.16% tốt hơn | 1/3 |
| Trend error ↓ | 1.971839 | **1.961033** | 0.55% tốt hơn | 2/3 |
| MLE XGBoost AUC ↑ | **0.731765** | 0.730041 | 0.24% kém hơn | 1/3 |
| Actionable held-out error ↓ | 0.006323 | **0.005097** | **19.38% tốt hơn** | **3/3** |
| Full held-out error ↓ | 0.011192 | **0.010293** | **8.03% tốt hơn** | **3/3** |

Seeds `20261447–20261449`; trajectory calibration seed `20261440`; 4,096 rows/seed.

### 7.2. Mechanism ablation trên Default

| Controller | Actionable error ↓ | Full held-out error ↓ | Online actions | Residual actions |
| --- | ---: | ---: | ---: | ---: |
| Final repair | 0.007035 | 0.009833 | 0 | 55 |
| Fixed stable dynamic gate | 0.005898 | 0.009133 | 94 | 0 |
| Event-triggered + repair | 0.006705 | 0.009629 | 30 | 46 |
| **Progressive dynamic gate** | **0.005652** | **0.009027** | 76 | 0 |
| Shuffled row topology + repair | 0.007035 | 0.009833 | 60 | 53 |

Progressive graph có `107` additions, `131` removals và `238` topology flips. Shuffled graph có `298` flips nhưng không có final gain, hỗ trợ causal interpretation rằng correct row-state assignment quan trọng hơn số action.

### 7.3. Shoppers

- seven readable actuator edges có cutoffs từ `0.102` tới `0.429`;
- ungated dynamic topology có `76` additions và `73` removals;
- online dynamic slightly worse final repair trên semantic error;
- graph precursor total bằng `0`, nên final controller abstain;
- proposed bằng final repair ở `3/3` seeds.

### 7.4. Diabetes

- 16 edges có cutoffs từ `0.061` tới `0.449`;
- graph precursor total bằng `0`;
- controller cấp `0` online edges và output bằng final repair;
- internal full held-out error: `0.006999` sau final repair/controller;
- final repair/proposed giảm actionable held-out error từ `0.014510` xuống `0.007059`, nhưng đây hoàn toàn là gain của final certifier;
- proposed và final repair có cùng Shape `1.927791`, Trend `7.840616`, XGBoost AUC `0.643297` và detection score `0.922631`;
- so với unconditional, Shape tốt hơn nhẹ nhưng Trend, utility và detection đều giảm. Vì không có online action, Diabetes là negative control cho claim về middle guidance.

### 7.5. News với constraint công khai

Năm inequality được lấy từ benchmark của ConstrainedDGM và compile thành typed graph; graph này không được tạo từ held-out News rows.

| Edge | Earliest readable `t` | TabDiff violation | Held-out violation |
| --- | ---: | ---: | ---: |
| title tokens <= content tokens | 0.735 | 0.0361 | 0.0260 |
| non-stop unique tokens <= non-stop words | 1.000 | 0.0012 | 0.0000 |
| self hrefs <= hrefs | 0.388 | 0.0620 | 0.0000 |
| self-reference min shares <= max shares | 0.388 | 0.1558 | 0.0010 |
| self-reference average shares <= max shares | 0.265 | 0.1472 | 0.0013 |

Trajectory probe cho thấy violation precision/recall tăng từ `0.410/0.105` ở high-noise states lên `0.936/0.838` trong khoảng `0.25 < t <= 0.50`, rồi gần `1.0/1.0` ở late states. Do các cạnh này luôn applicable, topology không flip; dynamics nằm ở violation state và edge-specific readability, không phải applicability lifecycle.

Graph precursor total bằng `2`, nên controller được phép action:

| Method | Held-out constraint error ↓ | Numeric KS ↓ | Actions | Rows changed |
| --- | ---: | ---: | ---: | ---: |
| TabDiff | 0.076626 | 0.033985 | 0 | 0 |
| Online progressive, trước final certifier | 0.008804 | 0.034210 | 1,834 | 1,075 |
| Final repair | **0.005649** | **0.034243** | 1,141 | 1,060 |
| Online progressive + cùng final certifier | **0.005649** | 0.034300 | 2,128 | 1,075 |

Online progressive guidance loại bỏ phần lớn violation trước endpoint, nhưng sau matched final certifier nó **chưa tốt hơn final repair** trên constraint error hoặc Numeric KS và dùng nhiều actions hơn. Common metrics ở seed `20261447` là:

| Metric | TabDiff | Final repair | Proposed |
| --- | ---: | ---: | ---: |
| Shape score ↑ | **96.6554** | 96.6134 | 96.6029 |
| Trend score ↑ | 89.1527 | 90.2583 | **90.5473** |
| Alpha precision ↑ | **97.1327** | 97.0579 | 97.0845 |
| Beta recall ↑ | 42.8825 | **43.0046** | 42.9020 |
| Detection score ↑ | **0.8882** | 0.8839 | 0.8804 |
| XGBoost regression NRMSE ↓ | 1.0126 | 1.0142 | **1.0077** |
| Public violation rate ↓ | 25.8789% | **0%** | **0%** |

So với final repair, proposed tốt hơn ở Trend và regression NRMSE trong khi cùng đạt zero public violations; đổi lại Shape, beta recall và detection giảm nhẹ. Đây là mixed fidelity frontier, chưa phải Pareto dominance. Hai seed `20261448–20261449` đang được xác nhận bằng Slurm array `1081096` mà không thay policy.

### 7.6. So với two-pass temporal router cũ

Two-pass router trước đó đạt 17–18% actionable gain nhưng cần conditional diffusion proposal passes, khoảng `158 s` trên Default và `50 s` trên Shoppers. Single-pass progressive controller đạt 19.38% mean actionable gain trên Default và chỉ dùng một reverse trajectory; screen runtime khoảng 5–7 giây cho 4,096 rows. Runtime chưa hoàn toàn implementation-matched nhưng single-pass rõ ràng practical hơn.

## 8. Negative results quan trọng

1. **Primal-dual row-edge strength bị bác bỏ.** Gần như không cải thiện Default và làm Trend trên Shoppers xấu rõ.
2. **Middle luôn tốt bị bác bỏ.** Shoppers và Diabetes không có positive precursor.
3. **Per-edge precursor quá bảo thủ.** Chỉ giữ 2/7 Default edges, làm actionable gain giảm còn khoảng 2.6%.
4. **Event-triggered chưa đủ.** Tiết kiệm online actions nhưng cần gần như toàn bộ final repair.
5. **Shuffled topology không tạo gain.** Nhiều flips/actions không đủ nếu edge gán sai row.
6. **Direct repeated LLM judgement không nên là main actuator.** Chi phí cao, output không ổn định và compile-once typed execution mạnh hơn trong focused probe.
7. **Static/frozen graph không giải thích được gain.** Dynamic topology tốt hơn static window trên Default, nhưng không universal trên Shoppers.

## 9. Các job mở rộng đang chạy

| Dataset | Graph source | Train/checkpoint | Pipeline jobs | Trạng thái lúc submit |
| --- | --- | --- | --- | --- |
| Diabetes | target-free BSCG đã có | TabDiff 8,000 epochs đã có | `1080985 -> 1080986 -> 1080987` | complete; controller abstains |
| News | 5 public ICLR-2024 constraints | TabDiff 8,000 epochs đã có | `1081001 -> 1081002 -> 1081003`; confirm `1081096_[0-1]` | one seed complete; two confirmation seeds queued/running |
| Adult | target-free statistical BSCG, 6 edges | train TabDiff 8,000 epochs | graph `1081004`, train `1081005`, probe/controller/eval `1081006–1081008` | graph complete; training running |
| Magic | target-free statistical BSCG, 5 edges | train TabDiff 8,000 epochs | graph `1081009`, train `1081010`, probe/controller/eval `1081011–1081013` | graph complete; training running |
| Adult–CuTS | 4 typed edges từ public implication 2/4 | dùng cùng Adult checkpoint `1081005` | `1081166 -> 1081167 -> 1081168` | submitted with checkpoint dependency |

News và Adult–CuTS là các benchmark quan trọng nhất trong batch này vì constraint suite đến từ accepted papers, không phải do work tự đề xuất. News cho substantial pre-endpoint correction nhưng chưa thắng matched final repair. Adult statistical và Magic kiểm tra cross-dataset graph discovery và authority selection; kết quả của chúng không được dùng để thay đổi policy trước evaluation. Magic hiện có 5 stable statistical edges nhưng tất cả đều vượt safety threshold 5%, nên branch này được dự kiến như một abstention/safety test chứ không phải positive mechanism benchmark.

## 10. Main contribution có thể viết thành paper

### Contribution 1 — Temporal active constraint graph

Biểu diễn constraint state ở độ phân giải `(row, edge, timestep)`, với learned per-edge readability và row-wise topology hysteresis. Đây là phần novelty chính.

### Contribution 2 — Calibration-gated progressive intervention

Tách “có thể đọc edge” khỏi “nên action”. Outer graph precursor cho phép abstention; inner progressive gate điều chỉnh evidence threshold theo reverse progress.

### Contribution 3 — Typed mixed-type actuation inside one trajectory

Numerical/categorical correction có kiểu, budget, cooldown và short hold, rejoin cùng trajectory; không cần retrain backbone hoặc thêm diffusion proposal pass.

### Contribution 4 — Evaluation protocol

So sánh cùng backbone/seed/graph và final certifier; dùng independent held-out graph, shuffled topology, static window, event trigger, final repair và public constraints. Report semantic validity cùng Shape, Trend, utility, coverage/privacy và runtime.

## 11. Vai trò hợp lý của LLM

LLM nên nằm ở **knowledge proposal layer**, không phải repeated row judge mặc định:

1. đọc schema/metadata/external domain description;
2. đề xuất relations, applicability và exceptions;
3. compile thành typed DSL;
4. train-only counterexample verification và bootstrap calibration;
5. chỉ verified edge mới đi vào temporal controller.

Như vậy paper có thể giữ motivation “foundation model bổ sung external semantics” nhưng main technical contribution và result không phụ thuộc API reliability. Ablation cần có:

- statistics-only graph;
- LLM-only verified graph;
- union graph;
- shuffled/unverified graph;
- same temporal controller cho tất cả graph sources.

## 12. Bước tiếp theo có giá trị thông tin cao nhất

1. Xác nhận News Trend/utility gain trên hai seed mới: constraint error đã hòa final repair; nếu fidelity gain không lặp lại, News được report như boundary case.
2. Đọc Adult/Magic authority decision trước khi tune. Ít nhất một dataset nên có online authority và gain; nếu cả hai abstain, outer precursor hiện quá conservative hoặc phenomenon không general.
3. Chạy Harpoon-style differentiable guidance và CDD/constraint-layer proxy trên đúng public News constraints.
4. Cross-backbone trên TabSyn chỉ sau khi cross-dataset result ổn; không nên mở architecture branch trước.
5. Sau khi policy freeze, chạy ba seed cho dataset có positive signal; không multiseed mọi negative branch.

## 13. Kết luận hiện tại

Work đã vượt qua giai đoạn “một ý tưởng LLM feedback ở middle step”. Main mechanism hiện có definition rõ, implementation tái lập được, một positive dataset với 19.38% actionable gain, hai calibrated abstention datasets, và mechanism controls hỗ trợ đúng row/timestep dynamics.

Điểm còn thiếu để đóng khung mạnh cho ICDE là **external/cross-dataset positive evidence**. News đã xác nhận cơ chế có thể hành động trên public constraints nhưng chưa thắng matched final repair; common fidelity/utility vẫn đang được đo. Adult/Magic frozen expansion đang trực tiếp kiểm tra generality. Nếu ít nhất một nhánh mới cho final Pareto gain, project có thể được viết quanh “temporal active constraint graph”. Nếu các dataset mới đều abstain hoặc ngang final repair, contribution nên được viết như selective controller có safety/efficiency benefit, không phải universal quality improvement.
