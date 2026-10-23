# Báo cáo technique: compositional dynamic constraint graph cho tabular diffusion

**Ngày chốt vòng thực nghiệm:** 10-10-2026  
**Backbone chính:** TabDiff, checkpoint Default epoch 8000  
**Dataset headline hiện tại:** Default of Credit Card Clients  
**Trạng thái:** đã đạt tiêu chí dừng về held-out semantic error; đủ để đóng khung method và bắt đầu viết, chưa đủ để claim SOTA.

> **Cập nhật baseline 10-10-2026:** matched three-seed Harpoon-style comparison và
> corrected common metrics được tổng hợp tại
> `COMMON_METRICS_AND_HARPOON_COMPARISON_20261010_VI.md`. Dynamic graph giảm
> held-out semantic error 61.2% so với TabDiff và 33.1% so với train-calibration-selected
> Harpoon proxy, trong khi Shape/Trend/Alpha/Beta/AUC vẫn ở mức TabDiff. Harpoon là
> local port, không phải official reproduction; method chưa đủ bằng chứng để claim
> general tabular SOTA.

---

## 1. Tóm tắt điều hành

Hướng nghiên cứu hiện tại không còn là gọi LLM để chấm từng bản ghi ở nhiều diffusion
steps. Method được thu hẹp thành:

> Xây một typed constraint graph đã được kiểm chứng; biên dịch các cạnh có thể nối
> tiếp thành compositional DAG; trong reverse diffusion, đọc predicted-clean state,
> cập nhật active graph theo từng row và timestep, thực thi typed action, rồi refresh
> graph để action ở cạnh trước có thể kích hoạt cạnh sau.

Natural motif chính trên Default là:

```text
BILL_AMT3 == 0
    -> PAY_4 == -2
    -> PAY_5 == -2
    -> PAY_6 == -2
```

Ba cạnh đều được mine từ train và kiểm tra trên các discovery, counterexample và
calibration partitions. Chúng là quan hệ xác suất, không phải hard invariants; method
do đó dùng uncertainty-aware calibration thay vì ép violation về 0.

Kết quả trung bình ba paired sampling seeds, 4,096 rows/seed, 50 reverse steps:

| Method | Macro violation ↓ | Held-out error ↓ | Calibration error ↓ | Actions |
| --- | ---: | ---: | ---: | ---: |
| **Dynamic graph** | **0.1266 ± 0.0013** | **0.0120 ± 0.0020** | **0.0091 ± 0.0013** | 85.3 |
| Frozen graph | 0.1300 ± 0.0018 | 0.0154 ± 0.0028 | 0.0125 ± 0.0018 | 64.0 |
| Late-only | 0.1412 ± 0.0057 | 0.0245 ± 0.0048 | 0.0237 ± 0.0057 | 29.7 |
| Final-only calibrated repair | 0.1304 ± 0.0003 | 0.0161 ± 0.0002 | 0.0129 ± 0.0003 | 62.7 |

Dynamic graph giảm held-out error:

- **21.8%** so với frozen graph;
- **50.6%** so với late-only;
- **25.5%** so với final-only repair;
- thắng cả ba baselines trên held-out error ở **3/3 seeds**.

Paired seed-bootstrap 95% intervals của absolute held-out reduction đều dương:

- frozen: `0.0034 [0.0025, 0.0048]`;
- late-only: `0.0125 [0.0095, 0.0166]`;
- final-only: `0.0041 [0.0029, 0.0063]`.

Khi giới hạn dynamic dùng đúng số actions của frozen ở từng seed, held-out gain vẫn
đạt **24.4%**, thắng 3/3 seeds. Nhưng macro violation và train-calibration error của
action-matched dynamic kém frozen. Vì vậy kết luận đúng là dynamic topology cải thiện
việc chọn row/edge có lợi cho held-out semantic distribution; full calibration gain
còn phụ thuộc vào các downstream actions bổ sung mà refresh mở ra.

Common fidelity gần như không đổi so với frozen. So với final-only, dynamic có shape
error tăng 0.0035 point, trend error tăng 0.0206 point và downstream AUC giảm 0.0050.
Đây là trade-off nhỏ nhưng phải được báo cáo, không được mô tả là dominance tuyệt đối.

---

## 2. Bài toán nghiên cứu hiện tại

### 2.1. Input và output

Input gồm:

- schema và kiểu dữ liệu;
- metadata hoặc data dictionary;
- aggregate evidence trên train;
- optional external knowledge hoặc natural-language request;
- pretrained mixed-type diffusion model;
- typed source graph `G_source`.

Output là synthetic table cùng trajectory audit:

- active row-edge set theo timestep;
- topology additions/removals;
- action được chọn và columns bị sửa;
- semantic validity và exception calibration;
- marginal/joint fidelity, utility và action cost.

### 2.2. Câu hỏi trung tâm

Một static constraint list chỉ biết kiểm tra mọi edge độc lập. Với compositional graph,
action của edge trước có thể đổi trạng thái dùng để xác định applicability của edge
sau. Câu hỏi trung tâm là:

> Dynamic refresh trong reverse trajectory có giúp truyền constraint qua nhiều cạnh,
> tốt hơn frozen topology, một late intervention và post-generation repair, trong khi
> giữ held-out semantic distribution và general table quality hay không?

### 2.3. Điều không còn được dùng làm claim

- “LLM feedback trong diffusion” nói chung;
- “constraint được áp ở mỗi reverse step”;
- “graph thay đổi theo row” nếu chỉ vì score/gradient phụ thuộc current row;
- “LLM tạo executable constraint” như contribution duy nhất;
- “middle injection luôn tốt hơn late/final repair”;
- “zero violation là tốt nhất” đối với probabilistic relations.

---

## 3. Motivation

### 3.1. Flat constraints không đủ tạo nhu cầu cho middle injection

Trên các unary hoặc flat conjunctive requests, late-only gần repair thường nằm trên
Pareto frontier. Trong scope-preserving conditional benchmark:

| Request | Middle + sparse late | Late-only | Kết luận |
| --- | ---: | ---: | --- |
| Range violation | 0.63% | **0.48%** | Late-only tốt hơn |
| Category violation | **0.64%** | 0.75% | Middle gain nhỏ, collateral lớn hơn |
| Both violation | **0.73%** | 1.07% | Middle gain nhỏ, collateral lớn hơn |

Vì vậy “đọc state ở middle rồi sửa một rule” không đủ motivation. Middle steps chỉ
có vai trò riêng khi còn thời gian để action thay đổi graph và kích hoạt action kế
tiếp.

### 3.2. Controlled chain xác nhận mechanism

Controlled graph:

```text
MARRIAGE == 1 -> AGE >= 50
MARRIAGE == 1 AND AGE >= 50 -> EDUCATION == 3
```

Edge thứ hai chưa active trước khi edge thứ nhất sửa AGE. Kết quả 4,096 samples:

| Variant | Chain violation ↓ | Education topology changes |
| --- | ---: | ---: |
| Dynamic | **1.63%** | 1,924 |
| Frozen | 74.21% | 0 |
| Late-only | 70.80% | 0 |

Đây là mechanism proof, không phải semantic benchmark: graph được xây có chủ đích.

### 3.3. Natural repayment chain cho tín hiệu ban đầu nhưng gain còn nhỏ

Chuỗi `PAY_0 -> PAY_2 -> ... -> PAY_6` lấy hoàn toàn từ verified Default graph.
Dynamic + Wilson đạt held-out error 1.21%, frozen 1.39%, late-only 1.32% và final-only
1.16%. Kết quả chứng minh transfer nhưng chưa hơn final repair.

Natural motif mới được chọn để có một root context rõ và hai downstream transitions.
Nó tạo propagation opportunities nhiều hơn chuỗi PAY thuần túy, nhưng vẫn gồm các
cạnh đã được train-partition audit chứ không phải controlled rules.

---

## 4. Vị trí so với related work

| Work | Phần đã được bao phủ | Phần khác của method hiện tại |
| --- | --- | --- |
| TabDiff, ICLR 2025 | Mixed numeric/categorical diffusion, feature-wise schedules | Dùng làm frozen backbone; không có semantic graph |
| TabSyn, ICLR 2024 | Continuous latent diffusion cho mixed-type table | Không có trajectory-dependent typed active graph |
| Harpoon, ICLR 2026 | Differentiable constraint guidance trong sampling | Constraint request cố định; current gradient có thể đổi nhưng không phải action-induced topology propagation |
| CTDF, ICAIF 2025 | Training-free feasibility operator ở mỗi reverse step | Đã phủ per-step hard enforcement; không xử lý probabilistic edge lifecycle và compositional active-set refresh |
| SCTab-Diff, preprint 2026 | LLM prior condition denoising | Prior được xác lập trước sampling; không refresh graph vì action trước kích hoạt edge sau |
| LCATS, KDD 2026 | LLM discover/verify constraints, diagnosis và post-repair | Baseline gần nhất cho offline graph + repair; không có middle trajectory propagation |
| Zhao et al., preprint 2026 | LLM agents tạo executable rules, counterexample revision, deterministic repair | Compiler/repair không đủ mới; method hiện tại đặt novelty ở online topology refresh |
| SAGE, ACL 2026 | Value-dependent dependency khi LLM là generator | Row/value graph gần gũi; khác ở diffusion trajectory và intervention-induced edge activation |
| PromptLoop, CVPR 2026 | Foundation-model feedback trên denoised image preview | Chứng minh loop pattern ngoài tabular; không có typed mixed-column constraint DAG |

Novelty boundary defendable nhất:

> **Intervention-induced, trajectory-dependent constraint propagation trong
> mixed-type tabular diffusion**, với typed scope-preserving actions và calibrated
> exception mass.

---

## 5. Tổng quan technique

```text
Schema + metadata + train aggregates + optional external knowledge
                         |
                         v
           LLM/data proposal and typed compilation
                         |
                         v
 discovery -> counterexample -> blind calibration verification
                         |
                         v
              verified source constraint graph
                         |
              exact literal unification
                         v
              acyclic compositional motifs
                         |
        +----------------+----------------+
        |                                 |
 predicted-clean row x_hat_0(t)      readability/timing
        |                                 |
        +---------------> active row-edge set
                                    |
                         calibrated action quota
                                    |
                      typed scope-preserving action
                                    |
                      inject into reverse transition
                                    |
                         refresh graph at t-1
```

Technique có hai phần tách biệt:

1. **Knowledge acquisition:** tạo và kiểm chứng source graph.
2. **Trajectory controller:** tìm active subgraph, chọn action, inject và refresh.

Headline result hiện chứng minh phần thứ hai. LLM không được gọi trong sampling loop.

---

## 6. Source graph acquisition

### 6.1. Ba nguồn tri thức

- empirical candidates mine trên discovery split;
- LLM external/domain proposals từ schema, glossary và aggregate hints;
- predefined/expert rules có provenance rõ.

LLM không nhận raw real rows. Nó chỉ nhận schema và aggregate statistics.

### 6.2. Propose–execute–refine protocol

Default high-quality graph run dùng GPT-5.4 cho proposal, external, critic và judge:

- 20 API calls;
- 177,176 tokens;
- 28 LLM-protocol relations được giữ;
- unified graph có 53 edges và 29 endpoint patterns;
- 98.1% edges có held-out shift không quá 0.05.

Protocol:

1. proposal roles tạo typed edges;
2. compiler type-check columns, operators và sensitive/target leakage;
3. executor đo discovery support/violation;
4. counterexample partition cung cấp aggregate failure evidence;
5. critic giữ, sửa hoặc loại edge;
6. semantic judge kiểm tra directionality, hardness và domain scope;
7. blind calibration quyết định classification;
8. graph đóng băng trước held-out audit.

Stronger LLM không tự động tạo generation gain: Default GPT-5.4 graph có semantic
coverage cao nhưng trong một sampler batch chỉ tạo hai actions. Điều này bác bỏ giả
thuyết “tăng model/API cost sẽ tự giải quyết online actionability”.

### 6.3. Typed edge DSL

Mỗi edge lưu:

- `applicability`: antecedent predicates;
- `requirements`: consequent predicates;
- `exceptions`: explicit valid exception groups;
- `classification`: hard, near-hard hoặc soft;
- `calibration`: support, violation count/rate;
- `source`, provenance và semantic rationale;
- `usage_role`: actuate, score-only hoặc monitor.

### 6.4. Vai trò thực tế của LLM trong phiên bản hiện tại

LLM là offline knowledge proposer/critic/compiler assistant. Online loop dùng local
deterministic executor. Lựa chọn này được support bởi request-conditioned probe:

| Path | Accuracy | Request-swap accuracy | Tokens |
| --- | ---: | ---: | ---: |
| GPT-4.1-mini direct judgement | 0.864 | 0.867 | 37,544 |
| GPT-5.4-mini direct judgement | 0.650 | 0.367 | 37,699 |
| GPT-4.1-mini NL→DSL→executor | **1.000** | **1.000** | 1,846 |
| GPT-5.4-mini NL→DSL→executor | **1.000** | **1.000** | 1,853 |

Trong controlled grammar, compile-once chính xác và rẻ hơn direct repeated judgement.
Điều này không chứng minh cross-domain language generalization, nhưng đủ để loại LLM
per-step judge khỏi main implementation.

---

## 7. Compositional motif compiler

### 7.1. Exact unification

Với hai edges:

```text
e_u: A_u(x) -> B_u(x)
e_v: A_v(x) -> B_v(x)
```

nếu normalized literal của `B_u` trùng `A_v`, compiler tạo structural dependency
`e_u -> e_v`. Chỉ literal equality được dùng ở vòng hiện tại; không suy luận xấp xỉ
hoặc ontology để tránh tạo chain không audit được.

### 7.2. Train-only eligibility

Edge được phép vào motif nếu:

- có đúng một applicability và một requirement literal;
- không có exception trong vòng headline để tránh confound;
- support ít nhất 100 trên từng train partition;
- train violation không quá 25%;
- max–min violation shift trên discovery/counterexample/calibration không quá 5 points;
- source là empirical mining trong headline graph.

Descendants được xếp theo train-only score kết hợp support, lift, violation và
cross-partition stability. Compiler kiểm tra column-level cycle và topologically sort
graph trước khi chạy final-repair baseline.

### 7.3. Headline motif

| Edge | Discovery | Counterexample | Calibration | Held-out audit |
| --- | ---: | ---: | ---: | ---: |
| `BILL_AMT3=0 -> PAY_4=-2` | 18.12% | 17.86% | 17.60% | 17.06% |
| `PAY_4=-2 -> PAY_5=-2` | 10.90% | 8.64% | 9.35% | 9.22% |
| `PAY_5=-2 -> PAY_6=-2` | 10.06% | 9.89% | 8.30% | 10.63% |

Các số là violation rates trong applicable rows. Edge cuối có calibration uncertainty
rõ hơn, tạo motivation cho confidence-bound calibration.

### 7.4. Selection caveat

Compiler không dùng held-out statistics để chọn descendants. Tuy nhiên root `emp_024`
được chọn trong exploratory phase sau khi repository đã có held-out audit artifacts.
Ba confirmation seeds đóng băng motif và policy trước khi sampling, nhưng vẫn dùng cùng
một held-out table. Vì vậy result hiện là strong exploratory evidence, chưa phải một
preregistered independent-test result. Paper run cần chọn root tự động bằng train-only
propagation opportunity trên dataset/checkpoint mới.

---

## 8. Dynamic active graph theo trajectory

### 8.1. Clean preview

Ở mỗi reverse step, denoiser tạo predicted-clean mixed-type record
`x_hat_0(x_t)`. DSL executor đọc bản ghi này và tính cho mỗi edge:

- applicability;
- satisfaction/violation;
- exception status;
- violation streak;
- topology additions/removals so với step trước.

### 8.2. Active window và stability gate

Headline configuration:

- 50 reverse steps;
- active window `t in [0.10, 0.50]`;
- `annealed8` streak gate;
- cooldown 3 steps;
- tối đa 3 actions cho mỗi row-edge.

`annealed8` yêu cầu nhiều quan sát liên tiếp ở đầu active window và giảm dần về một
quan sát khi state đã gần sạch. Nó ngăn early noisy preview có action authority nhưng
vẫn cho phép refresh muộn khi upstream edit vừa kích hoạt edge mới.

### 8.3. Dynamic, frozen và late-only

- **Dynamic:** recompute applicability ở mỗi eligible step.
- **Frozen:** giữ active mask quan sát được ở lần đầu; satisfaction vẫn được đọc lại,
  nhưng row mới đi vào applicability scope không được thêm.
- **Late-only:** chỉ action quanh `t≈0.10`, không còn đủ trajectory để propagate nhiều
  hops.

Ba variants dùng cùng backbone, graph, random seed, sample count, executor và action
candidate family.

---

## 9. Uncertainty-aware calibration

Đối với soft/near-hard edge, mục tiêu không phải zero violation. Với train calibration
rate `p=k/n`, method dùng one-sided Wilson upper bound với `z=1` làm exception target.

Gate chỉ action phần predicted final violation mass vượt allowed exception mass:

```text
allowed_e(t) = q_e * number_of_active_rows_e(t)
action_quota_e(t) = max(0, predicted_final_violations_e(t) - allowed_e(t))
```

Trong headline setting không dùng learned trajectory certifier; predicted final mass
được tính từ current violations sau stability gate. Wilson giảm nguy cơ over-correct
edge có calibration support hữu hạn.

Một `split_median` train-only policy đã được implement để phục vụ vòng sau nhưng chưa
được dùng để chọn headline result. Không có held-out tuning cho Wilson `z=1` trong
confirmation seeds.

---

## 10. Typed action layer

### 10.1. Candidate generation

Với mỗi violated edge, planner có thể tạo:

- direct requirement correction;
- alternate endpoint cho numeric column relation;
- applicability deactivation nếu scope không bị khóa;
- abstain;
- optional train-only conditional prototype.

Headline graph dùng `--lock-applicability-columns`, nên planner không được “thỏa” một
implication bằng cách sửa antecedent thành false.

### 10.2. Vì sao scope lock là bắt buộc

Ở conditional benchmark ban đầu, planner giảm violation bằng cách đổi `MARRIAGE`, làm
53.7–75% requested rows rời khỏi applicability scope. Validity tăng nhưng request bị
né. Sau khi khóa antecedent, scope loss chỉ còn khoảng 1–3% và comparison mới hợp lệ.

### 10.3. Component beam planner

Các violated edges chia thành connected components theo columns. Beam width 8 tìm
joint candidate tối thiểu hóa:

```text
100 * target violation
+ train-kNN density cost
+ 0.5 * normalized edit cost
+ optional graph collateral penalty
```

Headline graph chỉ categorical; direct literal actions đặt `PAY_k=-2`. Planner vẫn
cần xử lý nhiều edges cùng row và không được ghi hai action mâu thuẫn lên một cell.

---

## 11. Injection vào reverse diffusion

Method đọc `x_hat_0` để quyết định nhưng headline online mode là
`xstate_reward_calibrated`:

- numeric action, nếu có, được chuyển thành proximal change ở next reverse state;
- categorical action ghi selected category code sau MDLM reverse update;
- các cells không được action giữ nguyên base sampler transition;
- graph được decode và refresh ở step kế tiếp.

Do headline motif chỉ categorical, kết quả chính chứng minh discrete state injection
và topology propagation. Nó chưa đủ để kết luận cùng mức gain cho numeric relations.

---

## 12. Experimental protocol

### 12.1. Backbone và dữ liệu

- TabDiff checkpoint: `outputs/full_train_default_1033760/ema_model_epoch_8000.pt`;
- dataset: Default;
- 4,096 synthetic rows mỗi seed;
- seeds: `20261376`, `20261377`, `20261378`;
- 50 deterministic reverse steps;
- same-noise paired controls trong từng seed;
- một trained checkpoint, chưa có training-seed replication.

### 12.2. Baselines và ablations

1. unconditional TabDiff;
2. frozen active graph;
3. late-only action;
4. topologically ordered calibrated final repair;
5. action-matched dynamic;
6. dynamic + residual final repair;
7. longer path và multi-entry graph structural probes.

Không gọi các proxy này là faithful Harpoon hoặc CTDF reruns. Harpoon-style energy
proxy đã được thử riêng và dense energy gây collateral lớn; nó không thay thế author
baseline.

### 12.3. Primary metric

Với edge `e`:

```text
heldout_error_e = |synthetic_violation_rate_e - real_heldout_violation_rate_e|
```

Primary result là macro average trên graph. Đây là metric phù hợp hơn raw violation vì
relations là xác suất và real data có valid exceptions.

### 12.4. Secondary metrics

- macro/micro violation;
- train calibration error;
- Numeric KS và categorical TV;
- SDMetrics shape/trend;
- alpha precision và beta recall;
- LogisticDetection;
- downstream MLE XGBoost AUC;
- action events, rows actuated, topology flips và timing.

---

## 13. Main result: ba sampling seeds

| Variant | Macro violation ↓ | Held-out error ↓ | Calibration error ↓ | Numeric KS ↓ | Categorical TV ↓ | Actions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Dynamic** | **0.1266 ± 0.0013** | **0.0120 ± 0.0020** | **0.0091 ± 0.0013** | 0.0304 | 0.0183 | 85.3 |
| Frozen | 0.1300 ± 0.0018 | 0.0154 ± 0.0028 | 0.0125 ± 0.0018 | 0.0304 | 0.0181 | 64.0 |
| Late-only | 0.1412 ± 0.0057 | 0.0245 ± 0.0048 | 0.0237 ± 0.0057 | 0.0304 | **0.0177** | 29.7 |
| Final-only | 0.1304 ± 0.0003 | 0.0161 ± 0.0002 | 0.0129 ± 0.0003 | 0.0304 | 0.0179 | 62.7 |

### 13.1. Relative gains

| Baseline | Macro gain | Held-out gain | Calibration gain | Held-out wins |
| --- | ---: | ---: | ---: | ---: |
| Frozen | 2.6% | **21.8%** | **26.2%** | 3/3 |
| Late-only | **10.3%** | **50.6%** | **60.8%** | 3/3 |
| Final-only | 2.9% | **25.5%** | **29.4%** | 3/3 |

Tiêu chí dừng 10% held-out gain so với frozen và final-only đã đạt. Macro gain chỉ đạt
10% so với late-only, không đạt 10% so với frozen/final. Do đó headline phải ghi rõ
metric, không được nói “method cải thiện 20–25% tổng thể”.

### 13.2. Paired seed-bootstrap

| Baseline | Absolute held-out reduction | 95% seed-bootstrap interval |
| --- | ---: | ---: |
| Frozen | 0.0034 | [0.0025, 0.0048] |
| Late-only | 0.0125 | [0.0095, 0.0166] |
| Final-only | 0.0041 | [0.0029, 0.0063] |

Intervals chỉ bao phủ sampling-seed uncertainty. Chúng không bao phủ graph-selection,
dataset hoặc checkpoint-training uncertainty.

### 13.3. Middle-step attribution

Trên ba dynamic runs:

- 237/256 actions, tương đương **92.6%**, nằm ở `t in (0.25, 0.375]`;
- mean action timestep là khoảng `t=0.336`;
- chỉ 19 actions nằm ở `t in [0.10, 0.25]`;
- trung bình 484 topology flips mỗi seed;
- downstream actions trung bình: dynamic 54.3, frozen 39.3, late-only 8.7.

Kết quả không đến từ một final pulse được đặt tên là middle. Late-only thiếu reverse
steps để propagation, còn frozen có nhiều steps nhưng không nhận row-edge additions.

### 13.4. Dynamic + final certifier

Trên headline seeds, residual final repair sau dynamic không đổi metrics và action
count. Tức dynamic đã đưa output tới calibrated fixed point của cùng final repair
operator; result không dựa vào một post-hoc cleanup ẩn.

---

## 14. Action-matched control

Dynamic được cap đúng action count frozen sử dụng ở từng seed:

| Seed | Actions | Held-out dynamic/frozen | Gain | Macro dynamic/frozen | Calibration dynamic/frozen |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20261376 | 89/89 | 0.0160/0.0181 | 11.9% | 0.1363/0.1319 | 0.0188/0.0144 |
| 20261377 | 53/53 | 0.0097/0.0154 | 36.7% | 0.1326/0.1283 | 0.0151/0.0108 |
| 20261378 | 50/50 | 0.0095/0.0126 | 24.4% | 0.1326/0.1298 | 0.0151/0.0123 |

Mean held-out gain vẫn **24.4%**, thắng 3/3. Categorical TV của matched dynamic cũng
nhẹ hơn frozen. Nhưng macro gain là `-3.0%` và calibration gain `-31.5%`.

Interpretation:

- row/topology selection có giá trị độc lập với số actions đối với held-out metric;
- full method cần thêm downstream actions để đồng thời hạ macro và calibration error;
- không được dùng matched control để claim dynamic uniformly tốt hơn frozen.

---

## 15. Common quality và downstream utility

Mean trên ba seeds:

| Method | Shape ↓ | Trend ↓ | Alpha ↑ | Beta ↑ | Detection ↑ | MLE AUC ↑ | DCR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unconditional | 1.9833 | 2.2165 | 98.4736 | 52.5109 | 0.9555 | 0.7230 | 91.81 |
| **Dynamic** | 1.9591 | 2.2116 | **98.5218** | 52.4436 | 0.9578 | 0.7228 | 91.76 |
| Frozen | 1.9600 | 2.2119 | 98.4990 | 52.4533 | 0.9553 | 0.7238 | 91.76 |
| Late-only | 1.9666 | 2.2081 | 98.4888 | 52.4637 | **0.9579** | 0.7224 | 91.79 |
| Final-only | **1.9556** | **2.1909** | 98.5003 | **52.4761** | 0.9517 | **0.7278** | 91.80 |

Dynamic và frozen gần như giống nhau về general quality:

- shape difference `-0.0009`;
- trend difference `-0.0004`;
- AUC difference `-0.0010`.

So với final-only, dynamic:

- shape xấu hơn 0.0035 point;
- trend xấu hơn 0.0206 point;
- AUC thấp hơn 0.0050;
- alpha cao hơn 0.0215 point;
- detection score cao hơn 0.0061.

Do đó semantic gain không gây collapse, nhưng final-only còn lợi thế nhỏ về một số
fidelity/utility metrics. DCR của mọi method đều khoảng 91.8, xa ideal 50; đây là đặc
tính chung của checkpoint/evaluator và không được diễn giải là privacy improvement.

Quality jobs `1066839` hoàn thành toàn bộ expensive metrics và ghi CSV trước khi phần
Markdown renderer lỗi vì focused graph không có score-only column. Renderer đã được
sửa để chỉ chọn columns tồn tại; scientific metric files không cần chạy lại.

---

## 16. Structural robustness

Exploratory one-seed graph structures:

| Graph | Edges | Dynamic held-out | Frozen held-out | Late held-out | Dynamic gain vs frozen |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stable chain | 3 | **0.0134** | 0.0181 | 0.0299 | **26.3%** |
| Longer path | 5 | **0.0184** | 0.0214 | 0.0251 | 14.2% |
| Multi-entry DAG | 8 | **0.0270** | 0.0290 | 0.0440 | 7.0% |

Dynamic thắng frozen và late trên cả ba graph structures, nhưng relative gain giảm khi
graph rộng hơn và có nhiều calibration shifts. Điều này cho thấy graph quality và
motif coherence vẫn là bottleneck. Multi-entry v2 final-repair order không topological,
nên final-only result của run đó không được dùng. Compiler v3 đã topologically sort
graph trước mọi future final-repair comparison.

---

## 17. Hypothesis ledger cập nhật

| Hypothesis | Kết luận | Bằng chứng |
| --- | --- | --- |
| Direct LLM judgement ở từng step đáng chi phí | **Bác bỏ trong executable setting** | NL→DSL chính xác hơn và khoảng 20× ít tokens |
| Early predicted-clean state đủ tin cậy | **Bác bỏ** | Precision/recall thấp; feedback sớm sai |
| Tăng guidance strength giải quyết residual error | **Bác bỏ** | Scale 2→4 không đổi; hold hàng nghìn lần không giúp |
| Dense differentiable energy là strong solution | **Bác bỏ cho implementation hiện tại** | Scope loss/outside edits rất lớn; categorical middle yếu |
| Scope có thể tự do thay đổi | **Bác bỏ cho user/request semantics** | Planner né antecedent thay vì satisfy request |
| Middle injection tốt cho flat constraint | **Bác bỏ** | Late-only/final repair mạnh hơn hoặc Pareto tốt hơn |
| Dynamic refresh cần cho action-induced topology | **Ủng hộ mạnh** | Controlled chain 1.63% so với 70–74% |
| Mechanism transfer sang natural graph | **Ủng hộ** | Ba graph structures đều thắng frozen/late |
| Natural dynamic graph hơn final repair | **Ủng hộ trên headline motif** | Held-out gain 25.5%, 3/3 seeds |
| Gain chỉ do dynamic dùng nhiều actions | **Bác bỏ cho held-out metric** | Action-matched gain 24.4%, 3/3 |
| Dynamic thắng mọi metric | **Bác bỏ** | Matched macro/calibration kém; final AUC tốt hơn |
| LLM source graph cần thiết cho headline result | **Chưa được chứng minh** | Headline edges đều từ empirical mining |

---

## 18. Claim được phép và không được phép

### 18.1. Claim được phép

1. Exact-unification compiler tìm được natural compositional motifs từ verified graph.
2. Dynamic refresh tạo topology changes và tăng downstream actions trong middle steps.
3. Trên một natural Default motif, dynamic giảm held-out semantic error 21.8% so với
   frozen và 25.5% so với final repair, lặp lại ở ba sampling seeds.
4. Action-matched control giữ 24.4% held-out gain, nên result không chỉ do action count.
5. Common quality gần như không đổi so với frozen; final repair còn một utility gain nhỏ.

### 18.2. Claim chưa được phép

1. SOTA tabular synthesis.
2. Universal improvement trên mọi dataset/graph.
3. LLM per-step feedback tốt hơn deterministic verifier.
4. LLM knowledge tạo headline gain; current winner là empirical graph.
5. Dynamic uniformly hơn final repair trên tất cả semantic/fidelity metrics.
6. General numeric/text constraint propagation.
7. Independent test confirmation; motif root vẫn có exploratory selection exposure.

---

## 19. Đánh giá mức độ sẵn sàng cho paper

### 19.1. Đã đủ để bắt đầu viết

- problem formulation đã rõ;
- failure của flat middle guidance đã được chứng minh;
- mechanism controlled và natural đều có evidence;
- headline metric đạt 20%+ gain với 3/3 sampling seeds;
- frozen, late, final và action-matched controls đã có;
- common quality/utility đã có;
- implementation và outputs đủ tái chạy.

### 19.2. Chưa đủ để chốt ICDE/SOTA

- một dataset headline;
- một checkpoint training seed;
- root motif chịu exploratory selection bias;
- chưa có blind natural compositional benchmark trên dataset thứ hai;
- chưa có faithful same-task CTDF/Harpoon implementation;
- chưa có author LCATS-style graph/repair rerun trên đúng motif DSL;
- no long-text branch.

Đánh giá thực tế: core research project đã đủ hình thành; experimental completeness
cho submission khoảng 65–75%, tùy mức baseline/cross-dataset mà paper yêu cầu.

---

## 20. Minimal next work sau khi mở lại vòng thực nghiệm

Không nên tiếp tục sweep strength. Ba bước có giá trị cao nhất:

1. **Blind motif selection trên dataset/checkpoint mới.** Root ranking phải dùng
   train-only propagation opportunity, freeze trước khi mở test.
2. **Second checkpoint và 3 seeds.** Tách sampling uncertainty khỏi training uncertainty.
3. **Matched external baselines.** Chạy final iterative repair, CTDF/Harpoon-style
   feasibility và LCATS-style post-repair trên cùng graph, sample count và evaluator.

Chỉ sau ba bước này mới quyết định claim SOTA. LLM graph extension nên quay lại sau khi
dynamic mechanism đã được xác nhận cross-dataset; ưu tiên LLM tạo root/context edges
không có trong empirical graph, rồi verify bằng cùng protocol.

---

## 21. Reproducibility và artifacts

### 21.1. Code

- `scripts/build_natural_compositional_dag.py`: exact-unification DAG compiler.
- `scripts/rowwise_reverse_action_guidance.py`: active graph, calibration, planner và injection.
- `scripts/aggregate_natural_dag_confirmation.py`: 3-seed semantic aggregation.
- `scripts/aggregate_natural_dag_action_matched.py`: action-matched control.
- `scripts/aggregate_natural_dag_quality.py`: common quality aggregation.
- `scripts/evaluate_shoppers_baseline_suite.py`: common evaluator; focused-graph renderer đã sửa.

### 21.2. Slurm

- `slurm/465_natural_compositional_dag.sbatch`: structural graph probes.
- `slurm/466_natural_dag_stable_confirm.sbatch`: two confirmation seeds.
- `slurm/467_natural_dag_quality.sbatch`: common fidelity/utility metrics.
- `slurm/468_natural_dag_action_matched.sbatch`: exact action-count control.

### 21.3. Main outputs

- `outputs/natural_dag_stable_v3_20261010/`: frozen headline graph and schedule.
- `outputs/natural_dag_stable_confirmation_v2_20261010/`: 3-seed main result.
- `outputs/natural_dag_action_matched_aggregate_20261010/`: matched-action table.
- `outputs/natural_dag_quality_aggregate_20261010/`: fidelity/utility table.
- `outputs/controlled_chain_eval_1066305_1066308/`: controlled mechanism result.
- `outputs/verified_chain_wilson_dynamic_1066312_0/`: earlier natural chain result.

### 21.4. Jobs

- `1066791`: nine structural controls, completed 9/9.
- `1066822`: six stable-chain confirmation jobs, completed 6/6.
- `1066839`: metrics written 3/3; Slurm marked failed only at optional Markdown rendering.
- `1066848`: three action-matched controls, completed 3/3.

Không còn job nào đang chạy hoặc chờ tại thời điểm chốt báo cáo.

---

## 22. Kết luận

Vòng này đạt đúng loại evidence còn thiếu trước đó. Dynamic graph không được biện minh
bởi flat constraint hoặc “LLM nhìn intermediate sample”, mà bởi **compositional
propagation**: action ở middle trajectory làm một edge mới applicable, active graph
refresh phát hiện nó và còn đủ reverse steps để thực thi downstream action.

Trên natural three-edge Default motif, effect vượt ngưỡng dừng 10–20% ở primary
held-out semantic metric, lặp lại ba seeds và còn tồn tại khi match action count. Kết
quả chưa phải SOTA vì mới có một dataset/checkpoint và motif choice chưa blind hoàn
toàn, nhưng đã đủ mạnh để đóng khung technique, viết method/mechanism sections và
chuyển phần tiếp theo sang external validation thay vì tiếp tục local tuning.
