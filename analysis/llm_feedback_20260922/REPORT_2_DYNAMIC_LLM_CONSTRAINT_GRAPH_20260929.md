# Dynamic semantic constraint graph cho mixed-type tabular diffusion

## In short

Các work hiện đã:

- dùng LLM tạo prior, knowledge graph hoặc executable rules trước sampling;
- dùng static constraint/loss/feasibility operator trong reverse diffusion;
- dùng LLM hoặc verifier đọc intermediate estimate trong image/video diffusion;
- dùng value-dependent dependency graph khi LLM là generator chính;
- repair constraint sau khi tabular generator đã sinh xong.

=> Khoảng trống có thể theo đuổi là một **dynamic semantic constraint graph** cho mixed-type diffusion, trong đó active predicates, edges, applicability, exception và actuation scope thay đổi theo từng synthetic row và reverse step.

Kiến trúc dự kiến:

`schema + metadata + user request → LLM-compiled typed graph → type check/train-data verification → row/timestep active subgraph → selective diffusion guidance → independent verifier → residual repair`.

Điểm cần ghi chính xác: method cuối không nên dùng luật đơn giản `uncertainty cao → active edge`. Khi intermediate state còn quá nhiễu, uncertainty cao có thể phải dẫn tới `defer`. Edge chỉ nên được activate khi relation đủ readable, applicable, có xác suất violation cao và không phải valid exception.

| Thành phần | Mục tiêu | Trạng thái |
| --- | --- | --- |
| LLM compile semantic graph | Tạo typed nodes/edges từ metadata/request | Chưa chạy end-to-end; DSL compiler probe đã chạy |
| Train-data validation | Loại rule sai và tìm counterexamples | Constraint audit đã chạy |
| Dynamic active graph | Chọn graph riêng theo row/timestep | Mechanism probe đã chạy |
| Semantic applicability/violation graph | Phân biệt satisfied/violated/not-applicable/exception | Chưa implement |
| LLM update graph giữa sampling | Giải quyết ambiguity/conflict | Chưa thử |
| Selective actuator | Chỉ guide columns/edges liên quan | Column-level probe đã chạy |
| CTDF/Harpoon/repair baselines | So cùng constraint và budget | Proxy baselines đã chạy; official comparison chưa đủ |

Claim hiện có thể bảo vệ:

> Row- and timestep-dependent graph actuation là khả thi và active topology thực sự thay đổi. Preliminary mixed-type evidence cho thấy direct condition satisfaction và sampling efficiency tốt hơn fixed dependency graph, nhưng numeric fidelity giảm nhẹ. LLM graph constructor/controller chưa được kiểm chứng.

Chưa thể claim “LLM-controlled dynamic graph improves TabDiff”.

## 1. Bài toán

Cho một prior graph `G_prior` mô tả các semantic relations và một intermediate clean estimate `x_hat_0(x_t)`, mục tiêu là xây active graph riêng cho row `i` tại timestep `t`:

`G_(i,t) = Select(G_prior, request_i, x_hat_0(i,t), trajectory_i, uncertainty_i)`.

Khác với static graph:

- node/edge có thể active ở row này nhưng không active ở row khác;
- relation có thể chưa applicable ở early step;
- một constraint có thể đã satisfied nên không cần guidance;
- một subgroup có thể là valid exception;
- nhiều constraints có thể conflict và cần abstain/rollback.

Ví dụ:

```text
Prior graph:
  Never-worked  --usually_implies--> occupation is missing
  student       --supports---------> low working hours
  subgroup_Z    --exception_to-----> low working hours

Intermediate row at t1:
  workclass uncertain, occupation uncertain
  => defer relation

Intermediate row at t2:
  workclass = Never-worked, occupation = Engineer
  => relation applicable and probably violated

Intermediate row at t3:
  subgroup_Z becomes confident
  => exception node suppresses the actuation edge
```

## 2. Motivation

### 2.1. Static scope tạo validity–fidelity trade-off lớn

Ở Default support-mixed q950, constant guidance 4:

| Scope | Direct success | Numeric KS | Categorical TV | Normalized calls/valid |
| --- | ---: | ---: | ---: | ---: |
| All features | 0.5105 | 0.3730 | 0.1984 | 4.43 |
| Fixed target + top-1 dependency | 0.3144 | 0.1610 | 0.0834 | 7.86 |

Ở Shoppers support-mixed q950:

| Scope | Direct success | Numeric KS | Categorical TV | Normalized calls/valid |
| --- | ---: | ---: | ---: | ---: |
| All features | 0.4893 | 0.3205 | 0.3325 | 4.73 |
| Fixed target + top-1 dependency | 0.3111 | 0.2636 | 0.1969 | 8.68 |

=> All-feature guidance đạt success cao nhưng kéo lệch toàn record. Fixed narrow scope giữ fidelity tốt hơn nhưng bỏ nhiều valid trajectories. Dynamic graph có thể chọn context khác nhau theo row/step thay vì dùng một scope cố định.

### 2.2. Real dependencies có directionality và exceptions

Cross-dataset audit đầu tiên có 788 candidate relations:

- 80 empirical zero-exception candidates;
- 95 near-hard dynamic candidates;
- 334 probabilistic relations;
- 132 insufficient-support candidates;
- 85 sensitive/demographic audit-only relations.

Ví dụ:

- Shoppers: `Administrative = 0 → Administrative_Duration = 0` gần exact.
- Chiều ngược `Administrative_Duration = 0 → Administrative = 0` có khoảng 2–3% exceptions.
- Adult: `Own-child → age < 30` có khoảng 19.7% exceptions ở cả train và held-out.

=> Static undirected dependency hoặc hard global rule có thể xóa rare valid subgroups.

### 2.3. Intermediate preview không đáng tin ở mọi timestep

Trajectory probes và các work như PPAD cho thấy preview quá sớm có thể nhiễu. Numeric, categorical và text cũng không hội tụ cùng tốc độ. Vì vậy graph update cần readability threshold và abstention, không nên gọi LLM hoặc actuate ở mọi step.

## 3. Related work và novelty boundary

### 3.1. LLM-TabLogic và SCTab-Diff

LLM-TabLogic dùng LLM khám phá graph/relations trước diffusion và thay đổi representation. SCTab-Diff trích xuất semantic prior rồi condition reverse denoising; score vẫn phụ thuộc current row context.

=> Không thể claim chung rằng “graph phụ thuộc row” hoặc “LLM prior đi vào denoising” là mới. Cần chứng minh active structure/applicability/action thay đổi theo row/timestep.

### 3.2. Harpoon và CTDF

Harpoon có static constraint request nhưng gradient thay đổi theo current state. CTDF đưa feasibility operator vào mỗi reverse step cho mixed numeric/categorical constraints.

=> Dynamic graph chỉ khác nếu nó thực sự thêm/bỏ nodes/edges, đổi applicability, exception hoặc abstain. Nếu chỉ đổi gradient magnitude theo `x_t`, contribution đã bị các work này bao phủ.

### 3.3. LCATS, Zhao et al. và EVS

LCATS khám phá/validate constraints rồi repair sau sinh. Zhao et al. dùng counterexample-guided revision để tạo executable constraints rồi deterministic repair. EVS cũng xem LLM như noisy oracle và chọn verified constraint subset.

=> “LLM compile metadata thành executable rules” không đủ novelty một mình. Graph compiler là infrastructure; contribution phải nằm ở dynamic active graph và verified selective actuation.

### 3.4. SAGE

SAGE chỉ ra dependency có thể thay đổi theo giá trị của bản ghi và dùng sparse value-dependent graph khi LLM sinh table.

=> Đây là related work gần nhất về dynamic dependency. Khoảng khác biệt tiềm năng là generator chính vẫn là diffusion, graph được update trên intermediate denoised estimates, và actuator phải xử lý mixed continuous/categorical states.

### 3.5. PromptLoop, PPAD, Free2Guide và TTGen

Các work image/video đã dùng foundation-model feedback trên denoised preview, prompt update, branch scoring hoặc rollback.

=> Không claim LLM feedback inside diffusion là mới. Contribution phải đến từ adaptation sang typed tabular semantics: asymmetric dependencies, applicability, valid exceptions, rare-mode coverage và feature-specific convergence.

## 4. Proposed method đầy đủ

### 4.1. LLM tạo typed prior graph

LLM đọc:

- schema và column descriptions;
- allowed user request;
- data dictionary/domain documentation;
- aggregate train evidence hoặc sanitized counterexamples.

LLM trả về graph có kiểu:

```json
{
  "nodes": [
    {"id": "p1", "type": "predicate", "expression": "workclass == Never-worked"},
    {"id": "p2", "type": "predicate", "expression": "occupation == missing"},
    {"id": "e1", "type": "exception", "expression": "subgroup == Z"}
  ],
  "edges": [
    {"source": "p1", "target": "p2", "type": "implies", "confidence": 0.94},
    {"source": "e1", "target": "p2", "type": "exception_to", "confidence": 0.81}
  ]
}
```

Typed edges tối thiểu:

- `requires`;
- `implies`;
- `conflicts`;
- `exception_to`;
- `depends_on`;
- `actuates`.

### 4.2. Verification trước sampling

- schema/type check;
- executable tests;
- train-only support và exception rate;
- held-out evaluation chỉ sau khi graph đã đóng băng;
- sensitive/target relations mặc định audit-only;
- counterexample-guided revision;
- hard/soft/probabilistic classification.

### 4.3. Dynamic graph update

Semantic activation score dự kiến:

`a(i,e,t) = readability(i,t) * applicability(i,e,t) * P(violation | trajectory) * (1 - P(exception | trajectory))`.

Actions:

- `continue`;
- `guide(columns, strength)`;
- `defer`;
- `abstain`;
- `rollback/resample`.

LLM không cần chạy mọi step. Routine graph update do deterministic verifier hoặc learned lightweight selector thực hiện; LLM chỉ xử lý ambiguity/conflict/unseen semantics.

### 4.4. Selective actuator

Active edge được chuyển thành:

- feature mask;
- guidance strength;
- differentiable energy nếu relation executable;
- projection/rollback action nếu cần.

Actuator phải report columns bị tác động và không được tự động sửa protected attributes hoặc supervised targets nếu request không yêu cầu.

## 5. Method đã implement: dynamic column-level graph probe

Current probe chưa có LLM graph.

- Prior edges lấy từ normalized mutual information trên training split.
- Condition columns luôn active.
- Candidate columns được score riêng cho từng row và reverse step.
- Numeric response: absolute conditional denoiser delta.
- Categorical response: RMS conditional logit delta.
- Categorical entropy làm uncertainty factor.
- Chọn top-`k` columns rồi chỉ giữ conditional delta trên những columns đó.

Implemented score gần đúng:

`score(i,j,t) = conditional_response(i,j,t) * dependency_prior(j) * categorical_uncertainty(i,j,t)`.

Điểm khác với proposed full method:

- chưa có predicate nodes;
- chưa đánh giá applicable/satisfied/violated/exception;
- uncertainty cao hiện làm categorical edge dễ active hơn;
- numeric uncertainty chưa được model hóa;
- selection mới ở column level, chưa phải typed semantic edge level.

## 6. Experimental setup

- Backbone: existing TabDiff checkpoints.
- Datasets hiện có kết quả: Default; Shoppers đang trong full grid.
- Conditions: numeric và mixed rare-condition catalogs.
- Samples: `2,048`/condition.
- Reverse steps: `30`.
- Guidance profiles: constant, step-release, mid-pulse.
- Active-edge budgets: `k ∈ {1,2,4}`.
- Candidate pools: `{4,8}`.
- Seeds: ba.
- Full submitted grid: `216` runs.
- Fixed comparison: same backbone, adapter, condition, seed, noise, schedule và active-edge budget.

Automatic candidates loại supervised target và protected/sensitive columns, trừ khi chính request yêu cầu chúng.

## 7. Metrics

### 7.1. Direct success

Tỷ lệ trajectories thỏa toàn bộ user condition trước repair hoặc rejection. Cao hơn tốt hơn.

### 7.2. Normalized denoiser calls/valid

Logged metric là `2 / success`, vì mỗi step dùng một conditional và một unconditional pass. Với 30 steps, actual backbone evaluations là `30 × logged value`. Thấp hơn tốt hơn.

### 7.3. Numeric KS

Mean two-sample Kolmogorov-Smirnov statistic trên numeric columns giữa accepted synthetic cohort và held-out real cohort thỏa cùng condition. Miền `[0,1]`; thấp hơn tốt hơn.

### 7.4. Categorical TV

Mean total-variation distance trên categorical columns. Miền `[0,1]`; thấp hơn tốt hơn.

### 7.5. Graph diagnostics

- edge-selection rate;
- edge-change rate giữa consecutive steps;
- selection theo early/middle/late phase;
- fraction defer/abstain/rollback trong full method;
- LLM calls/tokens và false actuation rate.

## 8. Current results

### 8.1. Mixed condition smoke test

Default, mid-pulse guidance, dynamic `2-of-8` so với fixed top-2 graph, một seed:

| Rarity | Fixed success | Dynamic success | Relative success gain | Fixed calls | Dynamic calls | Fixed KS | Dynamic KS | Fixed TV | Dynamic TV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| q750 | 0.3429 | 0.3923 | +14.4% | 6.79 | 5.92 | 0.1042 | 0.1087 | 0.0514 | 0.0495 |
| q900 | 0.2878 | 0.3574 | +24.2% | 8.46 | 6.51 | 0.1309 | 0.1413 | 0.0659 | 0.0701 |
| q950 | 0.2339 | 0.3245 | +38.8% | 10.10 | 6.90 | 0.1456 | 0.1696 | 0.0907 | 0.0862 |
| q975 | 0.1947 | 0.3122 | +60.4% | 11.12 | 6.71 | 0.2176 | 0.2466 | 0.1061 | 0.1086 |
| Macro mean | 0.2648 | 0.3466 | +30.9% | 9.12 | 6.51 | 0.1496 | 0.1665 | 0.0785 | 0.0786 |

Diễn giải:

- direct validity tăng từ `26.48%` lên `34.66%`;
- normalized calls giảm `28.6%`;
- actual expected backbone evaluations giảm từ khoảng `273.5` xuống `195.3` cho một valid row;
- numeric KS xấu hơn `0.0169`, tương đương `11.3%` relative;
- categorical TV gần như không đổi.

=> Tín hiệu validity/cost khá rõ, nhưng chưa Pareto-dominant vì numeric fidelity giảm và mới có một mixed seed.

### 8.2. Numeric conditions

Default, constant guidance; `4-of-8` mới có một seed tại snapshot này:

| Active/Pool | Seeds | Fixed success | Dynamic success | Relative gain | Fixed KS | Dynamic KS | Fixed TV | Dynamic TV |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1/4 | 3 | 0.4642 | 0.4638 | -0.08% | 0.1435 | 0.1391 | 0.0739 | 0.0715 |
| 1/8 | 3 | 0.4642 | 0.4650 | +0.16% | 0.1435 | 0.1427 | 0.0739 | 0.0719 |
| 2/4 | 3 | 0.4782 | 0.4785 | +0.05% | 0.1441 | 0.1399 | 0.0771 | 0.0762 |
| 2/8 | 3 | 0.4782 | 0.4814 | +0.67% | 0.1441 | 0.1430 | 0.0771 | 0.0744 |
| 4/4 | 3 | 0.4899 | 0.4899 | 0.00% | 0.1440 | 0.1440 | 0.0761 | 0.0761 |
| 4/8 | 1 | 0.4903 | 0.5004 | +2.05% | 0.1433 | 0.1415 | 0.0772 | 0.0744 |

`4-of-4` là sanity control: dynamic selector phải chọn toàn bộ candidate pool, nên kết quả bằng fixed graph. Numeric-only gains hiện nhỏ; tín hiệu chính nằm ở mixed conditions.

### 8.3. Graph có thực sự dynamic không?

Mean edge-change rate giữa consecutive reverse steps trong mixed smoke khoảng `0.118`. Edge-selection frequencies cũng thay đổi giữa rows và early/middle/late phases.

=> Cơ chế thực sự đổi active topology, không chỉ đổi tên một static graph.

### 8.4. LLM compiler evidence

| Phương pháp | Four-way accuracy | Same-row/different-request | Token |
| --- | ---: | ---: | ---: |
| GPT-4.1-mini direct judgement | 0.864 | 0.867 | khoảng 37.5k |
| GPT-5.4-mini direct judgement | 0.650 | 0.367 | khoảng 37.7k |
| LLM → typed DSL → executor | 1.000 | 1.000 | khoảng 1.85k |

=> Bằng chứng này ủng hộ compile-once + deterministic execution. Nó phản đối việc gọi LLM trực tiếp ở mọi step cho executable constraints.

## 9. Những hypothesis đã bị bác bỏ hoặc thu hẹp

### 9.1. “LLM judge every step” — không được ủng hộ cho executable rules

Direct judgement vừa đắt vừa không ổn định hơn typed executor. Event-triggered LLM chỉ hợp lý cho ambiguity, exception, external knowledge hoặc text–structured conflict.

### 9.2. “Uncertainty cao nghĩa là edge phải active” — chưa đúng

Uncertainty có thể biểu thị preview chưa readable. Full method cần tách:

- epistemic/preview uncertainty;
- applicability uncertainty;
- violation probability;
- exception probability.

### 9.3. “Dynamic graph đã thắng fixed graph” — chưa đủ bằng chứng

Mixed smoke có success gain lớn nhưng KS xấu hơn. Numeric conditions gần trung tính. Full grid và confidence intervals chưa hoàn tất.

### 9.4. “Dynamic graph đã cần LLM” — chưa chứng minh

Current selector hoàn toàn deterministic. Nếu oracle/deterministic graph đã giải quyết đủ, LLM không cần thiết.

## 10. Khó khăn hiện tại

### 10.1. Novelty boundary rất sát related work

Static graph + state-dependent gradient không đủ mới vì Harpoon/SCTab-Diff/CTDF. LLM rule compiler không đủ mới vì Zhao et al./EVS/LCATS. Contribution phải là dynamic semantic active graph với applicability/exception và verified actuation.

### 10.2. Benchmark semantic còn yếu

Natural hard-rule violation của full checkpoints chỉ khoảng `1.6–1.8%`, dễ saturation. Mined correlations cũng không phải semantic ground truth.

### 10.3. Current graph mới ở column level

Một column có thể tham gia nhiều relations với conditions và exceptions khác nhau. Column mask không biểu diễn được edge semantics hoặc higher-order relation.

### 10.4. Causal interpretation không được phép

NMI/dependency graph chỉ là association graph. Không được diễn giải edge như causal relation hoặc domain invariant.

### 10.5. Rare exception preservation

Hard guidance có thể tăng measured validity bằng cách xóa một subgroup hợp lệ. Evaluation phải đo exception recall và rare-mode coverage, không chỉ violation rate.

### 10.6. Cost

Dynamic guidance đang cần conditional + unconditional pass ở mỗi step. LLM calls phải event-triggered; nếu gọi ở nhiều rows/steps, chi phí sẽ không cạnh tranh với deterministic guidance hoặc repair.

## 11. Hướng tiếp theo

### 11.1. Hoàn tất mechanism grid

Full grid `1048642[0-215]`; aggregate jobs `1048643–1048645`. Quyết định dựa trên:

- paired fixed-vs-dynamic graph;
- seed-cluster confidence intervals;
- Pareto frontier validity–KS–TV–cost;
- graph turnover diagnostics.

### 11.2. Oracle four-state semantic graph

Xây benchmark có bốn trạng thái:

- violated;
- satisfied;
- not applicable;
- valid exception.

Oracle graph quyết định active edge/action. Nếu oracle không thắng fixed projection và repair, không nên đầu tư thêm vào LLM controller.

### 11.3. Edge-level executor

Thay column selector bằng typed edge selector:

- antecedent/applicability test;
- consequent violation score;
- exception suppressor;
- conflict resolution;
- feature-specific actuation strength.

### 11.4. Readability/calibration

- categorical agreement/flip rate theo timestep;
- numeric error/stability theo timestep;
- defer threshold;
- calibrated confidence;
- early/middle/late invocation ablation.

### 11.5. LLM ablation sau oracle

So sánh:

1. static prior graph;
2. deterministic dynamic graph;
3. oracle semantic graph;
4. LLM compile-only graph;
5. fixed-schedule LLM update;
6. event-triggered LLM update;
7. shuffled-request và shuffled-edge controls.

### 11.6. Baselines bắt buộc

- no guidance;
- static condition/prior;
- fixed dependency graph;
- Harpoon-style differentiable guidance;
- CTDF-style per-step feasibility;
- final-only projection;
- LCATS/Zhao-style post-repair;
- best-of-N/verifier selection;
- oracle graph upper bound.

## 12. Đánh giá khả năng thành paper độc lập

Hướng này có thể thành paper độc lập với hướng rare user-condition generation, nhưng chưa ở trạng thái đủ kết quả.

Để độc lập, paper phải tập trung vào:

1. dynamic semantic graph problem và benchmark bốn trạng thái;
2. typed graph with applicability/exception/conflict;
3. selective mid-denoising actuation;
4. event-triggered LLM controller;
5. exception preservation và semantic validity, không lấy rare sampling cost làm claim chính.

Nếu paper chỉ dùng cùng condition adapter và báo success trên rare cohorts, reviewer sẽ xem đây là một ablation của paper 1. Phải tách task, method và metrics rõ ràng.

Mốc quyết định:

- oracle graph thắng fixed guidance/projection/repair: tiếp tục thành paper 2;
- oracle không thắng: dừng mid-denoising claim hoặc chuyển thành benchmark/analysis;
- oracle thắng nhưng LLM selector không đạt: có thể viết về calibrated graph selection, nhưng không claim LLM controller mạnh.

## 13. Result files

- `DYNAMIC_CONSTRAINT_GRAPH_EVIDENCE_20260929.md`
- `outputs/dynamic_graph_partial_full_20260929_065108/RESULTS.md`
- `outputs/dynamic_graph_partial_full_20260929_065108/summary.csv`
- `outputs/dynamic_graph_comparison_1048645/` sau khi full grid hoàn tất
- `outputs/temporal_scope_comparison_1048591/RESULTS.md`
- `outputs/cross_dataset_constraint_20260924/AUDIT_SUMMARY.md`
- `outputs/request_condition_aggregate_1031786/RESULTS.md`
- `outputs/ctdf_style_aggregate_1044372/RESULTS.md`
- `outputs/harpoon_style_aggregate_1044475/RESULTS.md`
