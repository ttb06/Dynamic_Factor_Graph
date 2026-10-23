# Bổ sung LCATS-style domain constraints vào dynamic graph

**Ngày:** 20/10/2026  
**Jobs:** `1101506` failed only at evaluator metadata; corrected run `1101671` completed. Exact endpoint control: `1101700`.  
**Dataset screening:** Online News Popularity

## 1. Graph trước đây có các dạng này chưa?

Chưa có dưới dạng explicit graph edge. Actuation graph News trước thay đổi này có 11 edges:

- 5 ordered-summary programs;
- 2 functional programs;
- 1 simplex sum;
- 1 conditional sum;
- 2 directional inequalities do LLM bổ sung.

Graph Adult chủ yếu là conditional categorical implications. Không graph nào có typed `is_integer`, non-negative, unit-interval hoặc categorical-domain primitive.

Một số range được bảo đảm ngầm bởi preprocessing/inverse transform, còn category codes được MDLM bảo đảm hợp lệ. Integer-valued semantics không được bảo đảm: sample News cũ có fractional outputs ở nhiều count columns, chẳng hạn khoảng 80.9% ở `n_tokens_content`, 64.7% ở `timedelta`, 6.2% ở `num_hrefs` và 5.2% ở `num_imgs`.

## 2. LCATS thực tế dùng gì?

Official LCATS Insurance graph có 14 constraints:

- 12 `domain_bound` constraints;
- 1 order constraint;
- 1 conditional constraint.

Các domain primitives gồm:

- `is_integer(children)`;
- `age >= 18`;
- `bmi >= 0`, `charges >= 0`;
- upper bounds như `age <= 120`;
- membership như `region in {...}`.

Vì vậy phần lớn official LCATS graph trên Insurance là unary schema/domain integrity, không phải causal relation phức tạp.

## 3. Phần mới đã implement

### Typed integer action

Thêm program:

```json
{
  "kind": "integer",
  "column": " n_tokens_content",
  "tolerance": 0.000001
}
```

Tại intermediate clean preview, target là nearest integer. Middle-step strength có thể nhỏ hơn 1; endpoint certifier dùng exact projection. Stability gate yêu cầu nearest-integer target không đổi giữa các preview liên tiếp trước khi active.

### Domain bounds

Bounds dùng typed literal inequality vốn đã có trong actuator:

```json
{
  "left_column": " num_hrefs",
  "operator": ">=",
  "right_kind": "literal",
  "right_value": 0
}
```

### Conservative News profile

Tạo 62 domain edges và verify toàn bộ trên train:

- 8 integer edges cho time/count columns;
- 12 non-negative lower bounds;
- unit-interval bounds cho LDA, rate, subjectivity và positive-polarity features;
- signed-unit bounds cho sentiment/negative-polarity features.

Tất cả 62 candidates có zero train violation. Chúng được ghép với 11 semantic/functional edges cũ thành graph 73 edges.

Không thêm integer action vào regression target `shares`: target actuation có thể trực tiếp thay đổi utility và gây leakage/confound. Không thêm mọi column tình cờ integral trong train; chỉ giữ schema-semantically integer columns.

Categorical membership không cần action trong TabDiff vì categorical reverse process đã chọn từ finite codebook. Nó vẫn có thể được báo như structural guarantee.

## 4. Thực nghiệm đang chạy

Job `1101506` chạy tuần tự trên cùng node và seed:

| Variant | Mục đích |
|---|---|
| Typed control | graph 11 edges cũ |
| Integer `.10` | integer domain riêng, weak guidance |
| Integer `.25` | integer domain riêng, stronger guidance |
| Bounds only | non-negative/range projection riêng |
| Domain only | integer + bounds, không semantic graph |
| Typed + domain | graph cũ cộng domain primitives |
| Full domain conditional | typed + domain + target-quantile transport |
| Domain final repair | exact endpoint control kiểu LCATS |

Common evaluator báo Shape, Trend, Alpha, Beta, R² và NRMSE. Domain auditor báo:

- any-row domain violation;
- macro integer violation;
- macro bound violation;
- per-edge violation.

## 5. Cách diễn giải

Domain constraints này **không phải novelty chính**; chúng là:

1. một baseline/primitive family gần LCATS;
2. bằng chứng action layer hỗ trợ unary, pairwise và hyperedge constraints;
3. phép thử xem intermediate enforcement có giữ common fidelity tốt hơn exact final repair không.

Chỉ giữ trong main method nếu integer/domain trajectory guidance cải thiện hoặc giữ Shape/Trend/utility tốt hơn endpoint snapping. Nếu chỉ đạt domain validity còn fidelity giảm, nó nên nằm trong ablation hoặc appendix như negative result.

## 5.1 Kết quả corrected run `1101671`

| Method | Shape ↑ | Trend ↑ | Alpha ↑ | Beta ↑ | R² ↑ | NRMSE ↓ | Integer macro violation ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Backbone | 97.259672 | 89.864502 | **95.377997** | 45.597331 | -2.464418 | 1.861295 | 20.428467% |
| Integer `.10` | 97.259672 | 89.864513 | 95.376369 | 45.597331 | **-2.439732** | **1.854651** | 20.425415% |
| Integer `.25` | **97.260181** | 89.864517 | 95.376369 | **45.598958** | -2.461658 | 1.860553 | **20.370483%** |
| Bounds only | 97.259672 | 89.864502 | **95.377997** | 45.597331 | -2.464418 | 1.861295 | 20.428467% |
| Full domain conditional | 97.242625 | **89.976646** | 95.213609 | 45.452474 | -2.360559 | 1.833183 | 20.455933% |

Kết luận:

- Bounds-only là exact no-op vì backbone đã thỏa mọi lower/upper domain bound.
- Integer `.25` tạo khoảng 42.7k projection events nhưng macro violation chỉ giảm `0.058` điểm phần trăm; các bước reverse sau gần như ghi đè integer projection.
- Integer `.10` cải thiện NRMSE khoảng `0.36%`, nhưng các fidelity changes khác gần bằng numerical noise.
- Full domain conditional tăng Trend `0.1121`, R² `0.1039` và giảm NRMSE `1.51%`, nhưng Shape, Alpha và Beta đều giảm. Improvement chủ yếu đến từ conditional transport, không phải unary domain edges.
- `DomainFinalRepair` trong job này kế thừa integer weight `.25`, nên không phải exact LCATS endpoint comparator. Job `1101700` đã sửa riêng control này bằng hard rounding/clipping weight 1.

Như vậy intermediate integer/bound injection hiện là negative result: constraint family hợp lệ nhưng không mang lại common-metric gain và không duy trì integrity tới output.

## 5.2 Exact endpoint comparator `1101700`

| Method | Shape ↑ | Trend ↑ | Alpha ↑ | Beta ↑ | R² ↑ | NRMSE ↓ | Any domain violation ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Backbone | 97.259672 | 89.864502 | **95.377997** | **45.597331** | -2.464418 | 1.861295 | 94.018555% |
| Exact final repair | **97.268319** | **89.864506** | 95.371487 | 45.527344 | **-2.444706** | **1.855992** | **0.000000%** |

Exact repair sửa `3,851/4,096 = 94.02%` rows và `6,694` cells (`3.40%` toàn bảng). Nó đạt zero integer/bound violation, đồng thời:

- Shape tăng `0.00865`;
- R² tăng `0.01971`;
- NRMSE giảm khoảng `0.285%`;
- Alpha giảm `0.00651`;
- Beta giảm `0.06999`.

Đối với unary domain constraints, endpoint rounding/clipping đơn giản hiệu quả hơn intermediate weak injection: validity hoàn hảo, compute thấp và common-metric distortion rất nhỏ. Vì vậy dynamic trajectory mechanism không có motivation từ riêng integer/non-negative constraints; các constraint này nên là structural postprocessor/control. Main dynamic claim phải tiếp tục dựa vào inter-column/conditional distribution factors, nơi action sớm có thể thay đổi dependency trajectory.

## 6. File tái lập

- `scripts/lcats_domain_graph.py`
- `scripts/dense_numeric_factor_messages.py`
- `slurm/807_news_lcats_domain_trajectory.sbatch`
- `outputs/news_lcats_domain_graph_20261020/`
