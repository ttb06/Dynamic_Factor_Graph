# Cập nhật phương pháp: dynamic distribution graph trong reverse diffusion

Ngày cập nhật: 2026-10-19

## 1. Kết luận ngắn

Nhánh có tín hiệu tốt nhất hiện tại không còn là “sửa mọi constraint violation”. Phương pháp đang tiến tới một **dynamic distribution graph**: mỗi factor biểu diễn thông tin train-fitted về marginal, dependency hoặc conditional distribution; factor được active theo trạng thái của row và timestep, rồi gửi correction có giới hạn vào dự đoán clean state ở middle reverse steps.

Trên News, hai operating point mới đều cải thiện đồng thời Shape, Trend, Alpha, Beta và downstream regression so với paired backbone:

- one-shot target-quantile transport, 4 bins, strength `.10`: NRMSE giảm `8.18%`, R² tăng `0.2884`, Alpha tăng `2.0265` điểm;
- two-pulse target-quantile transport, 4 bins, strength `.05`: NRMSE giảm `9.82%`, R² tăng `0.7792`, Alpha tăng `1.8663` điểm.

Đây vẫn là single-seed screening. Chưa được gọi là SOTA vì absolute R² còn âm, baseline thay đổi khá mạnh theo GPU dù paired comparison trong mỗi task vẫn hợp lệ, và chưa có multi-seed confidence interval.

## 2. Bài toán và thay đổi trọng tâm

### Cách đặt bài toán cũ

Graph chủ yếu chứa hard/near-hard rules. Online mechanism tìm violation rồi projection. Cách này có ba giới hạn:

1. Constraint do chính pipeline khám phá không thể là thước đo duy nhất.
2. Một edge đang satisfied không truyền thêm thông tin.
3. News graph loại target `shares`, nên gần như không thể tạo utility gain lớn bằng constraint projection cũ.

### Cách đặt bài toán hiện tại

Graph có nhiều loại factor:

1. **Typed semantic factor**: inequality, ordered relation, sum/function program đã được verify.
2. **Distribution factor**: train marginal và Gaussian-copula dependency.
3. **Conditional factor**: local joint distribution theo target category hoặc target quantile bin.
4. **Categorical density factor**: train marginal và PMI message giữa categorical nodes.

Với row (i), state (x_t), active graph thay đổi theo:

- giá trị/prediction hiện tại của row;
- target category hoặc target quantile bin hiện tại;
- state-specific distribution gap;
- readability/stability của edge theo diffusion time;
- trust region và pulse budget.

LLM vẫn có vai trò offline để đề xuất semantic edges. Online sampler không gọi LLM; nó dùng factor đã verify và train-fitted statistics.

## 3. Implement hiện tại

### 3.1 Numerical batch transport

File chính:

- `scripts/batch_distribution_transport.py`
- `scripts/dense_numeric_factor_messages.py`

Các mode đã có:

- `marginal`: monotone empirical quantile transport;
- `copula`: align train Gaussian-rank copula và marginals;
- `graph_copula`: chỉ align dependency nằm trên verified graph;
- `adaptive_copula`: mở thêm edge có state-specific correlation gap lớn;
- `sliced`, `graph_sliced`, `adaptive_sliced`: thử dependency bậc cao qua projection directions;
- categorical-target conditional transport;
- regression-target quantile conditional transport.

Regression-target quantile mode chia train target thành (K) bins. Ở mỗi reverse state, predicted target của từng row quyết định local operator đang active. Vì prediction có thể đổi theo timestep, membership và graph factor cũng đổi theo row/timestep.

### 3.2 Categorical factor messages

File: `scripts/categorical_logit_guidance.py`.

- Chỉ thay raw logits của cell còn masked trước MDLM update.
- Marginal message cân bằng phần probability mass còn có thể thay đổi.
- Pairwise message dùng smoothed PMI và normalized mutual information fit trên train.
- Message truyền hai chiều và active cả khi relation không violated.
- Timing coordinate dùng expected categorical mask probability. Bản log-sigma cũ đã được xác định là sai vì tạo empty interval; các run categorical trước khi sửa là no-op và không được dùng làm evidence.

### 3.3 Geometry và timing

- Correction đi qua clean preview rồi được đưa trở lại reverse update.
- Ellipsoidal trust region đo displacement theo learned per-feature sigma.
- One-shot hoặc một số pulse hữu hạn; không guidance dày ở mọi step.
- Typed projection và distribution messages có thể chạy trong cùng sampler.

## 4. Main screening result trên News

### 4.1 One-shot global copula, middle, trust radius 0.75

Job `1099203_2`, paired noise:

| Metric | Backbone | Dynamic graph | Thay đổi |
|---|---:|---:|---:|
| Shape ↑ | 97.4792 | 97.5466 | +0.0673; Shape error giảm 2.67% |
| Trend ↑ | 90.4259 | 90.4793 | +0.0535; Trend error giảm 0.56% |
| Alpha ↑ | 95.1401 | 97.1365 | +1.9964 |
| Beta ↑ | 42.1615 | 42.2778 | +0.1164 |
| R² ↑ | -1.5686 | -0.9202 | +0.6484 |
| NRMSE ↓ | 1.6027 | 1.3857 | giảm 13.54% |

Kết quả thắng 6/6 nhưng factor global chưa thể hiện rõ tính row-conditional.

### 4.2 Dynamic target-quantile graph: one-shot

Job `1099341_6`, 4 quantile bins, strength `.10`:

| Metric | Backbone | Dynamic graph | Thay đổi |
|---|---:|---:|---:|
| Shape ↑ | 97.4853 | 97.5601 | +0.0749; Shape error giảm 2.98% |
| Trend ↑ | 91.1478 | 91.1568 | +0.0089 |
| Alpha ↑ | 94.8853 | 96.9119 | +2.0265 |
| Beta ↑ | 42.3690 | 42.3958 | +0.0269 |
| R² ↑ | -0.8374 | -0.5489 | +0.2884 |
| NRMSE ↓ | 1.3555 | 1.2446 | giảm 8.18% |

Job `1099341_7`, strength `.25`, cũng thắng 6/6; fidelity/coverage cao hơn nhưng utility gain nhỏ hơn `.10`.

### 4.3 Dynamic target-quantile graph: two bounded pulses

Job `1099422_0`, 4 bins, strength `.05`, hai pulse cách nhau 0.18 log-sigma quantile:

| Metric | Backbone | Dynamic graph | Thay đổi |
|---|---:|---:|---:|
| Shape ↑ | 97.3764 | 97.4037 | +0.0272 |
| Trend ↑ | 91.3764 | 91.3996 | +0.0232 |
| Alpha ↑ | 94.9422 | 96.8085 | +1.8663 |
| Beta ↑ | 42.1297 | 42.2868 | +0.1571 |
| R² ↑ | -3.1706 | -2.3914 | +0.7792 |
| NRMSE ↓ | 2.0422 | 1.8416 | giảm 9.82% |

Strength `.10–.25` hoặc ba pulse thường làm Beta/utility giảm. Vì vậy recurrence chỉ có ích khi từng pulse yếu; “nhiều guidance hơn” đã bị bác bỏ.

## 5. Ablation quan trọng

### Timing/geometry

- One-shot, strength vừa, trust `.75`: thắng 6/6.
- Strength `1.0`, trust `1.5–3.0`: utility giảm mạnh; một run làm NRMSE xấu 31.6%.
- Không trust region: Shape error tăng khoảng 35.8%, Beta giảm 7.36 điểm.
- Repeated guidance toàn interval: không ổn định và thường mất coverage.
- Hai pulse yếu `.05`: có thể thắng 6/6; hai pulse mạnh hơn lại tạo trade-off.

### Dynamic conditional factor so với global factor

- Global two-pulse `.25`: Shape/Trend/Alpha tăng nhưng Beta và utility giảm.
- Target-quantile two-pulse `.05`: sáu metric đều tăng trong paired task.

Điều này ủng hộ dùng state/row-conditioned local distribution factor thay vì một correction global cố định. Vì các task có hardware nondeterminism, cần multi-seed confirmation trước khi claim effect size giữa hai cấu hình.

### Sliced transport

Sliced transport trực tiếp làm R² gần 0 nhưng phá Shape/Trend. Ghép thêm marginal rank calibration tạo post-hoc ceiling rất cao (Shape khoảng 99.82, Trend khoảng 94.09, NRMSE khoảng 1.00) nhưng Beta giảm và đây là direct endpoint calibration. Hiện chỉ xem là diagnostic upper bound, chưa dùng làm main method.

## 6. Diagnostic ceiling, không phải main result

Post-hoc 4-bin conditional copula, strength `.25`, trên cùng News baseline:

| Metric | Baseline | Diagnostic ceiling |
|---|---:|---:|
| Shape ↑ | 97.4792 | 97.5879 |
| Trend ↑ | 90.4259 | 90.8638 |
| Alpha ↑ | 95.1401 | 96.3464 |
| Beta ↑ | 42.1615 | 42.6099 |
| R² ↑ | -1.5686 | -0.5035 |
| NRMSE ↓ | 1.6027 | 1.2262 |

Nó chứng minh conditional factor chứa signal tốt, nhưng không chứng minh online mechanism đã khai thác hết signal đó.

## 7. Điều chưa được phép claim

1. Chưa thể claim SOTA.
2. Chưa thể claim multi-dataset consistency; Magic vẫn có Shape–Trend trade-off.
3. Chưa thể claim LLM edges tạo main gain; gain hiện chủ yếu đến từ train-fitted distribution factors.
4. Chưa thể dùng self-discovered constraint satisfaction làm primary metric.
5. Chưa thể coi certifier là main method; trên News nó thường giảm public violations nhưng làm fidelity/utility xấu.
6. Chưa được chọn cấu hình cuối bằng test score; final protocol phải tách validation để chọn strength/pulse/bin count rồi khóa test.

## 8. Jobs đang chạy/đã submit

| Job | Nội dung | Trạng thái tại thời điểm viết |
|---|---|---|
| `1099203` | global/graph copula one-shot, 3 datasets | hoàn tất |
| `1099227` | strength, interval, trust geometry | hoàn tất |
| `1099341` | target-quantile News + categorical Default pilot | News hoàn tất; categorical pilot cũ là no-op do schedule bug |
| `1099422` | bounded pulse News + categorical Default đã sửa | đang chạy/đánh giá |
| `1099434` | Adult categorical factor sweep | đã submit |

## 9. Bước quyết định tiếp theo

1. Khóa hai ứng viên News: one-shot `q4/.10` và two-pulse `q4/.05`; chạy ít nhất 3 seeds trên cùng GPU class.
2. Báo paired deltas và bootstrap confidence intervals, không chỉ raw scores.
3. Đánh giá categorical factors trên Default và Adult; giữ nhánh này chỉ khi tăng common fidelity/utility metrics, không dựa vào PMI nội bộ.
4. Thử factorized discriminator guidance nếu conditional transport vẫn để khoảng cách lớn so với diagnostic ceiling.
5. Chuyển hyperparameter selection sang validation split; test chỉ chạy một lần cho final table.

## 10. Claim tạm thời phù hợp nhất

> A row- and time-dependent distribution graph routes verified semantic and train-fitted conditional factors into bounded middle reverse-diffusion updates. Local target-state factors improve fidelity and utility more reliably than global or continuously repeated correction.

Claim này rộng hơn exception repair, trực tiếp nhắm common metrics, vẫn giữ main novelty ở dynamic graph theo row/state/timestep, và không phụ thuộc việc mọi LLM-proposed constraint đều đúng.
