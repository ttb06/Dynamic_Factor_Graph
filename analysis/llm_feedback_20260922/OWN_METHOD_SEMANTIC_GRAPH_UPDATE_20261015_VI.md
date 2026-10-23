# Cập nhật phương pháp: semantic graph và trajectory projection

Ngày cập nhật: 2026-10-15. Tài liệu này chỉ theo dõi phương pháp đang phát triển; các job baseline gần đây được giữ nguyên nhưng không thuộc phạm vi phân tích này.

## 1. Câu hỏi nghiên cứu hiện tại

Phương pháp được tách thành hai bài toán độc lập:

1. **Graph discovery:** có thể tạo một typed constraint graph chứa quan hệ có cơ sở ngữ nghĩa, không chỉ là correlation hoặc các biến thể threshold lặp lại hay không?
2. **Trajectory guidance:** với một graph đã được tin cậy, việc thay đổi active graph và action theo row state lẫn diffusion timestep có giữ constraint mà làm hỏng fidelity/utility ít hơn final-only repair hay không?

Chỉ khi cả hai nhánh đạt tiêu chí riêng mới chạy full pipeline `LLM discovery → verified graph → dynamic trajectory controller`. Thiết kế này tránh việc dùng satisfaction trên constraint tự tạo làm bằng chứng duy nhất.

## 2. Nhận định lại graph LLM cũ

Graph Default cũ giữ 28/58 đề xuất sau critic, judge và calibration; 23/28 cạnh có arity từ ba cột hoặc có conjunction/exception. Tuy nhiên, đọc trực tiếp các cạnh cho thấy structural complexity không đồng nghĩa với semantic diversity:

- 25/28 cạnh là soft relation;
- nhiều cạnh chỉ lặp mẫu `BILL_AMTk = 0 → PAY_AMT{k-1} = 0` ở các tháng khác nhau;
- nhiều exception về refund, prepayment hoặc posting delay làm edge dài hơn nhưng không tạo relation family mới;
- persistence của `PAY_k = -2` qua tháng kế tiếp chủ yếu là temporal correlation, không phải invariant;
- không có contextual column comparison được giữ lại.

Đánh giá thủ công sơ bộ:

| Nhóm relation | Ý nghĩa | Quyền hợp lý |
| --- | --- | --- |
| Bill bằng/không dương dẫn tới payment bằng 0 | Process prior hợp lý nhưng refund, prepayment và timing tạo nhiều trường hợp đúng mà rule không bao quát | `score_only`, không hard-actuate |
| Trạng thái `PAY_k=-2` tiếp tục ở tháng liền kề | Có tính dự báo nhưng chưa có semantic necessity | `score_only` hoặc diagnostic |
| Bill vượt credit limit nhưng status lại là no-use/no-bill | Mâu thuẫn codebook/accounting tương đối rõ nếu metadata xác nhận nghĩa của `-2` | Human review; có thể actuation sau metadata check |
| Bill dương chuyển sang không dương mà không có payment | Accounting explanation hợp lý nhưng thiếu refund/write-off/adjustment fields | `score_only`; không ép thành hard constraint |

Định lượng lại trên graph cũ cũng xác nhận nhận xét này. Các biến thể zero-bill/payment được giữ dưới dạng soft với calibration violation khoảng 0–4,68%; các cạnh `PAY=-2` persistence có thể ổn ở một cặp tháng nhưng nhiều chiều/tháng khác vi phạm khoảng 7,81–17,13% trên calibration và 8,94–13,40% trên held-out. Chỉ relation `bill vượt credit limit → status không được là -2` được judge giữ near-hard (calibration 0,26%, held-out 0%), nhưng vẫn cần xác nhận codebook trước khi có actuation authority.

Do đó graph cũ không được dùng nguyên khối làm actuator chỉ vì held-out shift thấp.

## 3. Pipeline graph discovery mới

### 3.1 Nguồn thông tin

LLM chỉ nhận schema, glossary, column types, frequent categorical values và aggregate statistics trên discovery split. Không có raw real row được gửi ra API. Proposal roles gồm domain/process knowledge, empirical structure và trajectory actionability. Critic chỉ thấy discovery/counterexample aggregates; semantic judge không thấy calibration/test.

### 3.2 Typed DSL

Mỗi cạnh gồm:

- `applicability`: conjunction xác định row state mà relation có hiệu lực;
- `requirements`: một hoặc nhiều literal/column predicates phải đúng;
- `exceptions`: các conjunction làm cạnh tạm deactive;
- `relation_family`, `hardness_prior`, `evidence_basis`;
- rationale, directionality và plausible counterexample.

Prompt mới ưu tiên equation, directional inequality, conditional denial constraint, accounting identity/bound, temporal ordering, process-coupled missingness và value-conditioned dependency. Mỗi atom phải qua counterfactual necessity test: bỏ atom phải thay đổi applicability, violation hoặc exemption; không được thêm atom chỉ để tăng arity.

### 3.3 Kiểm tra mới sau LLM

`audit_constraint_meaningfulness.py` thực hiện:

1. bỏ lần lượt từng applicability/requirement/exception atom và đếm số row whose executable decision changes;
2. đánh dấu atom counterfactually redundant;
3. so Jaccard của active/violation masks để tìm edge gần trùng;
4. đánh dấu near-constant consequent, low support, distributional/representation family và dataset-empirical rule;
5. tách `actuation_candidate`, `score_only`, `human_review`, `reject_or_revise`.

Audit không tự cấp action authority. Từng cạnh còn lại phải được đọc thủ công trước khi dùng.

### 3.4 Experiment đang chạy

- Job `1083779`: focused Luna profile `trajectory_actionable` trên Default, tối đa 4 candidates/call và 120.000 token toàn run.
- Job `1084008`: resume cùng output/cache của `1083779`, chốt sau một proposal round thay vì trả tiền cho round thứ hai đã cho tín hiệu lặp lại.
- Job `1083780`: semantic/counterfactual audit phụ thuộc.
- Job `1083932`: codebook-aware rerun, một proposal round, 100.000-token hard budget; chạy sau audit hiện tại để không gọi API trùng lúc.
- Job `1083933`: meaningfulness audit phụ thuộc cho codebook-aware graph.
- Terra và Luna đều strict-output đúng ở smoke test nhưng request schema lớn trước đây timeout ở upstream với usage bằng 0; Gemini trả sai typed schema và đã bị loại thủ công. Focused run giảm output mỗi call từ 5.000 xuống 3.000 tokens và giảm số candidates để tránh upstream timeout.

Các lần thử trước dừng fail-closed ở structured-output/schema validation; không có graph lỗi được dùng. Đọc trực tiếp output Gemini cho thấy nó lặp năm zero-bill/payment rules và ba `PAY=-2` persistence rules, đồng thời bỏ các trường hardness/evidence/counterexample. Toàn bộ output đó đã bị loại. Endpoint hiện dùng Chat Completions; parser chỉ nhận typed JSON và malformed candidate được ghi thành rejection thay vì làm pipeline crash.

Ba API calls đầu của focused run đã dùng 25.814 tokens. Đọc thủ công cho thấy structured-output đã ổn, nhưng chất lượng tri thức vẫn là nút thắt:

- hai cạnh `PAY_0 > 0 → BILL_AMT1 > 0` và `BILL_AMT2 ≤ 0 → PAY_AMT1 ≤/== 0` tiếp tục xuất hiện dưới nhiều cách diễn đạt; đây không phải relation family mới;
- persistence `PAY_0=-2 → PAY_2=-2` lại được đề xuất và vẫn chỉ nên là `score_only`;
- các exception như `PAY_AMT1>0` cho delinquency/bill relation nghe hợp lý nhưng có nguy cơ là hậu nghiệm do LLM tự dựng; chỉ được giữ nếu executable audit cho thấy exception có support và giảm violation ổn định trên các split;
- candidate `BILL_AMT2>0 ∧ PAY_AMT1=0 → PAY_0≠-1` có state-dependent applicability và nhiều repair endpoint, nên đáng review nhất. Tuy nhiên nó vẫn chỉ là soft guardrail vì `PAY_0` có thể lệch chu kỳ so với `BILL_AMT2/PAY_AMT1` và dữ liệu thiếu các kênh payment/adjustment;
- candidate overpayment và over-limit hiện thiếu semantic necessity; không được cấp actuation authority chỉ vì empirical verification tốt.

Kết luận tạm thời: LLM có thể cung cấp hypothesis/guardrail mới, nhưng ba calls đầu **chưa tạo được hard semantic constraint mới**. Kết quả này ủng hộ thiết kế fail-closed và phản bác việc coi graph LLM là ground truth tự động.

Run gốc dừng ở call `proposal_r2_stateful_process` vì router trả top-level `error` và không tính token cho call lỗi. Bốn proposal calls hợp lệ đã dùng tổng cộng 34.837 tokens và được cache. Job `1084008` resume với `rounds=1`: không gọi lại bốn calls đã trả phí, bỏ round hai không còn giá trị thông tin, rồi chỉ chạy critic/judge còn thiếu. Dependency của audit đã được nối lại sang resume job nên không còn trạng thái `DependencyNeverSatisfied`.

Focused run và audit đã hoàn tất:

| Mốc | Kết quả |
| --- | ---: |
| API calls hợp lệ | 12 proposal/review/judge calls + 1 failed call không tính token |
| Tổng token tính phí | 77.412 |
| LLM protocol proposals / retained | 14 / 8 |
| Hard / near-hard / soft sau judge | 0 / 0 / 8 |
| Audit recommendation | 5 `score_only`, 3 `reject_or_revise`, 0 auto-promotion |
| Redundant atoms / near-duplicate pairs | 0 / 1 |

Năm cạnh `score_only` đều là anomaly/monitoring signals, không phải ground-truth constraints. Cạnh có empirical violation thấp nhất là overpayment relation `BILL_AMT1>0 ∧ PAY_AMT1>BILL_AMT1 → BILL_AMT2>0` (train 0,46%, held-out 0,33%), nhưng vẫn phụ thuộc cycle alignment và omitted refund/credit/fee fields. Hai zero-bill/payment variants trùng gần nhau; `PAY_0=-2 → PAY_2=-2` bị `reject_or_revise` do undocumented code; `BILL_AMT1=0 → BILL_AMT2≤0` bị loại với train violation 27,21% và held-out 23,00%. Không cạnh nào đủ điều kiện làm actuator.

Kết quả này thu hẹp vai trò LLM một cách rõ ràng: trên Default, LLM graph hợp lý nhất là **soft action scorer/diagnostic guardrail**; hard actuation phải dùng external specification hoặc domain expert verification. Job codebook-aware tiếp theo kiểm tra liệu caveat ngay từ proposal stage có tăng diversity và giảm số candidate phải reject, chứ không cố ép sinh ra hard rule.

Protocol mới truyền thêm calendar metadata và các giới hạn nhận thức từ [UCI codebook](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients). Codebook xác định tháng của status, bill và payment fields, nhưng không xác định liability của chu kỳ nào được mỗi `PAY_AMTk` thanh toán; mô tả chính thức cũng không định nghĩa đầy đủ các code `-2` và `0` xuất hiện trong dữ liệu. Vì vậy proposer, critic và judge đều bị cấm gán hard/near-hard authority cho relation dựa trên cycle alignment hoặc code semantics chưa được tài liệu hóa. Các field bị thiếu như refund, write-off, fee, channel và posting timestamp cũng được ghi thành omitted-variable caveat thay vì để LLM tự bịa exception.

## 4. Trajectory mechanism mới trên constraint công khai

News dùng năm bất đẳng thức công khai từ ConstrainedDGM (ICLR 2024). Đây là setting phù hợp để đánh giá mechanism vì constraint không do phương pháp đề xuất và satisfaction có ý nghĩa độc lập hơn.

Controller hiện tại làm ba việc:

1. decode predicted clean sample tại mỗi reverse step;
2. active/deactive edge theo readability schedule, row state và hysteresis;
3. tạo candidate action, chấm bằng validity, train-density và edit cost, rồi inject selected correction trở lại predicted clean state.

### 4.1 Weighted balanced projection

Action cũ chỉ sửa toàn bộ left endpoint hoặc toàn bộ right endpoint. Với quan hệ `x ≤ y`, candidate mới giải bài toán nearest feasible point theo standardized squared displacement:

\[
\min_{\Delta x,\Delta y}
  (\Delta x/s_x)^2 + (\Delta y/s_y)^2
\quad\text{s.t.}\quad
  x+\Delta x \le y+\Delta y+\epsilon.
\]

Khi row vi phạm một lượng `g`, nghiệm boundary chia correction theo variance:

\[
\Delta x=-g\frac{s_x^2}{s_x^2+s_y^2},\qquad
\Delta y=+g\frac{s_y^2}{s_x^2+s_y^2}.
\]

Candidate này không được áp cứng. Action selector so nó với left-only, right-only và abstain. Vì vậy dynamic graph vẫn quyết định edge nào có hiệu lực, còn projection layer chỉ mở rộng tập action khả thi.

### 4.2 Focused experiments

- Job `1082530[0]`: balanced projection với L1 action cost; đã hoàn thành nhưng bị bác bỏ.
- Job `1083777`: strict-final rerun của corrected balanced projection với paired L2 action energy.
- Job `1083778`: standard evaluation phụ thuộc.
- Job `1083784`: time-routed action layer, balanced projection chỉ ở middle (`t ≥ 0,15`), endpoint correction ở late/final.
- Job `1083785`: standard evaluation phụ thuộc cho time-routed variant.
- Job `1083909`: readability-routed confirmation, chuyển sang endpoint correction tại `t<0,25`, đúng vùng trajectory probe bắt đầu đạt khoảng 0,92 precision/recall.
- Job `1083910`: standard evaluation phụ thuộc cho readability-routed variant.
- Job `1083969/1083970`: balanced-L2 rerun sau khi sửa lỗi action bookkeeping mô tả dưới đây.
- Job `1083996/1083997`: edge-specific routing; mỗi cạnh dùng timestep đầu tiên đạt cả violation precision và recall ≥0,90 trên trajectory calibration để chuyển từ balanced sang endpoint action.

Tất cả dùng cùng News checkpoint, seed `20261447`, 4,096 samples, 50 steps, trajectory schedule và final certifier. Chỉ action set/lookahead thay đổi.

L1 trace chỉ chọn balanced candidate 9/1.834 middle actions và 2/1.141 final-repair actions; output chỉ khác controller cũ ở 17/4.096 rows. Quan trọng hơn, phiên bản đầu đặt nghiệm đúng tại biên verifier tolerance nên sau decode/serialization còn 0,1709% violation, trong khi controller cũ đạt 0%. Cấu hình đó bị bác bỏ. Projection đã được sửa để đi vào semantic feasible interior, và lookahead dùng projection lỗi đã bị dừng thay vì tiếp tục tốn compute.

Paired-L2 run sau đó chọn balanced candidate ở 2.245/2.389 middle actions và cho kết quả: Shape 96,5864; Trend 90,6927; R² -0,00543; NRMSE 1,00271. So với matched final repair, Trend tăng 0,4102 điểm, R² tăng 0,01081 và NRMSE giảm 0,00538; Shape giảm 0,00868 điểm. Nó còn sót đúng 1/4.096 hard-constraint violation. Ban đầu residual này được nghi là do reward abstain; rerun `1083777` đã bác bỏ giải thích đó và giúp truy ra lỗi action bookkeeping ở đoạn dưới.

Rerun `1083777/1083778` cho thấy nguyên nhân của 1 residual violation không phải abstention: `changed_columns` dùng relative tolerance mặc định của `np.isclose`. Ở giá trị khoảng 5.183, một balanced delta bằng 0,036 bị coi là không đổi, nên chỉ endpoint còn lại được commit và phép chiếu hai phía mất bảo đảm feasibility. Verifier nội bộ sau action và evaluator độc lập vì thế bất đồng. Code đã đổi sang `rtol=0` và chỉ bỏ numerical round-off ở mức float32; delta gây lỗi hiện được giữ lại. Các job `1083909` và `1083969` đều chạy sau bản sửa. Bảng dưới vẫn ghi số cũ cho đến khi evaluator độc lập của rerun hoàn tất.

Final certifier cũng được đổi từ một pass sang bounded fixed-point passes. Lý do là các cạnh `min≤max` và `avg≤max` dùng chung endpoint; action cho cạnh sau có thể tái vi phạm cạnh trước. Mỗi pass vẫn dùng cùng branch selector, và dừng khi tất cả hard edges hợp lệ hoặc không còn action. Cả proposed lẫn matched final repair dùng đúng certifier này, nên đây là correctness fix chứ không phải lợi thế bất đối xứng.

Paired-L2 làm numeric drift thấp hơn controller one-sided ở online output (0,07324 so với 0,07564) nhưng có residual macro violation cao hơn (0,02158 so với 0,01445). Vì vậy job `1083784` kiểm tra một hypothesis có định hướng rõ: balanced action giữ manifold tốt ở middle, còn one-sided endpoint action bền hơn ở late steps. Đây là trajectory-dependent action geometry, không phải chỉ thay đổi guidance scale.

Time-routed run `1083784/1083785` đạt 0% final violation. So với matched final repair, Trend tăng 0,3738 điểm, R² tăng 0,01809 và NRMSE giảm 0,00896; Shape giảm 0,02292 điểm, alpha precision giảm 0,02060 điểm và beta recall giảm 0,06348 điểm. Online macro violation là 0,02148, chỉ tốt hơn rất ít so với balanced-only 0,02158 và vẫn kém one-sided 0,01445. Do đó kết quả là một **Pareto trade-off**: trajectory intervention giữ dependency/utility tốt hơn post-repair, nhưng chưa cải thiện đồng đều mọi fidelity metric.

Cutoff `0,25` của job `1083909` không phải một hyperparameter sweep tùy ý. Nó được chọn trước từ probe: dưới vùng này clean preview đã đủ chính xác để endpoint correction ổn định hơn; phía trên vẫn cho balanced projection giảm edit energy. Nếu biến thể này không tốt hơn `t=0,15` về online persistence hoặc standard metrics, không tiếp tục sweep cutoff.

Biến thể `1083996` bỏ global cutoff khỏi main mechanism. Trên News, train-only trajectory cho ngưỡng reliability theo cạnh lần lượt khoảng 0,5102; 0,3878; 0,3469; 0,3265 và 0,2449. Khi (t) lớn hơn ngưỡng của cạnh, preview violation chưa đủ ổn định nên selector được phép dùng balanced minimum-displacement action; khi (t) giảm qua ngưỡng, chỉ các endpoint action bền hơn được dùng. Đây là routing theo `edge × timestep`, không đọc test data và không thêm learned policy.

## 5. Bằng chứng đã có trước experiment mới

Trên News seed chính:

| Method | Shape ↑ | Trend ↑ | XGBoost R² ↑ | NRMSE ↓ | Violation ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| TabDiff | 96.6554 | 89.1527 | -0.02545 | 1.01264 | 25.8789% |
| Final repair | 96.6134 | 90.2583 | -0.02852 | 1.01416 | 0% |
| Dynamic trajectory + final certifier | 96.6029 | 90.5473 | -0.01537 | 1.00765 | 0% |
| Balanced L2 + final certifier | 96.5864 | **90.6927** | **-0.00543** | **1.00271** | 0.0244% trước action-bookkeeping fix |
| Time-routed (`t=0,15`) + final certifier | 96.5904 | 90.6321 | -0.01043 | 1.00520 | 0% |

Dynamic trajectory tốt hơn final repair về Trend, R² và NRMSE ở seed này. Hai confirmation seeds cho kết quả mixed: một seed tăng nhẹ Trend/R², một seed giảm. Vì vậy mechanism hiện có tín hiệu nhưng chưa ổn định; weighted projection được thiết kế để giảm distributional damage tại chính action layer.

Không nên mô tả bảng này là cải thiện lớn trên toàn bộ fidelity. Kết luận đúng ở thời điểm hiện tại là: trên cùng sample budget và cùng final certifier, mid-trajectory guidance cho điểm tốt hơn final-only repair về Trend, R² và NRMSE ở seed chính, trong khi Shape gần như ngang nhưng hơi thấp hơn. R² vẫn âm trên News nên utility tuyệt đối còn yếu; phần đáng chú ý là mức giảm thiệt hại tương đối so với matched repair, không phải một claim SOTA utility.

## 6. Điều kiện để nối full pipeline

Graph discovery phải thỏa đồng thời:

- có ít nhất một relation family mới ngoài các zero-bill/payment variants;
- relation có directionality dựa trên process/domain semantics;
- atom không redundant và edge không near-duplicate;
- hard/near-hard actuation candidate không dựa chỉ vào empirical correlation;
- human review không phát hiện codebook misunderstanding hoặc omitted-variable exception nghiêm trọng.

Trajectory mechanism phải:

- đạt public constraint violation bằng final repair;
- cải thiện ít nhất một standard fidelity/utility metric so với matched final repair;
- không chỉ đổi metric bằng cách sửa mạnh hơn hoặc tăng số action bất thường;
- balanced projection thực sự được chọn trong trace, không chỉ tồn tại trong code.

Nếu một trong hai nhánh không đạt, chưa chạy full pipeline. Nếu graph LLM chỉ sinh soft prior, LLM vẫn có thể giữ vai trò offline proposer/critic nhưng actuator chính nên dùng constraint công khai hoặc được domain expert xác nhận.

## 7. Claim boundary và quyết định tiếp theo

Claim method có thể bảo vệ nếu các rerun thành công là: **một typed constraint graph thay đổi theo row state và diffusion time, đồng thời route action geometry theo độ tin cậy của predicted-clean trajectory, có thể đạt cùng external-constraint satisfaction như final repair nhưng giảm một phần thiệt hại dependency/utility**. Đây không phải claim rằng LLM tự khám phá ground-truth constraint, cũng không phải claim rằng mọi metric fidelity đều tăng.

Ba quyết định tiếp theo đã được khóa trước khi đọc kết quả mới:

1. Nếu edge-specific routing đạt 0% violation theo evaluator độc lập và Pareto-dominates global routing trên Trend/utility mà không làm Shape giảm thêm đáng kể, dùng nó làm main mechanism.
2. Nếu edge-specific routing không hơn, giữ global `t=0,15` hoặc `t=0,25` tốt nhất như ablation và dừng tuning action cutoff; chuyển compute sang confirmation seed/dataset.
3. Nếu codebook-aware graph chỉ còn soft diagnostics, không inject nó như hard actuator. Full pipeline sẽ dùng các cạnh đó làm scorer/guardrail, còn satisfaction chính tiếp tục đánh giá trên constraint công khai hoặc expert-verified.

Vì vậy tiến độ hiện tại đủ để đóng khung method và viết phần problem/method, nhưng chưa đủ để claim SOTA: còn cần kết quả rerun sau correctness fix, ít nhất một confirmation seed và một dataset external-constraint nữa cho main table.
