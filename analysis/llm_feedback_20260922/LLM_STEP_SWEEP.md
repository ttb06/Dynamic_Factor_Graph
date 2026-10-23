# LLM-at-diffusion-step sweep: asynchronous experiment handoff

**Ngày submit:** 2026-09-23  
**Trạng thái tài liệu:** các kết quả full-checkpoint bên dưới đã chạy; semantic API và rollback sweep đang chạy/đợi queue. Không dùng trạng thái pending để claim.

## 1. Vì sao chuyển sang semantic stress-test

Full Adult TabDiff checkpoint (8,000 epochs) đã làm hard executable rules gần bão hòa: no-guidance chỉ còn khoảng `1.56–1.76%` violation. Post-repair đạt `0%`; direct typed state projection còn khoảng `0.065–0.13%` ở các biến thể tốt. Vì vậy Adult hard-rule setting hiện là **NO-GO cho claim rằng LLM là cần thiết**: thiếu positive cases tự nhiên và rule engine vừa rẻ vừa chính xác hơn.

Đợt mới giữ mục tiêu LLM can thiệp trong reverse diffusion, nhưng thu hẹp câu hỏi:

> Khi preview synthetic chứa lỗi semantic có kiểm soát, loại context nào và timestep nào giúp một LLM critic phát hiện đúng quan hệ, tránh sửa rare-but-valid record, rồi gate một typed rollback có sống sót tới mẫu cuối với chi phí thấp nhất?

Đây là mechanism/stress test. Nó chưa chứng minh lỗi tương tự xuất hiện tự nhiên với tần suất đáng kể.

## 2. Hypothesis mới và kill criteria

| ID | Hypothesis khả thi | Experiment quyết định | Kill criterion |
|---|---|---|---|
| S1 | Output contract `violated/satisfied/not_applicable/abstain` giảm false positive so với schema-only | `schema`, `typed`, `typed3` | Không tăng precision/rare-control specificity, hoặc gain chỉ đến từ việc prompt đã encode toàn bộ rule |
| S2 | Broad semantic rubric bắt lỗi world-knowledge/plausibility mà static rules không mô tả hết | `rubric`, `rubric41` so với `static` | Static verifier/rules bằng hoặc tốt hơn theo F1, distortion và cost |
| S3 | Hai preview liên tiếp hoặc delta cột giúp critic ở timestep sớm | `tpair`, `tdelta`, metrics theo `t` | Không có gain trước `t≈.10`, hoặc làm tăng false positive do preview nhiễu |
| S4 | Cost-sensitive abstention bảo vệ subgroup/exception hiếm | `abstain`, rare-valid France và Own-child age 45 controls | FPR trên controls vẫn cao, hoặc recall/cost-adjusted utility sụp |
| S5 | Model mạnh hơn cải thiện semantic discrimination đủ bù chi phí | `rubric41` vs `rubric` | Gain nhỏ hơn biến thiên/repeat hoặc token/latency tăng không đáng đổi |
| S6 | Self-consistency giúp lỗi ngẫu nhiên nhưng không cứu systematic bias | `typed3` vs `typed` | Không tăng precision/stability trên cùng case/timestep |
| S7 | LLM-gated typed rollback giảm final violation so với corrupt/no-feedback và tiến gần static/oracle rollback | 9 job `rb-*` | Không hơn no-feedback, hoặc kém static verifier rõ rệt với distortion tương đương |
| S8 | Lịch gọi tối ưu là sparse và late, không phải gọi mọi step | `schedule_frontier.csv`; rollback tại `.03/.10/.17` | Multi-call/earlier context không tạo Pareto gain trên F1–calls–tokens |

S2 là hypothesis niche đáng giữ nhất nếu có signal: **LLM là semantic relation proposer/router, còn compiler + typed actuator mới có quyền sửa state**. Nếu static rule condition thắng, kết luận đúng phải là bỏ LLM khỏi setting này.

## 3. Dữ liệu stress-test

Job `1028600` đã hoàn tất (`1m42s`) và tạo `220` case synthetic ở năm timestep thực tế: `.4138, .2414, .1724, .1034, .0345`.

- 5 corruption types: education functional dependency, numeric domain, spouse consistency, age–education plausibility, missingness coupling.
- Mỗi corrupt case có paired-clean control.
- Thêm rare-valid country và valid exception `Own-child, age=45` để đo over-correction.
- Chỉ synthetic previews/counterfactuals được gửi API; không gửi raw real rows.

Artifacts: `outputs/semantic_bank_1028600/semantic_cases.jsonl`, `bank_summary.csv`, `run_metadata.json`.

## 4. Job matrix đã submit

### Semantic critic audit (CPU/API)

| Job | Tag | Context/model |
|---:|---|---|
| 1028601 | schema | schema-only, `gpt-4.1-mini` |
| 1028602 | typed | typed contract, mini |
| 1028603 | rubric | semantic rubric, mini |
| 1028604 | static | executable/static rules in prompt, mini |
| 1028605 | fewshot | fabricated positives + rare-valid negative, mini |
| 1028606 | tpair | current + previous preview, mini |
| 1028607 | tdelta | current + changed columns, mini |
| 1028608 | abstain | false-positive cost 5×, mini |
| 1028609 | typed3 | typed contract, 3 repeats |
| 1028610 | rubric41 | semantic rubric, `gpt-4.1` |
| 1028611 | aggregate | `afterany` trên mười jobs |

Mỗi response dùng strict JSON schema qua Responses API và `store:false`. Job tổng hợp xuất precision/recall/F1, rare-control FPR, abstention, TP/1k tokens, latency và frontier cho lịch 1–3 calls với early stopping.

### Controlled end-to-end rollback (GPU/API)

Mỗi job dùng cùng checkpoint, seed, `96` records, `30` denoising steps và năm branch: clean/no-injection, corrupt/no-feedback, oracle rollback, static-verifier rollback, LLM rollback.

| Job | Condition | Target `t` |
|---:|---|---:|
| 1028613 | typed contract | .10 |
| 1028614 | semantic rubric | .10 |
| 1028615 | temporal pair | .10 |
| 1028616 | semantic rubric | .03 |
| 1028617 | typed contract | .03 |
| 1028618 | typed contract | .17 |
| 1028619 | semantic rubric | .17 |
| 1028620 | static rules through LLM | .10 |
| 1028621 | cost-sensitive abstention | .10 |
| 1028622 | aggregate | `afterany` trên chín jobs |

Rollback khôi phục các latent columns trước corruption, nên là **upper-bound actuator để isolate critic/timing**, không phải editing method hoàn chỉnh. Comparator static/oracle trong cùng sampler giúp bác bỏ LLM nếu LLM chỉ diễn đạt lại code rule.

## 5. Xem kết quả sau khi queue hoàn tất

Không cần đọc từng log trước. Hai thư mục aggregate là điểm vào chính:

1. `outputs/semantic_aggregate_1028611/overall_metrics.csv`
2. `outputs/semantic_aggregate_1028611/metrics_by_timestep_and_type.csv`
3. `outputs/semantic_aggregate_1028611/top_cost_adjusted_schedules.csv`
4. `outputs/rollback_aggregate_1028622/llm_detection_and_cost.csv`
5. `outputs/rollback_aggregate_1028622/branch_comparison.csv`

Nếu aggregate báo missing job, xem `run_metadata.json`, rồi đối chiếu `logs/<job-name>_<job-id>.out/.err`. Aggregator dùng `afterany`, nên vẫn thu được partial result nếu một model/API job thất bại.

## 6. Quy tắc quyết định hướng tiếp theo

- **GO có điều kiện cho verified semantic transactions** nếu rubric/temporal condition cải thiện held-out semantic detection so với static prompt, rare-control FPR thấp, và LLM rollback thu hồi phần đáng kể oracle gain mà không tăng distortion.
- **Pivot sang sparse verifier routing** nếu LLM chọn đúng relation/tool nhưng tự judge truth kém; đây vẫn là LLM-in-the-loop tại diffusion step, nhưng truth/action thuộc verifier typed.
- **NO-GO cho LLM trên Adult hard rules** nếu static verifier hoặc post-repair tiếp tục bằng/tốt hơn. Khi đó dataset tiếp theo phải có ontology/tool semantics, value-dependent exception hoặc relation không thể local-repair sau sampling.
- Không chọn timestep theo F1 đơn lẻ: chọn trên Pareto của final violation, false-positive correction, rare-control coverage, categorical agreement/numeric distance, calls/tokens và latency.

## 7. Reproducibility và API boundary

Scripts chính: `scripts/semantic_preview_bank.py`, `semantic_llm_sweep.py`, `semantic_sweep_aggregate.py`, `llm_rollback_sampler.py`, `rollback_aggregate.py`. Slurm entry points: `slurm/20` đến `24`.

API implementation theo official OpenAI guidance: strict Structured Outputs bằng `text.format`, và tắt lưu response bằng `store:false`. API key chỉ được load runtime từ env file và không ghi vào artifacts.

## 8. Quick check và local-LLM follow-up (2026-09-23)

Tất cả jobs `1028600–1028622` đã kết thúc, nhưng mọi OpenAI generation call đều trả `429 credit_balance_exhausted`. Do đó các zero LLM metrics là **infrastructure failure, không phải experimental result**.

Phần không phụ thuộc API vẫn cho signal rõ: controlled corruption tồn tại tới final sample với intended violation `100%`; oracle rollback và executable static-verifier rollback đưa intended violation về `0%` cho cả bốn relations. Sau rollback, categorical agreement với clean đạt `0.991–1.0` và numeric latent MAE nhỏ cho hầu hết relation. Như vậy typed mid-step actuation là khả thi, nhưng chưa có evidence rằng LLM gate tốt hơn static verifier.

Đã submit local fallback dùng models có sẵn trong cache, không gửi dữ liệu ra ngoài:

- environment setup: `1030216`;
- Llama-3.1-8B-Instruct × typed/rubric/temporal: `1030217–1030219`;
- Qwen3-8B × typed/rubric/temporal: `1030220–1030222`;
- Gemma-3-12B-IT × typed/rubric/temporal: `1030223–1030225`;
- aggregate: `1030226`.

Kết quả sẽ nằm tại `outputs/local_semantic_aggregate_1030226/`. Gate tiếp theo: chỉ nối local critic vào end-to-end sampler nếu nó có precision tốt trên paired-clean/rare controls và temporal context tạo Pareto gain ở timestep sớm.
