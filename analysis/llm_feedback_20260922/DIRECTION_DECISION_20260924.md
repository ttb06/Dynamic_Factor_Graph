# Direction decision after local-LLM sweep

> **Superseded evidence warning (2026-09-24):** the focused API F1 values in this file are invalid as scientific evidence because API-visible case IDs encoded `clean/corrupt`, paired variants shared batches, and controls were routed to candidate `none`. Keep them only for provenance. The leakage-controlled conclusion and replacement experiments are in `FOUNDATION_CONSTRAINT_DIRECTION_20260924.md` and `REQUEST_CONDITION_PROBE.md`.

**Ngày audit:** 2026-09-24  
**Evidence boundary:** các số trong Sections 1–4 đã chạy; Section 6 là jobs mới đang chạy/đợi queue.

## 1. Execution audit

| Jobs | Kết quả thực thi | Diễn giải |
|---|---|---|
| `1030217–1030219`, Llama-3.1-8B | Completed, 220/220 generations mỗi condition | Output bị mode-collapse sang Python code; `100%` không parse được. Đây là instruction/decoding failure, không phải semantic score hợp lệ. |
| `1030220–1030222`, Qwen3-8B | Completed, 220 cases mỗi condition | JSON parse thành công; dùng được để đánh giá. |
| `1030223–1030225`, Gemma-3-12B | Failed khi load | FP16 cần hơn GPU 22 GiB; chưa có semantic result. Retry dùng 8-bit đã submit. |
| `1030226`, aggregate | Completed quá sớm, chỉ load job đầu | Slurm dependency dùng colon hoạt động như alternative dependency trong setup này. Đã re-aggregate thủ công và đổi batch mới sang comma-separated AND dependencies. |

OpenAI sweep trước đó vẫn không có model result vì `credit_balance_exhausted`. Local Qwen là LLM evidence đầu tiên thực sự usable.

## 2. Main local-LLM result

220 cases gồm 100 corrupted positives, 100 paired-clean negatives và 20 rare/exception controls.

| Condition | Precision | Recall | F1 | Rare-control FPR | Unnecessary action | Tokens | Generation time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen typed | 0.352 | 0.190 | 0.247 | **0.00** | 0.036 | 95,123 | 83 s |
| Qwen broad rubric | 0.185 | 0.200 | 0.192 | 0.25 | 0.155 | 113,819 | 139 s |
| **Qwen temporal pair** | **0.525** | **0.210** | **0.300** | 0.05 | **0.014** | 128,585 | 147 s |

Temporal context tăng precision `+17.3 pp` so với typed contract và giảm false actions, nhưng recall chỉ tăng `2 pp`; broad rubric làm over-flagging nặng hơn. Multi-timestep scheduling không giúp: best cost-adjusted schedule là **một call** ở `t≈.034` hoặc `.103`, cùng precision `0.625`, recall `0.25`, F1 `0.357`. Gọi 2–3 lần không tăng F1 trên bank này.

## 3. Failure decomposition

Qwen không giải đồng đều năm relation. Exact recall theo corruption:

| Relation | Typed | Rubric | Temporal |
|---|---:|---:|---:|
| age–education plausibility | 0.95 | 1.00 | 0.70 |
| education FD | 0.00 | 0.00 | 0.15 |
| spouse consistency | 0.00 | 0.00 | 0.20 |
| numeric domain | 0.00 | 0.00 | 0.00 |
| missingness coupling | 0.00 | 0.00 | 0.00 |

Model chủ yếu dự đoán `age_education_plausibility` cho mọi loại lỗi. Confidence threshold không sửa được systematic confusion: tại threshold `.95`, temporal precision/recall vẫn khoảng `0.538/0.210`.

**Kết luận:** general-purpose “LLM xem row rồi tự tìm relation” hiện là **NO-GO**. Nó vừa bỏ sót 79% relations vừa đắt hơn verifier, trong khi static typed rollback đã đạt intended violation `0%` trên stress-test.

## 4. Điều còn có thể defend

Signal tích cực không phải LLM world knowledge tổng quát, mà là decomposition:

1. Một router/compiler xác định relation candidate và applicability domain.
2. LLM chỉ xác minh relation đó trên current/previous preview.
3. Typed actuator thực hiện atomic correction; independent verifier có quyền rollback action.

Đây là hướng **conditional GO** vì relation routing giải quyết đúng failure mode confusion giữa năm labels. Nhưng novelty phải nằm ở record/timestep-specific routing, abstention và transaction semantics; nếu router đã chứa đầy đủ executable rule thì LLM không còn đóng góp.

## 5. Formalized hypotheses

### H11 — Relation-routed verification

Cho LLM một candidate relation nhưng không cho nhãn valid/invalid sẽ tăng macro recall trên bốn relation bị bỏ sót mà không tăng paired-clean FPR.

- Primary gate: macro-F1 `>=0.60`, paired-clean FPR `<=0.10`.
- Strong gate để nối sampler: precision `>=0.80` trên candidate decisions.
- Kill: gain chỉ xuất hiện khi natural-language rubric encode trọn executable rule; khi đó static verifier là baseline thắng.

### H12 — Temporal evidence chỉ hữu ích sau routing

Current+previous preview giúp phân biệt transient anomaly với semantic violation sau khi relation đã cố định.

- Test: focused label vs focused rubric vs focused temporal, cùng model/cases.
- Support: temporal tăng precision/F1 và/hoặc calibrated abstention, đặc biệt ở `t>.10`.
- Kill: focused temporal không hơn focused rubric; khi đó chỉ gọi một lần late `t≈.03–.10`.

### H13 — Model-family/scale không cứu task formulation xấu

Llama-8B, Qwen-8B và Gemma-12B được so trên cùng focused contract. Model lớn hơn chỉ đáng dùng nếu tăng macro-F1/rare specificity đủ bù latency/VRAM.

- Kill: Gemma-12B không vượt Qwen-8B ít nhất `0.05` macro-F1 hoặc vẫn cần rule-complete rubric.

### H14 — End-to-end gate

Chỉ khi H11 đạt strong gate mới chạy LLM-gated rollback trong sampler. Experiment phải so cùng seed với corrupt/no-feedback, static verifier, oracle và post-repair.

- Success: thu hồi `>=80%` oracle violation reduction, với categorical/numeric distance không tệ hơn static rollback đáng kể.
- Kill direction: không đạt gate; pivot LLM sang **router of external semantic tools**, không làm truth judge.

## 6. Follow-up jobs submitted

| Job | Experiment |
|---:|---|
| `1031028` | install/check bitsandbytes for isolated local env |
| `1031031–1031033` | Llama-3.1-8B: focused label/rubric/temporal, strict system JSON contract |
| `1031034–1031036` | Qwen3-8B: focused label/rubric/temporal |
| `1031037–1031039` | Gemma-3-12B: same three conditions, 8-bit retry |
| `1031040` | AND-dependency aggregate after all nine terminate |

Expected entry point: `outputs/local_semantic_aggregate_1031040/`. If the focused sweep passes H11, the next implementation is one sparse local-Qwen/Gemma call at `t≈.03` or `.10`, not a multi-call loop.

API credit was re-tested after submission and is now active. A parallel strict-Structured-Output sweep was therefore added:

- `1031041–1031043`: GPT-4.1-mini focused label/rubric/temporal;
- `1031044–1031046`: GPT-5.4-mini focused label/rubric/temporal;
- `1031047`: GPT-4.1 focused temporal;
- `1031048`: GPT-5.4 focused temporal;
- `1031049`: AND-dependency aggregate.

This separates formulation effects from local-model capacity. API and local models receive the same 220 synthetic cases; no real row is transmitted, and Responses use `store:false`.

### Focused-label API result [executed update]

Two focused-label runs completed before rubric/temporal retries:

| Model | Precision | Recall | F1 | Rare-control FPR | Tokens | API latency |
|---|---:|---:|---:|---:|---:|---:|
| GPT-4.1-mini | 0.989 | 0.870 | 0.926 | 0 | 60,899 | 116 s |
| GPT-5.4-mini | 0.978 | 0.870 | 0.921 | 0 | 61,191 | 68 s |

Both models reached recall `1.0` on education FD, numeric domain, spouse consistency and age–education plausibility, but only `0.35` on missingness coupling. This supports H11's decomposition but does not yet show an LLM advantage over an executable verifier: the candidate relation label is supplied, and the rules in this stress test remain programmatically checkable. Focused rubric/temporal jobs failed on an implementation edge case for `intended_relation=none`; they provide no scientific result until rerun.

## 7. Artifact map

- Correct six-job aggregate: `outputs/local_semantic_reaggregate_20260924/`
- Full rollback re-aggregate: `outputs/rollback_reaggregate_20260923/`
- Raw local responses: `outputs/local_semantic_<condition>_<job>/responses.jsonl`
- Slurm logs: `logs/*_<job>.out` and `.err`
- API focused aggregate: `outputs/semantic_aggregate_1031049/`
