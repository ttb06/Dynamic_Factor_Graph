# Tóm tắt thay đổi hướng nghiên cứu: từ LLM loop judge đến compositional dynamic graph

**Phạm vi:** bản ngắn để đối chiếu các giai đoạn trên Adult, Shoppers và Default.  
**Kết luận hiện tại:** LLM phù hợp nhất với offline knowledge acquisition; main novelty
nằm ở trajectory-dependent graph refresh và multi-edge propagation.

---

## 1. So sánh ngắn

| Giai đoạn | Câu hỏi chính | LLM nằm ở đâu? | Kết quả quyết định | Hướng xử lý |
| --- | --- | --- | --- | --- |
| Direct LLM loop judge | LLM có thể đọc từng intermediate row và quyết định lỗi/action không? | Gọi lặp lại trên row/request | Direct judgement không ổn định và tốn khoảng 37.5k tokens/bank | Không dùng làm main online controller |
| NL→DSL compiler | LLM có thể compile semantics một lần rồi local executor chạy không? | Trước sampling | 100% execution equivalence, khoảng 1.85k tokens | Giữ LLM ở offline knowledge layer |
| Static verified graph | Verified rules có giúp Shoppers/Default không? | Tạo/critic/judge graph | Final repair rất mạnh; graph chỉ có ít actuator edges | Cần dynamic state và action audit |
| Row/timestep active graph | Dynamic topology có hơn frozen không? | LLM không chạy online | Shoppers có gain rõ; Default phụ thuộc graph/setting | Giữ topology control, bỏ universal claim |
| Stronger LLM discovery | Model tốt hơn có tự tạo generation gain không? | GPT-5.4 toàn pipeline | Default: 177k tokens, 28 relations nhưng chỉ 2 online actions | Coverage không đồng nghĩa actionability |
| Flat middle-step injection | Middle guidance có tự thân tốt hơn late/final không? | Không cần LLM | Late-only/final thường tốt hơn | Bác bỏ flat-middle framing |
| Compositional dynamic graph | Action trước có kích hoạt edge sau và cần refresh không? | LLM/data tạo source graph; executor online | Controlled gain rất lớn; natural Default gain 21.8–25.5% held-out | Main direction hiện tại |

---

## 2. Giai đoạn đầu: LLM chấm từng row hoặc timestep

Ý tưởng ban đầu là đưa `x_hat_0(x_t)`, schema và synthetic row cho LLM; LLM trả về
violation, columns, confidence và suggested change. Đây là adaptation hợp lý từ
PromptLoop/PPAD sang tabular, nhưng có ba vấn đề:

1. Preview sớm chưa readable nên feedback dễ sai.
2. Cùng row nhưng request khác phải cho quyết định khác; row plausibility đơn thuần
   không giải quyết được.
3. Free-form action khó chuyển thành typed operation an toàn cho mixed columns.

Một focused API bank ban đầu còn có label leakage qua case IDs và oracle candidate
`none`; result đó đã bị loại khỏi scientific evidence.

Corrected request-conditioned bank dùng opaque IDs và same-row/different-request pairs:

| Model/path | Four-way accuracy | Swap-pair accuracy | Tokens |
| --- | ---: | ---: | ---: |
| GPT-4.1-mini direct | 0.864 | 0.867 | 37,544 |
| GPT-5.4-mini direct | 0.650 | 0.367 | 37,699 |
| GPT-4.1-mini compile→DSL→execute | **1.000** | **1.000** | 1,846 |
| GPT-5.4-mini compile→DSL→execute | **1.000** | **1.000** | 1,853 |

Kết luận: khi semantics có thể executable, compile once + deterministic execution tốt
hơn repeated LLM judgement. Vì vậy LLM loop không bị bỏ hoàn toàn; nó được chuyển từ
online judge sang offline graph construction.

---

## 3. Shoppers: verified graph và active-edge guidance

Shoppers là nơi pipeline typed graph đầu tiên chạy end-to-end:

- LLM protocol dùng 110,513 tokens và 15 API calls;
- 8 source edges được giữ;
- qua actuation audit chỉ còn 3 actuator edges;
- các edges chính liên quan page count–duration và `BounceRates <= ExitRates`.

Early state-active result cho thấy calibration quan trọng:

| Method | Macro violation | Held-out error | Actions |
| --- | ---: | ---: | ---: |
| Fixed middle | 0.0011 | 0.0046 | 377 |
| State-active | 0.0013 | 0.0047 | 224 |
| Calibrated state-active | 0.0055 | 0.0006 | 29 |
| Calibrated final repair | 0.0053 | **0.0004** | 28 |

Bài học:

- zero violation có thể xóa legitimate exception mass;
- state gate giảm actions/drift;
- nhưng final repair vẫn rất mạnh vì graph đơn giản và action không tạo edge mới.

Ở broader dynamic-window experiments, Shoppers dynamic từng giảm khoảng 9.7% macro,
32.1% held-out và 41.3% calibration error so với frozen graph. Tuy nhiên graph-specific
effects và final repair khiến result này chưa đủ để định nghĩa contribution.

---

## 4. Default: graph coverage không đồng nghĩa online actionability

Default cho kết quả ít ổn định hơn Shoppers. Một số strong-topology runs có dynamic
gain khoảng 5–10% macro và 12.5–17.5% held-out so với frozen, nhưng matched-budget
results không luôn tốt.

Pipeline GPT-5.4 toàn bộ roles tạo:

- 177,176 tokens, 20 API calls;
- 28 LLM-protocol relations;
- 15 endpoint patterns;
- 22 external/hybrid relations.

Tuy vậy curation chỉ cho rất ít safe actuators và một sampler batch chỉ phát hai
actions. Điều này thay đổi giả thuyết dự án:

```text
Old assumption:
better LLM -> richer graph -> better generation

Observed reality:
better LLM -> richer verifier graph
              but not necessarily readable, exposed, actionable edges
```

Do đó không tiếp tục tăng API/model size. Bottleneck chuyển sang graph structure và
trajectory mechanism.

---

## 5. Vì sao hướng “dynamic graph theo row” vẫn chưa đủ

SAGE đã nghiên cứu value-dependent dependency khi LLM sinh table. SCTab-Diff cũng có
static LLM prior nhưng score phụ thuộc current row. Harpoon/CTDF áp constraint trong
reverse steps. Vì vậy chỉ nói “mỗi row có graph khác nhau” dễ bị related work bao phủ.

Các strict controls còn cho thấy:

- planner có thể sửa antecedent để né request nếu không scope-lock;
- dense energy gây outside edits rất lớn;
- tăng guidance scale hoặc hold không giải quyết residual;
- với flat constraints, late-only thường tốt hơn middle.

Gap được thu hẹp thành:

> Graph topology không chỉ phụ thuộc row hiện tại; nó phải được refresh sau action vì
> action ở cạnh trước có thể làm cạnh sau mới applicable.

Đây là điểm mà frozen graph, một late pulse và independent rule list không mô tả được.

---

## 6. Hướng hiện tại: compositional dynamic graph

### 6.1. Controlled evidence

Hai-edge chain có action `AGE` kích hoạt edge `EDUCATION`:

- dynamic chain violation: 1.63%;
- frozen: 74.21%;
- late-only: 70.80%.

Result chứng minh mechanism nhưng graph không phải semantic invariant.

### 6.2. Natural evidence trên Default

Verified empirical motif:

```text
BILL_AMT3 == 0 -> PAY_4 == -2 -> PAY_5 == -2 -> PAY_6 == -2
```

Ba-seed result:

| Comparison | Macro gain | Held-out gain | Calibration gain | Held-out wins |
| --- | ---: | ---: | ---: | ---: |
| Dynamic vs frozen | 2.6% | **21.8%** | **26.2%** | 3/3 |
| Dynamic vs late-only | **10.3%** | **50.6%** | **60.8%** | 3/3 |
| Dynamic vs final-only | 2.9% | **25.5%** | **29.4%** | 3/3 |

92.6% dynamic actions nằm ở middle interval `(0.25, 0.375]`. Khi match action count
với frozen, held-out gain vẫn 24.4%, nhưng macro/calibration kém frozen. Vì vậy dynamic
topology có selection value, còn full semantic gain cũng cần downstream action budget.

---

## 7. Direction thay đổi cụ thể như thế nào?

### 7.1. Vị trí của LLM

```text
Ban đầu:
LLM đọc từng intermediate row -> critique -> action mỗi vài steps

Hiện tại:
LLM/data/expert đề xuất typed graph trước sampling
-> deterministic verification
-> compositional compiler
-> local online executor refresh graph mỗi step
```

LLM vẫn quan trọng cho external knowledge, relation semantics, exception proposal và
cross-dataset graph construction. Nhưng LLM không nằm trên critical sampling path.

### 7.2. Đối tượng dynamic

```text
Ban đầu: feedback text thay đổi theo row/timestep
Sau đó: active edge set thay đổi theo row/timestep
Hiện tại: action thay đổi state -> state làm topology thay đổi -> downstream action
```

### 7.3. Contribution

```text
Ban đầu: LLM-in-the-loop tabular diffusion
Sau đó: row/timestep dynamic semantic graph
Hiện tại: intervention-induced compositional graph propagation
```

### 7.4. Benchmark và metric

```text
Ban đầu: rare/user-conditioned rows, raw validity
Sau đó: Shoppers/Default flat typed rules, dynamic-vs-frozen
Hiện tại: natural multi-edge motif, held-out semantic calibration,
          frozen/late/final/action-matched controls + common fidelity/utility
```

---

## 8. Đóng khung paper hiện tại

Một framing ngắn có thể dùng:

> Existing constraint-guided tabular generators mostly treat constraints as a fixed
> set to enforce. We study compositional constraints whose applicability changes as
> earlier interventions modify the reverse trajectory. We compile verified relations
> into typed acyclic motifs and repeatedly refresh the row-level active graph, enabling
> multi-hop propagation while calibrating probabilistic exceptions.

Ba contributions hợp lý:

1. problem formulation về intervention-induced active graph trong mixed-type diffusion;
2. verified compositional compiler + scope-preserving calibrated trajectory controller;
3. controlled và natural evidence với frozen, late, final và action-matched controls.

Không nên dùng “LLM per-step feedback” làm title claim. LLM knowledge acquisition nên
là component tạo source graph và future cross-dataset generalization.

---

## 9. Điều còn thiếu

- blind motif selection trên dataset thứ hai;
- second TabDiff training checkpoint;
- faithful same-task CTDF/Harpoon/LCATS-style comparisons;
- natural LLM-only edge tạo gain ngoài empirical graph;
- numeric/text compositional motifs;
- fully independent test table sau khi motif/hyperparameters đóng băng.

Vì vậy project hiện đã vượt qua giai đoạn “idea có work không” và có thể bắt đầu viết.
Phần còn lại là external validation và baseline completeness, không phải tiếp tục
tuning LLM loop hoặc tăng injection strength.

---

## 10. Artifacts chính

- Báo cáo technique đầy đủ:
  `TECHNIQUE_REPORT_COMPOSITIONAL_DYNAMIC_GRAPH_20261010_VI.md`.
- Main result:
  `outputs/natural_dag_stable_confirmation_v2_20261010/RESULTS.md`.
- Action-matched:
  `outputs/natural_dag_action_matched_aggregate_20261010/RESULTS.md`.
- Common quality:
  `outputs/natural_dag_quality_aggregate_20261010/RESULTS.md`.
- Corrected LLM request probe:
  `REQUEST_CONDITION_PROBE.md`.
- Historical full report:
  `FULL_PROJECT_REPORT_DYNAMIC_GRAPH_20261009_VI.md`.
