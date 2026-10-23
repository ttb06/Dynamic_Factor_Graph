# Refocus: dynamic graph discovery and injection along tabular diffusion

Ngày cập nhật: 03/10/2026

## 1. Quyết định sau feedback

Contribution chính chuyển sang **constraint graph thay đổi theo intermediate diffusion state**. Row-specific activation vẫn cần thiết, nhưng exception preservation chỉ là một safety metric, không phải motivation trung tâm.

Câu hỏi chính:

> Một specification graph được đề xuất từ LLM, training data và predefined rules có thể được tái kiểm chứng và instantiate thành các active subgraph khác nhau theo từng intermediate state; sau đó việc inject active graph ở middle denoising window có cải thiện validity–fidelity frontier so với static graph và final repair hay không?

## 2. Gap tổng quát

- LCATS có vòng LLM propose → executable validation → reject/retain, nhưng graph được khóa trước generation và repair sau sinh.
- SAGE có value-conditioned dependency graph, nhưng generator là autoregressive LLM và không có uncertainty của noisy diffusion state.
- LLM-TabLogic và SCTab-Diff tạo prior trước sampling; prior chủ yếu cố định theo table/request.
- Harpoon/CTDF can thiệp trong denoising, nhưng constraint specification cố định; gradient/projection thay đổi theo state chứ graph topology không được tái khám phá.

Khoảng trống không phải “LLM tạo graph” hay “constraint đi vào diffusion”, mà là:

> **State-conditioned graph discovery:** edge existence, reliability và row-level instantiation được cập nhật từ predicted-clean state, với abstention/persistence khi topology chưa ổn định.

## 3. Pipeline method dự kiến

### 3.1 Specification graph discovery

Ba nguồn được chuẩn hóa vào cùng typed graph:

1. LLM external knowledge từ schema/metadata và aggregate evidence;
2. train-discovered dependencies và constraint hypotheses;
3. predefined domain/user/public rules.

Vòng lặp kiểu LCATS được giữ lại nhưng mở rộng:

```text
propose candidate
→ type/execute check
→ train validation
→ counterexamples
→ refine direction/scope/hardness
→ calibration split
→ retain as hard, soft, uncertain, or rejected edge
```

Khác LCATS, không chỉ giữ zero-violation rules. Near-hard và probabilistic edges được giữ với confidence, support và provenance, nhưng không được hard-enforce mặc định.

### 3.2 State-conditioned graph discovery

Tại mỗi selected timestep:

```text
x_t → predicted clean row x_hat_0(x_t)
    → evaluate predicates and relation evidence
    → update edge support/confidence/lift/stability
    → active / inactive / abstain
    → G_(row,t)
```

Graph động ở hai cấp:

- **row topology:** edge xuất hiện khi antecedent/context đúng trong row state;
- **batch topology:** edge được enable khi relation vẫn có đủ support, confidence và lift trong current predicted-clean batch.

Persistence yêu cầu edge ổn định qua nhiều preview trước khi inject.

### 3.3 Injection

Thí nghiệm đầu giữ action cố định để cô lập giá trị của graph:

- cùng condition adapter hoặc guidance operator;
- so static graph, current-state graph, persistent dynamic graph và oracle-final graph;
- inject ở early, middle và late windows với matched compute.

Action routing chỉ được thêm sau khi dynamic graph chứng minh incremental value.

## 4. LCATS audit đã chạy

Official Insurance discovery tạo 14 constraints trên 7 columns:

- 12 unary domain bounds;
- 1 order relation;
- 1 conditional relation;
- chỉ 1 edge có explicit state predicate.

Điều này xác nhận LCATS loop là baseline tốt cho discovery/verification, nhưng graph cụ thể còn chủ yếu always-on và unary. Không nên dùng riêng Insurance LCATS graph để chứng minh dynamic topology.

Artifact: `outputs/lcats_graph_audit_20261003/RESULTS.md`.

## 5. General state-graph experiment đã hoàn tất

Job array `1054114` đã chạy Adult, Default và Shoppers với:

- official 8.000-epoch TabDiff checkpoints;
- 2.048 synthetic trajectories mỗi dataset;
- 50 reverse steps;
- tối đa 24 implications mỗi dataset;
- graph bank gồm hard, near-hard, probabilistic và low-support relations;
- một seed cho mechanism screening.

Đo hai lớp:

1. row-level active/violation precision–recall theo timestep;
2. batch-level edge discovery precision–recall với hard, near-hard và dependency profiles;
3. topology flip rate;
4. persistence gate so với one-shot state gate.

Aggregate job `1054115` đã hoàn tất.

Outputs:

```text
outputs/general_state_graph_<dataset>_1054114_<task>/
outputs/general_state_graph_aggregate_1054115/
```

### 5.1 Row-level edge instantiation

| Dataset | Early active P/R | Middle active P/R | Early violation P/R | Middle violation P/R |
| --- | ---: | ---: | ---: | ---: |
| Adult | 0.854/0.126 | 0.924/0.843 | 0.235/0.019 | 0.612/0.595 |
| Default | 0.827/0.486 | 0.966/0.968 | 0.548/0.059 | 0.719/0.798 |
| Shoppers | 0.824/0.340 | 0.973/0.879 | 0.468/0.011 | 0.816/0.551 |

Early là `t>0.75`; middle trong bảng là `0.25<t<=0.50`. Active edge nghĩa là antecedent đúng trong current row state. Violation edge yêu cầu cả antecedent đúng và consequent sai, nên khó đọc hơn. Kết quả quan trọng là xu hướng tăng từ early sang middle xuất hiện trên cả ba dataset, không chỉ Default.

### 5.2 Batch-level graph discovery

Với dependency profile, precision bằng 1.0 trong các bins được đánh giá, nhưng recall thay đổi mạnh:

| Dataset | Early recall | `0.50<t<=0.75` | Middle recall | Late recall |
| --- | ---: | ---: | ---: | ---: |
| Adult | 0.147 | 0.591 | 0.818 | 1.000 |
| Default | 0.677 | 0.917 | 0.989 | 1.000 |
| Shoppers | 0.132 | 0.744 | 0.929 | 1.000 |

Điều này cho thấy specification graph không nên được xem là fully instantiated ở early state: nhiều edge cuối cùng chưa có đủ support/confidence/lift trong predicted-clean batch. Middle state phục hồi 0.818–0.989 số edge cuối cùng tùy dataset.

Near-hard profile khó hơn nhưng cũng hội tụ ở middle/late. Persistent two-preview gate cải thiện nhẹ precision trong một số trường hợp, ví dụ Default middle từ 0.860 lên 0.895, đổi lại recall giảm từ 0.952 xuống 0.917. Persistence vì vậy là precision–recall control, không phải universal improvement.

## 6. Hypotheses

### H1 — LCATS-style strict graph quá đơn giản cho dynamic topology

Zero-violation filtering ưu tiên domain bounds và mechanical identities; phần lớn edge always-on. Giữ calibrated nonzero relations sẽ tạo graph phong phú hơn nhưng cần soft activation.

### H2 — Middle state là điểm đầu tiên topology đáng tin trên nhiều dataset — được ủng hộ

Row-level và batch-level edge precision/recall phải cùng tăng rõ từ early sang middle trên Adult, Default và Shoppers. Nếu chỉ xảy ra trên Default, claim chưa general.

### H3 — Persistence cải thiện graph discovery — được ủng hộ có điều kiện

Yêu cầu edge xuất hiện liên tiếp ở hai previews có thể tăng precision nhưng làm giảm recall; hiệu ứng nhỏ hoặc bằng không khi base gate đã chính xác. Đây nên là một configurable uncertainty control, không phải headline result.

### H4 — State discovery bổ sung giá trị ngoài row-only activation — có signal ban đầu

Batch-level re-estimation trì hoãn nhiều edge ở early state: dependency recall chỉ 0.132–0.677 tùy dataset và tăng lên 0.818–0.989 ở middle. Đây là topology change ngoài row-only activation, nhưng vẫn cần injection experiment để chứng minh utility.

### H5 — Middle injection chỉ đáng giữ nếu vượt same-graph baselines

Sau probe, cùng một graph và action operator phải được so ở final-only, fixed-middle, persistent-dynamic và oracle schedule. Dynamic method cần cải thiện joint/conditional fidelity hoặc compute-adjusted validity, không chỉ final satisfaction.

## 7. Cách diễn giải general

Method không được framing bằng một exception rule hoặc một subgroup cụ thể. Đơn vị tổng quát là:

```text
typed relation candidate
+ calibrated evidence
+ state predicate
+ trajectory stability
= dynamically instantiated edge
```

Exception chỉ là một trường hợp khiến edge inactive; các nguyên nhân khác gồm antecedent chưa hình thành, category còn đổi, numeric margin chưa ổn định, batch relation chưa có support hoặc edge confidence giảm.

## 8. Điều kiện quyết định

- Tiếp tục injection method nếu H2 và H3 lặp lại trên ít nhất hai dataset và H4 có incremental signal.
- Nếu row topology ổn nhưng batch topology không đổi, thu hẹp method về row–timestep active graph, không claim dynamic knowledge discovery.
- Nếu middle state đọc được graph nhưng injection không hơn final-only cùng graph/action, giữ trajectory result làm analysis và không claim generation improvement.
- Nếu strict LCATS graph chỉ gồm unary bounds, dùng LCATS làm discovery baseline nhưng đánh giá main trên graph bank nhiều relation families.

## 9. Khám phá graph bằng LLM và kiểm chứng dữ liệu

Đã chạy sáu job độc lập trên Adult, Default và Shoppers với GPT-4.1-mini và GPT-5.4-mini. Pipeline bắt chước cấu trúc vòng lặp của LCATS nhưng chỉ gửi schema và thống kê tổng hợp: LLM đề xuất implication có kiểu dữ liệu, bộ thực thi cục bộ đo support/confidence/lift/violation, rồi phản hồi kết quả tổng hợp cho vòng đề xuất tiếp theo.

| Dataset | Model | Retained | Strict | Near-hard | Soft | Tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Adult | GPT-4.1-mini | 10 | 5 | 4 | 1 | 12,306 |
| Adult | GPT-5.4-mini | 5 | 4 | 0 | 1 | 12,090 |
| Default | GPT-4.1-mini | 12 | 7 | 3 | 2 | 15,512 |
| Default | GPT-5.4-mini | 13 | 4 | 9 | 0 | 14,439 |
| Shoppers | GPT-4.1-mini | 14 | 12 | 1 | 1 | 12,164 |
| Shoppers | GPT-5.4-mini | 20 | 8 | 8 | 4 | 11,786 |

Pipeline tìm được nhiều họ quan hệ thay vì chỉ exception: thứ tự theo thời gian của trạng thái thanh toán, dependency giữa bill liên tiếp, quan hệ count–duration của hành vi duyệt web, missingness dependency và mapping giữa education–education number. Đặc biệt, Shoppers cho thấy quan hệ có hướng: `count=0 -> duration=0` có thể chính xác, trong khi chiều ngược lại chỉ near-hard. Đây là nguồn edge phù hợp cho dynamic graph hơn một bộ rule viết tay đơn lẻ.

Kết quả hiện tại mới là candidate discovery, chưa phải graph sẵn sàng inject. Audit phát hiện ba vấn đề tổng quát:

1. zero-violation có thể giữ các domain bound hiển nhiên hoặc implication gần tautology;
2. quan hệ ít support có thể đúng trên train nhưng sai mạnh trên heldout;
3. dù prompt yêu cầu tránh target leakage, model vẫn có thể sinh cạnh chứa target, nên cần bộ lọc xác định bằng code.

Phiên bản tiếp theo phải tách train thành discovery/calibration, yêu cầu edge đạt ngưỡng trên cả hai phần, giữ test chỉ để báo cáo, loại target/sensitive edges khỏi graph điều khiển, và chấm thêm non-vacuity. Exception handling chỉ còn là safety gate nhỏ.

Artifacts: `outputs/llm_verified_graph_*_1054118_*`.

## 10. Phạm vi method tổng quát nên triển khai

Đầu vào không phải một semantic request riêng lẻ mà là một specification graph được hợp nhất từ ba nguồn: external knowledge của foundation model, dependency khám phá từ train, và rule do domain expert cung cấp. Mỗi edge lưu predicate, relation type, hardness, provenance, support và calibration uncertainty.

Ở mỗi preview của reverse diffusion, graph được instantiate theo hai mức: antecedent của từng row và độ tin cậy của relation trên current batch. Scheduler chỉ inject edge đủ ổn định trong một cửa sổ giữa trajectory; action operator được giữ giống nhau giữa các baseline để đo đúng đóng góp của topology động.

So sánh chính cần là: unconditional TabDiff; static verified graph; LCATS-style strict graph; final-only repair/projection; fixed-middle graph; state-conditioned dynamic graph; và oracle topology. Claim mục tiêu là state-conditioned graph cải thiện validity–fidelity trade-off so với cùng graph/action nhưng lịch static hoặc final-only, không claim LLM tự thân tạo rule hoàn hảo.

## 11. Implementation mới: typed dynamic graph guidance

Pipeline mới đã thay thế “dynamic graph” cũ vốn chỉ chọn column theo conditional-response magnitude. Graph hiện gồm typed directional edges; mỗi edge lưu antecedent, consequent, hardness, evidence trên discovery/calibration, provenance và action eligibility.

Sampler thực hiện bốn bước tại mỗi predicted-clean preview:

1. instantiate antecedent và violation theo từng row;
2. kiểm tra persistence và drift của các column liên quan;
3. dùng edge-specific readability cutoff được học từ trajectory calibration độc lập;
4. áp dụng minimal typed projection lên consequent, với conflict priority xác định.

Mọi đối chứng online dùng cùng graph, cùng initial noise và cùng actuator. Khác biệt chỉ là static/fixed/dynamic activation schedule. Post-repair dùng bounded fixed-point passes để tạo đối chứng đủ mạnh.

Các script chính:

- `scripts/state_conditioned_graph_guidance.py`;
- `scripts/unify_verified_specification_graph.py`;
- `scripts/aggregate_state_graph_guidance.py`.

## 12. Unified graph discovery

Vòng foundation-model discovery đã được sửa thành discovery/calibration/test, loại target/sensitive edges bằng code và loại same-column/vacuous implications. Kết quả cho thấy vai trò của LLM không đồng đều:

| Dataset | Unified edges | Hard | Near-hard | Soft | Foundation-model edges |
| --- | ---: | ---: | ---: | ---: | ---: |
| Adult | 6 | 4 | 2 | 0 | 0 |
| Default | 32 | 2 | 11 | 19 | 2 |
| Shoppers | 32 | 7 | 9 | 16 | 3 |

Adult là negative result quan trọng: sau khi loại 19 same-column tautologies, 3 sensitive relations, 1 vacuous relation và các weak relations, không còn LLM-only edge nonvacuous nào. Vì vậy chưa có bằng chứng rằng foundation-model knowledge luôn bổ sung tốt hơn empirical discovery. Method phải cho phép abstain và giữ provenance, không ép LLM thành nguồn chính.

## 13. Injection results mới

Thí nghiệm dùng unified graph, 2.048 paired trajectories, 50 steps và official 8.000-epoch TabDiff checkpoints. Metric `hard/near violation` đo tỷ lệ vi phạm; `calibration error` đo sai lệch violation rate so với heldout thật; `conditional distance` đo KS/TV của consequent trong cohort mà antecedent đúng.

| Dataset | Method | Hard/near violation ↓ | Calibration error ↓ | Conditional distance ↓ | Numeric KS ↓ | Categorical TV ↓ | Actions |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Default | Unconditional | 0.0433 | 0.0230 | 0.0324 | 0.0333 | 0.0251 | 0 |
| Default | Fixed-middle | 0.0067 | 0.0230 | 0.0311 | 0.0317 | 0.0262 | 190 |
| Default | Fixed-late | 0.0055 | 0.0227 | 0.0310 | 0.0332 | 0.0256 | 79 |
| Default | Adaptive dynamic | 0.0063 | 0.0223 | 0.0307 | 0.0332 | 0.0254 | 73 |
| Default | **Calibrated adaptive** | **0.0201** | **0.0163** | **0.0270** | 0.0332 | **0.0249** | **42** |
| Shoppers | Unconditional | 0.0404 | 0.0191 | **0.0422** | **0.0246** | 0.0312 | 0 |
| Shoppers | Fixed-middle | 0.0030 | 0.0357 | 0.0616 | 0.0286 | 0.0330 | 288 |
| Shoppers | Fixed-late | 0.0000 | 0.0373 | 0.0630 | 0.0282 | 0.0327 | 381 |
| Shoppers | Adaptive dynamic | 0.0132 | 0.0300 | 0.0569 | 0.0280 | 0.0328 | 346 |
| Shoppers | **Calibrated adaptive** | **0.0272** | **0.0135** | 0.0433 | 0.0256 | **0.0311** | **120** |

### Diễn giải

- Ép satisfaction về 100% không phải mục tiêu đúng cho near-hard edges: fixed/static methods giảm violation nhưng calibration và conditional fidelity xấu rõ.
- Edge-specific timing giảm action và distortion so với một middle window dùng chung.
- Calibrated actuation chỉ sửa phần violation vượt mức calibration. Trên Default, nó đồng thời cải thiện violation, calibration, conditional fidelity và categorical TV so với unconditional.
- Trên Shoppers, calibrated actuation giảm violation khoảng một phần ba và cải thiện calibration, đổi lại conditional distance và numeric KS tăng nhẹ. Đây là Pareto trade-off chứ chưa phải thắng tuyệt đối.
- Adult đã bão hòa; calibrated method không action. Selective abstention là hành vi mong muốn, nhưng Adult không nên là main effect dataset.

Artifacts chính:

- `outputs/adaptive_graph_schedule_aggregate_1054249/RESULTS.md`;
- `outputs/unified_verified_graph_<dataset>_1054228_*/`;
- `outputs/nonvacuous_llm_graph_<dataset>_1054225_*/`.

## 14. Claim được thu hẹp sau thực nghiệm

Claim không nên là “middle injection luôn tốt hơn late repair” hoặc “LLM graph luôn giàu hơn statistical graph”; cả hai đều bị kết quả hiện tại phản bác.

Claim có tiềm năng hơn là:

> Một verified graph có typed hardness, kết hợp edge-specific trajectory readability và calibrated violation budget, có thể chọn lọc thời điểm và mức độ can thiệp, cải thiện validity–calibration trade-off so với static/fixed schedules vốn over-enforce probabilistic relations.

Phần foundation-model nên được định vị là một proposal source có khả năng bổ sung external relations, nhưng phải qua cùng nonvacuity, calibration và provenance audit. Contribution chính vẫn là graph instantiation và actuation trên trajectory.
# Cập nhật 2026-10-05: tách knowledge acquisition khỏi dynamic activation

Protocol sinh graph đã được nâng thành pipeline nhiều giai đoạn: external/domain proposal, discovery-only empirical mining, predefined hypotheses, typed compilation, counterexample-guided refinement, critic, semantic judge, blind calibration và frozen heldout audit. DSL mới hỗ trợ literal constraint, column-to-column comparison, conjunction và exception group. Báo cáo chi tiết, bảng kết quả đầu tiên và job đang chạy nằm tại [KNOWLEDGE_GRAPH_PROTOCOL_20261005_VI.md](KNOWLEDGE_GRAPH_PROTOCOL_20261005_VI.md).

Kết quả một-vòng hiện chưa chứng minh multi-stage tốt hơn single-pass vì baseline cũ chưa qua cùng semantic judge. Batch v3 đã sửa confound này. Contribution chính vẫn là dynamic graph activation theo diffusion state; knowledge acquisition chỉ được nâng thành contribution phụ nếu judged baseline comparison và cross-dataset audit ủng hộ.
