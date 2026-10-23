# Protocol sinh knowledge graph trước dynamic activation

Ngày cập nhật: 2026-10-05

## 1. Quyết định tạm thời

Tách dự án thành hai tầng:

1. **Knowledge acquisition:** sinh, biên dịch, kiểm chứng, phân loại và đóng băng một graph có provenance rõ ràng.
2. **Dynamic activation:** ở mỗi row và diffusion state, quyết định edge nào đủ applicable, readable và safe để active; sau đó chọn action và timestep can thiệp.

Đợt này tập trung vào tầng 1. Đây là lựa chọn hợp lý vì nếu graph đầu vào chứa quan hệ giả, dynamic activation tốt đến đâu cũng chỉ thực thi tri thức sai. Tuy nhiên, protocol sinh graph chủ yếu là hạ tầng và ablation; contribution chính của paper vẫn nên là **state-conditioned activation/intervention trên diffusion trajectory**, không nên claim riêng việc dùng LLM để khám phá và kiểm chứng constraint là mới.

## 2. Protocol hiện được implement

### 2.1. Nguồn tri thức

- **External/domain knowledge:** foundation model đọc dataset context, data dictionary, schema và giá trị categorical công khai.
- **Empirical structure:** code miner chỉ dùng `discovery split` để tìm directional implications; không import candidate đã fit trên toàn bộ train.
- **Predefined hypotheses:** một thư viện nhỏ, có provenance rõ, dùng như hypothesis chứ không được mặc định là đúng.
- **Directionality/exception role:** một agent riêng tập trung vào chiều quan hệ, conjunction và deactivation condition.

### 2.2. Vòng kiểm chứng

1. Chia train thành `60% discovery / 20% counterexample / 20% blind calibration`.
2. Ba vai trò đề xuất edge độc lập.
3. Biên dịch về typed DSL và chạy deterministic checks: kiểu dữ liệu, cột lạ, target leakage, sensitive attribute, tautology, support và vacuity.
4. Đánh giá trên counterexample partition; chỉ gửi aggregate counts cho vòng refinement thứ hai.
5. Critic có thể `keep / revise / reject` từng edge.
6. Semantic judge độc lập gán scope: `domain_general`, `dataset_semantic`, `dataset_empirical` hoặc reject.
7. Blind calibration quyết định empirical hardness: `hard / near_hard / soft`.
8. Chỉ sau khi graph đã frozen mới mở heldout để đo stability; heldout không tham gia selection.

Không raw row nào được gửi tới API. Payload chỉ gồm schema, data dictionary, frequent categories và aggregate statistics.

### 2.3. Typed graph DSL v2

Mỗi hyperedge gồm:

- `applicability`: conjunction có thể rỗng nếu constraint áp dụng toàn cục;
- `requirements`: một hoặc nhiều atom phải đồng thời thỏa;
- `exceptions`: OR giữa các exception group, AND trong mỗi group;
- atom có thể so sánh `column–literal` hoặc `column–column`;
- `relation_family`, `hardness_prior`, `evidence_basis`, rationale, directionality, counterexample và confidence;
- provenance qua proposal role, critic, judge và data split metrics.

DSL mới diễn tả được cả:

- `Administrative == 0 -> Administrative_Duration == 0`;
- `BounceRates <= ExitRates`;
- conjunction và exception group;
- temporal/accounting dependencies;
- representation/codebook dependencies như `education -> education.num`.

## 3. Kết quả một vòng đã hoàn tất

Job `1055674`, aggregate `1055675`. Đây là DSL cũ chỉ hỗ trợ column–literal; kết quả dùng để chẩn đoán protocol, không phải bảng cuối cùng.

| Dataset | Graph | Proposed | Retained | Hard | Near-hard | Soft | Families | Heldout stable ≤ 0.05 | Mean shift |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Adult | Strong single-pass | 23 | 19 | 16 | 1 | 2 | 3 | 0.947 | 0.0081 |
| Adult | Multi-stage LLM | 30 | 1 | 0 | 0 | 1 | 1 | 1.000 | 0.0064 |
| Adult | Empirical only | 12 | 10 | 6 | 1 | 3 | 1 | 0.900 | 0.0176 |
| Default | Strong single-pass | 19 | 14 | 5 | 5 | 4 | 3 | 1.000 | 0.0057 |
| Default | Multi-stage LLM | 28 | 9 | 0 | 0 | 9 | 3 | 0.778 | 0.0238 |
| Default | Empirical only | 30 | 30 | 0 | 11 | 19 | 1 | 0.900 | 0.0234 |
| Shoppers | Strong single-pass | 10 | 9 | 3 | 3 | 3 | 4 | 1.000 | 0.0134 |
| Shoppers | Multi-stage LLM | 19 | 5 | 0 | 3 | 2 | 3 | 1.000 | 0.0097 |
| Shoppers | Empirical only | 30 | 28 | 5 | 9 | 14 | 1 | 1.000 | 0.0117 |

Chi phí: Adult `58,881` tokens/6 calls; Default `103,746` tokens/8 calls; Shoppers `62,132` tokens/6 calls.

### Cách đọc đúng

- `Retained/proposed` là **protocol yield**, không phải precision vì chưa có semantic gold graph.
- Single-pass giữ nhiều edge hơn nhưng ở run này chưa qua cùng semantic judge, nên chưa thể kết luận nó tốt hơn protocol.
- Adult single-pass tìm được 15 ánh xạ `education -> education.num` có zero observed violation trên calibration và heldout. Protocol cũ loại chúng vì coi codebook là artifact. Audit cho thấy filter này quá mạnh: dependency có ý nghĩa ở scope `dataset_semantic` cần được giữ, còn tương quan giữa identifier tùy ý mới phải loại.
- Default protocol bổ sung 4 edge không có trong single-pass và 9 edge không trùng empirical graph, nhưng 2/9 edge có heldout shift lớn hơn 0.05; đây là failure signal thực, không nên che đi.
- Shoppers protocol giữ các process relation đúng chiều và ổn định, đồng thời loại một số correlation như `SpecialDay -> Month` hoặc OS–Browser mà single-pass dễ giữ lại.

### Tín hiệu sớm từ vòng refinement thứ hai

Adult và Shoppers của job `1055696` đã hoàn tất khi cập nhật file:

| Dataset | Một vòng: retained / mean shift | Hai vòng: retained / mean shift | Nhận xét tạm thời |
| --- | ---: | ---: | --- |
| Adult | 1 / 0.0064 | 5 / 0.0346 | Coverage tăng nhưng có một edge shift 0.099; chưa cho thấy loop đáng dùng |
| Shoppers | 5 / 0.0097 | 8 / 0.0021 | Thêm process edges và stability tốt hơn; 8/8 edge có shift ≤ 0.05 |

Đây chưa phải paired ablation hoàn toàn sạch vì API generation có thể khác giữa hai run. Vì kết quả khác nhau theo dataset, không nên claim refinement loop luôn tốt. Hypothesis hợp lý hơn là loop hữu ích khi schema có process semantics rõ (page count–duration), còn với Adult nó dễ sinh association nghề nghiệp mang tính dataset-specific.

## 4. Các hypothesis đang kiểm tra

### H-KG1 — Multi-stage verification tốt hơn single-pass

Sau khi cả hai cùng qua critic, semantic judge và blind calibration, protocol nhiều vai trò và refinement loop phải có:

- semantic acceptance cao hơn;
- ít spurious/codebook-ID correlations hơn;
- heldout shift không xấu hơn;
- coverage theo relation family và endpoint không chỉ tăng nhờ duplicate/paraphrase.

Batch v3 dùng baseline đã được kiểm tra công bằng.

### H-KG2 — External knowledge có incremental value ngoài empirical mining

Đo số edge `external_domain/hybrid` ổn định trên heldout nhưng không trùng empirical-only graph. Nếu phần lớn edge LLM chỉ diễn đạt lại train implication, LLM chưa có giá trị rõ ràng ở knowledge acquisition.

### H-KG3 — Counterexample-guided loop bổ sung edge có ích

So sánh một vòng và hai vòng theo unique canonical edge, family/endpoint coverage, semantic acceptance và heldout shift. Nếu vòng hai chỉ tạo paraphrase hoặc edge yếu hơn thì bỏ loop, không dùng số proposal lớn làm contribution.

### H-KG4 — Rich typed DSL cần thiết

DSL v2 phải thu được relation family mà DSL cũ không diễn tả được, đặc biệt column comparison, process, temporal/accounting và representation dependency. Đo cả số edge mới và chất lượng heldout, không chỉ expressiveness.

### H-KG5 — Scope calibration quan trọng hơn hard/soft label đơn lẻ

Phân biệt `domain_general`, `dataset_semantic`, `dataset_empirical` để tránh hai lỗi đối nghịch: loại nhầm codebook hợp lệ và nâng correlation ổn định thành domain invariant.

## 5. Metrics của bảng cuối

| Nhóm | Metric | Ý nghĩa |
| --- | --- | --- |
| Yield | retained/proposed | Tỷ lệ candidate sống qua protocol; không gọi là precision |
| Semantic | judge acceptance, scope distribution | Chất lượng và phạm vi claim của edge |
| Statistical | support, violation rate, Wilson interval | Mức empirical fit và uncertainty |
| Generalization | heldout violation shift, stable-within-0.05 | Edge có giữ hành vi ngoài calibration hay không |
| Incremental value | unique vs judged single-pass; non-overlap empirical | LLM/protocol thêm gì ngoài baseline |
| Diversity | relation families, endpoint patterns | Tránh graph lớn giả do paraphrase |
| Safety | target/sensitive rejection, exception coverage | Ngăn target leakage và normative rule |
| Cost | calls, tokens, latency | Chi phí để sinh/freeze graph |

Để claim mạnh hơn, vẫn cần một semantic gold subset được gán nhãn độc lập bởi người có hiểu dataset/domain. Heldout statistics một mình không chứng minh semantic truth.

## 6. Jobs đã submit

| Mục đích | Job | Trạng thái lúc viết |
| --- | --- | --- |
| Model/schema smoke v1 | `1055673` | Completed |
| Protocol một vòng, DSL cũ | `1055674`, aggregate `1055675` | Completed |
| Counterexample loop hai vòng, DSL cũ | `1055696`, aggregate `1055697` | Completed |
| Rich typed DSL v2 | `1055708`, aggregate `1055709` | Failed do API credit; aggregate đã hủy |
| V3: rich DSL + judged baseline công bằng + codebook scope | `1055713`, aggregate `1055714` | Failed do API credit; aggregate đã hủy |

### Budgeted rerun sau khi bổ sung giới hạn chi phí

Các batch `1055708` và `1055713` sau đó thất bại vì API credit tạm thời cạn; dependent aggregates đã được hủy thay vì để ở trạng thái `DependencyNeverSatisfied`. Đợt chạy thay thế chỉ dùng Shoppers:

- main run: `1056016`;
- hypothesis audit: `1056017`;
- token budget cứng: `165,000`;
- tối đa `4,500` output tokens mỗi API call;
- tối đa 7 candidates mỗi role, 2 refinement rounds;
- `gpt-4.1-mini` cho empirical proposal/critic;
- `gpt-5.4` cho external proposal và semantic judge;
- response, usage và error được checkpoint sau từng API call.

Một run duy nhất kiểm tra: multi-stage so với judged single-pass, giá trị riêng của round 2, rich typed DSL và external knowledge ngoài empirical mining. Không chạy thêm dataset hoặc seed.

Model thực tế được dùng: `gpt-4.1-mini` cho empirical proposal/critic và `gpt-5.4` cho external proposal/semantic judge. Run đầu dừng ở semantic judge vì response rỗng sau reasoning; sau khi bổ sung cache/resume, phân loại lỗi structured output và tách batch judge, job `1056060` đã tiếp tục từ 10 stage đã hoàn tất thay vì gọi lại API. Tổng chi phí cuối là **110,513/165,000 tokens**, 15 API calls.

### Kết quả budgeted run đã hoàn tất

| Graph | Proposed | Retained | Hard | Near-hard | Soft | Families | Endpoint patterns | Stable heldout ≤ 0.05 | Mean shift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Judged single-pass | 14 | 1 | 0 | 1 | 0 | 1 | 1 | 1 | 0.0017 |
| Multi-stage LLM | 46 | 8 | 5 | 3 | 0 | 3 | 6 | 8 | 0.0013 |
| Empirical-only | 30 | 28 | 5 | 9 | 14 | 1 | — | 28 | 0.0117 |
| Unified | 76 | 36 | 10 | 12 | 14 | — | — | 36 | 0.0094 |

Multi-stage graph bổ sung 7 stable unique edges so với judged single-pass, 2 stable edges từ refinement round 2, 2 column-comparison edges, và 2 external/hybrid edges không có trong empirical graph. Tuy nhiên, đây chưa phải bằng chứng rằng protocol thắng một strong single-pass về semantic quality: compiler của single-pass chỉ biên dịch đúng 1/14 mô tả, chủ yếu vì model ghi antecedent mà thiếu consequent trong `requirements`. Bảng này vì vậy chứng minh pipeline sinh được typed graph đa dạng và ổn định, nhưng baseline compiler cần sửa trước khi claim superiority.

Audit edge-level cũng cho thấy **source graph không thể đưa thẳng vào sampler**. Trong 8 edges có duplicate, threshold bị subsume, reverse action không hợp lý và một association ổn định nhưng không phải constraint (`SpecialDay > 0 -> Month = May`). Vì vậy graph dùng để verify được giữ rộng, còn graph dùng để actuation phải nhỏ hơn và có action semantics rõ.

## 7. Điều kiện quyết định trước khi sang dynamic activation

Chuyển trọng tâm sang dynamic activation khi v3 đạt cả ba điều kiện:

1. có một core graph nhỏ nhưng có semantic scope rõ, không chỉ là correlation catalog;
2. có incremental external/domain edges ngoài empirical-only baseline;
3. judged multi-stage protocol ít spurious hơn judged single-pass mà heldout stability không giảm đáng kể.

Nếu H-KG1 không được ủng hộ, dùng protocol đơn giản hơn: strong model một lần + deterministic verifier + independent judge. Contribution paper vẫn có thể nằm ở dynamic activation; không cần ép knowledge acquisition thành contribution nếu ablation không ủng hộ.

## 8. Kiểm định graph trước khi active theo denoising state

Ba job không dùng API đã hoàn tất:

| Tầng kiểm định | Job | Kết quả |
| --- | --- | --- |
| Lọc verifier graph thành actuation graph | `1056064` | 8 source edges → 7 verifier edges → **3 active edges** |
| Đọc trạng thái cạnh trên trajectory | `1056065` | Cả 3 cạnh có cutoff thỏa precision ≥ 0.90 và recall ≥ 0.80 |
| Điều kiện an toàn trước can thiệp | `1056066` | **Qualified = true**, đạt 6/6 tiêu chí |

Actuation graph gồm:

| Edge | Loại | Cutoff sớm nhất | Synthetic / heldout violation |
| --- | --- | ---: | ---: |
| `Administrative == 0 -> Administrative_Duration == 0` | Hard process | 0.571 | 0.0031 / 0.0000 |
| `Informational == 0 -> Informational_Duration == 0` | Hard process | 0.612 | 0.0043 / 0.0000 |
| `BounceRates <= ExitRates` | Near-hard column comparison | 0.286 | 0.0220 / 0.0154 |

`Cutoff` là reverse-time lớn nhất nơi state ước lượng đã dự báo active set cuối với precision/recall yêu cầu. Kết quả bác bỏ một global middle window duy nhất: hai process edges đã đủ đọc được ở nửa giữa sớm, nhưng column-comparison edge chỉ đáng tin muộn hơn. Đây là cơ sở thực nghiệm trực tiếp cho **edge-specific dynamic activation schedule**.

### Kết quả active-edge guidance

Job chính `1056090`, 4,096 samples, 50 denoising steps, cùng checkpoint và initial noise:

| Method | Macro violation ↓ | Heldout error ↓ | Numeric KS ↓ | Drift SD ↓ | Action events ↓ | Unique rows ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unconditional | 0.0091 | 0.0042 | **0.0259** | **0.0000** | **0** | **0** |
| Fixed middle | **0.0011** | 0.0046 | 0.0263 | 0.0028 | 377 | 120 |
| Edge-specific schedule, không state gate | 0.0013 | 0.0044 | 0.0266 | 0.0036 | 279 | 126 |
| State-active | 0.0013 | 0.0047 | 0.0262 | 0.0016 | 224 | 94 |
| Calibrated state-active | 0.0055 | 0.0006 | **0.0262** | 0.0006 | 29 | 29 |
| Final repair | **0.0000** | 0.0057 | 0.0262 | 0.0012 | 93 | 93 |
| Calibrated final repair | 0.0053 | **0.0004** | 0.0263 | **0.0003** | **28** | **28** |

`state_active` chỉ bật từng edge sau cutoff của chính nó, khi applicability tồn tại qua hai state liên tiếp và các cột liên quan đã đủ ổn định. So với edge-specific schedule có cùng cutoff nhưng không dùng state gate, nó giữ macro violation tương đương, giảm **19.7% action events**, **25.4% số row bị tác động**, và giảm **55.6% drift**. Đây là bằng chứng cơ chế rằng state-dependent active set tránh được can thiệp thừa.

Tuy nhiên, calibrated final repair hiện tốt hơn calibrated state-active về heldout error, drift và số row/action, trong khi KS chỉ chênh 0.0001. Lý do là hai hard edges có một sửa chữa hiển nhiên: khi count bằng 0 thì đặt duration bằng 0; còn near-hard `BounceRates <= ExitRates` vốn đã có violation rate gần dữ liệu thật nên calibrated methods không cần sửa. Với graph đơn giản này, sampler không có bài toán action selection đủ khó để thể hiện lợi ích của mid-step injection.

Kết luận hiện tại:

- **Pipeline hoạt động ổn định và dynamic activation đã được implement/verify.**
- **State gate có lợi so với static timing nếu buộc phải guidance trong sampler.**
- **Chưa có bằng chứng rằng guidance trong sampler cần thiết hơn calibrated post-repair.** Claim mạnh về middle-step injection vì vậy chưa được ủng hộ ở Shoppers graph hiện tại.
- Thí nghiệm tiếp theo chỉ đáng làm nếu graph có multi-column/multi-action constraints, nơi final repair không biết nên đổi cột nào hoặc gây joint-distribution damage; không nên chỉ tăng seeds cho ba edges hiện tại.

Một giới hạn nữa là cả ba active edges đều bắt nguồn từ predefined seeds rồi được LLM critic/judge kiểm chứng. Các LLM-discovered edges còn lại hữu ích cho verifier nhưng không an toàn để actuation. Do đó kết quả này chứng minh dynamic graph mechanism, chưa chứng minh LLM external knowledge acquisition tạo ra actuator tốt hơn rule library.

Đây vẫn là exploratory result: curation/qualification hiện có một heldout-stability guard. Không edge nào trong active graph được giữ hay loại chỉ vì guard này, nhưng paper run phải chọn graph hoàn toàn bằng discovery/counterexample/calibration split và để test split bất khả xâm phạm đến lần đánh giá cuối.

## 9. File tái chạy và kết quả

- Main protocol: `scripts/knowledge_graph_protocol.py`
- Model/schema smoke: `scripts/model_structured_output_smoke.py`
- Aggregate audit: `scripts/aggregate_knowledge_graph_protocol.py`
- Slurm: `slurm/178_model_access_smoke.sbatch`, `179_knowledge_graph_protocol.sbatch`, `180_aggregate_knowledge_graph_protocol.sbatch`
- Kết quả một vòng: `outputs/knowledge_graph_protocol_aggregate_1055675/RESULTS.md`
- Budgeted graph: `outputs/shoppers_budgeted_graph_1056016/RESULTS.md`
- Hypothesis audit: `outputs/shoppers_budgeted_graph_audit_1056061/RESULTS.md`
- Actuation curation: `outputs/typed_graph_curation_1056064/RESULTS.md`
- Trajectory probe: `outputs/typed_graph_trajectory_1056065/RESULTS.md`
- Qualification: `outputs/typed_graph_qualification_1056066/RESULTS.md`
- Dynamic actuation implementation: `scripts/typed_dynamic_graph_guidance.py`
- Matched active-edge comparison: `outputs/typed_dynamic_graph_guidance_1056090/RESULTS.md`
- Per-edge metrics: `outputs/typed_dynamic_graph_guidance_1056090/edge_metrics.csv`
