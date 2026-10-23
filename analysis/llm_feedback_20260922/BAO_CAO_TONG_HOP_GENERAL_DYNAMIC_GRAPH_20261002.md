# Báo cáo tổng hợp: constrained tabular synthesis với dynamic constraint graph

Ngày cập nhật: 02/10/2026

## 1. Kết luận ngắn

Dự án hiện **không còn được định nghĩa là “sinh dữ liệu theo semantic request của người dùng”**. Semantic request chỉ là một nguồn cung cấp constraint. Bài toán rộng hơn là:

> Sinh bảng mixed-type bằng diffusion dưới nhiều loại constraint, trong đó hệ thống phải xác định constraint nào đáng tin, constraint nào áp dụng cho từng bản ghi và từng thời điểm, nên tác động vào phần nào của bản ghi, bằng solver nào, rồi chứng nhận đầu ra bằng verifier độc lập.

Tên working formulation phù hợp nhất là **verified dynamic constraint graph for constrained tabular synthesis**.

Nên triển khai benchmark theo hai tầng:

1. **Tầng chung, tương thích với prior work:** constraint toàn cục, implication, range, category, conjunction/disjunction và statistical constraint trên các chương trình công khai của CuTS, Harpoon và LCATS. Tầng này trả lời câu hỏi hệ thống có cạnh tranh được với baseline có sẵn hay không.
2. **Tầng dynamic:** applicability theo row/subgroup, valid exception, graph thay đổi theo diffusion timestep hoặc distribution regime, ambiguity về endpoint và solver. Tầng này kiểm tra phần đóng góp riêng mà baseline tĩnh chưa giải quyết đầy đủ.

Có thể xây bài theo setting chung. Thực tế, đây là hướng an toàn hơn so với chỉ dùng benchmark tự đề xuất. Tuy nhiên, nếu chỉ giữ tầng constraint tĩnh/toàn cục thì novelty sẽ bị CuTS, Harpoon, CTDF và LCATS bao phủ. Hai tầng phải cùng tồn tại: tầng chung tạo tính hợp lệ của so sánh; tầng dynamic tạo research gap.

Trạng thái hiện tại:

- đã có evidence rõ rằng constraint applicability, reliability và solver tốt nhất thay đổi theo row, timestep, objective và regime;
- đã có evidence rằng mid-denoising có lợi cho joint fidelity và preservation, nhưng không tốt nhất cho target marginal fidelity;
- đã bác bỏ việc dùng LLM đánh giá trực tiếp từng row như thiết kế mặc định;
- đã chạy CuTS và LCATS official; CuTS German và Harpoon official vừa được submit;
- chưa đủ để tuyên bố state of the art hoặc chốt bài ICDE, vì chưa có một phương pháp dynamic hoàn chỉnh vượt strong official baselines trên ít nhất ba dataset dưới cùng protocol.

## 2. Bài toán nghiên cứu

### 2.1 Input và output

Input tổng quát gồm:

- một tabular generator, ưu tiên diffusion;
- schema và metadata;
- dữ liệu train dùng cho discovery/calibration;
- một tập constraint có thể đến từ user request, domain knowledge, metadata, statistical discovery hoặc foundation model;
- objective như validity, joint fidelity, subgroup preservation, utility hoặc cost.

Output gồm:

- bảng synthetic;
- trace cho biết constraint nào đã active, khi nào và solver nào đã can thiệp;
- independent certification về validity, fidelity, exception preservation, utility, diversity và cost.

### 2.2 Đơn vị constraint chuẩn hóa

Mỗi constraint được biểu diễn thành một typed program:

`constraint = (predicate, direction, applicability, exception, hardness, scope, provenance, verifier)`.

Ví dụ:

- global range: `35 < age < 55`;
- implication: `Administrative = 0 -> Administrative_Duration = 0`;
- subgroup: chỉ áp dụng relation cho một cohort;
- exception-aware implication: implication đúng với phần lớn row nhưng có một nhóm ngoại lệ hợp lệ;
- statistical constraint: trung bình, tỷ lệ hoặc correlation ở mức dataset;
- policy shift: edge đúng trong giai đoạn đầu nhưng không còn an toàn ở regime sau.

### 2.3 Dynamic nghĩa là gì?

Graph không được gọi là dynamic chỉ vì gradient thay đổi theo `x_t`. Harpoon đã có loss cố định nhưng gradient phụ thuộc state; SCTab-Diff có prior tương đối tĩnh nhưng score response vẫn phụ thuộc row context. Dynamic trong dự án này phải chỉ một trong các thay đổi có nội dung sau:

- **row dynamic:** edge áp dụng cho row này nhưng không áp dụng cho row khác;
- **subgroup dynamic:** cùng relation có hardness hoặc exception khác nhau giữa các nhóm;
- **trajectory dynamic:** độ tin cậy của applicability thay đổi theo `x_hat_0(x_t)`;
- **regime dynamic:** support/reliability của edge đổi theo time hoặc distribution shift;
- **objective dynamic:** stage, scope và solver tốt nhất đổi theo metric profile hoặc budget.

## 3. Động lực

### 3.1 Constraint tĩnh dễ xóa mode hợp lệ

Audit trên Adult, Default và Shoppers sinh 788 candidate relations, nhưng chỉ 80 relation không có observed exception trong cả hai split. Có 334 relation mang tính probabilistic và 132 relation thiếu support. Nếu coi toàn bộ relation khai phá được là hard constraint, hệ thống sẽ sửa cả những ngoại lệ hợp lệ và làm biến dạng phân phối.

Ví dụ rõ nhất:

- `Own-child -> age < 30` trên Adult có khoảng 19,7% exception ở cả train và test; không thể hard-enforce.
- `Administrative = 0 -> Administrative_Duration = 0` trên Shoppers không có observed exception.
- chiều ngược `Administrative_Duration = 0 -> Administrative = 0` có 2,2–3,1% exception.
- `ProductRelated_Duration = 0 -> ProductRelated = 0` sai tới khoảng 92–95%.

Cùng một cặp cột nhưng direction khác nhau dẫn tới semantics hoàn toàn khác. Đây là motivation trực tiếp cho graph có direction, applicability, support và exception.

### 3.2 Constraint không xác định duy nhất hành động

Biết một row vi phạm không cho biết nên sửa cột nào. Với `A -> B`, có thể:

- làm antecedent sai;
- sửa consequent;
- resample một vùng con;
- giữ row và abstain nếu uncertainty cao.

Các lựa chọn này đều có thể đạt final validity nhưng tạo counterfactual distortion rất khác nhau. Vì vậy graph cần một action layer, không chỉ một verifier.

### 3.3 Thời điểm can thiệp có trade-off thật

Ở đầu reverse process, state còn dễ thay đổi nhưng khó đọc. Gần cuối, state đã đọc được nhưng ít controllable hơn. Trajectory probe trên Default cho thấy edge-scope precision/recall tăng từ `0,669/0,242` ở đầu trajectory lên `0,922/0,931` ở giữa và gần `1,0` ở cuối. Một lịch gọi cố định không thể tối ưu đồng thời readability và controllability.

### 3.4 Benchmark tự đề xuất là chưa đủ

Nếu chỉ báo cáo subgroup hoặc exception bank do chính dự án tạo, reviewer có thể cho rằng task được thiết kế để khớp method. Vì vậy public constraints từ CuTS/Harpoon/LCATS phải là tầng đánh giá bắt buộc. Dynamic benchmark chỉ bổ sung trục còn thiếu, không thay thế benchmark chung.

## 4. Khoảng trống so với related work

| Nhóm công trình | Phần đã được giải quyết | Điều không được phép claim | Khoảng trống còn hợp lý |
| --- | --- | --- | --- |
| TabSyn, TabDiff | Backbone diffusion cho mixed-type table | Một backbone tabular diffusion mới nếu chỉ sửa sampler | Dùng trajectory và representation để nghiên cứu graph activation |
| Harpoon | Differentiable guidance trong reverse process | Đầu tiên đưa constraint/loss vào denoising | Selective applicability, exception, solver/stage routing và independent certification |
| CTDF | Feasibility operator ở mỗi reverse step cho numeric và categorical | Đầu tiên enforce hard mixed-type constraints per-step | Dynamic active graph và matched comparison với final projection |
| SCTab-Diff | Prior do LLM trích xuất condition reverse denoising | Đầu tiên dùng LLM prior trong diffusion | Constraint set thay đổi theo [**SAGE - ACL 2026 main.**](https://aclanthology.org/2026.acl-long.174/)row/regime/objective, có abstention và verification |
| CuTS | Training/fine-tuning theo executable constraint programs | Đầu tiên làm programmable constrained synthesis | Training-free hoặc low-cost control; dynamic applicability và graph lifecycle |
| LCATS | Constraint discovery, TabSyn generation, diagnosis và post-generation alignment | Đầu tiên dùng LLM để discover/repair constraint | Khi nào repair là chưa đủ; preservation và online dynamic activation |
| Zhao et al., EVS | LLM tạo executable rule, kiểm chứng và chọn rule | Compiler + validation tự nó là novelty | Cross-schema semantics, dynamic activation và generation-time action |
| PromptLoop, PPAD, Free²Guide, TTGen | Foundation model xem intermediate preview rồi feedback, rollback hoặc chọn nhánh | LLM-in-the-loop nói chung là mới | Typed mixed-type graph, applicability, exception và certified action |

Ranh giới claim chặt nhất là:

> Không claim “feedback phụ thuộc state” hoặc “constraint được inject trong denoising”. Claim mục tiêu là một graph được kiểm chứng, trong đó active constraints, topology, stage và solver được chọn có điều kiện theo row, trajectory, regime và objective.

LLM là optional compiler hoặc semantic transfer module. Nếu parser/statistics làm tốt ngang LLM, method vẫn phải hoạt động và báo đúng kết quả đó.

## 5. Phương pháp chung đề xuất

### 5.1 Bốn lớp graph

1. **Specification graph:** columns, predicates, direction, subgroup, exception và provenance.
2. **Evidence graph:** train support, held-out stability, confidence interval, hardness, sensitive/target flags và temporal history.
3. **Active graph:** subset edge áp dụng cho row `i` tại timestep `t` và regime `r`.
4. **Action graph:** endpoint, descendants, stage, solver, strength và budget được phép can thiệp.

### 5.2 Four-state activation

Mỗi cặp row-edge được phân vào một trong bốn trạng thái:

- applicable và violated;
- applicable và satisfied;
- not applicable;
- valid exception.

Mặc định chỉ trạng thái đầu được tác động. Ba trạng thái còn lại được dùng để đo false activation, collateral edit và exception loss.

### 5.3 Controller

Controller nhận:

- graph features: type, arity, direction, support, feasible mass, exception rate;
- state features: preview uncertainty, violation margin, flip history;
- objective: target fidelity, joint fidelity, intervention cost, runtime;
- candidate actions: projection, gradient guidance, conditional denoising, imputation, best-of-N, final repair hoặc abstain.

Controller trả về:

- active edges;
- target scope;
- stage;
- solver và strength;
- rollback/certification decision.

Thiết kế ban đầu không cần một neural router lớn. Có thể bắt đầu bằng calibrated rules hoặc contextual bandit trên graph features. Evidence hiện tại cho thấy một shallow router học trực tiếp solver label chưa đủ mạnh.

### 5.4 Foundation model

Foundation model chỉ nên được gọi ở những bước cần semantics khó viết bằng code:

- compile natural-language constraint sang typed DSL;
- map relation roles qua schema mới;
- đề xuất candidate exception/subgroup;
- xử lý mô tả không cấu trúc.

Sau đó type checker, executor và statistical verifier thực hiện phần còn lại. Không nên gọi LLM để phán xét lại mọi row ở mọi timestep.

## 6. Implementation hiện có

### 6.1 Backbone và checkpoint

- Repo chính: TabDiff.
- Checkpoint dùng cho phần lớn probe: `outputs/full_train_1028011/ema_model_epoch_8000.pt`.
- Config: `outputs/full_train_1028011/config.pkl`.
- Checkpoint khoảng 41 MB, dùng sampler đã instrument để lấy `x_hat_0` và can thiệp ở nhiều timestep.
- Các sửa đổi có sẵn trong `tabdiff/main.py` và `tabdiff/trainer.py` được giữ nguyên; đợt analysis không reset hoặc ghi đè chúng.

### 6.2 Dataset và split

- Adult, Default, Shoppers cho mixed-type constraint audit và generation probes.
- Insurance cho LCATS/TabSyn official.
- Beijing, News, Shoppers cho graph lifecycle và distribution shift audit.
- German Credit đang được thêm bằng benchmark CuTS official.
- Discovery chỉ dùng train/early split; held-out/late split dùng cho replication và evaluation.
- Không gửi raw real rows ra API; request probes dùng synthetic-only cases và opaque IDs.

### 6.3 Các thành phần đã implement

- constraint candidate mining và audit đa dataset;
- typed request bank và local deterministic executor;
- direct LLM judgement và compile-to-DSL probes;
- trajectory probe cho applicability/readability;
- subgroup/four-state activation;
- final repair, empirical repair, TabDiff imputation, best-of-8, conditional denoising và lookahead;
- sparse/full descendant scopes;
- solver grid và router audit;
- temporal graph discovery, Wilson filtering và sequential update;
- independent CuTS/LCATS result audit.

Các script quan trọng:

- `scripts/cross_dataset_constraint_audit.py`;
- `scripts/request_condition_probe.py`;
- `scripts/applicability_gate_trajectory.py`;
- `scripts/dynamic_graph_comparison.py`;
- `scripts/cross_relation_solver_router.py`;
- `scripts/intervention_stage_value_audit.py`;
- `scripts/temporal_graph_lifecycle.py`;
- `scripts/temporal_graph_change_detection.py`;
- `scripts/cuts_common_tabdiff_baselines.py`;
- `scripts/audit_lcats_official_outputs.py`.

### 6.4 Baseline labeling

Ba nhãn phải được tách rõ:

- **official:** chạy code tác giả, ví dụ CuTS, LCATS và sweep Harpoon vừa submit;
- **same-backbone proxy:** tái hiện cơ chế trên TabDiff để so matched checkpoint, ví dụ Harpoon-style gradient hoặc CTDF-style projection;
- **our method:** dynamic graph controller.

Proxy không được trình bày là reproduction của paper.

## 7. Giả thuyết đã kiểm tra

### H1 — Intermediate estimate có đủ đọc được không?

**Kết quả: được ủng hộ có điều kiện.**

- Early trajectory: edge precision/recall `0,669/0,242`.
- Mid trajectory: `0,922/0,931`.
- Late trajectory: gần `1,0`.
- Persistence ba preview tăng early precision lên `0,703`, nhưng recall giảm còn `0,204`.

Kết luận: preview giữa/cuối đủ dùng cho verifier; early feedback dễ sai. Persistence chỉ là một calibration knob nhỏ, chưa phải contribution độc lập.

### H2 — Có lỗi mà global/static graph không xử lý an toàn không?

**Kết quả: được ủng hộ.**

- 334/788 candidates là probabilistic; 132 thiếu support.
- Relation direction và subgroup tạo exception rate rất khác nhau.
- Graph frozen dưới temporal shift có 185 unsafe edges trên News và 3 trên Shoppers sau dedup.

Static graph có thể đạt validity trên một split nhưng over-enforce ở subgroup/regime khác.

### H3 — Mid-denoising có tốt hơn post-repair/resampling không?

**Kết quả: chỉ được ủng hộ cho một số objective; giả thuyết mạnh bị bác bỏ.**

Sau khi chỉ giữ output có active satisfaction ít nhất `0,999` và preserve exception:

| Objective profile | Gain của mid-denoising so với best non-mid | Tỷ lệ policy thắng |
| --- | ---: | ---: |
| Balanced | `+0,0118` | `60,0%` |
| Intervention-conservative | `+0,0370` | `76,7%` |
| Joint fidelity | `+0,0550` | `66,7%` |
| Target fidelity | `-0,1004` | `26,7%` |

Mid-denoising có giá trị cho joint structure và giảm can thiệp; final repair/resampling tốt hơn rõ khi chỉ ưu tiên target marginal. Do đó không thể claim feedback trong diffusion luôn tốt hơn repair.

### H4 — LLM có bổ sung gì so với parser/verifier?

**Direct judgement bị bác bỏ; compile-once được ủng hộ như interface.**

Trên 140 synthetic-only cases và 30 same-row/different-request pairs:

| Phương pháp | Accuracy bốn trạng thái | Swap-pair accuracy | Token |
| --- | ---: | ---: | ---: |
| GPT-4.1-mini direct | `0,8643` | `0,8667` | `37.544` |
| GPT-5.4-mini direct | `0,6500` | `0,3667` | `37.699` |
| GPT-4.1-mini compile + execute | `1,0000` | `1,0000` | `1.846` |
| GPT-5.4-mini compile + execute | `1,0000` | `1,0000` | `1.853` |
| Template parser | `1,0000` | `1,0000` | `0` API token |

Compile-to-DSL hiệu quả hơn direct judgement, nhưng template parser cũng hoàn hảo trên grammar kiểm soát. Bằng chứng hiện tại chỉ support LLM cho open-ended hoặc cross-schema specification, không support LLM per-row critic.

Một focused API result F1 khoảng `0,926` đã bị loại khỏi evidence vì payload từng gửi `case_id` làm lộ relation và nhãn clean/corrupt; rare controls còn được truyền candidate `none`. Kết quả đó không được dùng cho claim.

### H5 — Feedback có đáng tin và đáng chi phí không?

**Thiết kế direct LLM bị bác bỏ.**

Compile một lần rồi execute cục bộ dùng khoảng 1/20 số token và ổn định hơn. LLM chỉ nên nằm trên control plane, không nằm trong vòng lặp từng row/timestep, trừ khi có semantic ambiguity mà code không giải được.

### H6 — Text–structured consistency

**Chưa thử và đang được chủ động hoãn.**

Các dataset/backbone chính chưa có long text. Mở nhánh text lúc này sẽ làm benchmark rộng nhưng không giúp kiểm tra claim cốt lõi.

### H7 — Graph-selected topology tốt hơn fixed scope

**Được ủng hộ một phần.**

Sparse top-k descendant scope trên Shoppers giảm một số drift và numeric KS so với full numeric scope, nhưng có trade-off ở target KS. Không có scope duy nhất thắng mọi metric. Điều này support action graph, nhưng chưa đủ cho một learned topology claim.

### H8 — Một solver cố định đủ cho mọi constraint

**Bị bác bỏ về mặt oracle frontier; learned router chưa thành công.**

Grid 30 policies cho thấy không có universal winner. Oracle routing cải thiện normalized rank khoảng `0,073–0,134`. Tuy nhiên leave-one-relation-family-out random forest router chỉ cải thiện:

- balanced: `+0,0118`;
- target fidelity: `+0,0557`;

và giảm nhẹ ở:

- intervention-conservative: `-0,0016`;
- joint fidelity: `-0,0030`.

Exact oracle action accuracy chỉ `0,17–0,33`. Vì vậy solver heterogeneity là thật, nhưng router hiện tại không được dùng làm headline result.

### H9 — Graph cần cập nhật khi regime thay đổi

**Được ủng hộ trên hai trong ba dataset.**

Sequential controller chỉ dùng các window trước:

- News: unsafe exposure `1,000 -> 0,471`, safe retention `1,000`, weighted violation `0,0148 -> 0,0094`.
- Shoppers: unsafe exposure `1,000 -> 0,478`, safe retention `0,938`, weighted violation `0,0171 -> 0,0072`.
- Beijing: không cải thiện vì shift ngắn và không dự đoán được từ window trước.

Hypothesis mạnh “reactive update luôn tốt hơn frozen graph” bị bác bỏ. Gap hợp lý hơn là calibrated lifecycle với abstention/change detection và giới hạn predictability rõ ràng.

### H10 — Cross-dataset/foundation transfer

**Chưa đủ evidence.**

Audit đã chuẩn hóa relation roles qua nhiều table, nhưng chưa có leave-one-table-out graph compiler/controller vượt lexical retrieval hoặc statistical baseline. Đây là nhánh có tiềm năng làm bài rộng hơn, nhưng chưa nên đưa thành claim chính.

## 8. Kết quả baseline đã có

### 8.1 CuTS official trên Adult

CuTS đạt satisfaction `1,0` trên năm chương trình public. Base model có satisfaction từ `0,3256` đến `0,9851`. Total variation thay đổi nhỏ theo metric gốc của CuTS; line constraint có trade-off lớn hơn implication. Đây là official result một training seed và năm sampling repeats, chưa phải multi-seed reproduction.

### 8.2 TabDiff trên cùng CuTS programs

- Implication2/4 đã bão hòa: unconditional satisfaction khoảng `0,994`.
- Implication5: unconditional `0,9448`; repair và best-of-8 đạt `1,0`.
- `sex=Female`: repair đạt `1,0` nhưng categorical TV tăng `0,0155 -> 0,0890`.
- `35<age<55`: repair đạt `1,0` nhưng numeric KS tăng `0,0249 -> 0,0942`.

Public constraints xác nhận validity đơn thuần không đủ; phải đo distribution distortion. Đồng thời chúng cho thấy nhiều hard-rule task dễ bị saturation hoặc direct repair giải quyết.

### 8.3 LCATS official trên Insurance

LCATS đã chạy end-to-end phases 1–5:

- phát hiện 14 constraints;
- 12 là domain bounds/vocabulary;
- 2 là relational/conditional;
- cả 14 có zero violation trên train và held-out test;
- TabSyn đã train/sample;
- diagnostics và LLM reasoning đã chạy;
- phase 5 tạo ba aligned candidates và đưa 99/1.204 rows vào các violation buckets cần alignment.

Independent evaluator đã qua local smoke test trên chính official outputs:

| Output | Any-violation | Numeric KS | Categorical TV | Correlation MAE | TSTR NRMSE | TSTR R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TabSyn original | `0,0822` | `0,0359` | `0,0233` | `0,0141` | `0,5280` | `0,7212` |
| LCATS aligned | `0,0017` | `0,0318` | `0,0233` | `0,0150` | `0,5268` | `0,7224` |

LCATS sửa gần hết lỗi integer ở cột `children`, nhưng hai row vi phạm conditional constraint vẫn còn, nên official output chưa đạt zero violation theo verifier độc lập. Ba aligned candidates có cùng bảng metric này. Exact row edit rate là `1,0` và cell edit rate là `0,2884`, chủ yếu do dịch chuyển nhỏ dưới 0,5 ở các cột số; normalized numeric L1 chỉ `0,00486` và categorical edit bằng 0. Điều này cho thấy binary edit rate phải đi cùng magnitude, nếu không sẽ phóng đại distortion.

Điểm mạnh: baseline official chạy được và cải thiện validity rất mạnh mà fidelity/utility chỉ đổi nhỏ. Giới hạn: constraint set chủ yếu là schema invariants và zero-violation trên real data, benchmark dễ bão hòa; alignment vẫn để lại residual conditional violations và sửa nhẹ rất nhiều numeric cells. Job `1052790` đang chạy để lưu evaluator output vào artifact bền vững.

### 8.4 Baseline mới đã submit

| Job | Nội dung | Trạng thái khi viết báo cáo |
| --- | --- | --- |
| `1052787` | Tạo split German Credit cho CuTS | Hoàn tất |
| `1052788` | CuTS official German baseline mode | Đang chạy |
| `1052789` | CuTS official German constrained mode | Chờ baseline |
| `1052790` | Independent audit của LCATS official outputs | Đã submit |
| `1052791` | Môi trường biệt lập cho Harpoon official | Đã submit |
| `1052792` | Train official Harpoon/RePaint backbone trên Adult | Chờ setup |
| `1052793` | Harpoon và RePaint trên range/category/both/or | Chờ training |
| `1052794` | Chuẩn hóa CuTS German arrays thành bảng metric | Chờ CuTS constrained |
| `1052795` | Trích các official Harpoon/RePaint runs thành bảng | Chờ sweep |

Harpoon chạy trên snapshot riêng dưới `baseline_runs/harpoon_general_adult_20261002`, không sửa result cũ. CuTS German dùng code và constraint programs của tác giả. Không có proxy nào được gọi là official.

## 9. Benchmark tổng quát đề xuất

### 9.1 Tầng A — Public static constraints

Mục tiêu: compatibility và baseline credibility.

- CuTS Adult và German: implication, line, statistical/fairness programs.
- Harpoon Adult; sau đó Default/Shoppers nếu preprocessing/checkpoint chạy ổn: range, category, conjunction và disjunction.
- LCATS Insurance: discovery + TabSyn + post-repair.
- TabDiff unconditional, final repair, best-of-N và same-backbone gradient/projection.

Tầng này không nhằm chứng minh dynamic graph; nó kiểm tra method không thua các task chuẩn khi dynamic components bị tắt.

### 9.2 Tầng B — Dynamic applicability and action

Mục tiêu: kiểm tra contribution riêng.

- bốn trạng thái violated/satisfied/not-applicable/exception;
- same-row/different-constraint pairs;
- subgroup-specific hardness;
- ambiguous endpoint;
- changing regimes;
- multiple objectives và matched action budget.

Nên dùng relation families lấy từ dataset/public programs, rồi thêm applicability/exception split có train-only discovery. Như vậy benchmark không hoàn toàn được tạo từ đầu.

### 9.3 Tầng C — Cross-table foundation generalization

Mục tiêu: kiểm tra foundation-model role.

- hold out một dataset hoặc relation family;
- map schema description sang typed relation roles;
- local verifier được phép calibration, semantic encoder không được train lại;
- so template parser, lexical/embedding retrieval, statistical graph, LLM compiler và oracle DSL.

Tầng C chỉ nên trở thành main claim nếu có transfer thật; nếu không, giữ như analysis/ablation.

### 9.4 Protocol chung

Mọi method nhận cùng:

- train/test split;
- constraint program và verifier;
- số row yêu cầu;
- random seeds;
- generator checkpoint khi có thể;
- action/sampling budget được ghi rõ.

Mọi method trả:

- synthetic table;
- active constraint trace nếu có;
- runtime, GPU time, sampling passes, LLM calls/tokens;
- edit/rollback count.

## 10. Metrics

### 10.1 Validity

- row-level any-violation rate;
- per-constraint violation rate;
- applicable-row satisfaction;
- antecedent evasion;
- independent verifier score.

### 10.2 Preservation

- false action trên satisfied/not-applicable rows;
- valid-exception survival;
- subgroup coverage;
- rare-mode recall;
- row/cell edit rate và normalized edit magnitude.

### 10.3 Fidelity và utility

- numerical KS hoặc Wasserstein;
- categorical total variation;
- pairwise/joint correlation distance;
- conditional fidelity theo subgroup;
- propensity detection AUC;
- TSTR downstream utility;
- diversity/duplicate rate.

### 10.4 Cost

- sampling passes và best-of-N budget;
- wall-clock/GPU time;
- LLM calls, tokens và latency;
- number of projections/repairs/rollbacks;
- cost trên mỗi certified valid sample.

Không dùng chính loss/verifier đã guidance làm metric duy nhất.

## 11. Claim nên và không nên dùng

### Claim có thể nhắm tới

> Một verified dynamic constraint graph điều phối applicability, topology, intervention stage và solver để mở rộng validity–fidelity–preservation–cost frontier so với static guidance và fixed post-processing.

Ba contribution có thể hình thành:

1. General formulation nối public constrained synthesis với dynamic applicability, exceptions và graph lifecycle.
2. Controller có independent verification, chọn active graph và action theo row/state/objective.
3. Benchmark hai tầng với official baselines, four-state cases, distribution shift và matched budgets.

### Claim không nên dùng

- đầu tiên đưa LLM vào diffusion loop;
- đầu tiên đưa constraint vào reverse denoising;
- đầu tiên dùng LLM tạo hoặc kiểm chứng rule;
- LLM critic tốt hơn deterministic verifier;
- mid-denoising luôn tốt hơn repair;
- dynamic graph luôn tốt hơn frozen graph;
- state of the art trước khi official Harpoon/CuTS/LCATS và ít nhất ba dataset được so đầy đủ.

## 12. Đánh giá khả năng thành dự án ICDE

### Điểm mạnh

- Bài toán nằm ở giao của data quality, constraint management và synthetic data, phù hợp cộng đồng database hơn framđã ing thuần generative modeling.
- Đã có nhiều phản chứng, nên câu hỏi nghiên cứu không còn phụ thuộc một intuition duy nhất.
- General benchmark dùng prior-work tasks làm anchor, giảm rủi ro “tự tạo bài toán để thắng”.
- Dynamic graph có thể mở rộng ngoài LLM và ngoài inference-time, nên impact rộng hơn semantic request generation.

### Điểm yếu hiện tại

- Chưa có end-to-end dynamic controller là một method duy nhất; nhiều kết quả hiện là analysis hoặc oracle selection.
- Main generation evidence mới tập trung Default/Shoppers và một seed.
- Cross-table transfer chưa chứng minh.
- Official CTDF/SCTab-Diff chưa có common-table reproduction.
- Temporal monitoring hiện mới là graph audit, chưa nối vào generation under shift.
- Public hard constraints có nhiều task bão hòa hoặc repair rất mạnh.

### Đánh giá thẳng

- **Đủ tốt để thành research project:** có.
- **Đủ để viết submission mạnh ngay:** chưa.
- **Đủ để claim state of the art:** chưa.
- **Hướng có xác suất tốt nhất:** active graph + objective-conditioned stage/solver, dùng LLM như optional compiler; benchmark hai tầng.

## 13. Thực nghiệm quyết định tiếp theo

1. **Hoàn tất public baseline table:** đọc CuTS German, LCATS independent audit và Harpoon/RePaint official.
2. **Unified general runner:** chuẩn hóa output của CuTS/Harpoon/LCATS/TabDiff thành cùng validity, fidelity, utility và cost schema.
3. **Dynamic controller end-to-end:** chọn active edges và một trong ba action class: mid-denoising, resampling, final repair. Không bắt đầu bằng quá nhiều solver.
4. **Generation under graph shift:** frozen, last-window, EWMA/change detector, abstain và oracle trên ít nhất một dataset có checkpoint.
5. **Dataset thứ ba:** ưu tiên mixed-type table có đủ nontrivial relation và không bão hòa; chọn bằng train-only audit trước khi train backbone.
6. **Cross-table probe nhỏ:** leave-one-table-out relation-role mapping; dừng nhánh nếu không hơn retrieval/statistics.
7. **Replication:** chỉ sau khi method và baseline table ổn mới chạy 3–5 seeds cho main comparisons.

## 14. Ước lượng công việc

Với compute hiện có và không bị giới hạn queue, ước lượng hợp lý:

- 1–2 tuần: hoàn tất official baselines, unified evaluator và chọn dataset thứ ba;
- 2–3 tuần: implement controller end-to-end và generation-under-shift;
- 1–2 tuần: cross-table/foundation probe, ablation và failure analysis;
- 1–2 tuần: multi-seed main table, runtime/cost, utility và viết paper.

Tổng khoảng **5–9 tuần** cho một submission đủ chắc, nếu official environments không phát sinh lỗi lớn. Có thể có first complete main table trong 2–3 tuần. Nếu objective-conditioned controller vẫn không hơn fixed strong baseline, nên pivot sang paper về benchmark/analysis của dynamic constraint lifecycle thay vì cố giữ method claim.

## 15. Tiêu chí tiếp tục hoặc dừng

Tiếp tục method claim nếu:

- dynamic controller có Pareto region vượt best fixed baseline trên ít nhất ba datasets;
- gain tồn tại trước official baseline hoặc same-backbone faithful implementation;
- valid exceptions và inactive rows được preserve;
- intermediate stage có incremental value sau matched budget;
- runtime/call cost hợp lý.

Thu hẹp hoặc chuyển hướng nếu:

- final repair hoặc best-of-N dominate trên mọi public và dynamic task;
- graph update không dự đoán được shift tốt hơn abstention đơn giản;
- cross-table LLM không hơn parser/retrieval/statistics;
- benchmark dynamic chỉ tạo gain trên controlled synthetic cases.

## 16. Vị trí kết quả

- Báo cáo định hướng trước: `BAO_CAO_DINH_HUONG_DYNAMIC_GRAPH_20261001.md`.
- Cross-dataset audit: `outputs/cross_dataset_constraint_20260924/AUDIT_SUMMARY.md`.
- Request probe: `outputs/request_condition_aggregate_1031786/RESULTS.md`.
- CuTS official Adult: `outputs/cuts_official_result_audit_1045524/RESULTS.md`.
- CuTS sampling audit: `outputs/cuts_official_resample_audit_1045748/RESULTS.md`.
- TabDiff trên CuTS programs: `outputs/cuts_common_tabdiff_1051083/RESULTS.md`.
- Dynamic solver grid: `outputs/dynamic_graph_solver_default_1051370_0/` và `outputs/dynamic_graph_solver_shoppers_1051370_1/`.
- Trajectory graph gate: `outputs/persistence_graph_gate_1051548/`.
- Temporal lifecycle: `outputs/temporal_graph_lifecycle_1051556/`.
- Sequential change detection: `outputs/temporal_graph_change_1052277/`.
- Mid-stage value: `outputs/intervention_stage_value_1052276/RESULTS.md`.
- LCATS graph audit: `outputs/lcats_graph_audit_1051372/RESULTS.md`.
- LCATS official output audit: sau khi job `1052790` hoàn tất, xem `outputs/lcats_official_output_audit_1052790/RESULTS.md`.
- CuTS German audit: sau khi job `1052794` hoàn tất, xem `outputs/cuts_official_german_audit_1052794/RESULTS.md`.
- Harpoon/RePaint audit: sau khi job `1052795` hoàn tất, xem `outputs/harpoon_official_general_audit_1052795/RESULTS.md`.

## 17. Kết luận cuối

Setting chung không làm yếu dự án; ngược lại, nó giải quyết concern chính về baseline và benchmark legitimacy. Cách an toàn nhất là dùng constrained tabular synthesis như bài toán nền, dynamic constraint graph như đóng góp, và semantic request/foundation model như một interface có thể bật hoặc tắt.

Evidence hiện tại phản bác phiên bản đơn giản của ý tưởng ban đầu: LLM không nên trực tiếp phán xét từng preview, mid-denoising không luôn hơn repair, và graph update không luôn thắng frozen graph. Nhưng evidence cũng chỉ ra một gap bền hơn: constraint semantics và action đúng thay đổi theo row, trajectory, regime và objective; các baseline tĩnh giải từng mảnh nhưng chưa cung cấp một verified controller chung cho toàn bộ vòng đời đó.
