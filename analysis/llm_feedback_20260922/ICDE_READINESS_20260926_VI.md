# Đánh giá mức sẵn sàng cho ICDE và lộ trình thực nghiệm

**Ngày đánh giá:** 2026-09-26  
**Đích tham chiếu:** ICDE 2027 Research Track, Round 2 ngày 2026-11-11  
**Bằng chứng mới nhất:** target-scoped actuation aggregate `1040853`

## 1. Kết luận trực tiếp

### Có phải SOTA chưa?

**Chưa.** Hiện chưa có cơ sở khoa học để dùng từ SOTA vì:

1. chưa có so sánh faithful cùng backbone với CTDF, Harpoon và SCTab-Diff;
2. LCATS/Zhao mới được đại diện bằng mechanism proxy, chưa phải reproduction;
3. kết quả method chính chủ yếu trên Adult, còn Default/Shoppers mới dừng ở checkpoint/rule audit;
4. absolute conditional violation vẫn cao trên nhiều split;
5. chưa có downstream utility, privacy, multi-dataset confidence intervals và statistical test đầy đủ;
6. chưa có standardized benchmark mà điểm compiler/condition transfer hiện tại có thể được gọi là tốt nhất.

Tuy nhiên, dự án đã có **một số kết quả cạnh tranh trong phạm vi hẹp**:

- typed CFG 2 hơn best-of-2 trên unseen composition ở ngân sách denoiser xấp xỉ nhau;
- compile-once + executor hơn direct LLM judgement rõ về accuracy, request-swap và token cost;
- target-scoped high-CFG guidance giảm mạnh categorical/antecedent drift so với all-feature guidance;
- đã phát hiện và định lượng reliability–actuation dilemma mà các baseline tĩnh không trực tiếp đo.

Các kết quả này đủ để nói “promising mechanism evidence”, chưa đủ để nói “state of the art”.

### Có đủ để hình thành research project cho ICDE không?

**Có.** Problem phù hợp các chủ đề ICDE về data semantics, data quality/curation, heterogeneous data, foundation models for data engineering, benchmarking và responsible data management. Tuy nhiên framing phải là **semantic data generation/quality system**, không phải một architecture ML thuần túy.

Mức sẵn sàng hiện tại:

| Thành phần | Mức sẵn sàng | Nhận xét |
|---|---:|---|
| Problem/gap | 8/10 | Gap đã rõ sau khi loại bỏ các claim bị related work chiếm |
| Mechanism | 6/10 | Typed condition và scoped actuation có signal; inactive drift/exception chưa giải quyết |
| Novelty defensibility | 6/10 | Tốt nếu gắn open contract + applicability uncertainty + scoped actuator; yếu nếu chỉ nói LLM feedback |
| Baseline coverage | 4/10 | Proxy nhiều; thiếu CTDF/Harpoon/LCATS/SCTab faithful hoặc sufficiently matched implementation |
| Dataset breadth | 3/10 | Main method mới có Adult; Default/Shoppers chỉ mới audit checkpoint |
| Evaluation completeness | 4/10 | Có validity/fidelity/cost, thiếu utility/privacy/significance/cross-table |
| Artifact quality | 6/10 | Script và provenance tốt nhưng cần đóng gói config, one-command reproduction và public-safe release |
| Paper readiness | 4/10 | Có story và figures, chưa có main comparison table cấp submission |

Đánh giá tổng thể: **đủ mạnh để tiếp tục như một ICDE project; chưa nên submit ở trạng thái hiện tại**.

## 2. Track nào phù hợp hơn?

### Regular research paper

Phù hợp nếu trong 3–4 tuần tới có một method rõ ràng đạt Pareto improvement trên ít nhất ba dataset, cùng faithful/matched baselines. Main claim nên là:

> A schema-grounded open-contract interface and reliability-calibrated scoped actuator enable mixed-type diffusion to satisfy unseen row-conditional contracts while preserving non-target columns and valid exceptions.

Regular paper cần ít nhất một technical contribution rõ hơn benchmark, ví dụ:

- inactive-row restoration/trust-region actuator;
- uncertainty-calibrated active-set estimator;
- training objective bảo toàn immutable/non-target columns;
- cross-table contract encoder có transfer thật.

### Experiment, Analysis, and Benchmark paper

Hiện tại đây là đường **phù hợp và ít rủi ro hơn**. ICDE định nghĩa EAB là extensive evaluation hoặc benchmark tạo insight mới về strengths/weaknesses hay cách đánh giá mới. Dự án đã có nhiều insight đúng dạng này:

- direct LLM judgement vs compile-execute;
- parameter inversion của foundation embeddings;
- antecedent evasion và exception loss;
- readability window;
- reliability–actuation dilemma;
- per-feature scoped actuation;
- directionality và false invariants trên 788 candidate relations.

Một EAB submission có thể mang thesis:

> Existing validity evaluations for tabular generators conflate hard constraints, probabilistic dependencies, applicability and exceptions; a contract-centric benchmark exposes when conditioning, per-step guidance, selection and repair succeed or erase valid modes.

Nhược điểm: EAB bắt buộc cung cấp toàn bộ artifact tái lập. Benchmark phải rộng và ổn định, không thể chỉ là Adult pilots.

### Khuyến nghị

- Nếu deadline là ICDE 2027 Round 2: chuẩn bị **EAB-quality benchmark + một scoped actuator đơn giản** làm method companion.
- Nếu có thể dành thêm 10–14 tuần và chọn venue sau: phát triển thành regular paper với stronger actuator và faithful baselines.

## 3. Đánh giá kết quả mới nhất

### Phần thực sự tốt

Ở CFG 4, target-scoped actuation chứng minh một causal effect tương đối sạch:

- always-on: mean antecedent error giảm `0,0252`, categorical TV giảm `0,0071`, mean non-exception violation còn cải thiện `0,0022` so với all-feature;
- `global→gate@0,25`: antecedent error giảm `0,0244`, categorical TV giảm `0,0130`, numeric KS giảm `0,0041`;
- held-out composition: violation `0,4031→0,3811`, categorical TV `0,0360→0,0294`;
- exception split: antecedent error `0,0733→0,0126`, exception error `0,0184→0,0118`, categorical TV `0,0416→0,0242`.

Điều này hỗ trợ claim rằng **unscoped condition có thể thỏa rule bằng cách làm thay đổi antecedent/categorical context**, còn typed target mask ngăn shortcut đó.

### Phần chưa ổn

- Inactive-target error không giảm; ở `global→gate@0,25` còn tăng trung bình `0,2743`.
- Global early condition vẫn tác động target lên rows mà sau này được xác định là not applicable.
- Threshold/extrapolation không đồng thời tốt hơn all-feature ở mọi operating point.
- Exception absolute violation vẫn khoảng `0,68`, nên chưa thể gọi là exception-aware enforcement.
- High CFG làm validity tốt hơn nhưng có vùng fidelity xấu rõ; chưa có single operating point dominate các baseline.

Kết luận: target masking giải quyết **feature leakage**, chưa giải quyết **row leakage**.

## 4. Technical gap còn thiếu để thành regular paper

Phần method nên tập trung vào row leakage, không thêm LLM call tùy ý. Ba lựa chọn:

1. **Inactive-row restoration:** lưu unconditional target proposal hoặc short lookahead; khi late gate xác định not applicable, blend/rollback target về unconditional trajectory.
2. **Counterfactual trust region:** conditional delta trên target được nhân với soft applicability probability và giới hạn bằng per-feature trust region; uncertainty cao chỉ shape latent nhẹ.
3. **Applicability-preserving training:** paired same-noise loss buộc condition effect gần zero trên not-applicable/exception rows và monotonic trên applicable rows; loss phải đo terminal short rollout, không chỉ local `x̂₀`.

Lựa chọn 1 dễ triển khai nhất và trực tiếp kiểm tra nguyên nhân. Lựa chọn 3 có novelty tốt hơn nhưng tốn compute và rủi ro surrogate mismatch.

## 5. Thực nghiệm tối thiểu còn thiếu

### Nhóm bắt buộc cho cả regular và EAB

1. **Main method trên ít nhất 3 dataset**, ba seed:
   - Adult cho threshold/composition;
   - Shoppers cho directional implications và exceptions;
   - Default cho temporal-looking near-hard relations;
   - nên thêm HELOC, Bank Marketing hoặc dataset domain có metadata rõ.
2. **Matched baselines:** unconditional, static condition, typed adapter, Harpoon-style loss, CTDF-style per-step projection, final projection, post-repair, best-of-2/4, parser/oracle DSL và compile-execute.
3. **Natural và controlled evaluation tách riêng.** Controlled contracts chứng minh mechanism; natural constraints chứng minh relevance.
4. **Utility:** TSTR với ít nhất 3 downstream models; mean/std trên seeds/splits.
5. **Fidelity:** numerical KS/Wasserstein, categorical TV, pairwise/joint metric, conditional subgroup fidelity.
6. **Coverage:** rare category/pair coverage, antecedent prevalence, exception survival và mode dropping.
7. **Privacy:** tối thiểu DCR/NNDR và membership-inference proxy nếu claim synthetic-data quality rộng.
8. **Cost:** actual denoiser calls, verifier calls, tokens, GPU runtime, peak memory; không chỉ logical activation count.
9. **Statistical analysis:** bootstrap theo contract/dataset/seed; confidence interval và paired test với same initial noise.
10. **Robustness:** paraphrase, column rename, unseen threshold, unseen composition, held-out exception và held-out table.

### Nhóm bắt buộc nếu giữ LLM/foundation model trong claim

- ít nhất 2–3 API/open models;
- schema-grounded compile success và generated-counterexample equivalence;
- same-row/different-request pairs;
- direct judgement, template parser, oracle DSL, shuffled request và none-of-the-above controls;
- token/cost/latency và retry/invalid-program rate;
- held-out schema/table, vì Adult-only không chứng minh foundation representation cần thiết.

### Nhóm bắt buộc nếu claim dynamic feedback

- blank preview, shuffled preview và oracle-final-applicability;
- `(contract,t)` schedule không đọc preview;
- optimized one/two-pass runtime;
- actual matched compute;
- late repair với cùng compiler/actuator;
- rollback/restore rate và action survival.

## 6. Quy mô thực nghiệm ước lượng

Một matrix hợp lý, đã giảm bớt tổ hợp không cần thiết:

| Hạng mục | Số run ước lượng |
|---|---:|
| 4 datasets × 3 seeds × 4 core methods | 48 GPU runs |
| CTDF/Harpoon/final-repair/best-of-N baselines | 36–60 GPU runs |
| Main method ablation: scope, gate, restore, CFG | 36–54 GPU runs |
| Utility/privacy/fidelity evaluation | 100–200 CPU jobs |
| Compiler/verifier API benchmark | 6–12 API batches |
| Robustness: rename/paraphrase/exception/table | 24–40 evaluation runs |

Tổng cộng khoảng **120–180 GPU runs**, phần lớn có thể ngắn nếu reuse checkpoints, cộng 100–200 CPU jobs. Với 11 GPU jobs song song, raw compute có thể hoàn thành trong vài ngày; bottleneck thật là implementation, benchmark curation, debugging và phân tích.

## 7. Ước lượng thời gian

### Phương án gấp cho ICDE 2027 Round 2

Từ 2026-09-26 đến deadline 2026-11-11 có khoảng 6,5 tuần. Lịch khả thi nếu có ít nhất một người làm gần toàn thời gian và tài nguyên hiện tại:

| Tuần | Công việc | Deliverable |
|---|---|---|
| 1 | đóng băng thesis; implement inactive-row restoration; chuẩn hóa contract DSL cho Shoppers/Default | method specification, unit tests, smoke results |
| 2 | train/run three-dataset adapters; tích hợp CTDF/Harpoon-style matched baselines | first complete main table |
| 3 | ablation gate/scope/restore/CFG; utility, coverage và fidelity | Pareto plots, statistical tables |
| 4 | thêm dataset thứ tư; robustness và API compiler benchmark | cross-table/held-out evidence |
| 5 | privacy/cost, artifact cleanup, rerun failed cells, freeze numbers | reproducible release candidate |
| 6 | viết 12 trang, internal review, rebuttal-risk audit | submission draft |
| 0,5 | sửa cuối, format, supplementary và disclosure | final PDF/artifacts |

Khả năng hoàn thành:

- **EAB submission:** khả thi nhưng căng, khoảng 60–70% nếu ưu tiên benchmark và không mở thêm nhánh lớn.
- **Regular research paper:** khoảng 35–50% cho deadline này; phụ thuộc việc restoration/scoped method có thắng baseline trong hai tuần đầu.

### Phương án chất lượng cao hơn

Ước lượng 10–14 tuần:

- 2–3 tuần method/baseline engineering;
- 3–4 tuần multi-dataset experiments;
- 2 tuần robustness, utility, privacy và statistics;
- 1–2 tuần artifact/reproducibility;
- 2–3 tuần viết, internal review và rerun.

Nếu làm bán thời gian hoặc chỉ một người vừa code vừa viết, nên dự trù 14–18 tuần.

## 8. Cổng quyết định sau hai tuần

Sau hai tuần, chỉ tiếp tục regular-paper framing nếu đồng thời đạt:

1. main method tạo Pareto improvement trên ít nhất 2/3 dataset;
2. inactive-row restoration giảm drift ít nhất 25–30% mà giữ tối thiểu 80% validity gain;
3. CTDF/final repair không dominate toàn bộ frontier;
4. result giữ dấu qua ba seed;
5. exception/antecedent coverage không xấu hơn unconditional quá confidence interval.

Nếu không đạt, chuyển dứt khoát sang EAB framing. Negative results hiện đủ phong phú để tạo một benchmark paper tốt nếu mở rộng dataset và baseline một cách nghiêm túc.

## 9. Framing ICDE nên dùng

Không nên mở bài bằng “LLM improves diffusion”. Nên mở bằng vấn đề data engineering:

> Distributionally realistic synthetic tables may violate request-specific integrity semantics, while hard enforcement can erase legitimate exceptions and rare subgroups. Existing evaluations do not separate applicability, violation, exception and uncertainty, nor do they measure when intervention during generation is trustworthy.

Ba contribution phù hợp ICDE:

1. một typed semantic-contract abstraction và benchmark cho mixed-type synthetic data;
2. empirical characterization của reliability, actuation timing, exception survival và feature/row leakage;
3. một scoped conditional actuator hoặc restoration mechanism tạo Pareto improvement dưới matched budget.

ICDE 2027 liệt kê trực tiếp các chủ đề data semantics, heterogeneous data, foundation models for data engineering, data quality/curation và responsible data management. Tuy nhiên CFP cũng cảnh báo ML thuần túy không gắn data engineering có thể bị xem là ngoài phạm vi; paper phải nhấn mạnh integrity, data quality, contract execution, reproducibility và system cost.

## 10. Quyết định cuối

- **SOTA:** chưa thể claim.
- **Research project:** đủ mạnh, có gap và chuỗi bằng chứng thực nghiệm tốt.
- **ICDE:** phù hợp nếu framing theo data semantics/data quality; EAB hiện có xác suất thành công cao hơn regular track framing.
- **Thời gian tối thiểu:** 6–7 tuần rất tập trung cho một submission có thể gửi; 10–14 tuần để có bản regular-paper vững hơn.
- **Bước tiếp theo quan trọng nhất:** implement inactive-row restoration và chạy Shoppers/Default với CTDF/final-repair matched baselines. Đây là phép thử quyết định project là method paper hay benchmark/analysis paper.
