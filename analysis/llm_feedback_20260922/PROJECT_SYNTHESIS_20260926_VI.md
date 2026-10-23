# Tổng hợp dự án: foundation model và dynamic semantic constraints cho mixed-type tabular diffusion

> Bản cập nhật 28/09 với cross-dataset results, active-set causal controls, expanded API audit và limitation ledger: [`GAP_LIMITATION_LEDGER_20260928_VI.md`](GAP_LIMITATION_LEDGER_20260928_VI.md).

**Cập nhật:** 2026-09-26  
**Backbone thực nghiệm:** TabDiff trên Adult, Default và Online Shoppers  
**Phạm vi:** tổng hợp toàn bộ giả thuyết, bằng chứng đã chạy, kết quả bị vô hiệu, related work, ranh giới đóng góp và hàng đợi Slurm hiện tại  
**Quy tắc riêng tư:** dữ liệu thật chỉ được xử lý cục bộ; API ngoài chỉ nhận schema, contract do người nghiên cứu viết và bản ghi synthetic

## 1. Kết luận điều hành

Ý tưởng ban đầu — để LLM đọc `x̂₀(x_t)` tại một số reverse step rồi phản hồi cho các bước còn lại — không còn đủ hẹp để làm claim chính. PromptLoop, PPAD, Free²Guide và TTGen đã chiếm phần lớn không gian “foundation model quan sát intermediate diffusion state rồi cập nhật condition, reward hoặc nhánh sampling”. Trong tabular diffusion, Harpoon, CTDF, Constrained Discrete Diffusion và SCTab-Diff cũng đã cho thấy constraint hoặc prior có thể tác động trong denoising. LCATS, Zhao et al. và EVS đã bao phủ nhiều phần của chuỗi LLM khám phá rule, kiểm chứng rule và repair.

Hướng hiện tại đã được thu hẹp thành:

> **Open-contract conditional diffusion với applicability và exception theo từng bản ghi:** foundation model biên dịch một yêu cầu semantic chưa thấy thành typed contract được schema-grounding; diffusion model nhận contract làm condition; active set được quyết định theo bản ghi và theo độ tin cậy của trajectory; actuator chỉ được phép thay đổi các cột đã khai báo; mọi kết quả được đánh giá bằng verifier độc lập và so với projection, post-repair và best-of-N dưới ngân sách tương đương.

Bốn kết luận thực nghiệm quan trọng nhất hiện nay là:

1. **Không nên gọi LLM trực tiếp cho từng bản ghi hoặc từng timestep khi constraint có thể biên dịch.** Trên 220 trường hợp synthetic có language shift, hai model API đạt execution equivalence `1,000` khi compile một lần sang DSL và execute cục bộ, với khoảng 2.660 tokens. Direct judgement chỉ đạt accuracy `0,855` và `0,636`, dùng khoảng 61.500 tokens.
2. **Typed contract thực sự truyền được sang yêu cầu chưa thấy trong TabDiff.** Qua ba seed, typed AST tại CFG 2 tăng satisfaction so với null condition `0,171±0,023` trên held-out threshold và `0,194±0,023` trên unseen composition. Request ID gần như không transfer.
3. **Raw foundation embedding không đáng tin để mang operator và numerical parameter.** Paired-noise request swap cho thấy text và hybrid đảo chiều yêu cầu `hours 30→45`; typed AST đúng cả ba cặp. Factorized embedding, trong đó text chỉ mang semantic template còn AST mang parameter, sửa được chiều tác động nhưng chưa tốt hơn AST thuần.
4. **Dynamic row gating gặp reliability–actuation dilemma.** Applicability preview chỉ ổn định muộn; chờ tới `t<=0,25` làm mất gần hết tác dụng. Condition toàn cục sớm rồi chuyển sang row gate muộn phục hồi validity, nhưng tăng drift trên bản ghi không áp dụng. Đây là gap có bằng chứng rõ nhất cho phần “dynamic”.

Vị trí hợp lý của LLM hiện tại là **compiler và semantic router có quyền abstain**, không phải free-form judge trong inner loop. LLM chỉ nên nằm trong sampler nếu có một benchmark mà deterministic executor không biểu diễn được đầy đủ và online trajectory evidence chứng minh giá trị riêng qua blank-preview và shuffled-preview controls.

## 2. Định nghĩa chính xác phạm vi “dynamic”

Ba trục cần tách riêng:

| Trục | Định nghĩa | Ví dụ trong dự án | Có phải khoảng trống riêng không? |
|---|---|---|---|
| `request-dynamic` | Contract thay đổi giữa các yêu cầu sinh; giá trị, composition hoặc cách diễn đạt có thể chưa thấy khi train | `age>=45` sau khi chỉ train `35` và `50`; conjunction chưa thấy | Có thể bảo vệ nếu đo generalization thật |
| `row-dynamic` | Applicability, exception và uncertainty khác nhau giữa các bản ghi cùng một request | Chỉ áp dụng `hours>=45` cho `workclass=Private`, trừ `Exec-managerial` | Khoảng trống quan trọng nếu bảo toàn subgroup/exception |
| `trajectory-dynamic` | Intermediate preview cung cấp bằng chứng mới làm thay đổi active set hoặc actuator theo timestep | global condition sớm, row gate muộn; rollback khi gate không ổn định | Chỉ đáng giữ nếu hơn schedule không đọc preview |

Điểm phân biệt quan trọng là **constraint specification** và **state-dependent response**:

- SCTab-Diff giữ semantic prior tương đối tĩnh, nhưng score response vẫn phụ thuộc current row context ở mỗi bước.
- Harpoon giữ constraint request cố định, nhưng gradient thay đổi theo `x̂₀(x_t)`.
- CTDF áp feasibility operator ở mỗi reverse step.

Vì vậy dự án không claim “row/timestep-conditioned denoising” nói chung. Phần có thể claim phải là tổ hợp của **unseen request**, **schema-grounded typed parameters**, **row-specific applicability/exception**, **calibrated active set** và **scoped actuation**.

## 3. Câu hỏi nghiên cứu đã chỉnh sửa

### 3.1 Câu hỏi chính

> Một mixed-type diffusion generator có thể đáp ứng semantic contract chưa thấy, có điều kiện áp dụng và exception theo bản ghi, tốt hơn static conditioning, per-step feasibility guidance, best-of-N và post-repair ở ít nhất một vùng trên validity–fidelity–coverage–cost frontier hay không?

### 3.2 Câu hỏi về foundation model

> Foundation model nên tham gia ở lớp nào: biên dịch natural language sang typed contract, tạo representation cho condition, suy luận applicability, hay đọc intermediate preview để quyết định actuation?

Bằng chứng hiện tại trả lời sơ bộ:

- compiler frontend: có ích và tiết kiệm;
- raw text condition mang parameter: không an toàn;
- parameter-free semantic embedding: an toàn hơn nhưng chưa thêm giá trị trên Adult;
- direct row judge: yếu và tốn;
- online semantic judge: chưa chứng minh được nhu cầu;
- local executable applicability gate: có signal nhỏ nhưng exception chưa được giải quyết.

### 3.3 Câu hỏi trajectory cụ thể

> Khi applicability chỉ đọc được muộn nhưng condition cần tác động sớm, có thể tách semantic intent toàn cục khỏi active set theo bản ghi và giới hạn conditional delta vào mutable columns để đạt validity mà không làm biến dạng các bản ghi không áp dụng hay không?

Đây là câu hỏi đang được kiểm tra trực tiếp bởi loạt job `1040844–1040853`.

## 4. Những claim không được đưa ra

Dựa trên related work và kết quả âm, dự án không nên claim:

- LLM feedback trong diffusion là mới;
- constraint được đưa vào reverse process là mới;
- per-step hard constraint enforcement cho mixed numeric/categorical là mới;
- LLM khai phá và kiểm chứng executable rule là mới;
- LLM repair dữ liệu synthetic sau sinh là mới;
- value-dependent dependency trong table generation là mới;
- raw language embedding hiểu numerical threshold một cách đáng tin;
- validity tăng đồng nghĩa quality tăng;
- rule có zero observed exception trong train là universal invariant;
- các proxy baseline trong thư mục này là reproduction đầy đủ của paper tương ứng.

## 5. Kiến trúc nghiên cứu hiện tại

Pipeline mục tiêu gồm bảy lớp:

1. **Contract frontend:** natural-language request `q` được compile thành typed AST gồm `scope`, `antecedent`, `consequent`, `unless`, `mutable_columns`, `hardness`, `priority` và `abstention_policy`.
2. **Schema grounding:** mọi column, category, type, operator và numerical boundary được đối chiếu schema; invalid literal bị reject hoặc retry.
3. **Counterexample validation:** tự tạo boundary cases, categorical members, same-row/different-request pairs và exception cases để kiểm tra execution equivalence.
4. **Condition encoder:** so sánh request ID, AST, frozen text embedding và factorized text+AST. Parameter và operator không được để chỉ trong raw embedding.
5. **Conditional diffusion:** condition projection hoặc FiLM nhỏ trong TabDiff; condition dropout và CFG dùng cho controllability.
6. **Selective active set:** tại timestep `t`, active set lý tưởng là
   `C_i(t) = Retrieve(q_i) ∩ Applicable(x̂₀,i(t)) \ Exception(x̂₀,i(t))`,
   kèm uncertainty và quyền abstain.
7. **Typed actuator và independent verifier:** numerical projection/loss, categorical projection/remask, rollback hoặc branch selection; actuator chỉ sửa mutable columns; evaluator không được chỉ là chính rule/LLM đã guidance.

Trong các thí nghiệm hiện tại, phần semantic execution và applicability đều là oracle/executable. Chưa có claim end-to-end rằng LLM trong sampler làm tốt hơn rule engine.

## 6. Audit repository, môi trường và dữ liệu

### 6.1 Repository và môi trường

- TabDiff checkout: commit `5ecdb33`.
- Harpoon checkout được audit: commit `40dc8ce`.
- Không tìm thấy `AGENTS.md` trong phạm vi đã kiểm tra.
- Các thay đổi có sẵn của người dùng trong `tabdiff/main.py`, `tabdiff/trainer.py` và Harpoon được giữ nguyên.
- Tất cả script, log và output mới nằm dưới `analysis/llm_feedback_20260922/`.
- Python chính: `3.11.16`; PyTorch `2.5.1+cu124`.
- Job đã chạy trên T4, L4 và A10; các experiment adapter nhỏ thường dưới 300 MB allocated GPU memory; full checkpoint khoảng 0,7–0,8 GB peak allocation.

### 6.2 Dataset

| Dataset | Train / test | Numeric / categorical | Vai trò hiện tại | Rủi ro |
|---|---:|---:|---|---|
| Adult | 32.561 / 16.281 | 6 / 9 | trajectory probe, contract adapter, applicability và exception | demographic relation dễ bị hiểu sai thành norm; hard rule gần bão hòa |
| Default | 27.000 / 3.000 | 14 / 10 | checkpoint mới; temporal-looking implication và near-hard dependency | nhiều relation chỉ là probabilistic association |
| Shoppers | 11.097 / 1.233 | 10 / 8 | checkpoint mới; directional count–duration relation và rate ordering | vài rule support rất thấp; target-related rule không phải validity |

Không dataset nào trong ba tập hiện tại có long text. Nhánh text–structured được hoãn thay vì ép vào một backbone không phù hợp.

### 6.3 Full checkpoints mới

| Dataset | Epoch / update | GPU, thời gian | 4.096-sample fidelity | Duplicate rate |
|---|---:|---|---|---:|
| Default | 8.000 / 56.000 | A10, 5.362 s | numeric KS `0,0345`; categorical TV `0,0161` | `0` |
| Shoppers | 8.000 / 24.000 | A10, 2.856 s | numeric KS `0,0301`; categorical TV `0,0209` | `0,00415` |

Đây là bounded full-config training trong repo, không được gọi là paper reproduction.

## 7. Related work và ranh giới phương pháp

### 7.1 Phân loại theo thời điểm tín hiệu đi vào pipeline

| Work | Nguồn và độ tin cậy | Gap và method | Tín hiệu đi vào đâu, khi nào | Bài học cho dự án | Baseline feasibility |
|---|---|---|---|---|---|
| TabSyn | ICLR 2024, peer-reviewed, official code | Transformer VAE đưa mixed types vào continuous latent, sau đó score diffusion | Không có LLM/constraint; diffusion trong latent | `x̂₀` có thể decode cho critic nhưng action phải quay lại latent và chịu reconstruction error | Chưa có local checkpoint; train hai stage |
| TabDiff | ICLR 2025, peer-reviewed, official code local | Direct continuous-time diffusion cho numerical và categorical với feature-wise schedule | Mixed-type denoising; CFG cho imputation | Backbone phù hợp nhất để quan sát và sửa typed columns | Đã chạy trên ba dataset |
| Harpoon | ICLR 2026, peer-reviewed, code local | Differentiable inference loss trên predicted clean sample, gradient update reverse step | Trong denoising, response thay đổi theo `x̂₀` | Mọi claim về executable differentiable constraint phải so với Harpoon-style guidance | Official code có; cùng-backbone proxy đã chạy, reproduction chưa có |
| CTDF | ICAIF 2025; arXiv đăng 2026 | Training-free Euclidean/KL feasibility maps cho mixed-type TabDiff, áp hard affine/logical constraints | Feasibility operator ở mỗi reverse step | Không thể claim first per-step mixed-type enforcement; CTDF là baseline trực tiếp | Chưa tích hợp; cần same-rule matched-budget run |
| Constrained Discrete Diffusion | NeurIPS 2025 | KL/simplex projection, Gumbel-softmax và augmented Lagrangian cho discrete diffusion | Constraint projection xuyên denoising | Categorical action phải so với direct projection | Khác backbone; dùng technical proxy |
| LLM-TabLogic | preprint 2025 | LLM khám phá relation từ metadata và chọn compression/decompression trước latent diffusion | Trước train/sampling | Static metadata reasoning và representation selection đã có | Code có nhưng domain/pipeline khác |
| SCTab-Diff | preprint 2026 | LLM trích semantic prior, validate trên train; prior + column ID + current row context condition score network | Prior được đưa vào mọi reverse step; prior specification tĩnh, response theo state | “LLM prior trong denoising” không đủ mới; phải có unseen request/applicability/exception | Chưa có local checkpoint; cần static-prior proxy hoặc reproduction sau |
| LCATS | KDD 2026; official code | Constraint discovery/checking, TabSyn generation, diagnosis và optimization repair | Discovery trước, correction sau generation | Post-repair là baseline bắt buộc; nếu bằng hoặc hơn thì mid-sampling không cần thiết | Code không drop-in; hiện có deterministic repair proxy |
| Zhao et al. | preprint 2026, code link | Agent tạo equation/inequality/logical dependency, full-table validation, counterexample revision, deterministic repair | Discovery trước và repair sau | Compiler + validation + repair đã bị chiếm; novelty phải nằm ở request shift/selective generation | Chưa setup full; post-repair proxy hiện có |
| EVS | ICML Structured-FM workshop 2026 | LLM như noisy oracle để trích linear constraints, train verification và chọn subset giữ utility | Constraint extraction/selection trước generation | Không claim LLM extraction/filtering; cần request-specific unseen semantics | Workshop; có thể dùng làm compiler-selector baseline |
| SAGE | ACL 2026 | Value-aware pseudo-features và sparse dependency trong autoregressive LLM generator | Prefix/value-dependent guidance khi decode | Value-dependent relation không mới; diffusion-specific applicability vẫn còn | Generator paradigm khác |
| AFT-Tab | ACL 2026 | CTGAN cho structured columns, LLM cho long text, discriminator/PPO feedback | Training-time co-evolution | Chỉ liên quan khi text–structured consistency là trọng tâm | Không phù hợp ba dataset hiện tại |
| TabDLM | preprint 2026 | Continuous diffusion cho numeric và masked language diffusion cho categorical/text | Joint denoising, không có external critic | Backbone hợp lý nếu mở nhánh long text | Cần A100 80 GB theo setup paper; chưa ưu tiên |
| PromptLoop | CVPR 2026 | MLLM đọc denoised estimate rồi cập nhật prompt trong phần trajectory còn lại | Intermediate preview → revised condition | Generic feedback loop đã có; đóng góp phải là typed tabular transaction/applicability | Modality khác, không direct baseline |
| PPAD | preprint 2025 | Lookahead preview, error diagnosis, rollback và resampling | Intermediate preview/rollback | Preview nhiễu có thể làm feedback sai; bắt buộc calibrate, abstain và rollback | Image-only; mechanism control |
| Free²Guide | ICCV 2025 | Black-box LVLM reward và path-integral/branch guidance | Reward trong sampling, không gradient qua LVLM | Best-of-N/black-box reward là baseline bắt buộc | Best-of-2/4 proxy đã chạy |
| TTGen | CVPR 2025 workshop | LVLM sửa prompt từ clean latent, external score chọn nhánh | Intermediate prompt update + branch selection | Rất gần ý tưởng đầu, nhưng image-only và workshop | Không direct baseline |
| GRDM | NeurIPS 2025 | Joint graph diffusion cho relational database | Structural graph condition | Mở rộng dài hạn sang multi-table contract | Không thuộc pilot single-table |

### 7.2 Khoảng trống còn bảo vệ được

Sau khi loại bỏ phần đã có, khoảng trống hợp lý là:

- natural-language contract chưa thấy được compile thành schema-grounded program;
- program tách specification, applicability, exception, mutable fields và hardness;
- condition representation generalize sang numerical values và composition chưa thấy;
- applicability/exception được quyết định theo từng row với uncertainty rõ ràng;
- trajectory schedule được calibration theo độ ổn định của mixed-type preview;
- actuator bị giới hạn theo cột và kiểu dữ liệu để tránh antecedent evasion/collateral drift;
- đánh giá đồng thời validity, subgroup/exception survival, rare-mode coverage, fidelity, utility và cost;
- LLM có quyền abstain và không sở hữu ground-truth evaluator.

## 8. Feasibility matrix của baseline

| Baseline | Trạng thái | Cách dùng đúng trong nghiên cứu |
|---|---|---|
| Unconditional TabDiff | Đã chạy, ba dataset có checkpoint | Generator control |
| Request-ID conditional adapter | Đã chạy, ba seed | Non-semantic memorization control |
| Typed AST adapter | Đã chạy, ba seed | Main learned-condition baseline |
| Frozen text và raw hybrid | Đã chạy, ba seed | Foundation-condition ablation; hiện có failure về parameter direction |
| Factorized text+AST | Đã chạy, ba seed | Safety ablation: text không mang threshold/operator |
| Fixed typed guidance | Đã chạy trên pilot | Executable static guidance proxy |
| Harpoon-style numeric guidance | Đã chạy proxy | Không gọi là Harpoon reproduction |
| Applicability-gated conditional sampling | Đã chạy, ba seed | Row-dynamic mechanism |
| Delayed gate và two-phase schedule | Đã chạy, ba seed | Trajectory scheduling mechanism |
| Target-scoped conditional delta | Đang chạy, 3 guidance × 3 seed | Typed actuator/collateral-drift hypothesis |
| Final projection/post-repair | Đã chạy | LCATS/Zhao-style mechanism proxy |
| Best-of-2 và best-of-4 | Đã chạy, ba seed | Black-box selection under matched denoiser budget |
| CTDF | Chưa tích hợp | Direct per-step hard feasibility baseline cần bổ sung |
| SCTab-Diff | Chưa tái lập | Static-prior baseline; trước mắt dùng proxy và định vị rõ |
| LCATS full pipeline | Chưa chạy end-to-end | Insurance demo, OpenRouter, TabSyn/MILP environments; proxy phải ghi đúng nhãn |
| Direct API judgement | Đã chạy | Chi phí/độ tin cậy control; không phải main method |
| Compile-once + local executor | Đã chạy | Main foundation-model frontend baseline |

## 9. Audit rule và benchmark semantics

### 9.1 Bài học từ 788 candidates

Train-only discovery trên Adult, Default và Shoppers tạo 788 candidates:

| Loại | Số lượng | Cách sử dụng |
|---|---:|---|
| empirical executable | 80 | baseline, vẫn cần semantic review |
| near-hard dynamic | 95 | exception/applicability benchmark |
| probabilistic | 334 | không hard-enforce; dùng calibration stress test |
| demographic/sensitive | 85 | audit-only, không biến thành validity |
| target-related | 57 | distributional association, không semantic validity |
| insufficient support | 132 | stress test cho overfitting |
| exception-rate shift | 5 | held-out robustness |

### 9.2 Các cặp phản ví dụ quan trọng

- Shoppers `Administrative=0 → Administrative_Duration=0`: zero observed exception trên train/test; chiều đảo có `2,20%` train và `3,14%` test exception.
- `Informational=0 → Informational_Duration=0`: zero observed exception; chiều đảo có khoảng `2,3%` exception.
- `ProductRelated_Duration=0 → ProductRelated=0`: sai trên `95,31%` train và `91,78%` test applicable rows dù tên cột gợi quan hệ mạnh.
- `BounceRates<=ExitRates`: violation thật khoảng `1,4–1,5%`; không phải hard invariant.
- Adult `Own-child→age<30`: exception ổn định khoảng `19,7%`; hard enforcement sẽ xóa valid tail.
- Default `BILL_AMT1<=0→PAY_AMT1=0`: violation `33,0%` train và `28,1%` test; chỉ là dependency xác suất.

Các ví dụ này cho thấy benchmark phải phân biệt bốn trạng thái `satisfied`, `violated`, `not_applicable`, `exception`, đồng thời đo directionality và support. Chỉ nhìn mutual association hoặc natural-language plausibility là không đủ.

### 9.3 Audit checkpoint mới

Default checkpoint giữ tốt các near-hard rules được kiểm tra. Ví dụ `PAY_0=0→BILL_AMT1>0` có synthetic violation `0`; `PAY_0=-1→BILL_AMT1>0` là `1,01%`, cao hơn train `0,118%` nhưng support lớn. Shoppers checkpoint tái tạo đúng asymmetry: forward count→duration chỉ `0,31–0,41%` synthetic violation, reverse có `3,49–4,80%`; rule ProductRelated reverse vẫn sai `97,0%`. Đây là dataset tốt hơn Adult hard rules để kiểm tra exception-aware intervention.

## 10. Registry toàn bộ giả thuyết và kết quả

### H01 — Intermediate preview có readability window

**Giả thuyết.** `x̂₀(x_t)` chỉ đáng đánh giá sau một cửa sổ timestep, và numerical/categorical hội tụ khác nhau.

**Bằng chứng.** Known-`x₀` forward-noise probe cho categorical accuracy `97,37%` ở `t≈0,10` nhưng `87,46%` ở `t≈0,40`; numeric MAE lệch khoảng 24 lần. Trên free generation, category agreement với final row ở `t≈0,414` chỉ `81,84%`.

**Kết luận.** Được ủng hộ mạnh. Lịch gọi critic không thể chọn tùy ý.

### H02 — Static prior không bắt được semantic lỗi theo giá trị/subgroup

**Giả thuyết.** Có những lỗi cần record-specific applicability hoặc exception, không giải được bằng hard static prior.

**Bằng chứng.** Audit tìm thấy nhiều directional, probabilistic và exception-bearing relations. Tuy nhiên chưa có bằng chứng rằng LLM giải chúng tốt hơn typed executable program; phần lớn natural hard rules trên Adult đã được backbone mạnh học gần hết.

**Kết luận.** Có benchmark motivation, nhưng chưa chứng minh nhu cầu LLM. Cần coi đây là open question, không phải kết quả dương.

### H03 — Mid-sampling feedback tốt hơn final repair

**Giả thuyết ban đầu.** Sửa trong trajectory tạo cohort tự nhiên hơn và ít distortion hơn repair ở `t=0`.

**Bằng chứng pilot.** Dynamic correction ở `t≈0,10` còn `36,46%` hard-rule violation, trong khi post-repair đạt `0%`. Typed state projection mạnh giảm còn khoảng `1,76%` trên pilot nhưng gần như trùng post-repair về fidelity/utility. Trên full Adult checkpoint, no-guidance chỉ còn `1,56–1,76%`, post-repair `0%`, typed projection tốt nhất khoảng `0,065–0,13%`.

**Kết luận.** Bị bác bỏ trên hard executable Adult rules. Learned open-condition adapter vẫn có một Pareto region so với best-of-2, nhưng chưa vượt best-of-4/final projection tổng quát.

### H04 — LLM bổ sung kiến thức ngoài verifier/rule engine

**Giả thuyết.** LLM phát hiện semantic violation mà schema/rule engine không biểu diễn.

**Bằng chứng.** Schema-only GPT-4.1-mini bỏ sót 8/8 known violations và đề xuất sửa rare valid categories. Khi cung cấp exact rules, recall `100%` nhưng relation precision chỉ `28,57%`; typed applicability contract mới đạt hoàn hảo trên tập rất nhỏ vì contract đã encode verifier logic.

**Kết luận.** Bị bác bỏ trong hard-rule setting. Chưa thử thành công trên genuinely non-executable semantics.

### H05 — LLM feedback đáng tin và đáng chi phí

**Giả thuyết.** Per-row critique có precision, stability và benefit/token đủ cao.

**Bằng chứng.** Schema-only dùng 9.593 tokens trên 18 calls nhưng strict true positive bằng 0. Static-rules condition tăng recall nhưng tạo 20 relation-level false positives. Trong request-conditioned bank, direct judgement dùng khoảng 61.500 tokens/model và vẫn kém compile-once rõ rệt.

**Kết luận.** Repeated judgement bị bác bỏ như kiến trúc mặc định. Compile-once, validate và local execute được ủng hộ.

### H06 — Text–structured consistency

**Giả thuyết.** Long text tạo semantic conflict mà structured verifier không đủ.

**Bằng chứng.** Ba dataset hiện tại không có long text; TabDiff hiện tại không phải backbone tối ưu cho nhánh này.

**Kết luận.** Tạm hoãn. Chỉ mở lại với dataset/backbone như AFT-Tab hoặc TabDLM.

### H07 — Temporal persistence cải thiện dự báo final violation

**Giả thuyết.** Violation tồn tại ở hai preview liên tiếp và antecedent ổn định có precision tốt hơn một preview.

**Bằng chứng.** Persistence chỉ tăng precision khoảng `1,1` điểm phần trăm cho một rule, giảm recall `2–9` điểm; final violation tệ hơn one-shot late correction trong 6/6 blocks, trung bình `+0,716` điểm phần trăm.

**Kết luận.** Bị bác bỏ như main contribution.

### H08 — Selective recheck chống denoiser undo

**Giả thuyết.** Chỉ recheck rows đã sửa giảm violation với ít critic evaluations.

**Bằng chứng.** Targeted `.10→.03` giảm hard violation từ `37,43%` xuống `28,65%`, dùng khoảng 782 thay vì 1.024 record evaluations; dấu cải thiện ổn định 6/6 blocks. Nhưng direct state projection mạnh hơn và post-repair đạt `0%`.

**Kết luận.** Có ích như component cho weak actuator, không đủ làm claim trung tâm.

### H09 — Typed compiler quan trọng hơn LLM self-consistency

**Giả thuyết.** No-op filter, antecedent check và typed action contract loại systematic false positives tốt hơn repeat consensus.

**Bằng chứng.** Repeat intersection chỉ nâng precision từ `28,57%` lên `30,77%`; reject explicit no-op và intersection lên `57,14%`; typed contract đạt `100%` trên 18-call audit nhỏ.

**Kết luận.** Được ủng hộ về kiến trúc, nhưng trong hard-rule case compiler đủ mạnh để bỏ LLM.

### H10 — Actuator phải phụ thuộc feature type

**Giả thuyết.** Numerical `x̂₀` correction bị attenuation; direct state projection và categorical projection bền hơn.

**Bằng chứng.** Tại `t≈0,03`, `70,07%` education actions qua `x̂₀` vẫn vi phạm ở cuối; state projection giảm residual còn `0,53%`. Categorical direct-state actions có residual `0%` trong pilot.

**Kết luận.** Được ủng hộ về cơ chế; chưa chứng minh ưu thế so với post-repair.

### H11 — Compile once tốt hơn direct row judgement

**Giả thuyết.** LLM biên dịch request một lần sang typed DSL đáng tin và rẻ hơn gọi model trên từng row.

**Bằng chứng.** Trên 220 synthetic cases, compile→execute đạt `1,000` execution equivalence và swap-pair accuracy cho cả GPT-4.1-mini/GPT-5.4-mini, khoảng 2.660 tokens. Direct judgement đạt `0,855/0,636`, swap-pair `0,833/0,333`, khoảng 61.500 tokens.

**Caveat.** Compiler ban đầu tạo category literal không tồn tại; finite-bank execution vẫn `100%`. Schema vocabulary grounding và generated counterexamples mới đưa semantic exactness lên 10/10.

**Kết luận.** Được ủng hộ mạnh cho foundation-model frontend, không phải novelty độc lập vì Zhao/EVS đã làm các phần gần kề.

### H12 — Typed semantic condition transfer sang request chưa thấy

**Giả thuyết.** AST/text condition hơn request ID trên held-out values và compositions.

**Bằng chứng ba seed.** Typed CFG 2 gain `0,171±0,023` trên threshold và `0,194±0,023` trên composition; request ID chỉ `0,007±0,026` và `0,023±0,032`. Không có duplicate collapse quan sát được.

**Kết luận.** Được ủng hộ ở mức mechanism pilot. Cần cross-dataset và full utility/privacy evaluation.

### H13 — Learned condition có Pareto region so với selection/repair

**Giả thuyết.** Với ngân sách tương đương, condition adapter tạo conditional cohort tốt hơn chỉ chọn/sửa output.

**Bằng chứng ba seed.** Trên unseen composition, typed CFG 2 satisfaction `0,608`, cohort KS `0,110`; best-of-2 là `0,498/0,142`. Best-of-4 tốt hơn với `0,712/0,093` nhưng dùng bốn trajectories. Final projection đạt satisfaction `1,000` nhưng categorical TV và edit rate cao hơn rõ.

**Kết luận.** Được ủng hộ trong một vùng ngân sách hẹp; không phải dominance tổng quát.

### H14 — Foundation embedding thêm giá trị ngoài typed AST

**Giả thuyết.** Frozen text hoặc hybrid giúp semantic transfer.

**Bằng chứng.** Raw text/hybrid đảo request direction cho `hours 30→45` với margin khoảng `−0,648/−0,619`. Denoiser-level monotonic loss làm held-out margin tệ hơn `−1,294` dù endpoint train tốt hơn. Factorized template+AST sửa cả ba hướng và giảm collateral change, nhưng CFG-2 composition gain `0,162±0,034`, thấp hơn typed AST `0,194±0,023`.

**Kết luận.** Không được ủng hộ trên Adult. LLM/foundation embedding hiện chỉ chắc chắn có vai trò compiler; generator condition cần held-out-schema evidence.

### H15 — Row-wise applicability gate cải thiện condition without antecedent evasion

**Giả thuyết.** Gate từ unconditional preview chỉ bật condition cho applicable non-exception rows.

**Bằng chứng ba seed, CFG 2.** So với repair-preserve adapter, gate giảm violation thêm `0,032` trên train, `0,020` threshold và `0,036` extrapolation; composition chỉ `0,005`; exception tệ hơn `0,012`. CFG 4 tăng validity nhưng làm antecedent error và categorical TV xấu hơn.

**Kết luận.** Được ủng hộ một phần; exception và validity–coverage trade-off còn mở.

### H16 — Chỉ cần chờ gate đáng tin rồi mới actuation

**Giả thuyết.** Delay row gate đến `t<=0,25` giữ validity gain và giảm false activation.

**Bằng chứng.** Gate recall/accuracy cao ở late window, nhưng `t<=0,25` chỉ thu hồi violation reduction khoảng `−0,001/0,008/0,007/0,001` trên train/threshold/composition/extrapolation; `t<=0,10` gần như giống never-active.

**Kết luận.** Bị bác bỏ. Reliability muộn không đồng nghĩa action còn đủ thời gian ảnh hưởng.

### H17 — Global-early rồi row-gated-late giải reliability–actuation dilemma

**Giả thuyết.** Request-level condition có thể shape trajectory sớm; row gate muộn chỉ làm selection/refinement.

**Bằng chứng ba seed.** Trên bốn split không có exception, always-on gate giảm violation trung bình `0,0344`; `global→gate@0,25` giảm `0,0439`. Tuy nhiên inactive-target error tăng từ `1,195` lên `2,394`, numeric KS từ `0,0391` lên `0,0446`. Exception split không cải thiện rõ so với always-on.

**Kết luận.** Có signal về validity nhưng chưa đạt Pareto improvement. Kết quả tạo ra giả thuyết mới về target-scoped actuation.

### H18 — Target-scoped conditional delta giảm collateral drift

**Giả thuyết.** Chỉ giữ conditional delta ở mutable consequent column và giữ categorical logits unconditional sẽ bảo toàn phần lớn violation reduction, đồng thời giảm antecedent/exception/inactive drift.

**Thiết kế.** Paired initial noise; cùng adapter, contract và gate; so sánh `all-feature` với `target-only` ở guidance `1/2/4`, ba seed, các schedule always và `global→gate` tại `0,5/0,25/0,1`.

**Kết quả ba seed.** Tất cả jobs `1040844–1040853` đã hoàn tất. Ở CFG 4, so với all-feature actuation, target masking giảm mean antecedent error `0,0252` và categorical TV `0,0071` cho always-on schedule, trong khi mean non-exception violation còn tốt hơn `0,0022`. Với `global→gate@0,25`, masking giảm antecedent error `0,0244`, categorical TV `0,0130` và numeric KS `0,0041`; mean violation gần như ngang nhau (`+0,0008`), nhưng inactive-target error tăng `0,2743`.

Trên held-out composition tại CFG 4, target-scoped `global→gate@0,25` giảm violation từ `0,4031` xuống `0,3811` và categorical TV từ `0,0360` xuống `0,0294`. Trên exception split, nó giảm antecedent error từ `0,0733` xuống `0,0126` và categorical TV từ `0,0416` xuống `0,0242`, nhưng absolute violation vẫn cao `0,6813`.

**Kết luận.** Được ủng hộ cho việc ngăn condition đi đường tắt qua categorical antecedents và hạn chế collateral feature drift. Nó chưa giải quyết inactive-target drift do global early phase vẫn đẩy target trên mọi row, và chưa tạo hard-constraint satisfaction đủ mạnh để gọi là SOTA.

## 11. Kết quả two-phase schedule mới nhất

### 11.1 Reliability của gate

Tại CFG 2, relevant-category agreement chỉ khoảng `0,51–0,60` ở `t>0,75` nhưng `0,93–0,98` ở `t<=0,25`. Với unseen conjunction, early accuracy `0,962` che giấu recall chỉ `0,131`. Exception converges muộn nhất.

### 11.2 Causal schedule result

| Schedule | Mean violation reduction, 4 non-exception splits | Mean inactive-target error | Mean numeric KS | Diễn giải |
|---|---:|---:|---:|---|
| never | `0` | `1,085` | `0,0365` | no-actuation control |
| delayed `t<=0,25` | `0,0039` | `1,076` | `0,0369` | sạch nhưng hầu như quá muộn |
| always row gate | `0,0344` | `1,195` | `0,0391` | có hiệu lực dù early gate chưa ổn định |
| global→gate `t<=0,50` | `0,0414` | `1,777` | `0,0426` | thêm validity, tăng collateral shift |
| global→gate `t<=0,25` | `0,0439` | `2,394` | `0,0446` | validity tốt nhất trung bình, drift gần gấp đôi always |
| global→gate `t<=0,10` | `0,0438` | `2,424` | `0,0453` | không hơn `0,25`, drift cao hơn |

Kết luận phù hợp nhất là: two-phase scheduling giải được **timing**, nhưng chưa giải được **scope**. Một global condition sớm có thể đưa trajectory về vùng mong muốn, song latent response lan sang inactive rows và non-target features. Target-scoped actuation là kiểm tra nhân quả trực tiếp tiếp theo.

## 12. Các kết quả bị vô hiệu hoặc không được dùng làm bằng chứng

1. Focused API bank cũ có `case_id` chứa relation và `_clean/_corrupt`; batch đặt clean/corrupt pair cạnh nhau. Accuracy/F1 cao từ batch này bị label leakage và không được dùng.
2. Rare-control batch truyền `candidate_relation_to_verify='none'`, khiến false-positive rate bằng 0 không còn là exception evidence.
3. Một số local-model jobs OOM hoặc lỗi key; không có output hợp lệ để diễn giải.
4. Các aggregate đầu dùng comma-separated Slurm export bị cắt còn một run; đã được thay bằng aggregate đúng với colon-separated IDs.
5. Natural hard-rule Adult từ pilot checkpoint yếu không đại diện full model; full 8.000-epoch checkpoint mới là bằng chứng natural generation.
6. Gate reliability đang đo consistency với final generated row, không phải independent semantic truth. Nó chỉ là temporal-stability diagnostic.
7. Schedule implementation hiện vẫn tính hai denoiser passes ngay cả khi một branch về logic bị tắt; active-condition budget và wall-clock optimized budget chưa hoàn toàn trùng nhau.

## 13. Ba proposal đã thu hẹp

### Proposal A — Parameter-faithful open-contract diffusion

**Mục tiêu.** Compile natural-language request chưa thấy thành schema-grounded typed AST; condition TabDiff bằng AST; đánh giá transfer sang value/composition/table chưa thấy.

**Đóng góp có thể claim.** Một benchmark và method cho request-specific open contracts, numerical parameter fidelity, composition generalization và matched-budget actuator comparison trong mixed-type diffusion.

**Khác related work.** PromptLoop sửa prompt từ image preview; SCTab-Diff dùng prior khai phá từ metadata; Harpoon/CTDF nhận executable objective/feasibility operator; LCATS/Zhao repair output. Proposal này tập trung vào request shift và learned conditional cohort, với grounded executable interface.

**Bằng chứng hiện có.** Typed adapter transfer qua ba seed và tốt hơn best-of-2 trên unseen composition ở khoảng hai denoiser evaluations.

**Rủi ro.** Một grammar/parser nhỏ có thể đủ; best-of-4 hoặc projection có thể dominate khi tăng ngân sách; cross-table transfer chưa có.

**Thí nghiệm quyết định.** Adult + Default + Shoppers, thêm HELOC hoặc MIC; held-out parameter/composition/table; AST vs parser vs factorized FM; CTDF, Harpoon-style, post-repair và best-of-N dưới matched cost.

**Ước lượng.** 2–3 tuần cho multi-dataset adapters và metrics đầy đủ.

### Proposal B — Reliability-calibrated, scoped dynamic actuation

**Mục tiêu.** Tách semantic intent toàn cục khỏi row applicability. Giai đoạn sớm shape trajectory bằng request condition; giai đoạn muộn dùng calibrated gate; conditional delta chỉ đi vào mutable target columns; unstable rows abstain hoặc rollback.

**Đóng góp có thể claim.** Định lượng reliability–actuation dilemma trong mixed-type trajectories và một cơ chế two-phase, target-scoped để quản lý trade-off thay vì gọi critic ở timestep tùy ý.

**Khác related work.** Harpoon/CTDF không tập trung vào uncertainty của applicability/exception; PromptLoop/PPAD không có typed tabular active set và subgroup coverage; SCTab-Diff không nhận unseen request-specific exception program; LCATS tác động sau sampling.

**Bằng chứng hiện có.** Reliability window rõ, delayed-only thất bại, global-early phục hồi validity nhưng collateral drift tăng. Đây là chuỗi bằng chứng nhân quả tốt, dù final method chưa thắng Pareto.

**Rủi ro.** Target masking có thể làm mất condition effect vì adapter học distributed response; final repair có thể vẫn tốt hơn; online preview có thể không thêm gì ngoài schedule.

**Thí nghiệm quyết định.** Loạt target-scoped đang chạy; sau đó blank/shuffled preview, oracle-final-applicability và matched actual denoiser passes. Nếu preview controls không tạo unique gain, bỏ trajectory feedback khỏi main method.

**Ước lượng.** 3–5 tuần gồm optimized sampler, rollback và cross-dataset replication.

### Proposal C — Exception-aware abstaining verifier và benchmark

**Mục tiêu.** Xây benchmark bốn trạng thái với same-row/different-request pairs, hard distractors và naturally occurring exceptions. Deterministic tools xử lý executable clauses; foundation model chỉ route/verify phần ambiguous và có quyền abstain.

**Đóng góp có thể claim.** Calibration và exception survival là first-class metrics; phân biệt semantic applicability khỏi association; đánh giá chi phí và false action khi constraint bank lớn.

**Khác related work.** Zhao/EVS tập trung rule discovery/validation; LCATS sửa retained constraints; SAGE học endogenous dependencies. Proposal này nhấn mạnh external request, abstention, unseen exceptions và active-set control.

**Bằng chứng hiện có.** Cross-dataset audit cho thấy phần lớn candidates không phải hard constraints; direct FM judgement kém compiler-executor; exception gate hiện là điểm thất bại.

**Rủi ro.** Nếu mọi relation quan trọng đều code được, LLM không cần thiết. Adult có stereotype risk; cần domain metadata hoặc ontology tốt hơn.

**Thí nghiệm quyết định.** Candidate set `K={1,4,8,16}`, same-row/different-request, held-out table và exception family; compare NL-direct, NL→DSL→executor, oracle DSL, parser, shuffled request và none-of-the-above.

**Ước lượng.** 4–6 tuần, chủ yếu cho semantic benchmark và annotation/ground truth.

## 14. Trọng tâm contribution nên chọn

### Trọng tâm chính có rủi ro thấp nhất

**Open-contract conditional diffusion với parameter-faithful typed conditions.** Phần này đã có kết quả dương ba seed, có baseline matched budget và có failure analysis rõ. Foundation model được đặt ở compiler frontend, phù hợp với dữ liệu hiện có.

### Trọng tâm dynamic có tiềm năng cao hơn nhưng chưa chín

**Reliability-calibrated, target-scoped active set.** Đây là nơi có gap tốt nhất so với related work, nhưng phải chờ target-scoped jobs và cross-dataset replication. Nếu không tạo Pareto gain, nó nên trở thành analysis/negative result thay vì main method.

### Trọng tâm foundation-model condition dài hạn

Cross-table representation chỉ hợp lý nếu factorized foundation semantics giúp với schema/column mới. Adult result hiện không đủ: AST thuần tốt hơn hoặc tương đương. Cần leave-one-table-out, column rename/paraphrase và ontology-rich dataset trước khi claim FM condition có ích trong generator.

## 15. Tiêu chí quyết định tiếp tục hay dừng từng nhánh

### Tiếp tục open-contract adapter nếu

- typed/factorized condition giữ gain trên ít nhất hai dataset ngoài Adult;
- hơn request-ID và parser control trên unseen values/compositions;
- có ít nhất một Pareto region so với best-of-N, CTDF/Harpoon-style và final repair;
- numerical request swap luôn đúng chiều và collateral change được giới hạn.

### Dừng raw foundation embedding nếu

- vẫn có parameter inversion hoặc dependence vào surface form;
- AST thuần bằng hoặc hơn trên cross-table split;
- text branch chỉ tăng magnitude nhưng không tăng calibrated control.

### Giữ trajectory feedback chỉ nếu

- target-scoped/two-phase schedule giảm collateral drift mà giữ phần lớn validity gain;
- online preview hơn blank-preview, shuffled-preview và `(contract,t)` schedule;
- gain tồn tại dưới actual matched denoiser/verifier budget;
- exception survival và antecedent prevalence không xấu hơn post-repair.

### Không đưa LLM vào sampler nếu

- executable oracle cũng không hơn repair;
- foundation verifier không đạt calibrated precision trên exception/none-of-the-above;
- token/runtime cost cao hơn compile-once + local execute nhưng không có unique semantic recall.

## 16. Thực nghiệm tiếp theo có giá trị thông tin cao

1. **Đọc loạt target-scoped `1040844–1040853`.** Đây là kiểm tra gần nhất cho Proposal B.
2. **Cross-dataset conditional adapter trên Shoppers trước.** Dùng directional count→duration contracts, giữ reverse relation làm exception/probabilistic controls. Shoppers có semantics rõ hơn Default và checkpoint tốt.
3. **Default near-hard exception test.** Dùng `PAY_k`→`BILL_AMT` rules với support lớn, không hard-enforce các temporal associations 5–33% violation.
4. **Tích hợp CTDF-style feasibility projection trên cùng contract/sampler.** So per-step, final-only và learned condition với matched action budget.
5. **Oracle applicability upper bound.** Dùng final-row applicability để tách lỗi gate timing khỏi lỗi condition/actuator.
6. **Blank/shuffled preview controls.** Chỉ sau khi target-scoped actuation có signal; đây là phép thử quyết định online feedback có unique information hay không.
7. **Held-out-table foundation condition.** Chỉ nên submit sau khi catalog được chuẩn hóa qua ít nhất bốn dataset.

## 17. Slurm provenance và trạng thái hiện tại

### Các nhóm đã hoàn tất quan trọng

| Job IDs | Nội dung | Kết quả chính |
|---|---|---|
| `1028011–1028018` | Adult full checkpoint và actuator sweeps | natural hard rules gần bão hòa; projection gần repair |
| `1031779` | cross-dataset constraint audit | 788 candidates; đa số không phải hard semantics |
| `1033604–1033619` | direct vs compiler, grounding và counterexamples | compile-once thắng; grounding bắt lỗi finite-bank bỏ sót |
| `1033620–1033628` | condition representations, ba seed | AST transfer; request ID không transfer |
| `1033629–1033631` | actuator Pareto | CFG2 hơn best-of-2; best-of-4 mạnh hơn; projection validity tuyệt đối nhưng distortion |
| `1033729–1033738` | causal swap và text-dropout | raw text/hybrid parameter inversion |
| `1033750–1033759` | high-power implication/applicability | small gate gain; validity–coverage frontier |
| `1033763–1033768` | monotonic-loss hybrid | local loss không sửa terminal interpolation |
| `1033774–1033786` | gate reliability và delayed schedule | gate đáng tin muộn nhưng quá muộn để act |
| `1033791–1033796` | factorized foundation condition | sửa direction, chưa hơn AST |
| `1033760–1033761` | Default/Shoppers full training | hoàn tất checkpoint 8.000 epoch |
| `1033800–1033801` | cross-dataset checkpoint audit | fidelity tốt; directionality tái hiện |
| `1033803–1033807` | global-early→row-gated-late | validity tăng, collateral drift tăng |

### Jobs vừa submit

| Jobs | Guidance | Seeds | Mục tiêu |
|---|---:|---|---|
| `1040844–1040846` | 1 | `20260925–27` | hoàn tất; target-scoped vs all-feature ở low guidance |
| `1040847–1040849` | 2 | `20260925–27` | hoàn tất; operating point chính |
| `1040850–1040852` | 4 | `20260925–27` | hoàn tất; target scope kiểm soát categorical/antecedent drift |
| `1040853` | — | aggregate | hoàn tất; `outputs/scoped_actuation_aggregate_1040853/RESULTS.md` |

## 18. Hướng dẫn xem kết quả

### Bắt đầu từ các file tổng hợp

- Báo cáo này: `PROJECT_SYNTHESIS_20260926_VI.md`.
- Formulation gần nhất trước báo cáo này: `DYNAMIC_SEMANTIC_CONSTRAINTS_20260925.md`.
- Báo cáo pilot gốc và environment audit: `REPORT.md`.
- Follow-up về persistence, recheck và typed actuator: `NICHE_FOLLOWUP.md`.

### Bằng chứng định lượng chính

- Compiler/direct API: `outputs/request_condition_aggregate_1033608/RESULTS.md`.
- Grounded compiler retry: `outputs/request_condition_aggregate_1033619/RESULTS.md`.
- Three-seed condition transfer: `outputs/condition_adapter_aggregate_1033628/RESULTS.md`.
- Actuator Pareto: `outputs/contract_pareto_20260925/RESULTS.md`.
- Causal condition swap: `outputs/condition_swap_aggregate_20260925_v2/RESULTS.md`.
- High-power applicability: `outputs/implication_adapter_aggregate_1033759/RESULTS.md`.
- Gate trajectory: `outputs/applicability_gate_trajectory_aggregate_1033778/RESULTS.md`.
- Delayed schedules: `outputs/delayed_gate_schedule_aggregate_1033786/RESULTS.md`.
- Two-phase schedule: `outputs/delayed_gate_schedule_aggregate_1033807/RESULTS.md`.
- Factorized condition: `outputs/condition_adapter_aggregate_1033796/RESULTS.md`.
- Cross-dataset audit: `outputs/cross_dataset_constraint_20260924/AUDIT_SUMMARY.md`.
- Default checkpoint audit: `outputs/cross_dataset_checkpoint_default_1033800/`.
- Shoppers checkpoint audit: `outputs/cross_dataset_checkpoint_shoppers_1033801/`.
- Scoped actuator aggregate: `outputs/scoped_actuation_aggregate_1040853/RESULTS.md`.

### Tái chạy

- Slurm scripts mới: `slurm/47_scoped_actuation_schedule.sbatch`, `slurm/48_scoped_actuation_aggregate.sbatch`.
- Implementation mới: `scripts/scoped_actuation_schedule_probe.py`, `scripts/scoped_actuation_aggregate.py`.
- Toàn bộ provenance trước đó: `outputs/job_manifest.json` và `logs/`.

## 19. Kết luận hiện tại

Dự án đã đi từ một ý tưởng rộng về LLM feedback trong reverse diffusion sang một chương trình nghiên cứu cụ thể hơn:

- foundation model compile request, không trực tiếp sở hữu truth signal;
- typed AST giữ operator và parameter;
- diffusion học conditional cohort cho request chưa thấy;
- active set thay đổi theo row và trajectory, nhưng phải calibration;
- typed actuator giới hạn phạm vi thay đổi;
- exception, inactive rows và rare modes được đo như primary outcomes;
- post-repair, projection và best-of-N là đối thủ chính, không phải phụ lục.

Hiện đã đủ bằng chứng để tiếp tục **open-contract conditional diffusion**. Chưa đủ bằng chứng để claim LLM trong denoising loop là cần thiết. Cửa mở tốt nhất cho phần dynamic là target-scoped two-phase actuation; cửa mở tốt nhất cho foundation model là cross-table semantic grounding. Nếu hai nhánh này không vượt được AST + deterministic executor + repair dưới cùng ngân sách, paper nên tập trung vào open-contract benchmark/method và ghi online feedback như negative result có hệ thống.
