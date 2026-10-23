# Exploratory analysis: record-adaptive feedback trong mixed-type tabular diffusion

> **Báo cáo hiện hành (2026-10-11):** kết quả three-dataset mới nhất, protocol metric, negative results và paper outline nằm tại `PAPER_RESULTS_OUTLINE_20261011_VI.md`. Nội dung bên dưới được giữ như lịch sử của pilot ban đầu, không phải kết luận hiện tại.

> **Current direction update (2026-09-24):** this report preserves the original pilot. For the corrected API audit, cross-dataset evidence, literature boundary including CTDF/CTSyn, and the broader non-inference-time proposal, read `FOUNDATION_CONSTRAINT_DIRECTION_20260924.md`. The later focused-label API scores were found label-leaked and are explicitly invalidated there.

**Ngày chạy:** 2026-09-22  
**Backbone:** official TabDiff checkout, commit `5ecdb33`  
**Trạng thái:** có kết quả thực nghiệm nhỏ; đây là mechanism pilot, **không** phải reproduction đầy đủ của TabDiff/Harpoon/LCATS/SCTab-Diff.

## 0. Kết luận ngắn và quy ước bằng chứng

### Kết luận hiện tại

**Không nên tiếp tục với claim rộng “LLM feedback trong denoising làm tăng semantic validity”.** Với các hard rule Adult đã biết, fixed typed guidance và post-repair đều đưa violation từ `55.21%` về `0%` mà không cần LLM. Một correction theo bản ghi ở giữa trajectory (`t≈0.35`) chỉ đưa xuống `44.66%`; chuyển correction sang lúc preview đã ổn định hơn (`t≈0.10`) tốt hơn, `36.46%`, nhưng vẫn thua xa post-repair. Best-of-4 đạt `8.59%` nhưng làm rare-pair coverage giảm từ `85.71%` xuống `57.14%` và tốn 4× denoiser passes. Audit LLM schema-only bỏ sót toàn bộ 8/8 known violations; khi cung cấp exact rules, recall lên `100%` nhưng relation-level precision chỉ `28.57%`, tức LLM chủ yếu thi hành lại thông tin mà verifier đã có và còn tạo false alarms.

Hướng đáng làm tiếp phải hẹp hơn: **confidence-gated, exception-aware record feedback cho quan hệ mềm/không thể mã hóa sẵn**, với abstention và rollback; hard executable rules đi qua rule engine/repair. Adult hiện tại phù hợp để kiểm tra cơ chế và phản ví dụ, nhưng **không đủ để chứng minh sự cần thiết của LLM**.

### Quy ước

- **[RUN]**: số liệu thực sự đã chạy trong thư mục này.
- **[SOURCE]**: đọc từ bài/code nguồn; không phải kết quả tái lập.
- **[INFERENCE]**: diễn giải dựa trên [RUN]/[SOURCE].
- **[NOT RUN]**: chưa có bằng chứng thực nghiệm tại server này.

## 1. Repository, môi trường và Slurm audit

### 1.1 Repository

- Workspace là một collection, không phải một Git repo hợp lệ ở top-level. Hai repo liên quan là:
  - `baseline_repos/tabdiff`, commit `5ecdb33`.
  - `baseline_repos/harpoon`, commit `40dc8ce`.
- Không tìm thấy `AGENTS.md` trong phạm vi đã kiểm tra.
- Các sửa đổi có sẵn của người dùng được giữ nguyên:
  - TabDiff: `tabdiff/main.py`, `tabdiff/trainer.py`.
  - Harpoon: `experiments/imputation.csv`.
- Toàn bộ artefact mới nằm dưới `analysis/llm_feedback_20260922/`; không ghi đè checkpoint hoặc kết quả cũ.
- LCATS official code được checkout read-only để audit ở `external/LCATS`, commit `997018b`.

### 1.2 Compute và software

| Thành phần | Quan sát [RUN] |
|---|---|
| Login node | CPU only; 8 vCPU, 62 GiB RAM; không dùng để train/probe nặng |
| Slurm GPU | `prioritized`: V100/A40/T4/L4/L40S/A10; `aicentre`: A40; `aicentre-a100`: A100 |
| Pilot node/GPU | `a256-t4-01`, Tesla T4 |
| Python env dùng cho pilot | Python 3.11.16, PyTorch 2.5.1+cu124, CUDA build 12.4 |
| Core libs | NumPy 2.4.6, pandas 2.3.3, sklearn 1.9.0, SciPy 1.17.1 |
| Pilot training | 500 epochs = 4,000 optimizer updates; 531.34 s; peak allocated GPU memory 666 MB |
| Probe chính | 768 rows, 30 reverse steps, seed `20260922`; 35 s cho job gốc, 87 s cho rerun thêm late intervention |

Checkpoint cũ `debug/ckpt/ema_model_4.pt` chỉ train 4 epochs. Nó được dùng để kiểm tra đường chạy, không dùng làm bằng chứng chất lượng. Pilot 500 epochs vẫn rất nhỏ so với config/paper đầy đủ; mọi kết luận chất lượng phải được xem là sơ bộ.

### 1.3 Slurm provenance

| Job | ID | Trạng thái | Ghi chú |
|---|---:|---|---|
| Profile/rule audit | `1027375` | completed | Adult, Default, Shoppers |
| Bounded TabDiff training | `1027373` | completed | 500 epochs, T4 |
| Debug probe lần đầu | `1027374` | failed | thiếu `AnalysisDataset.__len__`; đã sửa |
| Debug probe retry | `1027379` | completed | smoke evidence |
| Pilot probe | `1027376` | completed | trajectory + 6 variants |
| LLM audit lần đầu | `1027377` | completed but experiment failed | model listed nhưng org chưa được verify cho `gpt-5-mini` |
| Pilot probe v2 | `1027386` | completed | thêm intervention `t=0.10`, output riêng |
| Withheld verifier audit | `1027387` | completed | rules không dùng trong guidance |
| LLM audit v2 | `1027389` | completed | `gpt-4.1-mini`, output riêng |
| LLM audit + static rules preflight | `1027392` | failed | transient model-access probe; không gửi case |
| LLM audit + static rules retry | `1027393` | cancelled | dừng sớm sau HTTP 429 để tránh spam API |
| Rate-limited static-rule retry | `1027394` | completed | 18/18 calls; 3:18 wall time do pacing 7 s/call |

Lỗi job `1027374` được giữ trong log để provenance; retry không ghi đè log cũ. LLM audit lỗi cũng được giữ riêng ở `outputs/llm_audit/`; retry ở `outputs/llm_audit_v2/`.

## 2. Baseline feasibility matrix

| Baseline/cơ chế | Code/checkpoint tại server | Có thể chạy ngay? | Cách dùng trong pilot | Nhãn đúng |
|---|---|---:|---|---|
| TabDiff | Official local repo; debug ckpt; pilot ckpt mới | Có | Backbone chính, direct mixed numeric/categorical diffusion | TabDiff backbone; pilot không phải paper reproduction |
| TabSyn | Không có local trained ckpt; LCATS vendored một implementation | Không ngay | Chỉ audit kiến trúc/khả năng decode latent | Chưa tái lập |
| Harpoon | Official local code + Adult checkpoint; env/code từng chạy imputation | Một phần | Cài cùng ý tưởng differentiable numeric loss vào TabDiff sampler | **Harpoon-style numeric proxy**, không phải Harpoon reproduction |
| Static prior/SCTab-Diff | Anonymous code link, không local checkpoint; kiến trúc khác TabDiff | Không ngay | Repeated fixed typed corrections trong reverse process | **Fixed-rule denoising proxy**, không phải SCTab-Diff |
| LCATS | Official repo local; chỉ ship insurance demo/data, cần OpenRouter, TabSyn train và MILP env | Không “drop-in” cho Adult | Deterministic typed repair trên output TabDiff | **LCATS-style post-repair proxy**, không phải LCATS reproduction |
| Zhao et al. agent repair | Official GitHub link từ paper, chưa setup | Không ngay | Được đại diện cơ chế bởi post-repair, nhưng không discovery agent | Post-repair proxy |
| Best-of-N / black-box reward | Không cần train | Có | 4 TabDiff branches, chọn row bằng executable verifier | Best-of-4 verifier |
| Record-adaptive mid-sampling feedback | Chưa có paper code tabular | Có dạng tối thiểu | Một correction typed theo preview ở `t≈0.35` và `t≈0.10` | Dynamic record-feedback proxy, chưa có LLM action loop |
| PromptLoop/PPAD/Free²Guide/TTGen | Image/video code/weights, modality khác | Không hợp lý làm baseline trực tiếp | Dùng để định vị novelty và thiết kế ablation | Related mechanism only |
| CDD | Discrete sequence diffusion, không phải mixed-table backbone | Không trực tiếp | Bài học cho categorical projection | Related mechanism only |

Lý do không chạy full LCATS: code official tổ chức thành sáu phase; phase 1/4/6 gọi OpenRouter, phase 2 train VAE+TabSyn, phase 5 giải MILP từng row, và repository chỉ cung cấp data demo `insurance`, không có checkpoint. Audit code xác nhận repair là post-generation MILP tối thiểu hóa weighted cell change; không có LLM quan sát `x_t`.

## 3. Related work: tín hiệu vào đâu và khi nào

| Work / độ tin cậy | Gap và method [SOURCE] | Tín hiệu vào generator | Bài học/gap còn lại cho đề tài | Khả năng baseline |
|---|---|---|---|---|
| [TabSyn, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/e9750610639c3e7a849cff746bf60dbd-Abstract-Conference.html), peer-reviewed + official code | Transformer VAE đưa mixed types vào latent liên tục; score diffusion trên latent; decode qua VAE. Đánh giá 6 datasets bằng marginals, pairwise, alpha/beta, utility, privacy; paper báo sampling <20 NFE. | Không LLM; không constraint guidance. | Có thể tính `x̂0` trong latent rồi decode cho critic, nhưng biến critique data-space ngược về latent cần decoder/Jacobian hoặc typed latent action; reconstruction error của VAE là thêm nguồn sai. | Code có, nhưng chưa có local ckpt; training hai stage. |
| [TabDiff, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5c882988ce5fac487974ee4f415b96a9-Abstract-Conference.html), peer-reviewed + official local code | Joint continuous-time diffusion trực tiếp cho numeric và categorical, feature-wise learnable schedules, mixed-type stochastic sampler; 7 datasets/8 metrics. | Không semantic LLM; classifier-free guidance cho imputation. | Direct data-space preview và typed column correction đơn giản hơn TabSyn; phù hợp backbone thăm dò. | **Đã chạy** local. |
| [Harpoon, ICLR 2026](https://openreview.net/pdf?id=G5g6tDg1ZE), peer-reviewed + official local code | Inference-time manifold guidance cho differentiable objective; lấy dirty `x̂0`, tính constraint/imputation loss, gradient về `x_t`, cập nhật mỗi reverse step. Paper đánh giá imputation và inequality constraints. | Differentiable loss trong denoising, không LLM. | Bất kỳ claim “constraint trong sampler” phải hơn baseline này. Categorical dùng continuous/OHE relaxation; semantic critique không khả vi vẫn là gap. | Official code chạy được nhưng backbone/preprocessing/task khác; pilot dùng proxy cùng TabDiff. |
| [LLM-TabLogic](https://arxiv.org/html/2503.02161), preprint + code link | LLM đọc metadata/description, nhóm quan hệ và nén deterministic columns trước latent diffusion kiểu TabSyn. Evaluation chủ yếu supply-chain; chính paper nêu metadata mơ hồ/domain hẹp/privacy metadata là hạn chế. | **Trước training/sampling**; không đọc intermediate sample. | Static metadata reasoning đã có; novelty không thể chỉ là “LLM + diffusion”. | Code link có nhưng datasets/domain và pipeline mới; không immediate. |
| [LCATS, KDD 2026 code](https://github.com/SJTU-DMTai/LCATS), conference metadata + official code; full ACM text không truy cập được trong phiên | Code: LLM discovery → executable checker; TabSyn generation → diagnostic buckets → LLM chọn repair weights → per-row MILP alignment → feedback/eval. | **Post-generation diagnosis/repair**; LLM không xem `x_t`. | Đối chứng mạnh: nếu repair đạt validity tương đương với ít distortion/cost thì mid-sampling feedback không đáng. Full-paper result claims không được dùng vì không đọc được full text; method được cross-check bằng code. | Không drop-in; insurance demo, OpenRouter, two envs, retrain TabSyn, MILP. |
| [SCTab-Diff](https://arxiv.org/html/2609.16069), preprint v1 (13 Sep 2026), anonymous code | GPT-4o-mini trích prior từ metadata, train-only validation; prior embedding + column ID + current row context condition score network qua cross-attention ở **mọi reverse step**; train với prior masking. 6 datasets; paper tự nêu phụ thuộc metadata, chỉ intra/inter symbolic rules, chưa privacy training. | **Static validated prior condition trong denoising**; current row context nhưng prior không được cập nhật bởi critique theo sample. | Novelty còn lại phải là observation/decision thay đổi theo row và timestep, hoặc reliable abstention/exception handling; không phải static prior. | Kiến trúc khác, không local ckpt; direct baseline tốn train mới. Fixed proxy chỉ là mechanism lower bound. |
| [Zhao et al.](https://arxiv.org/html/2608.15109), preprint + official code link | Agent tạo executable equations/inequalities/logical dependencies, full-table validation, counterexample revision; generator giữ nguyên, postprocessor sửa output. 7 public datasets/4 generator paradigms; paper báo zero measured violations chỉ cho retained/applicable validators và nêu không cover inter-row/cell-format constraints. | **Discovery trước**, **repair sau**. | Discovery/repair executable đã mạnh; zero violation không đồng nghĩa semantic completeness. Đề tài cần held-out verifier và uncovered/exception cases. | Có code nhưng chưa setup; post-repair proxy đã chạy. |
| [SAGE, ACL 2026](https://aclanthology.org/2026.acl-long.174/), peer-reviewed | LLM là generator; value-aware pseudo-features + pairwise MI; context selection hoặc logit correction thay đổi theo prefix/value. 6 datasets; limitation: pairwise MI bỏ sót higher-order, high-dimensional preprocessing. | Trong **autoregressive LLM decoding**, không phải diffusion feedback. | Value-conditioned dependencies không phải khoảng trống hoàn toàn. Đáng chú ý: bảng phụ Adult báo TabSyn `0.00%` violation cho education rule, SAGE-FS `1.32%`; rule này không chứng minh LLM cần thiết. | Không local code; generator paradigm khác. |
| [AFT-Tab, ACL 2026](https://aclanthology.org/2026.acl-long.209/), peer-reviewed | CTGAN structured generator + Qwen2.5-7B text generator; discriminator cung cấp PPO reward cho text và replay high-quality structured rows. Đề xuất TCCF; 3 long-text datasets. Limitation: RL compute, context length, không DP. | Feedback trong **training/co-evolution**, không reverse diffusion inference. | Chỉ liên quan khi có long text. Current 3 datasets không có long text, nên defer H6. | Cần 7B/RL/new datasets; không phù hợp pilot hiện tại. |
| [TabDLM](https://arxiv.org/html/2602.22586), preprint + code | Continuous diffusion cho numeric + masked language diffusion cho categorical/free text, LLaDA-8B. Paper thừa nhận sampling chậm hơn TabDiff/TabSyn và shared cross-modality schedule có thể suboptimal. | Joint diffusion; không LLM critic feedback. | Nếu chọn nhánh text, đây là backbone/đối chứng phù hợp hơn ghép text vào TabDiff. | Code có, paper dùng A100 80GB; không immediate trên T4. |
| [PromptLoop](https://arxiv.org/html/2510.00430), CVPR 2026 main | MLLM policy đọc denoised latent estimate ở denoising steps và sửa prompt; black-box diffusion/reward, policy học bằng terminal reward. Paper nêu overhead policy mỗi step/co-residency, và báo tăng ~5–23% runtime cho 1–5 improvement steps trong setting của họ. | **Intermediate preview → updated condition → remaining denoising**. | Ý tưởng vòng feedback không mới. Tabular contribution phải ở typed mixed-column action, reliability/exception, cost, và controlled baselines. | Modality/model khác; không direct baseline. |
| [PPAD](https://arxiv.org/html/2505.20053), preprint | Lookahead sketch khi `x_t` quá nhiễu; MLLM checker + corrective prompt; ping-pong rollback/resample. Ablation cho thấy semantic correction đơn độc có thể làm metric giảm vì preview nhiễu; rare concepts có thể suy giảm. | **Preview/lookahead giữa denoising**, rồi correct/rollback. | H1/H5 bắt buộc; abstention và rollback đáng hơn unconditional critique. | Image-only, no direct tabular code baseline. |
| [Free²Guide, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Kim_Free2Guide_Training-Free_Text-to-Video_Alignment_using_Image_LVLM_ICCV_2025_paper.html), peer-reviewed + code | Path-integral, gradient-free guidance dùng black-box LVLM reward; sample perturbation/branches, stitched frames để score video. | Black-box reward **trong sampling**, không sửa bằng gradient qua LVLM. | Best-of-N/path-integral reward là baseline cho bất kỳ LLM black-box guidance. | Video-only; best-of-4 proxy đã chạy. |
| [TTGen](https://openaccess.thecvf.com/content/CVPR2025W/ViSCALE/papers/Qiao_TTGen_Incorporating_Test-time_Scaling_to_Diffusion_Models_CVPRW_2025_paper.pdf), **CVPR workshop**, không phải main | Clean latent mỗi step → LVLM sửa prompt; CLIP chọn nhánh/latent tốt nhất. | Intermediate clean estimate + prompt update + branch selection. | Gần ý tưởng ban đầu; novelty phải tabular-specific và so với branch selection. | Workshop/image-only; không direct baseline. |
| [Constrained Discrete Diffusion, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/124c1cc9f7750f512ed81cdbf2fd5bf5-Abstract-Conference.html), main track | KL projection trên categorical simplex, Gumbel-softmax + augmented Lagrangian, chèn projection xuyên sampling; text/molecule/lexical tasks. Paper nêu compute overhead và giới hạn scale của open discrete DMs. | Differentiable constraint **trong discrete denoising**. | Typed categorical action nên so với/proxy projection trực tiếp, không chỉ prompt critique. | Không tabular backbone; có giá trị kỹ thuật cho categorical. |

## 4. Dataset và rule audit

### 4.1 Datasets

| Dataset | Train/test | Numeric/categorical | Missing | Rare categorical evidence | Long text |
|---|---:|---:|---:|---|---:|
| Adult | 32,561 / 16,281 | 6 / 9 | 0% | 16 rare `native.country` values, ~0.93% rows; rare workclass/marital/occupation values | Không |
| Default | 27,000 / 3,000 | 14 / 10 | 0% | rare codes ở EDUCATION/PAY columns | Không |
| Shoppers | 11,097 / 1,233 | 10 / 8 | 0% | rare OperatingSystems/Browser/TrafficType | Không |

`Missing=0%` ở đây nghĩa là không có pandas NaN trong processed split; Adult vẫn có sentinel category `?`, được giữ như một giá trị để audit missing-pair semantics.

Adult được chọn cho trajectory/intervention vì official local checkpoint/config và semantic relations dễ kiểm chứng. Default/Shoppers đã profile và là lựa chọn tốt cho phase kế tiếp, đặc biệt để tránh overfit proposal vào một rule Adult.

### 4.2 Rule audit chỉ dùng train để khám phá

| Rule | Train violation | Test violation | Quyết định |
|---|---:|---:|---|
| `education → education.num` (train-modal mapping) | 0 / 32,561 | 0 / 16,281 | Hard executable; LLM không cần sau khi discover |
| Husband ⇒ Male | 1 / 13,193 applicable (`0.0076%`) | 0 / 6,523 | Near-hard; giữ empirical exception thay vì gọi tuyệt đối |
| Wife ⇒ Female | 2 / 1,568 (`0.128%`) | 1 / 763 (`0.131%`) | Near-hard với exception |
| Husband/Wife ⇒ Married-civ-spouse | 21 / 14,761 (`0.142%`) | 14 / 7,286 (`0.192%`) | Near-hard với exception |
| `capital.gain/loss ≥ 0`, `hours.per.week ≥ 1` | 0 | 0 | Hard range/type rule |
| Own-child ⇒ age < 30 | **19.69%** applicable train là phản ví dụ | **19.66%** test | **Không được hard-enforce**; đây là valid tail subgroup |

Train-only candidate audit cũng tìm thấy các rule dễ overfit: `Preschool ⇒ <=50K` là 100% train nhưng có `3.125%` exception trong test; một số Browser/Traffic rules của Shoppers là exact trên support nhỏ nhưng có tới `7.14%` violation test. Đây là bằng chứng trực tiếp rằng “LLM/rule thấy hợp lý” không đủ; phải có held-out counterexample và minimum support.

## 5. H1 — Khi nào `x̂0(x_t)` đọc được?

### 5.1 Known-x0 forward-noise probe [RUN]

384 Adult test rows, 3 noise repeats; metric ở encoded space, pilot 500 epochs.

| `t` (0 = clean) | Numeric MAE | Categorical accuracy | Hard-rule violation của decoded preview |
|---:|---:|---:|---:|
| 0.00 | 0.0025 | 100.00% | 0.26% |
| 0.03 | 0.0040 | 99.15% | 1.48% |
| 0.10 | 0.0114 | 97.37% | 3.30% |
| 0.25 | 0.0760 | 92.77% | 13.19% |
| 0.40 | 0.2724 | 87.46% | 23.78% |
| 0.55 | 0.7993 | 80.70% | 45.66% |
| 0.70 | 1.1679 | 72.95% | 83.85% |
| 0.85 | 1.1827 | 65.10% | 93.49% |
| 1.00 | 1.1759 | 55.16% | 100.00% |

**[INFERENCE]** Một critic semantic ở `t≈0.4–0.7` sẽ nhìn rất nhiều artifact do noise, đúng phản ví dụ của PPAD. Trên backbone/checkpoint này, `t≤0.10` là cửa sổ hợp lý đầu tiên nếu đòi categorical agreement ≈95% và numeric MAE thấp. Không nên dùng một schedule chung cố định giữa datasets/checkpoints: threshold phải hiệu chuẩn từ validation trajectory.

### 5.2 Free-generation stability [RUN]

- Ở `t=0.414`, categorical agreement với record cuối chỉ `81.84%`, numeric MAE-to-final `0.1066`.
- Ở `t=0.103`, categorical agreement `95.82%`, numeric MAE-to-final `0.00028`.
- Preview rule violation không đơn điệu: `40.10%` ở `t=0.483`, rồi tăng dần tới final `55.21%`. Vì categorical cells còn masked/uncommitted, “preview đang hợp lệ” không đảm bảo final hợp lệ.

**Kết luận H1:** được ủng hộ mạnh: preview có một late readability window và các modality hội tụ khác nhau; gọi LLM giữa trajectory tùy ý là không defensible.

## 6. Cheap interventions và trade-off

Tất cả dùng cùng pilot checkpoint, seed, 768 rows và 30-step deterministic sampler. `runtime_seconds` bị ảnh hưởng warm-up/cache nên chỉ xem denoiser passes là cost proxy chắc chắn hơn. Violation chính dùng executable rules cũng dùng bởi các hard-rule baselines; vì vậy Section 6.2 bổ sung withheld verifiers không dùng trong guidance.

### 6.1 Main metrics [RUN]

| Variant | Guided violation ↓ | Num KS ↓ | Cat TVD ↓ | Joint MAE ↓ | Conditional income gap ↓ | Rare value/pair cov ↑ | AUROC/F1 ↑ | Passes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No guidance | 55.21% | 0.0846 | 0.1584 | 0.0307 | 0.1256 | 97.67% / 85.71% | .8177 / .5570 | 30 |
| Fixed-rule denoising proxy | **0.00%** | 0.0935 | 0.1592 | 0.0298 | 0.1215 | 95.35% / 85.71% | .8193 / .5515 | 30 |
| Dynamic row correction, `t≈0.35` | 44.66% | 0.0813 | 0.1571 | 0.0303 | 0.1271 | 97.67% / 85.71% | .8231 / .5678 | 30 |
| Dynamic row correction, `t≈0.10` | 36.46% | **0.0794** | 0.1545 | 0.0303 | 0.1293 | 97.67% / **92.86%** | .8238 / .5665 | 30 |
| Harpoon numeric proxy | 40.89% | **0.0784** | 0.1598 | 0.0307 | 0.1256 | 97.67% / 85.71% | .8226 / .5605 | 30 |
| Post-repair proxy | **0.00%** | 0.0838 | **0.1537** | 0.0299 | 0.1266 | 95.35% / 78.57% | .8145 / .5586 | 30 + repair |
| Best-of-4 verifier | 8.59% | 0.0865 | 0.1577 | **0.0294** | **0.1170** | 97.67% / **57.14%** | **.8330 / .5759** | **120** |

Các số nhỏ khác biệt AUROC/F1 là **một seed, không confidence interval**; không claim thắng chất lượng. Duplicate rate là 0 cho cả bảy variants, nhưng 768 rows quá ít để xem đó là bằng chứng diversity mạnh.

### 6.2 Withheld verifiers và exception preservation [RUN]

Các checker dưới đây **không** được dùng bởi fixed guidance, dynamic correction, post-repair hay best-of-4 score.

| Dataset/variant | Missing-pair XOR ↓ | Priv-house-serv ⇒ Private violation ↓ | Own-child age≥30 conditional rate (target test 19.66%) |
|---|---:|---:|---:|
| Real train | 0.0215% | 0% | 19.69% |
| Real test | 0.0184% | 0% | 19.66% |
| No guidance | 6.51% | 0.13% | **45.08%** |
| Fixed-rule denoising | 6.38% | 0.13% | 45.60% |
| Dynamic late | 6.51% | 0.13% | 45.08% |
| Post-repair | 6.51% | 0.13% | 45.08% |
| Best-of-4 | **6.64%** | 0.13% | 44.78% |

**[INFERENCE]** Tất cả interventions chỉ sửa các rule đã biết, không cải thiện withheld semantics; thậm chí generator pilot overproduces legitimate tail subgroup `Own-child, age≥30` hơn 2× real rate. Điều này vừa là model-quality limitation của pilot, vừa là cảnh báo: hard-enforce “Own-child ⇒ age<30” sẽ cho validity đẹp giả tạo bằng cách xóa subgroup hợp lệ.

### 6.3 Diễn giải cơ chế

1. Late correction tốt hơn mid correction: absolute reduction so với no-guidance là `18.75 pp` tại `t≈0.10`, so với `10.55 pp` tại `t≈0.35`. Điều này nhất quán với trajectory readability.
2. Fixed guidance và post-repair cùng đạt 0% trên guided rules. Fixed guidance làm numeric KS xấu hơn `+0.0089`; post-repair giữ numeric KS gần baseline nhưng rare-pair coverage giảm `7.14 pp`.
3. Best-of-4 cải thiện validity và utility proxy nhưng mất `28.57 pp` rare-pair coverage, là phản ví dụ mode selection rõ nhất.
4. Harpoon numeric proxy chỉ sửa quan hệ `education ↔ education.num`; nó giảm education violation `47.40% → 31.51%` nhưng không động đến categorical spouse rules. Kết quả này không được gọi là Harpoon reproduction.
5. Dynamic typed correction hiện không dùng LLM; nó là oracle/rule critic upper-bound cho phần **action execution**. Nếu LLM critique kém hơn oracle, end-to-end không thể vượt các con số này nếu không có cơ chế khác.

## 7. H5 — LLM feedback audit

### 7.1 Thiết kế

- Input: schema metadata + **synthetic preview row** tại khoảng `t=0.7, 0.4, 0.1`; không gửi raw real rows.
- Output strict JSON: `violated_relation`, `columns`, `applicability_condition`, `confidence`, `suggested_change`.
- Mỗi case gọi 2 lần; so với executable verifier độc lập với LLM; đo precision/recall, repeat disagreement, actionability, tokens, runtime.
- API dùng Responses Structured Outputs, `store:false`, và usage accounting theo [official OpenAI API reference](https://developers.openai.com/api/reference/resources/responses/methods/create).

### 7.2 Kết quả [RUN]

<!-- LLM_RESULTS_START -->
**Schema-only, `gpt-4.1-mini`:** 9 cases × 2 repeats = 18/18 calls thành công; 8,252 input + 1,341 output = 9,593 tokens; tổng API latency 26.68 s. Với giá niêm yết ngày chạy là $0.40/M input và $1.60/M output tokens ([official model page](https://developers.openai.com/api/docs/models/gpt-4.1-mini)), chi phí ước tính là **$0.00545 tổng, $0.00030/call**, chưa gồm hạ tầng sampler. So với strict executable verifier, micro precision = `0`, recall = `0`; mean repeat-set disagreement = `22.22%`; `44.44%` calls trả ít nhất một action có column hợp lệ.

Chi tiết theo timestep:

| `t` | Calls | TP / FP / FN | Tokens | API latency |
|---:|---:|---:|---:|---:|
| 0.690 | 6 | 0 / 0 / 6 | 2,822 | 7.08 s |
| 0.414 | 6 | 0 / 1 / 2 | 3,283 | 8.63 s |
| 0.103 | 6 | 0 / 1 / 0 | 3,488 | 10.97 s |

Không nên diễn giải precision `0` là mọi critique đều vô nghĩa: strict verifier chỉ cover rule set nhỏ. Tuy nhiên hai failure có ý nghĩa trực tiếp:

1. LLM bỏ sót toàn bộ `education ↔ education.num` mismatch vì schema-only không cung cấp mapping (`HS-grad` preview có code 10 thay vì 9; `Some-college` có 11 thay vì 10). Nếu phải đưa mapping vào prompt thì rule engine đã có cùng thông tin và check chính xác/rẻ hơn.
2. Ở hai repeats của cùng record hợp lệ, LLM đề nghị đổi `native.country=France` thành `United-States` vì phổ biến hơn đối với workclass/race. Đây là semantic hallucination có thể xóa rare mode. Nó cũng nhầm whitespace hợp lệ của source categories là lỗi formatting.

Vì strict TP = 0, schema-only condition có **0 detected violations per dollar** trong audit này; rẻ theo từng call không đồng nghĩa cost-effective.

**Static validated relations, cùng `gpt-4.1-mini`:** 9 cases × 2 repeats = 18/18 calls thành công. Khi prompt được cung cấp exact education mapping và spouse/range rules, relation-level micro recall tăng từ `0` lên **`100%`**, nhưng micro precision chỉ **`28.57%`** (`8 TP / 20 FP / 0 FN`). Repeat-set disagreement giảm còn `5.56%`; `66.67%` calls có output được parser xem là actionable. Tổng usage là 12,788 input + 1,608 output = 14,396 tokens, tương đương **$0.00769 tổng, $0.00043/call** theo cùng giá niêm yết. Tổng API latency là 29.92 s; Slurm wall time 3:18 vì chủ động chèn 7 s giữa calls để tránh rate limit.

| `t` | Calls | TP / FP / FN | Tokens | API latency |
|---:|---:|---:|---:|---:|
| 0.690 | 6 | 6 / 6 / 0 | 4,850 | 10.76 s |
| 0.414 | 6 | 2 / 7 / 0 | 4,785 | 8.90 s |
| 0.103 | 6 | 0 / 7 / 0 | 4,761 | 10.26 s |

Các FP không chỉ là lỗi canonicalization: model nhiều lần đặt `record_is_valid=false` rồi liệt kê Husband/Wife constraints cho `Not-in-family` hoặc `Other-relative`, dù chính câu giải thích nói điều kiện không áp dụng/no change needed. Nó còn yêu cầu sửa `capital.loss` dương thành non-negative. Một repeat ở `t=.103` trả sáu “violations” mà toàn bộ suggested changes đều nói không cần sửa. Vì vậy static prompt đã giúp đọc mapping được cung cấp, nhưng không cho thấy reasoning động vượt verifier; structured schema hiện tại cũng chưa đủ ép model phân biệt **checked-and-satisfied** với **violated**.

So sánh hai conditions tạo một kết luận H4/H5 khá rõ trong phạm vi pilot: không đưa rule thì LLM không biết mapping; đưa rule thì deterministic checker chính xác hơn, rẻ hơn và không misapply antecedent. `outputs/llm_audit_static_rules/` lưu đầy đủ raw structured outputs và per-call scoring.
<!-- LLM_RESULTS_END -->

## 8. Trạng thái giả thuyết

| H | Trạng thái | Bằng chứng |
|---|---|---|
| H1 preview readable theo timestep/modality | **Ủng hộ** | Known-x0 categorical 97.37% tại `t=.10` nhưng 87.46% tại `.40`, numeric MAE chênh 24×; late correction tốt hơn mid correction. |
| H2 có lỗi static prior không bắt được | **Chưa đủ; setting hiện tại nghiêng về bác bỏ** | Hard rules đều executable/static. Train/test cho thấy exception-bearing relations, nhưng chưa chứng minh LLM khám phá/giải đúng hơn counterexample-validated rule engine. |
| H3 dynamic feedback hơn repair | **Bác bỏ trên pilot Adult hard rules** | Dynamic late còn 36.46% violation; post-repair 0%. Fixed guidance cũng 0%. |
| H4 LLM thêm gì ngoài verifier | **Không thêm giá trị trên hard-rule case; no-go** | Schema-only bỏ sót 8/8 known violations; có exact rules thì recall 100% nhưng precision 28.57%. Typed oracle/rule engine chính xác và đã đủ cho guided rules; withheld semantics cũng không được cải thiện. |
| H5 đáng tin/đáng tiền | **Không đạt trong pilot; semantic/exception case vẫn chưa thử** | $0.00545 schema-only cho 0 TP; static-rules tốn $0.00769, đạt recall nhưng tạo 20 relation-level FP. Kết quả chỉ gồm 18 calls/condition nên không ngoại suy model-wide. |
| H6 text dài | **Defer/no-go trong repo hiện tại** | Adult/Default/Shoppers không có long text; ép nhánh này sẽ đổi backbone/dataset và cản thí nghiệm cơ bản. |

## 9. Proposal đã thu hẹp

### P1 — Readability-gated, abstaining critic cho exception-bearing relations

**Câu hỏi:** khi verifier executable không đủ vì relation có exception/subgroup, liệu critic chỉ được gọi sau khi per-modality stability vượt threshold, và có quyền `abstain`, có cải thiện held-out semantic validity mà không làm mất exception modes?

- Khác PromptLoop/TTGen: action là typed column proposal + abstain, schedule được calibration từ tabular trajectory, không phải prompt revision mỗi bước.
- Khác SCTab-Diff: feedback thay đổi theo row/preview; prior tĩnh chỉ là context.
- Khác Harpoon/CDD: critic có thể black-box/semantic; executable parts vẫn đi qua differentiable/projection rule engine.
- Khác LCATS/Zhao: correction xảy ra trước final sample nhưng phải beat post-repair dưới cùng budget.
- Experiment quyết định: 3 datasets, 3 seeds, relation sets chia `hard executable / probabilistic / held-out`; compare no guide, fixed, Harpoon/CDD proxy, repair, best-of-N, oracle dynamic, LLM dynamic; report subgroup coverage và abstention calibration.
- Kill criterion: LLM precision dưới verifier/rule discovery, hoặc không vượt repair trên Pareto validity–fidelity–cost.
- Công: 2–3 tuần nếu có curated semantic benchmark; rủi ro lớn nhất là ground truth cho soft semantics.

### P2 — Critique-to-action compiler với rollback, không gọi LLM trong inner loop

**Câu hỏi:** một lần LLM critique muộn có thể biên dịch thành executable typed action (projection, remask/resample, bounded numeric update), chạy verifier độc lập, rollback nếu fidelity/local likelihood xấu, rồi tiếp tục vài bước cuối hay không?

- Novelty nằm ở typed mixed-table action/rollback contract, không ở “LLM nhìn `x̂0`”.
- Direct baselines: post-repair (same compiler at `t=0`), fixed rule, Harpoon gradient, CDD-style categorical projection, best-of-N.
- Experiment quyết định: matched action budget; same critique/action applied at `t=.25,.10,.03,0`; đo success, rollback rate, cell distortion, log-proxy/fidelity, tokens.
- Kill criterion: `t=0` repair luôn đạt cùng/hơn kết quả hoặc continuation sau action undo correction.
- Công: 1–2 tuần cho TabDiff; rủi ro là không có principled likelihood score cho mixed categorical state.

### P3 — Hướng không-LLM: exception-safe constraint guidance benchmark

**Câu hỏi:** can thiệp hard rules làm mất legitimate rare modes ở mức nào, và confidence/support/counterexample gating có tạo Pareto frontier tốt hơn hard enforcement?

- Đây là hướng fallback có giá trị ngay cả khi LLM no-go.
- Novelty ứng dụng: benchmark/algorithm cho hard-vs-soft constraint typing, subgroup exception coverage và held-out counterexamples trên mixed-type diffusion.
- Experiment quyết định: inject deliberately over-broad train-perfect rules; compare hard repair/guidance với calibrated soft penalties/abstention. Sử dụng Default/Shoppers candidates có train-exact nhưng test exceptions.
- Công: 1–2 tuần; rủi ro thấp, nhưng contribution algorithm có thể cần mạnh hơn một benchmark.

## 10. Go/no-go và bước có giá trị thông tin cao nhất

### Quyết định

- **NO-GO** cho “LLM feedback per row để enforce hard executable rules trên Adult”. Static typed guidance và post-repair đã thắng rõ về validity; không có bằng chứng LLM cần thiết.
- **CONDITIONAL GO** cho P1/P2 chỉ khi dựng được benchmark soft semantic/exception mà:
  1. rule engine/static prior không biểu diễn đủ;
  2. held-out human/external verifier tồn tại;
  3. oracle dynamic correction có Pareto advantage so với repair/best-of-N trước khi thay oracle bằng LLM.
- **GO** cho P3 như fallback có rủi ro thấp và trực tiếp giải quyết rare-mode counterexample đã quan sát.

### Next experiment có information gain cao nhất

Không train thêm model lớn ngay. Chạy cùng protocol trên **Default và Shoppers** với 20–30 rule candidates được chia bằng held-out exceptions; thêm oracle dynamic action + post-repair cùng compiler. Nếu oracle dynamic **không** vượt repair trên validity–fidelity–rare-mode frontier, dừng nhánh LLM-in-denoising. Nếu có gap, mới gọi LLM trên chính subset mà static verifier bất định.

## 11. Cách xem và tái chạy

Quick view:

1. `outputs/pilot_probe_v2/trajectory_readability.png` — đồ thị H1.
2. `outputs/pilot_probe_v2/intervention_metrics.csv` — main baseline table.
3. `outputs/pilot_probe_v2/independent_verifier_metrics.csv` — withheld rules và exception subgroup.
4. `outputs/llm_audit_v2/` và `outputs/llm_audit_static_rules/` — schema-only/static-rule H5, raw JSONL, cost và scoring.
5. `outputs/profile/rule_audit.csv` và `train_only_rule_candidates.csv` — dataset/rule audit.
6. `outputs/job_manifest.json`, `logs/` — Slurm provenance/failures.

Tái submit theo `README.md`. Scripts chính:

- `scripts/profile_and_rules.py`
- `scripts/train_pilot.py`
- `scripts/tabdiff_probe.py`
- `scripts/posthoc_independent_audit.py`
- `scripts/llm_feedback_audit.py`

## 12. Hạn chế của đợt analysis

- Một pilot checkpoint, một seed, 768 synthetic rows; không có confidence interval.
- 500 epochs/4,000 updates không đại diện full TabDiff; baseline final vẫn có 55.21% guided-rule violation và withheld-rule gap lớn.
- Fixed/dynamic/repair baselines là proxies do analysis này viết, không phải paper reproductions.
- Main validity metric dùng chính hard rules làm action target; withheld audit giảm circularity nhưng chỉ có hai hard candidates và hai soft subgroup diagnostics.
- Chưa có human semantic annotation; executable verifier không đo toàn bộ semantic validity.
- Không chạy long-text branch; điều này có chủ đích vì datasets/backbone hiện tại không hỗ trợ.
- LCATS full paper không truy cập được qua ACM trong phiên; method được audit từ official code, nhưng không lặp lại claim evaluation từ abstract.

## 13. Follow-up hypothesis hẹp

Đã triển khai thêm temporal-persistence forecast, targeted recheck, critique compiler, timing sweep, replay-stability và mixed-type actuator audit. Kết quả/proposal cập nhật nằm ở [`NICHE_FOLLOWUP.md`](NICHE_FOLLOWUP.md). Tóm tắt: persistence gating đơn giản bị bác bỏ; targeted recheck chủ yếu bù cho weak x0 actuator; direct numeric state projection giảm violation xuống `1.76%` nhưng gần như trùng post-repair về fidelity/utility. Hướng đáng giữ là verified typed transaction/type-specific actuator trên constraint không local-repairable, không phải LLM judge trực tiếp.

## 14. Cập nhật full checkpoint và LLM-step sweep (2026-09-23)

Full 8,000-epoch Adult jobs `1028011/12/13/18` đã `COMPLETED`. Natural no-guidance hard violation chỉ còn khoảng `1.56–1.76%`; post-repair đạt `0%`, còn typed state projection tốt nhất khoảng `0.065–0.13%`. Kết quả này củng cố **NO-GO cho LLM nhằm enforce hard Adult rules**, vì backbone tốt gần như đã học chúng và static repair vẫn thắng.

Để kiểm tra phần semantic/dynamic thay vì tiếp tục thiếu positives tự nhiên, một synthetic semantic stress-test gồm `220` paired cases ở năm timestep đã được tạo bởi job `1028600`. Mười job critic/API, chín job end-to-end typed rollback và hai dependency aggregators đã được submit. Hypothesis, kill criteria, toàn bộ job IDs và đường dẫn đọc kết quả nằm trong [`LLM_STEP_SWEEP.md`](LLM_STEP_SWEEP.md). Các job pending không được dùng làm bằng chứng cho đến khi aggregate artifacts xuất hiện.

## 15. Local-LLM result và direction gate (2026-09-24)

OpenAI sweep không tạo model evidence vì API credit hết. Local Qwen3-8B sweep đã chạy thành công: temporal-pair condition tốt nhất nhưng chỉ đạt precision `0.525`, recall `0.21`, F1 `0.30`; lỗi tập trung gần như hoàn toàn ở age–education relation. Broad rubric tăng false positives; 2–3 timestep calls không hơn một late call. Vì vậy **NO-GO cho open-ended LLM judge**, còn **conditional GO cho relation-routed verifier → typed transaction**. Chi tiết failure decomposition, hypothesis H11–H14 và follow-up jobs nằm trong [`DIRECTION_DECISION_20260924.md`](DIRECTION_DECISION_20260924.md).

Update: API credit sau đó hoạt động trở lại. Focused candidate-relation verification đạt precision/recall/F1 `0.989/0.870/0.926` với GPT-4.1-mini và `0.978/0.870/0.921` với GPT-5.4-mini trên 220 controlled cases. Đây là evidence mạnh cho decomposition “route relation → verify”, nhưng chưa phải evidence rằng LLM hơn executable verifier hoặc mid-sampling hơn post-repair.
