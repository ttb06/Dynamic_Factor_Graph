# Định hướng ICDE: discovery tri thức, dynamic graph và độ dày thực nghiệm

## 1. Kết luận hiện tại

Bài chưa nên được đóng khung là “LLM tự tìm constraint rồi ép diffusion tuân theo”. Thực nghiệm hiện có cho thấy cách diễn đạt đó vừa khó bảo vệ vừa không đúng với bằng chứng:

- statistics-only mining tạo nhiều relation nhưng phần lớn là threshold implication đơn giản hoặc correlation đặc thù dataset;
- LLM có codebook và critic nghiêm ngặt chỉ giữ 2 relation mềm trên Default; cả hai chỉ đủ tư cách `score_only`, không đủ authority để sửa dữ liệu;
- lợi ích rõ nhất của method hiện tại nằm ở **khi nào và trên row nào một relation đã được xác thực nên ảnh hưởng tới reverse trajectory**, không nằm ở việc LLM là một hard verifier;
- trên common fidelity/utility, gain hiện còn nhỏ và không đồng đều; chưa có bằng chứng SOTA tổng quát.

Định vị hợp lý hơn:

> Một framework tạo và quản lý typed constraint graph từ nhiều nguồn tri thức, sau đó xây active graph riêng cho từng row và timestep để điều khiển reverse diffusion. External/expert constraints cấp hard authority; relation từ LLM hoặc thống kê chỉ được actuation sau semantic review, empirical verification và utility-preserving selection. Nếu chưa đủ bằng chứng, chúng chỉ làm score guardrail.

LLM vẫn có vai trò khoa học rõ ràng, nhưng giá trị của nó phải được đo bằng ablation `không LLM / LLM single-pass / LLM có counterexample loop`, không được giả định trước.

## 2. Vì sao constraint discovery hiện tại còn yếu

### 2.1 Statistics-only mining

Pipeline cũ tạo predicate từ category, zero/positive và quantile, rồi mine implication có confidence cao. Ưu điểm là executable, rẻ và dễ kiểm tra. Nhược điểm:

- dễ tìm lại nhiều threshold variant của cùng một relation;
- high confidence không chứng minh semantic validity;
- không biết hướng process/temporal thật;
- dễ nhầm rare support hoặc base-rate cao với hard constraint;
- graph nhiều edge nhưng effective semantic capacity thấp.

### 2.2 LCATS-style discovery

Code public LCATS đưa headers, type indices và 20 real rows cho LLM, giới hạn output vào bảy template, sau đó chỉ giữ candidate có zero violation trên full table. Cách này tạo executable constraint nhanh nhưng có ba rủi ro:

1. gửi real rows ra API, không phù hợp protocol riêng tư của dự án;
2. zero-violation filtering ưu tiên bound/ordering đơn giản và relation có support hẹp;
3. không phân biệt “đúng do process semantics” với “đúng tình cờ trong sample”.

Vì vậy project chỉ dùng một **LCATS-style privacy-safe proxy**: schema + aggregate, typed DSL, full-table verification; không gọi đó là reproduction.

Nguồn: [LCATS code](https://github.com/SJTU-DMTai/LCATS).

### 2.3 SAGE-style dependency discovery

SAGE không khám phá hard constraint. Method chia numerical value thành pseudo-feature, tính mutual information giữa current feature và past feature-value, rồi dùng sparse value-dependent dependency để chọn context/logit correction cho LLM generator. Đây là nguồn retrieval tốt cho dynamic relation, nhưng MI không cấp hướng, semantic validity hoặc authority.

Phần mới đã implement đúng ranh giới này: SAGE-style MI chỉ tạo `value_aware_dependency_signals`; LLM và verifier phải chuyển signal thành một typed hypothesis có semantics. Signal không bao giờ được actuation trực tiếp.

Nguồn: [SAGE, ACL 2026](https://aclanthology.org/2026.acl-long.174/).

### 2.4 EVS và agentic discovery

EVS tách Extract–Verify–Select: LLM là noisy oracle, data kiểm tra candidate, sau đó chọn subset giữ utility. Zhao et al. bổ sung machine-executable hypothesis, full-table validation và counterexample-guided revision. Hai bài này làm cho claim “LLM sinh executable constraints và kiểm tra bằng data” không còn mới nếu đứng riêng.

Phần project có thể khác biệt ở:

- constraint graph là typed graph có `applicability`, `requirements`, `exceptions`, `hardness`, `usage_role` và action endpoints;
- counterexample feedback là aggregate signature, không raw row;
- authority tách khỏi confidence: hard actuation, score-only và monitor;
- graph tiếp tục thay đổi theo row state và diffusion time sau discovery.

Nguồn: [EVS](https://openreview.net/pdf?id=1k9oK22A3R), [Zhao et al.](https://arxiv.org/abs/2608.15109).

## 3. Discovery protocol mới đã implement

### Stage A — Candidate retrieval

Ba nguồn được giữ tách biệt:

1. external metadata/codebook và published domain constraints;
2. SAGE-style value-aware MI trên train split;
3. empirical implication/mapping trên train split.

MI và empirical relation chỉ là retrieval evidence, không phải ground truth.

### Stage B — LLM semantic proposal

Các role độc lập đề xuất process hyperedge, conditional structure và exception/deactivation. Output phải theo typed DSL:

- `applicability`: khi edge có hiệu lực;
- `requirements`: điều kiện phải thỏa;
- `exceptions`: trạng thái deactivate;
- `relation_family`, `hardness_prior`, `evidence_basis`;
- directionality rationale và plausible counterexample.

Không gửi real row. Prompt chỉ nhận schema, aggregate distribution, codebook caveat và retrieval signals.

### Stage C — Deterministic execution và counterexample loop

Candidate được type-check và chạy trên discovery/counterexample splits. Phiên bản mới bổ sung aggregate counterexample signature:

- numeric median contrast giữa violating và satisfied cohorts;
- categorical enrichment của violating cohort;
- loại target, sensitive column và các column đã có trong edge.

Signature chỉ giúp LLM đặt giả thuyết applicability/exception ở vòng sau; không được tự động thêm predicate có correlation mạnh nhất.

### Stage D — Semantic judge và blind calibration

Independent critic/judge kiểm tra ambiguity, wrong direction, unsupported hardness, omitted fields và codebook limitations. Calibration split quyết định `hard`, `near_hard`, `soft` hoặc reject. Heldout chỉ mở sau khi graph đã freeze.

### Stage E — Authority assignment

- external/public constraint: có thể `actuate`;
- LLM relation được semantic review nhưng chưa đủ universal: `score_only`;
- relation chưa đủ ổn định: `monitor` hoặc reject.

Đây là điểm cần giữ trong paper: semantic confidence không đồng nghĩa actuation authority.

## 4. Kết quả discovery đã có

### Default codebook-aware run

| Graph | Proposed | Retained | Hard | Near-hard | Soft | Family | Token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Single-pass trước semantic judge | 3 | 2 | 0 | 2 | 0 | 2 | thuộc tổng dưới |
| Single-pass sau semantic judge | 3 | 2 | 0 | 0 | 2 | 2 | thuộc tổng dưới |
| Multi-role LLM protocol | 11 | 2 | 0 | 0 | 2 | 2 | 62,397 tổng |
| Statistics-only | 30 | 30 | 0 | 11 | 19 | 1 | 0 |
| Unified | 41 | 32 | 0 | 11 | 21 | 3 | 62,397 tổng |

Audit độc lập trên 2 edge LLM giữ lại:

| Relation | Train violations/support | Heldout violations/support | Authority |
| --- | ---: | ---: | --- |
| `BILL_AMT2 <= 0 -> PAY_AMT1 == 0` | 140/2,859 | 6/316 | score-only |
| `BILL_AMT2 == 0 -> PAY_AMT1 == 0` | 37/2,258 | 1/248 | score-only |

Hai relation có association thật nhưng cycle alignment không được codebook xác định. Kết luận: **LLM có ích để nêu soft process hypothesis, chưa đủ để cấp hard authority trên Default.**

## 5. Baseline đã chạy

### 5.1 Same-backbone Default: kết quả trung bình ba seed

Đây là evaluator chung trên cùng TabDiff checkpoint. Shape/Trend là error, thấp hơn tốt hơn.

| Method | Shape ↓ | Trend ↓ | Alpha ↑ | Beta ↑ | Detection ↑ | AUC ↑ | Heldout relation error ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TabDiff | 1.9833 | 2.2165 | 98.4736 | **52.5109** | 0.9528 | 0.7230 | 0.0309 |
| Dynamic graph | 1.9591 | 2.2116 | **98.5218** | 52.4436 | **0.9590** | 0.7228 | **0.0120** |
| Frozen graph | 1.9600 | 2.2119 | 98.4990 | 52.4533 | 0.9587 | 0.7238 | 0.0154 |
| Final repair | **1.9556** | **2.1909** | 98.5003 | 52.4761 | 0.9546 | **0.7278** | 0.0161 |
| Harpoon-style hard | 1.9772 | 2.2317 | 98.5284 | 52.4398 | 0.9570 | 0.7278 | 0.0179 |

Dynamic graph giảm heldout relation error 61.2% so với TabDiff, nhưng common fidelity/utility chỉ thay đổi rất nhỏ và AUC không tăng. Bảng này hỗ trợ row/edge calibration, chưa hỗ trợ SOTA general synthesis.

### 5.2 Official TabSyn rerun, cùng split/evaluator nhưng một run

| Dataset | Method | Shape error ↓ | Trend error ↓ | Alpha ↑ | Beta ↑ | Detection ↑ | XGBoost utility ↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Default | TabSyn official | 1.4810 | 1.6499 | 99.2198 | 47.7816 | 0.9615 | AUC 0.7127 |
| Shoppers | TabSyn official | 1.6652 | 1.9929 | 99.0039 | 46.3363 | 0.9709 | AUC 0.8817 |

Không được ghép các hàng này với mean ba seed như thể cùng uncertainty estimate. Nó cho thấy backbone choice vẫn ảnh hưởng lớn tới Shape/Trend.

### 5.3 Official Harpoon rerun bằng code tác giả, năm trials

| Dataset | Constraint | Violation accuracy/error field của code ↓ | Alpha ↑ | KS score ↑ | Detection ↑ | XGB utility ↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Adult | range | 28.0740 | 0.9220 | 0.9519 | 0.8684 | 0.9498 |
| Default | range | 4.8079 | 0.9003 | 0.9417 | 0.7988 | 0.7809 |
| Default | category | 17.6471 | 0.9478 | 0.9363 | 0.9284 | 0.7697 |
| Shoppers | category | 0.0000 | 0.5999 | 0.8974 | 0.8484 | 0.8741 |

Các constraint này là inference-time requests do Harpoon định nghĩa, không phải natural semantic constraints do discovery pipeline tìm ra. Vì vậy official rerun là baseline mechanism, không phải baseline discovery.

Nguồn: [Harpoon, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/e785d64baa2cde06c69d9b7a14636ef2-Abstract-Conference.html).

### 5.4 CDD-style categorical projection trên cùng TabDiff Default

| Method | Violation ↓ | Shape error ↓ | Trend error ↓ | Alpha ↑ | Beta ↑ | AUC ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Harpoon-style mạnh nhất | 32.5195 | 6.4278 | 3.5495 | 94.2149 | 41.8011 | 0.6622 |
| Final repair | 0.0000 | 6.5100 | 4.3902 | 96.3747 | 46.8280 | 0.7142 |
| CDD-style full trajectory | 0.0000 | **4.7070** | **3.5252** | 95.6308 | 46.9086 | 0.7054 |
| CDD-style delayed middle-to-late | 0.0000 | 5.5590 | 3.7373 | 95.5349 | 48.4140 | **0.7325** |

CDD paper không chạy tabular; đây là matched adaptation, không phải official reproduction. Kết quả quan trọng: projection bật từ state đã readable rồi duy trì tới cuối tốt hơn middle-only. Nguồn: [CDD, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/124c1cc9f7750f512ed81cdbf2fd5bf5-Abstract-Conference.html).

### 5.5 News với năm public constraints của ConstrainedDGM

Một seed, 4,096 samples, cùng checkpoint/evaluator:

| Method | Shape ↑ | Trend ↑ | Alpha ↑ | Beta ↑ | Detection ↑ | R² ↑ | NRMSE ↓ | Any violation ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TabDiff | **96.6554** | 89.1527 | **97.1327** | 42.8825 | 0.8883 | -0.0254 | 1.0126 | 25.8789% |
| Matched final repair | 96.6134 | 90.2583 | 97.0579 | **43.0046** | **0.8840** | -0.0285 | 1.0142 | **0.0000%** |
| Dynamic time-routed + certifier | 96.5904 | **90.6321** | 97.0373 | 42.9411 | 0.8803 | **-0.0104** | **1.0052** | **0.0000%** |

So với final repair, dynamic method tăng Trend 0.3738 điểm, R² 0.0181 và giảm NRMSE 0.0090; Shape/Alpha/Beta/Detection giảm nhẹ. Đây là Pareto trade-off có tín hiệu, chưa phải gain lớn hoặc uniform.

ConstrainedDGM constraint layer chuyển thẳng sample TabDiff sang feasible set đạt 0% violation và Trend 93.1562, nhưng Shape giảm từ 96.4965 xuống 96.1275 trong run khác. Official training-time TVAE rerun chưa hoàn chỉnh do device bug; matched TVAE samples đã có và evaluator đang chạy lại.

Nguồn: [Constrained Deep Generative Models, ICLR 2024](https://openreview.net/pdf?id=tBROYsEz9G).

### 5.6 Paper-reported SCTab-Diff: chỉ dùng làm context

Paper báo trên pipeline riêng:

| Dataset | Model | Shape ↑ | Trend ↑ | SA ↑ |
| --- | --- | ---: | ---: | ---: |
| News | TabDiff | 96.62 | 98.14 | **1.42** |
| News | SCTab-Diff | **96.90** | **98.60** | 1.24 |
| Shoppers | TabDiff | **97.94** | 97.88 | 99.19 |
| Shoppers | SCTab-Diff | 97.89 | **98.00** | **99.98** |
| Default | TabDiff | 98.37 | 96.46 | 99.97 |
| Default | SCTab-Diff | **98.57** | **97.89** | **100.00** |

Không so trực tiếp các số này với local News table: local TabDiff Trend chỉ khoảng 89–91, chứng tỏ split/checkpoint/evaluator khác đáng kể. Nguồn: [SCTab-Diff preprint](https://arxiv.org/html/2609.16069).

## 6. Các mechanism từ paper đang được đối chiếu

| Mechanism | Nguồn | Local status | Câu hỏi kiểm tra |
| --- | --- | --- | --- |
| Manifold/gradient guidance | Harpoon, ICLR 2026 | official rerun + matched proxy đã xong | dynamic graph có tốt hơn fixed inference request không? |
| Differentiable constraint layer | ConstrainedDGM, ICLR 2024 | same-sample transfer xong; matched TVAE đã train | training-time feasible layer có hơn online dynamic activation không? |
| KL feasibility projection | CDD, NeurIPS 2025 | matched categorical adaptation đã xong | categorical edge cần projection hay soft guidance? |
| Projection/primal–dual constrained sampling | Constrained Diffusers, NeurIPS 2025 | row-edge primal–dual jobs mới submit | adaptive multiplier có tăng feasibility với ít distortion hơn không? |
| Preview/lookahead/rollback | PPAD/TTGen line | matched branch/lookahead đã có; kết quả cũ không vượt rõ | chỉ giữ làm ablation, không mở rộng sweep hiện tại |
| Sparse value-aware graph | SAGE, ACL 2026 | MI retrieval mới tích hợp vào discovery | value-dependent retrieval có tăng semantic/public recall không? |
| Discover–verify–repair | LCATS, KDD 2026 | official code audited; privacy-safe proxy | LLM single-pass có thua counterexample loop không? |
| Extract–verify–select | EVS, 2026 workshop | selection principle đang tích hợp | satisfaction-only có chọn relation gây hại utility không? |

## 7. Jobs mới

| Job | Nội dung | Output dự kiến |
| --- | --- | --- |
| `1084361` | News discovery, SAGE retrieval + multi-role LLM + counterexample revision, budget 180k | `outputs/news_complex_discovery_1084361/` |
| `1084362` | counterfactual meaningfulness audit | `outputs/news_complex_discovery_audit_1084362/` |
| `1084363` | recall/direction trên năm public constraints bị giấu | `outputs/news_discovery_public_eval_1084363/` |
| `1084364` | merge public actuators với LLM score guardrails | `outputs/news_llm_guardrail_graph_1084364/` |
| `1084365[0-2]` | public-only guardrail và LLM guardrail weights 0.5/2.0 | `outputs/news_llm_guardrail_1084365_*` |
| `1084366[0-2]` | common News evaluation | `outputs/news_llm_guardrail_eval_1084366_*` |
| `1084380` | đánh giá lại matched ConstrainedDGM samples | `outputs/news_cdgm_matched_eval_1084380/` |
| `1084386[0-1]` | dynamic row-edge primal–dual, hai controller regimes | `outputs/news_primal_dual_1084386_*` |
| `1084387[0-1]` | common News evaluation cho primal–dual | `outputs/news_primal_dual_eval_1084387_*` |

Ba GPU jobs sửa correctness của balanced projection vẫn đang chờ priority: `1083909`, `1083969`, `1083996`.

## 8. Experimental matrix đủ dày cho ICDE

### Dataset axis

Nên có ít nhất bốn setting có authority khác nhau:

1. **News**: public hard numeric inequalities; benchmark chính cho mechanism.
2. **HELOC hoặc WiDS**: public constraints từ ConstrainedDGM; cross-dataset external graph.
3. **Insurance**: conditional constraint public trong LCATS; kiểm tra applicability thay đổi theo row.
4. **Shoppers**: mixed-type process relations; kiểm tra categorical/numeric active graph.

Default chỉ nên là soft-relation stress test, không phải dataset chính cho hard semantic validity.

### Baseline axis

Main table cần ít nhất:

- TabDiff unconditional;
- TabSyn hoặc backbone generative mạnh gần đây;
- final repair/projection;
- best-of-N + certifier;
- ConstrainedDGM layer;
- Harpoon;
- CDD/CTDF-style per-step projection;
- frozen graph;
- dynamic graph của project.

Discovery table tách riêng:

- statistics-only;
- SAGE-style value-aware retrieval;
- LLM single-pass/LCATS-style proxy;
- LLM Extract–Verify–Select;
- multi-role counterexample protocol;
- expert/public oracle.

### Metric axis

Primary:

- public/expert constraint violation;
- Shape, Trend;
- downstream AUC/F1 hoặc R²/NRMSE;
- Alpha precision, Beta recall.

Secondary:

- detection score và DCR/privacy proxy;
- valid-exception change rate khi có conditional constraint;
- action/edit cost, number of denoiser passes, runtime;
- LLM calls/tokens;
- trajectory precision/recall, edge activation flips và residual final repair.

Discovery-specific:

- recall và direction accuracy trên withheld public constraints;
- executable/typecheck rate;
- semantic precision qua blinded human review;
- heldout stability và support;
- utility loss sau enforcing selected subset;
- redundancy/effective graph capacity.

## 9. Điều kiện quyết định sau batch mới

LLM đáng giữ trong main method nếu thỏa ít nhất một trong hai:

1. multi-role/counterexample protocol tăng withheld public recall hoặc semantic precision rõ so với single-pass và statistics-only;
2. LLM score guardrails tăng Trend/utility hoặc giảm edit/residual repair so với public-only guardrail ở cùng 0% public violation.

Nếu không thỏa, LLM vẫn có thể nằm ở knowledge acquisition interface và ablation, nhưng không nên là core claim.

Dynamic graph đáng làm main contribution nếu primal–dual hoặc edge-specific routing đạt 0% public violation và tạo improvement nhất quán hơn final repair trên ít nhất một standard fidelity metric và một utility metric, sau đó lặp lại trên dataset public thứ hai.

Hiện tại project **đủ để bắt đầu viết problem, method và experimental protocol**, nhưng **chưa đủ để đóng main result hoặc claim SOTA**. Hai thiếu hụt quyết định là cross-dataset public constraints và significant matched gain trên common metrics.

## 10. Cập nhật kết quả ngày 16/10/2026

### 10.1 So sánh mechanism trên cùng News sample

Các hàng dưới đây dùng cùng checkpoint, seed `20261447`, 4,096 sample và năm public constraints. `Dynamic + certifier` là sample sau guidance giữa trajectory rồi chỉ dùng certifier cuối để xóa residual violation.

| Mechanism | Shape ↑ | Trend ↑ | Alpha ↑ | Beta ↑ | Detection ↑ | R² ↑ | NRMSE ↓ | Violation ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TabDiff | **96.6554** | 89.1527 | **97.1327** | 42.8825 | 0.8883 | -0.0254 | 1.0126 | 25.8789% |
| Matched final repair | 96.5951 | 90.2825 | 97.0486 | **43.0046** | **0.9050** | -0.0162 | 1.0081 | **0.0000%** |
| Dynamic balanced L2 + certifier | 96.5864 | **90.6927** | 97.1024 | 42.8711 | 0.8831 | **-0.0054** | **1.0027** | **0.0000%** |
| ICLR-2024 constraint layer | 96.2996 | 90.5950 | **97.1344** | 42.8988 | 0.8965 | -0.0293 | 1.0146 | **0.0000%** |

So với matched final repair, dynamic balanced L2 tăng **0.4102 điểm Trend**, tăng **0.0108 R²**, giảm **0.0054 NRMSE** và tăng nhẹ Alpha; đổi lại Shape giảm 0.0087, Beta giảm 0.1335 và Detection giảm 0.0219. Vì vậy đây là Pareto improvement về dependency fidelity và regression utility, không phải thắng đồng đều mọi metric.

So với official ICLR-2024 constraint layer được áp lên **đúng cùng unconditional sample**, dynamic method vẫn cùng đạt 0% violation nhưng tăng **0.2867 Shape**, **0.0977 Trend**, **0.0239 R²** và giảm **0.0119 NRMSE**. Constraint layer tốt hơn 0.0320 Alpha, 0.0277 Beta và 0.0134 Detection. Đây là matched evidence rõ nhất cho giá trị của trajectory intervention so với final-only feasibility projection.

### 10.2 Ablation về thời điểm và controller

| Variant sau certifier | Trend ↑ | R² ↑ | NRMSE ↓ | Online violation trước certifier ↓ | Online actions ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed middle switch, balanced L2 | **90.6927** | **-0.0054** | **1.0027** | 2.16% | 2,389 |
| Readability-routed switch | 90.6385 | -0.0141 | 1.0070 | 1.62% | 2,282 |
| Per-edge reliability routing | 90.5499 | -0.0209 | 1.0104 | **1.44%** | **1,834** |
| Primal–dual conservative | 90.4077 | -0.0162 | 1.0081 | 1.99% | 3,274 |
| Primal–dual responsive | 90.4152 | -0.0178 | 1.0088 | 1.78% | 3,793 |

Kết luận:

- per-edge routing giảm residual violation và số action, nhưng không cải thiện common fidelity/utility; không nên dùng làm main variant;
- primal–dual phản ứng đúng với violation nhưng tốn nhiều action hơn và không vượt fixed balanced L2; hypothesis này bị bác bỏ ở cấu hình hiện tại;
- fixed middle switch vẫn là lựa chọn tốt nhất cho main result. Đây là một negative result hữu ích: controller phức tạp hơn không tự động tạo sample tốt hơn.

Instantaneous-gate baseline mới giữ nguyên middle window nhưng bỏ trajectory-confidence/persistence. Trước final certifier, nó dùng 2,696 actions và còn 2.66% violation; main annealed trajectory gate dùng 2,389 actions và còn 2.16%. Như vậy main gate giảm **11.4% action** đồng thời giảm **18.8% residual violation**. Common fidelity/utility của baseline này đang được tính trong job `1085118`.

### 10.3 Matched ConstrainedDGM training baseline

| Model | Shape ↑ | Trend ↑ | Alpha ↑ | Beta ↑ | R² ↑ | NRMSE ↓ | Violation ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Matched TVAE | 78.3665 | **76.5262** | 10.4054 | **0.0002** | 0.0216 | 0.9891 | 30.5558% |
| Matched constrained TVAE | 73.0569 | 75.9558 | **37.3452** | 0.0000 | **0.0336** | **0.9831** | **0.0000%** |

Constraint-aware training đạt feasibility và tăng regression utility so với đúng TVAE base, nhưng làm giảm Shape/Trend vốn đã thấp hơn TabDiff rất nhiều. Kết quả này ủng hộ việc giữ backbone diffusion mạnh và tập trung vào inference-time graph guidance, thay vì đổi core generator sang constrained TVAE.

### 10.4 Trạng thái discovery dùng LLM

Job `1084361` ban đầu hoàn thành 12 API calls, dùng 175,388/180,000 token rồi dừng trước `judge_multi_role_1`; đây là lỗi budget guard, không phải API/model failure. Resume `1085075` dùng lại cache và hoàn tất với 14 calls, 188,790 token. Multi-role protocol giữ 14/32 edge, 6 relation families và 11 endpoint patterns, so với single-pass giữ 3/6 edge thuộc một family. Exact recall trên năm public constraints bị giấu chỉ đạt 0.20 cho cả single-pass và multi-role; statistics-only và SAGE-style value retrieval đạt 0.00. Vì public list không exhaustive, đây là recall chứ không phải precision.

Audit observational ban đầu gắn cờ mọi conjunction có 0 natural violation là redundant. Kết luận đó không hợp lệ cho hard invariant. Audit đã được sửa bằng minimally mutated executor-level counterexamples: mỗi atom được chủ động làm sai trong khi giữ các atom khác đúng. Sau sửa, 0/29 atom executable bị đánh dấu counterfactually redundant; graph gồm:

- 2 `actuation_candidate`: signed/absolute title-polarity identity và singleton self-reference min=mean=max;
- 2 `human_review`: self-link subset và nonempty self-reference ordering;
- 4 soft `score_only` zero-state/process dependencies;
- 6 reject/revise, gồm hai duplicate, các distributional relations thiếu codebook semantics và một temporal correlation không có process justification.

Các proposal cho thấy cả hai mặt cần báo cáo:

- giữ được hard subset relation `num_self_hrefs <= num_hrefs` và một số soft min/average/max hoặc zero-state process relations;
- loại các edge sai vị trí applicability, tolerance làm relation trở nên vacuous và hardness không được codebook hỗ trợ.

Không edge LLM nào được tự động actuate. Chỉ graph đã qua judge, calibration và corrected independent audit mới được merge; LLM-only edges vẫn là `score_only`. Job guardrail đầu tiên dùng audit lỗi nên có 0 LLM edge và chỉ là no-op control; corrected guardrail ablation là chuỗi `1085107`–`1085110`.

Corrected graph có 7 incremental score guards sau khi bỏ một edge trùng public constraint. Tuy nhiên public-only, LLM weight 0.5 và LLM weight 2.0 sinh ra các file `x0_reward_calibrated.csv` có **cùng SHA-256**, và output sau certifier cũng giống hệt. Do đó LLM guardrail không thay đổi bất kỳ action decision nào trên News. Đây là negative result trực tiếp cho incremental value của LLM trong setting hiện tại: các relation mới hoặc đã được backbone tôn trọng, hoặc không giao với repair endpoints của năm public constraints. LLM discovery nên là module tùy chọn/ablation, chưa nên là core performance claim.

Phân tích overlap cụ thể cho thấy public guidance tự giải quyết 197 LLM-edge violation events và chỉ tạo mới 7 events trên tổng 4,096 rows; các alternative repair candidates không làm score weight 0.5 hoặc 2.0 đổi lựa chọn. Vì vậy không có cơ sở để mở một weight sweep lớn chỉ nhằm tạo chênh lệch: phạm vi tác động tiềm năng quá nhỏ để cải thiện common fidelity/utility.

### 10.5 Chuỗi job mới

| Job | Mục đích |
| --- | --- |
| `1085075` | resume News discovery từ cache cũ, trần tổng 240k token |
| `1085076` | independent meaningfulness audit sau discovery |
| `1085077` | exact recall/direction trên năm public constraints bị giấu |
| `1085078` | build public actuator + audited LLM score-guardrail graph |
| `1085079[0-2]` | public-only so với LLM guardrail weights 0.5 và 2.0 |
| `1085080[0-2]` | common evaluator cho guardrail ablation |
| `1085081` | lần đầu same-sample evaluator; constraint sample tạo được nhưng evaluator gặp lỗi output directory |
| `1085082` | rerun đã hoàn tất: official ICLR-2024 layer trên đúng unconditional sample |
| `1085107`–`1085110` | corrected counterfactual audit, graph có 7 incremental LLM score guards, guidance và common evaluation |
| `1085117[0-1]` | full per-step projection proxy và middle-window/current-state gate baseline |
| `1085118[0-1]` | common News evaluator cho hai baseline trên |

Sau batch này, quyết định giữ LLM trong main method dựa trên incremental value so với public-only graph, không dựa trên số edge mà LLM đề xuất. Quyết định về main mechanism dựa trên same-sample comparison với ConstrainedDGM layer; nếu fixed middle guidance không tạo trade-off tốt hơn, contribution cần đóng khung là trajectory-aware Pareto control thay vì SOTA fidelity thuần túy.
