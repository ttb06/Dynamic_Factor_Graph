# Catch-up: Verified dynamic constraint graph cho tabular diffusion

Ngày tổng hợp: 02/10/2026

## 1. Bài toán hiện tại

Dự án không còn tập trung hẹp vào semantic request hoặc gọi LLM tại mọi diffusion step. Formulation hiện tại là:

> Xây dựng một verified dynamic constraint graph kết hợp rule, evidence thống kê và tri thức từ foundation model; sau đó quyết định edge nào active, inactive hoặc abstain cho từng row và từng trạng thái của reverse diffusion.

Graph gồm relation, direction, subgroup/applicability, exception, hardness, confidence và provenance. LLM là candidate proposer/compiler; typed executor và statistical verifier mới quyết định edge có được phép tác động hay không.

## 2. Bằng chứng mạnh nhất đã có

### 2.1 Graph chỉ trở nên đọc được ở mid/late trajectory

Probe: 2.048 synthetic rows, 50 reverse steps, 12 subgroup edges, Default, một seed.

| Timestep chuẩn hóa | Edge precision | Edge recall | Edge accuracy |
| --- | ---: | ---: | ---: |
| `(0.75,1.00]` | 0.6688 | 0.2416 | 0.9272 |
| `(0.50,0.75]` | 0.8162 | 0.6976 | 0.9657 |
| `(0.25,0.50]` | 0.9216 | 0.9311 | 0.9895 |
| `(0.10,0.25]` | 0.9874 | 0.9926 | 0.9986 |
| `[0,0.10]` | 0.9999 | 1.0000 | 1.0000 |

Ý nghĩa: early preview không đủ đáng tin để bật graph; method cần abstention, persistence hoặc uncertainty gate. Đây là bằng chứng trực tiếp nhất cho trajectory-aware graph.

### 2.2 Static enforcement phá valid exceptions

| Dataset | Method | Satisfaction | Valid exceptions bị đổi | Target KS |
| --- | --- | ---: | ---: | ---: |
| Default | Unconditional | 0.7024 | 0.0000 | 0.3256 |
| Default | Static final repair | 1.0000 | **0.9744** | 0.3129 |
| Default | Subgroup final repair | 1.0000 | 0.0000 | 0.3129 |
| Default | Best-of-8 graph + certify | 1.0000 | 0.0000 | **0.1586** |
| Default | Mid lookahead + certify | 1.0000 | 0.0000 | 0.2335 |
| Shoppers | Unconditional | 0.4470 | 0.0000 | 0.5642 |
| Shoppers | Static final repair | 1.0000 | **0.6491** | 0.5642 |
| Shoppers | Subgroup final repair | 1.0000 | 0.0000 | 0.5642 |
| Shoppers | Best-of-8 graph + certify | 1.0000 | 0.0000 | 0.2685 |
| Shoppers | Mid lookahead + certify | 1.0000 | 0.0000 | **0.2259** |

Ý nghĩa: contribution không nên là đạt 100% validity; static repair cũng làm được. Vấn đề đáng nghiên cứu là đạt validity mà không xóa valid subgroup exceptions và không làm hỏng conditional distribution.

### 2.3 Không có intervention stage hoặc solver tốt nhất cho mọi objective

| Objective | Mid gain so với best non-mid | Mid win fraction |
| --- | ---: | ---: |
| Balanced | +0.0118 | 0.6000 |
| Intervention-conservative | +0.0370 | 0.7667 |
| Joint fidelity | **+0.0550** | 0.6667 |
| Target fidelity | **−0.1004** | 0.2667 |

Mid-denoising có ích cho joint structure và preservation, nhưng thua final repair/resampling khi chỉ tối ưu target marginal. Điều này support objective-conditioned controller, không support claim rằng mid-guidance luôn tốt hơn repair.

### 2.4 Statistical graph có nhiều relation không thể hard-enforce

Cross-dataset audit trên Adult, Default và Shoppers tìm 788 candidates:

- 80 empirical zero-exception candidates;
- 95 near-hard dynamic candidates;
- 334 probabilistic relations;
- 132 insufficient-support candidates;
- 85 demographic/sensitive relations chỉ được audit;
- 5 exception-rate shift cases.

Kết luận: phần lớn relation được khai phá không phải invariant. Dynamic graph cần hardness, subgroup scope và abstention; graph discovery đơn thuần chưa đủ.

### 2.5 LLM phù hợp làm compiler hơn direct row judge

Trên request-conditioned synthetic bank 140 cases:

| Method | Four-way accuracy | Swap-pair exact | Token usage |
| --- | ---: | ---: | ---: |
| GPT-4.1 direct judgement | 0.8643 | 0.8667 | 37.544 |
| GPT-5.4 direct judgement | 0.6500 | 0.3667 | 37.699 |
| GPT-4.1 NL→DSL→executor | **1.0000** | **1.0000** | 1.846 |
| GPT-5.4 NL→DSL→executor | **1.0000** | **1.0000** | 1.853 |

Template/oracle parser cũng đạt 1.0 trên grammar kiểm soát. Vì vậy chưa có evidence rằng LLM cần thiết khi rule đã executable; vai trò hợp lý là open-ended semantic proposal, cross-schema transfer và compile-once interface.

## 3. Baseline đáng báo cáo

### 3.1 CuTS official self-run — Adult

CuTS đạt satisfaction 1.0 trên cả năm public programs. Balanced accuracy giữ khoảng 0.772–0.776. Năm repeated sampling runs dùng cùng trained model, không phải năm training seeds.

### 3.2 LCATS official output audit — Insurance

| Output | Violation | Numeric KS | Propensity AUC | TSTR R² |
| --- | ---: | ---: | ---: | ---: |
| TabSyn original | 0.0822 | 0.0359 | 0.5071 | 0.7212 |
| LCATS aligned | **0.0017** | **0.0318** | 0.5070 | **0.7224** |

LCATS là post-repair baseline rất mạnh. Nó sửa tất cả rows và 28.84% cells, dù normalized numeric L1 chỉ 0.00486. Dynamic method phải chứng minh lợi ích về selective intervention, exception preservation hoặc joint fidelity.

### 3.3 TabDiff trên public CuTS programs

Implication2/4 gần bão hòa ngay khi unconditional. Với line constraints, repair/best-of-8 đạt validity 1.0 nhưng fidelity giảm mạnh; ví dụ line1 categorical TV tăng từ 0.0155 lên 0.0890 với final repair và 0.2002 với best-of-8. Đây là bằng chứng validity–fidelity trade-off.

### 3.4 Paper-reported baselines

- SAGE có value-conditioned sparse graph nhưng không thống trị constraint validity: Adult education violation của SAGE-LC là 16.47%, so với GReaT 0% và TabSyn 2.32%.
- Harpoon mạnh trên range/AND/OR constraints nhưng vẫn còn khoảng 3–21% violation tùy dataset/constraint; đây là baseline trực tiếp cho per-step guidance.
- CuTS đạt 100% CSR bằng rejection sampling/fine-tuning + rejection sampling, tiếp tục cho thấy final validity không đủ phân biệt method.

## 4. Các kết quả không được dùng làm positive claim

1. Focused LLM result khoảng 0.926 F1 bị label leakage qua `case_id` và candidate `none`; không hợp lệ làm scientific evidence.
2. Full Adult checkpoint chỉ có khoảng 1.6–1.8% natural hard-rule violation; setting này quá bão hòa để chứng minh LLM cần thiết.
3. Direct LLM critique theo row vừa kém ổn định vừa đắt hơn typed executor.
4. Learned cross-relation solver router chỉ cải thiện nhỏ ở hai trong bốn objectives và kém fixed solver ở hai objective còn lại.
5. Dynamic monitoring giúp News/Shoppers nhưng không xử lý tốt transient shift ở Beijing; không thể claim dynamic graph luôn hơn frozen graph.
6. Mid-denoising không luôn hơn best-of-N hoặc post-repair.

## 5. Positioning so với related work

SAGE đã bao phủ value-dependent graph trong autoregressive LLM generation. SCTab-Diff đã đưa LLM prior vào reverse denoising. Harpoon và CTDF đã áp constraint trong diffusion steps. Zhao et al. và LCATS đã bao phủ LLM rule discovery, validation và post-repair.

Khoảng trống còn lại có thể bảo vệ là:

> Hybrid constraint knowledge được kiểm chứng trên dataset, có provenance và explicit/implicit applicability scope; active graph thay đổi theo row và uncertain diffusion state; controller có active/inactive/abstain và bảo toàn valid exceptions.

Không claim “đầu tiên dùng LLM trong diffusion”, “đầu tiên có dynamic graph”, hoặc “đầu tiên dùng LLM tạo rule”.

## 6. Mức độ sẵn sàng hiện tại

### Đã có

- evidence mạnh cho trajectory readability;
- evidence mạnh cho exception damage của static repair;
- cross-dataset statistical graph audit;
- public constraint baselines;
- official CuTS Adult và LCATS audit;
- visualization graph theo Default, News, Beijing;
- bằng chứng bác bỏ direct LLM judgement.

### Chưa có

- end-to-end method tự tạo verified hybrid graph rồi dùng graph đó trong sampler;
- official Harpoon rerun: environment setup job `1052791` thất bại, downstream jobs chưa chạy;
- CuTS German constrained run `1052789` vẫn đang chạy, audit đang chờ;
- trajectory trace đa dataset tương đương Default;
- independent semantic ground truth cho graph edges;
- kết quả nhiều seed và statistical significance cho final method.

## 7. Kết luận cho buổi catch-up

1. Research question đã rõ hơn: không phải LLM feedback nói chung, mà là verified dynamic graph dưới row- và trajectory-dependent applicability.
2. Bằng chứng hiện tại support feasibility, nhưng chưa phải state-of-the-art result.
3. Kết quả tốt nhất là graph không đọc được ở early trajectory và static enforcement phá 65–97% valid exceptions trong controlled settings.
4. Kết quả phản bác quan trọng là direct LLM judgement, universal mid-guidance và universal dynamic routing đều chưa được support.
5. Method tiếp theo nên tối giản: hybrid candidate graph → typed validation → explicit/implicit scope → three-state trajectory gate → cùng một actuator so với final repair và best-of-N.

## 8. Artifact nên mở trong buổi họp

- `BANG_TONG_HOP_PAPER_BASELINE_VA_KET_QUA_20261002.md`
- `outputs/dynamic_graph_visualization_20261002_v3/dataset_subgroup_graphs.png`
- `outputs/dynamic_graph_visualization_20261002_v3/default_edge_activation_by_timestep.png`
- `outputs/dynamic_graph_visualization_20261002_v3/RESULTS.md`
