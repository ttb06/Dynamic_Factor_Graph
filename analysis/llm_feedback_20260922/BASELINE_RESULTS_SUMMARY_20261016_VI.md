# Tổng hợp kết quả baseline — 16/10/2026

## 1. Quy tắc đọc bảng

- Chỉ các hàng trong cùng một bảng **local matched** mới được xếp hạng trực tiếp.
- News báo `Shape/Trend score` nên **cao hơn tốt hơn**. Các bảng Default cũ báo `Shape/Trend error` nên **thấp hơn tốt hơn**.
- `Paper-reported` không được trộn với local result vì split, checkpoint, sample count và evaluator khác.
- `Proxy/adaptation` nghĩa là mượn mechanism của paper và chạy trên TabDiff; không gọi là official reproduction.
- Main result hiện dùng một seed và 4,096 synthetic rows; chưa có uncertainty estimate.

## 2. Bảng baseline chính: News, cùng unconditional sample và evaluator

Năm hard inequalities lấy từ public constraint file của ConstrainedDGM. Tất cả phương pháp nhận cùng TabDiff sample trước can thiệp.

| Method | Loại baseline | Shape ↑ | Trend ↑ | Alpha ↑ | Beta ↑ | Detection ↑ | R² ↑ | NRMSE ↓ | Violation ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TabDiff | unconditional backbone | **96.6554** | 89.1527 | 97.1327 | 42.8825 | 0.8883 | -0.0254 | 1.0126 | 25.8789% |
| Final repair | post-generation repair | 96.5951 | 90.2825 | 97.0486 | **43.0046** | **0.9050** | -0.0162 | 1.0081 | **0.0000%** |
| ConstrainedDGM layer | official ICLR-2024 feasibility operator, final-only | 96.2996 | 90.5950 | **97.1344** | 42.8988 | 0.8965 | -0.0293 | 1.0146 | **0.0000%** |
| Dynamic middle + certifier | method hiện tại | 96.5864 | **90.6927** | 97.1024 | 42.8711 | 0.8831 | **-0.0054** | **1.0027** | **0.0000%** |

Gain của method hiện tại so với final repair:

- Trend `+0.4102` điểm;
- R² `+0.0108`;
- NRMSE `-0.0054`;
- đổi lại Detection `-0.0219`, Shape `-0.0087`, Beta `-0.1335`.

Gain so với ConstrainedDGM layer trên đúng cùng sample:

- Shape `+0.2867`;
- Trend `+0.0977`;
- R² `+0.0239`;
- NRMSE `-0.0119`;
- cùng đạt 0% violation;
- đổi lại Alpha `-0.0320`, Beta `-0.0277`, Detection `-0.0134`.

Kết luận: method là một Pareto improvement rõ trên dependency fidelity và regression utility, chưa thắng đồng đều tất cả metric.

## 3. News mechanism baselines, cùng seed/checkpoint

| Variant sau final certifier | Trend ↑ | R² ↑ | NRMSE ↓ | Online violation trước certifier ↓ | Online actions ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed middle, balanced L2 | **90.6927** | **-0.0054** | **1.0027** | 2.16% | 2,389 |
| Readability-routed switch | 90.6385 | -0.0141 | 1.0070 | 1.62% | 2,282 |
| Per-edge reliability routing | 90.5499 | -0.0209 | 1.0104 | **1.44%** | **1,834** |
| Primal–dual conservative | 90.4077 | -0.0162 | 1.0081 | 1.99% | 3,274 |
| Primal–dual responsive | 90.4152 | -0.0178 | 1.0088 | 1.78% | 3,793 |

Instantaneous middle gate dùng 2,696 actions và còn 2.66% online violation. Trajectory gate của main dùng 2,389 actions và còn 2.16%, tương ứng giảm 11.4% action và 18.8% residual violation.

Full per-step projection proxy đang chạy ở job `1085117_0`; common evaluation là job `1085118`. Không điền số tạm thời.

## 4. ConstrainedDGM training baseline, News nhưng generator riêng

Đây là official code với matched News split, nhưng dùng TVAE thay vì TabDiff nên không so trực tiếp với bảng 2.

| Model | Shape ↑ | Trend ↑ | Alpha ↑ | Beta ↑ | R² ↑ | NRMSE ↓ | Violation ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TVAE | **78.3665** | **76.5262** | 10.4054 | **0.0002** | 0.0216 | 0.9891 | 30.5558% |
| Constrained TVAE | 73.0569 | 75.9558 | **37.3452** | 0.0000 | **0.0336** | **0.9831** | **0.0000%** |

Constraint-aware training tăng feasibility và utility so với đúng TVAE base nhưng giảm Shape/Trend. Nó không thay thế được một diffusion backbone mạnh trong setting hiện tại.

## 5. Default: cùng TabDiff evaluator, trung bình ba seed

Ở bảng này Shape/Trend là **error**, thấp hơn tốt hơn.

| Method | Shape ↓ | Trend ↓ | Alpha ↑ | Beta ↑ | Detection ↑ | AUC ↑ | Heldout relation error ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TabDiff | 1.9833 | 2.2165 | 98.4736 | **52.5109** | 0.9528 | 0.7230 | 0.0309 |
| Dynamic graph | 1.9591 | 2.2116 | **98.5218** | 52.4436 | **0.9590** | 0.7228 | **0.0120** |
| Frozen graph | 1.9600 | 2.2119 | 98.4990 | 52.4533 | 0.9587 | 0.7238 | 0.0154 |
| Final repair | **1.9556** | **2.1909** | 98.5003 | 52.4761 | 0.9546 | **0.7278** | 0.0161 |
| Harpoon-style hard | 1.9772 | 2.2317 | 98.5284 | 52.4398 | 0.9570 | 0.7278 | 0.0179 |

Dynamic graph giảm heldout relation error 61.2% so với TabDiff, nhưng common fidelity/utility chỉ thay đổi nhỏ và không vượt final repair đồng đều.

## 6. Harpoon request trên Default: matched TabDiff proxy

Constraint và request semantics theo Harpoon; CDD là adaptation của categorical feasibility projection.

| Method | Violation ↓ | Shape error ↓ | Trend error ↓ | Alpha ↑ | Beta ↑ | Conditional AUC ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unconditional | 84.9121% | 10.0480 | 4.3902 | 75.5645 | 23.4274 | 0.6864 |
| Harpoon-style strong | 32.5195% | 6.4278 | 3.5495 | 94.2149 | 41.8011 | 0.6622 |
| Final repair | **0.0000%** | 6.5100 | 4.3902 | **96.3747** | 46.8280 | 0.7142 |
| CDD-style full trajectory | **0.0000%** | **4.7070** | **3.5252** | 95.6308 | 46.9086 | 0.7054 |
| CDD delayed middle-to-late | **0.0000%** | 5.5590 | 3.7373 | 95.5349 | **48.4140** | **0.7325** |

Đây là evidence rằng full projection tối ưu Shape/Trend, trong khi delayed middle-to-late tốt hơn về conditional AUC/Beta. CDD paper không chạy tabular; hai hàng CDD là proxy baseline.

## 7. Official public-code reruns

### TabSyn — ICLR 2024, một local run

| Dataset | Shape error ↓ | Trend error ↓ | Alpha ↑ | Beta ↑ | Detection ↑ | XGBoost utility ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Default | 1.4810 | 1.6499 | 99.2198 | 47.7816 | 0.9615 | AUC 0.7127 |
| Shoppers | 1.6652 | 1.9929 | 99.0039 | 46.3363 | 0.9709 | AUC 0.8817 |

TabSyn dùng generator khác. Kết quả cho thấy backbone choice ảnh hưởng Shape/Trend lớn hơn phần lớn gain do guidance.

### Harpoon — ICLR 2026, official code/checkpoint, năm trials

Các metric dưới đây dùng scale của official Harpoon evaluator (`Alpha/KS/Detection/utility` trong khoảng 0–1). `Violation field` giữ nguyên tên/semantics của code tác giả; thấp hơn tốt hơn.

| Dataset | Request | Violation field ↓ | Alpha ↑ | KS ↑ | Detection ↑ | XGB utility ↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Adult | range | 28.0740 | 0.9220 | 0.9519 | 0.8684 | 0.9498 |
| Default | range | 4.8079 | 0.9003 | 0.9417 | 0.7988 | 0.7809 |
| Default | category | 17.6471 | 0.9478 | 0.9363 | 0.9284 | 0.7697 |
| Shoppers | category | **0.0000** | 0.5999 | 0.8974 | 0.8484 | 0.8741 |

Official Harpoon constraints là inference-time requests, không phải semantic relations do discovery pipeline tìm ra. Vì thế baseline này đánh giá mechanism, không đánh giá discovery.

## 8. Paper-reported context: không xếp hạng với local table

### SCTab-Diff — preprint, 09/2026

| Dataset | Model | Shape ↑ | Trend ↑ | SA ↑ |
| --- | --- | ---: | ---: | ---: |
| News | TabDiff | 96.62 | 98.14 | **1.42** |
| News | SCTab-Diff | **96.90** | **98.60** | 1.24 |
| Shoppers | TabDiff | **97.94** | 97.88 | 99.19 |
| Shoppers | SCTab-Diff | 97.89 | **98.00** | **99.98** |
| Default | TabDiff | 98.37 | 96.46 | 99.97 |
| Default | SCTab-Diff | **98.57** | **97.89** | **100.00** |

Local News TabDiff Trend chỉ khoảng 89–91 trong pipeline hiện tại, khác đáng kể paper-reported 98.14. Do đó không được tuyên bố local method vượt SCTab-Diff từ hai bảng này.

## 9. Bảng nào nên dùng trong paper

Main table nên dùng **bảng 2** và bổ sung full per-step proxy sau khi job hoàn tất:

1. TabDiff;
2. final repair;
3. ConstrainedDGM final constraint layer;
4. full per-step projection/CTDF-style proxy;
5. instantaneous middle gate;
6. dynamic middle trajectory method.

Secondary tables:

- bảng 5 cho cross-seed Default robustness;
- bảng 6 cho Harpoon/CDD mechanism comparison;
- bảng 7 để chứng minh đã chạy official public-code baselines;
- bảng 8 chỉ làm context và ghi rõ paper-reported.

Không nên dùng constraint satisfaction trên graph tự discovery làm metric chính. Main claim phải dựa trên public hard constraints, Shape/Trend, downstream utility và matched edit/runtime cost.
