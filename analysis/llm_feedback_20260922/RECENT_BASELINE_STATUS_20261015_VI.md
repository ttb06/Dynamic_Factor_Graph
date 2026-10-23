# Trạng thái baseline gần đây và mức độ sẵn sàng viết bài

## 1. Kết luận ngắn

Phương pháp hiện tại đã có một kết quả cơ chế ổn định trên **Default**: so với cùng TabDiff checkpoint, dynamic graph làm giảm `actionable held-out violation error` **68.35%**, và so với final repair làm giảm **19.38%**, thắng cả ba seed. Tuy nhiên, phương pháp mới chỉ gần như hòa về fidelity/utility phổ biến: Shape tốt hơn 0.14%, Trend tốt hơn 0.55%, còn XGBoost AUC thấp hơn 0.06% so với TabDiff.

Vì vậy có thể bắt đầu viết problem formulation, method, protocol và phần mechanism study; chưa nên khóa claim “state of the art về data fidelity”. Main claim phù hợp với bằng chứng hiện tại là: **dynamic row–timestep graph activation giảm lỗi semantic ngoài mẫu tốt hơn final-only correction trong khi gần như bảo toàn fidelity và utility**.

## 2. Kết quả chính đã xác nhận, Default, ba seed

Mỗi seed dùng 4.096 bản ghi, cùng checkpoint, split, graph đánh giá độc lập và final certifier.

| Method | Shape error ↓ | Trend error ↓ | Alpha ↑ | Beta ↑ | Detection ↑ | XGB AUC ↑ | Graph error ↓ | Full held-out error ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TabDiff | 2.0415 | 1.9718 | 98.7555 | 52.6362 | **0.9446** | 0.7305 | 0.01611 | 0.01726 |
| Final repair | 2.0418 | 1.9718 | 98.7555 | 52.6367 | 0.9444 | **0.7318** | 0.00632 | 0.01119 |
| **Dynamic graph theo trajectory** | **2.0386** | **1.9610** | **98.7636** | **52.6373** | 0.9441 | 0.7300 | **0.00510** | **0.01029** |

So với TabDiff, gain tương đối lần lượt là 0.14% Shape, 0.55% Trend, **68.35% graph error** và **40.38% full held-out error**; AUC giảm 0.06%. So với final repair, graph error giảm **19.38%**, full error giảm **8.03%**, còn AUC giảm 0.24%.

## 3. Các baseline đã có

### Baseline cùng pipeline nhưng từ các đợt chạy cũ

Các số dưới đây dùng cùng split và evaluator, nhưng là single-seed/campaign cũ nên chỉ dùng để kiểm tra độ lớn, không trộn với bảng ba seed ở trên.

| Dataset | Method | Shape error ↓ | Trend error ↓ | Alpha ↑ | Beta ↑ | XGB AUC ↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Default | CTGAN (SDV) | 15.0455 | 17.6522 | 65.4671 | 19.7388 | 0.6663 |
| Default | TVAE (SDV) | 10.8578 | 6.8219 | 81.5438 | 23.6532 | 0.7315 |
| Default | TabDiff | 1.6429 | 1.7127 | 98.3884 | 52.5806 | 0.7482 |
| Default | DynamicGraph, phiên bản cũ | 1.6366 | 1.7083 | 98.4071 | 52.6066 | **0.7530** |
| Shoppers | CTGAN (SDV) | 14.2642 | 15.4354 | 83.8363 | 31.6911 | 0.8013 |
| Shoppers | TVAE (SDV) | 16.6945 | 9.4195 | 53.8737 | 18.1136 | 0.8773 |
| Shoppers | TabDiff | 1.7462 | 1.5848 | 98.3602 | 48.6995 | 0.8988 |

### Constraint guidance và correction

- **Harpoon, code chính thức:** đã chạy năm trial trên Adult, Default và Shoppers. Đây là baseline quan trọng về differentiable guidance, nhưng bộ constraint và thước đo `ViolationAcc` của họ khác graph hiện tại; chỉ so sánh trực tiếp khi chạy cùng constraint catalog. Harpoon-style proxy cùng pipeline cho Default cho held-out error 0.0179, trong khi dynamic graph phiên bản tương ứng đạt 0.0120; đây là bằng chứng hỗ trợ nhưng phải ghi rõ là proxy.
- **LCATS, code chính thức:** đã audit trên Insurance. Post-repair giảm any-violation từ 0.0822 xuống 0.0017, đồng thời sửa 100% số hàng và 28.84% số ô. Kết quả này phù hợp để minh họa trade-off của repair, không phải xếp hạng trực tiếp trên Default.
- **CuTS programs trên Adult:** final repair và best-of-8 đạt satisfaction 1.0, nhưng các constraint mạnh có thể làm Numeric KS tăng từ 0.0249 lên 0.0942–0.1146. Đây là bằng chứng rằng đạt constraint tuyệt đối không đồng nghĩa fidelity tốt.

## 4. Baseline 2024+ đang chạy mới

| Method | Publication | Vai trò | Dataset | Job |
| --- | --- | --- | --- | --- |
| ForestDiffusion | AISTATS 2024 | Strong general-purpose generator | Default, Shoppers; K=20 screen và K=100 gần mặc định | generation 1082464; evaluation 1082465 |
| TabSyn | ICLR 2024 Oral | Latent-diffusion predecessor trực tiếp | Default, Shoppers; code chính thức | generation 1082472; evaluation 1082473 |
| TabDiff | ICLR 2025 | Backbone và unguided baseline | Default, Shoppers | đã hoàn tất |
| Harpoon | ICLR 2026 | Differentiable constraint guidance | Adult, Default, Shoppers | code chính thức, năm trial đã hoàn tất |
| C-DGM | ICLR 2024 | Constraint layer/training objective | benchmark riêng của paper | để bảng phụ; không trộn dataset |

ForestDiffusion và TabSyn sẽ được chấm lại bằng đúng evaluator của phương pháp chính. Bảng tự động sau khi jobs kết thúc được tạo bởi job 1082474 (`643_recent_baseline_aggregate.sbatch`). Các job GPU dùng normal queue `prioritized`, không còn dùng `aicentre/unprivileged`. Campaign được throttle ở sáu task sinh baseline song song: bốn ForestDiffusion và hai TabSyn. TabSyn giữ A40; ForestDiffusion dùng bất kỳ GPU CUDA nào để giảm thời gian chờ.

## 5. Baseline không nên đưa vào cùng cột xếp hạng

- SCTab-Diff là preprint 2026 và dùng protocol/bảng điểm riêng. Có thể trích số paper làm context, nhưng không dùng số đó để tuyên bố thắng nếu chưa rerun cùng split.
- C-DGM, LCATS và CuTS có dataset/constraint program khác. Chúng nên ở bảng constraint-specific hoặc external validity.
- EPIC và HARMONIC là baseline LLM tabular 2024 có mục tiêu huấn luyện/benchmark khác, chi phí tái lập lớn và không cô lập contribution trajectory guidance. Chỉ bổ sung nếu reviewer yêu cầu LLM-generator comparison.

## 6. Điều kiện để khóa bài

Có thể bắt đầu viết ngay khoảng 60–70% phần paper không phụ thuộc bảng cuối. Để khóa main results cần:

1. TabSyn và ForestDiffusion hoàn tất cùng pipeline;
2. ít nhất một dataset ngoài Default cho gain dương thật sự, hoặc một cơ chế abstention được định nghĩa trước và đánh giá rõ;
3. Harpoon-style và final repair dùng đúng cùng graph/constraint set;
4. báo cáo mean ± std cho main comparison, kèm paired test/bootstrap CI;
5. giữ fidelity và utility là guardrail; semantic error là primary metric của contribution, không thay thế hoàn toàn metric chung.

Nếu baseline mới cho fidelity cao hơn rõ rệt, kết quả hiện tại vẫn có thể viết thành contribution về constraint-aware sampling, nhưng cần chứng minh controller gắn được lên backbone mạnh hơn hoặc ít nhất không làm mất lợi thế fidelity của backbone đó.
