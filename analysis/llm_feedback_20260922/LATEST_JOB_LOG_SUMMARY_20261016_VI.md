# Tổng hợp log mới nhất — 16/10/2026

## 1. Trạng thái

| Job | Nhánh | Trạng thái | Kết luận từ log |
| ---: | --- | --- | --- |
| 1085221_0 | SAGE, GPT-2 | Running | Rerun 3 epoch; checkpoint chuyển sang local scratch để không vượt disk quota. MI discovery đã hoàn tất. |
| 1085221_1 | SAGE, Qwen3-0.6B | Running | Rerun 2 epoch với LoRA; MI discovery đã hoàn tất. |
| 1085222 | SPADA | Running | Đã chuyển sang GPU partition; tiến trình đang dùng CPU/GPU bình thường, chưa có exception. |
| 1085223 | GReaT | Running | Đã sửa pin thành `chugchug==0.1.3`; đã vào training loop thành công. |
| 1085234 | LCATS phase 6 | Pending/rerun | Phase 1–5 và independent audit đã xong; rerun cuối sửa evaluator environment và Conda C++ runtime. |
| 1085151_0,2 | ForestDiffusion, Default | Completed | Hai cấu hình đã được đánh giá bằng local matched evaluator. |
| 1085225_1,3 | ForestDiffusion, Shoppers | Completed | Cả K=20 và official K=100 đã được đánh giá bằng cùng local evaluator. |

Không còn job `DependencyNeverSatisfied`.

## 2. ForestDiffusion trên Default

Shape và Trend dưới đây là error, thấp hơn tốt hơn.

| Method | Shape ↓ | Trend ↓ | Alpha ↑ | Beta ↑ | Detection ↑ | AUC ↑ | Heldout relation error ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TabDiff, trung bình 3 seed trước đó | 1.9833 | 2.2165 | 98.4736 | 52.5109 | 0.9528 | 0.7230 | 0.0309 |
| Dynamic graph, trung bình 3 seed trước đó | 1.9591 | 2.2116 | 98.5218 | 52.4436 | 0.9590 | 0.7228 | 0.0120 |
| ForestDiffusion K=20 | 9.4247 | 7.4322 | 94.1387 | 45.2865 | 0.9270 | 0.7053 | 0.1705 |
| ForestDiffusion K=100, official setting | 8.9813 | 6.1003 | 94.4882 | 47.8939 | 0.9357 | 0.6968 | 0.1166 |

K=100 tốt hơn K=20 trên phần lớn metric, nhưng vẫn thua rõ TabDiff và dynamic graph. So với TabDiff, Shape error cao khoảng 4.53 lần, Trend error cao 2.75 lần, AUC thấp 0.0262 và heldout relation error cao 3.77 lần. ForestDiffusion vì vậy là baseline hợp lệ nhưng không phải đối thủ sát nhất trên Default.

ForestDiffusion Shoppers đã sinh và đánh giá xong:

| Method | Shape ↓ | Trend ↓ | Alpha ↑ | Beta ↑ | Detection ↑ | AUC ↑ | Heldout relation error ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ForestDiffusion K=20 | 14.0348 | 5.6888 | 93.2747 | 44.4189 | 0.7917 | 0.8884 | 0.2320 |
| ForestDiffusion K=100, official setting | 14.5687 | 4.6032 | 91.0807 | 49.7168 | 0.8157 | 0.8992 | 0.2165 |

K=100 cải thiện Trend, Beta, detection, AUC và relation error nhưng giảm Shape và Alpha. Vì thế không có một cấu hình ForestDiffusion thống trị hoàn toàn cấu hình còn lại.

## 3. LCATS

Các phase đã hoàn tất:

1. Phase 1: sinh và kiểm chứng constraint;
2. Phase 2: huấn luyện TabSyn, dừng sớm ở epoch 2,200; thời gian train khoảng 486 giây;
3. Sampling: 1,204 hàng, khoảng 1.87 giây;
4. Phase 3: diagnostic hoàn tất.

Phase 4 và 5 sau đó đã chạy xong qua UnoRouter. Ba output repair giống nhau và independent audit đã hoàn tất. Phase 6 chính thức đang được rerun sau khi sửa tách môi trường agent/evaluator và C++ runtime của Conda.

Constraint package có 14 rule, nhưng toàn bộ là unary domain/bound/integer constraints: giới hạn age, BMI, children, charges và miền giá trị sex/smoker/region. Không có dependency liên cột hoặc context-dependent relation. Vì vậy lần chạy LCATS này chưa giải quyết yêu cầu knowledge graph phức tạp.

Diagnostic trên 1,204 synthetic rows:

- 1,116 hàng có ít nhất một violation;
- 88 hàng sạch;
- constraint `age is integer`: 1,114 violations;
- constraint `children is integer`: 84 violations.

Kết quả independent audit sau repair:

| Candidate | Row violation ↓ | Numeric KS ↓ | Categorical TV ↓ | Correlation MAE ↓ | Propensity AUC →0.5 | TSTR NRMSE ↓ | TSTR R² ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TabSyn trước LCATS | 0.9269 | 0.0370 | 0.0066 | 0.0118 | 0.5104 | 0.5254 | 0.7239 |
| LCATS repair | **0.0000** | **0.0299** | 0.0066 | **0.0115** | 0.5106 | 0.5335 | 0.7153 |

LCATS loại toàn bộ measured violations và cải thiện KS/correlation nhẹ, nhưng sửa tất cả hàng (28.49% số ô), đồng thời TSTR giảm nhẹ. Đây là bằng chứng trực tiếp cho trade-off giữa hard repair và utility, dù constraint discovery của lần chạy này vẫn chủ yếu là unary rules.

## 4. SAGE

Hai cấu hình đã qua bước setup và mutual-information discovery:

- 2,478 feature-value MI entries;
- MI trung bình 0.0076, median 0.0005, max 0.6558;
- GPT-2 chạy khoảng 5.2 mini-batch/giây;
- Qwen3-0.6B chạy khoảng 1.5 mini-batch/giây ở log gần nhất.

Chưa có synthetic CSV nên chưa được tính fidelity/utility. Các cảnh báo mamba lock lúc đầu chỉ do hai array task cùng khởi tạo môi trường; cơ chế file lock đã giúp cả hai đi tiếp.

## 5. GReaT và SPADA

GReaT đã được sửa thành `chugchug==0.1.3`; job `1085223` đã vào training loop và không còn lỗi package. SPADA được chuyển khỏi node bị launch failure sang GPU partition; job `1085222` đang train và chưa có exception.

## 6. Kết luận tạm thời

1. ForestDiffusion không đe dọa kết quả chính trên Default; TabDiff và dynamic graph tốt hơn rõ trên fidelity, utility và heldout relations.
2. LCATS repair đạt zero measured violation nhưng phải sửa toàn bộ hàng và làm giảm nhẹ downstream utility; graph sinh ra vẫn quá đơn giản so với dynamic graph của dự án.
3. SAGE là nhánh published LLM baseline gần nhất đang thực sự train; chưa thể kết luận fidelity/utility trước khi sinh mẫu.
4. Lỗi package của GReaT và lỗi Slurm launch của SPADA đều đã được sửa; hai rerun đang chạy.
