# Baseline và metric trên Shoppers

Tài liệu này tách rõ hai loại số liệu: kết quả do bài báo công bố và kết quả chạy cục bộ. Không so sánh trực tiếp hai loại nếu split, số mẫu hoặc phiên bản metric chưa khớp.

## 1. Baseline công bố trên đúng dataset Shoppers

Nguồn chính là [TabDiff, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/5c882988ce5fac487974ee4f415b96a9-Paper-Conference.pdf). Bài này so sánh tám baseline phổ biến trên cùng Shoppers: CTGAN, TVAE, GOGGLE, GReaT, STaSy, CoDi, TabDDPM và TabSyn. Code chính thức: [TabDiff](https://github.com/minkaixu/tabdiff), [TabSyn](https://github.com/amazon-science/tabsyn).

| Method | Shape error ↓ | Trend error ↓ | α-Precision ↑ | β-Recall ↑ | Detection ↑ | MLE AUC ↑ | DCR, 50 lý tưởng |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CTGAN | 21.15 | 13.08 | 76.97 | 31.80 | 0.7488 | 0.875 | – |
| TVAE | 24.51 | 18.67 | 58.19 | 19.78 | 0.2962 | 0.871 | – |
| GOGGLE | 22.33 | 23.90 | 86.95 | 9.79 | 0.1418 | 0.658 | – |
| GReaT | 14.51 | 45.16 | 78.88 | 44.90 | 0.4285 | 0.902 | – |
| STaSy | 9.37 | 8.49 | 89.65 | 37.24 | 0.5482 | 0.914 | 51.53 |
| CoDi | 31.84 | 17.78 | 94.95 | 20.82 | 0.2784 | 0.865 | 51.06 |
| TabDDPM | 2.72 | 6.61 | 88.55 | 47.79 | 0.8349 | 0.918 | 63.23 |
| TabSyn | **1.01** | 2.13 | 98.36 | 49.10 | 0.9662 | 0.914 | 52.90 |
| TabDiff | 1.28 | **1.74** | **99.11** | **49.75** | **0.9843** | **0.921** | **50.24** |

TabSyn trong bảng là reproduction do nhóm TabDiff báo cáo; không được ghi là kết quả chạy cục bộ của dự án này. CSV máy đọc được nằm ở `paper_shoppers_baselines.csv`.

## 2. Ý nghĩa ngắn gọn của metric

- **Shape error**: sai khác marginal của từng cột; dùng KS cho numeric và TV distance cho categorical, sau đó tổng hợp. Thấp hơn tốt hơn.
- **Trend error**: sai khác quan hệ từng cặp cột, tương ứng với `Column Pair Trends` trong SDMetrics. Thấp hơn tốt hơn.
- **α-Precision**: phần synthetic nằm trong support của real distribution; đo fidelity. Cao hơn tốt hơn.
- **β-Recall**: phần support của real distribution được synthetic bao phủ; đo coverage và mode coverage. Cao hơn tốt hơn.
- **Detection**: khả năng làm một classifier phân biệt real/synthetic bị thất bại; điểm cao hơn nghĩa là hai bảng khó phân biệt hơn.
- **MLE AUC**: train XGBoost trên synthetic, test trên real held-out. Cao hơn tốt hơn.
- **DCR**: xác suất nearest neighbor của synthetic thuộc train thay vì held-out; gần 50% là cân bằng, quá cao có nguy cơ gần train quá mức.

## 3. Metric bổ sung cho dynamic constraint graph

Các metric chung ở trên là bắt buộc nhưng không đo trực tiếp mục tiêu của method. Suite cục bộ mới bổ sung, và áp dụng giống nhau cho CTGAN, TVAE, TabDiff và các biến thể của method:

| Metric | Nội dung | Hướng tốt |
| --- | --- | --- |
| `actuate_macro_violation` | Violation trung bình trên các cạnh có quyền tạo action | Thấp |
| `score_only_macro_violation` | Violation trên các cạnh chỉ dùng chấm branch | Thấp, nhưng phải gần real |
| `heldout_violation_error` | Sai số giữa violation rate của synthetic và real held-out | Thấp |
| `calibration_violation_error` | Sai số so với calibration split đã dùng khi đóng băng graph | Thấp |
| `catalog_edge_coverage` | Tỷ lệ cạnh có đủ support trong synthetic để đánh giá | Cao |
| `active_row_edge_fraction` | Tỷ lệ cặp row–edge thực sự applicable | Chỉ chẩn đoán; không tối ưu đơn độc |

Điểm quan trọng là **violation bằng 0 không luôn tốt**. Với near-hard hoặc probabilistic relation, ép về 0 có thể xóa valid modes. Vì vậy metric chính cho paper nên là `heldout_violation_error`, đi cùng Shape, Trend, β-Recall và MLE.

## 4. Trạng thái chạy cục bộ

- TabDiff checkpoint có sẵn là backbone và unconditional baseline.
- CTGAN và TVAE đang được chạy bằng synthesizer có sẵn trong SDV, không phải implementation tự viết.
- TabDDPM và TabSyn hiện dùng số công bố làm mốc; chưa được tuyên bố reproduced cục bộ.
- Job metric mới sẽ đánh giá cùng một số mẫu và cùng code metric cho: TabDiff, CTGAN, TVAE, mid-lookahead, reverse-`x0` guidance và post-repair.

Kết quả cục bộ sẽ được ghi vào `outputs/shoppers_baseline_metric_suite_<job_id>/baseline_metrics.csv`; đây mới là bảng dùng cho so sánh trực tiếp method–baseline trong ablation hiện tại.

## 5. Kết quả chạy cục bộ đầu tiên

Tất cả phương pháp dưới đây dùng cùng 4,096 samples, cùng train/test split và cùng implementation metric. CTGAN/TVAE dùng synthesizer của SDV; đây không phải baseline tự viết. `α-Precision` và `β-Recall` chưa có trong bảng này vì bản `synthcity` cục bộ không tương thích với Torch environment; không thay thế chúng bằng metric khác cùng tên.

| Method | Shape error ↓ | Trend error ↓ | Detection ↑ | MLE AUC ↑ | DCR (%) | Actuate violation ↓ | Heldout violation error ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CTGAN-SDV | 14.264 | 15.435 | 0.750 | 0.801 | 89.72 | 0.3823 | 0.3063 |
| TVAE-SDV | 16.694 | 9.419 | 0.180 | 0.877 | 88.43 | 0.2758 | 0.1852 |
| TabDiff | 1.746 | 1.585 | 0.988 | 0.899 | 91.14 | 0.0070 | 0.0218 |
| DynamicGraph-MidLookahead | 1.739 | 1.509 | 0.981 | **0.900** | 91.16 | 0.0059 | 0.0209 |
| DynamicGraph-ReverseX0 | **1.733** | 1.512 | 0.983 | 0.898 | 91.14 | 0.0052 | 0.0207 |
| PostRepair | 1.735 | **1.501** | **0.998** | 0.900 | 91.19 | **0.0036** | **0.0205** |

Diễn giải ngắn:

- TabDiff đã vượt xa CTGAN/TVAE ở fidelity và semantic constraint preservation; đây là lý do hợp lý để dùng nó làm backbone.
- Mid-lookahead và reverse-x0 cải thiện violation so với unconditional TabDiff mà Shape/Trend gần như không đổi.
- Post-repair vẫn tốt nhất ở constraint metric hiện tại. Dynamic injection chưa thể được mô tả là vượt baseline mạnh nhất.
- DCR đều xa mức 50%; cách cài DCR cục bộ và số liệu công bố chưa cùng protocol, nên chỉ dùng so sánh nội bộ giữa các hàng, không so trực tiếp với bảng paper.

Artifact máy đọc được: `outputs/shoppers_baseline_metric_suite_1057825/baseline_metrics.csv`. Job `1057825` chỉ lỗi ở bước render Markdown sau khi CSV đã được ghi; retry sạch là `1057841`.
