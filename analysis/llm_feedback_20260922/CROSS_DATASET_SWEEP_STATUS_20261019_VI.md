# Mở rộng method sang nhiều dataset và bảng baseline ban đầu

## Mục tiêu của đợt chạy

Đợt này kiểm tra khả năng chuyển giao của controller tốt nhất từ News sang **Default** và **Shoppers**. Mỗi cấu hình dùng cùng checkpoint TabDiff, cùng noise seed và cùng evaluator để sinh bốn nhánh: TabDiff, final repair, DynamicGraph và DynamicGraph + certifier. Vì vậy chênh lệch giữa các nhánh là paired comparison, không bị lẫn với khác biệt do training seed.

Một thiếu sót cũ đã được sửa trước khi submit: sampler chỉ actuation đầy đủ cho các program cột–cột. Phần lớn graph Default/Shoppers lại là constraint numeric–literal có điều kiện. Projection và trajectory signal hiện hỗ trợ cả hai dạng, đồng thời vẫn áp dụng applicability, row-level stability mask và trust region.

## Sweep method đã submit

Slurm array `1098641`, tám tasks:

| Dataset | Cấu hình | Strength | Interval theo feature noise quantile | Trust radius |
| --- | --- | ---: | --- | ---: |
| Default | s050_mid | 0.50 | 0.10–0.40 | 0.35 |
| Default | s100_mid | 1.00 | 0.10–0.40 | 0.50 |
| Default | s150_mid | 1.50 | 0.10–0.40 | 0.75 |
| Default | s100_broad | 1.00 | 0.05–0.55 | 0.50 |
| Shoppers | s050_mid | 0.50 | 0.10–0.40 | 0.35 |
| Shoppers | s100_mid | 1.00 | 0.10–0.40 | 0.50 |
| Shoppers | s150_mid | 1.50 | 0.10–0.40 | 0.75 |
| Shoppers | s100_broad | 1.00 | 0.05–0.55 | 0.50 |

Các thành phần cố định gồm edge-wise automatic valley/isotonic routing, rank stability, persistence hai bước, middle-peak schedule, stochastic sampler và Heun predictor–corrector. Sweep chỉ thay ba hyperparameter có ý nghĩa trực tiếp, tránh search quá rộng trên test metrics.

## Baseline đã submit thêm

| Dataset | Baseline | Implementation | Job |
| --- | --- | --- | --- |
| Adult | ForestDiffusion | official repository, K=100 | `1098642` |
| Adult | TabSyn | official repository, full train/sample | `1098643` |
| Adult | CTGAN | SDV implementation, 300 epochs | `1098644_0` |
| Adult | TVAE | SDV implementation, 300 epochs | `1098644_1` |
| Adult | TabDiff | existing checkpoint/sample, matched re-evaluation | `1098647` |

## Bảng baseline matched hiện đã có

Các số dưới đây đều được tính lại tại local bằng cùng repository split, 4.096 rows và cùng metric code. Đây không phải số chép từ paper.

| Dataset | Method | Shape error ↓ | Trend error ↓ | Alpha ↑ | Beta ↑ | Detection ↑ | XGB AUC ↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Default | TabDiff | 2.3851 | 2.5518 | 96.2240 | 53.3024 | 0.9561 | 0.7175 |
| Default | TabSyn official | **1.4810** | **1.6499** | **99.2198** | 47.7816 | **0.9615** | 0.7127 |
| Default | ForestDiffusion official | 8.9813 | 6.1003 | 94.4882 | 47.8939 | 0.9357 | 0.6968 |
| Default | CTGAN | 15.2233 | 17.8281 | 64.4922 | 21.3704 | 0.6355 | 0.6695 |
| Default | TVAE | 10.8799 | 6.5209 | 82.3844 | 25.9066 | 0.5474 | **0.7244** |
| Shoppers | TabDiff | **1.6044** | 2.5302 | 96.3738 | 50.3320 | **0.9676** | 0.8978 |
| Shoppers | TabSyn official | 1.6652 | **1.9929** | **99.0039** | 46.3363 | 0.9539 | 0.8817 |
| Shoppers | ForestDiffusion official | 14.5687 | 4.6032 | 91.0807 | **49.7168** | 0.7925 | **0.8992** |
| Shoppers | CTGAN | 14.2642 | 15.4354 | 83.8363 | 31.6911 | 0.7090 | 0.8013 |
| Shoppers | TVAE | 16.6945 | 9.4195 | 53.8737 | 18.1136 | 0.1630 | 0.8773 |

DynamicGraph sẽ được thêm vào bảng sau khi array `1098641` hoàn tất. Cấu hình chính chỉ được chọn nếu cải thiện nhiều metric fidelity/utility và không chỉ giảm violation của graph tự định nghĩa. Final repair luôn được giữ ở cùng bảng paired để kiểm tra liệu trajectory intervention có thật sự cần thiết.

## Vị trí kết quả

- Method: `outputs/literal_dynamic_<dataset>_<config>_1098641_<task>/eval/RESULTS.md`.
- ForestDiffusion Adult: `outputs/forestdiffusion_adult_official_k100_1098642/eval/RESULTS.md`.
- TabSyn Adult: `outputs/tabsyn_official_adult_1098643/eval/RESULTS.md`.
- CTGAN/TVAE Adult: `outputs/sdv_adult_<method>_1098644_<task>/eval/RESULTS.md`.
- Log: `logs/llmtd-lit-xdata_*`, `logs/llmtd-adult-*`.

## Tiêu chí quyết định

Một cấu hình chỉ đáng giữ nếu đạt ít nhất một trong hai điều kiện: (1) cải thiện fidelity/utility phổ biến so với paired TabDiff mà không giảm mạnh metric khác; hoặc (2) giảm independent held-out graph error hơn final repair trong khi fidelity/utility nằm trong sai số nhỏ. Nếu chỉ tăng satisfaction trên constraint do chính pipeline khám phá nhưng làm xấu Shape, Trend hoặc XGB AUC, cấu hình đó không được dùng làm main result.

## Mở rộng lần hai: Adult, Magic và Diabetes

Slurm array `1098676` bổ sung 12 paired method tasks, bốn cấu hình cho mỗi dataset:

| Dataset | Graph có thể actuation trong sampler này | Vai trò trong screening |
| --- | ---: | --- |
| Adult | 2 numeric–literal edges có categorical applicability | Semantic transfer nhỏ nhưng rõ: education label → education number |
| Magic | 5 numeric–literal edges, cùng các component liên thông | Ứng viên transfer chính; có thêm positive copula-factor variant |
| Diabetes | 1 numeric–literal edge trong graph chủ yếu categorical | Kiểm tra giới hạn và tránh chỉ chọn dataset thuận lợi |

Mỗi dataset có `s050_mid`, `s100_mid`, `s100_broad`; cấu hình thứ tư là `s150_mid` cho Adult/Diabetes và `positive_factor` cho Magic. Positive factor truyền thông tin trên cả edge đang thỏa mãn thay vì chỉ sửa violation, đồng thời dùng copula transport để giữ marginal.

Baseline mới được submit:

| Job | Dataset | Baseline |
| --- | --- | --- |
| `1098677[0]` | Magic | ForestDiffusion official, K=100 |
| `1098677[1]` | Diabetes | ForestDiffusion official, K=100 |
| `1098678` | Magic | TabSyn official |
| `1098679[0-1]` | Magic | CTGAN, TVAE |
| `1098679[2-3]` | Diabetes | CTGAN, TVAE |
| `1098680[0]` | Adult | TabDDPM official |
| `1098680[1]` | Magic | TabDDPM official |

Adult TabDiff matched evaluation ban đầu lỗi vì raw Adult giữ khoảng trắng ở categorical label trong khi decoder bỏ khoảng trắng, làm binary target bị hiểu thành bốn lớp. Evaluator đã canonicalize representation này cho real, held-out và synthetic; run sửa là `1098668`. Đây là sửa protocol, không thay đổi model hoặc metric definition.

Output mới:

- Method: `outputs/extended_dynamic_<dataset>_<config>_1098676_<task>/eval/RESULTS.md`.
- ForestDiffusion: `outputs/forestdiffusion_<dataset>_official_k100_1098677_<task>/eval/RESULTS.md`.
- TabSyn Magic: `outputs/tabsyn_official_magic_1098678/eval/RESULTS.md`.
- CTGAN/TVAE: `outputs/sdv_<dataset>_<method>_1098679_<task>/eval/RESULTS.md`.
- TabDDPM: `outputs/tabddpm_official_<dataset>_1098680_<task>/eval/RESULTS.md`.

## LCATS: ưu tiên artifact đã hoàn tất các pha dùng API

LCATS hiện chỉ có pipeline phát hành sẵn và dữ liệu chuẩn bị đầy đủ cho **Insurance**. Các artifact của phase 1--5 đã tồn tại, gồm constraint catalog, exclusive buckets, mẫu TabSyn gốc và ba candidate đã align. Vì vậy đợt này không gọi lại discovery API và không huấn luyện lại generator.

| Job | Phạm vi | API mới | Output |
| --- | --- | --- | --- |
| `1098706` | Phase 6 chính thức (candidate selection/reflexion) và independent audit | Có thể có, chỉ ở phase 6 | `outputs/lcats_published_final_audit_1098706/RESULTS.md` |
| `1098709` | Đánh giá read-only toàn bộ artifact LCATS hiện có | Không | `outputs/lcats_existing_artifacts_audit_1098709/RESULTS.md` |

LCATS ở đây được ghi là **official code trên Insurance**, nhưng là baseline post-generation repair; không phải cùng backbone hoặc cùng dataset với các paired DynamicGraph runs. Việc port LCATS sang Adult/Magic cần tạo lại metadata, discovery và verifier bằng API, nên được để sau đúng theo ưu tiên hiện tại.
