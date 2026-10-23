# Đợt chạy baseline constraint public — 2026-10-12

## Mục tiêu

Đợt chạy này tách hai loại so sánh vốn không thể thay thế cho nhau:

1. **Tái lập bằng code chính thức**: giữ backbone, checkpoint, hyperparameter và protocol của tác giả.
2. **Matched adaptation**: đưa cơ chế cốt lõi của baseline vào cùng checkpoint, split, số mẫu và evaluator TabDiff của dự án.

Kết quả từ loại (1) cho biết code public có hoạt động theo công bố hay không. Kết quả từ loại (2) mới phù hợp để gán chênh lệch cho cơ chế guidance thay vì cho backbone hoặc evaluator.

## Baseline đã submit

| Baseline | Công bố | Trạng thái code | Cách chạy trong đợt này | Job |
| --- | --- | --- | --- | --- |
| Harpoon | ICLR 2026 | Official public code; checkpoint được huấn luyện lại cục bộ bằng code tác giả | 200 reverse steps, strength 0.2, 5 trials, Adult/Default/Shoppers, 4 loại constraint | `1069377`, aggregate `1069378` (xong) |
| Harpoon | ICLR 2026 | Matched adaptation | Cùng TabDiff Default checkpoint, 200 steps, cùng noise/seed, sweep gồm setting strength 0.2 | `1069379`, evaluate `1069380` |
| ConstrainedDGM | ICLR 2024 | Official public code + device compatibility patch | TVAE và constrained TVAE, News, 50 epochs, correlation ordering | retry `1069954`, evaluate `1069956` (đang chạy) |
| ConstrainedDGM | ICLR 2024 | Official code + device compatibility patch, matched split | Cùng train/test News của TabDiff; chỉ giữ hyperparameter và constraint layer của paper | retry `1069955`, evaluate `1069957` (đang chạy) |
| TabDiff + ConstrainedDGM layer | ICLR 2025 backbone + ICLR 2024 constraint layer | Cross-method matched transfer | Áp public constraint layer lên cùng sample TabDiff News, rồi chấm bằng evaluator chung | train `1069382`, sample `1069383`, evaluate `1069953` (xong) |
| CDD-style projection | NeurIPS 2025 | Matched adaptation; paper không đánh giá tabular | KL-nearest categorical projection ở từng reverse step trên cùng TabDiff Default checkpoint | `1069863`, evaluate `1069864` (xong); timing `1069865`/`1069866` |

LCATS (KDD 2026) đã có proxy weighted-repair trên cùng sample trong các run trước. Không gọi phase discovery chính thức ở đợt này vì implementation public gửi 20 real rows trong prompt; điều đó vi phạm quy tắc không đưa raw real rows ra API. Với atomic constraint của Harpoon, `LCATS-style atomic final repair` được chấm như đối chứng post-generation trong job `1069380`; đây là matched specialization, không được gọi là official reproduction.

CDD (NeurIPS 2025) là đối chứng method rất sát nhưng paper đánh giá text/molecule, không có implementation tabular chạy nguyên trạng. Job `1069863` vì vậy được ghi rõ là **CDD-style matched adaptation**, không phải reproduction. CTDF là NeurIPS 2025 GenAI4Fin workshop, không phải main-conference baseline; các kết quả `ctdf_style_*` cũ chỉ là same-backbone proxy.

## Kết quả đã có: matched Harpoon trên TabDiff Default

Các số dưới đây dùng cùng checkpoint, seed, 4.096 mẫu, 200 reverse steps và evaluator. Đây là phép so sánh cơ chế đáng tin hơn bảng chạy độc lập bằng backbone Harpoon.

| Constraint | Method | Violation ↓ | Shape error ↓ | Trend error ↓ | α-precision ↑ | β-recall ↑ | AUC ↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `AGE ≥ 50` | TabDiff | 91.5527 | 11.6014 | 13.6737 | 80.2974 | 27.8067 | 0.6954 |
| `AGE ≥ 50` | Harpoon-style, strength 0.2 | **0.0000** | **9.1445** | **8.7533** | 92.5890 | **46.9888** | 0.7025 |
| `AGE ≥ 50` | Final repair | **0.0000** | 11.0924 | 13.6737 | **94.0401** | 41.9827 | 0.6949 |
| `EDUCATION = 3` | TabDiff | 84.9121 | 10.0480 | 4.3902 | 75.5645 | 23.4274 | 0.6864 |
| `EDUCATION = 3` | Harpoon-style mạnh nhất | 32.5195 | **6.4278** | **3.5495** | 94.2149 | 41.8011 | 0.6622 |
| `EDUCATION = 3` | Final repair | **0.0000** | 6.5100 | 4.3902 | **96.3747** | **46.8280** | **0.7142** |
| Cả hai | TabDiff | 96.6797 | 15.8913 | 14.4610 | 51.6374 | 11.8129 | 0.6902 |
| Cả hai | Harpoon-style mạnh nhất | 20.5078 | **10.4982** | **14.8882** | 85.9871 | 46.8421 | **0.7288** |
| Cả hai | Final repair | **0.0000** | 11.7960 | 16.6397 | **89.1813** | **47.8947** | 0.6601 |

Diễn giải:

- Với numeric constraint, trajectory guidance đạt feasibility tuyệt đối và giữ conditional structure tốt hơn sửa cuối; đây là bằng chứng tích cực cho can thiệp trong trajectory.
- Với categorical constraint, target-logit energy chưa đủ mạnh: sửa cuối thắng rõ về feasibility, coverage và AUC. Đây là lý do cần CDD-style discrete projection làm baseline trực tiếp.
- Với mixed constraint, guidance vẫn còn 20.5% violation nhưng fidelity/AUC tốt hơn final repair. Thiết kế có cơ sở nhất hiện tại là trajectory guidance mềm cộng final certifier, thay vì khẳng định guidance đơn lẻ thay thế hard feasibility layer.

CDD-style matched adaptation đã hoàn tất cho categorical constraint:

| Method | Violation ↓ | Shape error ↓ | Trend error ↓ | α ↑ | β ↑ | AUC ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Harpoon-style mạnh nhất | 32.5195 | 6.4278 | 3.5495 | 94.2149 | 41.8011 | 0.6622 |
| Final repair | **0.0000** | 6.5100 | 4.3902 | **96.3747** | 46.8280 | **0.7142** |
| CDD-style, full trajectory | **0.0000** | **4.7070** | **3.5252** | 95.6308 | **46.9086** | 0.7054 |
| CDD-style, middle band | 26.0498 | 6.6545 | 3.7487 | 95.6452 | 44.4086 | 0.7207 |

CDD-style full trajectory là baseline mạnh về feasibility + Shape/Trend. Middle-band mở lại 813 states đã chốt sai nhưng vẫn để lại 26.05% violation vì constraint bị tắt trước giai đoạn cuối.

Timing follow-up `1069865`/`1069866` đã hoàn tất:

| Window | Violation ↓ | Shape error ↓ | Trend error ↓ | α ↑ | β ↑ | AUC ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full trajectory | **0.0000** | **4.7070** | **3.5252** | **95.6308** | 46.9086 | 0.7054 |
| Middle band rồi tắt | 26.0498 | 6.6545 | 3.7487 | 95.6452 | 44.4086 | 0.7207 |
| Delayed middle-to-late | **0.0000** | 5.5590 | 3.7373 | 95.5349 | 48.4140 | **0.7325** |
| Late-only | **0.0000** | 6.5112 | 3.8508 | 94.8336 | **49.3683** | 0.6683 |
| Final repair | **0.0000** | 6.5100 | 4.3902 | 96.3747 | 46.8280 | 0.7142 |

Kết quả tách được ba hiệu ứng. Full guidance giữ marginal/pairwise fidelity tốt nhất; late-only bảo đảm feasibility nhưng làm conditional utility kém; delayed middle-to-late giữ 0% violation đồng thời có AUC cao nhất, hơn full guidance 0.0271 và hơn final repair 0.0183. Vì vậy claim timing nên là **activate when the state becomes readable, then maintain feasibility**, không phải “chỉ tác động trong middle band”. Đây là baseline result, chưa phải kết quả của dynamic graph; method của dự án cần chứng minh bộ active edges theo row/state đạt hoặc vượt frontier này trên constraint phức tạp hơn một equality đơn cột.

## Kết quả chính của dynamic graph hiện tại

Trên Default, trung bình ba paired seeds:

| Method | Shape ↓ | Trend ↓ | α ↑ | β ↑ | AUC ↑ | Held-out constraint error ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TabDiff | 1.9833 | 2.2165 | 98.4736 | **52.5109** | 0.7230 | 0.0309 |
| Dynamic graph | 1.9591 | 2.2116 | **98.5218** | 52.4436 | 0.7228 | **0.0120** |
| Frozen graph | 1.9600 | 2.2119 | 98.4990 | 52.4533 | 0.7238 | 0.0154 |
| Final repair | **1.9556** | **2.1909** | 98.5003 | 52.4761 | **0.7278** | 0.0161 |
| Harpoon-style hard | 1.9772 | 2.2317 | 98.5284 | 52.4398 | 0.7278 | 0.0179 |

Dynamic graph giảm held-out constraint error **61.2%** so với TabDiff, **22.1%** so với frozen graph và **25.5%** so với final repair. Khi match chính xác số online actions của frozen graph, dynamic vẫn tốt hơn held-out error trung bình **24.4% trên cả 3/3 seeds**. Tuy nhiên Shape/Trend/α/β/AUC chỉ chênh rất nhỏ; do đó bằng chứng hiện tại ủng hộ **calibration và chọn đúng row/edge**, chưa ủng hộ claim SOTA về chất lượng sinh dữ liệu nói chung.

Cross-dataset còn yếu hơn:

- Shoppers: dynamic giảm actionable violation khoảng 55.8% so với unconditional nhưng gần như hòa final repair; AUC giảm khoảng 0.0042.
- Diabetes: dynamic/final đều đưa actionable violation về 0, nhưng trajectory không hơn final repair và AUC giảm khoảng 0.0057.
- Các proxy đưa external knowledge vào repair hoặc train-time prior chỉ làm metric chung thay đổi rất nhỏ và đôi khi giảm fidelity. Chưa có bằng chứng rằng graph-discovery tốt hơn tự động chuyển thành generation tốt hơn.

Vì vậy kết luận đúng ở thời điểm này là: **đủ tín hiệu để tiếp tục research project, chưa đủ bằng chứng để tuyên bố vượt SOTA**. Điểm mạnh nhất là state-dependent topology cải thiện held-out constraint calibration dưới matched action budget; điểm yếu nhất là lợi ích chưa lặp lại rõ trên nhiều dataset và chưa thắng final repair trên metric phổ biến.

## Baseline khác đáng đưa vào

| Mức ưu tiên | Work | Vai trò hợp lý | Cách dùng |
| --- | --- | --- | --- |
| 1 | Constrained Discrete Diffusion, NeurIPS 2025 | Per-step KL projection cho discrete state | Matched adaptation trên categorical TabDiff; job `1069863` |
| 1 | ConstrainedDGM, ICLR 2024 | Hard constraint layer có guarantee; accepted tabular baseline | Official News và matched split đang chạy |
| 1 | LCATS, KDD 2026 | LLM discovery + diagnosis + post-repair | Dùng public rules/output khi có; giữ matched repair proxy do privacy restriction |
| 2 | TabLogicEval, ICLR 2025 Tiny Papers | Independent evaluator HCS/MDI/DSI, không phải generator | Thêm metric độc lập để tránh chỉ chấm rule do chính dự án tạo |
| 2 | TabSyn, ICLR 2024 Oral | Strong unconditional generator | Dùng làm backbone/fidelity anchor, không dùng như constraint-guidance baseline |
| 2 | Diffuse Everything, ICML 2025 | Strong mixed-type diffusion backbone | Kiểm tra transfer của graph mechanism ngoài TabDiff sau khi claim chính ổn định |
| 3 | SAGE, ACL 2026 | Value-dependent knowledge cho LLM table generation | So sánh graph discovery/dependency, không đặt cùng cột sampler guidance |
| 3 | SCTab-Diff, preprint 2026 | Static semantic prior trong denoising | Baseline khái niệm quan trọng, nhưng phải ghi preprint và cần code/checkpoint trước khi gọi reproduction |

Không ưu tiên thêm CTGAN/TVAE vào bảng constraint chính: chúng là fidelity anchors và đã xuất hiện trong bảng TabDiff/SCTab-Diff, nhưng không kiểm tra contribution “dynamic graph trong reverse diffusion”.

## Metric chung

Các bảng local sẽ báo cáo:

- `Shape` và `Trend`: fidelity của marginal và pairwise distribution, cùng họ metric của TabDiff/SCTab-Diff;
- `α-precision` và `β-recall`: chất lượng và coverage trong không gian feature;
- detection score;
- downstream utility: XGBoost AUC cho classification, R² và normalized RMSE cho News regression;
- violation trên constraint public;
- số bước, runtime và tỷ lệ repair khi có.

Không ghép trực tiếp số fidelity từ paper vào cùng cột với local rerun nếu split, backbone hoặc evaluator khác nhau. Bảng chính nên dùng số **recomputed locally**; paper-reported number chỉ nằm ở cột tham khảo có nhãn rõ.

## Sanity check đã chạy

Evaluator News đã chạy smoke trên held-out real rows và cùng rows sau public constraint layer:

| Method | Shape ↑ | Trend ↑ | R² ↑ | NRMSE ↓ | Any violation ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Real held-out | 98.766 | 96.753 | 0.896 | 0.322 | 2.598% |
| Real held-out + ICLR-2024 layer | 98.279 | 96.753 | 0.893 | 0.328 | 0.000% |

Đây chỉ là pipeline sanity check, không phải kết quả synthetic. Nó cho thấy hai điểm cần report:

1. constraint layer đạt feasibility tuyệt đối;
2. chính held-out real data vẫn có 2.60% rows không thỏa tập constraint public, và hard correction làm Shape/R² giảm nhẹ.

Điều này tạo một phép kiểm tra có ý nghĩa cho dynamic graph: method không chỉ cần giảm violation mà phải tránh cưỡng ép mọi trạng thái khi constraint không phù hợp với row hiện tại.

## News bounded pilot đã chạy

TabDiff News được train 8.000 epochs/72.000 updates trên một A40 trong 1 giờ 57 phút, sau đó sinh 4.096 rows bằng 50 reverse steps. Đây là bounded pilot, không phải reproduction của số paper.

| Method | Shape ↑ | Trend ↑ | α ↑ | β ↑ | Detection | R² ↑ | NRMSE ↓ | Any violation ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TabDiff | **96.4965** | 90.9501 | **96.9007** | **44.5003** | **0.8928** | **-0.1747** | **1.0838** | 24.6094% |
| TabDiff + ICLR-2024 Constraint Layer | 96.1275 | **93.1562** | 96.8991 | 44.4482 | 0.8926 | -0.1809 | 1.0867 | **0.0000%** |

Hard layer sửa 24.61% synthetic rows, đưa violation về 0 và tăng Trend 2.21 điểm, nhưng Shape giảm 0.37 điểm; α/β/detection hầu như giữ nguyên. Utility regression của cả hai âm, nên checkpoint này không phù hợp làm bằng chứng chất lượng tổng thể. Kết quả chỉ hỗ trợ hai kết luận cơ chế: TabDiff pilot không tự học chắc các public inequalities, và hard projection có thể sửa feasibility nhưng không tự động cải thiện utility.

Các run ConstrainedDGM `1069857`/`1069858` đã train nhưng lỗi khi public inverse-transform tạo tensors trên CPU trong lúc decoder ở GPU. Compatibility patch chỉ đồng bộ device/dtype và chuyển output về CPU trước NumPy; không đổi loss, constraint hay architecture. Retry `1069954`/`1069955` đang chạy.

## Audit constraint News

File `data/news/news_constraints.txt` của ConstrainedDGM là nguồn chuẩn mà model dùng. Hàm NumPy audit riêng trong repository viết inequality thứ hai khác với file này. Evaluator của dự án lấy **text constraint file** làm chuẩn và ghi rõ sai khác, tránh vô tình chấm một rule khác rule đã được layer thực thi.

TabDiff compact hai nhóm one-hot `data_channel` và `weekday`; do đó raw column index không còn bảo toàn sau cột 11. Adapter mới tái dựng đúng layout 60 cột của paper theo tên cột trước khi gọi public constraint layer. Không được áp trực tiếp constraint index 27–29 lên compact TabDiff row vì khi đó chúng trỏ nhầm sang các cột LDA.

## API từ thời điểm này

- Credential duy nhất: `/home/cs.aau.dk/tungkvt/binhtt2/.env`.
- Base URL: `https://api.unorouter.com/v1`.
- Loader: `scripts/unorouter_env.py`; loader không in key ra log.
- Không job mới nào trong đợt baseline này cần API.
- Mọi job API tiếp theo phải dùng UnoRouter Chat Completions và chỉ gửi schema, aggregate metadata hoặc synthetic rows; không gửi raw real rows.

## Vị trí kết quả dự kiến

- `outputs/harpoon_official_5trial_aggregate_*/RESULTS.md`
- `outputs/harpoon_matched_eval_*/RESULTS.md`
- `outputs/news_public_constraint_eval_*/RESULTS.md`
- `outputs/news_matched_constraint_eval_*/RESULTS.md`

Các job evaluate có dependency với toàn bộ producer tương ứng, vì vậy sẽ không tạo bảng từ output thiếu.
