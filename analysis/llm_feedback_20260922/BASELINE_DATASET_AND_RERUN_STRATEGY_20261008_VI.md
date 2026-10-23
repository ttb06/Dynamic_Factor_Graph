# Chiến lược dataset và baseline cho dynamic constraint graph

**Ngày cập nhật:** 08-10-2026  
**Mục tiêu:** phân biệt số paper-reported, official-code rerun và matched local comparison.

## 1. Các dataset được dùng bởi những baseline gần nhất

| Work | Dataset trong evaluation liên quan | Phần phù hợp với dự án | Giao với dataset hiện tại |
| --- | --- | --- | --- |
| TabSyn, ICLR 2024 | Adult, Default, Shoppers, Magic, Beijing, News | General synthetic-data fidelity | Adult, Default, Shoppers |
| TabDiff, ICLR 2025 | Adult, Default, Shoppers, Magic, Beijing, News, Diabetes | Backbone và general fidelity | Tất cả dataset hiện có; main hiện là Shoppers, Default, Diabetes |
| Harpoon, ICLR 2026 | Tám dataset cho imputation; general constraints dùng Adult, Default, Shoppers | Inference-time differentiable guidance | Adult, Default, Shoppers |
| SAGE, ACL 2026 | Adult, HELOC, Iris, Diabetes, MIC, Housing; constraint examples trên Adult/Housing | Value-conditioned dependency graph | Adult, Diabetes |
| CuTS, ICML 2024 | Public logical programs được report rõ trên Adult và German | Constraint satisfaction, rejection/fine-tuning | Adult; German chỉ ở external protocol |
| LCATS, KDD 2026 | Reproducible code/demo hiện có trên Insurance | LLM discovery + TabSyn + post-repair | Chưa giao main datasets |
| CTDF, ICAIF 2025 | Finance-oriented constrained tables | Per-step feasibility operator | Chưa có official common-table run |

### Nhận xét về độ giao

- **Shoppers và Default** là hai dataset quan trọng nhất cho baseline comparison: cùng xuất hiện trong TabSyn, TabDiff và Harpoon; đồng thời dynamic graph hiện có tín hiệu actuation rõ.
- **Diabetes** là dataset thứ ba tốt cho generalization vì categorical-heavy, có trong TabDiff và SAGE, đồng thời đã tạo negative result cho mechanism. Nó không có Harpoon general-constraint benchmark chính thức.
- **Adult** có độ giao paper cao nhất nhưng natural graph hiện gần bão hòa. Nên giữ Adult cho public constraint programs và external reproduction, không dùng làm dataset chính để chứng minh dynamic trajectory.
- **Insurance** cần giữ cho LCATS external baseline. Chuyển toàn bộ method sang Insurance hiện chưa đáng vì graph official chủ yếu là unary bounds và có natural violation thấp.
- **German** chỉ cần cho CuTS external validation; không cần train thêm TabDiff nếu không đưa public CuTS programs thành một main task.

## 2. Có nên đổi dataset theo baseline hay chạy baseline trên dataset của mình?

Nên dùng hai tầng, nhưng mọi claim thắng/thua phải đến từ tầng thứ nhất.

### Tầng A — matched local comparison

Dataset: **Shoppers, Default, Diabetes**.

Tất cả method phải dùng:

- cùng train/test split;
- cùng TabDiff checkpoint nếu baseline là sampling intervention;
- cùng unconditional random stream khi khả thi;
- cùng graph/verifier;
- cùng sample count và reverse steps;
- cùng fidelity, utility, privacy và cost evaluator.

Baseline bắt buộc:

1. TabDiff unconditional;
2. final repair;
3. best-of-N hoặc rejection;
4. frozen graph và frozen-state;
5. shuffled row-edge assignment;
6. Harpoon-style differentiable guidance trên constraint subset chung;
7. CTDF-style per-step projection trên constraint subset chung;
8. dynamic graph.

Chỉ tầng này được dùng cho câu như “method tốt hơn baseline”. Harpoon-style và CTDF-style phải ghi là proxy nếu không chạy nguyên code tác giả.

### Tầng B — external official protocol

- Harpoon official: Adult, Default, Shoppers, author 70/30 split;
- CuTS official: Adult và German public programs;
- LCATS official: Insurance;
- paper-reported TabSyn/TabDiff/SAGE trên các dataset chuẩn.

Tầng này trả lời “implementation có tương thích và kết quả có cùng order of magnitude với literature hay không”. Không được gộp các số này vào một ranking với tầng A.

### Quyết định dataset

Không đổi main benchmark khỏi Shoppers–Default–Diabetes. Thay vào đó:

- chạy Harpoon trên Shoppers/Default vì đây là giao trực tiếp;
- giữ Diabetes làm categorical-heavy generalization và negative control;
- dùng Adult/Insurance/German như external benchmark;
- chỉ thêm dataset mới nếu có process/relational constraints không bão hòa, không phải chỉ để tăng số dataset.

## 3. Các paper có tự rerun baseline không?

Có, nhưng cách làm không đồng nhất.

### TabSyn

- Tác giả reproduce các baseline từ official code;
- một số architecture được tăng kích thước để gần khoảng 10M parameters;
- kết quả được tính trên 20 synthetic tables;
- TabDDPM được chỉnh nhẹ từ conditional sang unconditional generation.

Vì vậy số TabSyn không phải lúc nào cũng là con số copy nguyên từ paper gốc của từng baseline.

### TabDiff

- Kết quả TabDiff được tính trung bình trên 20 randomly sampled synthetic tables từ best-validated checkpoint;
- TabSyn được nhóm TabDiff tự reproduce;
- phần lớn số của các baseline khác trên sáu dataset cũ được lấy từ TabSyn vì TabDiff dùng cùng split/protocol;
- Diabetes được đánh giá bổ sung vì không có trong TabSyn benchmark cũ.

Đây là ví dụ hợp lệ của việc reuse số paper khi protocol thực sự giống, nhưng nguồn phải được ghi rõ.

### Harpoon

- Hầu hết baseline được implement từ public code;
- các diffusion methods dùng shared backbone để tách lợi ích của guidance;
- fixed seeds và năm trials được dùng để báo mean/std;
- general constraints tập trung Adult, Default và Shoppers.

### Quy tắc cho dự án này

Mỗi hàng kết quả phải có một trong ba nhãn:

1. `paper-reported` — số đọc từ paper/artifact;
2. `official-code rerun` — chạy code tác giả, nhưng phải ghi số trial/split;
3. `matched local run` — cùng protocol với method của mình.

Không dùng paper-reported mean/std để suy ra significance so với một local seed. Không gọi một proxy là official reproduction.

## 4. Baseline đã có trên server

| Baseline | Dataset | Evidence | Trạng thái |
| --- | --- | --- | --- |
| TabDiff/CTGAN/TVAE | Shoppers | Matched local | Hoàn thành |
| TabDiff dynamic/frozen/final repair | Shoppers, Default, Diabetes | Matched local | Hoàn thành/nâng cấp đang chạy |
| CuTS | Adult, German | Official-code rerun | Hoàn thành |
| LCATS + TabSyn | Insurance | Official-code rerun | Hoàn thành |
| Harpoon-style | Shoppers, Default | Same-backbone proxy | Hoàn thành, nhưng task chưa match graph hiện tại |
| CTDF-style | Shoppers, Default | Same-backbone proxy | Hoàn thành, numeric subset only |
| Harpoon | Adult, Default, Shoppers | Paper/repository-reported | Có artifact; official rerun mới đang chạy |

## 5. Jobs vừa submit

| Job | Nội dung | Evidence sau khi hoàn thành |
| --- | --- | --- |
| `1061898` | Chuẩn bị author split và isolated run roots cho Harpoon | Infrastructure only |
| `1061899[0-2]` | Train shared official Harpoon/RePaint backbone trên Adult, Default, Shoppers | Official-code rerun |
| `1061900[0-2]` | Harpoon và RePaint trên range/category/AND/OR, một compatibility trial | Official-code rerun, chưa phải full five-trial reproduction |
| `1061901` | Tách paper rows và rerun rows thành bảng riêng | Evidence audit |
| `1061903[0-3]` | CTGAN/TVAE trên Default và Diabetes, cùng repository split | Matched local general-fidelity baseline |
| `1061904[0-1]` | Đánh giá CTGAN/TVAE/TabDiff/dynamic/post-repair bằng cùng evaluator | Matched local comparison |

Harpoon compatibility run chỉ dùng một trial để phát hiện lỗi môi trường và protocol. Nếu cả ba dataset chạy sạch và kết quả cùng order of magnitude với artifact, mới mở rộng lên năm trials. Điều này tránh tốn compute cho một pipeline chưa được xác nhận.

## 6. Việc nên rerun tiếp theo sau batch hiện tại

Ưu tiên:

1. mở rộng Harpoon lên năm trials nếu compatibility run sạch;
2. implement matched Harpoon/CTDF subset trên chính frozen union graph của Shoppers và Default;
3. chạy evaluator chung cho ba dataset với TabDiff, dynamic, final repair và baseline generators;
4. chỉ sau đó cân nhắc TabSyn local hoặc backbone robustness.

Không ưu tiên ngay:

- thêm nhiều GAN/VAE cũ ngoài CTGAN/TVAE;
- đưa Insurance thành main dataset chỉ để có LCATS overlap;
- dùng Adult làm main dynamic result khi violation đã bão hòa;
- chép paper numbers của SAGE/TabSyn vào cột local và gọi đó là head-to-head comparison.

## 7. Bảng kết quả nên có trong paper

1. **General quality:** Shoppers, Default, Diabetes; TabDiff, CTGAN, TVAE, dynamic và post-repair; Shape, Trend, Alpha-Precision, Beta-Recall, Detection, utility và DCR.
2. **Matched constraint mechanism:** dynamic, frozen, shuffled, final repair, best-of-N, Harpoon-style và CTDF-style trên cùng graph subset.
3. **Official external evidence:** Harpoon Adult/Default/Shoppers, CuTS Adult/German và LCATS Insurance; tách khỏi main ranking.
4. **Trajectory ablation:** early/middle/late, refresh policy, state readability và transition subgroup.

Thiết kế này vừa giữ tính so sánh với literature, vừa bảo vệ claim cốt lõi rằng topology thay đổi theo row và denoising state cung cấp thông tin vượt static/final-only controls.
