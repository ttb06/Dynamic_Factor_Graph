# Đánh giá bằng metric phổ biến và Harpoon-style baseline

**Ngày chạy:** 10-10-2026  
**Dataset:** Default of Credit Card Clients  
**Backbone:** TabDiff epoch 8000  
**Protocol:** 3 paired sampling seeds, 4.096 rows/seed, 50 reverse steps

## 1. Mục tiêu

Vòng này trả lời hai câu hỏi:

1. Dynamic graph có giữ chất lượng bảng theo các metric thường dùng trong synthetic
   tabular data hay chỉ cải thiện metric constraint tự đặt ra?
2. Method có còn lợi thế khi so với một Harpoon-style per-step energy baseline trên
   đúng checkpoint, graph, seed và số mẫu hay không?

Mọi con số trong báo cáo là kết quả chạy local. Không trộn số công bố từ paper vào
bảng matched comparison.

## 2. Baseline và điều kiện so sánh

- **TabDiff:** official sampler và checkpoint local; không guidance.
- **Frozen graph:** cùng action layer với method nhưng active topology được đóng băng.
- **Final repair:** cùng graph và calibration policy, chỉ sửa sau khi sinh xong.
- **Harpoon-style hard:** tại mỗi reverse step, đọc predicted-clean state, xác định
  các row-edge đang vi phạm và tăng target logit của consequent trước discrete update.
- **Harpoon-style calibrated:** chỉ guidance phần violation mass vượt train-only
  Wilson target.
- **Dynamic graph:** explicit active graph được refresh theo row và timestep, dùng
  annealed persistence, calibrated quota, component action planning và direct
  state intervention trong middle window.

Harpoon-style baseline là **local rerun variant**, không phải official reproduction.
Lý do là TabDiff dùng absorbing categorical diffusion; categorical response được
port thành target-logit energy thay vì gradient trên continuous one-hot state.
Graph hiện tại có ba categorical consequents nên proxy bao phủ đúng các action target
của main experiment.

Strength `{1,2,4,8,16,32,64}` được sweep. Operating point được chọn bằng
**calibration error**, không nhìn held-out error:

- hard: weight 8;
- calibrated: weight 32.

Strength 0 tái tạo official TabDiff sample chính xác từng cell, xác nhận paired RNG.

## 3. Định nghĩa metric

| Metric | Ý nghĩa | Tốt hơn khi |
| --- | --- | --- |
| Shape error | Sai khác marginal distribution của từng cột | Thấp |
| Trend error | Sai khác pairwise column trends | Thấp |
| Alpha-Precision | Synthetic rows nằm trong support của dữ liệu thật | Cao |
| Beta-Recall | Mức synthetic data phủ support/modes của dữ liệu thật | Cao |
| Detection | Khả năng real-vs-synthetic classifier bị đánh lừa | Cao, gần 1 |
| MLE XGBoost AUC | Train trên synthetic, test trên held-out real | Cao |
| DCR | Tỷ lệ synthetic rows gần train hơn held-out | Gần 50% |
| Macro violation | Trung bình violation rate của ba graph edges | Phụ thuộc hardness |
| Held-out violation error | Trung bình `abs(synthetic violation rate - held-out real rate)` | Thấp |
| Calibration violation error | Sai lệch tương tự so với train-only calibration split | Thấp |

DCR được sửa để train và held-out reference pools có cùng kích thước. Trước sửa,
train pool lớn hơn làm score khoảng 91,8% dù method gần như không đổi; con số đó
không nên dùng. Với equal-sized pools, kết quả nằm quanh 52,4% và có nghĩa rõ hơn.

## 4. Kết quả matched comparison

Trung bình trên ba sampling seeds:

| Method | Shape ↓ | Trend ↓ | Alpha ↑ | Beta ↑ | Detection ↑ | AUC ↑ | DCR, 50 ideal | Held-out error ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TabDiff | 1.9833 | 2.2165 | 98.4736 | **52.5109** | 0.9528 | 0.7230 | 52.42 | 0.0309 |
| **Dynamic graph** | **1.9591** | 2.2116 | 98.5218 | 52.4436 | **0.9590** | 0.7228 | 52.38 | **0.0120** |
| Frozen graph | 1.9600 | 2.2119 | 98.4990 | 52.4533 | 0.9587 | 0.7238 | 52.36 | 0.0154 |
| Final repair | 1.9556 | **2.1909** | 98.5003 | 52.4761 | 0.9546 | **0.7278** | 52.43 | 0.0161 |
| Harpoon-style hard | 1.9772 | 2.2317 | **98.5284** | 52.4398 | 0.9570 | **0.7278** | 52.43 | 0.0179 |
| Harpoon-style calibrated | 1.9742 | 2.2016 | 98.4898 | 52.5054 | 0.9581 | 0.7253 | 52.38 | 0.0272 |

### 4.1. So với TabDiff

Dynamic graph:

- giảm Shape error `0.0242` điểm, tương đương `1.22%`;
- giảm Trend error `0.0050` điểm, tương đương `0.22%`;
- tăng Alpha-Precision `0.0482` điểm;
- giảm Beta-Recall `0.0673` điểm;
- tăng Detection `0.0062`;
- thay đổi AUC `-0.0002`, về thực tế là giữ nguyên utility;
- giảm held-out semantic error **61.2%**.

Real-train XGBoost AUC bằng đúng evaluator là `0.7706`. TabDiff đạt `0.7230` và
dynamic đạt `0.7228`, tức dynamic giữ khoảng `93.8%` real-data utility ceiling.

Các chênh lệch Shape/Trend/Alpha/Beta/AUC nhỏ hơn độ lệch giữa ba sampling seeds.
Vì vậy kết luận đúng không phải dynamic graph cải thiện general quality đáng kể, mà
là nó **giữ chất lượng ở mức TabDiff trong khi cải thiện semantic distribution**.

### 4.2. So với Harpoon-style proxy

| Method | Macro violation ↓ | Held-out error ↓ | Calibration error ↓ | Runtime guidance/seed |
| --- | ---: | ---: | ---: | ---: |
| Dynamic graph | 0.1266 | **0.0120** | **0.0091** | 8.12 s |
| Harpoon-style hard, weight 8 | **0.1253** | 0.0179 | 0.0121 | 2.62 s |
| Harpoon-style calibrated, weight 32 | 0.1471 | 0.0272 | 0.0296 | 2.63 s |

Dynamic graph giảm:

- held-out error khoảng **33.1%** so với Harpoon-style hard;
- calibration error khoảng **24.8%** so với Harpoon-style hard.

Harpoon hard có raw macro violation thấp hơn dynamic khoảng `0.0013`, đồng thời AUC
cao hơn `0.0050`. Nếu bài toán coi mọi edge là hard constraint, đây là lợi thế thật
của Harpoon. Nhưng ba edges trong graph đã được audit là quan hệ xác suất. Khi tăng
weight từ 8 lên 16–64, raw violation tiếp tục giảm tới khoảng `0.109`, trong khi
held-out error tăng trở lại từ `0.0179` lên `0.0235–0.0370`. Ép violation về 0 do đó
không phải mục tiêu đúng trong setting này.

Harpoon nhanh hơn khoảng 3,1 lần trong phần guidance. Dynamic dùng trung bình 85,3
explicit action events; Harpoon hard phát khoảng 7.273 row-edge guidance requests,
trong đó khoảng 2.564 còn editable khi categorical state chưa unmask. Hai count này
không cùng đơn vị, nhưng cho thấy dynamic là controller thưa và đắt hơn trên mỗi action.

## 5. Kết luận khoa học

### Bằng chứng hiện đã đủ tốt

1. Method không mua semantic gain bằng một degradation rõ rệt ở marginal fidelity,
   joint fidelity, support coverage, downstream utility hoặc DCR.
2. Dynamic graph tốt hơn TabDiff, frozen graph, final repair và matched
   Harpoon-style energy trên metric phù hợp với probabilistic constraints.
3. Lợi thế so với Harpoon không đến từ một strength yếu: sweep tới 64 cho thấy
   frontier đảo chiều do over-enforcement.
4. DCR quanh 52,4% cho mọi method không cho thấy guidance làm tăng train proximity.

### Điều chưa được phép claim

- Chưa phải SOTA tổng quát trên synthetic tabular data: mới một dataset, một trained
  checkpoint và ba sampling seeds.
- Chưa phải official Harpoon reproduction.
- Không thể claim dynamic graph thắng trên hard satisfaction; Harpoon hard đạt raw
  violation thấp hơn và previous global-request experiments còn cho thấy energy
  guidance có thể đạt zero violation.
- Harpoon-style loss cũng re-evaluate state ở từng step, nên nó đã có implicit
  row/state-dependent activation. Novelty không thể chỉ viết là “graph thay đổi theo
  state”; phải đặt ở explicit calibrated lifecycle, persistence, sparse typed action
  và intervention-induced compositional propagation.
- General quality gains rất nhỏ và chưa có ý nghĩa thống kê; headline phải là
  semantic calibration với preserved quality, không phải overall fidelity SOTA.

## 6. Đánh giá hiện tại

Kết quả **đủ tốt để giữ dynamic graph làm main method và bắt đầu đóng khung bảng
thực nghiệm**. Trên benchmark hiện tại, method có semantic gain lớn (`61.2%` so với
TabDiff, `33.1%` so với strong Harpoon proxy) và không làm giảm utility đáng kể.

Để thành main SOTA claim cho paper, tối thiểu còn cần:

1. lặp lại matched table trên một dataset thứ hai có compositional probabilistic graph;
2. thêm ít nhất một trained checkpoint seed hoặc backbone thứ hai;
3. chạy hard-constraint control để ghi rõ vùng Harpoon thắng;
4. báo paired confidence interval cho semantic gain và quality deltas;
5. nếu có thể, đối chiếu một official constraint-guidance implementation; nếu không,
   giữ nhãn proxy xuyên suốt paper.

## 7. Artifacts

- Harpoon proxy implementation: `scripts/harpoon_graph_proxy.py`
- Harpoon strength sweep: `outputs/harpoon_natural_graph_aggregate_v3_20261010/`
- Common metric comparison: `outputs/harpoon_natural_common_quality_aggregate_v3_20261010/`
- Per-seed common metrics: `outputs/harpoon_natural_quality_s20261376_1067018_0/`,
  `outputs/harpoon_natural_quality_s20261377_1067018_1/`,
  `outputs/harpoon_natural_quality_s20261378_1067018_2/`
- Main dynamic confirmation: `outputs/natural_dag_stable_confirmation_v2_20261010/`
- Main technique report: `TECHNIQUE_REPORT_COMPOSITIONAL_DYNAMIC_GRAPH_20261010_VI.md`
