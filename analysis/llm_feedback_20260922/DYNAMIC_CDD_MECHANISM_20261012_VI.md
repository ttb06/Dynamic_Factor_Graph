# Dynamic graph kết hợp feasibility projection kiểu CDD

## Ranh giới với CDD

Constrained Discrete Diffusion (CDD, NeurIPS 2025) không đánh giá trên dữ liệu bảng. Bài báo dùng discrete sequence diffusion cho kiểm soát độc tính văn bản, thiết kế phân tử và instruction-constrained completion. CDD đưa một phép chiếu tối thiểu KL vào sampling để decoded sequence thỏa tập constraint đã cho. Vì vậy, contribution của dự án này không thể là “đưa constraint vào từng reverse step” hay “KL projection cho categorical diffusion”.

Khoảng trống được kiểm tra ở đây là **tập khả thi thay đổi theo row và trạng thái trajectory**. CDD giả định constraint set được chỉ định trước; method của ta xây dựng `C_i(t)` từ dynamic graph, bằng applicability của edge trên `xhat_0(t)`, độ ổn định của evidence và hardness đã calibrate trên train.

## Mechanism pilot

1. Denoiser tạo `xhat_0(t)` cho toàn bộ row.
2. Mỗi edge được đánh giá applicability trên row. Edge chỉ đủ điều kiện sau hai preview liên tiếp đồng ý và sau cutoff đọc được (`t <= 0.5`).
3. Bản persistent chuyển edge sang trạng thái committed. Edge đã committed được giữ đến cuối trajectory để sample không thể né constraint bằng cách đổi antecedent.
4. Với categorical consequent, action layer chiếu phân phối denoiser sang phân phối gần nhất theo `D_KL(q || p)` mà target là argmax khả thi. Nếu category đã unmask sai, coordinate được re-mask rồi tiếp tục reverse transition.
5. Với soft edge, chỉ violation mass vượt exception rate calibrate từ train mới được sửa. Hai cách chọn action cùng budget được so sánh: chi phí KL nhỏ nhất và violation nặng nhất.
6. Pilot chạy trên DAG Default `BILL_AMT3 -> PAY_4 -> PAY_5 -> PAY_6`, TabDiff checkpoint cố định, 4.096 mẫu, 50 step, paired seed.

Đây là domain adaptation của CDD, không phải official reproduction. Pilot mới có exact projection cho categorical equality; numeric/mixed joint projection chưa được triển khai.

## Kết quả một seed

| Method | Macro violation | Sai lệch so với held-out | Sai lệch so với train calibration | Shape error (%) | Trend error (%) | Alpha precision (%) | Beta recall (%) | MLE AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TabDiff | 0.1637 | 0.0407 | 0.0462 | 2.4058 | 2.5072 | 98.6638 | 52.5049 | 0.7168 |
| Static CDD, hard, all steps | 0.0000 | 0.1231 | 0.1175 | 2.1613 | 2.6239 | 99.2650 | 51.9010 | 0.7260 |
| Dynamic, transient middle, hard | 0.0554 | 0.0677 | 0.0621 | 2.4236 | 2.5997 | 98.9143 | 50.6055 | 0.7278 |
| Dynamic, persistent, hard | 0.0095 | 0.1136 | 0.1080 | 2.5683 | 2.9514 | 99.0174 | 50.5192 | 0.7133 |
| Dynamic, persistent, calibrated, min-KL | 0.1302 | **0.0199** | **0.0127** | 2.3096 | **2.4198** | 98.7831 | 50.7145 | 0.7231 |
| Dynamic, persistent, calibrated, severity | 0.1334 | 0.0204 | 0.0159 | 2.3322 | 2.4485 | 98.7535 | 51.4274 | **0.7300** |
| Calibrated final repair (corrected) | 0.1175 | **0.0103** | **0.0005** | đang đánh giá lại | đang đánh giá lại | đang đánh giá lại | đang đánh giá lại | đang đánh giá lại |

Diễn giải:

- Ép mọi soft relation thành hard constraint tạo `0 violation` nhưng làm sai lệch dependency so với held-out tăng gấp khoảng ba lần. Đây không phải kết quả tốt dù satisfaction tuyệt đối.
- Dynamic calibrated min-KL giảm sai lệch held-out khoảng 51% và sai lệch train calibration khoảng 73% so với TabDiff, đồng thời cải thiện Shape, Trend và MLE AUC trên paired run.
- Transient middle không đủ: constraint có thể bị phá sau khi action ngừng. Middle phù hợp làm thời điểm quyết định active set; feasibility cần được giữ đến cuối hoặc được kiểm tra lại bằng hysteresis.
- Min-KL và severity tạo trade-off có ý nghĩa: min-KL giữ fidelity tốt hơn; severity cho utility cao hơn.
- Calibrated final repair hiện tốt hơn online method trên chính constraint calibration. Do đó chưa thể claim online projection thắng repair. Claim tiềm năng phải dựa trên việc online guidance cải thiện joint distribution/utility hoặc constraint composition mà final repair không đạt, và cần xác nhận nhiều seed/dataset.

## Hypothesis tiếp theo có giá trị

1. **Finite-state edge controller:** dormant -> candidate -> active -> committed, với hysteresis riêng cho hard và soft edge. So với latch vĩnh viễn, controller có thể giảm false activation mà vẫn chống antecedent evasion.
2. **Joint graph projection:** giải đồng thời các edge active bằng augmented Lagrangian/alternating projection, thay vì projection tuần tự. Đây là bước cần thiết cho graph có cycle hoặc nhiều consequent xung đột.
3. **Mixed-type geometry:** KL projection cho categorical, Mahalanobis hoặc covariance-aware projection cho numerical; action selector chọn hướng sửa có chi phí phân phối nhỏ nhất.
4. **Compositional test:** đo thời điểm edge sau xuất hiện vì edge trước đã được projection; so với static graph, final repair và branch selection trên graph dài hơn ba cạnh.
5. **Fair post-repair comparison:** matched train-calibrated exception budget, số row thay đổi và compute. Nếu repair vẫn tốt ngang online trên fidelity/utility, giới hạn claim ở trajectory-level composition hoặc chuyển online layer thành selective fallback thay vì core method.

## Artifacts

- Implement: `scripts/dynamic_cdd_graph_projection.py`
- Sampling result đầu tiên: `outputs/dynamic_cdd_graph_1069974/`
- Corrected calibrated-repair rerun: `outputs/dynamic_cdd_graph_1069977/`
- Common evaluator đầu tiên: `outputs/dynamic_cdd_graph_eval_1069975/`
- Corrected common evaluator: Slurm `1069978` (pending/running tùy thời điểm đọc)

