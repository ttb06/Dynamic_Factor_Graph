# Phân tích dynamic graph theo row và denoising state

Ngày chạy: 2026-10-15  
Dataset: **Default of Credit Card Clients**  
Backbone: TabDiff checkpoint epoch 8000  
Slurm job: `1081262` (`COMPLETED`, 2 phút 01 giây trên A40)  
Replay: 4.096 synthetic rows, 50 reverse steps, seed `20261440`

## 1. Vì sao chọn Default

Default hiện là dataset có bằng chứng mechanism rõ nhất: progressive temporal controller giảm actionable held-out violation error từ `0.006323` xuống `0.005097` (**19,38%**) và full held-out error từ `0.011192` xuống `0.010293` (**8,03%**) qua 3/3 seeds. Nó cũng có nhiều quan hệ tuần tự giữa `PAY_*`, `BILL_AMT*` và `PAY_AMT*`, nên dễ quan sát graph thay đổi theo row và theo trajectory hơn Shoppers.

Phân tích này dùng:

- **verified catalog:** 16 edges còn lại sau bootstrap discovery và train/held-out validation;
- **deployed graph:** 7/16 edges đủ điều kiện để controller tác động;
- **unguided replay:** dùng để đo state tự nhiên của từng row-edge, không bị action làm thay đổi trajectory;
- **guided run độc lập:** chỉ dùng để kiểm tra rằng mechanism thực sự đã tạo topology change và action event.

Không có real row nào được xuất. Tất cả ví dụ bên dưới là synthetic rows.

## 2. Định nghĩa trạng thái của một edge

Với edge có dạng `IF antecedent THEN consequent`, tại mỗi predicted-clean state `x̂₀(t)` và mỗi row, edge nhận một trong năm trạng thái:

| State | Ý nghĩa |
|---|---|
| `unreadable` | Timestep còn sớm hơn reliability cutoff của edge; chưa đánh giá applicability hoặc violation. |
| `inactive` | Antecedent không đúng sau row-wise EMA hysteresis; edge không thuộc active graph của row. |
| `satisfied` | Antecedent đúng và consequent đang đúng. |
| `violated` | Antecedent đúng, consequent sai, nhưng violation chưa tồn tại đủ lâu để hành động. |
| `trigger_ready` | Violation đã tồn tại ít nhất `K(t)` bước; edge đủ bằng chứng để chuyển sang action layer. |

`K(t)` giảm từ 8 ở vùng sớm xuống 1 ở cuối trajectory. Raw applicability được làm ổn định bằng EMA hysteresis: nếu antecedent vừa đổi trạng thái và row-edge còn volatile, thay đổi phải lặp lại 3 lần mới được chấp nhận. Vì vậy graph không chỉ phụ thuộc timestep mà còn phụ thuộc lịch sử state của chính row.

## 3. Timing ở mức toàn bộ population

Tổng số cặp row-edge là `4.096 × 16 = 65.536`.

| Reverse time | Readable row-edge pairs | Satisfied | Violated, chưa đủ evidence | Trigger-ready | Nhận xét |
|---:|---:|---:|---:|---:|---|
| `t=0.5918` | 4.096 | 333 | 7 | 0 | Edge đầu tiên vừa đủ ổn định; chưa nên action. |
| `t=0.4490` | 16.384 | 1.753 | 263 | 1 | Trigger-ready đầu tiên xuất hiện, vẫn rất thưa. |
| `t=0.3673` | 24.576 | 2.524 | 124 | 241 | Đây là thời điểm guided run bắt đầu action. |
| `t=0.2449` | 49.152 | 4.913 | 139 | 527 | Nhiều numerical relations bắt đầu đọc được. |
| `t=0.1020` | 57.344 | 5.784 | 3 | 692 | Violation tạm thời gần như đã được phân loại thành ready hoặc resolved. |
| `t=0.0000` | 65.536 | 6.799 | 0 | 725 | Toàn bộ catalog đọc được; 88,5% row-edge pairs không applicable. |

Kết quả này không ủng hộ việc bật toàn bộ graph ở early steps. Edge-level readability tạo một curriculum tự nhiên: relation về trạng thái thanh toán đọc được trước, relation có điều kiện `BILL_AMT == 0` chỉ đáng tin ở late-middle/late steps.

## 4. Ví dụ row-level đáng report

### S1 — cùng row, ba edges cùng chuyển sang action-ready

Synthetic row index `570` có 12/16 edges active ở ít nhất một bước.

- `bscg_000`: `PAY_6=-2 ∧ BILL_AMT4≤0 → PAY_5=-2`
  - `unreadable → inactive` tại `t=0.5918`;
  - `inactive → violated` tại `t=0.4898`;
  - `violated → trigger_ready` tại `t=0.3878`, sau streak 6.
- `bscg_001`, phiên bản antecedent chặt hơn với `BILL_AMT4=0`, chỉ trở nên readable tại `t=0.4286` và ready tại `t=0.3469`.
- `bscg_009`: `PAY_4=-2 → PAY_5=-2` có cùng target nhưng khác applicability; nó active muộn hơn `bscg_000` và ready tại `t=0.3469`.

Đây là ví dụ rõ rằng active graph không phải một mask toàn cục: ba relation cùng target nhưng được bật ở thời điểm khác nhau do cutoff và antecedent khác nhau.

### S2 — topology bật, tắt, rồi bật lại

Synthetic row index `3459` cho thấy `bscg_000`:

`satisfied (t=0.5918) → violated (0.5510) → inactive (0.5102) → violated (0.1633) → trigger_ready (0.1224)`.

Đây là bằng chứng trực tiếp cho state-dependent topology. Một frozen graph hoặc final-only rule không biểu diễn được việc relation chỉ applicable ở một số đoạn của trajectory.

### S3 — các numerical edges chỉ đáng đánh giá muộn

Synthetic row index `3963`:

- `BILL_AMT2=0 → PAY_AMT1=0` chỉ readable ở `t=0.2449`, ready tại `0.1837`;
- `BILL_AMT3=0 → PAY_AMT2≤0` chỉ readable ở `t=0.1429`, ready tại `0.1020`;
- `BILL_AMT4=0 → PAY_AMT3≤0` chỉ readable ở `t=0.0612`, ready tại `0.0408`.

Điều này cho thấy schedule cần **per-edge**, không chỉ một global middle-step window.

### S5 — violation tự phục hồi, không nên sửa ngay

Với synthetic row index `4017`, cả `bscg_000` và `bscg_009` chuyển:

`satisfied → violated` tại `t=0.4694`, rồi trở lại `satisfied` ở `t=0.4082` trước khi đủ evidence.

Nếu action layer sửa ngay khi nhìn thấy một-step violation, nó sẽ can thiệp vào lỗi mà denoiser tự giải quyết. Đây là motivation trực tiếp cho temporal persistence gate.

### S7 — active subgraph lớn nhưng vẫn thay đổi theo state

Synthetic row index `602` có 12 active edges, 13 state transitions và 86 trigger-ready states. Đáng chú ý, `bscg_010: PAY_5=-2 → PAY_6=-2` chuyển `violated → trigger_ready` ở `t=0.2653`, sau đó thành `inactive` ở `t=0.1020`. Vì antecedent biến mất, edge phải bị deactive thay vì tiếp tục ép consequent đến cuối.

## 5. Kiểm tra với guided controller

Guided run độc lập dùng deployed graph 7 edges, seed `20261447`, ghi nhận:

- 107 topology additions;
- 131 topology removals;
- 238 topology flips;
- 76 action events;
- action đầu tiên tại `t=0.3673`, khi `K(t)=5`;
- action tập trung từ middle-late đến late steps, mạnh nhất quanh `t=0.1837`, `0.1020`, `0.0612`, `0.0408`.

Như vậy có hai bằng chứng tách biệt:

1. unguided replay cho thấy relation thực sự đổi giữa inactive/satisfied/violated/ready theo row và state;
2. guided run cho thấy controller sử dụng các thay đổi này để quyết định thời điểm hành động.

Không được hiểu `trigger_ready` trong replay là một action đã xảy ra. Nó chỉ nói rằng row-edge đã vượt readability, stabilized applicability và persistence gate.

## 6. Kết luận có thể dùng trong report

Kết quả hỗ trợ claim hẹp nhưng rõ: **một constraint edge không nên được xem là active cố định trong toàn bộ reverse process**. Applicability và confidence của edge thay đổi theo predicted-clean state; một số violation tự phục hồi, trong khi các numerical relations chỉ đủ đọc ở late-middle/late steps. Row-wise topology stabilization kết hợp per-edge readability và temporal evidence gate tạo active subgraph khác nhau cho từng row và từng timestep.

Phân tích này chưa chứng minh toàn bộ improvement đến từ topology dynamics; để claim causal cần ablation đối chiếu cùng action layer giữa frozen graph, global schedule và row-time dynamic graph.

## 7. Files tái chạy

- Replay exporter: `analysis/llm_feedback_20260922/scripts/export_row_edge_timeline.py`
- Slurm script: `analysis/llm_feedback_20260922/slurm/634_export_row_edge_timeline.sbatch`
- Raw timeline: `analysis/llm_feedback_20260922/outputs/row_edge_timeline_default_1081262/row_edge_timeline.json`
- Aggregate counts: `analysis/llm_feedback_20260922/outputs/row_edge_timeline_default_1081262/aggregate_state_counts.csv`
- Visualization builder: `analysis/llm_feedback_20260922/scripts/build_row_edge_timeline_viz.py`

