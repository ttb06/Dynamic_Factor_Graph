# Các bảng kết quả nên report — Default

Dataset duy nhất trong bản report hiện tại: **Default of Credit Card Clients**.  
Backbone chính: **TabDiff, ICLR 2025**, checkpoint epoch 8000, 50 reverse steps.  
Số synthetic rows: 4.096 mỗi seed.

## 1. Quy tắc tổ chức bảng

Nên tách kết quả thành ba nhóm, không đưa tất cả methods vào một bảng:

1. **External/generator baselines:** TabDiff, Harpoon-style và proposed method. Đây là bảng đầu tiên người đọc cần thấy.
2. **Post-generation baselines:** final repair và proposed method. Đây là research question khác: can thiệp trong trajectory có hơn sửa ở endpoint hay không.
3. **Mechanism ablations:** frozen topology, shuffled topology, event-triggered gate và progressive gate. Những dòng này không phải external baselines.

Không đưa số Harpoon do paper báo cáo vào cùng self-run table: paper dùng các range/category/AND/OR constraints khác graph và verifier của dự án. Dòng hiện tại phải ghi **Harpoon-style proxy**, không ghi “Harpoon reproduction”.

## 2. Table 1 — Main comparison với external baselines

Đây nên là bảng kết quả đầu tiên. Bảng chỉ chứa unconditional backbone, fixed-constraint diffusion guidance và proposed dynamic graph; không chứa final repair hoặc frozen/shuffled ablations.

### 2.1 Semantic profile và distortion

Kết quả matched ba seeds trên Default graph 8 cạnh của đợt Harpoon comparison:

| Method | Publication/status | Held-out profile error ↓ | Calibration error ↓ | Macro violation | Numeric KS ↓ | Categorical TV ↓ | Interventions ↓ | Added runtime ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| TabDiff | ICLR 2025; official backbone | 0.05452 ± 0.00995 | 0.03991 | 0.17545 | 0.03086 | **0.01763** | 0 | 0 |
| Harpoon-style proxy | Harpoon: ICLR 2026; local proxy | 0.02792 ± 0.00421 | 0.02122 | **0.13746** | **0.03078** | 0.02075 | 7.130,3 editable cells | **3,68 s** |
| **Dynamic constraint graph** | Proposed | **0.02250 ± 0.00079** | **0.01852** | 0.13849 | 0.03083 | 0.02013 | **195,3 actions** | 72,81 s |

Các gain được phép ghi trong caption/text:

- proposed giảm held-out profile error **58,7%** so với TabDiff;
- giảm **19,4%** so với Harpoon-style proxy;
- Harpoon-style có raw macro violation thấp hơn proposed, nhưng profile error cao hơn. Điều này cho thấy ép violation xuống thấp chưa chắc tái tạo đúng legitimate violation rate của real held-out data;
- Numeric KS gần như giữ nguyên; categorical TV của cả hai guidance methods cao hơn TabDiff nhẹ.

`Added runtime` là overhead của guidance trong implementation hiện tại, không phải tổng chi phí train backbone. Runtime của proposed chưa được tối ưu, vì vậy không nên dùng bảng này để claim efficiency.

### 2.2 Standard fidelity và utility của cùng external baselines

| Method | Shape error ↓ | Trend error ↓ | Alpha precision ↑ | Beta recall ↑ | Detection ↑ | XGBoost AUC ↑ | DCR, 50 ideal |
|---|---:|---:|---:|---:|---:|---:|---:|
| TabDiff | **2.0030** | **2.2268** | 98.4531 | **52.5347** | 0.9553 | 0.7227 | 52.2705 |
| Harpoon-style proxy | 2.0310 | 2.3122 | **98.5917** | 52.3161 | 0.9535 | 0.7238 | **52.1240** |
| **Dynamic constraint graph** | 2.0305 | 2.2843 | 98.5426 | 52.3958 | **0.9574** | **0.7265** | 52.2217 |

Thông điệp đúng của bảng này:

- semantic-profile gain không gây quality collapse;
- proposed không thắng TabDiff trên Shape/Trend;
- proposed tốt nhất nhẹ về Detection và XGBoost AUC trong comparison này;
- không được claim proposed là SOTA về general tabular fidelity.

### 2.3 Cảnh báo phiên bản

Table 1 hiện là matched comparison hợp lệ nhưng dùng graph 8 cạnh của đợt trước. Progressive controller mới nhất dùng graph 7 cạnh. Bản cuối cần rerun ba dòng `TabDiff / Harpoon-style / Progressive dynamic` trên cùng graph 7 cạnh, seeds `20261447–20261449`, rồi thay Table 1. Không trộn số của graph 8 cạnh và graph 7 cạnh trong cùng một statistical comparison.

## 3. Table 2 — General synthesis baselines, auxiliary table

Bảng này cho thấy method vẫn nằm trong vùng chất lượng của các tabular generators phổ biến. Đây là one-seed local comparison; phù hợp appendix hoặc preliminary report, chưa nên thay Table 1.

| Method | Venue/status | Shape error ↓ | Trend error ↓ | Alpha precision ↑ | Beta recall ↑ | Detection ↑ | XGBoost AUC ↑ | Held-out profile error ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CTGAN-SDV | NeurIPS 2019; SDV implementation | 15.0455 | 17.6522 | 65.4671 | 19.7388 | 0.6161 | 0.6663 | 0.2741 |
| TVAE-SDV | NeurIPS 2019; SDV implementation | 10.8578 | 6.8219 | 81.5438 | 23.6532 | 0.5463 | 0.7315 | 0.1993 |
| TabDiff | ICLR 2025; official backbone | 1.6429 | 1.7127 | 98.3884 | 52.5806 | **0.9474** | 0.7482 | 0.0187 |
| **Dynamic graph** | Proposed, one seed | **1.6366** | **1.7083** | **98.4071** | **52.6066** | 0.9422 | **0.7530** | **0.0140** |

Không thêm TabSyn/TabDDPM bằng số copy từ paper vào bảng này. Chỉ thêm khi rerun bằng cùng split, sample size và evaluator. Nếu chưa rerun, ghi chúng trong related-work/baseline feasibility table, không điền số vào result table.

## 4. Table 3 — Dynamic trajectory so với final repair

Đây là comparison mới nhất của progressive controller 7 cạnh, ba seeds `20261447–20261449`. Nó phải nằm ở bảng riêng vì final repair là post-processing baseline, không phải generator/guidance baseline cùng loại với TabDiff và Harpoon.

| Metric | Final repair | Progressive dynamic | Relative change | Paired wins |
|---|---:|---:|---:|---:|
| Actionable held-out error ↓ | 0.006323 | **0.005097** | **19,38% tốt hơn** | **3/3** |
| Full held-out error ↓ | 0.011192 | **0.010293** | **8,03% tốt hơn** | **3/3** |
| Shape error ↓ | 2.041838 | **2.038613** | 0,16% tốt hơn | 1/3 |
| Trend error ↓ | 1.971839 | **1.961033** | 0,55% tốt hơn | 2/3 |
| XGBoost AUC ↑ | **0.731765** | 0.730041 | 0,24% kém hơn | 1/3 |

Caption nên nói rõ:

- graph discovery và trajectory calibration được đóng băng trước ba evaluation seeds;
- profile error được đo bằng independent evaluation graph, không chỉ bằng 7 guidance edges;
- gain chính nằm ở semantic profile, trong khi downstream AUC giảm nhẹ;
- residual final repair sau progressive controller không thay đổi kết quả trong pilot seed, cho thấy online controller đã tới fixed point của cùng certifier.

## 5. Table 4 — Mechanism ablation

Một seed cố định `20261447`, cùng checkpoint, graph, row count và evaluator:

| Controller | Actionable error ↓ | Full held-out error ↓ | Online actions | Residual actions |
|---|---:|---:|---:|---:|
| Final repair | 0.007035 | 0.009833 | 0 | 55 |
| Fixed-streak dynamic gate | 0.005898 | 0.009133 | 94 | 0 |
| Event-triggered + repair | 0.006705 | 0.009629 | 30 | 46 |
| **Progressive dynamic gate** | **0.005652** | **0.009027** | 76 | 0 |
| Shuffled row topology + repair | 0.007035 | 0.009833 | 60 | 53 |

Interpretation cần report:

- progressive evidence schedule tốt hơn fixed streak và event-only;
- shuffled topology có 60 online actions nhưng không tạo semantic gain;
- số action hoặc số topology flips tự nó không giải thích improvement; row–edge assignment đúng mới quan trọng;
- final repair cần ít action hơn nhưng có held-out profile error cao hơn.

## 6. Table 5 — Deployed graph và edge-level schedule

Đây là bảng mô tả chính xác method đang điều khiển cái gì. Có thể đặt cuối Method section hoặc đầu Experiment section.

| Edge | Typed relation | Class | Train support | Train coverage | Calibration violation | Earliest readable `t` |
|---|---|---|---:|---:|---:|---:|
| `bscg_000` | `PAY_6=-2 ∧ BILL_AMT4≤0 → PAY_5=-2` | hard | 2.478 | 9,18% | 0,00% | 0,5918 |
| `bscg_001` | `PAY_6=-2 ∧ BILL_AMT4=0 → PAY_5=-2` | hard | 2.125 | 7,87% | 0,00% | 0,4490 |
| `bscg_002` | `BILL_AMT3=0 → PAY_AMT2≤0` | near-hard | 2.577 | 9,54% | 1,98% | 0,1429 |
| `bscg_003` | `BILL_AMT6=0 → PAY_AMT5=0` | near-hard | 3.603 | 13,34% | 1,58% | 0,1020 |
| `bscg_004` | `BILL_AMT2=0 → PAY_AMT1=0` | near-hard | 2.258 | 8,36% | 1,64% | 0,2449 |
| `bscg_005` | `BILL_AMT5=0 → PAY_AMT4=0` | near-hard | 3.146 | 11,65% | 1,72% | 0,1633 |
| `bscg_008` | `BILL_AMT4=0 → PAY_AMT3≤0` | near-hard | 2.870 | 10,63% | 1,43% | 0,0816 |

Ý nghĩa của bảng:

- mỗi edge có support và legitimate violation mass khác nhau;
- mỗi edge trở nên readable ở một timestep khác nhau;
- `bscg_000` có thể được đánh giá ở middle steps, trong khi `bscg_008` chỉ đáng tin rất muộn;
- vì vậy một global guidance schedule là không đủ.

## 7. Table 6 — Khi nào topology và action thay đổi?

Một seed `20261447`, deployed graph 7 cạnh:

| Reverse-time region | Topology flips | Action events | Diễn giải |
|---|---:|---:|---|
| `t > 0.65` | 0 | 0 | Preview ngoài active window; graph không có authority. |
| `0.50 < t ≤ 0.65` | 27 | 0 | Một số edge bắt đầu readable, nhưng temporal evidence chưa đủ. |
| `0.25 < t ≤ 0.50` | 126 | 6 | Phần lớn topology discovery xảy ra ở middle trajectory. |
| `0.10 < t ≤ 0.25` | 68 | 37 | Active graph đã ổn định hơn; action tăng mạnh. |
| `t ≤ 0.10` | 17 | 33 | Late corrections xử lý persistent violations còn lại. |
| **Tổng** | **238** | **76** | 107 additions và 131 removals. |

Bảng này hỗ trợ hai claim khác nhau:

1. **Topology dynamics xuất hiện chủ yếu ở middle trajectory**: 126/238 flips trong `(0.25, 0.50]`.
2. **Actuation không chỉ xảy ra ở middle**: phần lớn action được hoãn tới middle-late/late vì evidence gate yêu cầu preview đủ ổn định.

Do đó wording chính xác là: method khám phá và cập nhật active graph ở middle steps, sau đó thực hiện phần lớn corrections khi evidence đã tích lũy đủ. Không nên claim mọi action đều nằm ở middle steps.

## 8. Định nghĩa metric cần đặt ngay dưới bảng

### Held-out constraint-profile error

Với edge `e`:

\[
E_{profile}=\frac{1}{|E|}\sum_e\left|r_e^{syn}-r_e^{test}\right|.
\]

`r_e` là violation rate chỉ trên rows mà edge applicable. Metric thấp hơn nghĩa synthetic data tái tạo tốt hơn conditional semantic profile của real held-out data.

### Actionable held-out error

Profile error chỉ trên subset edges có đủ bằng chứng để action. Đây là metric trực tiếp nhất cho actuator nhưng không được báo một mình.

### Full held-out error

Profile error trên independent evaluation graph đầy đủ. Metric này kiểm tra gain có generalize ngoài guidance edges hay không.

### Macro violation

Trung bình raw violation rate qua edges. Thấp hơn không luôn tốt hơn, vì real data có legitimate violations. Dùng làm diagnostic, không dùng làm primary metric.

### Shape/Trend

Trong các bảng self-run ở đây, chúng được ghi dưới dạng **error percentage**, nên thấp hơn tốt hơn. Không trộn với paper sử dụng Shape/Trend score, nơi cao hơn tốt hơn.

### Alpha precision/Beta recall

- Alpha precision: synthetic samples nằm trong support của real distribution tốt đến đâu;
- Beta recall: synthetic data bao phủ real modes tốt đến đâu.

### Detection

Khả năng phân biệt real và synthetic được đổi thành quality score; cao hơn tốt hơn theo evaluator hiện dùng.

### DCR

Distance-to-closest-record privacy diagnostic. `50` là mức train-vs-heldout cân bằng lý tưởng trong evaluator; không diễn giải giá trị thấp/cao đơn thuần như một privacy guarantee.

## 9. Bảng nào đặt ở đâu

### Main paper

1. Table 1.1 — external baselines: semantic profile và distortion.
2. Table 1.2 — standard fidelity/utility của cùng methods.
3. Table 3 — proposed so với final repair.
4. Table 4 — mechanism ablation.
5. Table 5 — deployed graph; có thể rút gọn còn Edge, Class, Violation và Cutoff.

### Appendix

- general synthesis baselines CTGAN/TVAE;
- full edge statistics và bootstrap uncertainty;
- trajectory phase table;
- per-seed numbers;
- action choices, affected-row drift, density delta và runtime profiling.

## 10. Việc còn thiếu trước khi khóa bảng paper

Ưu tiên duy nhất để làm Table 1 hoàn toàn nhất quán:

1. rerun TabDiff no-guidance, Harpoon-style và progressive dynamic trên **cùng deployed graph 7 cạnh**;
2. dùng ba seeds `20261447–20261449`;
3. dùng cùng independent 29-edge evaluation graph;
4. chạy cùng quality evaluator cho Shape, Trend, Alpha, Beta, Detection và XGBoost AUC;
5. report cả intervention count và added runtime;
6. giữ tên `Harpoon-style proxy` trừ khi implementation khớp code chính thức.

Cho đến khi rerun hoàn tất, Table 1 dùng được cho internal report/catch-up, nhưng bản submission phải ghi rõ nó thuộc graph 8 cạnh và không ghép trực tiếp với progressive 7-edge Table 3.

