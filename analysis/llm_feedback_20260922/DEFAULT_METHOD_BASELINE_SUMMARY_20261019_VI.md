# Tổng hợp method và baseline trên Default

Ngày tổng hợp: 2026-10-19  
Dataset: **Default of Credit Card Clients**  
Backbone của method: **TabDiff (ICLR 2025)**, checkpoint epoch 8000, 50 reverse steps  
Quy mô đánh giá chính: 4.096 synthetic rows mỗi seed

## 1. Method hiện tại

Phiên bản giữ lại là **progressive dynamic constraint graph**:

1. Graph đã được khám phá và kiểm chứng offline; controller không gọi LLM trong sampling.
2. Mỗi edge có applicability condition, typed relation, độ cứng và readability cutoff riêng.
3. Ở mỗi reverse step, `x_hat_0` được decode; applicability được tính lại cho từng row-edge và ổn định bằng EMA/hysteresis.
4. Số bước xác nhận cần thiết giảm dần từ đầu tới cuối active window: chỉ edge đủ ổn định mới được actuate.
5. Typed action sửa predicted-clean state rồi đưa nó trở lại chính reverse trajectory; không chạy diffusion lần hai.
6. Nếu calibration-only precursor test không cho thấy lợi ích của online control, controller abstain và dùng final certifier.

Graph deploy hiện tại có **7 edge**. Đây là phiên bản mới hơn graph 8 edge dùng trong comparison với Harpoon-style bên dưới.

## 2. Matched comparison với backbone và mechanism controls

Bảng này dùng cùng TabDiff checkpoint, sample seed, split và evaluator. Chỉ **TabDiff** là published baseline trong bảng. `Harpoon-style` là local adaptation của ý tưởng Harpoon, còn final repair là endpoint ablation; không trình bày hai dòng này như kết quả chính thức của các paper. Dynamic graph trong bảng là phiên bản 8 edge, chưa phải bảng cuối của progressive controller 7 edge.

| Method | Shape error ↓ | Trend error ↓ | Alpha ↑ | Beta ↑ | Detection ↑ | XGB AUC ↑ | Held-out relation error ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TabDiff | 1.9833 | 2.2165 | 98.4736 | **52.5109** | 0.9528 | 0.7230 | 0.0309 |
| Harpoon-style hard proxy | 1.9772 | 2.2317 | **98.5284** | 52.4398 | 0.9570 | **0.7278** | 0.0179 |
| Harpoon-style calibrated proxy | 1.9742 | 2.2016 | 98.4898 | 52.5054 | 0.9581 | 0.7253 | 0.0272 |
| Final repair | **1.9556** | **2.1909** | 98.5003 | 52.4761 | 0.9546 | **0.7278** | 0.0161 |
| **Dynamic graph** | 1.9591 | 2.2116 | 98.5218 | 52.4436 | **0.9590** | 0.7228 | **0.0120** |

### Gain của DynamicGraph

So với TabDiff:

- Held-out relation error giảm **61,2%**.
- Shape error giảm **1,22%**; Trend error giảm **0,22%**.
- Alpha tăng `0,0482` điểm phần trăm.
- Beta giảm `0,0673` điểm phần trăm và AUC giảm `0,0002`: gần như giữ nguyên, nhưng không phải gain.

So với Harpoon-style hard mechanism control:

- Held-out relation error giảm khoảng **33,0%**.
- Shape/Trend gần tương đương; Harpoon-style có AUC cao hơn `0,0050`.

So với final repair:

- Held-out relation error giảm khoảng **25,5%**.
- Final repair vẫn tốt hơn nhẹ về Shape, Trend và AUC.

Kết luận đúng từ bảng này: dynamic graph cải thiện mạnh conditional semantic profile nhưng chỉ giữ hoặc cải thiện rất nhỏ fidelity/utility phổ biến. Kết quả này không chứng minh đã vượt official Harpoon.

## 3. Phiên bản hiện tại: trajectory ablation so với final-only repair

Graph 7 edge, ba unseen seeds `20261447–20261449`; graph discovery và trajectory calibration đã đóng băng trước evaluation.

| Metric | Final repair | Progressive dynamic | Thay đổi tương đối | Paired wins |
| --- | ---: | ---: | ---: | ---: |
| Actionable held-out error ↓ | 0.006323 | **0.005097** | **19,38% tốt hơn** | **3/3** |
| Full held-out error ↓ | 0.011192 | **0.010293** | **8,03% tốt hơn** | **3/3** |
| Shape error ↓ | 2.041838 | **2.038613** | 0,16% tốt hơn | 1/3 |
| Trend error ↓ | 1.971839 | **1.961033** | 0,55% tốt hơn | 2/3 |
| XGBoost AUC ↑ | **0.731765** | 0.730041 | 0,24% kém hơn | 1/3 |

Final repair ở đây là **matched final-only control**, không phải external baseline. Kết quả mạnh nhất hiện tại là progressive controller thắng control này trên cả ba seed về hai held-out graph metrics, đồng thời Shape/Trend không suy giảm trung bình. Điểm yếu còn lại là XGBoost AUC giảm `0,001724`.

### Final repair hiện được implement thế nào?

Implementation đang report là một endpoint control tương đối mạnh, không phải phép overwrite đơn giản:

1. Chỉ chạy sau khi TabDiff đã sinh xong clean rows.
2. Recompute applicability và violation cho từng row-edge tại endpoint.
3. Với near-hard edge, giữ lại violation mass được calibration trên train thay vì ép mọi row thỏa constraint.
4. Với row cần sửa, tạo nhiều typed candidate actions và chọn action bằng reward gồm train-density, edit cost và collateral graph effect.
5. Lặp hữu hạn qua graph để xử lý các edge dùng chung endpoint; dừng khi hết hard residual hoặc không còn thay đổi.

Các bản đơn giản hơn có thể dùng làm sanity ablation: one-pass overwrite consequent; nearest feasible projection; hoặc iterative projection không có calibration/reward. Chúng không nên được gọi là baseline chính. Matched final-only control mạnh ở trên hữu ích hơn cho câu hỏi khoa học vì nó dùng cùng graph, actuator và calibration với proposed method; khác biệt chủ yếu còn lại là **can thiệp ở endpoint hay trong trajectory**.

## 4. Cơ chế có thật sự cần dynamic activation không?

Một seed cố định, cùng checkpoint, graph và evaluator:

| Controller | Actionable error ↓ | Full held-out error ↓ | Online actions | Residual actions |
| --- | ---: | ---: | ---: | ---: |
| Final repair | 0.007035 | 0.009833 | 0 | 55 |
| Fixed-streak dynamic gate | 0.005898 | 0.009133 | 94 | 0 |
| Event-triggered + repair | 0.006705 | 0.009629 | 30 | 46 |
| **Progressive dynamic gate** | **0.005652** | **0.009027** | 76 | 0 |
| Shuffled row topology + repair | 0.007035 | 0.009833 | 60 | 53 |

Shuffled topology vẫn thực hiện 60 online actions nhưng kết quả bằng final repair. Vì vậy gain không chỉ đến từ việc sửa nhiều row; row-edge assignment và thời điểm activation đúng mới có tác dụng. Progressive schedule cũng tốt hơn fixed streak dù dùng ít action hơn.

## 5. Một paired seed trên standard fidelity/utility

Bảng này giữ nguyên một unconditional sample và so trực tiếp DynamicGraph với endpoint repair.

| Method | Shape error ↓ | Trend error ↓ | Alpha ↑ | Beta ↑ | Detection ↑ | XGB AUC ↑ | Held-out error ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TabDiff | 1.6429 | 1.7127 | 98.3884 | 52.5806 | **0.9474** | 0.7482 | 0.01869 |
| Post-repair | 1.6434 | 1.7127 | 98.3884 | 52.5806 | 0.9493 | 0.7462 | 0.01496 |
| **DynamicGraph** | **1.6366** | **1.7083** | **98.4071** | **52.6066** | 0.9422 | **0.7530** | **0.01404** |

Trong run này DynamicGraph cải thiện đồng thời Shape, Trend, Alpha, Beta, AUC và held-out error so với TabDiff. Tuy nhiên đây mới là một seed, nên chỉ dùng làm supportive result; bảng ba seed ở mục 2–3 có giá trị cao hơn.

## 6. Published generator baselines trên cùng dataset

Đây là các local rerun dùng cùng repository split, 4.096 rows và cùng metric implementation. Chúng dùng output riêng của từng generator và hiện chỉ có một run, nên thích hợp để đặt bối cảnh hơn là làm paired significance test.

| Generator | Implementation | Shape error ↓ | Trend error ↓ | Alpha ↑ | Beta ↑ | Detection ↑ | XGB AUC ↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TabDiff | official backbone | 2.3851 | 2.5518 | 96.2240 | **53.3024** | 0.9561 | 0.7175 |
| **TabSyn** | official code | **1.4810** | **1.6499** | **99.2198** | 47.7816 | **0.9615** | 0.7127 |
| ForestDiffusion | official code | 8.9813 | 6.1003 | 94.4882 | 47.8939 | 0.9357 | 0.6968 |
| CTGAN | SDV implementation | 15.2233 | 17.8281 | 64.4922 | 21.3704 | 0.6355 | 0.6695 |
| TVAE | SDV implementation | 10.8799 | 6.5209 | 82.3844 | 25.9066 | 0.5474 | **0.7244** |

TabSyn là generator mạnh nhất về marginal/joint fidelity trong bảng này. Điều đó cho thấy backbone choice tạo chênh lệch Shape/Trend lớn hơn gain hiện tại của graph controller. Kết quả chưa chứng minh DynamicGraph đạt SOTA chung; claim phù hợp hơn là **một trajectory controller bổ sung cho diffusion backbone**, với gain rõ nhất ở conditional relation profile.

CTGAN và TVAE là published models nhưng chỉ nên để ở appendix vì quá cũ/yếu so với diffusion baselines. Main baseline table nên ưu tiên TabDiff, TabSyn, ForestDiffusion và bổ sung TabDDPM khi matched run hoàn tất. Các phương pháp constraint-guided chỉ được gọi là external baseline khi chạy official code/protocol; local Harpoon-style và CDD-style adaptations thuộc mechanism ablation.

## 7. Cách đọc metric

- **Shape error:** sai khác marginal distribution theo từng cột; thấp hơn tốt hơn.
- **Trend error:** sai khác dependence/joint trend; thấp hơn tốt hơn.
- **Alpha precision:** tỷ lệ synthetic samples nằm trong support của real data; cao hơn tốt hơn.
- **Beta recall:** mức synthetic data bao phủ modes của real data; cao hơn tốt hơn.
- **Detection:** quality score từ real-vs-synthetic detector; cao hơn tốt hơn trong evaluator hiện tại. Chỉ xem là metric bổ trợ.
- **XGBoost AUC:** train classifier trên synthetic, test trên real held-out; cao hơn tốt hơn.
- **Held-out relation error:** trung bình sai khác giữa conditional violation rate của synthetic và real held-out; thấp hơn tốt hơn. Đây không phải raw satisfaction và không mặc định coi mọi real violation là sai.

## 8. Kết luận hiện tại

1. Method có evidence tốt cho claim hẹp nhưng hợp lý: row- và timestep-dependent graph activation giúp tái tạo conditional relation profile tốt hơn TabDiff và các matched mechanism controls.
2. Progressive controller mới nhất cải thiện matched final-only repair ổn định `3/3` seeds trên held-out graph metrics.
3. General fidelity/utility phần lớn được giữ, nhưng gain nhỏ; AUC chưa ổn định và đôi khi giảm nhẹ.
4. TabSyn vẫn mạnh hơn đáng kể về Shape/Trend, nên chưa đủ bằng chứng để claim SOTA synthetic tabular data nói chung.
5. Trước bảng paper cuối cần rerun published baselines và proposed method trên cùng protocol. `Harpoon-style / final repair / frozen / shuffled` được tách sang ablation; không ghép graph 8-edge với progressive 7-edge.

## 9. Nguồn kết quả

- `outputs/harpoon_natural_common_quality_aggregate_v3_20261010/RESULTS.md`
- `outputs/progressive_graph_controller_aggregate_20261015/RESULTS.md`
- `outputs/local_baseline_suite_default_1062130_0/RESULTS.md`
- `outputs/matched_generator_baseline_reeval_default_1091649_0/RESULTS.md`
- `DEFAULT_PAPER_TABLES_20261015_VI.md`
