# Experiment log — dynamic constraint graph, 08-10-2026

## 1. Kết luận ngắn

Batch này hoàn tất toàn bộ Slurm array tasks/aggregate jobs liên quan đến Diabetes controls, bootstrap, adaptive refresh và graph-aware action selection. Tất cả jobs mới kết thúc `ExitCode 0:0`; không có traceback, OOM hoặc dependency failure.

Kết luận chính:

1. Dynamic middle-step topology có effect lặp lại trên Shoppers và Diabetes khi so uncapped với frozen-state; paired hierarchical bootstrap không cắt 0 trên hai dataset này.
2. Effect không action-efficient một cách phổ quát. Khi cap action, dynamic vẫn tốt trên Shoppers nhưng thua frozen trên Default và Diabetes.
3. Middle tốt hơn early, nhưng late có absolute validity tốt nhất trên Diabetes. Claim hợp lệ là middle là vùng sớm nhất đủ đọc và còn thời gian để trajectory phản ứng, không phải middle luôn tối ưu.
4. Final repair vẫn là baseline rất mạnh; trên Diabetes nó tốt hơn mid-step guidance về validity và overall held-out error.
5. Volatility-adaptive hysteresis theo flip-rate bị bác bỏ. Topology flip cao không đồng nghĩa preview noise; nhiều flip là state transition thật.
6. Dùng score-only graph chỉ để rerank action hiện tại không thay đổi bất kỳ decision hay metric nào; candidate action set cần phong phú hơn trước khi graph-wide reward có giá trị.
7. Kết quả hiện tại chưa phải state of the art và chưa đủ để claim vượt Harpoon/CTDF/LCATS trên common task.

## 2. Dataset thứ ba: Diabetes

Graph discovery dùng 19 API calls và 199,622 tokens. Unified graph có 26 edges; catalog cuối có 22 edges gồm 6 `actuate`, 15 `score_only`, 1 `monitor`.

### 2.1 Dynamic so với frozen

| Window | Topology | Actuation heldout error ↓ | Overall heldout error ↓ | KS ↓ | TV ↓ | Actions |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Full | Dynamic current | **0.005911** | **0.012383** | 0.020365 | 0.019937 | 24.0 |
| Full | Dynamic stable | 0.006222 | 0.012528 | 0.020374 | 0.019905 | 18.3 |
| Full | Frozen state | 0.006964 | 0.012844 | 0.020365 | **0.019879** | 11.7 |
| Middle | **Dynamic** | **0.001077** | **0.011112** | **0.020038** | 0.020092 | 70.7 |
| Middle | Frozen state | 0.003543 | 0.012311 | 0.020056 | **0.020014** | 28.3 |

Uncapped dynamic-middle thắng frozen 3/3 seed, nhưng dùng nhiều hơn khoảng 42 actions.

### 2.2 Action budget, timing và smoothing

| Control | Variant | Actuation error ↓ | Overall error ↓ | Actions |
| --- | --- | ---: | ---: | ---: |
| Cap 20 | Dynamic | 0.005213 | 0.015524 | 20.0 |
| Cap 20 | **Frozen state** | **0.004740** | **0.015294** | 19.3 |
| Timing cap 20 | Early | 0.006152 | 0.015892 | 20.0 |
| Timing cap 20 | Middle | 0.005213 | 0.015524 | 20.0 |
| Timing cap 20 | **Late** | **0.003478** | **0.014544** | 20.0 |
| Uncapped | Hysteresis 2 | 0.000742 | 0.012851 | 69.3 |
| Uncapped | Hysteresis 3 | **0.000622** | **0.012803** | 60.0 |

Shuffled topology chỉ tạo trung bình 0.7 actions nên không phải matched-budget control hợp lệ trên Diabetes.

## 3. Paired bootstrap cho dynamic-middle

Positive improvement là `frozen error − dynamic error`. Conditional bootstrap resample synthetic rows; hierarchical bootstrap resample cả rows và ba seeds.

| Dataset | Improvement ↑ | Seed wins | Hierarchical 95% CI | P(improvement > 0) |
| --- | ---: | ---: | ---: | ---: |
| Shoppers | **0.001519** | 3/3 | **[0.000994, 0.002057]** | 1.000 |
| Default | 0.001963 | 2/3 | [-0.003091, 0.007210] | 0.769 |
| Diabetes | **0.002466** | 3/3 | **[0.001270, 0.003956]** | 1.000 |

Bootstrap hỗ trợ effect trên Shoppers và Diabetes, không hỗ trợ kết luận chắc trên Default. Đây vẫn chỉ là sampling-seed uncertainty; chưa có nhiều independent training/checkpoint seeds.

## 4. Same-node refresh ablation

Cross-job comparison được phát hiện có GPU-architecture confound. Confirmation batch chạy `dynamic`, `hysteresis-3` và `adaptive` tuần tự trong cùng task/GPU. SHA-256 của unconditional samples khớp 9/9 dataset-seed groups.

| Dataset | Policy | Actuation error ↓ | Overall error ↓ | Actions | Flip rate |
| --- | --- | ---: | ---: | ---: | ---: |
| Shoppers | **Dynamic** | **0.001783** | **0.021300** | 74.7 | 0.000921 |
| Shoppers | Hysteresis 3 | 0.002216 | 0.021516 | 63.3 | 0.001054 |
| Shoppers | Adaptive | **0.001783** | **0.021300** | 74.7 | 0.001222 |
| Default | Dynamic | 0.008438 | 0.019393 | 179.3 | 0.002146 |
| Default | **Hysteresis 3** | **0.006127** | **0.018116** | 160.7 | 0.001647 |
| Default | Adaptive | 0.007431 | 0.018524 | 163.7 | 0.001857 |
| Diabetes | **Dynamic** | **0.001100** | **0.013474** | 83.3 | 0.009240 |
| Diabetes | Hysteresis 3 | 0.001220 | 0.013511 | 67.0 | 0.007378 |
| Diabetes | Adaptive | 0.001220 | 0.013511 | 67.0 | 0.007378 |

Adaptive controller chọn persistence 3 ở khoảng 4.2% checkpoints trên Shoppers, 35.0% trên Default và 91.7% trên Diabetes. Nó không vượt policy cố định tốt nhất ở dataset nào. Diabetes có flip rate cao nhất nhưng dynamic tốt nhất, trực tiếp bác bỏ flip-rate heuristic.

## 5. Score-only graph guardrail

Mỗi Diabetes actuation edge chia sẻ cột với 12–13 score-only edges. Guardrail phạt action tạo violation mới trên score-only graph nhưng không sửa pre-existing score-only violations.

| Graph weight | Actuation error ↓ | Score-only error ↓ | Overall error ↓ | KS ↓ | TV ↓ | Actions |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.001541 | 0.014146 | 0.010198 | 0.019838 | 0.019638 | 74.0 |
| 2 | 0.001541 | 0.014146 | 0.010198 | 0.019838 | 0.019638 | 74.0 |

Parity đạt 3/3. Action choices cũng giống hệt: tổng mỗi seed chỉ thay đổi theo seed, còn weight 0/2 đều chọn cùng số `direct_requirements` và `deactivate_0`. Kết luận: score-only graph không phân biệt được candidate set hiện tại; không tiếp tục sweep weight.

## 6. Hypothesis ledger

| Hypothesis | Trạng thái | Bằng chứng |
| --- | --- | --- |
| Predicted-clean state đọc được theo timestep | Ủng hộ | Precision/recall tăng từ early đến late; early yếu, middle đã đủ dùng |
| Dynamic topology có thể tốt hơn frozen | Ủng hộ có điều kiện | Shoppers và Diabetes uncapped; bootstrap dương |
| Dynamic topology action-efficient phổ quát | Bác bỏ | Matched budget dương trên Shoppers, âm trên Default/Diabetes |
| Middle luôn là timing tốt nhất | Bác bỏ | Diabetes: late > middle > early ở cùng 20 actions |
| Middle là vùng sớm nhất hữu ích | Ủng hộ | Middle tốt hơn early trên Shoppers, Default và Diabetes |
| Flip-rate đủ để chọn refresh policy | Bác bỏ | Diabetes flip cao nhất nhưng dynamic vẫn tốt hơn hysteresis |
| EMA adaptive hysteresis vượt fixed policies | Bác bỏ | Không thắng best fixed policy trên 3/3 datasets |
| Score-only graph reranking cải thiện action | Bác bỏ với candidate set hiện tại | Không đổi action/metric ở 3/3 seeds |
| Row-specific assignment quan trọng | Ủng hộ trên Shoppers | Correct-row giảm actuation error khoảng 23.5% ở exact 40-action budget |
| Mid-step guidance vượt post-repair | Chưa được ủng hộ chung | Final repair thắng Diabetes và nhiều Shoppers settings; Default có trường hợp dynamic tốt hơn |
| LLM external knowledge cần thiết | Chưa chứng minh | Diabetes retained LLM graph chủ yếu dựa schema semantics; external/hybrid retained count bằng 0 trong source audit |

## 7. Baseline status

| Baseline | Trạng thái | Kết quả/giới hạn |
| --- | --- | --- |
| TabDiff | Official checkpoint/sampler, local runs | Common backbone và exact-noise baseline |
| LCATS + TabSyn | Official code, local rerun | Insurance violation `0.0822 → 0.0017`; cell edit `0.2884`; khác common task |
| CuTS | Official code, local rerun | German implication/line constraints đạt satisfaction 1.0; khác common task |
| Harpoon | Author artifact available | Local official setup lỗi CUDA/PyTorch environment; artifact numbers không trộn vào local table |
| Harpoon-style | Local proxy | Không được gọi là official reproduction |
| CTDF-style | Local numeric proxy | Thiếu full mixed-type Euclidean/KL feasibility maps |
| Final repair | Same compiler/local | Baseline rất mạnh, thường đạt exact actuation validity |

## 8. Claim có thể bảo vệ và claim chưa thể dùng

Claim có thể bảo vệ ở mức exploratory:

- Row- and trajectory-dependent graph topology có thể cải thiện semantic calibration so với topology đóng băng mà không làm đổi đáng kể KS/TV.
- Lợi ích xuất hiện rõ nhất khi predicted-clean row đã đủ đọc nhưng chưa hội tụ; early feedback không đáng tin.
- Refresh policy phải phân biệt state transition thật với preview noise; update mọi step và hysteresis cố định đều không tốt trên mọi dataset.

Claim chưa thể dùng:

- state of the art;
- dynamic luôn tốt hơn frozen/final repair;
- middle luôn là timestep tốt nhất;
- LLM cần thiết ở inference time;
- external knowledge graph tạo gain;
- vượt faithful Harpoon/CTDF/LCATS trên common protocol.

## 9. Hướng tiếp theo có giá trị thông tin cao

1. Thiết kế transition certifier: dự đoán một topology change là genuine hay noise bằng agreement/lookahead/counterfactual survival, học hoặc calibrate chỉ trên train/validation trajectory.
2. Mở rộng action candidate generator. Graph-wide scoring chỉ hữu ích khi có nhiều semantic-equivalent corrections thực sự khác nhau; candidate set hiện quá nghèo.
3. Port ít nhất một faithful inference-time baseline trên cùng backbone/task: Harpoon hoặc CTDF. Giữ proxy tách riêng.
4. Chạy common-task LCATS/final repair cùng compiler và action budget; validity đơn thuần không đủ.
5. Xác nhận trên second backbone hoặc second training checkpoint trước khi viết claim robustness.
6. Đánh giá graph discovery bằng source ablation thật: schema/statistics/external knowledge, nhưng chỉ sau khi external edges vượt semantic validation và được phép tác động sampler.

Không nên tiếp tục tune EMA threshold hoặc graph weight trên ba dataset hiện tại; hai hướng đó đã có falsification đủ rõ.

## 10. Artifacts và logs

- Main updated report: `ICDE_SUBMISSION_SPRINT_20261007.md`
- Diabetes aggregate: `outputs/next_batch_aggregate_1060463/RESULTS.md`
- Diabetes controls: `outputs/diabetes_controls_aggregate_1060481/RESULTS.md`
- Paired bootstrap: `outputs/bootstrap_dynamic_topology_1060488/RESULTS.md`
- Same-node refresh confirmation: `outputs/adaptive_paired_aggregate_1060655/RESULTS.md`
- Graph guardrail: `outputs/graph_guardrail_aggregate_1060665/RESULTS.md`
- LCATS official audit: `outputs/lcats_official_output_audit_1052790/RESULTS.md`
- CuTS official German audit: `outputs/cuts_official_german_audit_1052794/RESULTS.md`
- Slurm stdout/stderr: `logs/llmtd-*1060451*`, `logs/llmtd-*1060452*`, `logs/llmtd-*1060453*`, `logs/llmtd-*1060464*`, `logs/llmtd-*1060478*`, `logs/llmtd-*1060479*`, `logs/llmtd-*1060480*`, `logs/llmtd-*1060488*`, `logs/llmtd-*1060654*`, `logs/llmtd-*1060664*`.

Các sampling jobs trong batch 1060478–1060665 không gọi external API; API cost mới bằng 0.

## 11. Transition survival audit và online confirmation

Jobs `1060882[0-8] → 1060883` đo transition survival trên Shoppers, Default và Diabetes với calibration/evaluation seed split. Chờ 5 observations tăng evaluation precision so với immediate update:

| Dataset/state | Immediate precision | Wait-5 precision | Recall | Precision gain |
| --- | ---: | ---: | ---: | ---: |
| Default applicability | 0.7006 | 0.8313 | 0.9140 | +0.1307 |
| Default violation | 0.5574 | 0.7156 | 0.9007 | +0.1582 |
| Diabetes applicability | 0.6966 | 0.8247 | 0.8951 | +0.1281 |
| Diabetes violation | 0.5184 | 0.7904 | 0.9230 | +0.2720 |
| Shoppers applicability | 0.9655 | 0.9776 | 0.9955 | +0.0121 |
| Shoppers violation | 0.6005 | 0.9078 | 0.8980 | +0.3073 |

Policy selection chọn immediate applicability cho Shoppers vì nó đã vượt precision 0.95; các state còn lại chọn wait-5 dù chưa đạt 0.95. Kết quả support temporal survival như một certifier feature, nhưng không support global persistence: applicability Shoppers không cần trì hoãn.

Batch online `1060901[0-8] → 1060902` đã được submit trên ba seeds mới. Mỗi task chạy tuần tự immediate, wait-3, dynamic+violation-wait-5, topology+violation-wait-5 và frozen+wait-5, sau đó kiểm tra unconditional SHA-256 parity. Không có API call mới.
