# Cập nhật method, end-to-end training và mức độ tiệm cận SOTA

Ngày cập nhật: 2026-10-10  
Dataset chính trong đợt này: **Default of Credit Card Clients**  
Backbone: **TabDiff**, checkpoint epoch 8000  
Đơn vị xác nhận chính: 4096 synthetic rows × 3 paired sampling seeds

## 1. Kết luận ngắn

Kết quả tốt nhất hiện tại **không phải** checkpoint được fine-tune bằng graph loss. Cấu hình mạnh và ổn định nhất là:

1. TabDiff được continued-training 150 epochs với learning rate nhỏ, dùng như control không có graph loss;
2. dynamic constraint graph được đánh giá lại theo từng row và từng reverse step;
3. chỉ can thiệp trong middle trajectory, `t∈[0.10,0.50]`, với stability gate `annealed8`;
4. target của mỗi soft edge là median của ba train-only partitions: discovery, counterexample và calibration;
5. action được chọn bằng component beam search, giữ nguyên applicability columns và giới hạn ba action cho mỗi row-edge.

Trên ba paired seeds, cấu hình này đạt:

| Cấu hình | Held-out error ↓ | Numeric KS ↓ | Categorical TV ↓ | Actions | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: |
| TabDiff, không guidance | 0.0309 ± 0.0084 | 0.0304 | 0.0175 | 0 | 0 |
| Pretrained + dynamic middle, split-median | 0.0047 ± 0.0010 | 0.0304 | 0.0188 | 112.7 | 9.32 s |
| **Continued-training + dynamic middle, split-median** | **0.0039 ± 0.0006** | 0.0308 | 0.0188 | 112.3 | 9.38 s |
| Continued-training + final repair, split-median | 0.0055 ± 0.0004 | 0.0309 | 0.0184 | 88.3 | 0 s |
| Harpoon-style hard proxy | 0.0179 | 0.0305 | 0.0190 | — | — |

Gain của cấu hình tốt nhất:

- giảm **87.5%** held-out error so với TabDiff không guidance;
- giảm **17.5%** so với pretrained dynamic graph đã được retune;
- giảm **29.6%** so với final repair dùng cùng continued-training checkpoint;
- giảm khoảng **78%** so với Harpoon-style hard proxy đã chạy trên cùng TabDiff backbone và graph.

Đây là kết quả nội bộ tốt nhất trên benchmark hiện tại. Tuy nhiên, **chưa đủ để tuyên bố SOTA ngoài bài**, vì benchmark chính mới có một dataset và ba soft edges liên thông; Harpoon proxy là bản port cùng backbone, không phải official reproduction; CTDF và Harpoon chính thức giải bài toán constraint khác.

## 2. Bài toán hiện tại

Ta không còn mô tả bài toán là “LLM sửa rare subgroup”. Bài toán tổng quát hơn là:

> Cho một tập semantic relations đã được compile và kiểm chứng trên train, tại mỗi reverse step của tabular diffusion, hãy xác định graph nào đang active cho từng predicted-clean row, và chọn can thiệp tối thiểu để cải thiện semantic distribution mà không làm hỏng fidelity, diversity và utility.

Graph không chỉ thay đổi theo row. Nó thay đổi dọc trajectory vì:

- applicability predicate có thể chưa ổn định ở early steps;
- categorical parent có thể flip và kích hoạt hoặc vô hiệu hóa downstream edges;
- một edge chỉ được actuation khi violation state đủ bền;
- soft edge có target rate và độ bất định, không phải mọi active row đều phải được “sửa”.

Trong Default, graph thực nghiệm đang dùng là một chain ba cạnh:

```text
BILL_AMT3 == 0
    └── usually PAY_4 == -2
          └── usually PAY_5 == -2
                └── usually PAY_6 == -2
```

Đây là soft temporal/distributional graph. Violation hợp lệ phải được giữ lại ở đúng tỷ lệ; zero violation không phải mục tiêu.

## 3. Pipeline method hiện tại

### 3.1 Graph discovery và verification

Pipeline graph có ba nguồn ứng viên:

- external/domain knowledge từ LLM;
- train-only statistical discovery;
- predefined rules hoặc schema semantics.

Mỗi candidate được compile về typed predicates, sau đó đi qua:

1. type checking và executable validation;
2. discovery partition;
3. counterexample partition;
4. calibration partition;
5. semantic judge và role assignment;
6. frozen held-out audit chỉ dùng để báo cáo.

Mỗi edge có `classification`, `usage_role`, support, violation rate, directionality và provenance. Target mới cho soft edge không lấy một calibration split duy nhất mà lấy:

\[
r_e^{robust}=\operatorname{median}
\{r_e^{discovery},r_e^{counterexample},r_e^{calibration}\}.
\]

Việc này giải quyết một failure mode đã quan sát: cạnh `PAY_5→PAY_6` có calibration rate 0.083 nhưng held-out rate 0.106. Ép đúng 0.083 làm model under-generate relation này.

### 3.2 Dynamic active graph trong trajectory

Tại reverse step `t`, denoiser tạo predicted-clean state `x̂₀,t`. Với row `i`, active graph là:

\[
G_i(t)=\{e: Applicable_e(\hat{x}_{0,t}^{(i)})
\land Stable_e(i,t)\land Authorized_e\}.
\]

`Stable_e` dùng streak gate, hiện là `annealed8`: early/mid state cần violation xuất hiện bền hơn trước khi actuator được phép can thiệp. Điều này tránh phản ứng với noisy preview.

### 3.3 Action layer

Action layer hiện không chọn một sửa đổi tùy ý. Nó:

1. khóa applicability columns để không “giải” implication bằng cách xóa antecedent;
2. tìm action candidates trên connected active component;
3. chấm điểm bằng residual violation, edit cost và train-density;
4. dùng component beam search width 8;
5. giới hạn action budget và cooldown;
6. chỉ actuation phần excess violation so với target rate.

Với soft graph, split-median target cho kết quả tốt hơn point target và Wilson target. Đây là thay đổi lớn nhất trong đợt thực nghiệm này.

## 4. End-to-end objective đã implement

Script mới hỗ trợ graph mixed-type:

`scripts/finetune_mixed_graph_calibrated_tabdiff.py`

Khác GraphFT cũ, objective mới không chỉ dùng các real rows đã hợp lệ và không giới hạn ở numerical constraints. Nó dùng toàn bộ real training rows, bao gồm observed violations/exceptions.

Với mỗi edge, tại middle-noise state, model dự đoán:

- soft applicability `a_e(x̂₀,t)`;
- soft satisfaction `s_e(x̂₀,t)`;
- conditional violation rate `Σa_e(1-s_e)/Σa_e`.

Loss tổng quát:

\[
L=L_{TabDiff}
+\lambda_a L_{applicability}
+\lambda_s L_{satisfaction}
+\lambda_r L_{rate}
+\lambda_c L_{trajectory-consistency}.
\]

Trong các run chính:

- `λa=0.1`, `λs=0.1`, `λr=2`, `λc=0`;
- time window `[0.10,0.50]`;
- full denoiser hoặc decoder/head-only;
- noise schedules frozen;
- 150 epochs, learning rate `2e-6` cho full denoiser;
- checkpoint vẫn giữ format TabDiff chuẩn nên sampler cũ load trực tiếp được.

Objective này có hai target variants:

- `point`: rate của calibration partition;
- `split_median`: median của ba train-only partitions.

## 5. Kết quả chi tiết

### 5.1 Baseline matched trước đợt end-to-end

Ba paired seeds, cùng TabDiff backbone và cùng graph:

| Method | Held-out error ↓ | Calibration error ↓ | Shape ↓ | Trend ↓ | Alpha ↑ | Beta ↑ | Detection ↑ | AUC ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TabDiff | 0.0309 | 0.0341 | 1.9833 | 2.2165 | 98.4736 | 52.5109 | 0.9528 | 0.7230 |
| Dynamic graph, Wilson | 0.0120 | 0.0091 | 1.9591 | 2.2116 | 98.5218 | 52.4436 | 0.9590 | 0.7228 |
| Frozen graph | 0.0154 | — | 1.9600 | 2.2119 | 98.4990 | 52.4533 | 0.9587 | 0.7238 |
| Final repair | 0.0161 | — | 1.9556 | 2.1909 | 98.5003 | 52.4761 | 0.9546 | 0.7278 |
| Harpoon-style hard proxy | 0.0179 | 0.0121 | 1.9772 | 2.2317 | 98.5284 | 52.4398 | 0.9570 | 0.7278 |
| Harpoon-style calibrated proxy | 0.0272 | 0.0296 | 1.9742 | 2.2016 | 98.4898 | 52.5054 | 0.9581 | 0.7253 |

Dynamic graph đã preserve common metrics; gain không đến từ hy sinh fidelity tổng quát.

### 5.2 End-to-end training với point target

Kết quả một seed ban đầu trông rất mạnh:

- graph-trained + point dynamic: held-out error 0.0081;
- pretrained + point dynamic: 0.0091;
- continued-training + point dynamic: 0.0100;
- same-checkpoint final repair: 0.0096.

Nhưng ba-seed confirmation đã bác bỏ kết luận đó:

| Checkpoint | Mechanism | Held-out error ↓ | Point-calibration error ↓ | Actions | Runtime |
| --- | --- | ---: | ---: | ---: | ---: |
| Pretrained | dynamic middle | **0.0094 ± 0.0014** | 0.0042 | 127.3 | 10.02 s |
| Continued-training control | dynamic middle | 0.0097 ± 0.0007 | 0.0052 | 124.7 | 9.98 s |
| Graph-trained | dynamic middle | 0.0098 ± 0.0018 | **0.0019** | **38.3** | **5.84 s** |
| Graph-trained | final repair | 0.0103 ± 0.0011 | 0.0009 | 37.7 | 0 s |

Kết luận đúng:

- graph training làm point-calibration tốt hơn và giảm khoảng 70% actions;
- graph training không cải thiện held-out accuracy trung bình;
- single-seed gain không được dùng làm claim.

### 5.3 Split-median target: kết quả mạnh nhất

Ba-seed confirmation:

| Checkpoint | Mechanism | Held-out error ↓ | Split-median error ↓ | Actions | Runtime |
| --- | --- | ---: | ---: | ---: | ---: |
| Pretrained | dynamic middle | 0.0047 ± 0.0010 | 0.0040 | 112.7 | 9.32 s |
| Pretrained | final repair | 0.0055 ± 0.0003 | **0.0005** | 88.0 | 0 s |
| **Continued-training control** | **dynamic middle** | **0.0039 ± 0.0006** | 0.0053 | 112.3 | 9.38 s |
| Continued-training control | final repair | 0.0055 ± 0.0004 | **0.0004** | 88.3 | 0 s |
| Point-graph-trained checkpoint | dynamic middle | 0.0088 ± 0.0042 | 0.0052 | **32.3** | **5.54 s** |

Hai cách đọc cùng đúng và cần báo cáo đồng thời:

1. dynamic middle có held-out error tốt hơn final repair khoảng 29.6%; đây là evidence tốt cho can thiệp mềm trong trajectory;
2. final repair khớp split-median target chặt hơn. Dynamic method giữ residual violations và tình cờ/hoặc nhờ regularization gần held-out distribution hơn. Vì vậy chưa được claim rằng dynamic method “calibrate train target tốt hơn”.

### 5.4 Robust graph training

Sau khi đổi training target sang split-median, checkpoint robust đạt:

| Mechanism | Held-out error ↓ | Point-calibration error ↓ | Actions | Runtime |
| --- | ---: | ---: | ---: | ---: |
| Unconditional | 0.0200 ± 0.0055 | 0.0216 | 0 | 0 |
| Dynamic middle | 0.0053 ± 0.0014 | 0.0064 | 39.3 | 5.96 s |
| Dynamic middle + residual final repair | **0.0045 ± 0.0005** | 0.0061 | 41.3 | 5.96 s |
| Final repair | 0.0055 ± 0.0004 | 0.0061 | 38.3 | 0 |

Robust graph training giúp giảm actions và variance, nhưng pure dynamic vẫn không thắng continued-training dynamic 0.0039. Hybrid đạt 0.0045, tốt hơn final-only của chính checkpoint nhưng vẫn chưa là cấu hình accuracy tốt nhất.

### 5.5 Trajectory readability sau fine-tuning

Earliest readable cutoff, `t` càng lớn nghĩa là đọc được càng sớm:

| Edge | Continued-training | Point-graph-trained | Kết luận |
| --- | ---: | ---: | --- |
| `BILL_AMT3→PAY_4` | 0.265 | 0.245 | xấu hơn nhẹ |
| `PAY_4→PAY_5` | 0.531 | 0.531 | không đổi |
| `PAY_5→PAY_6` | 0.469 | 0.327 | xấu hơn rõ |

Vì vậy giả thuyết “graph loss làm các edge readable sớm hơn” hiện **bị bác bỏ**.

### 5.6 Common metrics của cấu hình mạnh

Single paired seed, cùng evaluator:

| Method | Shape ↓ | Trend ↓ | Alpha ↑ | Beta ↑ | Detection ↑ | AUC ↑ | DCR, 50 ideal | Held-out error ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TabDiff | 2.4058 | 2.5072 | 98.6638 | 52.5049 | 0.9322 | 0.7168 | 51.37 | 0.0407 |
| Pretrained + split-median dynamic | 2.3075 | 2.3816 | **98.8087** | 52.3926 | **0.9524** | 0.7209 | 51.44 | 0.0046 |
| **Continued + split-median dynamic** | 2.3268 | 2.3959 | 98.7181 | 52.3796 | 0.9416 | **0.7296** | 51.56 | **0.0043** |
| Continued + final repair | 2.3427 | 2.4168 | 98.7034 | 52.3942 | 0.9420 | 0.7183 | 51.49 | 0.0054 |
| Robust graph-trained dynamic | **2.3016** | **2.3554** | 98.6530 | **52.5765** | 0.9289 | 0.7103 | **51.00** | 0.0045 |

Kết quả không cho thấy quality collapse. Cấu hình accuracy tốt nhất cải thiện Shape, Trend, Alpha, Detection và downstream AUC so với TabDiff trong seed này; Beta giảm nhẹ 0.125 điểm.

## 6. Các hypothesis được giữ hoặc bác bỏ

| Hypothesis | Trạng thái | Bằng chứng |
| --- | --- | --- |
| Dynamic graph ở middle steps tốt hơn frozen graph | Ủng hộ | Wilson run: held-out 0.0120 so với frozen 0.0154, ba seeds |
| Middle-step intervention tốt hơn final repair | Ủng hộ trong split-median setting | 0.0039 so với 0.0055, gain 29.6% |
| Chỉ tăng guidance strength sẽ tạo breakthrough | Bác bỏ | Harpoon sweep over-enforce sau weight 8 và held-out xấu đi |
| Point calibration của một split là đủ | Bác bỏ | `PAY_5→PAY_6` shift 0.083→0.106; split-median giảm held-out error mạnh |
| Full graph fine-tuning tăng held-out accuracy | Chưa ủng hộ | point-trained 0.0098, pretrained 0.0094; robust-trained 0.0053, continued control 0.0039 |
| Full graph fine-tuning làm edge readable sớm hơn | Bác bỏ | hai trên ba edge cutoff xấu hơn |
| Full graph fine-tuning giảm intervention cost | Ủng hộ | actions giảm từ khoảng 127 xuống 38; runtime giảm khoảng 42% |
| Graph guidance giữ common fidelity | Ủng hộ sơ bộ | KS/TV gần như không đổi; Shape/Trend không xấu đi |
| Single-seed improvement đủ chọn method | Bác bỏ | gain +11% ở seed đầu chuyển thành -3.7% trung bình ba seeds |

## 7. Đánh giá mức độ SOTA và khả năng thành paper

### 7.1 Điều có thể claim ngay trong nội bộ

- Trên frozen three-edge Default graph, robust train-only split calibration + dynamic middle-step activation là cấu hình có held-out semantic error thấp nhất.
- Nó thắng matched final repair khoảng 30% và matched Harpoon-style proxy khoảng 78% trên metric semantic chính.
- Common fidelity và downstream AUC được giữ hoặc cải thiện trong current evaluation.
- Dynamic method có lợi thế rõ so với exact post-repair khi train constraint estimate có sampling uncertainty/partition shift.

### 7.2 Điều chưa thể claim

- Chưa thể nói SOTA tabular generation nói chung.
- Chưa thể nói thắng official Harpoon hoặc CTDF trên protocol của họ.
- Chưa thể nói graph training tốt hơn pretrained backbone về accuracy.
- Chưa thể nói LLM là nguyên nhân tạo gain; graph hiện tại chủ yếu từ statistical discovery + semantic verification.
- Chưa thể nói method generalize vì kết quả chính mới trên Default và ba edges.

### 7.3 Novelty boundary

- Harpoon đã có differentiable per-step guidance; official code hiện có tại <https://github.com/adis98/Harpoon>.
- CTDF đã có per-step mixed-type feasibility operator và zero hard-constraint violations.
- LCATS/Zhao đã có constraint discovery, validation và post-generation repair.
- SCTab-style work đã condition denoiser bằng static prior/current row context.

Do đó novelty không nên là “constraint vào mỗi reverse step”. Claim phòng thủ hơn là:

> Một verified soft constraint graph có topology thay đổi theo predicted-clean row và trajectory state, kết hợp train-only robust rate calibration và selective component-level actuation, giúp generalize semantic distributions tốt hơn frozen guidance và exact repair trong khi giữ common tabular quality.

Đây là project khả thi cho ICDE nếu mở rộng benchmark và official baselines trong phần tiếp theo.

## 8. Architecture nên phát triển nếu tiếp tục end-to-end

Không nên tiếp tục full-denoiser fine-tuning với graph loss hiện tại như main method. Nó giảm action cost nhưng không thắng accuracy control. Architecture hợp lý hơn là **Graph Residual Adapter** đặt sau predicted-clean head và trước reverse update.

### 8.1 Graph Residual Adapter

1. TabDiff backbone tạo numerical `x̂₀` và categorical logits.
2. Mỗi column trở thành một node state nhỏ:
   - numerical node: projected scalar + uncertainty;
   - categorical node: projected probability vector;
   - thêm time embedding và column semantic embedding.
3. Mỗi compiled edge tính soft applicability, exception probability và reliability.
4. Hai graph-message-passing layers truyền message chỉ trên active edges.
5. Adapter sinh residual cho numerical prediction hoặc categorical logits.
6. Time gate làm residual bằng 0 ở early-noisy states và gần late endpoint; authority tập trung ở middle trajectory.
7. Null graph phải cho output đúng bằng frozen TabDiff.

### 8.2 Training schedule đề xuất

Giai đoạn 1: freeze backbone, train adapter bằng:

\[
L=L_{row-state}+L_{robust-rate}+L_{trajectory-consistency}
+\lambda_{id}KL(p_{adapter}\|p_{base})_{inactive}.
\]

Giai đoạn 2: unfreeze decoder cuối với learning rate nhỏ hơn adapter 10–20 lần.

Graph dropout và shuffled-edge negatives là bắt buộc để model không học thuộc ba relations. Nếu chỉ train một graph ba cạnh, architecture mới không đủ general để thành contribution.

### 8.3 Kill criteria trước khi đầu tư lớn

Chỉ tiếp tục adapter nếu pilot trên ít nhất hai graph/datasets đạt đồng thời:

- held-out semantic error giảm ít nhất 10% so với continued-training + dynamic sampler;
- không xấu hơn 0.1 điểm ở Alpha/Beta và không xấu hơn 0.01 AUC;
- action count giảm ít nhất 30%;
- thắng final repair trên ít nhất 2/3 seeds;
- edge readability không muộn hơn continued-training control.

Nếu không đạt, end-to-end training nên là ablation âm; paper giữ pretrained/continued TabDiff + dynamic sampler.

## 9. Thứ tự thực nghiệm tiếp theo

1. **Mở rộng benchmark trước architecture:** thêm 2–3 connected soft subgraphs trên Default, sau đó một dataset thứ hai; freeze graph trước confirmation.
2. **Official/matched baselines:** official Harpoon trên protocol gốc; matched-backbone Harpoon proxy trên graph mới; final repair; frozen topology; CTDF-style projection cho hard subset.
3. **Uncertainty-aware target:** target interval hoặc posterior thay vì một scalar median; report cả held-out error và distance tới train uncertainty set.
4. **Graph Residual Adapter pilot:** frozen backbone, graph dropout, two datasets; không full joint training ngay.
5. **Common metrics ba seeds cho best configuration:** Shape, Trend, Alpha, Beta, Detection, MLE, DCR, runtime và action count.
6. **LLM role ablation:** statistical-only graph, LLM-only graph, hybrid verified graph; gain của sampler phải tách khỏi gain của discovery.

## 10. Reproducibility và artefacts

### Code mới

- `analysis/llm_feedback_20260922/scripts/finetune_mixed_graph_calibrated_tabdiff.py`
- `analysis/llm_feedback_20260922/scripts/aggregate_mixed_graph_end_to_end.py`
- Slurm files `473`–`483` trong `analysis/llm_feedback_20260922/slurm/`

### Kết quả chính

- Point confirmation: `outputs/mixed_graph_confirmation_20261010/`
- Split-median confirmation: `outputs/mixed_graph_splitmedian_confirmation_v2_20261010/`
- End-to-end training runs: `outputs/mixed_graph_finetune_rerun_*_1067447_*/`
- Robust graph training: `outputs/mixed_graph_robust_finetune_1067475/`
- Robust confirmation: `outputs/mixed_graph_robust_confirm_s*_1067485_*/`
- Trajectory recalibration: `outputs/mixed_graph_trajectory_*_1067457_*/`
- Common metrics, point setting: `outputs/mixed_graph_point_quality_1067465/`
- Common metrics, best split-median setting: `outputs/best_splitmedian_common_quality_1067489/`
- Matched baseline table: `outputs/harpoon_natural_common_quality_aggregate_v3_20261010/`

### Slurm jobs quan trọng

| Job | Nội dung | Trạng thái |
| --- | --- | --- |
| 1067300 | mixed-type objective smoke | completed |
| 1067447 | point end-to-end training + controls | completed |
| 1067452 | point checkpoint evaluation | completed |
| 1067457 | trajectory readability recalibration | completed |
| 1067458 | point/Wilson policy sweep | completed |
| 1067466 | point three-seed confirmation | completed |
| 1067475 | robust split-median end-to-end training | completed |
| 1067476 | split-median three-seed checkpoint comparison | completed |
| 1067485 | robust graph-trained three-seed confirmation | completed |
| 1067489 | common metrics for best split-median methods | completed |

## 11. Quyết định hiện tại

Nên đóng khung paper quanh **dynamic active graph in middle denoising + uncertainty-aware soft-edge calibration**, không quanh full end-to-end graph training.

End-to-end objective mới là một ablation có giá trị:

- chứng minh denoiser có thể hấp thụ graph và giảm action cost;
- đồng thời chỉ ra full fine-tuning dễ overfit rate của một split và không bảo đảm edge readability;
- từ đó biện minh cho frozen backbone hoặc residual adapter có identity regularization.

Kết quả đã đủ để bắt đầu viết Problem, Method và Experimental Protocol. Chưa nên khóa Abstract/SOTA claim cho đến khi có ít nhất một graph lớn hơn và một dataset thứ hai với cùng pattern: dynamic middle thắng exact repair mà common quality không giảm.

## 12. Mở rộng đang chạy: graph-rate controller

Phân tích ba seed của cấu hình tốt nhất cho thấy sai số không phân bố đều:

- `emp_024` và `emp_005` đã gần held-out rate, thường chỉ sai khoảng 0.002–0.004;
- `emp_012` còn sai khoảng 0.005–0.010 và chi phối phần lớn macro held-out error;
- nguyên nhân cấu trúc là action trên cạnh trước có thể đổi giá trị parent của cạnh sau, từ đó làm cạnh sau active/deactive và đổi cả tử số lẫn mẫu số của conditional violation rate.

Phiên bản cũ calibrate từng cạnh riêng nên chưa mô hình hóa trực tiếp tác động liên cạnh này. Extension mới thêm **graph-rate controller**:

1. với mỗi candidate row action ở middle step, tính thay đổi active count và violation count trên toàn bộ graph;
2. dùng target từ ba train partitions, không dùng held-out rate;
3. chọn greedily tập action làm vector conditional rates toàn graph gần target nhất;
4. giữ nguyên dynamic topology, annealed state gate và component action layer để comparison chỉ thay controller.

Sweep một seed đang so sánh:

- split-median + graph-rate ranking;
- upper endpoint của train-split uncertainty set;
- split-median Wilson upper với hai mức uncertainty;
- late-only control;
- graph-rate controller trong late window.

Slurm jobs:

- `1067504`: sáu GPU configurations;
- `1067510`: aggregate phụ thuộc `afterok`;
- output tổng hợp dự kiến: `outputs/graph_rate_controller_selection_20261010/`.

Đây là model extension có quan hệ trực tiếp với contribution chính: guidance không còn xem graph như danh sách rule độc lập mà ra quyết định trên graph có active set thay đổi theo row và reverse state. Chỉ cấu hình thắng prior split-median controller trên held-out error mà không làm xấu KS/TV mới được đưa sang confirmation nhiều seed.

Disk cleanup cùng ngày giảm `analysis/.../outputs` từ khoảng 15 GB xuống 9.8 GB; checkpoint EMA cuối, base checkpoint, metric và log đều được giữ. Chi tiết nằm trong `DISK_CLEANUP_20261010.md`.

### 12.1 Kết quả selection và paired confirmation

Trên seed dùng để chọn controller, split-median + graph-rate đạt:

| Controller | Held-out error ↓ | KS ↓ | TV ↓ | Actions | Gain so với local |
| --- | ---: | ---: | ---: | ---: | ---: |
| split-median local | 0.004290 | 0.034980 | 0.018756 | 139 | — |
| split-median graph-rate | **0.003658** | 0.034987 | 0.018810 | 129 | **14.7%** |
| split-Wilson 0.25 graph-rate | 0.003812 | 0.034987 | 0.018810 | 115 | 11.1% |
| split-upper graph-rate | 0.007513 | 0.034970 | 0.018647 | 103 | -75.1% |

`split-upper` quá lỏng và bị bác bỏ. Late-only cũng xấu hơn local khoảng 20%, nên gain không thể giải thích bằng việc đơn giản dời action về cuối trajectory.

Paired confirmation ba seed cho kết quả thận trọng hơn:

| Method | Held-out error ↓ | KS ↓ | TV ↓ | Actions | Runtime (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| continued unguided | 0.030052 ± 0.009133 | 0.030857 | 0.017627 | 0.0 | 0.00 |
| split-median final repair | 0.005486 ± 0.000368 | 0.030857 | 0.018438 | 88.3 | 0.00 |
| split-median local | 0.003861 ± 0.000636 | 0.030830 | 0.018770 | 112.3 | 9.38 |
| split-median graph-rate | **0.003717 ± 0.000312** | 0.030841 | 0.018733 | 103.3 | 27.07 |

Diễn giải đúng:

- graph-rate tốt hơn final repair 32.3% và thắng 3/3 seed;
- tốt hơn local controller khoảng 3.7% nếu so ratio của mean errors, nhưng mean paired relative gain chỉ 0.9% và thắng 2/3 seed;
- variance giảm gần một nửa và action count giảm khoảng 8%;
- runtime tăng gần 3 lần do implementation hiện tại thử candidate row tuần tự bằng Pandas/kNN;
- vì gain trên local controller còn nhỏ, graph-rate hiện là extension/ablation tích cực, chưa phải main claim mới.

Common metrics một seed cho thấy không có trade-off lớn: Shape 2.3364, Trend 2.4102, Alpha 98.7278, Beta 52.4805, detection 0.9419, XGBoost AUC 0.7226 và DCR 51.4893. So với local controller, Shape/Trend xấu nhẹ, Alpha/Beta và DCR tốt nhẹ, AUC giảm khoảng 0.007. Cần giữ toàn bộ vector metric thay vì chọn riêng semantic error.

Artifacts:

- selection: `outputs/graph_rate_controller_selection_20261010/RESULTS.md`;
- confirmation: `outputs/graph_rate_confirmation_20261010/RESULTS.md`;
- common metrics: `outputs/graph_rate_common_quality_1067514/RESULTS.md`.

### 12.2 Scale theo kích thước graph

Một seed paired trên graph tự nhiên lớn hơn cho thấy controller có ích rõ khi graph có đủ tương tác nhưng vẫn được lọc:

| Graph | Method | Held-out error ↓ | TV ↓ | Actions |
| --- | --- | ---: | ---: | ---: |
| 8 cạnh | final repair | 0.027700 | 0.019657 | 287 |
| 8 cạnh | local controller | 0.026737 | 0.020444 | 301 |
| 8 cạnh | **graph-rate controller** | **0.021696** | **0.019135** | **236** |
| 12 cạnh | final repair | 0.026556 | 0.019698 | 290 |
| 12 cạnh | **local controller** | **0.021105** | 0.022264 | 376 |
| 12 cạnh | graph-rate controller | 0.021955 | **0.019362** | **265** |

Trên graph 8 cạnh, graph-rate tốt hơn local 18.9%, tốt hơn final repair 21.7% và giảm action 21.6%. Đây là tín hiệu mạnh nhất của extension mới.

Trên graph 12 cạnh, graph-rate xấu hơn local 4.0% dù vẫn tốt hơn final repair 17.3% và giảm action 29.5%. Bốn cạnh bổ sung gồm nhiều empirical soft relations có rate shift đáng kể; controller bám train targets nhưng target đó không generalize đồng đều. Negative result này xác định rõ limit và ablation cần có: **graph quality/reliability quan trọng hơn edge count**.

Do đó claim method nên là joint controller cho **verified, calibrated dynamic graph**, không phải “càng nhiều knowledge edges càng tốt”. Bảng chi tiết một seed nằm tại `outputs/graph_rate_expanded_summary_20261010/RESULTS.md`.

Paired confirmation ba seed trên graph 8 cạnh đã hoàn thành:

| Method | Held-out error ↓ | Calibration error ↓ | KS ↓ | TV ↓ | Actions |
| --- | ---: | ---: | ---: | ---: | ---: |
| continued unguided | 0.054517 ± 0.009953 | 0.039907 | 0.030857 | 0.017627 | 0.0 |
| final repair | 0.027546 ± 0.000856 | **0.013383** | 0.030857 | 0.018972 | 180.7 |
| local controller | 0.025005 ± 0.001500 | 0.020099 | 0.030865 | 0.020532 | 218.0 |
| **graph-rate controller** | **0.022496 ± 0.000792** | 0.018524 | **0.030832** | **0.020128** | **195.3** |

Paired graph-rate gain:

- 57.7% so với continued unguided, thắng 3/3 seed;
- 9.7% so với local controller, thắng 3/3 seed;
- 18.2% so với final repair, thắng 3/3 seed.

So ratio của mean errors, graph-rate tốt hơn local khoảng 10.0%. Đây là con số nên dùng trong phần kết quả chính; không dùng 18.9% của seed selection như claim tổng quát.

Common metrics một seed cũng ủng hộ rằng gain không chỉ do over-enforcement:

| Method | Shape error ↓ | Trend error ↓ | Alpha ↑ | Beta ↑ | XGBoost AUC ↑ | DCR (50 ideal) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| local controller | 2.3387 | 2.4116 | 98.8276 | 52.2201 | 0.7210 | 51.4648 |
| final repair | 2.3173 | 2.2907 | **98.8932** | 52.3063 | 0.7185 | **51.0742** |
| **graph-rate controller** | **2.3085** | **2.2554** | 98.8564 | **52.4561** | **0.7218** | 51.2939 |

Graph-rate có macro raw violation cao hơn local (0.1385 so với 0.1348) nhưng held-out rate error thấp hơn. Đây là kết quả có ý nghĩa: mục tiêu của soft semantic graph không phải ép violation về thấp nhất, mà phải bảo toàn đúng conditional exception mass của dữ liệu thật.

Artifacts cuối:

- three-seed: `outputs/graph_rate_8edge_confirmation_20261010/RESULTS.md`;
- common metrics: `outputs/graph_rate_8edge_common_quality_1067521/RESULTS.md`;
- Slurm confirmation: `1067520`; common metrics: `1067521`.

## Cập nhật cross-dataset và Harpoon ngày 2026-10-10

Đánh giá chi tiết mới nhất được tách tại
[`CROSS_DATASET_HARPOON_ASSESSMENT_20261010_VI.md`](CROSS_DATASET_HARPOON_ASSESSMENT_20261010_VI.md).

Kết quả chính: trên Default 8-edge, dynamic graph-rate đạt held-out error
`0.02250±0.00079`, tốt hơn Harpoon-style matched proxy `0.02792±0.00421`
(`19.4%`) và final repair `0.02755±0.00086` (`18.2%`). Shoppers cho thấy
controller tổng quát hóa nhưng dynamic/frozen-state hòa nhau; Diabetes chỉ đạt
edge readability ở late steps và dynamic guidance hòa final repair. Do đó claim
hiện tại là trajectory-calibrated selective enforcement, chưa phải dynamic
topology luôn thắng trên mọi dataset.

Ablation bổ sung sau đó xác nhận dynamic graph-rate tốt hơn frozen-state trên
Default 8-edge `6.33%` về held-out error và thắng `3/3` seed, với action budget
gần như bằng nhau. Kết quả này hỗ trợ trực tiếp main novelty, nhưng Shoppers và
Diabetes vẫn cho thấy gain phụ thuộc edge readability và trajectory dynamics.
