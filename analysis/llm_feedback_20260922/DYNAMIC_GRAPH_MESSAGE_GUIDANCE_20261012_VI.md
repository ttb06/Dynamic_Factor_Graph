# Dynamic Graph Message Guidance theo diffusion time

**Cập nhật:** 12/10/2026  
**Trạng thái:** cơ chế mới đã chạy trên Default với graph 3 và 8 cạnh; đã có kiểm tra theo seed, thời điểm bắt đầu guidance, độ mạnh và khả năng lan truyền qua graph. Đây vẫn là kết quả exploratory, chưa phải bảng SOTA cuối cùng.

## 1. Kết luận ngắn

Cơ chế đáng theo tiếp nhất hiện tại là **message passing xác suất trực tiếp trên clean categorical logits trong reverse diffusion**:

1. Mỗi cạnh biểu diễn một quan hệ có hướng, ví dụ `PAY_4 = -2 -> PAY_5 = -2`.
2. Ở mỗi bước, mô hình không quyết định cạnh chỉ bằng row đã `argmax`; nó tính xác suất antecedent đúng từ phân phối `x_hat_0`.
3. Xác suất này là trọng số active của cạnh đối với từng row và từng timestep.
4. Message chỉ đi xuôi từ antecedent sang consequent. Không sửa antecedent để tránh “né điều kiện”.
5. Ba lượt message passing trong một timestep cho phép thay đổi ở node trước làm active/deactive cạnh sau.
6. Message được cộng vào clean logits trước categorical reverse transition.

Trên seed khám phá của Default, graph 3 cạnh đạt held-out constraint error **0,00889**, so với TabDiff **0,04066**, dynamic CDD proxy **0,01994** và calibrated final repair **0,01031**. Tuy nhiên trung bình ba sampling seed là **0,01311 ± 0,00591**, vẫn thấp hơn TabDiff nhưng chưa thắng ổn định final repair **0,01025 ± 0,00009**. Do đó kết luận đúng hiện tại là:

- cơ chế có hiệu quả rõ so với sampling không guidance;
- propagation qua graph tạo thêm gain, không chỉ do tăng guidance strength;
- bắt đầu ở giữa trajectory tốt hơn bắt đầu quá sớm hoặc quá muộn trên seed kiểm tra;
- chưa đủ bằng chứng để claim SOTA hay thắng post-repair ổn định.

## 2. Ý tưởng mượn từ các hướng liên quan

| Nguồn ý tưởng | Thành phần liên quan | Cách dùng trong bài này | Khác biệt cần giữ khi viết |
|---|---|---|---|
| [GILC, ICML 2026](https://proceedings.mlr.press/v306/dou26a.html) | Sửa trực tiếp clean logits trong discrete diffusion, không cần Jacobian | Cộng message vào logits của consequent trước reverse transition | GILC không phải đóng góp của ta; phần cần chứng minh là graph theo row/timestep, applicability xác suất và propagation có hướng trên tabular graph |
| [LGD, ICML 2023](https://proceedings.mlr.press/v202/song23k.html) | Guidance nên xét bất định thay vì chỉ dùng một point estimate | Dùng xác suất antecedent dưới `x_hat_0` thay vì row `argmax` | Hiện dùng factorized analytic probability, chưa phải Monte Carlo estimator đầy đủ |
| [Constrained Discrete Diffusion, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/124c1cc9f7750f512ed81cdbf2fd5bf5-Abstract-Conference.html) | Constraint layer nằm trong từng bước discrete denoising | Dynamic feasibility/message layer cho categorical columns | CDD không chạy trực tiếp benchmark tabular này; dynamic graph và soft empirical constraints là setting khác |
| [Sampling Demons, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/eeab2e00835c71d64458ad1821e05664-Paper-Conference.pdf) | Sinh nhiều candidate perturbation và chọn bằng black-box reward | Đã được phản ánh bởi các lookahead/best-of-N experiment trước | Trong repo, local rollback/lookahead tốn hơn nhưng không ổn định hơn final repair, nên không lặp lại làm main method |
| [Free2Guide, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Kim_Free2Guide_Training-Free_Text-to-Video_Alignment_using_Image_LVLM_ICCV_2025_paper.html) | Gradient-free control từ black-box reward | Gợi ý branch weighting và reward control | Các experiment branch selection trước chỉ cho gain nhỏ; direct graph messages hiện đơn giản và mạnh hơn |
| Belief propagation / sum-product | Truyền message dọc factor graph | Đã thử directed sum-product message | Kết quả kém linear gap message, nên chỉ report như negative result |

### Những hướng đã có trong repo nhưng không nên chạy lại làm main

- rollback và lookahead nhiều horizon: giảm số action nhưng không cải thiện quality ổn định;
- EMA/hysteresis để active edge: không thắng đồng đều;
- selective re-mask: giảm raw violations nhưng over-correct, làm held-out distributional error xấu hơn;
- final repair: vẫn là baseline rất mạnh và phải giữ;
- dynamic CDD/min-KL projection: baseline in-denoising gần nhất hiện có trong cùng pipeline.

## 3. Method hiện tại

Với cạnh có hướng `e: A -> B`, tại timestep `t` và row `i`:

\[
a_{i,e,t} = P(A_e=1\mid \hat{x}_{0,t}^{(i)})
\]

`a` là trọng số active mềm của cạnh. Với soft constraint, target satisfaction được lấy từ calibration split của train:

\[
r_e = 1 - \text{violation-rate}_{e,calibration}.
\]

Nếu `q_{i,e,t}` là xác suất consequent đúng, linear message hiện dùng:

\[
\Delta \ell_{i,e,t}
= \eta\,w(t)\,a_{i,e,t}\,[r_e-q_{i,e,t}]_+.
\]

`Delta l` được cộng vào clean logit của value đích. Sau mỗi pass, probability của các node được tính lại; do đó message ở `PAY_4` có thể thay đổi mức active của cạnh `PAY_4 -> PAY_5`, rồi tiếp tục tới `PAY_6`.

### Các lựa chọn thiết kế quan trọng

- **Có hướng:** chỉ sửa consequent, không giảm xác suất antecedent để thỏa constraint giả tạo.
- **Soft calibration:** không ép mọi empirical relation thành invariant cứng.
- **Dynamic theo row và timestep:** cùng một cạnh có trọng số khác nhau theo trạng thái hiện tại của từng row.
- **Bắt đầu ở giữa trajectory:** mặc định chỉ guidance khi `t <= 0,5`; trước đó preview chưa đủ đáng tin.
- **Training-free:** dùng checkpoint TabDiff đã train; LLM chỉ thuộc upstream graph proposal, không nằm trong inner denoising loop.

Code chính: `scripts/dynamic_graph_logit_messages.py`.

## 4. Graph kiểm tra

### Graph 3 cạnh liên thông

1. `BILL_AMT3 = 0 -> PAY_4 = -2`;
2. `PAY_4 = -2 -> PAY_5 = -2`;
3. `PAY_5 = -2 -> PAY_6 = -2`.

Đây là chuỗi phù hợp để kiểm tra propagation: message ở node trước có thể làm thay đổi active set của cạnh sau trong cùng timestep.

Graph 8 cạnh mở rộng thêm các entry từ `BILL_AMT1/2/4` và chuỗi `PAY_2 -> PAY_3 -> PAY_4 -> PAY_5 -> PAY_6`. Tất cả là **soft empirical relation**, không được diễn giải như luật tài chính tuyệt đối.

## 5. Kết quả chính trên seed khám phá

Dataset: Default; 4.096 synthetic rows; cùng checkpoint, seed, split, graph và evaluator. Lower tốt hơn đối với Shape error, Trend error và held-out constraint error.

| Method | Shape err. ↓ | Trend err. ↓ | α-precision ↑ | β-recall ↑ | Detection ↑ | MLE AUC ↑ | Held-out graph err. ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| TabDiff | 2,4058 | 2,5072 | 98,6638 | 52,5049 | 0,9371 | 0,7168 | 0,04066 |
| Dynamic CDD/min-KL proxy | 2,3096 | 2,4198 | 98,7831 | 50,7145 | 0,9336 | 0,7231 | 0,01994 |
| Calibrated final repair | 2,3179 | 2,4268 | 98,7713 | 52,4577 | 0,9389 | 0,7172 | 0,01031 |
| Message passing, 1 pass | 2,3428 | 2,4229 | 98,7078 | 52,4561 | 0,9376 | 0,7226 | 0,02818 |
| **Message passing, 3 passes** | **2,3078** | 2,3641 | 98,8055 | 52,4935 | **0,9394** | **0,7253** | **0,00889** |
| 1 pass, strength ×3 | **2,3078** | **2,3476** | **98,8494** | **52,5228** | 0,9389 | 0,7237 | 0,01525 |
| 3 passes, frozen routing | 2,3095 | 2,3827 | 98,7680 | 52,5000 | 0,9388 | 0,7240 | 0,01215 |

Trên seed này, 3-pass message:

- giảm graph error **78,1%** so với TabDiff;
- giảm **55,4%** so với dynamic CDD proxy;
- giảm **13,8%** so với final repair;
- đồng thời cải thiện Shape, Trend, α-precision, detection và MLE AUC so với TabDiff.

Đây là seed dùng để chọn `eta=8`, nên không được dùng một mình làm con số kết luận cuối.

## 6. Ablation: gain có thực sự đến từ graph propagation?

| Variant | Held-out graph err. ↓ | Diễn giải |
|---|---:|---|
| 1 pass, `eta=8` | 0,02818 | Không đủ thời gian lan qua chuỗi |
| 3 passes, tổng strength chuẩn hóa | 0,02962 | Chỉ chia nhỏ cùng tổng strength là chưa đủ |
| 1 pass, `eta=24` | 0,01525 | Tăng strength giúp rõ nhưng chưa đạt 3-pass |
| 3 passes, frozen antecedent routing | 0,01215 | Cùng số pass nhưng không cập nhật active set từ node upstream |
| 3 passes, dynamic routing | **0,00889** | Tốt nhất |

Kết quả này hỗ trợ hai thành phần riêng:

1. guidance cần đủ mạnh;
2. cập nhật routing sau mỗi pass vẫn tạo gain thêm so với matched strong guidance và frozen routing.

Chênh lệch dynamic routing với frozen routing là bằng chứng gần nhất cho contribution “graph thay đổi theo trạng thái trong trajectory”. Cần lặp ablation này trên dataset khác trước khi claim mạnh.

## 7. Ablation theo thời điểm bắt đầu guidance

| Bắt đầu guidance | Shape err. ↓ | Trend err. ↓ | Held-out graph err. ↓ | Message events | Nhận xét |
|---|---:|---:|---:|---:|---|
| Muộn, `t <= 0,25` | 2,3559 | 2,4455 | 0,02565 | 3.388 | Quá ít thời gian để correction truyền qua chuỗi |
| **Giữa, `t <= 0,50`** | **2,3078** | **2,3641** | **0,00889** | 40.384 | Điểm cân bằng tốt nhất trên seed này |
| Sớm, `t <= 0,75` | 2,5570 | 2,7571 | 0,03096 | 150.252 | Over-guidance từ preview còn bất định |

Lưu ý: đây là **thời điểm bắt đầu**, không phải chỉ guidance trong một cửa sổ giữa. Sau khi bắt đầu, method tiếp tục guidance tới cuối trajectory. Cách phát biểu chính xác là “mid-start trajectory guidance”.

### Certainty gate thay cho lịch thời gian cứng

Một thử nghiệm bổ sung cho method bắt đầu sớm tại `t=0,75`, nhưng nhân active weight với một sigmoid gate theo độ chắc chắn của antecedent:

| Variant bắt đầu tại `t=0,75` | Shape err. ↓ | Trend err. ↓ | MLE AUC ↑ | Held-out graph err. ↓ | Message events |
|---|---:|---:|---:|---:|---:|
| Không gate | 2,5570 | 2,7571 | **0,7293** | 0,03096 | 150.252 |
| Certainty threshold 0,50 | 2,3578 | 2,4305 | 0,7177 | 0,01995 | 133.728 |
| Certainty threshold 0,75 | **2,3405** | **2,4022** | 0,7248 | **0,01740** | 101.532 |

Gate theo posterior state sửa được phần lớn tác hại của early guidance, tự giảm số message, đồng thời đưa Shape/Trend trở lại tốt hơn TabDiff. Tuy nhiên nó vẫn chưa tốt bằng fixed mid-start 0,00889 về graph error; vì vậy hiện nên xem đây là cơ chế ổn định hóa và hướng để phát triển adaptive schedule, không phải default cuối cùng.

## 8. Kết quả graph 8 cạnh

| Method | Shape err. ↓ | Trend err. ↓ | α-precision ↑ | MLE AUC ↑ | Held-out graph err. ↓ |
|---|---:|---:|---:|---:|---:|
| TabDiff | 2,4058 | 2,5072 | 98,6638 | 0,7168 | 0,06481 |
| Previous x-state reward | 2,2999 | 2,2533 | 98,8792 | 0,7153 | 0,02303 |
| Calibrated final repair | 2,3173 | 2,2907 | 98,8932 | 0,7185 | 0,02770 |
| Message, 1 pass | **2,2803** | 2,2689 | 98,8380 | 0,7149 | 0,04202 |
| **Message, 3 passes** | 2,3146 | 2,2814 | **99,0452** | 0,7151 | **0,02215** |
| Directed sum-product, 3 passes | 2,2722 | **2,2289** | 98,9561 | 0,7210 | 0,02852 |

Graph 8 cạnh cho thấy mechanism scale qua nhiều cạnh hơn:

- 3-pass giảm held-out graph error **65,8%** so với TabDiff;
- tốt hơn previous x-state reward khoảng **3,8%** và final repair khoảng **20,0%** trên graph error;
- nhưng không đứng đầu mọi fidelity/utility metric. Linear message tốt hơn sum-product về graph error, còn sum-product tốt hơn về Shape/Trend/AUC trên seed này.

## 9. Kiểm tra ba sampling seed

Ba seed dùng cùng checkpoint và graph; đây là thay đổi random sampling, chưa phải ba training seed.

| Method | Shape err. ↓ | Trend err. ↓ | MLE AUC ↑ | Held-out graph err. ↓ |
|---|---:|---:|---:|---:|
| TabDiff | 1,9833 ± 0,3659 | 2,2165 ± 0,2568 | 0,7230 ± 0,0092 | 0,03092 ± 0,00844 |
| Message, 3 passes | 1,9959 ± 0,2702 | 2,2668 ± 0,0847 | 0,7227 ± 0,0063 | 0,01311 ± 0,00591 |
| Dual middle schedule, 3 passes | 1,9804 ± 0,2942 | 2,2427 ± 0,1368 | **0,7285 ± 0,0050** | 0,01548 ± 0,00091 |
| Calibrated final repair | **1,9634 ± 0,3071** | 2,2433 ± 0,1598 | 0,7230 ± 0,0050 | **0,01025 ± 0,00009** |

Phân tích trung thực:

- fixed 3-pass giảm graph error trung bình khoảng **57,6%** so với TabDiff;
- dual middle ổn định hơn về graph error và có MLE AUC cao nhất;
- final repair vẫn tốt hơn về graph error trung bình và Shape;
- fixed 3-pass làm Trend xấu nhẹ trung bình, nên seed khám phá Pareto-dominant không lặp lại hoàn toàn.

## 10. Negative results có giá trị

### Directed sum-product

Trên graph 3 cạnh, sum-product 3-pass đạt graph error 0,03086, kém linear 3-pass 0,02613 ở cùng grid `eta=4`. Vì vậy hiện chưa có lý do dùng belief-propagation update phức tạp làm default.

### Selective re-mask

Re-mask làm raw macro violation thấp hơn, nhưng graph error so với held-out distribution tăng. Nó đang ép consequent quá mạnh và làm mất soft exceptions tự nhiên.

### Guidance quá sớm

`start_t=0,75` tạo hơn 150 nghìn message events và cho held-out error 0,03096. Đây là phản ví dụ trực tiếp cho giả thuyết “guidance càng sớm/càng nhiều càng tốt”.

## 11. Claim có thể viết và claim chưa được phép viết

### Claim đã có support sơ bộ

1. Soft graph messages trong clean-logit space có thể cải thiện semantic graph fidelity mà không cần retrain backbone.
2. Row/timestep-dependent routing tạo gain ngoài việc tăng tổng guidance strength trên graph chuỗi 3 cạnh.
3. Có một mid-start sweet spot: quá muộn không đủ sửa, quá sớm gây over-guidance.
4. Method mở rộng từ 3 lên 8 cạnh và vẫn tốt hơn TabDiff, final repair và previous x-state reward về graph error trên seed kiểm tra.

### Chưa được claim

1. Chưa SOTA: mới một dataset, một backbone checkpoint và ba sampling seed.
2. Chưa thắng final repair ổn định trên graph 3 cạnh.
3. Chưa chứng minh cross-dataset hoặc mixed numeric/categorical messages.
4. Chưa chứng minh LLM cần thiết trong inner loop; method hiện tại không gọi LLM khi sampling.
5. Chưa được dùng held-out result này như final test vì `eta` và schedule đã được chọn sau khi xem kết quả.

## 12. Hướng experiment có giá trị cao nhất tiếp theo

1. **Đóng split đúng:** discovery train, tuning validation, final test chỉ mở một lần. Chọn `eta`, pass và `start_t` trên validation.
2. **Cross-dataset:** chọn một graph liên thông ở Shoppers hoặc Adult; nếu không có categorical chain tự nhiên thì dùng heterogeneous motif có numeric node nhưng không bịa hard rule.
3. **Adaptive temporal gate:** kích hoạt theo entropy/uncertainty của antecedent thay vì fixed `start_t`; so với fixed mid-start bằng cùng message budget.
4. **Node-degree normalization:** graph lớn có nhiều message đổ vào cùng node; chuẩn hóa theo in-degree để hạn chế over-guidance.
5. **Graph-quality stress test:** oracle/train-verified graph, shuffled topology, reversed edges và low-confidence graph. Đây là cách chứng minh method dùng đúng cấu trúc graph chứ không chỉ bias marginal.
6. **End-to-end option chỉ sau đó:** distill message target hoặc train denoiser với graph dropout. Chưa nên làm trước khi training-free method thắng ổn định post-repair.

## 13. File và job

- Method: `scripts/dynamic_graph_logit_messages.py`
- Main grid: jobs `1070239`, evaluator `1070240`
- Sum-product: jobs `1070259`, `1070260`
- Graph 8 cạnh: jobs `1070261`, evaluator sửa và chạy lại `1070291`
- Seed xác nhận: jobs `1070284`, `1070285`
- Propagation/matched-budget ablation: jobs `1070289`, `1070290`
- Time-window ablation: jobs `1070293`, `1070294`
- Certainty-gated early guidance: jobs `1070309`, `1070310`
- Graph 3 cạnh: `outputs/natural_dag_stable_v3_20261010/graph.json`
- Graph 8 cạnh: `outputs/natural_dag_multi_entry_v3_20261010/graph.json`

## 14. Đánh giá hướng nghiên cứu

Hướng này **đủ tốt để tiếp tục thành method paper**, vì đã có một phenomenon riêng và đo được: active topology thay đổi theo posterior state, propagation trong cùng timestep có tác dụng, và thời điểm bắt đầu có optimum nội tại. Nó mạnh hơn cách diễn giải chung chung “đưa constraint vào diffusion”.

Nhưng kết quả hiện tại mới ở mức **proof-of-mechanism**, chưa ở mức paper-ready. Điểm cần giải quyết không phải thêm nhiều engineering variant, mà là ba bằng chứng nền tảng: validation/test sạch, cross-dataset, và thắng hoặc tạo Pareto trade-off rõ ràng so với post-repair/dynamic projection. Nếu ba phần này thành công, framing hợp lý là:

> Training-free, trajectory-conditioned probabilistic message passing over verified semantic graphs for mixed-type tabular diffusion.

LLM nên được đặt ở graph discovery/semantic enrichment upstream; contribution sampling không phụ thuộc LLM API. Cách tách này làm paper mạnh hơn và tránh claim rằng LLM judgement ở mỗi step là nguyên nhân của gain khi thực nghiệm hiện chưa hỗ trợ điều đó.

## 15. Cập nhật ablation không LLM

Audit tiếp theo cho thấy cả 8 cạnh headline đều có nguồn gốc `empirical_mining`; LLM chỉ review/viết lại metadata cho một số cạnh. Một pipeline hoàn toàn không API và không mở test đã khai phá 30 candidate, lọc còn graph 8 cạnh có 7/8 program trùng headline.

Trên ba sampling seed, no-LLM fixed message giảm graph error trung bình từ 0,03923 xuống 0,02331; dual-middle đạt 0,02958 trong khi giữ Shape gần như nguyên vẹn. No-LLM fixed message cũng giảm lỗi trên headline graph từ 0,05380 xuống 0,02844.

Vì vậy main framing được điều chỉnh: graph discovery và guidance chính không dùng LLM. LLM chỉ còn là optional extension cho external constraints hoặc natural-language policies. Chi tiết nằm trong `NO_LLM_GRAPH_ABLATION_20261012_VI.md`.
