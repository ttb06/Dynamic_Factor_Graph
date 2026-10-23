# Đánh giá cross-dataset, Harpoon và độ vững của contribution

Ngày cập nhật: 2026-10-10.

## 1. Kết luận ngắn

Hướng nghiên cứu **đủ mạnh để tiếp tục form thành paper**, nhưng chưa đủ bằng chứng để gọi là SOTA tổng quát trên synthetic tabular data.

Kết quả mạnh nhất hiện tại nằm trên Default với graph 8 cạnh: dynamic graph-rate guidance giảm held-out constraint error `19.4%` so với Harpoon-style matched proxy, `18.2%` so với final repair và `9.7%` so với local row-wise controller. Fidelity và downstream utility không giảm có hệ thống. Điểm mạnh của method không phải là ép violation rate thấp nhất bằng mọi giá, mà là đưa violation profile gần dữ liệu thật hơn bằng số can thiệp nhỏ.

Cross-dataset cho thấy boundary rõ:

- Shoppers: controller tổng quát hóa, giảm error `26.5%` so với unconditional và nhỉnh hơn final repair; nhưng dynamic và frozen-state gần như bằng nhau trong protocol mới.
- Diabetes: các categorical constraints chỉ readable ở `t≈0.082`. Khi can thiệp đúng cửa sổ late-valid, guidance và final repair bằng nhau; dynamic topology không tạo thêm gain. Đây là negative result có giá trị: không thể claim middle-step luôn tối ưu.
- Vì vậy claim mạnh hiện tại nên là **trajectory-calibrated selective constraint enforcement**. Dynamic topology đã được xác nhận trên Default 8-edge (`6.33%`, thắng `3/3` seed), nhưng không phải dataset-independent.

## 2. Bài toán và method đang kiểm tra

Cho predicted-clean state `x̂₀(x_t)` tại mỗi reverse step, method thực hiện:

1. Đọc graph đã được discovery, kiểm chứng bằng train/calibration và gán vai trò `actuate`, `score_only`, `monitor`.
2. Với từng row và timestep, xây active graph từ các cạnh đã readable, đang applicable và có violation mass vượt target rate học từ calibration split.
3. Action layer sinh các typed corrections khả thi, khóa antecedent columns khi cần và đánh giá counterfactual bằng graph-level certifier, density và edit cost.
4. Chỉ đưa action được chấp nhận trở lại denoising trajectory; final repair là baseline riêng, không được tính là mid-step gain.

Khác biệt cần nhấn mạnh với Harpoon không phải “loss thay đổi theo state”: Harpoon cũng có state-dependent gradient. Khác biệt có thể bảo vệ là **authority to intervene thay đổi theo edge/row/timestep**, bao gồm readability, applicability, calibrated target rate, graph conflict và abstention. Harpoon dùng fixed constraint program và liên tục đẩy loss; method này chọn một active subgraph rồi thực hiện sparse typed actions.

## 3. Datasets và graph

| Dataset | Graph dùng khi sampling | Nodes | Edges | Actuating edges | Kiểu quan hệ chính |
| --- | ---: | ---: | ---: | ---: | --- |
| Default | natural DAG 8-edge | 9 | 8 | 8 | chuỗi BILL/PAY, categorical target, soft empirical rates |
| Shoppers | trajectory portfolio | 15 | 27 | 7 | numeric implication, column relation, hard và near-hard |
| Diabetes | trajectory portfolio | 6 | 17 | 4 | categorical multi-antecedent/multi-consequent, chủ yếu hard |

Mỗi graph chỉ dùng train/calibration để discovery và selection. Evaluation graph lớn hơn, độc lập với tập cạnh được quyền actuation: Shoppers 30 edges, Diabetes 27 edges.

## 4. Default 8-edge: matched comparison với Harpoon

Ba seed, 4,096 rows/seed, cùng TabDiff checkpoint đã continued-train, cùng graph, cùng deterministic sampler. Harpoon weight `8` được chọn trên seed development `20261379` bằng calibration error rồi xác nhận trên ba seed `20261376–78`. Đây là **Harpoon-style TabDiff proxy**, không phải official reproduction: categorical energy được cộng vào logits trước absorbing discrete update.

| Method | Held-out error ↓ | Calibration error ↓ | Macro violation | Numeric KS ↓ | Cat. TV ↓ | Interventions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Continued TabDiff | 0.05452 ± 0.00995 | 0.03991 | 0.17545 | 0.03086 | 0.01763 | 0 |
| Final repair | 0.02755 ± 0.00086 | **0.01338** | 0.14563 | 0.03086 | 0.01897 | 180.7 |
| Harpoon-style hard, w=8 | 0.02792 ± 0.00421 | 0.02122 | **0.13746** | 0.03078 | 0.02075 | 7,130 editable |
| Local row-wise controller | 0.02501 ± 0.00150 | 0.02010 | 0.13484 | 0.03087 | 0.02053 | 218.0 |
| **Dynamic graph-rate** | **0.02250 ± 0.00079** | 0.01852 | 0.13849 | **0.03083** | 0.02013 | **195.3** |

Diễn giải:

- Dynamic graph-rate thấp hơn Harpoon `19.4%` về held-out error và dùng khoảng `36.5×` ít interventions hơn.
- Harpoon có macro violation hơi thấp hơn (`0.13746` so với `0.13849`) nhưng xa target rate thật hơn. Điều này minh họa lỗi over-enforcement: giảm violation tuyệt đối không đồng nghĩa fidelity constraint tốt hơn.
- Dynamic graph-rate tốt hơn final repair `18.2%` và thắng local controller cả 3/3 seed (`9.7%` paired gain).
- Nhược điểm hiện tại: runtime sampler khoảng `72.8 s`, trong khi Harpoon proxy chỉ `3.68 s`. Method đang tốt hơn về intervention efficiency nhưng kém về wall-clock efficiency.

### Dynamic topology so với frozen-state trên Default

| Method | Held-out error ↓ | Calibration error ↓ | Macro violation | TV ↓ | Actions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen-state graph-rate | 0.02402 | 0.02051 | **0.13779** | **0.02006** | 197.3 |
| **Dynamic graph-rate** | **0.02250** | **0.01852** | 0.13849 | 0.02013 | 195.3 |

Dynamic refresh giảm held-out error trung bình `6.33%` và thắng `3/3` seed. Action count gần như bằng nhau, nên gain không đến từ việc dynamic method được sửa nhiều hơn. Frozen có macro violation thấp hơn một chút nhưng calibration/held-out error cao hơn; đây tiếp tục là dấu hiệu over-enforcement thay vì match empirical rate.

### Common synthetic-data metrics trên cùng ba seed

| Method | Shape err. ↓ | Trend err. ↓ | Alpha precision ↑ | Beta recall ↑ | Detection ↑ | XGBoost AUC ↑ | DCR, 50 ideal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Continued TabDiff | 2.0030 | 2.2268 | 98.4531 | **52.5347** | 0.9553 | 0.7227 | 52.2705 |
| Final repair | **1.9847** | **2.2017** | 98.5738 | 52.4729 | 0.9523 | 0.7241 | **52.0101** |
| Harpoon-style | 2.0310 | 2.3122 | **98.5917** | 52.3161 | 0.9535 | 0.7238 | 52.1240 |
| **Dynamic graph-rate** | 2.0305 | 2.2843 | 98.5426 | 52.3958 | **0.9574** | **0.7265** | 52.2217 |

Không có bằng chứng về fidelity collapse. Dynamic method gần như hòa Harpoon về Shape, tốt hơn về Trend, detection và downstream AUC; final repair vẫn tốt nhất ở Shape/Trend/DCR. Vì các chênh lệch common metrics nhỏ, không nên claim SOTA fidelity, chỉ có thể nói semantic gain không phải trả bằng degradation rõ rệt.

## 5. Shoppers: mixed numerical constraints

Một seed mới, 4,096 rows, cùng checkpoint/graph/evaluation graph:

| Method | Held-out error ↓ | Calibration error ↓ | KS ↓ | TV ↓ | Actions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unconditional | 0.0204 | 0.0181 | 0.0267 | 0.0228 | 0 |
| Final repair | 0.0154 | 0.0120 | 0.0267 | 0.0228 | 117 |
| Dynamic `xstate` | **0.0150** | 0.0121 | **0.0266** | 0.0228 | 146 |
| Frozen-state `xstate` | **0.0150** | 0.0121 | **0.0266** | 0.0228 | 146 |
| Dynamic + final | **0.0147** | **0.0120** | 0.0266 | 0.0228 | 156 |

Controller giảm held-out error `26.5%` so với unconditional và khoảng `2.6%` so với final repair. Tuy nhiên active labels gần như không thay đổi trong cửa sổ được chọn, nên dynamic và frozen-state bằng nhau. Kết quả cũ với protocol mạnh hơn cho dynamic tốt hơn frozen-state khoảng `7.1%` trên hai seed, nhưng result mới buộc ta không được claim topology gain là dataset-independent.

## 6. Diabetes: boundary về timestep readability

Các actuating edges có cutoff chủ yếu `t=0.0816`; do đó cửa sổ middle `0.15≤t≤0.60` tạo đúng `0` mid-step actions. Với cửa sổ late-valid `0.05≤t≤0.20`:

| Method | Held-out error ↓ | Calibration error ↓ | KS ↓ | TV ↓ | Actions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unconditional | 0.0166 | 0.0165 | 0.0226 | 0.0207 | 0 |
| Final repair | **0.0113** | **0.0112** | 0.0226 | 0.0209 | 21 |
| Dynamic late-valid | **0.0113** | **0.0112** | 0.0226 | **0.0208** | 21 |
| Frozen-state late-valid | **0.0113** | **0.0112** | 0.0226 | **0.0208** | 21 |

Guidance giảm error `31.9%` so với unconditional nhưng không hơn repair và không cần topology refresh. Đây là result nên giữ trong paper như một boundary condition: optimal intervention time phụ thuộc feature-type/edge readability; middle-step chỉ có ý nghĩa khi state còn thay đổi nhưng đã đủ semantic signal.

## 7. Official Harpoon results: chỉ dùng làm external reference

Official-code audit trước đây đã chạy Adult, Default và Shoppers với constraint suites `range`, `category`, `both`, `or`. Ví dụ one-trial rerun của `harpoon_ohe` có ViolationAcc `23.60` trên Adult-both, `10.16` trên Default-both và `25.17` trên Shoppers-both; paper-reported five-trial values lần lượt `3.37`, `2.98`, `12.05`.

Không đặt các số này chung ranking với bảng trên vì backbone, constraint suite, split, metric scale và trial count khác nhau. Vai trò của official rerun là chứng minh baseline chạy được trên nhiều dataset; vai trò của matched proxy là kiểm tra mechanism trên cùng backbone và graph.

## 8. Contribution hiện đã đủ tốt chưa?

### Phần đã đủ mạnh

1. **Motivation thực nghiệm rõ:** fixed aggressive guidance có thể giảm macro violation nhưng làm sai empirical constraint rates; Default cho ví dụ trực tiếp giữa Harpoon và graph-rate controller.
2. **Matched strong baselines:** cùng backbone/graph/seed đã có unconditional, final repair, local controller và Harpoon-style energy guidance.
3. **Không chỉ báo semantic metric:** common fidelity, detection, downstream AUC và privacy-distance không cho thấy collapse.
4. **Có boundary/negative result:** Shoppers không luôn có topology gain; Diabetes không hỗ trợ middle-step do readability muộn. Điều này làm claim đáng tin hơn.

### Phần chưa đủ để gọi SOTA hoặc khóa paper

1. Dynamic topology thắng frozen-state ổn định trên Default (`6.33%`, 3/3 seed), nhưng hòa trên protocol mới của Shoppers và Diabetes; cần tránh diễn đạt thành universal gain.
2. Matched Harpoon hiện đầy đủ nhất trên graph categorical Default; Shoppers mixed numerical graph chưa có Harpoon proxy bao phủ toàn bộ operators.
3. Default standard metrics mới có ba seed nhưng Shoppers/Diabetes mới chỉ có constraint/KS/TV; cần downstream utility và detection cùng protocol nếu đưa vào main table.
4. Chưa có end-to-end comparison với official TabDiff/TabSyn training budgets trên toàn bộ datasets. Checkpoints hiện là local bounded runs; không được gọi là paper reproduction.
5. Graph discovery và sampling đang cùng đóng góp; cần freeze graph rồi ablate riêng trajectory activation, target-rate controller và action selection để reviewer xác định nguồn gain.

## 9. Claim nên dùng và claim cần tránh

Claim phù hợp:

> A verified constraint graph can be converted into a row- and trajectory-specific active graph, and rate-aware sparse actuation can match real constraint profiles better than fixed energy guidance or post-generation repair on datasets whose constraints become readable before the terminal steps.

Không nên claim:

- LLM feedback trong diffusion là mới;
- per-step constraints là mới;
- dynamic graph luôn hơn repair;
- method đạt SOTA synthetic-data fidelity;
- Diabetes ủng hộ middle-step guidance.

## 10. Bước tiếp theo có giá trị thông tin cao nhất

1. Thêm Harpoon-style mixed proxy với operator coverage report trên Shoppers; không được silently skip unsupported edges.
2. Xác nhận Shoppers controller/final result thêm hai seed và phân tích vì sao active graph ít flip hơn Default.
3. Đưa edge readability distribution thành một figure: Default/Shoppers có usable middle region, Diabetes chỉ readable gần cuối. Đây có thể là một contribution phân tích độc lập và giải thích khi nào method nên abstain.
4. Tối ưu runtime của component-beam/action certifier; hiện semantic gain tốt nhưng chậm hơn fixed energy guidance gần 20 lần.

## 11. Artifacts

- Default main result: `outputs/graph_rate_8edge_confirmation_20261010/RESULTS.md`
- Matched Harpoon sweep: `outputs/harpoon_default_8edge_1067609/RESULTS.md`
- Matched Harpoon confirmation: `outputs/harpoon_default_8edge_confirm_s*_1067613_*/`
- Common-quality comparison: `outputs/harpoon_dynamic_common_s*_1067619_*/RESULTS.md`
- Shoppers pair: `outputs/cross_dynamic_shoppers_*_1067608_*/RESULTS.md`
- Diabetes middle no-op: `outputs/cross_dynamic_diabetes_*_1067608_*/RESULTS.md`
- Diabetes late-valid: `outputs/diabetes_x0_pair_*_1067629_*/RESULTS.md`
- Default frozen-state confirmation: `outputs/default_8edge_frozen_s*_1067698_*/RESULTS.md`
- Official Harpoon audit: `outputs/harpoon_official_multidata_1061929/RESULTS.md`
