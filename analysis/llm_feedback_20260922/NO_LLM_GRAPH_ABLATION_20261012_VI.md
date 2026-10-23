# Ablation về sự cần thiết của LLM trong dynamic graph pipeline

**Ngày:** 12/10/2026  
**Dataset:** Default  
**Kết luận:** LLM hiện không cần thiết cho main pipeline. Phần graph tạo gain trong các experiment headline vốn đã bắt nguồn từ empirical mining; pipeline train-only, không API vẫn tạo được graph gần như tương đương và guidance vẫn mạnh.

## 1. Câu hỏi kiểm tra

Các kết quả trước có thể được giải thích bởi hai thành phần:

1. LLM tạo/review semantic graph;
2. dynamic graph message guidance trong reverse diffusion.

Ablation này tách hai thành phần bằng cách tạo graph mới chỉ từ train data, không gọi API, không đọc test và không sử dụng quyết định hoặc description của LLM.

## 2. Audit provenance của graph headline

Graph headline 8 cạnh có các đặc điểm:

- cả 8/8 cạnh có `source = empirical_mining`;
- ba cạnh có thêm provenance `multi_role_critic`, nhưng executable program đã tồn tại từ empirical mining;
- LLM chủ yếu viết lại mô tả, family và rationale cho một số cạnh;
- không có cạnh headline nào chỉ được đề xuất từ external LLM knowledge.

Do đó các kết quả headline trước đây không chứng minh incremental value của LLM.

## 3. Pipeline không LLM

### Giai đoạn 1: train-only empirical mining

Train được chia thành:

- discovery: 16.200 rows;
- counterexample: 5.400 rows;
- calibration: 5.400 rows.

Các implication được khai phá chỉ trên discovery rồi kiểm tra độc lập trên counterexample và calibration. Target và sensitive columns không được đưa vào graph guidance.

Kết quả:

- 30 candidate được khai phá;
- 30 candidate qua three-way train validation;
- không mở test file;
- không đọc API key và không gọi model.

### Giai đoạn 2: deterministic graph selection

Các gate được cố định:

- calibration support tối thiểu 300;
- lift tối thiểu 1,5;
- violation rate tối đa 0,25;
- spread giữa discovery/counterexample/calibration tối đa 0,08;
- consequent phải là categorical equality và không phải target;
- loại cycle và transitive shortcut;
- giữ tối đa 8 cạnh theo association/stability score.

Kết quả: 30 candidate → 17 eligible → 8 cạnh DAG.

## 4. Graph thuần dữ liệu được chọn

1. `PAY_3 = -2 -> PAY_4 = -2`;
2. `PAY_5 = -2 -> PAY_6 = -2`;
3. `PAY_4 = -2 -> PAY_5 = -2`;
4. `PAY_0 = -2 -> PAY_2 = -2`;
5. `PAY_2 = -2 -> PAY_3 = -2`;
6. `BILL_AMT3 = 0 -> PAY_4 = -2`;
7. `BILL_AMT4 = 0 -> PAY_5 = -2`;
8. `BILL_AMT1 = 0 -> PAY_2 = -2`.

Graph này trùng 7/8 executable programs với graph headline, Jaccard 0,778. Khác biệt duy nhất:

- data-only chọn `PAY_0=-2 -> PAY_2=-2`;
- headline chọn `BILL_AMT2=0 -> PAY_3=-2`.

Điều này cho thấy empirical miner tự khôi phục gần như toàn bộ cấu trúc headline.

## 5. Kết quả ba sampling seed trên graph thuần dữ liệu

Mỗi method dùng cùng TabDiff checkpoint, 4.096 samples và seed ghép cặp. Lower tốt hơn với Shape, Trend và graph error.

| Method | Shape err. ↓ | Trend err. ↓ | α-precision ↑ | Detection ↑ | MLE AUC ↑ | Data-graph err. ↓ |
|---|---:|---:|---:|---:|---:|---:|
| TabDiff | 1,9833 ± 0,3659 | 2,2165 ± 0,2568 | 98,4736 | 0,9504 | 0,7230 | 0,03923 ± 0,00900 |
| Data graph, fixed 3-pass | 2,0700 ± 0,1921 | 2,3941 ± 0,1329 | **98,6345** | 0,9500 | **0,7271** | **0,02331 ± 0,00494** |
| **Data graph, dual middle** | **1,9821 ± 0,2816** | 2,2324 ± 0,1015 | 98,6008 | **0,9511** | 0,7247 | 0,02958 ± 0,00199 |
| Data-graph final repair | 1,9521 ± 0,3127 | **2,2021 ± 0,1827** | 98,5017 | 0,9500 | 0,7247 | 0,03471 ± 0,00559 |

Diễn giải:

- fixed 3-pass giảm data-graph error **40,6%** so với TabDiff và tốt hơn final repair trên cả ba seed;
- fixed 3-pass đổi lại bằng Shape/Trend kém hơn trung bình;
- dual-middle giảm graph error **24,6%**, giữ Shape gần như nguyên vẹn, đồng thời cải thiện nhẹ α-precision, detection và MLE AUC;
- dual-middle là operating point phù hợp hơn nếu fidelity là điều kiện chính.

## 6. Đánh giá chéo để tránh self-evaluation

Cùng synthetic outputs được chấm trên cả data-only graph và headline graph.

| Sampling method | Eval graph | Graph err. ↓ | Gain so với TabDiff |
|---|---|---:|---:|
| TabDiff | Data-only | 0,03923 | – |
| Data graph, fixed 3-pass | Data-only | **0,02331** | 40,6% |
| Data graph, dual-middle | Data-only | 0,02958 | 24,6% |
| TabDiff | Headline | 0,05380 | – |
| Data graph, fixed 3-pass | Headline | **0,02844** | 47,1% |
| Data graph, dual-middle | Headline | 0,04269 | 20,7% |

Data-only guidance vẫn cải thiện mạnh trên graph không dùng để guidance. Do hai graph chia sẻ 7/8 cạnh, đây không phải external semantic generalization hoàn toàn, nhưng đủ loại bỏ giải thích rằng gain chỉ do tự chấm trên graph riêng.

Headline guidance có graph error thấp hơn data-only guidance, nhưng không thể quy gain này cho LLM:

- toàn bộ headline programs cũng được empirical mining phát hiện;
- hai graph chỉ khác một cạnh;
- headline fixed guidance có Shape/Trend trung bình xấu hơn data-only fixed guidance.

## 7. Độ ổn định của graph discovery split

Pipeline no-LLM được chạy lại với ba train partition seeds.

- số cạnh chung của cả ba graph: 6/8;
- pairwise overlap: 6/8, 7/8 và 7/8;
- toàn bộ graph đều phục hồi temporal chain `PAY_0/PAY_2 -> PAY_3 -> PAY_4 -> PAY_5 -> PAY_6`;
- khác biệt chủ yếu nằm ở bill-entry edge.

Graph discovery tương đối ổn định nhưng chưa hoàn toàn deterministic đối với entry edges. Đây là uncertainty nên được report, không nên ẩn bằng cách chỉ chọn một graph đẹp.

Hai alternate-split graphs cũng được đưa qua sampler với cùng sampling seed:

| Train partition seed | Method | Shape err. ↓ | Trend err. ↓ | Graph err. ↓ |
|---:|---|---:|---:|---:|
| 20261006 | TabDiff | 2,4058 | 2,5072 | 0,04750 |
| 20261006 | Fixed message | **2,2863** | **2,3008** | 0,02474 |
| 20261006 | Dual-middle | 2,2834 | 2,3212 | **0,02241** |
| 20261007 | TabDiff | 2,4058 | 2,5072 | 0,04353 |
| 20261007 | **Fixed message** | **2,2960** | **2,3487** | **0,02666** |
| 20261007 | Dual-middle | 2,3082 | 2,3704 | 0,02742 |

Fixed message giảm graph error lần lượt **47,9%** và **38,7%**, đồng thời cải thiện Shape/Trend trên cả hai alternate partitions. Cả fixed và dual-middle đều tốt hơn final repair về graph error ở hai run này.

## 8. Quyết định framing

### Main pipeline nên bỏ LLM

Pipeline chính đề xuất:

1. train-only empirical implication mining;
2. counterexample và calibration verification;
3. deterministic DAG selection;
4. row/timestep-dependent probabilistic active graph;
5. direct clean-logit message guidance trong reverse diffusion.

Contribution chính trở thành:

> Trajectory-conditioned probabilistic message passing over train-verified dependency graphs for tabular diffusion.

Framing này sạch hơn vì:

- không phụ thuộc model/API vendor;
- không có chi phí token hoặc reproducibility risk;
- graph source và selection hoàn toàn audit được;
- ablation hiện tại cho thấy mechanism, không phải LLM, là nguồn gain chính.

### Vai trò còn lại của LLM

LLM chỉ nên là optional extension hoặc appendix cho:

- external/domain constraints không thể suy ra từ train;
- semantic naming/explanation;
- cold-start khi support quá ít;
- user-provided natural-language policies.

Không nên để “LLM-generated graph” trong main claim nếu chưa có benchmark mà external knowledge tạo incremental gain ngoài empirical miner.

## 9. Giới hạn còn lại

1. Kết quả hiện mới trên Default và categorical consequents.
2. Hai graph chia sẻ phần lớn cạnh; chưa kiểm tra external unseen constraints.
3. Fixed guidance vẫn làm fidelity xấu trên hai sampling seed.
4. Graph selection thresholds hiện được chọn cho exploratory analysis; cần validation protocol trước final test.
5. Temporal chain trong Default khá rõ; cần Shoppers hoặc dataset khác để chứng minh phương pháp không chỉ khai thác một schema đặc biệt.

## 10. File và jobs

- Train-only miner: `scripts/mine_empirical_bank_no_llm.py`
- Deterministic selector: `scripts/select_empirical_graph_no_llm.py`
- Fully regenerated bank: `outputs/no_llm_empirical_bank_e2e_20261012/`
- Fully regenerated graph: `outputs/no_llm_empirical_graph_e2e_20261012/`
- Three-seed message jobs: `1070330`
- Headline graph matched jobs: `1070331`
- Cross evaluation: `1070332`
- Split-stability graph discovery: `1070340`
- Alternate-split guidance: jobs `1070342`, `1070343`
