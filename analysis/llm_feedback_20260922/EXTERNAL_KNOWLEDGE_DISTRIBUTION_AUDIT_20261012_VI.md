# Audit: external knowledge có cải thiện chất lượng phân phối hay không?

Ngày cập nhật: 2026-10-12.

## 1. Kết luận hiện tại

Nhận xét rằng dự án chỉ đánh giá bằng metric tự xây là chưa hoàn toàn đúng: các run
chính đã có `Shape`, `Trend`, `alpha-Precision`, `beta-Recall`, `C2ST`, `TSTR-AUC`
và `DCR`. Đây đều là metric phổ biến trong TabDiff, TabSyn, SCTab-Diff và các
framework đánh giá synthetic tabular data. Tuy nhiên, phần phê bình quan trọng là
đúng: **dynamic graph hiện chưa cải thiện các metric chuẩn một cách nhất quán**.

Kết quả ba seed đang được đóng băng cho bảng chính:

| Dataset | Method | Shape error ↓ | Trend error ↓ | alpha-Precision ↑ | beta-Recall ↑ | TSTR-AUC ↑ |
|---|---|---:|---:|---:|---:|---:|
| Default | TabDiff | 2.0030 | 2.2268 | 98.4531 | 52.5347 | 0.7227 |
| Default | Dynamic trajectory | 2.0305 | 2.2843 | 98.5426 | 52.3958 | 0.7265 |
| Shoppers | TabDiff | 1.5925 | 1.7499 | 98.4801 | 48.7099 | 0.8984 |
| Shoppers | Dynamic + residual repair | 1.6019 | 1.8242 | 98.0219 | 48.9589 | 0.8945 |
| Diabetes | TabDiff | 1.8570 | 8.5872 | 91.5137 | 53.7495 | 0.6535 |
| Diabetes | Dynamic + residual repair | 1.8684 | 8.7238 | 91.7361 | 53.7174 | 0.6500 |

Diễn giải:

- Default có AUC tăng `0.0038`, nhưng Shape, Trend và beta-Recall giảm nhẹ.
- Shoppers giảm ở Shape, Trend, alpha-Precision và AUC; chỉ beta-Recall tăng nhẹ.
- Diabetes tăng alpha-Precision, nhưng Shape, Trend và AUC giảm nhẹ.
- Do đó chưa thể claim external knowledge làm generator tốt hơn về phân phối chung.
  Claim hiện tại chỉ được hỗ trợ trên constraint-profile fidelity và một số mechanism
  probes về trajectory-dependent topology.

Audit evaluator cũng phát hiện `SDMetrics LogisticDetection` dùng shuffled
cross-validation nhưng không đặt `random_state`; vì vậy cùng một file có thể nhận
C2ST hơi khác nhau nếu được chấm nhiều lần trong một process. Common evaluator đã
được sửa để reset seed trước metric này. C2ST cũ chỉ nên đọc ở mức multi-seed, không
diễn giải chênh lệch một run rất nhỏ.

Một run Shoppers cũ với graph nhỏ hơn cho thấy tín hiệu dương nhưng chưa ổn định:
weak middle guidance giảm Shape error từ `1.3074` xuống `1.2968`, Trend error từ
`1.5476` xuống `1.5343`, và tăng AUC từ `0.9054` lên `0.9080`. Khi guidance mạnh
hơn, Trend và AUC có thể tăng nhưng Shape và coverage giảm. Điều này chỉ ra rằng
**graph selection và intervention strength**, không chỉ cơ chế injection, đang là
nút thắt.

## 2. Vì sao constraint tốt chưa chắc làm metric phân phối tốt hơn?

Metric phân phối toàn cục đo khoảng cách giữa `p_synthetic(x)` và `p_real(x)`. Nếu
generator đang học khá đúng cả các exception và các quan hệ xác suất, việc ép một
soft rule về satisfaction gần 100% sẽ làm thay đổi chính phân phối cần khớp. Khi đó
validity theo rule tăng nhưng Trend, coverage hoặc utility giảm là kết quả hợp lý.

Vì vậy phải tách ba trường hợp:

1. **Unconditional generation:** external knowledge chỉ nên sửa những quan hệ thật
   sự sai so với train/held-out distribution. Shape, Trend, alpha/beta, C2ST và TSTR
   là metric chính.
2. **Conditional generation:** khi user yêu cầu một cohort khác phân phối gốc, phải
   so với real held-out subset thỏa condition, không so với toàn bộ real table.
3. **Logical consistency:** chỉ meaningful nếu rule có semantic ground truth hoặc
   đã được train-validation chặt; metric này không được thay thế distribution và
   utility metrics.

## 3. Bài học từ các phương pháp external-knowledge gần nhất

### 3.1 SCTab-Diff: học denoiser có điều kiện theo prior

[SCTab-Diff](https://arxiv.org/html/2609.16069) dùng schema và textual description
để LLM đề xuất intra-column semantics và inter-column symbolic rules, kiểm chứng
trên train, encode chúng thành semantic tokens, rồi condition score network bằng
cross-attention trong quá trình train và denoise. Đây không phải post-hoc guidance.

Trên hai dataset trùng với dự án:

| Dataset | Method | Shape ↑ | Trend ↑ | Semantic accuracy ↑ |
|---|---|---:|---:|---:|
| Default | TabDiff | 98.37 | 96.46 | 99.97 |
| Default | SCTab-Diff | 98.57 | 97.89 | 100.00 |
| Shoppers | TabDiff | **97.94** | 97.88 | 99.19 |
| Shoppers | SCTab-Diff | 97.89 | **98.00** | 99.98 |

Bài học quan trọng là gain phân phối có thật nhưng nhỏ và không đồng nhất: Shoppers
Shape còn giảm `0.05` điểm. Phần đáng thử ở dự án là train-time prior regularization
hoặc residual condition adapter; không nên kỳ vọng inference guidance tự tạo gain
lớn trên Shape/Trend.

### 3.2 LLM-TabLogic và TabKG: bỏ biến deterministic trước khi train

[LLM-TabLogic](https://arxiv.org/html/2503.02161) và
[TabKG](https://arxiv.org/html/2605.26823) không yêu cầu diffusion học lại các
quan hệ deterministic. Graph đã verify được chuyển thành DAG; chỉ các root/independent
columns được sinh, còn dependent columns được reconstruct bằng lookup, formula hoặc
temporal mapping theo topological order.

TabKG so trực tiếp với TabSyn cùng latent-diffusion family. Kết quả paper cho thấy
trade-off, không phải dominance tuyệt đối:

| Dataset | Metric | TabSyn | TabKG | Thay đổi |
|---|---|---:|---:|---:|
| Retail | Density | 96.38 | 96.46 | +0.08 |
| Retail | Pairwise correlation | **94.81** | 92.52 | -2.29 |
| Retail | alpha-Precision | 98.48 | **99.15** | +0.67 |
| Retail | beta-Recall | 22.62 | **27.85** | +5.23 |
| Retail | TSTR-AUC | 71.55 | **71.90** | +0.35 |
| Purchasing | Density | **98.61** | 98.14 | -0.47 |
| Purchasing | Pairwise correlation | **98.63** | 92.34 | -6.29 |
| Purchasing | alpha-Precision | 96.73 | **97.29** | +0.56 |
| Purchasing | beta-Recall | 15.30 | **17.32** | +2.02 |
| Purchasing | TSTR-AUC | 62.26 | **63.45** | +1.19 |

Hai kết luận áp dụng trực tiếp:

- compression/decompression phù hợp với exact functional dependencies, không phù
  hợp với soft relations có 8–24% exception như graph Default hiện tại;
- ngay cả khi logical consistency tăng mạnh, pairwise distribution metric vẫn có
  thể giảm. Do đó paper của mình không nên hứa mọi fidelity metric cùng tăng.

### 3.3 LCATS: verify rồi repair với objective có trọng số

[LCATS](https://github.com/SJTU-DMTai/LCATS) dùng LLM propose executable constraints,
kiểm chứng với dữ liệu, TabSyn sinh bảng, sau đó diagnosis và weighted repair. Run
official local trên Insurance giảm any-violation từ `0.0822` xuống `0.0017` và KS từ
`0.0359` xuống `0.0318`. Tuy nhiên exact row edit rate bằng `1.0`; normalized numeric
L1 nhỏ (`0.00486`) cho thấy binary edit rate phải đi kèm edit magnitude.

Phần đáng chuyển sang action layer là: tạo nhiều correction candidates, chấm
constraint feasibility, train-density và edit cost, rồi chọn thay vì luôn project
vào boundary. Dự án đã có density/edit scorer; benchmark mới kiểm tra lại scorer
này nhưng chỉ trên exact verified edges.

### 3.4 SAGE: value-aware sparse dependency thay vì dense context

[SAGE, ACL 2026](https://aclanthology.org/2026.acl-long.174/) discretize numeric
features thành tối đa 16 pseudo-features, categorical values thành binary
pseudo-features, rồi tính mutual information. Khi autoregressive generation, Feature
Selector bỏ context không liên quan; Logit Correction đổi logits theo mức informative
của context hiện tại. SAGE cải thiện downstream utility so với các LLM generators và
cho thấy hai cơ chế phù hợp với constraint khác nhau; Feature Selector tốt hơn trên
Adult semantic relation, Logit Correction tốt hơn trên Housing spatial constraint.

Điểm có thể mượn không phải generator LLM, mà là **value-aware sparse context** cho
action selection: khi sửa một dependent column, chỉ dùng các current-value states có
mutual information cao để tìm conditional prototype. Đây là biến thể được submit
trong benchmark mới.

## 4. Thực nghiệm transfer mới

### 4.1 Job `1069025`: post-generation mechanisms trên cùng TabDiff sample

Ba dataset: Default, Shoppers, Diabetes. Không dùng held-out để select edge hoặc
chọn action.

Các phương pháp:

1. `LCATS-weighted-repair proxy`: chỉ exact verified edge; chọn consequent repair,
   alternate endpoint hoặc abstention bằng train-kNN density và edit cost.
2. `TabKG-kNN-reconstruct proxy`: với violated exact edge, tìm train rows active và
   valid gần nhất rồi reconstruct requirement columns; không copy cả row.
3. `SAGE-sparse-reconstruct proxy`: giống trên nhưng context cho nearest-neighbour
   được chọn theo mutual information của current value states.
4. Hai policy: chỉ semantic actuators, và toàn bộ exact verified actionable edges.

Default hiện không có exact dependency trong graph chính; kết quả đúng của các
proxy này phải là abstention. Đây là negative control quan trọng, tránh dùng
TabKG-style deterministic reconstruction cho soft payment-history relations.

Output sau khi hoàn tất:

`outputs/external_knowledge_{default,shoppers,diabetes}_1069025_{0,1,2}/RESULTS.md`

Kết quả Shoppers đầu tiên đã hoàn tất. Chỉ `33/4096` rows (`0.78%`) cần sửa trên
sáu exact actuator edges:

| Method | Shape error ↓ | Trend error ↓ | alpha ↑ | beta ↑ | TSTR-AUC ↑ | Held-out graph error ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Unconditional | 1.7537 | **1.4924** | **98.8288** | 48.5775 | **0.8980** | 0.01934 |
| LCATS-weighted repair proxy | **1.7297** | 1.5100 | **98.8288** | 48.5840 | 0.8920 | 0.01789 |
| TabKG-kNN reconstruct proxy | 1.7378 | 1.5158 | 98.8233 | **48.5872** | 0.8925 | **0.01788** |
| SAGE-sparse reconstruct proxy | 1.7392 | 1.5143 | 98.8233 | **48.5872** | 0.8918 | **0.01788** |

Đây là negative/mixed result có giá trị: exact-only correction giảm Shape error
khoảng `0.8–1.4%` tương đối và giảm graph error, nhưng làm Trend error tăng khoảng
`1.2–1.6%` và AUC giảm khoảng `0.0055–0.0062`. Sparse MI context chưa thắng
all-context reconstruction. Vì chỉ 33 rows bị sửa mà AUC dao động khá rõ, kết quả
AUC một seed này phải được xem là diagnostic; không chạy multi-seed trừ khi hai
dataset còn lại cho tín hiệu cùng chiều.

### 4.2 Job `1069028`: train-time semantic-prior regularization proxy

Hai dataset có exact relations: Shoppers và Diabetes. Mỗi dataset có một continued
fine-tuning control và một graph-regularized model. Cùng checkpoint, seed, số epoch,
learning rate và sample count; chỉ khác graph loss ở middle-noise states. Đây là
proxy cho nguyên lý train-time conditioning của SCTab-Diff, không phải reproduction
cross-attention architecture.

Output sau khi hoàn tất:

`outputs/semantic_prior_proxy_{dataset}_{continue,semantic_prior}_1069028_*/evaluation/RESULTS.md`

Job `1069032` đã được đặt dependency để tự tổng hợp cả hai nhánh sau khi jobs trên
kết thúc, không cần polling. Output dự kiến:

`outputs/external_knowledge_aggregate_1069032/RESULTS.md`

## 5. Điều kiện quyết định

Một external-knowledge mechanism chỉ được đưa vào main method nếu thỏa cả ba:

1. giảm Trend error hoặc tăng alpha-Precision/TSTR-AUC so với paired TabDiff hoặc
   continued-training control;
2. không làm Shape error tăng quá `0.10` điểm và không làm beta-Recall giảm quá
   `0.25` điểm ở operating point chọn trước;
3. gain giữ được trên ít nhất hai dataset trước khi chạy multi-seed.

Nếu reconstruction thắng nhưng train-time regularization không thắng, external
knowledge nên nằm ở typed action layer. Nếu regularization thắng, method nên chuyển
từ sampler-only sang backbone + trajectory controller. Nếu cả hai không thắng,
đóng góp hợp lý hơn là **Pareto-preserving constraint control**: tăng verified
semantic consistency với bounded distribution/utility degradation, không claim cải
thiện unconditional distribution fidelity.

## 6. Metric protocol cần dùng từ đây

Main table nên luôn có:

- `Shape`: trung bình similarity của marginal numeric KS và categorical TV;
- `Trend`: similarity của pairwise correlations/contingencies;
- `alpha-Precision`: synthetic samples có nằm trong support của real data;
- `beta-Recall`: synthetic data có phủ được support/diversity của real data;
- `C2ST`: real và synthetic khó phân biệt đến mức nào;
- `TSTR-AUC` hoặc RMSE: train trên synthetic, test trên real held-out;
- `DCR`: privacy diagnostic;
- independent semantic validity, nhưng chỉ trên verified/annotated rules;
- runtime, GPU time, action count và API cost.

Với user-conditioned generation, cùng các metric trên phải được tính đối với real
held-out cohort phù hợp condition. Không dùng unconditional real distribution làm
reference cho một distribution đã được user yêu cầu thay đổi.
