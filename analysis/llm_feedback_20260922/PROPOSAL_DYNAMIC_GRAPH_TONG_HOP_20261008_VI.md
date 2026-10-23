# Proposal tổng hợp: Dynamic Constraint Graph cho mixed-type tabular diffusion

**Ngày cập nhật:** 08-10-2026  
**Backbone thực nghiệm hiện tại:** TabDiff  
**Dataset chính:** Shoppers, Default, Diabetes  
**Trạng thái:** pipeline end-to-end đã chạy; đã có bằng chứng cho dynamic topology trên hai dataset, nhưng chưa đủ để claim state of the art.

---

## 1. Tóm tắt dự án

Dự án nghiên cứu cách dùng một **constraint graph thay đổi theo từng row và theo reverse-diffusion trajectory** để cải thiện semantic consistency của dữ liệu bảng synthetic.

Thay vì áp dụng một tập rule cố định cho tất cả row và mọi timestep, phương pháp gồm bốn tầng:

1. xây source graph từ tri thức ngoài, schema, thống kê train và predefined hypotheses;
2. kiểm chứng và biên dịch graph sang typed DSL thực thi được;
3. tại mỗi predicted-clean row \(\hat{x}_{0,i}(t)\), suy ra trạng thái riêng của từng edge;
4. khi một violation đủ đáng tin, chọn typed action và đưa bản \(\hat{x}_0\) đã sửa trở lại reverse update.

Đóng góp dự kiến không phải là “constraint guidance trong diffusion” nói chung. Harpoon và CTDF đã can thiệp constraint trong reverse diffusion; SCTab-Diff cũng đã condition denoising bằng prior được LLM trích xuất. Khoảng trống mà dự án theo đuổi hẹp và rõ hơn:

> **Graph topology và constraint state thay đổi theo từng predicted-clean row trong trajectory; chỉ những chuyển trạng thái đủ đáng tin mới được phép phát action.**

Bằng chứng hiện tại cho thấy hướng này có tín hiệu thật nhưng không phổ quát:

- dynamic-middle tốt hơn frozen-state trên Shoppers và Diabetes trong uncapped setting;
- effect bootstrap trên hai dataset này có khoảng tin cậy không cắt 0;
- correct row–edge assignment vẫn tốt hơn shuffled assignment trên Shoppers khi action budget bằng nhau;
- early state chưa đủ đọc, middle là vùng sớm nhất tạo lợi ích rõ;
- nhưng final repair vẫn rất mạnh, late can thiệp tốt hơn middle trên Diabetes nếu chỉ tối ưu validity;
- dynamic topology không luôn action-efficient;
- flip-rate, fixed hysteresis và graph-weight reranking đơn giản chưa tạo một controller tổng quát.

Vì vậy project hiện đã đủ để phát triển thành một research project, nhưng claim cuối phải dựa vào **transition-certified dynamic topology**, không dựa vào số violation cuối cùng hay vào việc dùng LLM đơn thuần.

---

## 2. Bài toán nghiên cứu

### 2.1. Input và output

Cho training table \(D_{train}\), một mixed-type diffusion generator, schema/data dictionary và một tập semantic relations có provenance khác nhau. Cần sinh synthetic table vừa:

- giữ marginal và joint fidelity;
- giữ downstream utility và coverage;
- giảm semantic violations;
- không ép empirical exceptions thành hard invariants;
- không chỉnh sửa row không liên quan;
- có chi phí inference hợp lý.

### 2.2. Vì sao static graph chưa đủ

Một edge có thể thay đổi trạng thái trong trajectory vì ba nguyên nhân:

1. **Readability:** các cột liên quan chưa ổn định nên chưa thể đánh giá edge.
2. **Applicability:** antecedent chưa đúng ở state hiện tại, nhưng có thể đúng ở state sau.
3. **Satisfaction:** edge applicable nhưng requirement đang satisfied hoặc violated.

Với row \(i\), edge \(e\) và timestep \(t\):

\[
s_{i,e,t}\in\{\text{unreadable},\text{inactive},\text{satisfied},\text{violated},\text{uncertain}\}.
\]

Active graph là:

\[
G_{i,t}=\{e\in G_0:\operatorname{Readable}(e,t)\land
\operatorname{Applicable}(e,\hat{x}_{0,i}(t))\land
\neg\operatorname{Exception}(e,\hat{x}_{0,i}(t))\}.
\]

Graph này thay đổi cả theo row và trajectory. Đây là đối tượng chính của paper.

### 2.3. Câu hỏi nghiên cứu đã thu hẹp

> So với static/frozen graph, differentiable rule guidance và final repair, liệu transition-certified row-wise graph có dùng middle trajectory để giảm semantic error mà vẫn giữ fidelity, utility và action efficiency hay không?

Ba phần phải được chứng minh riêng:

1. predicted-clean state có đủ đọc ở middle trajectory;
2. row- và trajectory-specific topology chứa thông tin ngoài static graph;
3. thông tin đó tạo gain sau khi kiểm soát action budget, fidelity và strong post-processing baselines.

---

## 3. Motivation thực nghiệm

### 3.1. Predicted-clean graph không đáng tin ở early steps

Probe 2,048 rows, 50 reverse steps và 12 edges cho thấy:

| Timestep | Edge precision | Edge recall | Edge accuracy |
| --- | ---: | ---: | ---: |
| Early, \(t>0.75\) | 0.6688 | 0.2416 | 0.9272 |
| \(0.50<t\le0.75\) | 0.8162 | 0.6976 | 0.9657 |
| \(0.25<t\le0.50\) | 0.9216 | 0.9311 | 0.9895 |
| \(0.10<t\le0.25\) | 0.9874 | 0.9926 | 0.9986 |
| Late | 0.9999 | 1.0000 | 1.0000 |

Early accuracy nhìn cao vì phần lớn row–edge pair là negative; recall mới cho thấy active graph thật sự bị bỏ sót nhiều. Middle region là điểm đầu tiên precision và recall cùng đủ cao, trong khi sampler vẫn còn thời gian phản ứng.

### 3.2. Static repair đạt validity nhưng không giải quyết selection

CuTS và LCATS cho thấy satisfaction gần 1.0 hoặc violation gần 0 có thể đạt bằng fine-tuning, rejection hoặc repair. Trong controlled policies của dự án, static repair cũng đạt satisfaction 1.0 nhưng thay đổi 97.44% valid exceptions trên Default và 64.91% trên Shoppers nếu không có subgroup applicability.

Target vì vậy không nên là “đạt zero violation”. Bài toán quan trọng hơn là edge nào applicable cho row nào, lúc nào edge đủ đáng tin, action nào nên chọn, và chi phí phân phối/utility của action đó.

### 3.3. Middle trajectory có signal nhưng không luôn là optimum cuối

Middle dynamic–frozen gap rõ hơn early và thường rõ hơn late. Tuy nhiên late intervention có absolute validity tốt nhất trên Diabetes. Cách diễn giải đúng là:

> Middle là vùng sớm nhất graph state đủ đọc và graph evolution còn tạo khác biệt; late có thể dễ sửa hơn vì sample gần hoàn tất nhưng khi đó lợi thế so với final repair giảm mạnh.

---

## 4. Vị trí so với related work

| Work | Constraint/knowledge đi vào đâu | Có quan sát state trung gian? | Có topology riêng theo row và timestep? | Khoảng trống còn lại |
| --- | --- | --- | --- | --- |
| TabSyn | Diffusion trong latent liên tục của VAE | Không phải contribution chính | Không | Backbone/representation, không giải quyết dynamic semantics |
| TabDiff | Mixed numerical/categorical diffusion với feature-wise schedules | Có thể truy cập predicted-clean estimate | Không | Backbone chính hiện tại |
| Harpoon | Loss khả vi tác động trong sampling | Loss thay đổi theo \(\hat{x}_0(t)\) | Constraint request cố định | Không giải quyết semantic applicability/exception topology thay đổi |
| CTDF | Feasibility operator ở từng reverse step, mixed-type | Có | Constraint program cố định | Per-step constraint enforcement không còn là novelty |
| SCTab-Diff | LLM prior được condition vào reverse denoising | Context hiện tại ảnh hưởng score | Prior/specification cơ bản cố định | Chưa tập trung vào verified row-edge transition và abstention |
| LCATS | LLM discovery, diagnosis và post-generation repair | Không | Diagnostic bucket theo output | Strong baseline cho graph discovery và repair |
| Zhao et al. | Agent sinh executable rules, counterexample revision, deterministic repair | Không | Không | LLM rule discovery/verification riêng không đủ novelty |
| EVS | LLM noisy oracle, train verification và constraint selection | Không | Không | LLM extraction/filtering riêng không đủ novelty |
| SAGE | Value-conditioned dependency khi LLM autoregressively sinh row | Theo prefix/value hiện tại | Row-dependent, không phải reverse-diffusion trajectory | Threat gần nhất cho row-wise dependency; temporal diffusion state vẫn khác |
| PromptLoop | MLLM đọc denoised image estimate và cập nhật prompt | Có | Feedback thay đổi theo image trajectory | Ý tưởng tổng quát đã tồn tại ở image; cần tabular-specific problem và evidence |
| PPAD | Preview/lookahead, chẩn đoán rồi rollback/resample | Có | Có feedback theo preview | Gợi ý transition confidence và rollback; cảnh báo preview quá nhiễu |
| Free²Guide/TTGen | Black-box reward hoặc clean-latent feedback trong visual diffusion | Có | Có | Gợi ý branch scoring; không giải quyết typed mixed-table semantics |

Novelty boundary:

- không claim LLM feedback trong denoising là mới nói chung;
- không claim per-step constraint enforcement là mới;
- không claim LLM sinh executable rules là mới;
- không claim row-dependent dependencies nói chung là mới vì SAGE đã gần;
- claim dự kiến là **verified typed graph có topology và constraint state thay đổi theo row trong predicted-clean diffusion trajectory, cùng transition certification và typed intervention**.

---

## 5. Pipeline sinh source graph

### 5.1. Ba nguồn tri thức

1. **External/domain knowledge:** foundation model đọc schema, data dictionary, context và categorical vocabulary.
2. **Empirical structure:** deterministic miner tìm directional implications và conditional dependencies chỉ trên discovery split.
3. **Predefined hypotheses:** seed relations có provenance rõ, được coi là hypotheses chứ không mặc định đúng.

LLM không nhận raw real rows. Payload chỉ gồm schema/metadata và aggregate statistics.

### 5.2. Propose–execute–refine loop

Training data được chia 60% discovery, 20% counterexample và 20% blind calibration. Protocol:

1. nhiều proposer roles tạo candidate edges;
2. compiler chuyển natural language sang typed DSL;
3. executor kiểm tra kiểu dữ liệu, cột lạ, tautology, support, vacuity, target leakage và sensitive attributes;
4. counterexample split cung cấp aggregate evidence cho critic;
5. critic `keep/revise/reject`;
6. semantic judge gán scope;
7. blind calibration gán empirical hardness;
8. heldout chỉ được mở sau khi graph frozen để audit shift.

### 5.3. Typed graph DSL

Mỗi hyperedge gồm:

- `applicability`: conjunction xác định antecedent;
- `requirements`: một hoặc nhiều conditions phải thỏa;
- `exceptions`: OR giữa exception groups, AND trong từng group;
- column–literal hoặc column–column atoms;
- relation family, source, scope, hardness và confidence;
- discovery/calibration/heldout support và violation statistics;
- `usage_role`: `actuate`, `score_only` hoặc `monitor`.

Ví dụ: `Administrative == 0 → Administrative_Duration == 0`, `BounceRates <= ExitRates`, medication-state conjunctions trên Diabetes, hoặc codebook relation `education → education.num` ở scope `dataset_semantic`.

### 5.4. Kết quả graph discovery

Budgeted Shoppers run dùng 110,513/165,000 tokens và 15 API calls:

| Graph | Proposed | Retained | Hard | Near-hard | Soft | Families | Stable heldout ≤ 0.05 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Judged single-pass | 14 | 1 | 0 | 1 | 0 | 1 | 1 |
| Multi-stage LLM | 46 | 8 | 5 | 3 | 0 | 3 | 8 |
| Empirical-only | 30 | 28 | 5 | 9 | 14 | 1 | 28 |
| Unified | 76 | 36 | 10 | 12 | 14 | — | 36 |

Multi-stage graph tạo stable relation families và column comparisons. Tuy nhiên single-pass compiler chỉ biên dịch đúng 1/14 descriptions, nên chưa được dùng bảng này để claim multi-stage hơn strong single-pass.

Diabetes run dùng 19 API calls và 199,622 tokens:

- unified graph: 26 edges, 4 families, 19 endpoint patterns;
- catalog cuối: 22 edges;
- 6 `actuate`, 15 `score_only`, 1 `monitor`;
- heldout violation shift trung bình 0.0010;
- external/hybrid retained actuator chưa đủ để chứng minh external knowledge là cần thiết.

### 5.5. Bài học từ graph acquisition

- Retained/proposed là protocol yield, không phải precision.
- Heldout stability không đồng nghĩa semantic truth.
- Source graph không được đưa thẳng vào sampler.
- Graph verification nên rộng; graph actuation phải nhỏ hơn và có action semantics rõ.
- Graph acquisition hiện là supporting contribution; main novelty vẫn nằm ở trajectory mechanism.

---

## 6. Từ source graph đến dynamic graph

### 6.1. Graph curation

Mỗi edge được gán vai trò:

- `actuate`: semantics và correction direction đủ rõ;
- `score_only`: hữu ích để đánh giá branch nhưng chưa an toàn để trực tiếp sửa;
- `monitor`: chỉ audit, không tác động sampler.

Shoppers graph đầu tiên đi từ 8 source edges → 7 verifier edges → 3 actuator edges. Expanded catalog sau đó được dùng để tránh benchmark chỉ có ba rules. Diabetes có 6 actuator edges đa cột và 15 score-only edges.

### 6.2. Readability schedule

Với mỗi edge, trajectory probe so active/violation state của \(\hat{x}_0(t)\) với final generated row. Edge chỉ được can thiệp sau cutoff đạt precision/recall tối thiểu.

| Shoppers actuator | Cutoff sớm nhất | Synthetic/heldout violation |
| --- | ---: | ---: |
| Administrative count → duration | 0.571 | 0.0031 / 0.0000 |
| Informational count → duration | 0.612 | 0.0043 / 0.0000 |
| BounceRates ≤ ExitRates | 0.286 | 0.0220 / 0.0154 |

Kết quả support edge-specific readability, nhưng ablation sau đó cho thấy common cutoff đôi khi cạnh tranh hoặc tốt hơn. Edge-specific cutoff hiện là implementation option, chưa phải independent contribution.

### 6.3. Dynamic state và controls

- `dynamic`: recompute applicability/satisfaction từ row hiện tại;
- `frozen-state`: đóng băng state khi bước vào window;
- `periodic`: refresh mỗi \(k\) steps;
- `hysteresis`: commit state sau \(k\) observations giống nhau;
- `shuffled-row`: giữ edge support nhưng phá row assignment;
- `oracle-final`: diagnostic dùng final topology, không phải deployable method.

Các controls tách timing, topology evolution và row–edge assignment.

---

## 7. Action layer và injection

### 7.1. Candidate actions hiện tại

Khi một edge violated, action layer tạo:

1. `direct_requirements`: sửa requirement trực tiếp;
2. `alternate_endpoint`: với column–column relation, sửa endpoint còn lại;
3. `deactivate_antecedent`: sửa applicability để edge không còn applicable;
4. `abstain`: không sửa.

Candidate được chấm theo:

\[
J(a)=100\,\mathbf{1}[\text{edge vẫn violated}]
+\lambda_d D_{kNN}(a,D_{train})
+\lambda_e\operatorname{EditCost}(a,x)
+\lambda_g\operatorname{GraphPenalty}(a).
\]

`GraphPenalty` chỉ phạt violation mới trên score-only edges, không ép sửa empirical exception có sẵn.

### 7.2. Calibration gate

Hard relations có thể sửa toàn bộ violations. Near-hard/soft relations chỉ sửa phần vượt quá violation rate trên calibration split, ưu tiên severity cao. Mục tiêu là không biến probabilistic relation thành hard invariant.

### 7.3. Predicted-clean injection

Action được áp lên decoded predicted-clean row. Chỉ cells đã sửa được encode trở lại; cells khác giữ nguyên denoiser output để tránh decode–encode drift.

\[
d_t=\frac{x_t-\hat{x}'_0(t)}{\sigma_t},\qquad
x_{t-\Delta}=x_t+(\sigma_{t-\Delta}-\sigma_t)d_t.
\]

Categorical actions sửa code tương ứng trong reverse categorical update. Các biến thể đã thử gồm direct \(\hat{x}_0\) correction, branch-score-select và rollback/resampling.

### 7.4. Giới hạn action layer

Graph-wide guardrail trên Diabetes cho kết quả hoàn toàn giống khi weight bằng 0 và 2. Candidate choices cũng giống nhau. Candidate set hiện quá nghèo để graph score phân biệt.

Không nên quét thêm graph weight. Hướng đúng là sinh candidate phong phú hơn: feasible values từ conditional train neighborhoods, nhiều endpoint/multi-column branches, antecedent deactivation, downstream edge closure và abstention.

---

## 8. Kết quả end-to-end chính

### 8.1. Main 3-seed, 8,192 rows, 50 steps

| Dataset | Policy | Actuation heldout error ↓ | Overall heldout error ↓ | Actions |
| --- | --- | ---: | ---: | ---: |
| Shoppers | Dynamic current | 0.000642 ± 0.000584 | 0.019361 | 108.3 |
| Shoppers | **Dynamic stable** | **0.000609 ± 0.000549** | **0.019322** | 107.3 |
| Shoppers | Frozen state | 0.001372 ± 0.001006 | 0.019642 | 93.7 |
| Default | **Dynamic current** | **0.003677 ± 0.001588** | **0.016150** | 52.0 |
| Default | Dynamic stable | 0.003821 ± 0.002374 | 0.016251 | 46.7 |
| Default | Frozen state | 0.004011 ± 0.002603 | 0.016306 | 51.0 |

Dynamic thắng 3/3 Shoppers seeds. Default chỉ thắng 2/3 và variance lớn.

### 8.2. Dynamic-middle so với frozen-middle

Positive improvement là `frozen error − dynamic error`.

| Dataset | Improvement ↑ | Seed wins | Hierarchical bootstrap 95% CI | P(improvement > 0) |
| --- | ---: | ---: | ---: | ---: |
| Shoppers | **0.001519** | 3/3 | **[0.000994, 0.002057]** | 1.000 |
| Default | 0.001963 | 2/3 | [-0.003091, 0.007210] | 0.769 |
| Diabetes | **0.002466** | 3/3 | **[0.001270, 0.003956]** | 1.000 |

Effect được support trên Shoppers và Diabetes, chưa support trên Default. Bootstrap resample synthetic rows và sampling seeds, chưa bao phủ training-seed uncertainty.

### 8.3. Matched action budget

| Dataset | Actions dynamic/frozen | Dynamic error ↓ | Frozen error ↓ | Kết luận |
| --- | ---: | ---: | ---: | --- |
| Shoppers | 33.0 / 33.0 | **0.002498** | 0.003116 | Dynamic thắng 3/3 |
| Default | 73.3 / 73.3 | 0.017417 | **0.009903** | Dynamic thua 3/3 |
| Diabetes | 20.0 / 19.3 | 0.005213 | **0.004740** | Dynamic thua 3/3 |

Dynamic topology chưa action-efficient một cách phổ quát. Shoppers là bằng chứng dương sạch nhất; Default và Diabetes là negative controls quan trọng.

### 8.4. Row assignment control

Trên Shoppers với đúng 40 actions:

| Topology | Actuation error ↓ | Overall error ↓ | KS ↓ | Wins |
| --- | ---: | ---: | ---: | ---: |
| Correct row-dynamic | **0.003307** | **0.020994** | **0.026137** | 3/3 |
| Shuffled row-dynamic | 0.004326 | 0.021497 | 0.026265 | 0/3 |

Correct row assignment giảm actuation error khoảng 23.5%. Đây là evidence trực tiếp rằng row-specific topology chứa thông tin, không chỉ tạo nhiều actions hơn.

### 8.5. Temporal region

| Dataset | Early | Middle | Late | Kết luận |
| --- | ---: | ---: | ---: | --- |
| Shoppers, dynamic–frozen gap | Không có | Rõ | Nhỏ hơn middle | Middle support mạnh nhất |
| Default, dynamic–frozen gap | Âm/không ổn định | Dương nhưng variance cao | Gần 0 | Chưa robust |
| Diabetes, cap 20 absolute error | 0.006152 | 0.005213 | **0.003478** | Late tốt nhất về absolute validity |

Middle không được claim là timing tối ưu tuyệt đối. Claim hợp lệ là **middle là vùng sớm nhất đủ đọc và còn thể hiện khác biệt giữa evolving và frozen graph**.

### 8.6. Refresh policy

Same-node paired confirmation, unconditional hash khớp 9/9 groups:

| Dataset | Dynamic | Hysteresis-3 | Adaptive | Policy tốt nhất |
| --- | ---: | ---: | ---: | --- |
| Shoppers actuation error | **0.001783** | 0.002216 | **0.001783** | Dynamic |
| Default | 0.008438 | **0.006127** | 0.007431 | Hysteresis-3 |
| Diabetes | **0.001100** | 0.001220 | 0.001220 | Dynamic |

EMA controller dựa trên flip-rate không thắng fixed policy tốt nhất ở dataset nào. Diabetes có flip rate cao nhất nhưng dynamic vẫn tốt nhất, bác bỏ giả thuyết “nhiều flip đồng nghĩa nhiều noise”. Đây là lý do trực tiếp cho transition-certifier batch mới.

### 8.7. Fidelity và utility

| Dataset | Method | Shape ↓ | Trend ↓ | α-Precision ↑ | β-Recall ↑ | Detection ↑ | MLE AUC ↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Shoppers | TabDiff | 1.3933 | 1.8358 | 98.9079 | **49.9300** | 0.9906 | 0.9043 |
| Shoppers | Dynamic-stable | **1.3778** | 1.7967 | **98.9158** | 49.9051 | 0.9912 | **0.9108** |
| Shoppers | Final repair | 1.3792 | **1.7911** | 98.8208 | 49.9737 | **0.9924** | 0.9062 |
| Default | TabDiff | 1.7829 | 1.9035 | 97.7084 | **52.6055** | 0.9488 | 0.7395 |
| Default | Dynamic-current | **1.7792** | 1.9033 | **97.7111** | 52.5993 | 0.9478 | 0.7445 |
| Default | Final repair | 1.7800 | 1.9035 | 97.7084 | 52.6055 | **0.9543** | **0.7451** |

Không thấy fidelity/utility collapse. Tuy nhiên chênh lệch nhỏ, chưa phải SOTA evidence.

### 8.8. Robustness theo số reverse steps

Dynamic giảm actuation error so với frozen ở Shoppers và Default tại 30 và 100 steps. Overall error tốt hơn ở 3/4 settings; sai khác âm còn lại chỉ khoảng \(4\times10^{-5}\). Effect không chỉ tồn tại ở sampler 50 steps, nhưng vẫn cần independent training checkpoints.

---

## 9. Baseline status

| Baseline | Trạng thái | Kết quả liên quan | Vai trò trong paper |
| --- | --- | --- | --- |
| TabDiff | Official backbone/checkpoint, local paired runs | Unconditional reference | Main generator baseline |
| Final repair | Same graph/compiler/local | Rất mạnh, thường đạt exact validity | Baseline bắt buộc |
| Frozen graph | Same sampler/action budget | Main causal control | Tách topology evolution |
| Periodic/hysteresis graph | Same sampler | Dataset-dependent | Refresh-policy ablation |
| Best-of-N + certifier | Local | Có thể giữ validity và cải thiện distribution | Compute-matched selection baseline |
| LCATS + TabSyn | Official local rerun | Any violation 0.0822 → 0.0017; KS 0.0359 → 0.0318 | Strong discovery + post-repair reference; chưa common task |
| CuTS | Official local rerun | Năm German programs đạt satisfaction 1.0 | Constraint baseline; chưa common task |
| Harpoon | Author artifacts; local setup lỗi CUDA/PyTorch | Paper cho thấy differentiable guidance mạnh | Cần faithful common-task port hoặc ghi artifact-only |
| Harpoon-style | Local proxy | Có validity–distortion frontier | Không gọi là official reproduction |
| CTDF-style | Local numeric proxy | Feasibility cao, có distortion trade-off | Cần mixed-type faithful implementation |
| CTGAN/TVAE | Local quality baselines | Có quality table | Generator breadth, không phải mechanism control |

Chưa có common-task faithful Harpoon/CTDF/LCATS comparison, nên chưa được gọi là SOTA. Official results khác dataset hoặc constraint không được trộn thành bảng xếp hạng trực tiếp.

---

## 10. Hypothesis ledger

| Hypothesis | Trạng thái | Evidence |
| --- | --- | --- |
| Early \(\hat{x}_0\) đủ đọc | **Bác bỏ** | Early recall rất thấp |
| Middle/late-middle đủ đọc | **Ủng hộ** | Precision/recall tăng mạnh từ middle |
| Dynamic topology tốt hơn frozen | **Ủng hộ có điều kiện** | Shoppers, Diabetes uncapped; bootstrap dương |
| Dynamic action-efficient phổ quát | **Bác bỏ** | Matched budget âm trên Default/Diabetes |
| Correct row–edge assignment quan trọng | **Ủng hộ trên Shoppers** | Exact-40 action shuffled control |
| Middle luôn tốt nhất | **Bác bỏ** | Diabetes late tốt hơn middle |
| Middle là vùng sớm nhất hữu ích | **Ủng hộ** | Middle > early trên cả ba dataset |
| Refresh mọi step luôn tốt nhất | **Bác bỏ** | Default thích periodic/hysteresis |
| Flip-rate chọn được refresh policy | **Bác bỏ** | Adaptive controller không thắng; Diabetes phản ví dụ |
| Score-only reranking cải thiện action | **Bác bỏ với candidate hiện tại** | Weight 0 và 2 giống nhau |
| Mid-step luôn hơn final repair | **Bác bỏ** | Final repair thắng nhiều settings |
| Direct LLM judgement từng step đáng chi phí | **Bác bỏ trong executable setting** | NL→DSL→executor chính xác hơn, rẻ hơn |
| External LLM knowledge là cần thiết | **Chưa chứng minh** | External retained actuators còn ít |
| Multi-stage hơn strong single-pass | **Chưa chứng minh** | Single-pass compiler chưa đủ công bằng |
| Dynamic method là SOTA | **Chưa chứng minh** | Thiếu faithful baselines và second backbone/checkpoint |

---

## 11. Method đề xuất cho phiên bản paper

Tên tạm thời:

> **Trajectory-Conditioned Dynamic Constraint Graphs for Mixed-Type Tabular Diffusion**

Giữ method ở năm component.

### Component 1 — Verified source graph

LLM/statistical/predefined candidates → typed compiler → deterministic executor → counterexample refinement → scope/hardness calibration → frozen graph.

### Component 2 — Row-edge state estimator

Đọc \(\hat{x}_{0,i}(t)\) và gán `unreadable/inactive/satisfied/violated/uncertain`. Readability dùng calibrated timestep cutoff; applicability và satisfaction được thực thi bằng DSL.

### Component 3 — Transition certifier

Đây là phần method mới cần phát triển. Với state change \(s_{i,e,t-1}\ne s_{i,e,t}\), certifier ước lượng:

\[
p_{i,e,t}=P(s_{i,e,t}\text{ là chuyển trạng thái thật}\mid
\hat{x}_0(t-k:t), e, \text{column stability}, t).
\]

Chỉ commit topology change khi confidence vượt threshold; nếu không thì giữ state trước hoặc abstain. Certifier phải phân biệt genuine convergence transition, one-step preview noise, oscillation gần decision boundary và true applicability change. Không dùng global flip-rate vì đã bị bác bỏ.

### Component 4 — Typed action router

Sinh nhiều feasible corrections, chấm validity, graph side effects, train-density và edit cost, sau đó chọn Pareto-safe action hoặc abstain.

### Component 5 — Predicted-clean injection

Chỉ sửa cells được chọn trong \(\hat{x}_0(t)\), đưa lại vào numerical reverse direction và categorical reverse update. Không cần gọi LLM trong mỗi reverse step; LLM có thể nằm ở graph acquisition hoặc offline action-template generation.

---

## 12. Transition survival và certification

### 12.1. Hypotheses

- **H-TC1:** state change tồn tại qua một số observations có final-state precision cao hơn immediate refresh.
- **H-TC2:** số observations cần chờ khác nhau giữa datasets và applicability/violation state; global hysteresis không tối ưu.
- **H-TC3:** wait policy chọn trên calibration trajectory tăng transition precision trên unseen sampling seeds mà vẫn giữ đủ genuine-transition recall.

### 12.2. Thiết kế

- Shoppers, Default, Diabetes; ba sampling seeds mỗi dataset.
- 8,192 rows cho Shoppers/Default; 4,096 cho Diabetes.
- 50 steps; active window \(t\in[0.10,0.75]\).
- Quan sát applicability và violation transitions.
- Đánh giá chờ 1–5 observations giống nhau.
- Final row chỉ là evaluation label, không làm input.
- Seed nhỏ nhất mỗi dataset dùng chọn wait đạt precision 0.95; hai seed còn lại dùng evaluation.

Metrics: certified transition precision, genuine-event recall, acceptance rate, precision delta so với immediate update, tách theo state và timestep bin.

### 12.3. Kết quả calibration/evaluation split

Array `1060882[0-8]` và aggregate `1060883` đã hoàn tất `ExitCode 0:0`. Seed nhỏ nhất của mỗi dataset được dùng cho policy selection; hai seeds còn lại chỉ dùng evaluation.

| Dataset | State | Selected wait | Evaluation precision/recall | Immediate precision | Precision gain | Acceptance |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Default | Applicability | 5 | 0.8313 / 0.9140 | 0.7006 | **+0.1307** | 0.7704 |
| Default | Violation | 5 | 0.7156 / 0.9007 | 0.5574 | **+0.1582** | 0.7016 |
| Diabetes | Applicability | 5 | 0.8247 / 0.8951 | 0.6966 | **+0.1281** | 0.7561 |
| Diabetes | Violation | 5 | 0.7904 / 0.9230 | 0.5184 | **+0.2720** | 0.6053 |
| Shoppers | Applicability | 1 | 0.9655 / 1.0000 | 0.9655 | +0.0000 | 1.0000 |
| Shoppers | Violation | 5 | 0.9078 / 0.8980 | 0.6005 | **+0.3073** | 0.5940 |

Kết quả ủng hộ H-TC1 và H-TC2:

- temporal survival dự báo final correctness tốt hơn raw flip;
- violation state cần filtering mạnh hơn applicability state;
- Shoppers applicability là phản ví dụ quan trọng: chờ thêm không cần thiết;
- policy 5 observations giữ khoảng 89.8–92.3% genuine violation transitions nhưng loại 29.8–40.6% raw transition proposals.

Mục tiêu precision 0.95 chỉ đạt cho Shoppers applicability. Vì vậy kết quả chưa đủ để gọi đây là high-confidence certifier hoàn chỉnh. Nó chứng minh **survival là feature có ích**, không chứng minh threshold 5 là optimal hoặc deployable ở mọi dataset.

### 12.4. Jobs end-to-end tiếp theo

| Job | Nội dung | Trạng thái khi viết |
| --- | --- | --- |
| `1060882[0-8]` | 3 datasets × 3 seeds transition-survival probe | Completed |
| `1060883` | Calibration/evaluation aggregate | Completed |
| `1060901[0-8]` | Paired online immediate/wait-3/dynamic-wait-5/topology-wait-5/frozen-wait-5 trên ba seeds mới | Completed |
| `1060902` | Same-node parity check và aggregate | Completed |

Online batch chỉ chạy `x0_reward_calibrated` để tránh lặp lại ba mechanisms không liên quan. Với Shoppers, pre-registered policy giữ applicability dynamic nhưng chờ violation 5 observations. Với Default/Diabetes, policy chờ 5 observations cho cả topology và violation. Các policies chạy tuần tự trong cùng Slurm task để tránh GPU-architecture confound. Batch không gọi API và không phát raw real rows.

**Cập nhật sau khi jobs hoàn tất:** `1060901–1060902` hoàn tất thành công, parity đạt 9/9. Pre-registered transition policy chỉ cải thiện so với immediate trên Default, nhưng thua trên Shoppers và Diabetes. `dynamic_wait5` là variant tốt nhất trên Default (`0.004497` so với immediate `0.009669`), trong khi immediate vẫn tốt nhất trên Shoppers và Diabetes. Vì vậy H-TC3 bị bác bỏ ở dạng policy tổng quát; final-state transition precision không đủ làm objective cho online action timing.

### 12.5. Điều kiện quyết định

Implement online transition certifier nếu evaluation precision tăng nhất quán trên ít nhất hai datasets, recall không giảm quá mạnh và selected wait giải thích được failure của fixed hysteresis. Nếu survival không dự báo final correctness, không xây learned controller; dùng policy đơn giản theo dataset và hạ claim xuống selective temporal guidance.

---

## 13. Hướng thực nghiệm tiếp theo

### Ưu tiên 1 — Transition-certified online graph

Sau `1060882–1060883`, implement đúng một controller: calibrated persistence theo edge/state/timestep hoặc short lookahead agreement. So sánh trong cùng task/GPU:

- dynamic every-step;
- best fixed hysteresis;
- transition-certified dynamic;
- frozen-state;
- final repair.

Phải báo action count và matched-budget result.

### Ưu tiên 2 — Richer action candidate generator

Chỉ làm sau transition controller. Ablation tối thiểu:

- direct-only;
- direct + alternate endpoint;
- full candidates without graph score;
- full candidates with graph score.

### Ưu tiên 3 — Faithful common-task baselines

Tối thiểu cần một inference-time baseline chạy trên cùng TabDiff checkpoint và constraint subset: Harpoon hoặc CTDF, final projection/repair, và best-of-N + independent certifier. Mọi method dùng cùng samples, seeds, verifier và constraints.

### Ưu tiên 4 — Independent checkpoint/backbone

Sau khi method đóng băng, thêm ít nhất một training checkpoint seed hoặc port sang TabSyn latent diffusion. Nếu effect biến mất, hạ kết luận xuống TabDiff-specific mechanism.

### Ưu tiên 5 — Graph-source ablation

Chỉ chạy khi source-specific edges thực sự đi vào actuator/branch proposal: empirical-only, LLM/schema-only, predefined-only và unified. Không chạy ablation giả khi variants có cùng actuation graph.

---

## 14. Metrics cho bảng cuối

### Semantic validity

- macro violation rate và heldout violation-rate error;
- per-edge satisfaction và applicability precision/recall;
- valid exceptions changed;
- antecedent evasion rate.

### Dynamic-graph mechanism

- transition precision/recall;
- topology additions/removals/flips;
- row–edge assignment control;
- action events, unique rows, actions per corrected violation;
- timing và refresh frequency.

### Fidelity, utility và cost

- numerical KS, categorical TV, Shape/Trend;
- α-Precision/β-Recall và joint/subgroup fidelity;
- TSTR/MLE utility, detection và privacy nếu đủ thời gian;
- runtime, GPU-hours, denoiser evaluations, API calls/tokens và edited cells.

Independent verifier không được chỉ là đúng graph đã dùng để guidance. Cần withheld/human-audited subset hoặc broader relation audit để giảm circular evaluation.

---

## 15. Claim boundary

### Đã có evidence

1. Mixed-type predicted-clean graph có temporal readability profile rõ; early feedback không đáng tin.
2. Row- và trajectory-dependent graph topology có thể cải thiện semantic calibration so với frozen topology.
3. Correct row assignment có causal signal trên Shoppers khi action budget bằng nhau.
4. Dynamic benefit rõ nhất ở middle region trên Shoppers; effect lặp lại uncapped trên Diabetes.
5. Một refresh policy chung không phù hợp mọi dataset.

### Chỉ dùng sau batch/method tiếp theo

1. Transition certification tốt hơn fixed hysteresis.
2. Dynamic method action-efficient trên nhiều dataset.
3. Graph-wide scoring cải thiện action selection.
4. External LLM knowledge tạo incremental generation gain.

### Chưa được phép dùng

1. State of the art.
2. Dynamic graph luôn hơn final repair.
3. Middle luôn là timing tối ưu.
4. Per-step guidance hoặc LLM-in-the-loop là mới nói chung.
5. LLM online cần thiết cho executable relations.

---

## 16. Đánh giá mức sẵn sàng cho ICDE

### Đã đủ

- problem formulation và novelty boundary đã rõ;
- pipeline end-to-end và controls đã chạy;
- có positive, negative và falsification evidence;
- có ba datasets và official baseline audits;
- fidelity/utility chưa cho thấy collapse.

### Chưa đủ

- chưa có transition-certified method thắng fixed policies;
- chưa có faithful common-task Harpoon/CTDF comparison;
- chưa có second training checkpoint/backbone;
- graph-source contribution chưa tạo generation gain độc lập;
- independent semantic gold/verifier còn yếu;
- action layer chưa đủ phong phú.

Đánh giá hiện tại: **đã là project có research gap hợp lý và có thể viết problem/method ngay, nhưng chưa phải submission-ready SOTA paper**. Mốc quyết định là transition-certifier result và một faithful same-task inference baseline.

---

## 17. Lộ trình

### 3–5 ngày tới

1. đọc `1060882–1060883`;
2. chọn hoặc bác bỏ transition-survival certifier;
3. implement một online policy duy nhất;
4. chạy paired three-dataset ablation;
5. freeze temporal window và refresh rule.

### 5–10 ngày tiếp theo

1. common-task Harpoon/CTDF/final-repair/best-of-N comparison;
2. richer action candidates nếu graph score có cơ hội tạo khác biệt;
3. independent checkpoint confirmation;
4. final fidelity/utility/cost/bootstrap tables.

Sau khi method đóng băng, không mở thêm engineering branches; chỉ chạy reviewer-critical ablations, hoàn thiện graph-source audit, qualitative trajectories, figures và reproducibility package.

---

## 18. Artifacts chính

- `KNOWLEDGE_GRAPH_PROTOCOL_20261005_VI.md`
- `EXPERIMENT_LOG_20261008.md`
- `ICDE_SUBMISSION_SPRINT_20261007.md`
- `ACTIVE_GRAPH_THEO_THOI_GIAN_REPORT_20261006.md`
- `BANG_TONG_HOP_PAPER_BASELINE_VA_KET_QUA_20261002.md`
- `outputs/bootstrap_dynamic_topology_1060488/RESULTS.md`
- `outputs/adaptive_paired_aggregate_1060655/RESULTS.md`
- `outputs/diabetes_controls_aggregate_1060481/RESULTS.md`
- `outputs/graph_guardrail_aggregate_1060665/RESULTS.md`
- `scripts/transition_survival_probe.py`
- `scripts/aggregate_transition_survival.py`
- `slurm/258_transition_survival_multidataset.sbatch`
- `slurm/259_aggregate_transition_survival.sbatch`

---

## 19. Kết luận

Hướng nghiên cứu nên tiếp tục, nhưng formulation cần giữ kỷ luật:

> Không phải “LLM sửa synthetic row trong diffusion”, mà là **LLM/statistics-derived verified graph được instantiate thành row-specific graph, graph này thay đổi theo predicted-clean trajectory, và chỉ certified transitions mới phát typed actions vào reverse diffusion**.

Evidence hiện tại đủ để nói dynamic topology là một phenomenon có thật trên ít nhất hai datasets. Evidence chưa đủ để nói mechanism hiện tại tốt hơn strong repair hoặc constraint-guidance methods. Transition certification, richer action choice và faithful common-task baselines là ba phần quyết định liệu project có tiến từ exploratory mechanism study thành một ICDE paper cạnh tranh hay không.

---

## 20. Cập nhật: tăng cường injection và cải thiện graph discovery

### 20.1. Sửa cách diễn giải main result

Nhận xét rằng kết quả dynamic graph hiện chưa mạnh so với baseline là đúng. So sánh dynamic với frozen chỉ chứng minh:

> Khi cùng dùng một graph và action layer, cập nhật topology theo trạng thái hiện tại tốt hơn giữ topology cũ.

Nó **không** chứng minh method hoàn chỉnh tốt hơn post-repair, best-of-N hoặc constraint guidance mạnh. Frozen graph là control cố ý yếu; vì vậy mức giảm 48.8% trên Shoppers và 69.6% trên Diabetes là evidence về cơ chế state evolution, không phải evidence SOTA.

### 20.2. Hai bottleneck đã tách được

#### Bottleneck A — injection chưa đủ mạnh

- Categorical action đã hard-set state ở bước hiện tại.
- Numeric correction trước đây chỉ thay đổi predicted-clean estimate với hệ số 1. Tuy nhiên Euler update làm displacement thực tế bị co theo `1 - sigma_next / sigma`; correction vì thế có thể bị denoiser rửa trôi.
- Mỗi row-edge chỉ được quyết định action một lần; không có target memory ở các bước sau.

Đã implement hai biến thể:

1. **Guidance strength**: scale clean-target bằng hệ số cố định hoặc chuẩn hoá theo noise step để đạt một state displacement mục tiêu.
2. **State-aware hold**: giữ target thêm `H` bước khi edge vẫn active; edge deactive thì release ngay. Đây vẫn là dynamic graph, không phải static clamping.

#### Bottleneck B — effective graph capacity thấp

| Dataset | Tổng cạnh | Actuator | Actuator exact duy nhất | Semantic core duy nhất | Endpoint pattern | Actuator có calibration violation = 0 | Cạnh thật sự phát action qua 3 runs | Graph guardrail weight cũ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Shoppers | 17 | 4 | 4 | 4 | 4 | 3 | 4 | 0 |
| Default | 24 | 5 | 4 | 4 | 4 | 0 | 5 | 0 |
| Diabetes | 22 | 6 | 6 | 4 | 4 | 6 | 3 | 0 |

Các vấn đề cụ thể:

- Default có hai proposal cùng executable program.
- Diabetes có sáu actuator nhưng chỉ bốn semantic core; chỉ ba cạnh từng phát action trong ba runs.
- Diabetes actuator hoàn toàn categorical; Default actuator hoàn toàn numeric. Điều này giải thích một phần dataset dependence của injection.
- Nhiều cạnh `score_only` tồn tại nhưng trước đây `graph_weight=0`, nên chúng chỉ được report chứ chưa thực sự bảo vệ action khỏi tạo violation mới.
- Curation ưu tiên support cao và violation thấp, dẫn đến graph dễ bị saturated: verifier tốt nhưng ít tạo challenge phân biệt method.

Artifact audit: `outputs/graph_capacity_audit_20261008_v2/RESULTS.md`.

### 20.3. Kết quả guidance-strength ban đầu

Kết quả dưới đây là tuning seed; hai confirmation seeds tách riêng đang chạy/đã hoàn thành một phần.

| Dataset | Setting | Actuator violation | Sai lệch so với calibration | Macro violation | Numeric KS | Categorical TV |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Shoppers | scale 1 | 0.006166 | 0.002562 | 0.062418 | 0.024185 | 0.023069 |
| Shoppers | scale 4 | **0.005701** | **0.002097** | **0.062340** | 0.024256 | 0.023139 |
| Default | scale 1 | 0.009603 | **0.018172** | 0.071431 | 0.030362 | 0.012895 |
| Default | scale 4 | **0.003779** | 0.022683 | **0.068077** | 0.030486 | 0.013139 |
| Default | step-normalized 0.05 | 0.026250 | **0.010964** | 0.077184 | 0.030451 | 0.012935 |
| Diabetes | scale 1 | 0.001098 | 0.001098 | 0.020589 | 0.019208 | 0.019805 |
| Diabetes | step-normalized 0.10 | **0.000926** | **0.000926** | **0.018349** | 0.020882 | 0.020724 |

Diễn giải:

- Shoppers có tín hiệu sạch: scale 4 giảm actuator violation 7.5% và calibration error 18.2% so với scale 1, gần như không đổi fidelity.
- Default thể hiện trade-off quan trọng. Scale 4 tăng semantic satisfaction mạnh và tốt hơn final repair trên absolute actuator violation trong tuning seed, nhưng over-enforce near-hard relations nên xa calibration distribution hơn. Step-normalized 0.05 làm điều ngược lại: bảo toàn legitimate violation mass tốt hơn nhưng satisfaction thấp hơn.
- Diabetes chủ yếu categorical, nên numeric scale cố định gần như không có tác dụng; step-normalized chỉ khác khi action chọn cách deactive một numeric antecedent. Đây là evidence rằng một guidance schedule chung cho mọi representation là không hợp lý.
- Vì vậy paper nên report **validity–calibration frontier**, không chọn một scalar metric để che trade-off.

Confirmation Shoppers trên hai seed mới đều giữ đúng chiều cho scale 4: calibration error giảm từ 0.003343 xuống 0.002903 và từ 0.002753 xuống 0.002373. Default calibration-first setting cũng giảm calibration error trên cả hai seed, nhưng làm absolute satisfaction tệ hơn. Diabetes không lặp lại gain của tuning seed: selected và scale 1 giống hệt nhau trên cả hai confirmation seeds, phù hợp với việc graph này gần như hoàn toàn categorical.

Nhánh **Default validity-first scale 4** đã hoàn thành hai seed độc lập:

| Seed | Method | Actuator violation | Calibration error | Macro violation | Numeric KS |
| --- | --- | ---: | ---: | ---: | ---: |
| 20261082 | scale 1 | 0.008124 | **0.018337** | 0.066523 | **0.026459** |
| 20261082 | scale 4 | **0.006150** | 0.020312 | **0.065185** | 0.026544 |
| 20261083 | scale 1 | 0.008296 | **0.019774** | 0.069603 | **0.029753** |
| 20261083 | scale 4 | **0.004684** | 0.021778 | 0.066365 | 0.029850 |

Đây là exact paired rerun: scale 1 và scale 4 chạy trong cùng GPU allocation, unconditional CSV có cùng SHA-256 ở cả hai seed. Scale 4 giảm actuator violation 24.3% và 43.5%, đồng thời giảm macro violation 2.0% và 4.7%; Numeric KS chỉ tăng 0.000085 và 0.000097. Calibration error tăng 10.8% và 10.1%. Trong chính mỗi scale-4 run, final repair trên cùng base sample đạt 0.012850 và 0.013524, nên scale-4 mid-injection thắng final repair rõ về satisfaction và macro violation. Tuy nhiên calibration error tệ hơn final repair; kết quả này chứng minh injection mạnh có hiệu lực, đồng thời xác nhận over-enforcement là rủi ro chính chứ không còn là giả thuyết.

### 20.4. Kết quả dùng score-only graph làm action guardrail

| Dataset | Graph weight | Actuator violation | Calibration error | Numeric KS | Categorical TV |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shoppers | 0 | 0.006293 | 0.002689 | 0.024668 | 0.022862 |
| Shoppers | 1 | **0.005705** | **0.002102** | 0.024906 | **0.020045** |
| Default | 0 | 0.008537 | **0.017925** | 0.031848 | **0.011894** |
| Default | 3 | **0.008336** | 0.018126 | **0.028284** | 0.012457 |
| Diabetes | 0 hoặc 1 hoặc 3 | 0.000515 | 0.000515 | 0.021911 | 0.021708 |

Kết luận tạm thời:

- Guardrail tạo gain có ý nghĩa trên Shoppers và giảm Numeric KS mạnh trên Default.
- Không có tác dụng trên Diabetes, cho thấy score-only graph ở dataset này không phân biệt được các action candidates hiện có.
- Đây không phải một global hyperparameter đủ dùng; graph cần được thiết kế theo actionability và conflict coverage.

### 20.4a. State-aware hold ban đầu

| Dataset | Hold steps | Actuator violation | Calibration error | Macro violation | Số cell re-injection |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shoppers | 0 | 0.006357 | 0.002754 | 0.059798 | 0 |
| Shoppers | 2 | **0.006327** | **0.002724** | 0.059809 | 115 |
| Shoppers | 4 | **0.006327** | **0.002724** | **0.059782** | 168 |
| Shoppers | 8 | **0.006327** | **0.002724** | **0.059782** | 215 |
| Default | 0 | 0.013796 | **0.013575** | 0.069069 | 0 |
| Default | 2 | **0.009114** | 0.017347 | **0.067241** | 131 |
| Default | 4 | 0.009447 | 0.017015 | 0.067272 | 212 |
| Default | 8 | 0.009435 | 0.017027 | 0.067342 | 295 |
| Diabetes | 0 | 0.001618 | 0.001618 | 0.023732 | 0 |
| Diabetes | 2 | 0.001618 | 0.001618 | 0.023732 | 177 |
| Diabetes | 4 | 0.001618 | 0.001618 | 0.023732 | 337 |
| Diabetes | 8 | 0.001618 | 0.001618 | 0.023732 | 592 |

Hold dài không tiếp tục tạo gain: Shoppers bão hoà sau 2–4 bước; Default tốt nhất ở 2 bước về satisfaction/macro. Điều này ủng hộ một short-horizon proximal correction, không ủng hộ hard clamping kéo dài. Default vẫn lặp lại trade-off over-enforcement. Diabetes hoàn toàn không đổi dù có tới 592 cell re-injections, nên repeated categorical clamping là computation thừa ở graph hiện tại và bị bác bỏ.

### 20.5. Graph discovery v2

Protocol mới không đơn thuần tăng số cạnh. Nó thay objective proposal sang **trajectory-actionable graph**:

1. cạnh phải có applicability có khả năng thay đổi khi row evolve;
2. có violation và action rõ ràng;
3. ưu tiên process/temporal/accounting/cross-field relations;
4. tránh demographic subgroup, rare exception và duplicated summary fields;
5. ưu tiên semantic/endpoint diversity thay vì nhiều threshold variant;
6. typecheck, counterexample critique, independent semantic judge và calibration vẫn giữ nguyên;
7. actuator được deduplicate cả exact program lẫn semantic core;
8. graph mới và graph cũ sẽ được đánh giá trên cùng một frozen union audit graph, tránh việc method tự đổi thước đo.
9. sau trajectory probe, một selector chỉ được **demote**, không được tự promote, các actuator; selector ưu tiên generator error exposure, middle-state flips và endpoint diversity. Heldout không tham gia bước này.

Models: GPT-4.1 cho proposal/critic và GPT-5.4 cho external knowledge/judge; budget tối đa 200k tokens mỗi dataset; không gửi raw rows; heldout không dùng để chọn graph.

Kết quả cấu trúc graph v2 đã hoàn thành:

| Dataset | Version | LLM-protocol retained | Endpoint patterns trong LLM graph | External/hybrid | Curated actuator |
| --- | --- | ---: | ---: | ---: | ---: |
| Shoppers | cũ | 7 | 6 | 1 | 4 |
| Shoppers | v2 | **14** | **8** | **3** | **12** |
| Default | cũ | 9 | 8 | 2 | **5** |
| Default | v2 | **16** | **13** | **4** | 3 |
| Diabetes | cũ | 7 | 5 | 0 | **6** |
| Diabetes | v2 | **12** | **11** | **5** | 5 |

V2 tăng rõ coverage, endpoint diversity và external/hybrid grounding ở cả ba datasets, nhưng không tự động tăng số actuator: Default và Diabetes có ít actuator hơn vì nhiều relation mới chỉ đủ authority cho score/monitor. Đây mới là improvement về graph discovery, chưa phải generation gain. Shoppers vẫn có nhiều proposal thuộc cùng page-count/duration family, nên trajectory portfolio tiếp tục áp endpoint-diversity cap trước evaluation.

Common-verifier evaluation đầu tiên trên Shoppers đã có kết quả. Graph cũ và v2 đều được đo trên cùng frozen union gồm 30 cạnh:

| Guidance graph | Union macro violation | Union heldout error | Union calibration error | Numeric KS | Actions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Graph cũ | 0.060418 | 0.018888 | 0.017067 | 0.024217 | 74 |
| Graph v2 | **0.059064** | **0.017413** | **0.015274** | **0.024033** | 401 |

V2 giảm union macro violation 2.2%, heldout error 7.8% và calibration error 10.5%, đồng thời Numeric KS tốt hơn nhẹ. Đây là evidence đầu tiên rằng graph discovery quality tạo generation gain trên một thước đo chung, không phải do đổi verifier. Tuy nhiên final repair trên graph v2 vẫn mạnh hơn về union macro/error (0.056376 macro), và số actions tăng 5.4 lần; do đó cost/action efficiency chưa tốt và chưa thể claim baseline dominance.

Default cho kết quả ngược lại: graph v2 không phát action nào trên evaluation seed, nên output bằng unconditional (union macro 0.065328), trong khi graph cũ phát 120 actions và đạt 0.062100. Calibration-trajectory portfolio cũng không phát action trên evaluation seed. Điều này xác nhận graph coverage trên real-data partitions không đồng nghĩa generator exposure ổn định. Một nhánh all-GPT-5.4 đã được submit để tách hai khả năng: proposal model chưa đủ mạnh hay bản thân Default thiếu nhiều semantic hard relations có thể actuation.

### 20.6. Jobs đã submit

| Nhóm | Job | Nội dung |
| --- | --- | --- |
| Guidance strength tuning | `1060911` | 7 schedules × 3 datasets, hoàn thành |
| Guidance selection | `1060912` | A-priori objective, hoàn thành |
| Independent confirmation | `1060913` | 2 seed × 3 datasets |
| Graph guardrail | `1060935` | graph weight 0/0.5/1/3 × 3 datasets, hoàn thành |
| Graph discovery v2 | `1060952` | trajectory-actionable protocol × 3 datasets |
| Curation v2 | `1060953` | verified, deduplicated role assignment |
| Trajectory probe v2 | `1060954` | learn readability per edge |
| Frozen union audit graph | `1060955` | common evaluation graph cũ + mới |
| Graph v2 comparison | `1060956` | old/new × graph weight 0/1, paired noise |
| State-aware hold | `1060959` | hold 0/2/4/8 × 3 datasets |
| Default validity confirmation | `1060966` | scale 4 trên hai seed độc lập |
| Trajectory-actionable portfolio | `1060979` | semantic authority + calibration trajectory + diversity |
| Portfolio evaluation | `1060980` | common frozen union verifier, graph weight 1 |
| Exact paired strength confirmation | `1060981` | scale 1/4 trong cùng GPU allocation + SHA-256 parity |
| Default high-quality discovery | `1060998` | GPT-5.4 cho toàn bộ proposal/critic/judge roles |
| Default high-quality downstream | `1060999–1061002` | curation, trajectory, union verifier, old/new comparison |

### 20.7. Tiêu chí quyết định sau batch này

Method chỉ được xem là tiến bộ thật nếu thỏa đồng thời:

1. mạnh hơn scale-1 dynamic graph trên independent seeds;
2. cải thiện so với final repair hoặc tạo validity–fidelity point không bị final repair dominate;
3. graph v2 tốt hơn graph cũ trên **cùng union verifier**, không chỉ trên verifier do chính nó sinh;
4. gain không chỉ đến từ over-enforcing near-hard rules;
5. dynamic topology vẫn tốt hơn frozen sau khi guidance strength đã được match.

Nếu graph v2 không tăng gain, kết luận hợp lý sẽ là graph discovery không phải bottleneck chính; khi đó nên freeze graph protocol và tập trung mechanism. Nếu state-aware hold/strong injection chỉ tăng satisfaction nhưng phá calibration, contribution nên chuyển sang calibrated adaptive strength thay vì tiếp tục tăng force.

## 21. Cập nhật sau graph v2 và strong-guidance batch

### 21.1. Graph discovery v2: kết quả đầy đủ trên common verifier

Tất cả các method trong bảng dưới được đánh giá bằng cùng một frozen union graph của từng dataset. Vì vậy graph mới không được tự thay đổi thước đo để tạo gain.

| Dataset | Guidance graph | Union macro violation | Heldout error | Calibration error | Numeric KS | Actions |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Shoppers | Graph cũ | 0.060418 | 0.018888 | 0.017067 | 0.024217 | 74 |
| Shoppers | Graph v2 | **0.058668** | **0.017168** | **0.015340** | **0.024034** | 404 |
| Shoppers | Final repair, graph v2 | 0.055609 | 0.015698 | 0.013376 | 0.024132 | 247 |
| Default | Graph cũ | **0.062100** | **0.015912** | **0.016688** | **0.029409** | 120 |
| Default | Graph v2 | 0.065328 | 0.018356 | 0.018519 | 0.029525 | 0 |
| Default | Final repair, graph cũ | 0.062209 | 0.015237 | 0.017115 | 0.029475 | 21 |
| Diabetes | Graph cũ | 0.020570 | 0.012780 | 0.012730 | 0.018114 | 100 |
| Diabetes | Graph v2 | **0.019885** | **0.012095** | **0.012045** | **0.018087** | 111 |
| Diabetes | Final repair, graph v2 | 0.019198 | 0.011408 | 0.011359 | 0.018223 | 26 |

Kết luận:

- graph v2 tạo gain thật trên Shoppers và Diabetes nhưng thất bại hoàn toàn trên Default;
- Shoppers trả giá bằng số actions tăng từ 74 lên 404, nên tăng coverage chưa đồng nghĩa tăng hiệu quả;
- final repair vẫn mạnh hơn online guidance ở cả Shoppers và Diabetes trong setting scale 1;
- graph discovery không phải bottleneck duy nhất. Một graph có coverage và authority tốt vẫn có thể không gặp violation trên distribution mà generator sinh ra.

### 21.2. Trajectory portfolio và graph capacity

Selector trajectory-only giữ 7/12 actuators trên Shoppers, 1/3 trên Default và 4/5 trên Diabetes. Heldout không được dùng để chọn cạnh.

| Dataset | Graph | Macro violation | Heldout error | Calibration error | Actions | Kết luận |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Shoppers | Full v2 | **0.058668** | **0.017168** | **0.015340** | 404 | Chất lượng tốt hơn nhẹ |
| Shoppers | Portfolio | 0.058910 | 0.017421 | 0.015535 | **270** | Giảm 33.2% actions với loss nhỏ |
| Default | Full v2/portfolio | 0.065328 | 0.018356 | 0.018519 | 0 | Không có generator exposure |
| Diabetes | Full v2 | 0.019885 | 0.012095 | 0.012045 | 111 | Bằng portfolio trong run này |
| Diabetes | Portfolio | 0.019885 | 0.012095 | 0.012045 | 111 | Cạnh bị bỏ là inert |

Portfolio hiện là cách giảm action hợp lý trên Shoppers, nhưng không phải một contribution độc lập. Nó không cứu được Default và không tạo khác biệt trên Diabetes.

### 21.3. Nâng toàn bộ Default discovery lên GPT-5.4 không giải quyết failure

Run all-GPT-5.4 dùng 177,176 tokens và 20 API calls. LLM protocol tạo được 28 relation giữ lại, 15 endpoint patterns và 22 relation external/hybrid. Tuy nhiên curation chỉ giữ được hai actuators từ ba candidates sau deduplication. Khi chạy generation, graph này chỉ phát hai actions và gần như bằng unconditional; graph cũ phát 112 actions và tốt hơn rõ.

Kết quả này bác bỏ giả thuyết đơn giản rằng Default thất bại chỉ vì proposal model chưa đủ mạnh. Bottleneck chính là giao của ba điều kiện:

1. relation có semantic authority;
2. relation có executable action an toàn;
3. generator thực sự đi qua violated state đủ thường xuyên.

Do đó không tiếp tục tăng model/API cost cho graph discovery ở giai đoạn này. GPT-5.4 có thể tăng semantic coverage nhưng không tự tạo online actionability.

### 21.4. Guidance strength và graph portfolio trên Shoppers

Bảng dưới là trung bình hai independent seeds, 8,192 samples/seed, cùng unconditional sample theo SHA-256.

| Graph | Guidance | Macro violation | Heldout error | Calibration error | Numeric KS | Actions |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Full v2 | scale 1 | 0.058679 | 0.015612 | 0.014168 | 0.024856 | 427 |
| Full v2 | scale 4 | **0.057306** | **0.014537** | **0.012973** | 0.024934 | 364 |
| Full v2 | hard 4 / near-hard 1 | 0.058747 | 0.015692 | 0.014257 | 0.024885 | 420 |
| Full v2 | hard 4 / near-hard 2 | 0.058465 | 0.015613 | 0.013976 | **0.024875** | 398 |
| Portfolio | scale 1 | 0.059079 | 0.015976 | 0.014401 | 0.024868 | 283 |
| Portfolio | scale 4 | **0.057641** | **0.014849** | **0.013435** | **0.024799** | 274 |
| Portfolio | hard 4 / near-hard 1 | 0.059025 | 0.016031 | 0.014462 | 0.024874 | 285 |
| Portfolio | hard 4 / near-hard 2 | 0.058897 | 0.016022 | 0.014393 | 0.024871 | 277 |

Các phát hiện chính:

- scale 4 thắng nhất quán scale 1 trên validity và calibration, gần như không đổi KS;
- portfolio scale 4 dùng ít actions hơn full scale 4 khoảng 24.7%, đổi lại macro violation tăng 0.000335;
- hardness-adaptive guidance không thắng. Phần lớn actuator hữu ích được gắn nhãn `near_hard`, nên rule `hard=4, near-hard=1` vô tình làm yếu chính các cạnh cần can thiệp;
- vì vậy hardness label chỉ nên dùng cho audit/calibration, chưa nên dùng trực tiếp làm guidance controller.

### 21.5. Dynamic guidance kết hợp residual final repair

Trên Shoppers full graph, trung bình hai seeds:

| Method | Macro violation | Heldout error | Calibration error | Numeric KS | Tổng actions | Residual repair actions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dynamic, scale 4 | 0.057306 | 0.014537 | 0.012973 | 0.024934 | 364.0 | — |
| Final repair only | 0.055726 | 0.013814 | 0.012758 | 0.025106 | 251.5 | 251.5 |
| Dynamic scale 4 + residual repair | **0.055119** | **0.013290** | **0.012310** | **0.024952** | 489.5 | **125.5** |

Dynamic + residual repair tốt hơn final repair only khoảng 1.1% macro violation, 3.8% heldout error và 3.5% calibration error. Nó giảm khoảng 50.1% số residual repairs, nghĩa là online guidance đã giải quyết trước một nửa số lỗi mà postprocessor phải sửa.

Tuy nhiên tổng số interventions vẫn lớn hơn final repair only. Do đó đây là evidence về chất lượng/compositionality, chưa phải evidence về cost dominance. Paper phải báo đồng thời online actions, residual actions và tổng actions.

### 21.6. Matched-strength topology control

Scale 4 được dùng cho cả dynamic và frozen-state, cùng graph, seed, schedule, action layer và unconditional sample.

| Dataset | Topology | Macro violation | Heldout error | Calibration error | Actions |
| --- | --- | ---: | ---: | ---: | ---: |
| Shoppers | Dynamic | **0.057306** | **0.014537** | **0.012973** | 364.0 |
| Shoppers | Frozen-state | 0.059156 | 0.016049 | 0.014638 | 259.0 |
| Default | Dynamic | **0.056747** | **0.013225** | **0.013829** | 95.5 |
| Default | Frozen-state | 0.059385 | 0.014802 | 0.014058 | 58.0 |

Dynamic giảm macro violation 3.1% trên Shoppers và 4.4% trên Default; heldout error giảm lần lượt 9.4% và 10.7%. Đây là evidence tốt hơn trước rằng lợi ích topology vẫn tồn tại khi guidance strength được match.

Nhưng `frozen_state` cố định cả active edge set và violation state ở đầu window, nên đây chưa phải static control mạnh nhất. Job `1061883` đang chạy thêm:

- `frozen`: chỉ cố định active edge set, vẫn đánh giá satisfaction hiện tại;
- `shuffled`: giữ support nhưng phá row-edge assignment;
- `oracle_final`: dùng final active set như diagnostic upper bound;
- `dynamic` và `frozen_state` trong cùng allocation để kiểm tra parity.

Chỉ sau control này mới nên chốt claim về dynamic topology.

### 21.7. Default guidance-strength frontier

Một paired seed mới cho thấy gain bão hòa từ scale 2; tăng tiếp lên 3–4 không tạo cải thiện đáng kể.

| Scale | Macro violation | Heldout error | Calibration error | Numeric KS | Actions |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 0.077629 | 0.023555 | 0.026386 | **0.027458** | 121 |
| 1.0 | 0.075139 | 0.021066 | 0.024995 | 0.027467 | 103 |
| 1.5 | 0.072971 | 0.019451 | 0.024875 | 0.027484 | 96 |
| 2.0 | **0.071760** | **0.018953** | 0.024753 | 0.027568 | 94 |
| 3.0 | 0.071944 | 0.019169 | 0.024870 | 0.027595 | 94 |
| 4.0 | 0.071865 | 0.018990 | **0.024727** | 0.027560 | 94 |

Scale 2 là operating point hợp lý nếu ưu tiên macro/heldout error; scale 4 tốt hơn rất nhẹ về calibration error. Khác với graph-specific metric ở Section 20.3, common union verifier trong run này không cho thấy calibration xấu đi ở scale cao. Điều này cho thấy kết luận over-enforcement phụ thuộc evaluation slice; cả actuator-only và union-level metrics phải được giữ trong paper.

### 21.8. Best-current hybrid trên ba dataset

Đây là một seed mới mỗi dataset và chỉ dùng để kiểm tra portability, chưa phải main table cuối.

| Dataset | Method | Macro violation | Heldout error | Calibration error | Numeric KS | Categorical TV |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Default | Unconditional | 0.062468 | 0.018687 | 0.015945 | 0.030482 | 0.012713 |
| Default | Dynamic | **0.056325** | **0.014041** | **0.013434** | **0.030420** | 0.012869 |
| Default | Final repair | 0.058745 | 0.014964 | 0.013774 | 0.030491 | **0.012713** |
| Diabetes | Unconditional | 0.020963 | 0.013173 | 0.013124 | 0.021319 | **0.019517** |
| Diabetes | Dynamic | 0.016665 | 0.008875 | 0.008825 | **0.021075** | 0.020477 |
| Diabetes | Dynamic + repair | 0.016216 | 0.008426 | 0.008377 | **0.021075** | 0.020495 |
| Diabetes | Final repair | **0.016169** | **0.008379** | **0.008330** | 0.021319 | 0.019693 |
| Shoppers | Unconditional | 0.061361 | 0.018737 | 0.017843 | **0.024727** | 0.023721 |
| Shoppers | Dynamic | 0.058559 | 0.016288 | 0.015164 | 0.024849 | **0.023715** |
| Shoppers | Dynamic + repair | 0.055494 | 0.014595 | 0.013842 | 0.025039 | **0.023715** |
| Shoppers | Final repair | **0.055369** | **0.014503** | **0.013746** | 0.025039 | 0.023721 |

Default là positive case rõ trong seed này: dynamic thắng final repair trên ba semantic errors với fidelity gần như giữ nguyên và không còn residual repair. Diabetes là negative case: final repair vẫn nhỉnh hơn; categorical TV của online guidance xấu hơn. Shoppers portfolio hybrid gần hòa final repair, trong khi full graph hai-seed ở Section 21.5 cho gain nhỏ. Điều này xác nhận graph capacity/action coverage ảnh hưởng trực tiếp tới lợi ích của online mechanism.

### 21.9. Trạng thái các hypothesis sau batch

| Hypothesis | Trạng thái | Evidence |
| --- | --- | --- |
| Tăng graph coverage luôn tăng generation quality | Bác bỏ | Default v2/HQ không phát action |
| Nâng proposal model lên GPT-5.4 giải quyết graph failure | Bác bỏ | 177k tokens nhưng chỉ hai actuators, hai actions |
| Stronger mid-step injection có hiệu lực | Ủng hộ | scale 4 thắng scale 1 trên Shoppers; scale 2–4 thắng trên Default |
| Một global hardness-adaptive schedule tốt hơn constant scale | Bác bỏ ở implementation hiện tại | hard/near-hard schedules đều thua scale 4 |
| Dynamic topology vẫn có gain khi match guidance strength | Ủng hộ tạm thời | hai seeds trên Shoppers và Default, nhưng strong `frozen` control còn pending |
| Dynamic guidance bổ sung cho final repair | Ủng hộ có điều kiện | full Shoppers hai seeds giảm errors và residual repair; chưa giảm tổng interventions |
| Method hiện đã dominate final repair trên mọi dataset | Bác bỏ | Diabetes và portfolio Shoppers còn thua/hoà |

### 21.10. Jobs mới

| Job | Nội dung | Mục tiêu quyết định |
| --- | --- | --- |
| `1061459[0-1]` | Full/portfolio × constant/hardness-adaptive guidance, hai seeds | Chọn graph capacity và strength controller |
| `1061460[0-3]` | Dynamic vs frozen-state, scale 4, Shoppers/Default, hai seeds | Match guidance strength |
| `1061461` | Default scale 0.5–4 frontier | Tìm saturation và calibration trade-off |
| `1061462[0-2]` | Best-current hybrid trên ba datasets | Portability smoke test |
| `1061883[0-4]` | Dynamic/frozen/frozen-state/shuffled/oracle-final trên ba datasets | Strong causal topology controls |
| `1061884[0-3]` | Default và Diabetes dynamic + residual repair trên hai seeds mới | Xác nhận baseline comparison |
| `1061885` | Aggregate tự động sau hai jobs trên | Sinh bảng CSV và Markdown tái lập được |

Tạm thời không submit thêm graph-discovery API jobs. Điểm có giá trị thông tin cao nhất hiện nay là kiểm tra strong static control và khả năng bổ sung final repair; tăng số relation hoặc đổi LLM tiếp sẽ không giải quyết bottleneck đã quan sát.

Chiến lược dataset, quy tắc phân biệt paper-reported/official-rerun/matched-local và các baseline jobs mới được ghi riêng trong `BASELINE_DATASET_AND_RERUN_STRATEGY_20261008_VI.md`.

## 22. Cập nhật strong controls và injection frontier mở rộng

### 22.1. “Gain 20–30%” phải gắn với baseline và metric cụ thể

Gain dưới đây là **relative error reduction** so với unconditional sample được ghép cùng noise:

\[
100\times\frac{E_{\text{unconditional}}-E_{\text{method}}}{E_{\text{unconditional}}}.
\]

Không được hiểu đây là tăng 20–30 điểm phần trăm satisfaction. Cũng không được dùng một metric thuận lợi để che việc metric khác hoặc fidelity xấu đi.

| Dataset/setting | Macro error giảm | Heldout error giảm | Calibration error giảm | Nhận xét |
| --- | ---: | ---: | ---: | --- |
| Default, dynamic scale 4, topology-control seeds | 7.9% | 19.7% | 6.0% | Có hiệu quả nhưng chưa đạt 20–30% đồng đều |
| Default, dynamic + repair confirmation | 6.3% | 15.9% | 4.7% | Scale 4 gần bão hòa; tăng force đơn thuần khó đủ |
| Diabetes, dynamic + repair confirmation | **31.4%** | **51.9%** | **53.9%** | Vượt mục tiêu, nhưng final repair vẫn nhỉnh hơn nhẹ |
| Shoppers, full dynamic + repair, Section 21.5 | 10.7% | 26.7% | 27.2% | Đạt 20–30% trên heldout/calibration, chưa đạt trên macro |

Kết luận hiện tại: mục tiêu 20–30% đã đạt trên một số error metrics và dataset, nhưng contribution quan trọng hơn là phần gain **so với frozen graph và post-repair**. Phần này hiện nhỏ hơn. Không nên tối ưu báo cáo bằng cách chỉ chọn Diabetes hoặc chỉ chọn heldout error.

### 22.2. Default strong topology controls đã hoàn tất

Trung bình hai seeds, 8,192 samples/seed, scale 4 và cùng union verifier:

| Topology | Macro error | Heldout error | Calibration error | Actions |
| --- | ---: | ---: | ---: | ---: |
| Dynamic | **0.056747** | **0.013225** | **0.013829** | 95.5 |
| Frozen active set | 0.059319 | 0.014783 | 0.014049 | 64.5 |
| Frozen active set + frozen violation state | 0.059385 | 0.014802 | 0.014058 | 58.0 |
| Shuffled row-edge assignment | 0.061380 | 0.016295 | 0.014490 | 61.0 |
| Oracle-final active set | 0.056190 | 0.013591 | 0.014739 | 48.0 |

Dynamic tốt hơn `frozen active set` khoảng 4.3% macro error và 10.5% heldout error. Shuffled control gần unconditional, cho thấy gain phụ thuộc row-edge assignment chứ không chỉ do thêm perturbation. Oracle-final nhỉnh hơn dynamic về macro nhưng thua dynamic trên heldout và calibration: biết final active set không đủ để thay thế việc theo dõi state giữa trajectory.

Đây là control mạnh hơn `frozen_state` trước đó. Tuy nhiên mức chênh dynamic–frozen chưa đạt 20–30%; nó hỗ trợ claim về topology động nhưng chưa đủ để claim một khoảng cách rất lớn với static guidance.

### 22.3. Xác nhận dynamic + residual repair trên seed mới

| Dataset | Method | Macro error | Heldout error | Calibration error | Numeric KS | Actions |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Default | Unconditional | 0.063715 | 0.017703 | 0.017134 | 0.028349 | 0 |
| Default | Final repair | 0.061137 | 0.015125 | **0.015737** | 0.028336 | 17 |
| Default | Dynamic scale 4 | **0.059693** | **0.014812** | 0.016330 | **0.028280** | 96 |
| Diabetes | Unconditional | 0.018836 | 0.011403 | 0.010996 | 0.022579 | 0 |
| Diabetes | Final repair | **0.012826** | **0.005393** | **0.004986** | **0.022579** | 23 |
| Diabetes | Dynamic | 0.013042 | 0.005609 | 0.005202 | 0.022619 | 111 |
| Diabetes | Dynamic + residual repair | 0.012913 | 0.005480 | 0.005073 | 0.022619 | 111.5 |

Default là case phù hợp với online guidance: dynamic tốt hơn final repair về macro/heldout error và KS, nhưng calibration error vẫn kém. Diabetes cho gain lớn so với unconditional nhưng không chứng minh cần mid-step injection, vì final repair đạt kết quả tốt hơn với ít actions hơn. Đây là negative control cần giữ trong paper.

### 22.4. Vì sao cần sweep rộng hơn thay vì chỉ tăng weight

`numeric_guidance_scale` chỉ quyết định độ lớn displacement của một numeric clean estimate. Nó không tăng số row được graph kích hoạt và không tác động trực tiếp lên categorical correction. Vì vậy bốn bottleneck được tách riêng:

1. **Strength:** scale `1, 2, 4, 8, 12, 16, 24, 32` để đo saturation/overshoot.
2. **Persistence:** cho phép cùng row-edge được action `1, 2, 4, 8` lần và thử hold `2/4` steps; đây là sweep quan trọng cho categorical edges.
3. **Timing:** cửa sổ hẹp giữa trajectory tới cửa sổ rộng `t=0.10–0.75`, kết hợp scale 4/8.
4. **Action selection:** graph guardrail weight `0, 1, 3, 10, 30` để kiểm tra liệu action mạnh hơn có cần đồng thời tránh phá các cạnh khác.

Mỗi frontier dùng một seed và paired noise để sàng lọc. Chỉ operating point không làm xấu fidelity/calibration mới được xác nhận bằng seed mới. Đây là nguyên tắc chống chọn cấu hình theo test result.

### 22.5. Jobs mới

| Job | Nội dung | Trạng thái lúc submit |
| --- | --- | --- |
| `1061927` | Pin Opacus 1.4.0 trong isolated Harpoon env và import-test | Đợi CPU |
| `1061928[0-2]` | Rerun official Harpoon/RePaint trên Adult, Default, Shoppers | Phụ thuộc `1061927` |
| `1061929` | Audit paper-reported và official-rerun riêng biệt | Phụ thuộc `1061928` |
| `1061930[0-1]` | Wide numeric injection strength trên Shoppers/Default | Đợi GPU |
| `1061931[0-2]` | Action persistence/hold trên ba datasets | Đợi GPU |
| `1061932[0-1]` | Window × strength trên Shoppers/Default | Đợi GPU |
| `1061933[0-1]` | Graph reward frontier trên Shoppers/Default | Đợi GPU |
| `1061934` | Aggregate mọi injection frontier thành CSV/Markdown | Phụ thuộc bốn frontier |
| `1061941[0-1]` | Dynamic/frozen/frozen-state tại broad window `0.15–0.60` | Đợi GPU; kiểm tra gain riêng của topology |

Harpoon sampling batch cũ `1061900` không tạo được kết quả hợp lệ dù Slurm ghi `COMPLETED`: mọi author command dừng khi Opacus 1.5.4 gọi `torch.nn.RMSNorm`, API không có trong PyTorch 2.2.2. Batch sửa môi trường không thay đổi author code hoặc checkpoint.

## 23. Kết quả frontier hoàn chỉnh và method thích nghi

### 23.1. Global scale không thể đồng thời tối ưu validity và drift

Trên Shoppers, scale lớn tạo gain semantic rõ nhưng nhanh chóng tăng drift:

| Scale | Macro error giảm | Heldout error giảm | Calibration error giảm | Numeric drift |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 6.56% | 21.87% | 23.99% | 0.00210 |
| 8 | 7.79% | 20.39% | 16.75% | 0.01431 |
| 12 | **8.64%** | 25.24% | 19.42% | 0.03714 |
| 16 | 8.51% | **27.88%** | 22.56% | 0.04943 |
| 24 | 7.07% | 23.58% | **25.48%** | 0.06821 |
| 32 | 5.91% | 19.86% | 23.83% | 0.09261 |

Trên Default, scale 4 tốt nhất về macro và heldout error; tăng lên 8–32 làm gain giảm. Kết quả này bác bỏ giả thuyết “cứ tăng injection weight sẽ tăng quality”. Scale lớn là evidence rằng mechanism có khả năng điều khiển sampler, nhưng không phải operating point an toàn.

### 23.2. Broad dynamic window làm rõ contribution topology

Một seed mới, scale 4, window `0.15–0.60`, cùng noise và common verifier:

| Dataset | Topology | Macro error | Heldout error | Calibration error | Actions |
| --- | --- | ---: | ---: | ---: | ---: |
| Shoppers | Dynamic | **0.058728** | **0.013808** | **0.011237** | 755 |
| Shoppers | Frozen | 0.065070 | 0.020342 | 0.019157 | 817 |
| Shoppers | Frozen state | 0.062279 | 0.016464 | 0.014239 | 321 |
| Default | Dynamic | **0.056977** | **0.013808** | **0.013515** | 107 |
| Default | Frozen | 0.059960 | 0.015787 | 0.013857 | 75 |
| Default | Frozen state | 0.062165 | 0.017814 | 0.015432 | 8 |

Trên Shoppers, dynamic giảm 9.7% macro error, 32.1% heldout error và 41.3% calibration error so với frozen graph. Frozen graph thậm chí làm xấu hơn unconditional dù dùng nhiều actions hơn dynamic. Đây là evidence mạnh nhất hiện tại rằng cập nhật row–edge topology trong trajectory quan trọng hơn chỉ tăng số intervention.

Trên Default, dynamic vẫn tốt hơn frozen nhưng khoảng cách nhỏ hơn: khoảng 5.0% macro, 12.5% heldout và 2.5% calibration. Claim vì vậy nên là dataset/graph-dependent, không phải universal dominance.

### 23.3. Các nút điều khiển đã bão hòa

- Graph reward weight 1, 3, 10 và 30 cho kết quả giống hệt trên cả hai dataset. Weight 1 đã đủ đổi candidate khi guardrail có khả năng phân biệt; tăng tiếp không tạo candidate mới.
- Cho phép 1→8 actions trên Shoppers giảm heldout error thêm khoảng 3.1% tương đối nhưng actions tăng từ 421 lên 670. Default và Diabetes hoàn toàn không đổi.
- Full-middle window tạo 1,848–1,946 actions trên Shoppers nhưng không thắng broad window. Coverage quá rộng không thay thế đúng timing.

Các kết quả này loại ba hướng engineering ít triển vọng: tiếp tục tăng graph weight, repeated categorical clamp, và mở window tới gần toàn trajectory.

### 23.4. Method mới: feedback-adaptive trust-region guidance

Sampler mới không dùng một scale cố định cho mọi row, edge và timestep. Với một numeric action tại timestep `t`, strength được đặt theo:

\[
\lambda_{i,e,t}=\min\left(\lambda_{\max},
\lambda_0\,g_{\text{late}}(t)\,r^{n_{i,e}}\right),
\]

trong đó `g_late(t)` tăng khi trajectory đi từ đầu tới cuối active window, còn `n_{i,e}` là số lần row–edge đó đã tái vi phạm. Clean-target displacement sau đó bị chặn bởi trust region:

\[
\Delta \hat{x}_0 \leftarrow \operatorname{clip}(\Delta \hat{x}_0,-\delta,\delta).
\]

Ý nghĩa:

1. correction sớm được giữ nhẹ vì preview còn nhiễu;
2. correction muộn được tăng lực khi semantic state đáng tin hơn;
3. chỉ các constraint khó và tái vi phạm mới nhận scale lớn;
4. trust region ngăn scale 12–32 gây drift cực đoan cho toàn bộ action.

Đây là controller theo row, edge và trajectory state; khác với static global guidance weight. Implementation đã được thêm vào `rowwise_reverse_action_guidance.py` dưới mode `feedback_adaptive` và ghi cả realized scale, số cell bị clip vào trace.

### 23.5. Jobs ngày 09/10/2026

| Job | Nội dung | Vai trò |
| --- | --- | --- |
| `1062126[0-1]` | Constant scale, trust-region cap, feedback adaptation và step-normalized control | Tách contribution của từng thành phần |
| `1062127[0-1]` | Feedback controller trên late/mid/broad/full-middle windows | Chọn timing |
| `1062128[0-1]` | Feedback controller với dynamic/frozen/frozen-state | Kiểm tra contribution topology sau khi đổi actuator |
| `1062131` | Aggregate lại frontier cũ sau khi bỏ dependency `tabulate` | Tạo artifact CSV/Markdown |
| `1062132` | Aggregate ba feedback batches | Tạo frontier mới |
| `1062129` → `1062130` | Bổ sung XGBoost vào isolated env và chạy lại CTGAN/TVAE baseline evaluation | Hoàn thiện baseline table |

Frontier chỉ dùng một seed để chọn operating region. Cấu hình tốt chỉ được đưa vào main result sau paired confirmation seed và kiểm tra không làm xấu KS/TV, drift, calibration hoặc số interventions một cách không hợp lý.

## 24. Phát triển actuator để tăng gain mà không làm yếu claim

### 24.1. Phân rã metric cho thấy bottleneck không chỉ nằm ở guidance strength

Trên các run mạnh hiện có, improvement trên nhóm cạnh thực sự được graph cho phép action lớn hơn nhiều so với macro metric trên toàn evaluation graph:

| Dataset/setting | Actuate macro giảm | Actuate heldout giảm | Toàn graph macro giảm | Toàn graph heldout giảm |
| --- | ---: | ---: | ---: | ---: |
| Shoppers, scale 16 | 52.5% | 82.4% | 12.1% | 28.8% |
| Default, dynamic promotion prototype | 69.6% | 65.6% | 15.3% | 46.6% |
| Diabetes, dynamic | 94.0% | 94.0% | 33.2% | 50.7% |

Macro metric còn chứa `score_only` và `monitor` edges. Vì vậy tăng scale trên cùng ba actuator không thể tạo gain 20–30% trên mọi dataset. Có hai hướng hợp lệ: discovery thêm actuator có semantics đáng tin, hoặc promote một phần `score_only` edge bằng calibration gate. Không được âm thầm tính metric chỉ trên các edge thuận lợi.

### 24.2. Calibrated dynamic promotion

Một `score_only` edge chỉ được promote thành action edge khi:

1. violation tồn tại liên tiếp qua ít nhất `K` clean previews;
2. train calibration violation rate của edge không vượt ngưỡng `epsilon`;
3. số row được sửa chỉ là violation mass vượt quá exception rate quan sát trên train;
4. active set vẫn được recompute theo row và timestep.

Kết quả sớm, chưa phải confirmation, trên Default (`epsilon=0.02`, `K=3`, không dùng trust-region cap):

| Method | Macro error | Macro giảm | Heldout giảm | Calibration giảm | Numeric drift | Actions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dynamic actuate-only | 0.05912 | 9.2% | 21.8% | 8.9% | 0.00030 | 107 |
| Dynamic + calibrated promotion | **0.05474** | **16.0%** | **37.7%** | 16.3% | 0.00087 | 315 |
| Final-only repair, cùng promoted edge set | 0.05768 | 11.4% | 37.1% | **27.8%** | gần 0 | 54 |

Online promotion tốt hơn final-only counterpart về macro error và gần tương đương heldout error; final repair vẫn tốt hơn về calibration error và rẻ hơn. Đây là trade-off cần report đầy đủ. Batch cũ dùng `max_guidance_delta=0.75` cho cả control nên bị confounded và không dùng để chọn method.

Safety audit sau đó cho thấy toàn bộ promoted edges trên Default và Shoppers ở batch này vốn là `empirical_mining` edges. Một số cạnh như `TrafficType -> Browser` là correlation rõ ràng, không phải semantic constraint. Vì vậy các số trên chỉ chứng minh **coverage bottleneck và tiềm năng của selective promotion**, chưa được dùng làm main result. Pipeline mới thêm một FM semantic judge: chỉ nhận schema program và aggregate discovery/calibration statistics, không nhận real rows hoặc heldout statistics; model phải loại correlation và chỉ approve domain/logical/process/accounting/temporal constraints. Main promotion result sẽ dùng graph sau semantic review, không dùng threshold-only graph.

Frontier threshold-only đã hoàn tất và xác nhận đây là một upper-bound diagnostic:

| Dataset | Promotion setting | Macro giảm | Heldout giảm | Calibration giảm | Numeric drift |
| --- | --- | ---: | ---: | ---: | ---: |
| Default | none | 9.2% | 21.8% | 8.9% | 0.00030 |
| Default | rate <=2% | 16.0% | 37.7% | 16.3% | 0.00087 |
| Default | rate <=5% | 17.9% | **41.6%** | 17.1% | 0.00096 |
| Default | rate <=10% | **19.2%** | 36.2% | **20.5%** | 0.00111 |
| Shoppers | none | 7.9% | 18.4% | 23.6% | 0.00461 |
| Shoppers | rate <=2% | 12.0% | 31.6% | 37.7% | 0.00469 |
| Shoppers | rate <=5% | **12.8%** | **32.7%** | **40.1%** | 0.00471 |

Feedback-adaptive promotion trên Shoppers đạt 16.6% macro, 39.3% heldout và 36.3% calibration reduction nhưng drift tăng lên 0.02056. Do đó nó là điểm high-validity/high-drift, chưa phải operating point chính.

Trên một seed Default khác, semantic coverage mở rộng vẫn giữ topology contribution: dynamic đạt macro/heldout/calibration error `0.05179/0.01271/0.01207`, frozen active set đạt `0.05753/0.01541/0.01187`, còn frozen state đạt `0.05907/0.01807/0.01328`. Dynamic tốt hơn frozen khoảng 10.0% macro và 17.5% heldout, dù calibration kém frozen nhẹ. Vì vậy gain không chỉ đến từ việc thêm nhiều rule actions.

### 24.3. Interior action thay boundary projection

Action hiện tại đưa inequality đúng lên decision boundary, ví dụ `x <= c` thành `x=c` hoặc `x <= y` thành `x=y`. Một denoising update nhỏ sau đó có thể làm constraint vi phạm trở lại. Actuator mới dùng scale-normalized feasible margin:

\[
x \le c \Rightarrow x^*=c-m\,\sigma_x,\qquad
x \ge y \Rightarrow x^*=y+m\,\sigma_x.
\]

`m` được quét trong `{0, 0.02, 0.05, 0.10, 0.20}`. Đây là một proximal/interior correction nhỏ, không hard-clamp row tới cuối trajectory. Hypothesis là margin vừa phải tăng persistence và giảm số re-actions; margin lớn sẽ tăng drift và bị loại.

### 24.4. Paired trajectory portfolio

Một diagnostic compute-heavy tạo nhiều middle-guidance trajectories từ cùng noise và chọn branch theo từng row bằng graph discovery trên train, kNN density và edit cost. Evaluation graph không tham gia selection. Mục đích không phải dùng portfolio tám nhánh làm final method, mà kiểm tra liệu một global strength đang che giấu heterogeneity giữa rows. Nếu portfolio thắng rõ, method tiếp theo là một short-lookahead controller; nếu không, branch search bị loại.

Kết quả Shoppers không ủng hộ phát triển branch search lớn: best single branch đạt macro/heldout error `0.05490/0.01317`, còn portfolio tốt nhất về macro đạt `0.05483/0.01423`. Portfolio giảm drift từ `0.02804` xuống `0.00440` nhưng mất heldout gain. Kết luận: per-row selection có thể là fidelity controller, không phải nguồn semantic gain chính; chưa nên tăng compute cho tree search.

### 24.5. Kiểm tra utility/fidelity bằng baseline evaluator chung

Trên Diabetes, cùng train/test split và evaluator:

| Method | Shape error | Trend error | Detection | XGBoost AUC | Heldout semantic error |
| --- | ---: | ---: | ---: | ---: | ---: |
| TabDiff | **1.884** | **11.302** | **0.9363** | 0.6459 | 0.01317 |
| Dynamic graph | 1.919 | 11.831 | 0.9314 | 0.6466 | 0.00888 |
| Dynamic graph + repair | 1.921 | 11.805 | 0.9384 | **0.6469** | 0.00843 |
| Post-repair | 1.887 | 11.630 | 0.9229 | 0.6434 | **0.00838** |

Dynamic guidance giảm semantic error khoảng 32.6% so với TabDiff và giữ downstream AUC, nhưng shape/trend/detection hơi xấu hơn. Post-repair vẫn thắng nhẹ semantic error; Diabetes tiếp tục là negative control cho claim rằng mid-step injection luôn cần thiết.

### 24.6. Jobs mới

| Job | Nội dung | Câu hỏi quyết định |
| --- | --- | --- |
| `1062146[0-2]` | Calibrated promotion không cap trên ba datasets | Coverage có tăng gain mà vẫn giữ calibration không? |
| `1062147[0-1]` | Dynamic/frozen/frozen-state sau promotion | Gain có thực sự cần topology động không? |
| `1062151[0-2]` | Interior margin frontier trên ba datasets | Action vào feasible interior có bền hơn boundary action không? |
| `1062152[0-1]` | Margin × topology control | Margin có làm mất contribution của dynamic graph không? |
| `1062154[0-1]` | Feedback-adaptive paired trajectory portfolio | Row-specific strength có đáng phát triển thành controller không? |
| `1062155[0-1]` | Portfolio trên wide-strength runs đã có | Diagnostic nhanh bằng paired outputs cũ |
| `1062156[0-1]` | Calibration rate và persistence frontier | Chọn vùng promotion an toàn quanh 2% |
| `1062157[0-1]` | Factorial margin × promotion × feedback | Hai cải tiến có bổ sung nhau hay chỉ trùng hiệu ứng? |
| `1062158` | Aggregate toàn bộ frontier mới | Artifact chung, không tự chọn theo audit metric |
| `1062162[0-2]` | FM semantic review cho empirical score-only edges | Loại correlation trước khi edge được quyền action |
| `1062163[0-2]` | Dynamic guidance với semantic-promoted graph | Đo gain thực sau safety filter trên ba datasets |
| `1062165[0-2]` | Semantic review độc lập bằng model thứ hai | Đo model disagreement, tránh một judge tự quyết |
| `1062166[0-2]` | Giao của hai semantic reviews | Chỉ promote edge được cả hai model xác nhận |
| `1062167[0-2]` | Dynamic guidance với consensus graph | Main safety-filtered promotion candidate |

Chỉ cấu hình thỏa đồng thời ba điều kiện mới được xác nhận bằng seed mới: tốt hơn unconditional, tốt hơn final-only counterpart hoặc frozen topology trên metric liên quan tới claim, và không làm xấu fidelity/exception calibration quá mức. Mục tiêu 20–30% không được đạt bằng cách hard-enforce các empirical relations vốn có exception hợp lệ.

## 25. Graph size, connectivity và multi-column coordination

### 25.1. Vấn đề cần tách rõ

Graph Shoppers đang dùng có 27 typed edges trên 15/18 cột, nhưng chỉ có 11 unique undirected links, 5 connected components và không có edge arity từ ba cột trở lên. Vì vậy “dynamic” hiện chủ yếu đến từ active set thay đổi theo row/timestep; static graph structure vẫn khá đơn giản.

Không thể tăng graph tùy ý rồi dùng chính các edge mới làm evaluator. Thí nghiệm mới giữ evaluator ẩn hiện tại cố định, cùng checkpoint, sampling seed và sample count. Các empirical edge bổ sung chỉ được làm `score_only`: chúng có thể ngăn một candidate action tạo ra violation mới nhưng không được tự khởi phát sửa row. Do đó gain, nếu có, phản ánh action coordination tốt hơn thay vì hard-enforce thêm correlation.

### 25.2. Controlled variants

| Variant | Edges | Score-only | Components | Largest component | Arity ≥ 3 | Thay đổi chính |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `base` | 27 | 7 | 5 | 4 | 0 | Catalog hiện tại |
| `role_expanded` | 27 | 15 | 5 | 4 | 0 | Chuyển monitor ổn định thành action guardrails; không đổi topology |
| `bridge_guardrails` | 28 | 8 | 4 | 7 | 0 | Thêm một empirical bridge qua calibration-only selection |
| `dense_connected` | 33 | 21 | 4 | 7 | 0 | Tăng edge budget và guardrail coverage |
| `validated_hyperedges` | 29 | 9 | 4 | 6 | 2 | Thêm hai multi-column edges từ protocol cũ đã qua critic/judge |
| `dense_hypergraph` | 35 | 23 | 3 | 10 | 2 | Kết hợp dense graph, bridge và multi-column edges |

Hai hyperedge cũ gồm một relation bốn cột nối `PageValues/ProductRelated/ProductRelated_Duration` với `BounceRates`, và một conditional relation ba cột trên `VisitorType/ExitRates/BounceRates`. Chúng vẫn chỉ có quyền score để tránh biến semantic uncertainty thành action authority.

Ba contrast quyết định:

1. `role_expanded` so với `base`: guardrail coverage có ích dù connectivity không đổi hay không;
2. `bridge_guardrails` so với `base`: một component-joining edge có cải thiện action choice hay không;
3. `dense_hypergraph` so với `dense_connected`: multi-column coordination có thêm giá trị ngoài edge count hay không.

Mỗi variant chạy ở `graph_weight ∈ {1,4}`. Nếu graph dày chỉ thắng ở weight lớn nhưng drift tăng, đó là reward scaling chứ chưa phải bằng chứng connectivity. Nếu `bridge_guardrails` hoặc `validated_hyperedges` thắng với cùng/ít edges hơn `role_expanded`, topology hoặc high-order structure mới là lời giải thích đáng tin hơn.

### 25.3. Connectivity-aware knowledge discovery

`knowledge_graph_protocol.py` có thêm profile `connectivity_aware`. LLM chỉ nhận schema, typed programs, component membership và aggregate statistics; không nhận raw rows hoặc heldout statistics. Ba proposal roles tìm:

1. semantic/process bridges giữa các components;
2. conjunction/hyperedge mà các atoms thực sự jointly necessary;
3. coordination guardrail giúp chọn giữa nhiều repair endpoints.

Mọi proposal vẫn qua executable typecheck, discovery support, counterexample split, calibration split, adversarial critic và semantic judge. Giảm component count không phải acceptance criterion: edge không có process semantics phải bị loại dù làm graph đẹp hơn.

### 25.4. Jobs ngày 09/10/2026

| Job | Nội dung | Output dự kiến |
| --- | --- | --- |
| `1062181` | Tạo sáu controlled graph variants | `outputs/connectivity_variants_1062181/` |
| `1062190[0-5]` | Đo readability/violation stability cho cả 6 graph bằng cùng seed | `outputs/connectivity_trajectory_*_1062190_*/` |
| `1062191[0-11]` | Paired dynamic guidance, 6 graph × 2 graph weights | `outputs/connectivity_guidance_*_1062191_*/` |
| `1062192` | Tổng hợp gain với fixed independent evaluator | `outputs/connectivity_aggregate_1062192/RESULTS.md` |
| `1062182` | LLM connectivity-aware graph generation, token cap 180k | `outputs/connectivity_graph_generation_1062182/` |
| `1062216[0-5]` | Per-edge readability schedule cho cả actuator và score-only guardrail | `outputs/connectivity_edge_schedule_*_1062216_*/` |
| `1062217` | Tổng hợp riêng trajectory-aware schedule | `outputs/connectivity_edge_aggregate_1062217/RESULTS.md` |

Tiêu chí kết luận: graph expansion chỉ được giữ nếu giảm heldout/calibration error trên evaluator cũ, không tăng KS/TV hoặc drift đáng kể, và gain không chỉ xuất hiện khi dùng dense graph làm evaluator của chính nó. Nếu mọi variant gần như bằng `base`, bottleneck nằm ở semantic actuator coverage hoặc action mechanism, không nằm ở số cạnh hay connected components.

Trajectory probe paired cho thấy toàn bộ 35 edges trong `dense_hypergraph` đều đạt readability criterion ở ít nhất một timestep. Hai prior complex hyperedges có cutoff rất khác nhau: khoảng `t=0.551` cho relation bốn cột và `t=0.204` cho conditional relation ba cột. Batch đầu `1062209` phát hiện score-only guardrail chưa được timestep-gated trong action scorer và vì vậy không dùng làm bằng chứng. Implementation đã được sửa; batch thay thế `1062216` dùng per-edge cutoff cho cả actuator lẫn guardrail để tránh cho một hyperedge late-readable ảnh hưởng action selection từ quá sớm. Batch common-schedule vẫn được giữ như một aggressive control.

### 25.5. Kết quả: connectivity đơn thuần không phải bottleneck

Kết quả paired với common schedule:

| Variant | Weight | Macro error | Heldout error | Gain macro so với base | Gain heldout | Numeric KS | Drift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `base` | 1 | 0.05467 | 0.01190 | 0.0% | 0.0% | 0.02422 | 0.00794 |
| `role_expanded` | 1 | 0.05467 | 0.01190 | 0.0% | 0.0% | 0.02422 | 0.00794 |
| `bridge_guardrails` | 1 | 0.05462 | 0.01184 | +0.1% | +0.5% | 0.02422 | 0.00751 |
| `dense_connected` | 1 | 0.05462 | 0.01184 | +0.1% | +0.5% | 0.02421 | 0.00774 |
| `validated_hyperedges` | 1 | 0.05986 | 0.01300 | −9.5% | −9.2% | 0.02370 | 0.00760 |
| `dense_connected` | 4 | 0.05977 | 0.01287 | −9.3% | −8.1% | 0.02332 | 0.00789 |

Tăng số guardrail từ 7 lên 15 nhưng giữ nguyên topology cho output giống hệt base, cho thấy các edge bổ sung không thay đổi action choice. Một empirical bridge làm component count giảm 5→4 chỉ tạo improvement khoảng 0.1–0.5%, quá nhỏ để thành contribution. Strong graph weight hoặc hyperedge được bật quá sớm có thể đổi action sai và làm error tăng khoảng 8–10%.

Sau khi score-only guardrail được timestep-gated đúng theo edge readability:

| Variant | Edges | Components | Macro error | Heldout error | Macro gain | Heldout gain | Actions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `base` | 27 | 5 | 0.05582 | 0.01330 | 0.0% | 0.0% | 358 |
| `bridge_guardrails` | 28 | 4 | 0.05580 | 0.01332 | +0.0% | −0.2% | 353 |
| `dense_connected` | 33 | 4 | 0.05576 | 0.01337 | +0.1% | −0.6% | 345 |
| `validated_hyperedges` | 29 | 4 | 0.05582 | 0.01330 | 0.0% | 0.0% | 358 |
| `dense_hypergraph` | 35 | 3 | 0.05576 | 0.01337 | +0.1% | −0.6% | 345 |

Per-edge timing loại được mức suy giảm do hyperedge nhưng không tạo positive gain. Nó giảm số action đáng kể so với common schedule, đổi lại semantic error tuyệt đối cao hơn. Vì vậy “nhiều edge hơn” và “ít connected components hơn” bị bác bỏ như một mechanism đủ mạnh trong action layer hiện tại.

Connectivity-aware LLM generation dùng GPT-4.1/GPT-5.4, 19 calls và 177,417 tokens: 50 proposals, 7 retained, gồm bốn candidates arity 3–5. Leave-one-atom-out audit trên discovery/counterexample/calibration cho kết quả:

| Hyperedge | Arity | Necessary antecedents | Nonredundant requirements | Empirically minimal |
| --- | ---: | ---: | ---: | --- |
| `PageValues ∧ Informational_Duration → Administrative` | 3 | 2/2 | 1/1 | Có |
| `VisitorType ∧ PageValues ∧ BounceRates → Product activity` | 5 | 1/3 | 0/2 | Không |
| `OperatingSystems ∧ Administrative → Duration ∧ Browser` | 4 | 2/2 | 2/2 | Có về thống kê, còn yếu về semantics |
| `PageValues ∧ Information ∧ Administration → Product activity` | 5 | 0/3 | 0/2 | Không |

Hai trong bốn hyperedges phức tạp là conjunction bị inflate: phần lớn atom không cần thiết. Hai edge qua minimality vẫn chưa đủ semantic authority; đặc biệt device/browser relation có nguy cơ chỉ là platform correlation. Chúng chưa được promote thành actuator.

Kết luận method: graph connectivity chỉ hữu ích nếu action layer thực sự ra quyết định jointly trên một component. Scorer hiện tại vẫn sửa từng violated edge và dùng hàng xóm như veto độc lập; do đó nối components không tự tạo message passing hoặc coordinated action. Hướng tiếp theo hợp lý là component-level joint candidate selection hoặc local energy trên active subgraph, với edge-count/connectivity variants ở đây làm ablation âm. Không nên tiếp tục tăng cạnh bằng threshold hoặc LLM prompt mà không thêm atom-minimality và semantic authority checks.

## 26. End-to-end graph-consistent diffusion fine-tuning

### 26.1. Trạng thái trước thí nghiệm

Các main results dynamic graph hiện tại đều dùng TabDiff checkpoint `ema_model_epoch_8000.pt` đã train trên từng dataset. Denoiser và learned noise schedules được load rồi đặt ở evaluation mode; graph chỉ tham gia sampling/action selection. Một số nhánh cũ đã train condition adapter hoặc phần MLP, nhưng chưa có thí nghiệm nào fine-tune toàn bộ TabDiff denoiser bằng typed dynamic-graph objective.

Đây là gap method hợp lý: inference-time action sửa output của một pretrained model chưa từng được tối ưu để tạo clean preview có semantic structure ổn định ở middle timesteps. End-to-end training có thể làm graph state đọc được sớm hơn, giảm excess violations trước guidance và giúp middle-step intervention cần ít action hơn. Tuy nhiên phải so với control train thêm cùng số updates; nếu không thể phân biệt contribution của graph loss với continued training.

### 26.2. Conservative auxiliary objective

Fine-tuning bắt đầu từ cùng pretrained Shoppers checkpoint. Toàn bộ 10.65M denoiser parameters được train; numerical/categorical noise schedules được frozen để không thể thay đổi noise process nhằm game objective.

Với mỗi training row và authorized edge, graph loss chỉ được áp dụng khi clean real row đã nằm trong applicability scope và thỏa constraint. Real rows là observed exceptions không bị relabel hoặc ép sửa. Tại timestep lấy mẫu, denoiser dự đoán `x_hat_0`; numerical predicates được chuyển sang encoded QuantileTransformer space và dùng fuzzy differentiable satisfaction. Objective là:

\[
L = L_{\text{TabDiff}} + \lambda L_{\text{graph-valid clean rows}}.
\]

11/12 actuate edges của Shoppers được compile; `BounceRates <= ExitRates` tạm bỏ khỏi auxiliary loss vì hai cột dùng feature-specific nonlinear transforms, nên so sánh trực tiếp trong encoded space không bảo toàn raw-space order. Edge này vẫn nằm trong sampling và independent evaluation graph.

Điểm hạn chế cần ghi rõ: objective hiện dạy consequent consistency trên clean-valid examples, chưa trực tiếp học graph-conditioned generation request và chưa dùng categorical semantic loss. Đây là end-to-end mechanism pilot, không phải full training formulation cuối cùng.

### 26.3. Cấu hình exploratory

Smoke job `1062228` chạy sạch: ba updates mất 3.4 giây, peak GPU memory 1.18 GB, graph loss ban đầu `0.0088` so với total denoising loss `3.33`, parameter displacement sau một epoch `0.00078`. Dựa trên scale này, sweep dùng:

| Variant | Graph weight | Timestep của graph loss | Mục đích |
| --- | ---: | --- | --- |
| `pretrained` | — | — | Checkpoint gốc |
| `continue` | 0 | — | Extra-training control |
| `graph1_mid` | 1 | `[0.15, 0.60]` | Mild graph objective |
| `graph5_mid` | 5 | `[0.15, 0.60]` | Trung bình, khoảng 1.3% initial total loss |
| `graph20_mid` | 20 | `[0.15, 0.60]` | Aggressive, khoảng 5% initial total loss |
| `graph5_uniform` | 5 | `[0,1]` | Kiểm tra middle-specific curriculum |

Mỗi fine-tune chạy 500 epochs/1,500 updates, learning rate `1e-5`, EMA `0.997`, một seed. Sau train, mỗi checkpoint được trajectory-probe lại bằng cùng seed và 8,192 samples, rồi đánh giá cả unconditional generation, dynamic middle guidance, guided-plus-final repair và final-only repair bằng frozen independent evaluator.

### 26.4. Jobs

| Job | Nội dung | Output |
| --- | --- | --- |
| `1062230[0-4]` | Continued training và bốn graph-loss configurations | `outputs/e2e_graph_finetune_*_1062230_*/` |
| `1062231[0-5]` | Paired trajectory/readability probe | `outputs/e2e_graph_trajectory_*_1062231_*/` |
| `1062232[0-5]` | Unconditional + dynamic guidance + repair evaluation | `outputs/e2e_graph_eval_*_1062232_*/` |
| `1062233` | Aggregate pretrained/control/graph-finetuned results | `outputs/e2e_graph_aggregate_1062233/RESULTS.md` |

Điều kiện giữ hướng end-to-end: graph variants phải tốt hơn cả pretrained và `continue`; không chỉ giảm violation trên chính training graph mà phải giảm heldout/calibration error trên evaluator cũ; KS/TV không xấu rõ; và ít nhất một trong hai tín hiệu phải xuất hiện: unconditional validity tốt hơn hoặc readability cutoff dịch về timestep sớm hơn. Nếu graph fine-tuning chỉ tái tạo hiệu ứng repair hoặc làm mất coverage, training-time integration không đáng thêm vào main method.

### 26.5. Kết quả end-to-end đầu tiên

Toàn bộ jobs `1062230`–`1062233` hoàn tất sạch. Bảng dưới dùng 8,192 mẫu, cùng seed, cùng independent evaluation graph; `continue` là control train thêm cùng số updates nhưng không có graph loss.

| Variant | Output | Macro error | Heldout error | Calibration error | Numeric KS | Cat. TV | Actions |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pretrained | Unconditional | 0.06427 | 0.01695 | 0.01720 | **0.02534** | **0.02203** | 0 |
| Continue | Unconditional | 0.06493 | 0.01758 | 0.01765 | 0.02553 | 0.02229 | 0 |
| GraphFT, `lambda=5`, middle | Unconditional | **0.06268** | **0.01628** | **0.01454** | 0.02674 | 0.02237 | 0 |
| Pretrained | Dynamic middle guidance | **0.05915** | **0.01336** | 0.01384 | **0.02557** | **0.02177** | 368 |
| GraphFT, `lambda=5`, middle | Dynamic middle guidance | 0.05924 | 0.01367 | **0.01242** | 0.02713 | 0.02233 | **245** |
| Pretrained | Final repair | 0.05840 | **0.01282** | 0.01316 | **0.02547** | **0.02203** | 258 |
| GraphFT, `lambda=5`, middle | Final repair | **0.05829** | 0.01334 | **0.01182** | 0.02707 | 0.02237 | **202** |

So với pretrained unconditional, GraphFT `lambda=5` giảm macro error 2.48%, heldout error 3.96% và calibration error 15.45%; continued training làm cả ba xấu hơn. Vì vậy graph loss thực sự tạo tín hiệu, không chỉ hưởng lợi từ train thêm. Tuy nhiên Numeric KS xấu 5.53% và Cat. TV xấu 1.56%.

Sau dynamic guidance, GraphFT giảm 33.42% số actions và calibration error tốt hơn 10.29%, nhưng macro gần như hòa và heldout error xấu hơn pretrained-guided 2.34%. Nói cách khác, model đã internalize một phần các actuator relations, nhưng chưa tạo complementary gain sau sampling guidance. `lambda=20` và graph loss trên toàn bộ timestep giảm macro nhiều hơn nhưng làm Numeric KS xấu khoảng 29–32%; hai cấu hình này bị loại.

Readability cutoff trung bình gần như không dịch chuyển (`0.291` sang `0.293`). Do đó chưa có bằng chứng graph fine-tuning làm semantic state đọc được sớm hơn. Kết luận tạm thời: giữ training-time integration như nhánh phụ để giảm action burden; chưa dùng làm main contribution.

### 26.6. Common quality/utility evaluation

Các output ở trên được chấm lại bằng cùng evaluator gồm SDMetrics shape/trend, AlphaPrecision/BetaRecall, detection, XGBoost utility và DCR.

| Method | Shape error | Trend error | Alpha precision | Beta recall | Detection | XGBoost AUC | Heldout semantic error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TabDiff | **1.307** | 1.548 | 97.929 | **50.518** | 0.9699 | 0.9054 | 0.01691 |
| TabDiff + dynamic | 1.348 | 2.346 | 98.003 | 50.258 | 0.9812 | 0.9018 | 0.01331 |
| TabDiff + final repair | 1.303 | **1.541** | 97.949 | 50.479 | 0.9651 | 0.9034 | **0.01278** |
| GraphFT | 1.367 | 1.577 | 97.983 | 50.068 | 0.9691 | 0.9079 | 0.01623 |
| GraphFT + dynamic | 1.386 | 2.084 | **98.091** | 49.880 | 0.9686 | 0.9039 | 0.01363 |
| GraphFT + final repair | 1.387 | 1.530 | 98.011 | 50.085 | 0.9793 | **0.9109** | 0.01329 |

Dynamic guidance trên pretrained giảm heldout semantic error 21.25%, nhưng shape error xấu 3.09% và trend error xấu 51.57%; final repair giảm semantic error 24.43% trong khi shape/trend gần như giữ nguyên. Đây là phản ví dụ quan trọng: middle-step injection hiện tại tạo semantic gain nhưng làm đổi pairwise structure nhiều hơn repair cuối. GraphFT làm mức suy giảm trend của dynamic nhỏ hơn (`+34.65%` thay vì `+51.57%`) và tăng AUC khi dùng final repair, nhưng chưa giải quyết bottleneck chính.

CTGAN và TVAE local baselines vẫn kém xa TabDiff về fidelity trên Shoppers: shape error lần lượt 14.26 và 16.69 so với khoảng 1.3–1.7 của TabDiff trong các batches tương ứng. Những số này chỉ dùng để xác nhận backbone diffusion mạnh hơn older GAN/VAE baselines; semantic absolute values không được trộn với bảng trên vì catalog/evaluator version khác.

### 26.7. Hard versus near-hard training objective

Một batch mới tách graph loss theo semantic hardness. Hard-only `lambda=5/10` làm Numeric KS tăng mạnh lên `0.02879/0.03206`; near-hard-only giữ KS gần hơn (`0.02580`) nhưng làm heldout validity xấu. Sau matched strong guidance, hard-only `lambda=5` đạt heldout error `0.01398` và KS `0.02940`; near-hard-only đạt `0.01468/0.02618`. Cả hai đều kém all-edge checkpoint epoch 100 (`0.01306/0.02620`) và final-only baseline về semantic error. All-edge `lambda=5` tốt hơn từng phần, nhưng kết quả không ủng hộ thêm semantic-hardness routing vào main method.

## 27. Đánh giá trạng thái và batch tiếp theo

### 27.1. Trạng thái claim

Project đã có đủ bằng chứng để bắt đầu viết problem statement, method overview, benchmark protocol và negative findings. Chưa đủ để claim SOTA hoặc khóa main-result table.

Các phần đã có bằng chứng tương đối ổn:

1. Clean preview ở early steps không đủ tin cậy; edge-specific middle timing là cần thiết.
2. Dynamic row/timestep graph tốt hơn frozen topology trong các paired controls, nổi bật trên Default: macro error giảm khoảng 10.0% và heldout error giảm 17.5% so với frozen active graph.
3. Dynamic guidance giữ TabDiff-level marginal fidelity tốt hơn CTGAN/TVAE và có thể giảm heldout semantic error 20–30% trên một số datasets.
4. Dense/connectivity-expanded graph không tự tạo gain; edge quality và action mechanism quan trọng hơn edge count.
5. Final repair là đối thủ rất mạnh và hiện thắng trên Shoppers common metrics; mọi claim phải là trade-off hoặc chỉ rõ setting nơi trajectory intervention thắng.

Các phần chưa đủ:

1. Chưa có faithful same-task rerun của Harpoon/CTDF/LCATS trên cùng graph và split; các official-code outputs hiện chỉ là external evidence hoặc khác task.
2. Chưa chứng minh middle-step injection cải thiện cả semantic validity và pairwise fidelity so với final repair trên Shoppers.
3. End-to-end graph training chưa tạo earlier readability và chưa bổ sung gain sau dynamic guidance.
4. Main operating point còn phụ thuộc dataset; cần một selection rule dựa trên calibration, không chọn bằng heldout audit.

### 27.2. Jobs đã submit để đóng các câu hỏi trên

| Jobs | Nội dung | Câu hỏi |
| --- | --- | --- |
| `1062264` → corrected `1062339` → `1062340` | Checkpoint frontier 100–500 epochs, scale khớp main run | Early stopping có giữ semantic gain nhưng giảm KS drift không? |
| corrected `1062341` → `1062342` | `numeric_guidance_scale in {1,2,4}` × action budget `{1,4}` × pretrained/GraphFT | GraphFT có đạt cùng validity với guidance yếu hơn hoặc ít action hơn không? |
| `1062356` | Common evaluator cho toàn bộ recalibration frontier | Operating point nào giữ shape/trend/utility tốt nhất? |
| `1062285` → corrected `1062343` → `1062344` | Hard-only, near-hard-only và matched strong guidance | Semantic hardness nào gây drift/generalization failure? |
| `1062269` | Common evaluator cho six core outputs | Đã hoàn tất; cho thấy final repair hiện là baseline cần vượt |
| `1062365` → `1062366/1062367` | Confirmation trên duy nhất GPU T4: pretrained/epoch100/epoch500 × bốn controllers | Loại sai khác do GPU architecture và chốt Pareto frontier |
| `1062375` → `1062376` | Weak middle shaping + sparse final certification trên T4 | Hybrid có giữ utility của middle injection và validity của final repair không? |

Batch `1062267/1062268` không dùng làm main evidence vì sweep ban đầu thay class-specific scale trong khi `feedback_adaptive` đọc `numeric_guidance_scale`. Lỗi đã được audit từ trace và corrected batch ở trên thay thế nó.

Corrected sweep tiếp tục cho thấy một vấn đề reproducibility: cùng seed/checkpoint cho unconditional outputs giống nhau trong cùng GPU family, nhưng khác nhẹ giữa T4 và A10. Vì vậy corrected sweep `1062341` chỉ dùng để chọn vùng scale; mọi comparison cuối được rerun bằng `--gres=gpu:t4:1` trong `1062365`. Gain trong từng task vẫn được tính so với paired unconditional output của chính task đó.

### 27.3. Quyết định sau batch corrected

Nếu GraphFT với scale 1–2 đạt heldout error gần pretrained scale 4, giảm actions và khôi phục trend error, giữ training-time integration làm phần thứ hai của method: graph-consistent denoiser + lightweight dynamic controller. Nếu mọi GraphFT point vẫn bị final repair trội đồng thời về semantic và fidelity, bỏ end-to-end branch khỏi main method và tập trung contribution vào verified dynamic active graph, cùng một fidelity-aware middle controller. Nếu checkpoint 100 tạo Pareto point tốt hơn epoch 500, chọn checkpoint bằng calibration error + KS trên validation, không dùng heldout audit.

## 28. T4-matched confirmation và quyết định hiện tại

### 28.1. Controller operating point

Batch `1062365` khóa toàn bộ runs vào GPU T4 để loại sai khác architecture. Bốn controller profiles dùng cùng graph, schedule, seed và action mechanism.

| Model | Profile | Macro error | Heldout error | Calibration error | Numeric KS | Cat. TV | Actions |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pretrained | scale 1, budget 4 | 0.05965 | 0.01382 | 0.01337 | 0.02541 | 0.02191 | 616 |
| Pretrained | **scale 2, budget 4** | **0.05868** | **0.01298** | **0.01322** | **0.02537** | 0.02188 | 457 |
| Pretrained | scale 4, budget 1 | 0.05925 | 0.01348 | 0.01403 | 0.02561 | **0.02174** | **357** |
| Pretrained | scale 4, budget 4 | 0.05915 | 0.01336 | 0.01384 | 0.02557 | 0.02177 | 368 |
| Epoch 100 GraphFT | scale 1, budget 4 | 0.06013 | 0.01356 | 0.01231 | 0.02629 | 0.02218 | 578 |
| Epoch 100 GraphFT | **scale 2, budget 4** | **0.05919** | **0.01294** | **0.01223** | **0.02618** | 0.02215 | 415 |
| Epoch 100 GraphFT | scale 4, budget 1 | 0.05979 | 0.01308 | 0.01285 | 0.02625 | 0.02208 | **291** |
| Epoch 100 GraphFT | scale 4, budget 4 | 0.05972 | 0.01306 | 0.01270 | 0.02620 | **0.02204** | 305 |

Pretrained scale 2 là point tốt nhất trong nhóm không fine-tune: giảm heldout error 23.4% so với unconditional và gần final repair (`0.01298` so với `0.01282`) trong khi KS/TV tốt hơn nhẹ. Epoch-100 scale 2 giảm heldout thêm 0.32%, calibration error giảm 7.5% và actions giảm 9.2% so với pretrained scale 2; đổi lại macro error xấu 0.87%, KS xấu 3.2% và TV xấu 1.3%. Hai điểm nằm trên một Pareto frontier, không có dominance.

Epoch 500 không còn cạnh tranh: calibration tốt nhưng heldout error và fidelity đều xấu hơn. Nếu giữ GraphFT, chỉ giữ epoch 100 như secondary variant; epoch 500 bị loại.

### 28.2. Common metrics cho các point chính

| Method | Shape error | Trend error | Alpha precision | Beta recall | XGBoost AUC | Heldout semantic error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pretrained unconditional | 1.3074 | 1.5476 | 97.9289 | **50.5176** | 0.9054 | 0.01691 |
| Pretrained final repair | 1.3025 | 1.5411 | 97.9492 | 50.4785 | 0.9034 | **0.01278** |
| Pretrained middle, scale 1 | **1.2968** | **1.5343** | 97.9647 | 50.4362 | **0.9080** | 0.01377 |
| Pretrained middle, scale 2 | 1.3035 | 1.5437 | 97.9452 | 50.3760 | 0.9006 | 0.01293 |
| Epoch-100 final repair | 1.3354 | **1.5379** | 97.8670 | **50.3052** | 0.9046 | 0.01293 |
| Epoch-100 middle, scale 2 | **1.3280** | 1.5471 | 97.8304 | 50.2327 | **0.9074** | **0.01289** |

Scale 1 là utility/fidelity-oriented point: shape, trend và AUC đều tốt hơn final repair nhưng semantic error còn cao hơn khoảng 7.8%. Scale 2 gần final repair về semantics nhưng không thắng final repair toàn cục. Epoch-100 scale 2 thắng final repair của chính checkpoint đó về shape, heldout error và AUC, nhưng thua pretrained final repair về semantic error và fidelity. Vì vậy GraphFT có interaction thật với middle guidance, nhưng effect size hiện quá nhỏ để làm claim thứ hai độc lập.

### 28.3. Direct topology controls

| Model | Topology | Macro error | Heldout error | Calibration error | Actions |
| --- | --- | ---: | ---: | ---: | ---: |
| Pretrained | **Dynamic** | **0.05868** | **0.01298** | **0.01322** | 457 |
| Pretrained | Frozen | 0.05924 | 0.01307 | 0.01339 | 517 |
| Pretrained | Frozen state | 0.05945 | 0.01309 | 0.01363 | **392** |
| Epoch 100 | **Dynamic** | **0.05919** | **0.01294** | **0.01223** | 415 |
| Epoch 100 | Frozen | 0.05960 | 0.01302 | 0.01236 | 495 |
| Epoch 100 | Frozen state | 0.06037 | 0.01334 | 0.01279 | **360** |

Dynamic active graph thắng frozen graph trên macro, heldout và calibration error cho cả hai model states. Trên pretrained, gain so với frozen khoảng 0.94% macro và 0.72% heldout; trên epoch 100 khoảng 0.70% và 0.66%. So với frozen-state, gain lớn hơn, đặc biệt epoch 100 giảm heldout khoảng 3.0%. Effect trên Shoppers nhỏ nhưng nhất quán; bằng chứng mạnh hơn vẫn là Default control trước đó, nơi dynamic thắng frozen khoảng 10.0% macro và 17.5% heldout.

### 28.4. Weak-middle plus final certification

Hybrid scale 2 gần như tái tạo final-only output. Trên pretrained, heldout error `0.01281` so với `0.01278` final-only, nhưng cần 468 thay vì 258 actions và AUC thấp hơn. Trên epoch 100, hybrid cải thiện nhẹ so với final-only của cùng checkpoint: heldout `0.01287` so với `0.01293`, shape `1.3293` so với `1.3354`, AUC `0.9052` so với `0.9046`; đổi lại trend/coverage hơi xấu và actions tăng từ 236 lên 435. Effect quá nhỏ so với cost, nên hybrid không được giữ làm default method.

### 28.5. Quyết định

1. Main method tiếp tục là verified dynamic graph + row/timestep active-set recomputation + middle clean-preview action. GraphFT không phải phần bắt buộc.
2. Hai operating points được giữ: scale 1 cho fidelity/utility, scale 2 cho balanced semantic validity. Scale 4 chỉ là high-strength ablation vì làm trend error tăng mạnh.
3. Final repair là baseline mạnh nhất trên Shoppers. Claim hợp lệ hiện tại là dynamic graph tạo một semantic–fidelity frontier và có lợi trên một số datasets/settings, chưa phải universal dominance.
4. End-to-end GraphFT epoch 100 được giữ như optional secondary experiment; epoch 500, hard-only, near-hard-only, dense graph và hybrid bị loại khỏi main pipeline.
5. Chưa claim SOTA. Điều còn thiếu quan trọng nhất là faithful same-task Harpoon/CTDF và post-repair baseline trên cùng frozen graph, sau đó paired multi-seed confirmation trên Default và Shoppers.

Jobs `1062391[0-3]` và `1062392` đã được submit để bổ sung hai T4 seeds cho scale-2 dynamic/frozen/final-repair comparison; kết quả sẽ được gộp với seed hiện tại thành confirmation ba seeds.

## 29. Component-level dynamic action planner

### 29.1. Động lực từ failure mode của actuator cũ

Confirmation ba seed đã hoàn tất trong `1062391` và corrected aggregate `1062396`. Dynamic scale 2 thắng frozen topology trên heldout error ở cả ba seeds, trung bình `0.93% ± 0.74%`, nhưng vẫn kém final repair `1.22% ± 0.34%`. Vì effect dynamic topology lặp lại nhưng nhỏ, bước tiếp theo không tiếp tục tăng guidance weight hoặc số edges một cách cơ học.

Audit trên 8,192 unconditional Shoppers rows cho thấy 578 rows vi phạm ít nhất một actuator edge; 224/578 rows (`38.8%`) đồng thời vi phạm ít nhất hai edges. Hai rules trên cùng cặp `Administrative–Administrative_Duration` cùng vi phạm ở 182 rows. Actuator cũ chọn action tuần tự cho từng edge, khóa column đã sửa và chỉ dùng các score-only edges như guardrail. Vì vậy một graph liên thông vẫn bị xử lý như danh sách constraints độc lập; đây cũng giải thích vì sao các thí nghiệm tăng connectivity trước đó gần như không tạo gain.

### 29.2. Kiến trúc mới

Ở mỗi timestep, method vẫn decode `x_hat_0(x_t)`, áp dụng edge-specific readability và tái tính applicability/violation cho từng row. Với một row, các gated edges được nối thành components nếu dùng chung column. Mỗi component sinh các candidate operations đã có: sửa consequent, sửa endpoint còn lại của column relation, deactivate antecedent khi hợp lệ, hoặc abstain. Beam search nhỏ tìm một joint correction:

\[
E(y;x,t)=\lambda_v\sum_{e\in C_{x,t}} w_e\,\mathbf{1}[e(y)\text{ violated}]
+\lambda_g\,\Delta\text{CollateralGraph}(y,x)
+\lambda_d\,\text{DensityCost}(y)
+\lambda_e\,\text{EditCost}(y,x).
\]

`CollateralGraph` chỉ phạt violations mới được candidate tạo ra; method không ép sửa các empirical exceptions đã tồn tại trước action. Joint target tiếp tục được inject vào clean prediction trước reverse update. Như vậy graph tham gia trực tiếp vào action planning, thay vì chỉ đóng vai trò gate.

Implementation ở `scripts/rowwise_reverse_action_guidance.py` với `--action-planner component_beam`. Default vẫn là `edge_local`, nên toàn bộ controls cũ không đổi. Smoke test cho thấy beam width 4 và 8 cho output giống nhau; beam 4 được giữ.

### 29.3. Full Shoppers result

8,192 samples, cùng T4/checkpoint/seed/graph/evaluator với main table:

| Method | Macro | Heldout | Calibration | KS | TV | Actions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unconditional | 0.06427 | 0.01695 | 0.01720 | **0.02534** | 0.02203 | 0 |
| Final repair | **0.05840** | **0.01282** | **0.01316** | 0.02547 | 0.02203 | **258** |
| Edge-local middle, scale 1 | 0.05965 | 0.01382 | 0.01337 | 0.02541 | 0.02191 | 616 |
| Component middle, scale 1 | 0.05884 | 0.01304 | 0.01335 | 0.02542 | **0.02181** | 602 |
| Edge-local middle, scale 2 | 0.05868 | 0.01298 | **0.01322** | 0.02537 | 0.02188 | 457 |
| Component middle, scale 2 | 0.05877 | 0.01293 | 0.01332 | 0.02541 | **0.02181** | 425 |
| Component scale 1 + final certification | 0.05856 | 0.01287 | 0.01324 | 0.02543 | **0.02181** | 622 |

Component planning đặc biệt có ích khi guidance yếu: scale-1 heldout error giảm từ `0.01382` xuống `0.01304`, đóng khoảng 78% khoảng cách giữa edge-local scale 1 và final repair. Ở scale 2, gain nhỏ hơn nhưng actions giảm 7.0% và TV tốt hơn. Graph guardrail weight `{1,5,10}` và violation energy `{30,100}` tạo output giống nhau, cho thấy result không nhạy với hai engineering constants này; chúng bị loại khỏi main ablation.

### 29.4. Common quality/utility result

| Method | Shape error | Trend error | XGBoost AUC | Heldout error |
| --- | ---: | ---: | ---: | ---: |
| Unconditional | 1.3074 | 1.5476 | 0.9054 | 0.01691 |
| Final repair | 1.3025 | 1.5411 | 0.9034 | **0.01278** |
| Component scale 1 | **1.2886** | **1.5313** | 0.8991 | 0.01299 |
| Component scale 1 + certification | **1.2886** | 1.5333 | **0.9089** | 0.01283 |
| Component scale 2 | 1.3005 | 1.5424 | 0.9033 | 0.01288 |

Component scale 1 + certification chỉ kém final repair khoảng 0.39% heldout error, nhưng shape tốt hơn 1.07%, trend tốt hơn 0.51% và XGBoost AUC tăng từ `0.9034` lên `0.9089`. Đây là Pareto point mạnh nhất hiện tại, chưa phải universal dominance vì action count/runtime lớn hơn.

Refinement một chiều quanh scale 1–2 cho thấy scale `1.5` tốt hơn hai endpoints. Online component scale 1.5 đạt macro `0.05871`, heldout `0.01280`, calibration `0.01318`, KS `0.02537`, TV `0.02181`; sau final certification các số tương ứng là `0.05850/0.01275/0.01309/0.02537/0.02181`. So với final-only, certified variant tốt hơn heldout khoảng 0.59%, calibration 0.55%, KS 0.39% và TV 0.98%, nhưng macro còn xấu 0.17% và actions tăng từ 258 lên 532.

Common evaluator xác nhận online scale 1.5 là balanced point tốt hơn: so với final repair, shape error giảm `1.3025 → 1.2925`, trend giảm `1.5411 → 1.5392`, XGBoost AUC tăng `0.9034 → 0.9049`, heldout semantic error giảm `0.012775 → 0.012755`. Certified scale 1.5 giảm heldout sâu hơn tới `0.012699`, nhưng AUC giảm xuống `0.9015`; do đó online scale 1.5 được ưu tiên làm main quality point, certified variant là high-validity point.

Lưu ý selection protocol: scale 1.5 được chọn sau khi đọc seed đầu. Batch `1062428[0-4]` và aggregate `1062429` đã rerun exact edge-local/component pair trên hai seeds mới. Component planner thắng edge-local heldout error trên cả ba seeds, trung bình `0.44% ± 0.24%`; đây là gain nhỏ nhưng lặp lại. Tuy nhiên component + final certification chỉ thắng final-only trên seed discovery và thua trên hai seeds mới; trung bình gain là `-1.05% ± 1.63%`. Do đó kết quả không ủng hộ claim SOTA so với final repair, nhưng ủng hộ contribution hẹp hơn rằng joint active-subgraph planning tốt hơn edge-local trajectory intervention.

Failure mode còn lại là action necessity: online controller can thiệp khoảng 527 lần cho 8,192 rows, trong khi final repair chỉ cần khoảng 268 lần. Một phần violations giữa trajectory tự biến mất dưới base denoiser. Batch `1062452[0-2]` triển khai defer-or-act policy: từ cùng current state, method tạo no-action và guided next states, decode clean preview ở bước kế tiếp, rồi chỉ giữ action nếu guided branch giảm weighted target-subgraph violation energy nghiêm ngặt so với no-action branch. Đây là one-step counterfactual lookahead, khác one-shot full-trajectory lookahead cũ ở chỗ nó chạy closed-loop tại mỗi action opportunity và có no-action counterfactual matched cho chính row đó.

Target-only lookahead được reroute sang idle A40 và hoàn tất trong `1062466`. Nó giữ 14/59 component actions, giảm intervention cost mạnh, nhưng làm online heldout error xấu từ `0.01999` lên `0.02063`; sau final certification vẫn kém final-only (`0.01986` so với `0.01950`). Global active-graph energy trong `1062470` không thay đổi bất kỳ quyết định nào. Horizon 2, 3 và 5 trong `1062475` cũng cho đúng cùng 14 accepted actions và cùng output. Endpoint diagnostic horizon 10/25/50 trong `1062487` chỉ giữ 10/57 actions; online heldout error `0.01913` vẫn kém final repair `0.01854`, còn guided-plus-final đạt `0.01890`. Kết luận: rollout lookahead là một action-efficiency control tốt nhưng không phải quality-improving architecture cho action family hiện tại.

Về execution infrastructure, fixed-T4 smoke job bị scheduler dự đoán chờ hơn một ngày dù các GPU khác đang idle. Exploratory jobs hiện dùng partition `aicentre` hoặc `aicentre-a100`, QoS `unprivileged` và time limit 30–45 phút; confirmation/main-table jobs vẫn dùng T4. Helper `slurm/submit_route.sh` chuẩn hóa bốn routes: `fast` cho A40, `fast-a100` cho A100, `opportunistic` cho bất kỳ GPU nào trong prioritized partition, và `t4` cho kết quả cần architecture matching.

### 29.5. Trajectory-certified graph portfolio

Hai actuator edges của Shoppers có quá ít final-positive examples trong trajectory probe (`r2_010`: 3; `r1_005`: 0) nên violation precision không thể được đánh giá ổn định. Một control chỉ bỏ quyền actuation của hai edges này nhưng giữ nguyên independent evaluation graph. Component scale 2 khi đó đạt heldout `0.01290`, macro `0.05862`, TV `0.02181` với 411 actions. Sau final certification, heldout còn `0.01285`, chỉ kém final repair khoảng 0.20%, trong khi KS/TV tốt hơn. Đây là portfolio calibration hợp lệ vì selection dùng trajectory/calibration statistics, không nhìn heldout labels.

### 29.6. Cross-dataset status và jobs mới

Default component scale 4 bước đầu đạt heldout `0.01238` so với final repair `0.01321`, macro `0.05622` so với `0.05838`, với KS gần như giữ nguyên. Exact edge-local control trên cùng T4/seed đang chạy trước khi attribution gain cho component planner.

| Job | Nội dung |
| --- | --- |
| `1062401[0-3]` | Smoke edge-local và component beam `{2,4,8}`; hoàn tất |
| `1062405[0-5]`, `1062406` | Full component strength/energy sweep và aggregate; hoàn tất |
| `1062412[0-2]` | Trajectory-certified actuation portfolio; hoàn tất |
| `1062415[0-3]` | Component planner trên Default và Diabetes |
| `1062419` | Common Shoppers quality/utility metrics; hoàn tất |
| `1062420[0-3]` | Refine guidance scale `{1.25,1.5,1.75,2.5}` quanh vùng tốt |
| `1062424[0-2]` | Exact edge-local cross-dataset controls |

Trạng thái claim: component planner đã tạo một Pareto improvement rõ trên Shoppers scale 1 và kết quả mạnh trên Default, nhưng chưa được gọi là SOTA. Điều kiện tiếp theo là exact paired controls, common metrics cho filtered/winner configuration, ít nhất ba seeds cho operating point đã chọn, và faithful same-task CTDF/Harpoon/post-repair baselines.

## 30. Structural action authority và precision-onset timing

### 30.1. Structural authority

Component planner ban đầu cho phép cả component chỉ có một violated edge. Điều này dùng cùng architecture nhưng không tận dụng thông tin topology nhiều hơn edge-local action. Job `1062484` thêm một authority rule tổng quát: middle intervention chỉ được phép nếu active component của row có ít nhất (K) gated edges; các component nhỏ hơn được defer cho final certifier.

| (K) | Middle actions | Online macro | Online heldout | Certified macro | Certified heldout | Certified KS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 267 | **0.053711** | **0.015906** | 0.053461 | 0.015756 | 0.026139 |
| 2 | 52 | 0.056079 | 0.017664 | **0.053314** | 0.015236 | **0.026112** |
| 3 | 2 | 0.058124 | 0.018272 | 0.053407 | **0.015144** | 0.026161 |
| Final repair | 0 middle / 126 final | — | — | 0.053407 | **0.015144** | 0.026161 |

(K=2) giảm 80.5% middle actions so với (K=1). Sau certification nó cải thiện macro và KS nhẹ so với final repair, nhưng heldout error xấu khoảng 0.6% và cần tổng cộng 137 actions thay vì 126. (K=3) gần như không còn middle action và tái tạo đúng final repair. Structural authority vì vậy là Pareto/efficiency control có thể giải thích được, nhưng không tạo dominance.

### 30.2. Precision-onset thay cho recall-gated onset

Trajectory schedule hiện tại mở edge ở timestep lớn nhất đồng thời đạt active/violation precision tối thiểu 0.90 và recall tối thiểu 0.80. Điều kiện recall phù hợp khi mục tiêu là đọc đầy đủ final graph tại một timestep, nhưng quá bảo thủ cho selective control: một row chưa được nhận diện có thể được xét lại ở bước sau; rủi ro trực tiếp nằm ở false positive trong các row đã được nhận diện.

Biến thể mới dùng minimum current support và Wilson lower confidence bound của violation precision. Chỉ actuator cutoffs thay đổi; score-only và monitor graph giữ nguyên để không confound guardrails/evaluator. Ví dụ với lower bound 0.65, sáu actuator edges mở sớm hơn từ khoảng 0.08 tới 0.22 normalized reverse time; các edges không đủ final-positive support giữ reference cutoff. Implementation nằm trong `scripts/build_precision_onset_schedule.py`.

Job `1062490[0-3]` so sánh exact paired reference schedule và lower bounds 0.75/0.65/0.50 trên 4,096 Shoppers samples:

| Schedule | Online macro | Online heldout | Calibration error | KS | TV | Actions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Reference precision+recall | **0.058731** | **0.014898** | 0.011878 | 0.024119 | 0.024539 | **335** |
| Precision onset, LCB 0.75 | 0.058815 | 0.015031 | **0.011866** | **0.024037** | 0.024225 | 367 |
| Precision onset, LCB 0.65 | 0.059212 | 0.015411 | 0.011885 | 0.024134 | 0.024329 | 384 |
| Precision onset, LCB 0.50 | 0.059322 | 0.015341 | 0.011909 | 0.024085 | **0.024085** | 409 |

Precision-only onset cải thiện một vài fidelity/calibration values rất nhỏ, nhưng làm heldout semantic error xấu 0.9–3.4% và tăng 9.6–22.1% actions. Sau final certification, mọi biến thể vẫn kém final-only trên heldout error. Giả thuyết early selective onset do đó bị bác bỏ ở Shoppers. Recall đang đóng vai trò stability gate thực sự: nó ngăn action ở thời điểm row-level state chưa đủ đầy đủ để lựa chọn correction tốt, không chỉ trì hoãn coverage. Main method giữ reference precision+recall schedule.

### 30.3. Conditional prototype action

Action candidates cũ gồm sửa consequent, sửa endpoint đối diện, deactivate applicability và abstain. Job `1062502` thêm một manifold-aware candidate: với mỗi active component, tìm train-only rows đang active và thỏa toàn bộ component, chọn (K) neighbors gần row hiện tại trên các context columns ngoài component, rồi tạo prototype bằng median cho numeric và mode cho categorical columns. Chỉ các involved columns được thay; không gửi real rows ra API và không copy toàn bộ một training record.

| Prototype (K) | Số lần được chọn | Online macro | Online heldout | TV | Actions |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0.057888 | **0.009357** | 0.028718 | 104 |
| 8 | 4 | **0.057821** | 0.009424 | **0.028648** | **101** |
| 16 | 2 | **0.057821** | 0.009424 | **0.028648** | 103 |
| 32 | 2 | **0.057821** | 0.009424 | **0.028648** | 103 |

Prototype hiếm khi thắng direct boundary correction dưới joint density/edit scorer. Nó tạo một trade-off rất nhỏ về macro/TV nhưng làm heldout error xấu hơn. Kết quả chưa ủng hộ việc biến retrieval/prototype thành module chính; implementation được giữ để làm action-family ablation.

Tổng hợp các refinement experiments và quyết định method nằm ở `outputs/methodology_refinement_20261009/RESULTS.md`.
