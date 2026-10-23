# ICDE submission sprint

**Mục tiêu:** đóng băng main experiments và ablation quan trọng trong 7–11 ngày; deadline dự kiến 12-11-2026.

## 1. Method được ưu tiên

Method tạm thời được khóa ở mức tối thiểu:

1. verified typed constraint graph được xây trước sampling;
2. predicted-clean row tạo edge state `unreadable / inactive / satisfied / violated`;
3. graph chỉ được đọc trong middle/late-middle region;
4. violation state phải tồn tại qua ít nhất hai checkpoints trước khi phát action;
5. typed action sửa predicted clean estimate rồi đưa lại vào reverse direction.

Không thêm learned controller hoặc online LLM trước khi các batch hiện tại hoàn thành.

## 2. Evidence mới chốt ngày 07-10

### Main 3-seed, 8,192 rows, 50 steps

| Dataset | Policy | Actuate violation ↓ | Actuate heldout error ↓ | Overall heldout error ↓ | Actions |
| --- | --- | ---: | ---: | ---: | ---: |
| Shoppers | Dynamic current | 0.004828 ± 0.000676 | 0.000642 ± 0.000584 | 0.019361 ± 0.002035 | 108.3 |
| Shoppers | **Dynamic stable** | **0.004775 ± 0.000663** | **0.000609 ± 0.000549** | **0.019322 ± 0.001998** | 107.3 |
| Shoppers | Frozen state | 0.005558 ± 0.001107 | 0.001372 ± 0.001006 | 0.019642 ± 0.002057 | 93.7 |
| Default | **Dynamic current** | **0.011771 ± 0.002456** | **0.003677 ± 0.001588** | **0.016150 ± 0.002320** | 52.0 |
| Default | Dynamic stable | 0.012282 ± 0.002902 | 0.003821 ± 0.002374 | 0.016251 ± 0.002634 | 46.7 |
| Default | Frozen state | 0.012456 ± 0.003424 | 0.004011 ± 0.002603 | 0.016306 ± 0.002613 | 51.0 |

Dynamic state thắng frozen-state ở cả ba Shoppers seeds. Trên Default, effect nhỏ và có variance lớn; dynamic thắng 2/3 seeds. Stable gate tốt nhất trên Shoppers, còn current gate có mean tốt nhất trên Default. Vì vậy paper không nên claim stable gate luôn tốt; nó là confidence-control ablation, còn core contribution vẫn là dynamic state.

### Temporal evidence với stable gate

| Dataset | Region | Dynamic heldout error ↓ | Frozen heldout error ↓ | Dynamic overall ↓ | Frozen overall ↓ |
| --- | --- | ---: | ---: | ---: | ---: |
| Shoppers | Early | 0.005272 | 0.005272 | 0.023566 | 0.023566 |
| Shoppers | Middle | **0.003278** | 0.005019 | **0.022361** | 0.023452 |
| Shoppers | Late | **0.001345** | 0.002062 | **0.021442** | 0.021818 |
| Default | Early | 0.018437 | **0.017682** | 0.022174 | **0.021787** |
| Default | Middle | **0.010543** | 0.014352 | **0.018641** | 0.020659 |
| Default | Late | 0.004796 | **0.004755** | **0.016354** | 0.016397 |

Early không có lợi; middle tạo dynamic–frozen gap rõ trên cả hai dataset. Late đạt absolute validity tốt trên Shoppers nhưng dynamic và frozen gần như hòa trên Default. Vì vậy middle hiện là vùng can thiệp có bằng chứng trực tiếp nhất cho lợi ích của graph evolution trước khi sample đã hội tụ; late chủ yếu hưởng lợi từ việc bản ghi đã dễ đọc.

### Temporal evidence trên ba seed mới

Các improvement dưới đây là `frozen-state − dynamic`, vì vậy số dương có lợi cho dynamic graph.

| Dataset | Region | Actuation-error improvement ↑ | Win rate | Overall-error improvement ↑ | Win rate |
| --- | --- | ---: | ---: | ---: | ---: |
| Shoppers | **Middle** | **0.001519 ± 0.000446** | **3/3** | **0.000927** | **3/3** |
| Shoppers | Late | 0.000416 ± 0.000211 | 3/3 | 0.000204 | 3/3 |
| Default | **Middle** | **0.001963 ± 0.004673** | 2/3 | **0.000653** | 2/3 |
| Default | Late | -0.000247 ± 0.000483 | 1/3 | 0.000001 | 2/3 |

Middle giảm actuation error tương đối khoảng 48.8% trên Shoppers (`0.003116 → 0.001596`) và 19.8% trên Default (`0.009903 → 0.007940`). Bằng chứng Shoppers ổn định; Default còn variance lớn. Late không hỗ trợ dynamic graph trên Default, nên claim paper phải là **state-dependent middle intervention**, không phải “dynamic luôn tốt ở mọi timestep”.

Dynamic-middle phát trung bình nhiều hơn 38.3 actions trên Shoppers và 82 actions trên Default so với frozen-state, nên batch `1060348` đã ép dynamic dùng đúng action budget của frozen theo từng seed:

| Dataset | Actions dynamic/frozen | Dynamic actuation error ↓ | Frozen actuation error ↓ | Paired win rate | Dynamic overall ↓ | Frozen overall ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Shoppers | 33.0 / 33.0 | **0.002498** | 0.003116 | **3/3** | **0.020208** | 0.020613 |
| Default | 73.3 / 73.3 | 0.017417 | **0.009903** | 0/3 | 0.020965 | **0.019239** |

Kết luận phải tách theo dataset. Trên Shoppers, lợi ích dynamic vẫn tồn tại khi số action bằng tuyệt đối, do đó không thể giải thích chỉ bằng việc can thiệp nhiều hơn. Trên Default, kết quả đảo chiều hoàn toàn; gain trước đó đến từ action volume hoặc cách phân bổ action, chưa phải bằng chứng cho graph evolution. Không được dùng Default để hỗ trợ claim action-efficient dynamic graph.

### Tần suất refresh và row assignment

| Dataset | Topology | Actuation error ↓ | Overall error ↓ | Actions |
| --- | --- | ---: | ---: | ---: |
| Shoppers | Dynamic mỗi step | **0.001596** | **0.019686** | 71.3 |
| Shoppers | Frozen | 0.003116 | 0.020613 | 33.0 |
| Shoppers | Periodic k=3 | 0.002391 | 0.020526 | 88.3 |
| Shoppers | Periodic k=5 | 0.002183 | 0.020444 | 92.7 |
| Shoppers | Periodic k=10 | 0.002164 | 0.020397 | 87.0 |
| Default | Dynamic mỗi step | 0.007940 | 0.018586 | 155.3 |
| Default | Frozen | 0.009903 | 0.019239 | 73.3 |
| Default | Periodic k=3 | 0.007084 | 0.017776 | 171.0 |
| Default | **Periodic k=5** | **0.005610** | 0.017296 | 165.7 |
| Default | Periodic k=10 | 0.006502 | **0.017154** | 117.3 |

Per-step refresh tốt nhất trên Shoppers nhưng không tối ưu trên Default. Periodic refresh k=5/k=10 vừa cải thiện validity vừa cải thiện overall error trên Default. Do đó phương pháp không nên đồng nhất “dynamic” với “refresh mọi step”; update frequency là thành phần cần được chọn hoặc kích hoạt theo độ tin cậy của state.

Temporal hysteresis chỉ commit một thay đổi active/inactive khi state mới tồn tại qua hai hoặc ba checkpoints. Kết quả ba seed ban đầu cho thấy đây không phải cải tiến phổ quát:

| Dataset | Topology | Actuation error ↓ | Overall error ↓ | Actions |
| --- | --- | ---: | ---: | ---: |
| Shoppers | Dynamic mỗi step | **0.001596** | **0.019686** | 71.3 |
| Shoppers | Hysteresis 2 | 0.002426 | 0.020547 | 85.3 |
| Shoppers | Hysteresis 3 | 0.002571 | 0.020629 | 82.3 |
| Default | Dynamic mỗi step | 0.007940 | 0.018586 | 155.3 |
| Default | Hysteresis 2 | 0.006680 | 0.017989 | 157.3 |
| Default | **Hysteresis 3** | **0.004479** | **0.017498** | 149.0 |

Hysteresis 3 giảm actuation error khoảng 43.6% trên Default nhưng làm Shoppers kém hơn. Tuy nhiên dynamic và hysteresis trong bảng này đến từ hai Slurm jobs có GPU architecture khác nhau; chỉ so sánh hysteresis-2 với hysteresis-3 trong cùng job là paired. Do đó đây là tín hiệu tạo hypothesis, chưa phải confirmation. Batch `1060654` đang chạy lại cả ba policy tuần tự trên cùng GPU. Nếu kết quả giữ, nó bác bỏ việc chọn một persistence cố định cho mọi dataset và ủng hộ graph refresh theo độ ổn định của state.

Same-node confirmation đã hoàn tất và SHA-256 của unconditional samples khớp ở 9/9 dataset-seed groups:

| Dataset | Refresh policy | Actuation heldout error ↓ | Overall heldout error ↓ | Actions | Normalized flip rate |
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

Adaptive EMA-threshold policy không vượt policy cố định tốt nhất trên bất kỳ dataset nào: nó bằng dynamic trên Shoppers, bằng hysteresis-3 trên Diabetes và nằm giữa hai policy trên Default. Hypothesis “topology flip rate cao đồng nghĩa state noise” cũng bị bác bỏ: Diabetes có flip rate cao nhất nhưng dynamic vẫn tốt hơn nhẹ. Một controller tương lai phải phân biệt **genuine state transition** với **preview noise**, không chỉ đếm flip. Không tiếp tục tune threshold trên ba dataset này.

### Score-only graph trong action selection

Diabetes có 6 actuation edges và 15 score-only edges; mỗi actuation edge chia sẻ cột với 12–13 score-only edges. Một guardrail được thử: trong các action đã thỏa edge chính, phạt candidate nào tạo violation mới trên score-only graph, nhưng không ép sửa violation có sẵn.

| Graph guard weight | Actuation error ↓ | Score-only error ↓ | Overall error ↓ | KS ↓ | TV ↓ | Actions |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.001541 | 0.014146 | 0.010198 | 0.019838 | 0.019638 | 74.0 |
| 2 | 0.001541 | 0.014146 | 0.010198 | 0.019838 | 0.019638 | 74.0 |

Unconditional hash khớp 3/3 seed, nhưng toàn bộ metric và action-choice counts giống hệt. Hypothesis bị bác bỏ: candidate set hiện chỉ gồm sửa requirement trực tiếp, đổi endpoint, deactivate antecedent hoặc abstain; các lựa chọn này không tạo xung đột mà score-only graph phân biệt được. Không sweep thêm weight. Nếu phát triển action layer, cần sinh candidate phong phú hơn hoặc dùng score-only edges để đề xuất nhánh, không chỉ rerank các action hiện có.

Shuffled-row run ban đầu chưa thể kết luận vì action count mất cân bằng nghiêm trọng: Shoppers phát 1,643 actions và overall error tăng lên 0.047057. Fixed-budget rerun trên Shoppers đã hoàn tất:

| Topology | Actions | Actuation error ↓ | Overall error ↓ | Numeric KS ↓ | Paired wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| Correct row-dynamic | 40 | **0.003307** | **0.020994** | **0.026137** | **3/3** |
| Shuffled row-dynamic | 40 | 0.004326 | 0.021497 | 0.026265 | 0/3 |

Với cùng hard budget, correct row assignment giảm actuation error khoảng 23.5% và thắng cả ba seed. Đây là bằng chứng trực tiếp rằng row-specific state mang thông tin hữu ích, không chỉ tăng số lần can thiệp. Trên Default, dynamic phát đủ 40 actions nhưng shuffled chỉ tìm được trung bình 33.3 actions; dynamic không thắng nhất quán (`0.017717` so với `0.017570`). Vì vậy kết luận row-specific hiện chỉ được hỗ trợ chắc trên Shoppers, còn Default là kết quả âm.

### Diabetes graph discovery

Pipeline dùng 19 API calls và 199,622 tokens, tạo unified graph 26 edges thuộc bốn relation families và 19 endpoint patterns. Catalog cuối giữ 22 edges: 6 `actuate`, 15 `score_only`, 1 `monitor`. Các actuation edges chủ yếu là multi-column medication-state relations giữa `diabetesMed`, `change`, `insulin` và các thuốc cụ thể; đây là graph phức tạp hơn các zero-count rules trên Shoppers.

Held-out shift trung bình của unified graph là 0.0010 và toàn bộ retained edges có shift không quá 0.05. Tuy vậy, stability chỉ chứng minh quan hệ giữ được qua split, không chứng minh mọi edge là luật y khoa phổ quát.

Sampling đã tạo được signal trên dataset thứ ba:

| Window | Topology | Actuation heldout error ↓ | Overall heldout error ↓ | Numeric KS ↓ | TV ↓ | Actions |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Full | Dynamic current | **0.005911 ± 0.001007** | **0.012383** | 0.020365 | 0.019937 | 24.0 |
| Full | Dynamic stable | 0.006222 ± 0.000960 | 0.012528 | 0.020374 | 0.019905 | 18.3 |
| Full | Frozen state | 0.006964 ± 0.001282 | 0.012844 | 0.020365 | **0.019879** | 11.7 |
| Middle | **Dynamic** | **0.001077 ± 0.000659** | **0.011112** | **0.020038** | 0.020092 | 70.7 |
| Middle | Frozen state | 0.003543 ± 0.000302 | 0.012311 | 0.020056 | **0.020014** | 28.3 |

Dynamic-middle thắng frozen-state ở 3/3 seed và giảm actuation error tương đối khoảng 69.6%; overall error giảm khoảng 9.7%, còn KS/TV gần như không đổi. Tuy nhiên action count chưa cân bằng. Final repair đạt actuation error `0.000000` và overall error `0.010376`, tốt hơn dynamic-middle trên chính các rule có actuator. Do đó Diabetes hiện hỗ trợ **dynamic topology tốt hơn frozen topology trong trajectory**, nhưng phản bác claim rằng mid-step intervention đã hơn post-repair. Batch `1060479` và `1060480` đang kiểm tra action-matched topology và timing trước khi chốt claim.

Các controls tiếp theo đã hoàn tất:

| Control | Variant | Actuation heldout error ↓ | Overall heldout error ↓ | Actions |
| --- | --- | ---: | ---: | ---: |
| Topology, cap 20 | Dynamic | 0.005213 | 0.015524 | 20.0 |
| Topology, cap 20 | **Frozen state** | **0.004740** | **0.015294** | 19.3 |
| Topology, cap 20 | Shuffled | 0.010728 | 0.018319 | 0.7 |
| Timing, cap 20 | Early | 0.006152 | 0.015892 | 20.0 |
| Timing, cap 20 | Middle | 0.005213 | 0.015524 | 20.0 |
| Timing, cap 20 | **Late** | **0.003478** | **0.014544** | 20.0 |
| Smoothing, uncapped | Hysteresis 2 | 0.000742 | 0.012851 | 69.3 |
| Smoothing, uncapped | **Hysteresis 3** | **0.000622** | **0.012803** | 60.0 |

Diabetes bác bỏ action-efficiency claim: khi gần cân bằng số action, frozen-state tốt hơn dynamic ở 3/3 seed. Shuffled control không đạt matched budget vì graph bị tráo gần như không sinh actionable violation; dòng này chỉ là diagnostic và không được dùng làm causal evidence. Middle tốt hơn early ở 3/3 seed, nhưng late tốt hơn middle ở 3/3. Vì vậy paper chỉ có thể nói middle là vùng **sớm nhất đã đủ đọc để tạo lợi ích**, không phải vùng có validity tuyệt đối tốt nhất. Hysteresis 3 cho thấy lọc topology noise có ích; cần so với dynamic trên cùng seed trước khi xem đây là method improvement.

Paired Poisson row-bootstrap với 2,000 lần lặp, sau đó bootstrap thêm trên ba seed, cho kết quả:

| Dataset | Frozen − dynamic actuation error ↑ | Seed wins | 95% hierarchical interval | P(improvement > 0) |
| --- | ---: | ---: | ---: | ---: |
| Shoppers | **0.001519** | 3/3 | **[0.000994, 0.002057]** | 1.000 |
| Default | 0.001963 | 2/3 | [-0.003091, 0.007210] | 0.769 |
| Diabetes | **0.002466** | 3/3 | **[0.001270, 0.003956]** | 1.000 |

Khoảng tin cậy này xác nhận effect theo synthetic rows và seeds trên Shoppers/Diabetes, nhưng không xác nhận Default. Đây là uncertainty audit trên ba sampling seeds, chưa thay thế confirmation run với nhiều training/checkpoint seeds.

Chưa chạy `LLM-only / empirical-only / unified` như ba sampler methods vì implementation hiện tại chỉ cho `actuate` edges sửa row; empirical edges đều được giới hạn ở `score_only`. Do đó unified và LLM-only đang có cùng action graph, còn empirical-only không có actuator. Source ablation chỉ có ý nghĩa sau khi score-only edges thực sự tham gia branch reward, hoặc phải được trình bày như graph-discovery audit thay vì generation baseline.

### Trạng thái các baseline từ code chính thức

| Baseline | Nguồn số | Dataset/task | Kết quả chính | Có thể so trực tiếp? |
| --- | --- | --- | --- | --- |
| TabDiff | Official backbone, checkpoint và sampler; local run | Shoppers, Default, Diabetes | Là unconditional baseline dùng chung noise/seed với mọi intervention | Có |
| LCATS + TabSyn | Official repository; local rerun và independent audit | Insurance | Any-violation `0.0822 → 0.0017`; KS `0.0359 → 0.0318`; TSTR R² `0.7212 → 0.7224`; cell edit `0.2884` | Chưa, khác dataset/rule; nhưng là bằng chứng mạnh rằng post-repair gần loại hết violation |
| CuTS | Official repository; local rerun | German | Năm implication/line programs đạt satisfaction `1.0`; TV tăng từ khoảng `0.0009–0.0010` lên `0.0011–0.0027`; utility thay đổi theo program | Chưa, khác task; cần common-task port nếu dùng trong bảng xếp hạng |
| Harpoon | Artifact CSV của tác giả | Shoppers/Default general constraints | Violation của Harpoon: Shoppers `0.0–20.55%`, Default `0.91–7.78%` tùy constraint family | Không trộn số trực tiếp; protocol và constraint khác. Local official rerun chưa hoàn tất |
| Harpoon-style | Local proxy trên TabDiff | Common contract probes | Tạo Pareto validity–distortion nhưng chưa thay thế official method | Chỉ là cơ chế đối chứng, phải ghi `proxy` |
| CTDF-style | Local numeric proxy | Common contract probes | Per-step projection đạt feasibility cao nhưng gây fidelity/distortion trade-off | Chỉ là `proxy`; chưa có mixed-type KL feasibility đầy đủ |

Các baseline chính thức hiện chứng minh hai giới hạn quan trọng của dự án: (i) final satisfaction bằng 1 không đủ tạo novelty vì CuTS/LCATS đã làm được; (ii) per-step constraint guidance không mới vì Harpoon/CTDF đã bao phủ. Claim hợp lệ phải nằm ở **topology thay đổi theo predicted-clean row và trajectory**, cùng matched action budget, fidelity và cost. Chưa có common-task faithful Harpoon/CTDF/LCATS comparison, nên chưa được gọi kết quả hiện tại là state of the art.

### Độ bền theo số reverse steps

| Dataset | Steps | Dynamic actuation error ↓ | Frozen actuation error ↓ | Dynamic overall ↓ | Frozen overall ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shoppers | 30 | **0.000138** | 0.000653 | **0.024246** | 0.024549 |
| Shoppers | 100 | **0.000982** | 0.001914 | **0.018923** | 0.019299 |
| Default | 30 | **0.003843** | 0.004068 | 0.019879 | **0.019839** |
| Default | 100 | **0.003972** | 0.004659 | **0.014589** | 0.014716 |

Dynamic graph giảm actuation error ở cả bốn cấu hình. Overall error tốt hơn ở ba trên bốn cấu hình; sai khác âm duy nhất là `4.0e-5` tại Default/30 steps. Như vậy hiệu ứng không phụ thuộc riêng lịch 50 steps, nhưng cần các seed bổ sung nếu muốn claim định lượng mạnh cho robustness.

### Fidelity và utility trên ba sampling seeds

| Dataset | Method | Shape error ↓ | Trend error ↓ | α-Precision ↑ | β-Recall ↑ | Detection ↑ | MLE AUC ↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Shoppers | TabDiff | 1.3933 | 1.8358 | 98.9079 | **49.9300** | 0.9906 | 0.9043 |
| Shoppers | Dynamic-stable | **1.3778** | 1.7967 | **98.9158** | 49.9051 | 0.9912 | **0.9108** |
| Shoppers | Final repair | 1.3792 | **1.7911** | 98.8208 | 49.9737 | **0.9924** | 0.9062 |
| Default | TabDiff | 1.7829 | 1.9035 | 97.7084 | **52.6055** | 0.9488 | 0.7395 |
| Default | Dynamic-current | **1.7792** | 1.9033 | **97.7111** | 52.5993 | 0.9478 | 0.7445 |
| Default | Final repair | 1.7800 | 1.9035 | 97.7084 | 52.6055 | **0.9543** | **0.7451** |

Các số này được tính lại bằng cùng split, sample và evaluator cục bộ; dùng để so sánh nội bộ giữa method, không trộn trực tiếp với số paper nếu protocol khác. Kết quả hiện tại cho thấy semantic gain không đánh đổi đáng kể fidelity/utility, nhưng chưa cho phép claim SOTA tổng quát. Trên Shoppers, final repair vẫn mạnh hơn về actuation validity; trên Default, dynamic guidance tốt hơn final repair về actuation error (`0.003677` so với `0.006801`).

## 3. Job DAG hiện tại

| Job | Nội dung | Vai trò trong paper |
| --- | --- | --- |
| `1060265[0-5]` | 3 seeds × Shoppers/Default; current, stable, frozen-state | Main mechanism table; completed |
| `1060266[0-5]` | Stable early/middle/late × dynamic/frozen | Temporal ablation |
| `1060308[0-5]` | Shape, Trend, α-Precision, β-Recall, Detection, MLE, DCR | Main fidelity/utility table; completed |
| `1060367 → 1060368 → 1060369` | Compact-resume Diabetes graph → catalog → trajectory | Completed; 22 retained catalog edges |
| `1060370[0-2]` | Diabetes sampling attempt | Failed at unseen-category decode; no scientific result |
| `1060451[0-2]` | Fixed Diabetes full-window: current/stable/frozen | Dataset thứ ba; completed |
| `1060452[0-2]` | Diabetes middle-window: dynamic/frozen | Dynamic thắng frozen 3/3; completed |
| `1060453[0-5]` | Dynamic vs shuffled row, fixed 40-action budget | Shoppers dương, Default âm; completed |
| `1060463` | Aggregate three batches above | Completed |
| `1060464[0-5] → 1060468` | Topology hysteresis 2/3 checkpoints | Default dương, Shoppers âm; completed |
| `1060478[0-2]` | Diabetes hysteresis 2/3 | Hysteresis 3 tốt hơn; completed |
| `1060479[0-2]` | Diabetes dynamic/frozen/shuffled với cap 20 actions | Frozen thắng dynamic; shuffled không đạt budget; completed |
| `1060480[0-2]` | Diabetes early/middle/late với cap 20 actions | Late > middle > early; completed |
| `1060481` | Aggregate ba Diabetes controls | Completed |
| `1060488` | Paired row/seed bootstrap cho dynamic vs frozen trên ba dataset | Shoppers/Diabetes CI dương, Default không chắc; completed |
| `1060644[0-8] → 1060645` | Volatility-adaptive refresh trên ba dataset | Cross-job pilot; không dùng làm paired evidence |
| `1060654[0-8] → 1060655` | Same-node paired dynamic/hysteresis-3/adaptive | 9/9 parity; adaptive hypothesis rejected; completed |
| `1060664[0-2] → 1060665` | Score-only graph guardrail cho action selection | 3/3 parity; no action/metric change; rejected |
| `1060282[0-3]` | 30/100-step robustness | Sampler robustness; completed |
| `1060283[0-5]` | Persistence depth 1/2/3 | Confidence-gate ablation; completed |
| `1060298` | Recompute CTGAN/TVAE/TabDiff table với α/β | Baseline completion; completed |
| `1060309` | Aggregate mean/std và paper-ready CSV | Reporting; completed |
| `1060317[0-11]` | 3 seeds × middle/late × dynamic/frozen; paired aggregation | Chốt temporal claim; completed |
| `1060348[0-5] → 1060354` | Dynamic-middle với action budget bằng frozen theo từng seed | Loại confound số action; completed |
| `1060355[0-5]` | Tráo active rows nhưng giữ edge support theo step | Completed; action imbalance makes it diagnostic only |
| `1060356[0-5]` | Refresh graph mỗi 3/5/10 steps | Completed; periodic refresh helps Default |

Lỗi Diabetes ban đầu ở `1060268` là structured JSON bị cắt tại giới hạn 5,000 output tokens. Pipeline resume dùng lại API calls thành công, giảm critic batch từ 16 xuống 6, chỉ gửi schema của các cột liên quan cho từng review block, cân bằng candidate theo proposal role trước final judge và đặt trần toàn run ở 225,000 tokens.

Lưu ý reproducibility mới: cùng seed nhưng chạy ở A10 và T4 có thể tạo unconditional sample khác nhau đủ để làm lệch KS/violation. Vì vậy không dùng cross-job/cross-architecture numbers như paired ablation. Batch `1060654` chạy ba topology tuần tự trong cùng task/GPU và aggregate chỉ hợp lệ nếu SHA-256 của ba unconditional sample trùng nhau cho từng dataset-seed.

## 4. Kế hoạch 7–11 ngày

| Ngày | Deliverable |
| --- | --- |
| 07–09/10 | Chốt dynamic-state formulation, main seeds, temporal và persistence ablations |
| 08–11/10 | Hoàn tất Diabetes hoặc quyết định thay dataset nếu graph không có đủ actuation edges |
| 09–13/10 | Matched final repair, frozen-state, TabDiff; hoàn thiện Harpoon-style/CTDF-style comparison |
| 11–15/10 | Quality, utility, coverage, runtime, cost và paired confidence intervals |
| 15–18/10 | Freeze method/config; chỉ rerun lỗi hoặc reviewer-critical ablation |

Sau 18/10, ưu tiên chuyển compute sang confirmation seeds và writing, không tiếp tục mở rộng method.

## 5. Tiêu chí đóng băng main method

Giữ dynamic constraint-state graph nếu:

- dynamic hơn frozen-state theo paired mean trên ít nhất hai datasets;
- early không tạo gain, middle tạo gain có ý nghĩa;
- Shape/Trend/α/β/MLE không giảm đáng kể;
- ít nhất một dataset cho kết quả cạnh tranh hoặc hơn final repair;
- gain không biến mất ở 30/100 reverse steps.

Nếu Diabetes không tạo natural violation signal hoặc không có ít nhất ba actuation edges, không ép dùng nó; thay bằng một benchmark process/medical khác và ghi Diabetes là negative dataset audit.
