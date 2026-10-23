# Cập nhật thực nghiệm: dynamic graph, trajectory actionability và persistence

Ngày cập nhật: 2026-10-09

## 1. Kết luận ngắn

Kết quả mới ủng hộ một phiên bản hẹp và rõ hơn của bài toán:

> Không phải mọi constraint hợp lệ đều nên được inject vào reverse diffusion. Giá trị của dynamic graph nằm ở việc chọn đúng cạnh cho đúng row và đúng trạng thái trajectory, chỉ can thiệp khi violation đã đủ ổn định, rồi dùng full graph để certification ở cuối.

Hai kết quả mạnh nhất hiện tại:

1. Trên Shoppers, một portfolio gồm hai hướng của quan hệ `Administrative–Administrative_Duration`, được chọn chỉ bằng pilot/calibration, giúp `middle + final` tốt hơn final repair về heldout semantic error khoảng 0,54% và tăng XGBoost AUC khoảng 0,68 điểm phần trăm. Inject toàn bộ graph lại làm xấu semantic error khoảng 1,45% và cần gần gấp đôi số actions.
2. Trên Default, full dynamic graph trong middle window tốt hơn final repair về heldout semantic error khoảng 7,5% trên ba seed 4.096 rows. Persistence gate `stable8` giữ gain khoảng 6,1%, giảm số middle actions từ khoảng 53 xuống 30, và cải thiện calibration error khoảng 2,2% trong batch xác nhận 4.096 rows.

Kết quả chưa đủ để claim SOTA. Baseline mạnh nhất hiện đã matched hoàn toàn là final typed repair; Harpoon/CTDF/LCATS chưa được tái chạy faithfully trên cùng graph, split và evaluator. Kết quả 8.192 rows cũng cho effect nhỏ hơn, khoảng 2,1% heldout gain cho `stable8 + final`, nên cần báo mean, variance và negative seed thay vì chỉ báo best run.

## 2. Method hiện tại

Pipeline đang có bốn lớp:

1. **Offline graph discovery.** LLM đề xuất quan hệ từ schema, codebook, aggregate statistics và seed rules. Mỗi relation được compile thành typed predicates, kiểm tra trên discovery/counterexample/calibration splits, phân hardness và gán role `actuate`, `score_only` hoặc `monitor`.
2. **Trajectory state estimation.** Ở mỗi reverse step, decoder tạo clean preview `x̂₀`. Với từng row, hệ thống tính applicability, satisfaction, violation streak, topology flips và edge-specific readability.
3. **Dynamic active graph và action layer.** Chỉ các edge phù hợp với row, timestep, readability và persistence gate mới vào active graph. Các edge có shared columns được xử lý theo connected component. Planner so sánh sửa consequent, deactivate antecedent và abstain bằng energy gồm constraint violation, graph guardrail, train-density và edit cost.
4. **Injection và certification.** Numeric/categorical corrections được inject vào `x̂₀` trước reverse update. Một `final graph` độc lập có thể repair residual violations sau sampling. Việc tách middle graph khỏi final graph cho phép dùng portfolio nhỏ trong trajectory nhưng không bỏ coverage ở certification.

LLM hiện tham gia ở graph discovery, semantic review và typed compilation; không gọi API trong từng sampling step. Đây là quyết định có chủ ý: các audit trước cho thấy direct LLM judgement đắt, không ổn định và kém compile-once + deterministic executor. Nếu quay lại LLM-in-the-loop, vai trò hợp lý hơn là cập nhật graph/edge policy có kiểm chứng, không phải trực tiếp sửa cell ở mọi step.

## 3. Thay đổi implementation trong đợt này

- Thêm `--final-graph`: middle guidance và final certification có thể dùng hai graph khác nhau.
- Thêm builder cho actuator portfolio có provenance rõ ràng.
- Thêm persistence gates `stable8`, `stable10`, `stable12`.
- Sửa global action-budget bug của component planner: budget trước đây được cắt riêng từng edge trước joint planning, nên có thể vượt cap. Sau sửa, budget được enforce sau khi biết số component actions thực tế.
- Giữ evaluation graph hoàn toàn độc lập với guidance graph trong mọi bảng dưới đây.

Các file chính:

- `scripts/rowwise_reverse_action_guidance.py`
- `scripts/build_selected_actuator_portfolio.py`
- `slurm/375_trajectory_benefit_portfolio.sbatch` đến `slurm/390_default_stable8_scale_common.sbatch`

## 4. Shoppers: chọn actuator thay vì inject toàn graph

### 4.1 Edge-level causal screening

Pilot 2.048 rows chạy từng actuator riêng, luôn so với cùng final repair. Chỉ hai edge có improvement đồng thời trên macro violation và calibration-side objective:

| Edge | Ý nghĩa | Middle actions | Online − final, macro | Online − final, heldout | Online − final, calibration |
| --- | --- | ---: | ---: | ---: | ---: |
| `seed_000` | `Administrative=0 → Administrative_Duration=0` | 10 | +0,000137 | +0,000146 | +0,000137 |
| `r1_020` | `Administrative_Duration>0 → Administrative>0` | 11 | +0,000163 | +0,000172 | +0,000163 |

Dấu dương nghĩa là middle intervention tốt hơn final-only. Các edge còn lại bằng 0 hoặc âm; năm near-hard edges đều không tạo improvement nhất quán. Hai edge được chọn tạo một relation hai chiều, nhưng edge active thay đổi theo trạng thái row: một edge xử lý count bằng 0, edge kia xử lý duration dương.

### 4.2 Validation ba seed chưa dùng để chọn

4.096 rows/seed, seeds `20261350–20261352`, pretrained TabDiff, cùng noise và full final graph.

| Method | Macro violation ↓ | Heldout semantic error ↓ | Calibration error ↓ | Actions |
| --- | ---: | ---: | ---: | ---: |
| Final repair | 0,054664 | 0,014041 | **0,013362** | 123,3 |
| Full graph middle | 0,055266 | 0,014283 | 0,013645 | 232,0 |
| Full graph middle + final | 0,054963 | 0,014245 | 0,013520 | 243,3 |
| Two-edge portfolio middle | 0,059481 | 0,017196 | 0,016752 | 11,3 |
| **Two-edge portfolio middle + full final** | **0,054549** | **0,013965** | 0,013387 | **124,7** |

Portfolio-certified giảm heldout error khoảng 0,54% và macro violation khoảng 0,21% so với final repair, với trung bình chỉ 1,3 actions bổ sung. Full graph certified làm heldout error xấu khoảng 1,45% và dùng thêm khoảng 120 actions.

Hai single-edge controls cho thấy portfolio không chỉ cộng cơ học:

- `seed_000 + final`: heldout mean 0,014026.
- `r1_020 + final`: heldout mean 0,014005.
- Hai-edge dynamic portfolio + final: heldout mean 0,013965.

Ở mỗi seed, active state có thể chọn hướng phù hợp hơn; portfolio đạt mean tốt hơn cả hai fixed single-edge variants.

### 4.3 Common fidelity và utility metrics

| Method | Shape error ↓ | Trend error ↓ | XGBoost AUC ↑ | Heldout semantic error ↓ |
| --- | ---: | ---: | ---: | ---: |
| Final repair | **1,53249** | **1,91819** | 0,89915 | 0,013993 |
| Full graph + final | 1,53252 | 1,91988 | 0,90419 | 0,014198 |
| **Portfolio + final** | 1,53322 | 1,92215 | **0,90599** | **0,013918** |

Portfolio cải thiện semantic error và AUC, đổi lại shape/trend fidelity xấu hơn rất nhẹ. Đây là Pareto improvement theo semantic/utility, không phải dominance trên mọi metric.

## 5. Default: middle trajectory tạo gain lớn hơn final repair

### 5.1 Full graph và actuator portfolio

Pilot từng edge cho thấy bốn trong năm actuators giảm macro violation; `r1_001` không kích hoạt. Portfolio v1 đã dùng calibration error như hard filter và loại `r1_017`; validation ba seed cho thấy quyết định này sai: portfolio v1 làm heldout error xấu hơn final repair khoảng 2,2%.

Portfolio v2 giữ mọi actuator có positive semantic effect và chỉ deduplicate constraint tương đương. Trên seeds `20261357–20261359`:

| Method | Macro violation ↓ | Heldout semantic error ↓ | Calibration error ↓ | Actions |
| --- | ---: | ---: | ---: | ---: |
| Final repair | 0,064462 | 0,016734 | **0,018918** | 9,7 |
| Portfolio v2 + final | 0,061848 | 0,015612 | 0,019001 | 52,0 |
| **Full dynamic graph** | **0,061508** | **0,015483** | 0,019028 | 54,0 |

Full dynamic graph giảm heldout error khoảng 7,5% và macro violation khoảng 4,6% so với final repair. Portfolio v2 vẫn giảm heldout khoảng 6,7% nhưng không hơn full graph. Vì vậy actionability screening là hữu ích trên Shoppers, nhưng chưa phải một universal selector.

Common metrics trên cùng ba seeds:

| Method | Shape error ↓ | Trend error ↓ | XGBoost AUC ↑ | Heldout semantic error ↓ |
| --- | ---: | ---: | ---: | ---: |
| Unconditional | 1,96476 | **2,04207** | **0,73344** | 0,019538 |
| Final repair | 1,95899 | **2,04207** | 0,73252 | 0,016734 |
| **Full dynamic graph** | **1,93875** | 2,04557 | 0,73153 | **0,015483** |

Middle guidance đồng thời cải thiện shape error và semantic validity, nhưng trend gần như ngang và AUC giảm khoảng 0,10 điểm phần trăm so với final repair.

### 5.2 Early, middle và late

Native controller, ba seed 4.096 rows:

| Timing | Macro violation ↓ | Heldout semantic error ↓ | Calibration error ↓ | Actions |
| --- | ---: | ---: | ---: | ---: |
| Early forced + final | 0,064388 | 0,016613 | 0,020435 | 12,3 |
| Late | 0,063817 | 0,016432 | 0,019074 | 13,0 |
| **Middle** | **0,061508** | **0,015483** | **0,019028** | 54,0 |
| Final repair | 0,064462 | 0,016734 | **0,018918** | 9,7 |

Middle tốt hơn late khoảng 5,8% heldout error và tốt hơn final repair khoảng 7,5%. Early previews có rất nhiều topology flips và cho kết quả kém.

Tuy nhiên, sau khi sửa budget bug và match đúng số actions với late, middle đạt heldout 0,017914 còn late đạt 0,016432. Do đó:

- Có bằng chứng cho native middle policy vì nó có nhiều cơ hội can thiệp hữu ích hơn.
- Chưa có bằng chứng rằng mỗi middle action tốt hơn một late action.
- Claim phù hợp là **trajectory coverage + state persistence**, không phải “middle timestep tự thân tối ưu”.

## 6. Persistence-aware active graph

Persistence gate chỉ activate một row-edge sau khi violation tồn tại liên tục qua K clean previews. Đây là state-dependent graph transition, không phải static rule threshold.

### 6.1 Validation 4.096 rows, ba seed chưa dùng để chọn K

| Method | Macro violation ↓ | Heldout semantic error ↓ | Calibration error ↓ | Actions |
| --- | ---: | ---: | ---: | ---: |
| Final repair | 0,060227 | 0,015264 | 0,014965 | 8,7 |
| Stable5 | **0,057794** | 0,014444 | 0,015305 | 53,3 |
| **Stable8 + final** | 0,058770 | **0,014340** | **0,014639** | 30,3 |
| Stable12 + final | 0,059779 | 0,014839 | 0,014847 | **13,7** |

Stable8 là quality point: heldout giảm khoảng 6,1%, macro giảm 2,4%, calibration giảm 2,2% so với final repair. Stable12 là efficiency point: chỉ thêm khoảng 5 actions, vẫn giảm heldout khoảng 2,8% và calibration khoảng 0,8%.

Một seed của stable8 xấu hơn final repair khoảng 1,2% heldout; vì vậy effect không hoàn toàn uniform.

### 6.2 Xác nhận 8.192 rows trên ba seed khác

| Method | Macro violation ↓ | Heldout semantic error ↓ | Calibration error ↓ | KS ↓ | TV ↓ | Actions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unconditional | 0,064030 | 0,017828 | 0,017125 | 0,027697 | **0,012172** | 0 |
| Final repair | 0,061066 | 0,014864 | **0,015374** | 0,027701 | **0,012172** | 20,3 |
| **Stable8 + final** | **0,060313** | **0,014557** | 0,015548 | **0,027685** | 0,012244 | 55,0 |

Ở cỡ mẫu lớn hơn, gain còn khoảng 2,1% heldout và 1,2% macro; calibration và TV xấu nhẹ. Hai seed dương, một seed âm khoảng 0,8%. Đây là evidence có thật nhưng chưa phải large-margin dominance.

### 6.3 Common metrics cho persistence frontier

| Method | Shape error ↓ | Trend error ↓ | XGBoost AUC ↑ | Heldout semantic error ↓ |
| --- | ---: | ---: | ---: | ---: |
| Final repair | 1,81404 | **2,07398** | **0,73965** | 0,015264 |
| Stable5 | 1,81072 | 2,07467 | 0,73648 | 0,014444 |
| **Stable8 + final** | **1,80989** | 2,07469 | 0,73476 | **0,014340** |
| Stable12 + final | 1,81315 | 2,07343 | 0,73601 | 0,014839 |

Stable8 cải thiện shape khoảng 0,23% và semantic error khoảng 6,1%, nhưng giảm AUC khoảng 0,49 điểm phần trăm. Stable12 giữ fidelity gần final hơn nhưng semantic gain nhỏ hơn.

### 6.4 Guidance scale sau khi cố định stable8

Ba seed mới `20261368–20261370`, 4.096 rows/seed:

| Method | Shape error ↓ | Trend error ↓ | XGBoost AUC ↑ | Macro violation ↓ | Heldout semantic error ↓ | Calibration error ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Final repair | 2,05164 | 2,28275 | 0,73561 | 0,058745 | 0,015166 | **0,012811** |
| Stable8, scale 2 | 2,04712 | 2,28084 | **0,73659** | 0,057362 | 0,014889 | 0,013418 |
| Stable8, scale 3 | 2,04611 | **2,27708** | 0,73585 | 0,057240 | 0,014777 | 0,013329 |
| Stable8, scale 4 | **2,04560** | 2,27751 | 0,73175 | **0,057177** | **0,014528** | 0,013060 |

Scale 2 là balanced point: heldout error giảm khoảng 1,8%, shape/trend tốt hơn nhẹ và AUC tăng khoảng 0,10 điểm phần trăm. Scale 4 là semantic-quality point: heldout giảm khoảng 4,1%, macro giảm khoảng 2,6% và shape/trend tốt hơn, nhưng AUC giảm khoảng 0,39 điểm phần trăm. Không có một scale thống trị trên mọi metric.

## 7. Hypotheses: giữ, sửa và bác bỏ

| Hypothesis | Kết luận hiện tại |
| --- | --- |
| Inject mọi valid constraint vào middle steps sẽ tốt hơn | Bác bỏ trên Shoppers; full graph tốn actions và xấu hơn final repair. |
| Một actuator portfolio nhỏ có thể thắng full graph | Được ủng hộ trên Shoppers, chưa tổng quát trên Default. |
| Middle intervention luôn tốt hơn late intervention | Bác bỏ nếu match action budget; native middle tốt hơn nhờ coverage, không nhờ per-action efficiency. |
| Persistence có thể lọc noisy state mà không mất hết coverage | Được ủng hộ; stable8 là quality point, stable12 là efficiency point trên Default. |
| Calibration error nên là hard filter khi chọn actuator | Bác bỏ; nó loại một edge quan trọng trên Default. Nên xem calibration như một objective trong Pareto selection. |
| Một guidance scale có thể thống trị mọi metric | Bác bỏ; scale 2 là balanced point, scale 4 là semantic-quality point. |
| Dynamic graph đã thắng SOTA | Chưa đủ bằng chứng; thiếu faithful same-task Harpoon/CTDF/LCATS và effect còn phụ thuộc dataset/seed. |

## 8. Contribution có thể viết

Claim an toàn và có khoảng cách với related work:

1. **Trajectory-conditioned graph activation.** Graph topology được recompute theo row state, applicability, readability và violation persistence; không phải chỉ dùng static constraint request với gradient thay đổi theo sample.
2. **Actionability-aware constraint routing.** Constraint discovery và constraint actuation được tách riêng. Valid semantic edges có thể chỉ làm monitor/score guardrail; chỉ edge có trajectory benefit mới được actuator authority.
3. **Connected-component action and residual certification.** Joint action xử lý các edge liên thông và cho phép portfolio nhỏ trong reverse process, trong khi full graph vẫn bảo đảm residual coverage ở cuối.
4. **Evaluation protocol có negative results.** So sánh unconditional, full graph, selected portfolio, early/middle/late, action-matched timing, final repair và independent evaluation graph; báo cả semantic validity, calibration, fidelity, utility và action cost.

Không nên claim:

- per-step constraint enforcement là mới;
- LLM-generated constraints hoặc executable rule repair là mới;
- middle timestep luôn tối ưu;
- kết quả hiện tại đã là SOTA.

## 9. Điểm yếu còn lại

1. Default graph có vài rationale từ LLM chưa dùng đúng domain codebook. Ví dụ diễn giải `PAY_* = -2` là “severe delinquency” không đáng tin; edge có empirical support nhưng semantic authority cần review lại.
2. Actuator selection hiện dùng small pilot và có thể overfit. Shoppers generalize tốt qua seed; Default selector v1 thất bại.
3. Utility trade-off chưa được loại bỏ. `stable8/scale4` cho semantic gain lớn nhưng có thể giảm AUC; `stable8/scale2` giữ AUC tốt hơn nhưng semantic gain nhỏ hơn.
4. Action cost cao hơn final repair. Stable8 cần khoảng 30–55 actions so với 9–20 final actions; stable12 giảm khoảng cách nhưng gain nhỏ.
5. Chưa có faithful matched baselines từ Harpoon, CTDF, LCATS/Zhao. Vì vậy bảng hiện chứng minh mechanism, chưa chứng minh SOTA.
6. Kết quả mạnh nhất nằm ở Default; Shoppers chỉ có gain nhỏ. Cần thêm ít nhất một dataset có temporal/process constraints tốt, codebook rõ và không dựa vào target leakage.

## 10. Hướng tiếp theo ưu tiên

1. Freeze ba operating points trên Default: `stable8/scale4` cho semantic quality, `stable8/scale2` cho utility balance và `stable12` cho action efficiency. Không tiếp tục controller sweep rộng.
2. Audit/rebuild Default graph với codebook chính xác; yêu cầu LLM cite field semantics và reject relation có rationale sai dù empirical rate tốt.
3. Port faithful CTDF-style per-step projection và Harpoon-style differentiable guidance lên cùng typed graph, backbone, seeds và action/compute budget.
4. Chạy thêm một dataset process/temporal có codebook rõ; ưu tiên dataset mà constraint có nhiều state changes trong trajectory.
5. Báo confidence interval trên ít nhất 5 seeds cho main table; không chọn hyperparameter bằng heldout graph.
6. Đóng phần paper problem/method ngay: graph discovery là offline support module; contribution chính là dynamic activation, persistence và actionability routing trong reverse diffusion.

## 11. Jobs và artifacts

| Nội dung | Job/artifact |
| --- | --- |
| Shoppers actuator ablation | `1062555`, `outputs/actuator_ablation_graphs_shoppers_20261009/` |
| Shoppers portfolio validation | `1062570` |
| Shoppers single-edge/timing/common metrics | `1062576`, `1062577`, `1062578` |
| Default edge pilot | `1062591` |
| Default portfolio v1/v2 | `1062596`, `1062607` |
| Default common metrics và timing | `1062613`, `1062614` |
| Action-matched timing sau bug fix | `1062626` |
| Persistence discovery/validation | `1062629`, `1062636` |
| Persistence common metrics | `1062645` |
| Stable8 8.192-row validation | `1062646` |
| Scale frontier và ba-seed validation | `1062651`, `1062655`, `1062656`, `1062665` |

Tất cả jobs trong tài liệu này chạy trên A40 exploratory route. Main-table confirmation cuối cần khóa GPU architecture, environment và evaluator version.
