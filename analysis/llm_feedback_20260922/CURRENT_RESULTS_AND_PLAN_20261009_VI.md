# Dynamic constraint graph cho tabular diffusion: trạng thái kết quả và kế hoạch

> Cập nhật mới nhất về trajectory actionability, actuator portfolio, timing và persistence: `TRAJECTORY_ACTIONABILITY_UPDATE_20261009_VI.md`.
>
> Bản handoff tự chứa, tổng hợp toàn bộ bài toán, method, implementation, versions, kết quả và roadmap: `FULL_PROJECT_REPORT_DYNAMIC_GRAPH_20261009_VI.md`.

Ngày cập nhật: 09/10/2026.

## Kết luận ngắn

Hướng nghiên cứu đã đủ rõ để bắt đầu viết problem statement, method và experimental protocol. Kết quả chưa đủ để claim SOTA hoặc khóa main table.

Kết quả mạnh nhất hiện tại là:

- dynamic graph thay đổi theo row và timestep thắng frozen graph nhất quán;
- middle guidance giảm heldout semantic error khoảng 20–25% so với TabDiff unconditional tại operating point phù hợp;
- scale vừa (`2`) gần đạt semantic validity của final repair trong khi giữ KS/TV tốt;
- nhưng final repair vẫn là baseline mạnh nhất trên Shoppers và thường rẻ hơn;
- GraphFT tạo tín hiệu thật nhưng chỉ dịch chuyển Pareto frontier, chưa cải thiện toàn bộ metrics.

Do đó claim nên tập trung vào **state-dependent active graph và trajectory intervention**, không tập trung vào exception, rare subgroup, số cạnh graph, hoặc end-to-end training.

## 1. Bài toán và method hiện tại

Knowledge graph được tạo từ schema/domain knowledge, train-only statistical discovery, executable validation, counterexamples và semantic review. Mỗi edge có applicability, requirement, hardness, confidence và action authority.

Trong reverse diffusion, clean preview `x_hat_0(x_t)` được decode. Với mỗi row, method recompute:

`active edges = readable at t ∩ applicable to current row ∩ authorized for action`.

Action layer tạo các candidate corrections, sau đó chọn bằng typed verifier, train-only density và edit cost. Correction được đưa vào clean prediction rồi reverse update tiếp tục. LLM chủ yếu dùng trong offline knowledge discovery/review; runtime action dùng executable graph để ổn định chi phí và khả năng kiểm chứng.

## 2. Main T4-matched result

8,192 Shoppers samples, cùng TabDiff checkpoint, split, graph, evaluator và seed.

| Method | Macro error | Heldout error | Calibration error | Numeric KS | Cat. TV | Actions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TabDiff unconditional | 0.06427 | 0.01695 | 0.01720 | **0.02534** | 0.02203 | 0 |
| Final repair | **0.05840** | **0.01282** | **0.01316** | 0.02547 | 0.02203 | **258** |
| Dynamic middle, scale 1 | 0.05965 | 0.01382 | 0.01337 | 0.02541 | 0.02191 | 616 |
| Dynamic middle, scale 2 | 0.05868 | 0.01298 | 0.01322 | **0.02537** | **0.02188** | 457 |
| Dynamic middle, scale 4 | 0.05915 | 0.01336 | 0.01384 | 0.02557 | 0.02177 | 368 |
| Epoch-100 GraphFT + scale 2 | 0.05919 | 0.01294 | **0.01223** | 0.02618 | 0.02215 | 415 |

Scale 2 là operating point chính: heldout error giảm 23.4% so với unconditional và chỉ kém final repair khoảng 1.2%, trong khi KS/TV tốt hơn nhẹ. GraphFT epoch 100 giảm calibration error và actions, nhưng làm macro/KS/TV xấu hơn.

## 3. Common quality và utility metrics

| Method | Shape error | Trend error | XGBoost AUC | Heldout semantic error |
| --- | ---: | ---: | ---: | ---: |
| TabDiff unconditional | 1.3074 | 1.5476 | 0.9054 | 0.01691 |
| Final repair | 1.3025 | 1.5411 | 0.9034 | **0.01278** |
| Middle scale 1 | **1.2968** | **1.5343** | **0.9080** | 0.01377 |
| Middle scale 2 | 1.3035 | 1.5437 | 0.9006 | 0.01293 |
| Epoch-100 final repair | 1.3354 | **1.5379** | 0.9046 | 0.01293 |
| Epoch-100 middle scale 2 | **1.3280** | 1.5471 | **0.9074** | **0.01289** |

Scale 1 là fidelity/utility point; scale 2 là semantic point. Strong scale 4 làm pairwise trend error tăng mạnh và không nên là default.

CTGAN/TVAE local baselines kém xa TabDiff về shape/trend fidelity trên Shoppers. Chúng xác nhận backbone choice nhưng không phải đối thủ chính cho dynamic guidance.

## 4. Cross-dataset evidence

Các hàng trong bảng dùng evaluator matched trong từng dataset; không so absolute values giữa datasets.

| Dataset | Method | Shape error | Trend error | XGBoost AUC | Heldout semantic error | Gain so với TabDiff |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Default | TabDiff | 1.6429 | 1.7127 | 0.7482 | 0.01869 | — |
| Default | Dynamic graph | **1.6366** | **1.7083** | **0.7530** | **0.01404** | **24.9%** |
| Default | Final repair | 1.6434 | 1.7127 | 0.7462 | 0.01496 | 19.9% |
| Diabetes | TabDiff | **1.8840** | **11.3025** | 0.6459 | 0.01317 | — |
| Diabetes | Dynamic graph | 1.9195 | 11.8309 | **0.6466** | 0.00888 | 32.6% |
| Diabetes | Final repair | 1.8873 | 11.6304 | 0.6434 | **0.00838** | **36.4%** |
| Shoppers | TabDiff | 1.3074 | 1.5476 | **0.9054** | 0.01691 | — |
| Shoppers | Dynamic graph, scale 2 | 1.3035 | 1.5437 | 0.9006 | 0.01293 | 23.5% |
| Shoppers | Final repair | **1.3025** | **1.5411** | 0.9034 | **0.01278** | **24.4%** |

Default ủng hộ middle intervention rõ nhất. Diabetes là negative control: post-repair thắng semantic metric. Shoppers cho thấy hai method gần nhau và cần report trade-off.

## 5. Direct test cho dynamic topology

| Model | Topology | Macro error | Heldout error | Calibration error |
| --- | --- | ---: | ---: | ---: |
| Pretrained | **Dynamic** | **0.05868** | **0.01298** | **0.01322** |
| Pretrained | Frozen | 0.05924 | 0.01307 | 0.01339 |
| Pretrained | Frozen state | 0.05945 | 0.01309 | 0.01363 |
| Epoch 100 | **Dynamic** | **0.05919** | **0.01294** | **0.01223** |
| Epoch 100 | Frozen | 0.05960 | 0.01302 | 0.01236 |
| Epoch 100 | Frozen state | 0.06037 | 0.01334 | 0.01279 |

Effect trên Shoppers nhỏ nhưng nhất quán. Default trước đó cho effect lớn hơn: dynamic tốt hơn frozen khoảng 10.0% macro và 17.5% heldout. Đây là bằng chứng trực tiếp nhất cho contribution “graph thay đổi dọc trajectory”.

## 6. Hypotheses đã bác bỏ hoặc thu hẹp

| Hypothesis | Kết luận |
| --- | --- |
| Càng tăng guidance càng tốt | Bác bỏ; scale 4 phá trend, scale 2 tốt hơn |
| Nhiều edge/connectivity cao tự tạo gain | Bác bỏ; improvement chỉ 0.0–0.5%, hyperedge có thể làm xấu 8–10% |
| GraphFT mạnh và lâu hơn tốt hơn | Bác bỏ; epoch 500 drift nhiều, epoch 100 tốt hơn |
| Hard-only graph objective an toàn hơn | Bác bỏ; Numeric KS tăng mạnh |
| Weak middle + final repair tạo gain lớn | Bác bỏ; gần final-only nhưng tốn khoảng 1.8–2.6 lần actions |
| Dynamic luôn thắng final repair | Bác bỏ; phụ thuộc dataset và metric |
| Readability trở nên sớm hơn sau GraphFT | Chưa được ủng hộ; mean cutoff hầu như không đổi |

## 7. Baseline status

| Baseline | Trạng thái | Có dùng để claim trực tiếp không? |
| --- | --- | --- |
| TabDiff | Local checkpoint, matched split/evaluator/seed | Có |
| CTGAN, TVAE | Local SDV training, matched split; quality metrics hoàn chỉnh | Có, nhưng là weak backbone baselines |
| Final typed repair | Matched graph, seed, verifier và samples | Có; đối thủ mạnh nhất hiện tại |
| Best-of-N | Đã chạy ở các batches trước; compute-heavy selection baseline | Có với đúng batch/evaluator |
| Harpoon official code | Đã chạy official tasks, nhưng khác constraint protocol | Chỉ external evidence |
| CTDF-style | Proxy local; chưa faithful same-task official implementation | Không claim paper reproduction |
| LCATS official | Đã audit Insurance outputs; khác dataset/task | Chỉ external evidence |
| CuTS official | Đã audit German constraints; khác dataset/task | Chỉ external evidence |

Vì chưa có faithful Harpoon/CTDF/LCATS trên cùng frozen graph và split, chưa được viết “SOTA”.

## 8. Kế hoạch ưu tiên

1. Freeze hai main operating points: scale 1 và scale 2 trên pretrained TabDiff; GraphFT epoch 100 chỉ là secondary variant.
2. Port faithful same-task baselines: differentiable guidance kiểu Harpoon, per-step feasibility/projection kiểu CTDF, và post-repair kiểu LCATS/Zhao trên cùng graph/evaluator.
3. Confirm Default và Shoppers với 3 seeds cho TabDiff, dynamic scale 1/2, frozen topology và final repair. Diabetes giữ làm negative control, không tune để ép thắng.
4. Chọn hyperparameters chỉ bằng train/calibration split; heldout graph chỉ mở cho final table.
5. Đóng knowledge-discovery protocol: provenance, counterexamples, atom minimality, semantic authority và graph coverage. Không tiếp tục tăng edge count nếu action layer vẫn edge-local.
6. Bắt đầu viết ngay phần problem, gap, method, benchmark và negative results. Main experiments/baselines cần khoảng 7–10 ngày làm việc; full paper polish và additional datasets cần thêm khoảng 1–2 tuần.

### Confirmation ba seed

Batch `1062391[0-3]` đã hoàn tất; aggregate đầu `1062392` lỗi do cột method bị ghi đè và được sửa trong `1062396`. Kết quả ba seed:

| Method | Macro error | Heldout error | Calibration error | Numeric KS | Cat. TV | Actions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unconditional | 0.06387 ± 0.00090 | 0.01903 ± 0.00285 | 0.01729 ± 0.00168 | 0.02507 ± 0.00103 | 0.02218 ± 0.00255 | 0 |
| Final repair | **0.05809 ± 0.00031** | **0.01488 ± 0.00232** | **0.01301 ± 0.00082** | 0.02518 ± 0.00098 | 0.02218 ± 0.00255 | 247.7 |
| Dynamic, scale 2 | 0.05843 ± 0.00029 | 0.01506 ± 0.00238 | 0.01310 ± 0.00088 | 0.02518 ± 0.00082 | **0.02199 ± 0.00242** | 419.7 |
| Frozen topology | 0.05885 ± 0.00051 | 0.01521 ± 0.00251 | 0.01312 ± 0.00090 | **0.02508 ± 0.00090** | 0.02208 ± 0.00245 | 449.3 |

Dynamic giảm heldout error `0.93% ± 0.74%` so với frozen trong cả ba seed, nhưng còn kém final repair `1.22% ± 0.34%`. Đây là effect có tính lặp lại nhưng nhỏ; không được mô tả là dominance.

## 9. Kiến trúc action mới: connected-component planner

Audit trên unconditional sample 8,192 rows cho thấy 578 rows vi phạm ít nhất một actuator edge và 224/578 rows (`38.8%`) đồng thời vi phạm ít nhất hai edges. Riêng hai relations gần tương đương trên `Administrative–Administrative_Duration` cùng vi phạm ở 182 rows. Action layer cũ xử lý từng edge tuần tự, nên không dùng được cấu trúc connected component và có thể chọn hai corrections không nhất quán.

Planner mới xây active subgraph riêng cho từng row/timestep, tách thành các connected components theo shared columns, sinh các endpoint/deactivation/abstention candidates và dùng beam search để tối thiểu hóa một energy chung:

`active constraint violations + newly introduced graph violations + train-density cost + edit cost`.

Điểm quan trọng là graph bây giờ tham gia trực tiếp vào quyết định joint action, không chỉ làm gate. Beam width 4 đủ hội tụ trong smoke test; beam 8 cho output giống beam 4.

Smoke test 1,024 rows, cùng checkpoint/seed:

| Planner | Macro error | Heldout error | Calibration error | KS | TV | Actions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Edge-local | **0.05713** | 0.01697 | 0.01451 | 0.03185 | 0.03759 | 57 |
| Component beam 2 | 0.05739 | 0.01684 | 0.01435 | **0.03152** | **0.03731** | 55 |
| Component beam 4 | 0.05738 | **0.01683** | 0.01436 | 0.03162 | **0.03731** | **51** |
| Final repair | 0.05675 | 0.01620 | **0.01433** | 0.03172 | 0.03759 | 30 |

Tín hiệu ban đầu phù hợp với hypothesis: component planning giảm heldout error, TV và số interventions, nhưng chưa vượt final repair và macro error hơi xấu hơn edge-local. Full run cần xác nhận đây không phải noise cỡ mẫu nhỏ.

Full strength refinement trên 8,192 rows tìm thấy scale `1.5` là point tốt nhất hiện tại:

| Method | Macro error | Heldout error | Calibration error | KS | TV | Actions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Final repair | **0.05840** | 0.01282 | 0.01316 | 0.02547 | 0.02203 | **258** |
| Component scale 1.5 | 0.05871 | 0.01280 | 0.01318 | **0.02537** | **0.02181** | 514 |
| Component scale 1.5 + certification | 0.05850 | **0.01275** | **0.01309** | **0.02537** | **0.02181** | 532 |

Common evaluator cho thấy component scale 1.5 không chỉ cải thiện semantic metric trên seed chọn: so với final repair, online variant giảm shape error `1.3025 → 1.2925`, trend error `1.5411 → 1.5392`, tăng XGBoost AUC `0.9034 → 0.9049`, và giảm heldout error `0.012775 → 0.012755`. Certified variant giảm heldout xuống `0.012699` nhưng AUC thấp hơn.

Confirmation ba seed trong `1062429` cho thấy component planner thắng edge-local nhất quán `0.44% ± 0.24%` heldout error, nhưng component + certification không thắng final-only: trung bình còn kém `1.05% ± 1.63%`. Vì vậy scale 1.5 là một Pareto point tốt trên seed discovery, không phải SOTA result đã xác nhận. Failure mode mới là nhiều middle actions sửa lỗi mà denoiser sẽ tự sửa hoặc final certifier xử lý rẻ hơn; batch `1062452[0-2]` đang thử one-step counterfactual lookahead để chỉ chấp nhận action có causal benefit ở bước kế tiếp.

Target-only counterfactual lookahead đã được chuyển sang A40 và hoàn tất trong job `1062466`: nó chỉ giữ 14/59 component actions, nhưng online heldout error xấu từ `0.01999` lên `0.02063`; sau certification vẫn kém final-only (`0.01986` so với `0.01950`). Global active-graph energy trong `1062470` cho kết quả giống hệt target-only. Horizon 2, 3 và 5 trong `1062475` cũng giữ đúng 14/59 actions và tạo cùng output. Endpoint diagnostic horizon 10/25/50 trong `1062487` giữ 10/57 actions, nhưng online heldout error `0.01913` vẫn kém final repair `0.01854`; certification cũng không khắc phục được. Vì vậy rollout lookahead giúp giảm số action nhưng không tạo quality gain, và không được giữ làm main method.

Về Slurm, job T4 `1062452` bị dự đoán chờ tới tối hôm sau dù A10/L40S/A40 còn rảnh, vì nó khóa GPU type và xin 3 giờ. Smoke jobs hiện được route qua `aicentre` hoặc `aicentre-a100` với `qos=unprivileged` và time limit ngắn; main-table confirmation vẫn khóa T4. Helper `slurm/submit_route.sh` hỗ trợ `fast`, `fast-a100`, `opportunistic`, và `t4`.

Các batches refinement gần đây đều đã hoàn tất:

- `1062405[0-5]` + aggregate `1062406`: 8,192 rows; beam 4; guidance scale `{1,2,4}`, graph guardrail weight `{1,5,10}`, và selective energy; hoàn tất.
- `1062412[0-2]`: trajectory-certified actuator portfolio; hoàn tất.
- `1062428[0-4]` + `1062429`: xác nhận scale 1.5 trên ba seeds; component thắng edge-local nhưng chưa thắng final-only.
- `1062466[0-2]`: target-only counterfactual lookahead; hoàn tất và không được giữ.
- `1062470[0-2]`: global-graph one-step lookahead; hoàn tất, không cải thiện target-only.
- `1062475[0-3]`: horizon 2/3/5; hoàn tất, output không đổi so với horizon 1.
- `1062487[0-2]`: endpoint-lookahead upper bound; hoàn tất, giảm actions nhưng vẫn thua final repair.

Tiêu chí giữ architecture: full-sample heldout/calibration error phải tốt hơn edge-local; component + final certification phải ít nhất khớp final repair và cải thiện một fidelity/utility metric; gain không được đến từ bỏ evaluator edges.

## 10. Artifact chính

- Full report: `PROPOSAL_DYNAMIC_GRAPH_TONG_HOP_20261008_VI.md`
- T4 controller frontier: `outputs/e2e_t4_confirmation_1062366/RESULTS.md`
- Common metrics: `outputs/e2e_t4_common_metrics_1062367/RESULTS.md`
- Topology controls: `outputs/e2e_t4_topology_1062384/RESULTS.md`
- Hybrid audit: `outputs/e2e_weak_hybrid_common_1062376/RESULTS.md`
- End-to-end checkpoint frontier: `outputs/e2e_checkpoint_frontier_1062340/RESULTS.md`
- Three-seed confirmation: `outputs/main_scale2_multiseed_1062396/RESULTS.md`
- Component-planner smoke outputs: `outputs/component_planner_smoke_*_1062401_*/`
- Full component frontier: `outputs/component_planner_aggregate_1062406/RESULTS.md`
- Selected common metrics: `outputs/selected_component_common_1062427/RESULTS.md`
- Scale-1.5 confirmation: `outputs/selected_component_multiseed_1062429/RESULTS.md`

## 11. Structural action authority và timing mới

Job `1062484` kiểm tra việc chỉ cho middle controller can thiệp khi một row có nhiều violated edges trong cùng connected component. Trên 4,096 Shoppers samples:

| Minimum edges/component | Middle actions | Online macro | Online heldout | Certified macro | Certified heldout |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 267 | **0.05371** | **0.01591** | 0.05346 | 0.01576 |
| 2 | 52 | 0.05608 | 0.01766 | **0.05331** | 0.01524 |
| 3 | 2 | 0.05812 | 0.01827 | 0.05341 | **0.01514** |
| Final repair | 126 | — | — | 0.05341 | 0.01514 |

Ngưỡng 2 giảm 80.5% middle actions và sau certification cải thiện macro/KS nhẹ so với final repair, nhưng heldout error còn xấu khoảng 0.6% và tổng actions cao hơn. Ngưỡng 3 gần như triệt tiêu middle intervention rồi tái tạo final repair. Kết luận: connected-component authority là một efficiency/Pareto control hợp lý, chưa phải source của SOTA gain.

Timing hypothesis mới tách safety khỏi coverage. Readability schedule cũ yêu cầu cả precision và recall cao; script `scripts/build_precision_onset_schedule.py` mở actuator sớm bằng Wilson lower bound của violation precision và minimum support. Job `1062490[0-3]` đã bác bỏ giả thuyết rằng precision-only onset tạo quality gain: reference đạt online heldout `0.014898`; các mức lower bound 0.75/0.65/0.50 lần lượt đạt `0.015031/0.015411/0.015341` và dùng nhiều actions hơn (`367/384/409` so với `335`). Một vài fidelity/calibration metrics cải thiện rất nhỏ, nhưng không bù được heldout loss. Recall trong schedule vì vậy đang đóng vai trò stability gate hữu ích, không chỉ là yêu cầu coverage; reference schedule được giữ.

Action-family hypothesis trong `1062502` thêm conditional prototype từ train-only neighbors thỏa active component, aggregate bằng median/mode để tránh copy toàn bộ real row. Prototype chỉ được chọn 2–4 lần trên 2,048 samples. Nó giảm nhẹ macro error, TV và actions nhưng làm heldout error xấu từ `0.009357` lên `0.009424`. Nhánh này được giữ như ablation về manifold-aware repair, không thêm vào main architecture.

Tổng hợp riêng của các refinement experiments: `outputs/methodology_refinement_20261009/RESULTS.md`.
