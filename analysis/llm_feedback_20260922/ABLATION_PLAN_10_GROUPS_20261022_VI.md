# Kế hoạch ablation tập trung — 10 nhóm

Mục tiêu: đủ causal evidence cho paper nhưng tránh Cartesian sweep. Adult là positive case; Shoppers là transfer case; News dùng cho trajectory/public constraints; Default và Magic dùng làm boundary cases.

## Nguyên tắc chạy

- Screening dùng 2 seeds; chỉ confirm arm tốt nhất bằng 3–5 fresh seeds.
- Ablation chỉ thay selector phải reuse exact generated pool.
- Ablation thay trajectory/action phải paired cùng seed, checkpoint và base noise trong một Slurm task.
- Không dùng test metrics để chọn hyperparameters.
- Main paper ưu tiên Nhóm 1–7; Nhóm 8–10 có thể đưa một phần vào appendix.

## 1. End-to-end causal chain

**Câu hỏi:** gain đến từ generation, oversampling, generic C3 hay shared-graph repair?

**Arms tối thiểu:**

1. `B`: backbone;
2. `G`: dynamic graph generator;
3. `R`: random pool subset;
4. `old`: frozen generic C3;
5. scalar graph selector;
6. constrained full redraw;
7. proximal graph-first;
8. proximal quality-first;
9. deficit-gated final.

**Dataset:** Adult + Shoppers.  
**Compute:** phần lớn reuse pool; chỉ `B/G` cần generation.  
**Paper output:** bảng causal chain chính và quality–graph Pareto plot.

## 2. Graph source và authority

**Câu hỏi:** gain cần loại graph nào, và verification/authority có thực sự quan trọng không?

**Arms tối thiểu:**

- no graph;
- train-statistical graph;
- verified typed graph;
- mixed typed + distribution graph.

Focused sub-ablation: no-LLM versus LLM-proposed + deterministic verification.

**Dataset:** Adult + News.  
**Compute:** cần regenerate.  
**Paper output:** graph-source table, edge count/support và discovery precision/recall.

## 3. Dynamic activation và routing

**Câu hỏi:** row/time-dependent activation có tốt hơn fixed global factor không?

**Arms tối thiểu:**

- global/static;
- row-conditioned only;
- time-conditioned only;
- row + time dynamic;
- final dynamic without uncertainty/abstention.

**Dataset:** Adult + News.  
**Compute:** paired generation.  
**Paper output:** main mechanism ablation và active-edge-over-time figure.

## 4. Timing và pulse schedule

**Câu hỏi:** thời điểm correction nào hữu ích và continuous guidance có gây off-manifold không?

**Arms tối thiểu:**

- early one-shot;
- middle one-shot;
- late one-shot;
- continuous guidance.

Optional: two weak middle pulses nếu cần giải thích variance.

**Dataset:** Adult + News.  
**Compute:** paired generation.  
**Paper output:** timing curve, fidelity/coverage/runtime trade-off.

## 5. Action family, strength và trust

**Câu hỏi:** operator nào thực sự tạo gain và bounded geometry đóng góp gì?

**Arms tối thiểu:**

- projection only;
- numerical transport only;
- categorical message only;
- full typed actions;
- full without trust region;
- full với strength `weak/locked/strong`.

**Dataset:** Adult cho categorical; News cho numerical/typed.  
**Compute:** paired generation.  
**Paper output:** action-family factorial và action magnitude/runtime table.

## 6. C3 và output-selection objective

**Câu hỏi:** graph-aware selection tốt hơn generic selection theo cơ chế nào?

**Arms tối thiểu:**

- no selection;
- uniform subset;
- marginal/pair raking;
- frozen consensus C3;
- additive scalar graph;
- constrained full redraw;
- proximal final.

**Dataset:** Adult + Shoppers; dùng Magic làm redraw failure case.  
**Compute:** reuse pool.  
**Paper output:** set-selection table và candidate Pareto frontier.

## 7. Proximal repair và deficit gate

**Câu hỏi:** sparse-repair budget và abstention threshold ảnh hưởng robustness như thế nào?

Không chạy full grid. Dùng sequential axes:

- swap fraction: `0`, `0.25%`, `0.5%`, `1%`;
- required graph gain: `0`, `5%`, `10%`;
- quality budget: `0`, `1.5%`, `3%`;
- activation threshold: `0`, `1e-3`, `2e-3`.

Giữ ba axes ở locked value khi sweep axis còn lại.

**Dataset:** Adult positive + Default boundary.  
**Compute:** reuse pool.  
**Paper output:** sensitivity plots và no-op/repair-rate analysis.

## 8. Pool budget và output scale

**Câu hỏi:** gain có chỉ đến từ oversampling lớn hoặc một output size cụ thể không?

**Arms tối thiểu:**

- pool ratio `1×`, `2×`, `4×`;
- output size `4k`, `8k`;
- optional sampler steps `25/50` nếu runtime là concern.

**Dataset:** Adult + Shoppers.  
**Compute:** cần generate pool mới cho pool-ratio arms; subset sizes có thể reuse pool lớn.  
**Paper output:** quality versus pool-compute curve.

## 9. Coverage, subgroup và robustness

**Câu hỏi:** gain tổng thể có che mất rare-subgroup collapse hoặc phụ thuộc một vài factors không?

**Arms tối thiểu:**

- no coverage barrier;
- integrated train-coverage;
- final factor-aware coverage;
- leave-one-factor-out;
- noisy/low-authority edge injection;
- final graph unchanged.

Báo rare-group recall, target-conditioned coverage, applicability support, hard excess và external metrics.

**Dataset:** Adult + Default.  
**Compute:** phần lớn reuse pool.  
**Paper output:** subgroup table và robustness-to-graph-noise figure.

## 10. Efficiency, privacy và reproducibility

**Câu hỏi:** gain có thực tế, an toàn và tái lập được không?

**Measurements tối thiểu:**

- generation/selection wall time;
- peak GPU/host memory;
- guidance calls, active edges/rows/cells;
- number/percentage membership swaps;
- graph compilation và LLM token cost;
- DCR, duplicate rate, nearest-neighbor rate;
- protected/target-column audit;
- one-command replay từ frozen manifest.

**Dataset:** tất cả main datasets; detailed trace chỉ cần Adult + News.  
**Compute:** không cần sweep lớn.  
**Paper output:** efficiency/privacy table và reproducibility appendix.

## Thứ tự ưu tiên trong hai tuần

### Main paper bắt buộc

1. Nhóm 1 — causal chain;
2. Nhóm 3 — dynamic routing;
3. Nhóm 4 — timing;
4. Nhóm 5 — action/trust;
5. Nhóm 6 — selection semantics;
6. Nhóm 7 — proximal/deficit gate;
7. Nhóm 10 — runtime/reproducibility.

### Appendix hoặc chạy nếu còn compute

8. Nhóm 2 — graph source/LLM isolation;
9. Nhóm 8 — pool/output scale;
10. Nhóm 9 — expanded subgroup/noisy-edge robustness.

## Stop rule

- Không mở method mới sau khi final protocol được khóa.
- Không confirm arm không thắng cả paired primary metrics hoặc vi phạm utility floor.
- Một arm fail fresh seeds được giữ làm negative result; không nới gate hậu nghiệm.
- Dừng ablation khi mỗi contribution chính đã có một causal comparison, một positive dataset và một boundary/failure dataset.

