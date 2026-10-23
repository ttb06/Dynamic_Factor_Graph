# Follow-up: temporal reliability, selective recheck và typed actuators

**Ngày:** 2026-09-22  
**Phạm vi:** follow-up trên Adult TabDiff pilot; mọi guidance vẫn là rule/oracle proxy, không được gọi là LLM end-to-end.  
**Quy ước:** `[RUN]` đã chạy; `[INFERENCE]` diễn giải; `[PENDING]` job đã submit nhưng chưa dùng để kết luận.

## 1. Hypothesis hẹp và kill criteria

| ID | Hypothesis | Kill criterion | Trạng thái hiện tại |
|---|---|---|---|
| H7 | Violation tồn tại ở hai preview liên tiếp, với antecedent category ổn định, dự báo final violation tốt hơn một preview | Precision không tăng nhất quán hoặc recall/validity giảm | **Bác bỏ trên pilot** |
| H8 | Recheck muộn chỉ trên row đã sửa giúp chống denoiser undo với ít critic evaluations hơn recheck toàn bộ | Không cải thiện validity, hoặc actuator đúng làm recheck không còn cần | **Bác bỏ như main idea; là workaround cho weak actuator** |
| H9 | No-op filter, repeat consensus và typed applicability contract cứu được precision của LLM critique | Precision vẫn thấp; hoặc contract phải encode toàn bộ verifier | **Contract hiệu quả nhưng phụ thuộc verifier** |
| H10 | Numeric và categorical cần actuator khác nhau: x0 update cho numeric bị attenuation, discrete state update bền hơn | Direct numeric state projection không cải thiện durability/Pareto | **Ủng hộ cơ chế; chưa hơn repair** |

## 2. H7 — temporal persistence không đủ giá trị

### Forecast cuối trajectory [RUN]

Ba seed × 512 records, 30 steps. Tại `t≈.10`, instantaneous preview đã có precision rất cao khi dự báo final violation:

| Rule | Instant precision / recall | Persistent + parent-stable precision / recall |
|---|---:|---:|
| education FD | 0.988 / 0.953 | 0.989 / 0.934 |
| Husband→Male | 0.937 / 0.847 | 0.947 / 0.796 |
| Wife→Female | 0.938 / 0.872 | 0.928 / 0.784 |
| spouse→married | 0.982 / 0.748 | 0.981 / 0.672 |

Persistence chỉ tăng precision đáng kể khoảng `+1.1 pp` cho Husband rule, giảm precision ở hai rule khác, và giảm recall `2–9 pp`. Trong hai GPU replays × ba seeds, `persistent_late_once` có final violation **cao hơn** `late_once` ở cả 6/6 blocks, trung bình `+0.716 pp`.

**[INFERENCE]** Không nên dùng “hai preview đồng ý” làm contribution chính. Preview muộn của executable verifier vốn đã có precision cao; lỗi LLM education trước đó đến từ thiếu mapping, không phải chủ yếu do temporal noise.

Files: `outputs/niche_persistence_1027702/trajectory_violation_forecast.csv` và `trajectory_violation_forecast.png`.

## 3. H8 — selective recheck có signal, nhưng là component

### Ba-seed result [RUN]

| Variant | Hard violation | Critic record-evals proxy | Rare-pair coverage | Residual của initial actions |
|---|---:|---:|---:|---:|
| no guidance | 55.47% | 0 | 78.57% | — |
| one call `t=.10` | 37.43% | 512 | 78.57% | 56.33% |
| persistent one-call | 38.15% | 1,024 | 80.95% | 56.48% |
| targeted recheck `.10→.03` | **28.65%** | **781.7** | 78.57% | 40.86% |
| recheck all `.10→.03` | 26.95% | 1,024 | 76.19% | 40.86% |
| post-repair | **0%** | 0 | 71.43% | 0% |

Targeted recheck giảm `8.80 pp` so với one-call trong completed replay, dùng ít hơn `23.7%` record evaluations so với recheck-all. Qua hai hardware replays × ba seeds, improvement của targeted recheck so với one-call giữ dấu ở **6/6** blocks, trung bình `-8.14 pp`. Recheck-all chỉ tốt hơn targeted thêm `1.95 pp`, cũng giữ dấu 6/6, nhưng rare-pair coverage giảm trung bình `2.38 pp`.

### Timing sweep [RUN]

| Schedule | Hard violation | Critic record-evals | Rare-pair coverage |
|---|---:|---:|---:|
| single `.35` | 45.51% | 512 | 78.57% |
| single `.17` | 39.78% | 512 | 80.95% |
| single `.10` | 37.43% | 512 | 78.57% |
| single `.03` | 35.16% | 512 | 73.81% |
| targeted `.17→.03` | 31.12% | 772.7 | 80.95% |
| targeted `.10→.03` | **28.65%** | 781.7 | 78.57% |

Single ultra-late `.03` với **x0 actuator** không thay thế được recheck: nó kém `.10→.03` `6.51 pp` validity và `4.76 pp` rare-pair coverage. Tuy vậy Section 5 cho thấy đây chủ yếu là actuator attenuation: direct state projection một lần ở `.03` tốt hơn cả hai. H8 không nên là main idea; targeted recheck chỉ còn là fallback khi action không thể project trực tiếp.

Files: `outputs/niche_timing_sweep_1028009/`.

## 4. H9 — compiler quan trọng hơn self-consistency

Audit dùng lại đúng 9 cases × 2 repeats/condition; không gọi API mới.

| Condition/gate | Precision | Recall | F1 |
|---|---:|---:|---:|
| schema-only raw | 0 | 0 | — |
| static rules raw | 28.57% | 100% | 44.44% |
| static + repeat intersection | 30.77% | 100% | 47.06% |
| static + reject explicit no-op | 44.44% | 100% | 61.54% |
| no-op + repeat intersection | 57.14% | 100% | 72.73% |
| typed antecedent/range contract | 100% | 100% | 100% |

Typed contract loại critique cho Husband/Wife rule khi relationship là `Not-in-family/Other-relative`, và loại sign correction khi numeric hiện đã thỏa range. Kết quả hoàn hảo chỉ trên tập rất nhỏ; quan trọng hơn, contract đã encode một phần rule verifier.

**[INFERENCE]** Self-consistency không cứu systematic false positive. Kiến trúc hợp lý phải là `LLM proposes relation/evidence → executable compiler checks status/applicability → typed action`, không để free-form suggestion trực tiếp sửa sampler. Nhưng trên hard rules hiện tại, compiler đủ thông tin để bỏ LLM hoàn toàn.

Files: `outputs/niche_critique_gate_1027700/`.

## 5. H10 — mixed-type actuator là signal mạnh nhất

Ba seed × 512 records, cùng preview/rule critic và timing; chỉ thay cách tác động numeric:

| Variant | Hard violation | Education violation | Numeric KS | Rare-pair | Utility AUROC |
|---|---:|---:|---:|---:|---:|
| no guidance | 55.47% | 49.22% | 0.08996 | 78.57% | 0.81245 |
| x0 correction `.03` | 35.16% | 34.77% | 0.08531 | 73.81% | 0.80932 |
| **state projection `.03`** | **1.76%** | **0.91%** | 0.09170 | 73.81% | 0.80250 |
| state projection `.10` | 6.05% | 3.91% | 0.09139 | 78.57% | 0.80501 |
| state targeted recheck `.10→.03` | 5.34% | 3.19% | 0.09139 | 78.57% | 0.80498 |
| post-repair | **0%** | **0%** | 0.09158 | 71.43% | 0.80232 |

Ở single `.03`, `70.07%` education actions qua x0 prediction vẫn vi phạm cuối trajectory; direct state projection giảm residual xuống `0.53%`. Các categorical actions đã có residual 0% vì implementation vốn ghi trực tiếp discrete next-state. Đây là bằng chứng type-specific rất rõ và ổn định ở cả ba seed.

Nhưng state projection gần như trùng post-repair về numeric KS, joint proxy và utility; cả hai giảm AUROC khoảng `1 pp` so với no-guidance trong small TSTR. Vì vậy chưa có Pareto advantage. File: `outputs/niche_actuator_1028014/`.

## 6. Replay stability và giới hạn metric

Hai run cùng seed trên L4/T4 cho qualitative contrasts ổn định nhưng số tuyệt đối có nhiễu:

- late-one-call cải thiện violation so với no-guidance trong 6/6 blocks (`-17.25 pp` trung bình);
- targeted recheck cải thiện so với one-call trong 6/6 (`-8.14 pp`);
- persistence tệ hơn one-call trong 6/6 (`+0.716 pp`);
- utility AUROC và conditional-gap không giữ dấu;
- rare-pair coverage có mean absolute replay difference `9.18 pp`, tối đa `21.43 pp`, vì denominator/support nhỏ.

Do đó chỉ dùng hard-violation contrasts để planning hiện tại; chưa claim fidelity/utility gain. Files: `outputs/niche_replay_stability_1028010/`.

## 7. Novelty boundary

Không claim generic adaptive-timestep guidance: [Trust Sampling, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/aa8164b8a8e484fe31885657fea14997-Paper-Conference.pdf) đã dùng timestep-dependent trust schedule và manifold gate. Không claim intermediate state dự báo outcome là mới: [Probe-Select, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Guo_Toward_Early_Quality_Assessment_of_Text-to-Image_Diffusion_Models_CVPR_2026_paper.html) làm early image-quality prediction. Hiện tượng later denoising hấp thụ intervention cũng đã được quan sát ngoài tabular, ví dụ [Purification Paradox, CVPR 2026 workshop](https://openaccess.thecvf.com/content/CVPR2026W/SAFE/html/Kamble_The_Purification_Paradox_Dissecting_and_Exploiting_Generative_Vulnerability_Bands_in_CVPRW_2026_paper.html).

Khoảng trống còn defend được phải là **mixed-type, relation-specific actuation và verified transaction semantics**, cộng evaluation về exception/rare-mode; không phải scheduling đơn thuần.

## 8. Proposal nên giữ

### P1 — Verified typed feedback transaction

LLM chỉ trả relation candidate, involved columns, evidence và enum `violated / satisfied / not_applicable / abstain`. Compiler kiểm tra antecedent/type/range, chọn actuator theo kiểu cột, commit action; independent verifier và local fidelity score quyết định rollback. Một targeted recheck chỉ chạy trên committed rows.

- So với PromptLoop: không update prompt tự do; action là atomic typed transaction.
- So với SCTab-Diff: relation/action thay đổi theo record và preview, không chỉ static prior.
- So với Harpoon: hỗ trợ non-differentiable semantic/tool result và categorical actuator.
- So với LCATS: commit trước khi trajectory kết thúc và đo action survival/rollback.
- Kill gate: typed oracle phải hơn post-repair trên constraint không local-repairable; sau đó LLM+compiler phải đạt ≥90% oracle gain với calibrated abstention.

### P2 — LLM làm sparse verifier router, không làm judge

Với domain có nhiều ontology/API/rule tools, LLM chọn verifier nào cần gọi cho từng stable preview; tool/verifier tạo truth signal và typed action. Contribution là cost-aware routing + denoising transaction, không phải kiến thức nội tại của LLM.

- Chỉ feasible khi số verifier lớn, sparse applicability, và cost evaluate-all đáng kể.
- Adult/Default/Shoppers hiện không đáp ứng; hard rules quá rẻ, Browser/Traffic IDs không có semantics.
- Dataset tiếp theo phải có meaningful metadata và external ground truth, ví dụ temporal/event records hoặc coded domain có public ontology.

### P3 — Type-specific actuator controller

Categorical correction dùng remask/projection; numeric equality dùng state projection hoặc differentiable loss; text dùng constrained regeneration. Controller học/chọn actuator và thời điểm từ rule type + preview stability; action survival là metric riêng.

- Đây là component kỹ thuật có signal mạnh nhất, nhưng novelty risk cao nếu chỉ là schedule/projection.
- Cần H10 và full-checkpoint validation trước khi nâng thành proposal chính.

## 9. Jobs và validation gate

| Job | Status | Vai trò |
|---:|---|---|
| `1027699` | failed after outputs | L4 replay; lỗi chỉ ở final print, CSV hoàn chỉnh, giữ provenance |
| `1027700` | completed | critique contract audit |
| `1027702` | completed | corrected persistence/recheck, T4 replay |
| `1028009` | completed | timing sweep |
| `1028010` | completed | cross-replay stability |
| `1028014` | completed | typed actuator ablation |
| `1028011` | submitted | Adult 8,000-epoch training; 500-epoch pilot chỉ bằng 6.25% config |
| `1028012` | dependency | persistence/recheck trên full checkpoint |
| `1028013` | dependency | timing sweep trên full checkpoint |
| `1028018` | dependency | typed actuator trên full checkpoint |

Không dùng full-run jobs để claim cho tới khi Slurm báo `COMPLETED` và outputs được audit.
