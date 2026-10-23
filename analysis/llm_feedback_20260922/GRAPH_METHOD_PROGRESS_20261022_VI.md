# Báo cáo tiến độ graph-consistent output selection

Ngày chốt vòng hiện tại: 2026-10-22. Toàn bộ generation, selection, audit và evaluation chạy qua Slurm; login node chỉ dùng để chuyển file, `sbatch`, `squeue`, `sacct` và đọc log nhỏ. Không có agent nào chạy trên server.

## Kết luận ngắn

Kết quả tốt nhất hiện tại là **graph-proximal quality-first selection** (`pqs005_g10`). Method chỉ đổi membership của 20/4096 hàng (0,5%) quanh selector C3 đã khóa, nhưng dùng đúng active factor graph của dynamic guidance để quyết định row nào cần swap. Trên fresh Shoppers, method được xác nhận ở mức `coherent`: graph consistency tăng trên 3/3 seed, vẫn thắng dynamic generator rất lớn ở Shape/Trend, đạt 3/5 local SOTA metric records, và phần giảm so với C3 nằm trong utility floor đã khóa trước.

Không nên claim cross-dataset comprehensive ở thời điểm này: Magic cải thiện C3 và graph nhưng chưa thắng dynamic `G` ở Trend; full-redraw Pareto không generalize utility; Diabetes full screens còn chạy.

## Vấn đề khoa học cần giải quyết

Dynamic middle injection dùng factor graph ở row/trajectory level, trong khi C3 cũ chọn output bằng generic marginals/pairs. Hai tầng này dùng hai semantic objective khác nhau. Failure mode là dynamic guidance có thể sửa từng row, nhưng terminal selection lại làm mất coverage hoặc relation của các factor-conditioned subgroups.

Mục tiêu của nhánh mới là dùng chung factor graph ở cả hai tầng mà vẫn giữ C3 cũ làm ablation độc lập:

1. Graph xác định constraint nào được actuate trong diffusion.
2. Dynamic guidance can thiệp theo timestep và theo row.
3. Output selector dùng lại đúng active factors để đo relation, support coverage và hard-factor excess.
4. Generic C3 vẫn là quality anchor, không bị thay bằng một graph objective tùy ý.

## Các method đã thử

### 1. Additive scalar graph selector

Các arm `graph_only`, `gc_lite`, `gc_bal`, `gc_strong`, `gc_cov`, `gc_hard` cộng graph relation/coverage/hard penalties vào C3 objective.

Ưu điểm: chứng minh graph statistics có thể được cải thiện; functional smoke giảm relation `.019140 → .014700`, coverage `.000562 → .000517`, hard excess `.008126 → .007500`.

Nhược điểm: scalarization tạo trade-off khó kiểm soát. Fresh confirmation:

| Dataset | Arm | Shape/Trend gain vs C3 | Graph relation/coverage gain | Quyết định |
|---|---|---:|---:|---|
| Diabetes | `gc_lite` | `+.032037/+.262422` | `+.000204/+0` | Không đủ graph wins toàn diện |
| Magic | `gc_bal` | `-.031124/-.370569` | `+.001453/+0` | Reject |
| Shoppers | `gc_cov` | `-.177187/-.434525` | `+.002952/+.000769` | Reject |

Kết luận: giữ làm ablation cho thấy “graph objective có tác dụng”, nhưng không dùng làm method chính.

### 2. Graph-constrained full-redraw Pareto

`pa_g0_q10`, `pa_g10_q15`, `pa_g25_q20`, `pa_g50_q30` chuyển scalar graph penalty thành feasibility constraints. Candidate phải đạt graph gain định trước, không làm xấu coverage/hard excess, và nằm trong train-only quality tolerance quanh C3. Trong feasible set, method chọn theo generic quality rồi graph relation.

Screen Magic rất tốt với `pa_g25_q20`:

- feasible `3/3`;
- graph relation gain `+.004907`;
- Shape/Trend gain vs C3 `+.019458/+.082075`;
- Alpha/Beta/AUC `+.276206/-.091688/-.002787`;
- `3/5` local SOTA records; mọi gate đều pass.

Nhưng fresh confirmation thất bại:

- graph vẫn tăng `+.002760`, feasible `3/3`;
- Shape/Trend vs C3 `-.048167/+.028892`;
- Trend vs dynamic `G` `-.011890`;
- chỉ `2/5` local SOTA records.

Kết luận: explicit constraints đúng hơn scalar penalty, nhưng redraw toàn bộ subset vẫn làm utility không ổn định. Đây là ablation quan trọng để motivate proximal repair.

### 3. Graph-proximal graph-first repair

Method bắt đầu từ exact C3 selected set. Với mỗi active factor, residual giữa train state distribution và anchor distribution tạo first-order score cho thao tác remove/add. Candidate chỉ swap 0,5–5% membership và phải giữ generic-quality, coverage, hard-excess constraints. Graph-first chọn feasible candidate có graph relation thấp nhất.

Kết quả trung gian cho thấy chỉ 20–41 swaps đã giảm graph relation 20–30%, nhưng tối ưu graph quá mạnh gây Shape cost nhỏ nhưng có hệ thống trên Shoppers. Nhánh này được giữ làm ablation “maximize graph correction”.

### 4. Graph-proximal quality-first repair — method tốt nhất

Method giữ cùng candidate neighborhood và constraints, nhưng đổi thứ tự lexicographic:

1. yêu cầu graph relation gain tối thiểu 10% hoặc 25%;
2. trong feasible set, chọn generic C3 train objective tốt nhất;
3. dùng graph relation làm tie-break.

Winner trên Shoppers và Magic đều là `pqs005_g10`: 20 swaps (0,5%), graph gain tối thiểu 10%, generic quality tolerance 3%, 256 train-only draws.

Screen 3 seed:

| Dataset | Shape/Trend gain vs C3 | Graph relation gain | Alpha/Beta/AUC gain | SOTA wins | Gates |
|---|---:|---:|---:|---:|---|
| Magic | `+.063301/+.004510` | `+.003291` | `+.053711/-.108507/+.001075` | 3/5 | all pass |
| Shoppers | `+.000664/+.003940` | `+.002239` | `+.084093/+.011936/-.001981` | 2/5 | all pass |

Fresh confirmation 3 seed:

| Dataset | Shape | Trend | Alpha | Beta | AUC | Gain vs G (Shape/Trend) | Gain vs C3 | Graph relation/coverage gain | SOTA | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Shoppers | 1.102046 | 1.803424 | 98.909617 | 48.103841 | .902240 | `+.389162/+.234395` | `-.024145/-.030366` | `+.002233/+.000619` | 3/5 | **coherent** |
| Magic | .956562 | 2.334173 | 98.803655 | 48.089193 | .923665 | `+.246281/-.033541` | `+.008858/+.007241` | `+.001749/+0` | 2/5 | reject strict dynamic gate |

## Novelty và claim hợp lý

Tên working: **Factor-Shared Graph-Proximal Output Repair**.

Claim có thể bảo vệ bằng evidence hiện tại:

- sampling và output selection dùng chung active factor graph, khép semantic gap của C3 cũ;
- terminal correction là sparse membership repair, không sửa giá trị generated row;
- C3 là frozen proximal anchor nên method cũ vẫn chạy độc lập và vẫn mạnh trong ablation;
- graph gain được định nghĩa bằng shared audit function, không dùng internal objective khác để claim;
- selection chỉ dùng train/pool statistics; evaluator/test metrics không đi vào search;
- Shoppers fresh confirmation cho thấy graph consistency tăng trong khi giữ dynamic main-metric gains và local SOTA coverage.

Chưa nên claim:

- comprehensive cross-dataset SOTA;
- Magic thắng dynamic generator trên cả Shape và Trend;
- graph-proximal repair luôn cải thiện C3 trên mọi metric.

## Ablation design đề xuất cho bài

| Arm | Ý nghĩa |
|---|---|
| `G` | Dynamic graph-guided generator, không terminal selection |
| `old` | `G +` frozen generic C3; chứng minh method cũ vẫn work |
| `graph_only` | Chỉ graph statistics; đo utility loss nếu bỏ C3 |
| `gc_*` | Additive scalar graph penalty; baseline scalarization |
| `pa_*` | Explicit graph constraints nhưng full subset redraw |
| `ps*_q*` | Sparse proximal, graph-first |
| `pqs005_g10` | Sparse proximal, quality-first; proposed final |

Ablation quan trọng nhất là `old → pa_g25_q20 → pqs005_g10`: nó tách riêng đóng góp của graph feasibility, redraw distance, và proximal/quality-first ordering.

## Tracking và reproducibility

- Scalar graph screen/confirmation: `1112400`, `1112692`, `1112695`, `1112696`.
- Pareto screen/early summary/fresh confirmation: `1114186`, `1114223`, `1114225`, `1114226`.
- Graph-first proximal screen: `1114201`, summary `1114202` (còn chờ Diabetes).
- Quality-first screen/early summary/fresh confirmation: `1114215`, `1114235`, `1114240`, `1114241`.
- Full dependent summaries: `1114187`, `1114202`, `1114216`.
- Remote ledger: `analysis/llm_feedback_20260922/EXPERIMENT_LEDGER_20261021.md`.
- Mỗi output root chứa `provenance.txt`, exact selected row indices, `selection_metadata.json`, factor-level graph audit và evaluator CSV.

## Trạng thái tiếp theo

Ba full screens vẫn đang hoàn tất phần Diabetes có integrated-coverage scoring. Chúng chỉ nhằm lấy ablation/cross-dataset evidence còn thiếu; không có thêm sweep lớn được mở. Sau khi summaries `1114187/1114202/1114216` xong, bước tiếp theo hợp lý là khóa bảng final và chỉ chạy thêm confirmation nếu Diabetes thật sự pass các gate đã định trước.
