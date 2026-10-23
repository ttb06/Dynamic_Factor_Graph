# LCATS published baseline — kết quả local ngày 16/10/2026

## Phạm vi reproduction

- Code: repository LCATS hiện có trên server.
- Dataset: Insurance, đúng dataset demo do repository cung cấp.
- Generator: TabSyn của LCATS.
- LLM phases 1, 4 và 6 được chuyển sang UnoRouter qua key trong project `.env`.
- Model LLM: `gpt-5.6-luna`.
- Phase 1–5 đã chạy; independent evaluator đã hoàn thành.
- Job `1085217` đang retry phase 6 để lấy thêm metric theo evaluator chính thức của LCATS.

## Pipeline đã hoàn thành

1. Phase 1 sinh và kiểm chứng 14 constraints.
2. Phase 2 train TabSyn; early stopping ở epoch 2,200, khoảng 486 giây.
3. Sinh 1,204 records, khoảng 1.87 giây.
4. Phase 3 tạo violation buckets.
5. Phase 4 gọi UnoRouter cho ba buckets và sinh ba weight profiles mỗi bucket:
   `conservative`, `uncertainty_flexible`, `balanced`.
6. Phase 5 giải MILP bằng HiGHS; cả ba candidates repair thành công 1,116 records được đánh dấu vi phạm.
7. Independent audit đánh giá validity, marginal/joint fidelity, propensity, TSTR utility và repair cost.

## Constraint graph thực tế

Constraint package có 14 rules nhưng đều là unary constraints:

- domain bounds của age, BMI, children và charges;
- integer constraints của age và children;
- categorical domains của sex, smoker và region.

Không có inter-column dependency, subgroup condition hoặc semantic context relation. Trước repair, các violation gần như hoàn toàn nằm ở hai integer constraints:

- age integer: 1,114 records;
- children integer: 84 records;
- tổng cộng 1,116/1,204 records có ít nhất một violation.

## Independent result

Lower is better, ngoại trừ TSTR R2. Propensity AUC lý tưởng ở 0.5.

| Candidate | Any violation ↓ | Numeric KS ↓ | Categorical TV ↓ | Corr. MAE ↓ | Propensity AUC →0.5 | TSTR NRMSE ↓ | TSTR R2 ↑ | Row edit ↓ | Cell edit ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TabSyn original | 0.9269 | 0.0370 | 0.0066 | 0.0118 | 0.5104 | 0.5254 | **0.7239** | — | — |
| LCATS trial 0 | **0.0000** | **0.0299** | 0.0066 | **0.0115** | 0.5106 | 0.5335 | 0.7153 | 1.0000 | 0.2849 |
| LCATS trial 1 | **0.0000** | **0.0299** | 0.0066 | **0.0115** | 0.5106 | 0.5335 | 0.7153 | 1.0000 | 0.2849 |
| LCATS trial 2 | **0.0000** | **0.0299** | 0.0066 | **0.0115** | 0.5106 | 0.5335 | 0.7153 | 1.0000 | 0.2849 |

## Diễn giải

So với TabSyn original, LCATS:

- giảm violation từ 92.69% xuống 0%;
- giảm mean numeric KS khoảng 19.1%;
- giảm max numeric KS khoảng 24.1%;
- giảm correlation MAE khoảng 2.9%;
- categorical TV gần như không đổi;
- TSTR R2 giảm 0.0086 và NRMSE tăng 0.0081;
- sửa 100% records và 28.49% cells.

Ba LLM weight profiles cho kết quả cuối giống hệt nhau. Với constraint set chỉ gồm unary bounds/integer rules, nghiệm feasibility gần như bị xác định bởi projection; LLM-generated weights không tạo ra khác biệt quan sát được. Đây là một negative result quan trọng: LLM reasoning layer chỉ có ý nghĩa khi graph có nhiều action routes hoặc inter-column alternatives.

## Ý nghĩa cho dynamic graph project

LCATS là baseline post-generation mạnh về hard validity, nhưng kết quả này đồng thời làm rõ khoảng trống:

1. discovery mặc định chỉ tìm được graph rất đơn giản trên Insurance;
2. repair đạt zero violation bằng cách sửa rất rộng;
3. utility giảm nhẹ;
4. ba action-weight profiles không tạo ra trade-off khác nhau;
5. không sử dụng state của reverse trajectory để quyết định edge/action.

So sánh phù hợp cho bài hiện tại không chỉ là final violation. Cần báo đồng thời validity, fidelity, utility và intervention cost; đồng thời kiểm tra dynamic middle guidance có đạt cùng validity với ít row/cell edits hơn LCATS hay không.

## Artefacts

- Independent result: `outputs/lcats_published_final_audit_1085172/RESULTS.md`.
- Full metrics: `outputs/lcats_published_final_audit_1085172/candidate_metrics.csv`.
- Candidate details: `outputs/lcats_published_final_audit_1085172/details.json`.
- LCATS aligned candidates: `baseline_repos/lcats/synthetic/insurance/aligned_data_trial_{0,1,2}.csv`.
- Phase 4 weights: `baseline_repos/lcats/data/insurance/bucket_weights.json`.
