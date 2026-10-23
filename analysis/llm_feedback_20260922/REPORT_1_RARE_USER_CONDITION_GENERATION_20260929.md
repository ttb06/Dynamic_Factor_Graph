# Sinh synthetic tabular data cho rare user conditions bằng diffusion

## In short

Các tabular generators hiện nay thường được đánh giá ở bài toán sinh toàn bộ phân phối hoặc một số conditions đã biết. Khi user đưa một condition hiếm tại inference time, cách đơn giản nhất là sinh unconditional rồi rejection sampling. Cách này đúng về mặt nguyên tắc nhưng chi phí tăng gần theo nghịch đảo support của condition.

Ví dụ trên các thí nghiệm đã chạy:

- Default, mixed condition q975 có held-out support `1.26%`: rejection cần trung bình `69.48` trajectories cho một valid row, trong khi condition adapter cần `14.79`, tương đương speedup `4.70× [4.37, 4.98]` qua tám seeds.
- Shoppers, mixed condition q975 có held-out support `0.35%`: rejection cần `165.23` trajectories/valid row, condition adapter cần `22.70`, tương đương speedup `7.28× [6.97, 7.62]`.

=> Hướng nghiên cứu là xây một **reusable condition mechanism** cho diffusion, nhận user condition mới mà không phải fine-tune riêng một model cho mỗi request:

`natural-language request → typed condition program → schema/type check → condition adapter → diffusion sampler → independent verifier → optional residual repair`.

| Thành phần | Vai trò | Trạng thái |
| --- | --- | --- |
| Typed condition program | Biểu diễn threshold, equality và composition | Đã chạy |
| LLM compiler | Chuyển natural language thành executable condition | Đã chạy probe |
| Condition adapter | Condition TabDiff bằng một interface dùng lại | Đã chạy trên Adult, Default, Shoppers |
| Rare-condition curriculum | Học nhiều mức support thay vì một threshold cố định | Đã chạy ablation |
| Rejection, repair, best-of-N | Baselines theo cùng condition | Rejection/repair đã chạy; best-of-N cần chuẩn hóa thêm |
| Official CuTS | Per-program constrained generation baseline | Đã chạy trên Adult |
| Cross-table condition model | Shared parameters qua schema/table mới | Chưa hoàn tất |

Claim hiện có thể bảo vệ:

> Conditioned diffusion giảm đáng kể sampling cost cho rare cohorts so với rejection sampling, nhưng lợi ích validity/cost đi kèm conditional-fidelity trade-off. Dense condition curriculum và backbone adaptation có thể cải thiện frontier trên một số tables, nhưng chưa generalize đồng đều.

Chưa thể claim foundation model hoặc state of the art vì chưa có một shared model chạy trên held-out schemas.

## 1. Bài toán

Cho một bảng thật với phân phối `p_data(x)` và một user condition `C`, mục tiêu là sinh:

`x ~ p_data(x | C)`.

Condition có thể gồm:

- numeric threshold: `LIMIT_BAL ≥ q975`;
- categorical equality: `segment = rare_category`;
- mixed composition: đồng thời một numeric threshold và một categorical predicate;
- condition family hoặc threshold chưa xuất hiện trong adapter training.

Bài toán không chỉ yêu cầu mọi output thỏa `C`. Một method tốt còn phải giữ phân phối của các cột còn lại giống real cohort thỏa `C`, bảo toàn rare modes, downstream utility và chi phí hợp lý.

Ví dụ:

```text
User request:
  Generate customers with a very high credit limit in a specified rare segment.

Typed program:
  LIMIT_BAL >= train_quantile(0.975)
  AND segment == value_k
```

Nếu condition chỉ chiếm `0.5%` phân phối, rejection sampling cần xấp xỉ 200 trajectories cho một accepted row. Một condition generator hữu ích phải tăng direct success mà không kéo tất cả các cột về một prototype hoặc làm mất các subgroups hợp lệ.

## 2. Motivation

### 2.1. Rejection sampling trở nên đắt khi condition hiếm

Nếu unconditional acceptance probability là `p(C)`, expected trajectories cho một valid row là gần `1 / p(C)`. Chi phí tăng rất nhanh ở q950, q975 và q990.

### 2.2. Per-condition fine-tuning đạt validity cao nhưng adaptation cost lớn

Official CuTS đạt constraint score `1.0` trên cả năm Adult programs đã chạy. Tuy nhiên, constrained fine-tuning mất gần hai giờ, trong khi baseline sampling mất vài phút. Năm resamples dùng cùng một fine-tuned model nên mới chỉ đo sampling variance, chưa đo training variance.

=> Một reusable condition interface có thể có giá trị nếu xử lý nhiều user requests bằng cùng parameter set hoặc adaptation rất nhỏ.

### 2.3. Hard projection và repair không giải quyết toàn bộ conditional distribution

Final projection có thể đưa validity lên `1.0`, nhưng nhiều rows phải bị sửa. Ví dụ trong hybrid condition + final projection:

- Default held-out composition, CFG 1: success trước repair `0.4055`; sau projection `1.0`, nhưng edit rate `0.5945`.
- Shoppers held-out composition, CFG 1: success trước repair `0.4600`; sau projection `1.0`, edit rate `0.5400`.

CTDF-style numeric projection proxy cũng đạt satisfaction `1.0` trên Default, nhưng target distortion ở held-out composition là `0.4689` khi project mọi step. Đây là proxy, không phải reproduction đầy đủ của CTDF vì chưa có categorical KL feasibility mapping.

=> Validity bằng `1.0` không tự động chứng minh generator học đúng conditional distribution.

## 3. Related work và khoảng trống

### 3.1. TabDiff và TabSyn

TabDiff và TabSyn là các mixed-type diffusion backbones mạnh. Chúng tập trung vào fidelity của toàn bảng; bản thân backbone không giải quyết bài toán một model nhận nhiều unseen user conditions với rarity thay đổi mạnh.

### 3.2. CuTS

CuTS cung cấp constrained tabular synthesis và là baseline mạnh cho declarative conditions. Kết quả official cho thấy constraint validity có thể đạt `1.0` với utility gần giữ nguyên. Khoảng trống cho hướng này không phải “conditioned generation chưa tồn tại”, mà là:

- per-program adaptation cost;
- generalization sang threshold/composition chưa thấy;
- cost khi phục vụ nhiều requests;
- conditional fidelity và rare-mode coverage.

### 3.3. Harpoon và CTDF

Harpoon đưa differentiable loss vào sampling; CTDF đưa feasibility operator vào mỗi reverse step. Hai work này là baseline trực tiếp cho hard or differentiable constraints. Hướng hiện tại chỉ khác nếu chứng minh reusable condition representation và unseen-condition generalization, không chỉ thêm một guidance loss khác.

Harpoon-style proxy hiện tại xác nhận guidance strength tạo trade-off. Trên Default held-out composition:

- unconditional satisfaction `0.1838`, KS `0.3642`;
- proxy `eta=0.1`: satisfaction `0.3649`, KS `0.3467`;
- proxy `eta=0.3`: satisfaction `0.7103`, KS `0.3189`, nhưng target distortion tăng lên `0.5989`.

Đây là proxy squared-hinge trên predicted-clean numeric lower bounds, không phải faithful Harpoon reproduction.

### 3.4. CTSyn và hướng foundation/cross-table

Điểm tham chiếu quan trọng là condition transfer qua nhiều tables/schemas. Hiện project mới có nhiều table-specific backbones/adapters; chưa có một shared parameter set. Vì vậy “foundation” mới là hướng tiếp theo, không phải kết quả đã đạt.

## 4. Proposed method hiện tại

### 4.1. Compile user request thành typed program

LLM không trực tiếp đánh giá từng generated row. Nó compile request một lần thành representation có kiểu:

```json
{
  "predicates": [
    {"column": "LIMIT_BAL", "operator": ">=", "value": 500000},
    {"column": "segment", "operator": "==", "value": "rare_group"}
  ],
  "logic": "AND"
}
```

Sau đó:

- schema checker kiểm tra column name và data type;
- deterministic executor kiểm tra semantics;
- counterexample/property tests kiểm tra parser;
- không gửi raw real rows cho external API.

Request-conditioned probe cho kết quả:

| Phương pháp | Four-way accuracy | Same-row/different-request | Token |
| --- | ---: | ---: | ---: |
| GPT-4.1-mini direct judgement | 0.864 | 0.867 | khoảng 37.5k |
| GPT-5.4-mini direct judgement | 0.650 | 0.367 | khoảng 37.7k |
| LLM → typed DSL → deterministic executor | 1.000 | 1.000 | khoảng 1.85k |

=> LLM phù hợp làm compiler; deterministic code phù hợp làm routine executor.

### 4.2. Condition adapter

Condition program được mã hóa thành vector có cấu trúc rồi inject vào TabDiff denoiser. Sampling dùng classifier-free guidance:

`prediction = unconditional + g * (conditional - unconditional)`.

Các variants đã thử:

- fixed thresholds hoặc dense train-quantile curriculum;
- condition projection only;
- low-core update;
- full backbone update;
- condition dropout.

### 4.3. Independent verifier và residual repair

Verifier đánh giá exact satisfaction bằng executable condition. Repair chỉ được dùng như bước cuối và phải report:

- edit rate;
- numeric distortion;
- conditional KS/TV sau repair;
- số rows cần repair.

## 5. Experimental setup

### 5.1. Datasets đã dùng cho generator experiments

- Adult;
- Default of Credit Card Clients;
- Online Shoppers Purchasing Intention.

Diabetes backbone và condition representations đã được chuẩn bị nhưng experiment chưa hoàn tất. Magic chủ yếu numeric và đã được hạ ưu tiên. Beijing và Online News hiện chỉ dùng trong constraint audit do regression-target preprocessing chưa được sửa hoàn chỉnh.

### 5.2. Rare-condition protocol

- Thresholds được lấy từ training quantiles, không từ held-out data.
- Held-out real cohort thỏa condition được dùng làm conditional-fidelity reference.
- Unconditional pool: `65,536` samples/seed.
- Conditional generation: `4,096` samples/condition/seed.
- Main rare-support audit: tám independent adapter seeds.
- Numeric conditions gồm q500, q750, q900, q950, q975, q990.
- Mixed conditions gồm q750, q900, q950, q975 kết hợp categorical predicate.

### 5.3. Metrics

- **Direct success:** tỷ lệ trajectories thỏa toàn bộ request trước repair/rejection.
- **Trajectories per valid:** expected sampling cost; thấp hơn tốt hơn.
- **Speedup:** rejection cost chia cho conditioned-generation cost.
- **Numeric KS:** mean Kolmogorov-Smirnov distance giữa accepted synthetic cohort và held-out real cohort; thấp hơn tốt hơn.
- **Categorical TV:** mean total-variation distance trên categorical columns; thấp hơn tốt hơn.
- **Edit rate/distortion:** mức can thiệp của repair.
- **BAC/downstream utility:** balanced accuracy của downstream model khi baseline hỗ trợ.

## 6. Main results

### 6.1. Rare mixed conditions: conditioned generation vs rejection

| Dataset | Rarity | Held-out support | Rejection success | Condition success | Rejection trajectories/valid | Condition trajectories/valid | Speedup [95%] |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Default | q750 | 9.91% | 0.1207 | 0.2535 | 8.48 | 4.11 | 2.06× [2.01, 2.12] |
| Default | q900 | 4.44% | 0.0554 | 0.1691 | 18.31 | 6.34 | 2.89× [2.78, 3.00] |
| Default | q950 | 2.39% | 0.0278 | 0.1217 | 36.44 | 9.73 | 3.74× [3.57, 3.95] |
| Default | q975 | 1.26% | 0.0145 | 0.0934 | 69.48 | 14.79 | 4.70× [4.37, 4.98] |
| Shoppers | q750 | 8.19% | 0.1016 | 0.2516 | 11.31 | 4.65 | 2.43× [2.39, 2.49] |
| Shoppers | q900 | 1.62% | 0.0273 | 0.1328 | 37.29 | 8.89 | 4.19× [4.09, 4.30] |
| Shoppers | q950 | 0.87% | 0.0132 | 0.0954 | 78.02 | 14.13 | 5.52× [5.33, 5.74] |
| Shoppers | q975 | 0.35% | 0.0064 | 0.0770 | 165.23 | 22.70 | 7.28× [6.97, 7.62] |

=> Lợi ích về sampling cost tăng cùng rarity. Đây là bằng chứng mạnh nhất của hướng 1.

### 6.2. Fidelity trade-off ở q975

Qua tám seeds, condition adapter so với rejection:

| Dataset | Success gain [95%] | KS change [95%] | TV change [95%] |
| --- | ---: | ---: | ---: |
| Default mixed q975 | +0.079 [+0.069, +0.086] | +0.011 [+0.005, +0.018] | +0.029 [+0.024, +0.034] |
| Shoppers mixed q975 | +0.071 [+0.066, +0.076] | +0.003 [+0.000, +0.007] | -0.015 [-0.017, -0.012] |

Negative KS/TV favors conditioned generation. Default cho thấy validity/cost cải thiện nhưng fidelity xấu hơn. Shoppers gần giữ numeric fidelity và cải thiện categorical TV.

=> Không có một kết luận “condition adapter luôn cải thiện cả validity và fidelity”. Dataset heterogeneity là vấn đề thật.

### 6.3. Adapter/curriculum ablation

So với fixed-threshold projection-only:

| Dataset | Setting | Fixed success | Dense-full success | Fixed accepted KS/TV | Dense-full accepted KS/TV | Kết luận |
| --- | --- | ---: | ---: | --- | --- | --- |
| Adult support-mixed | guidance 1 | 0.0306 | 0.0739 | 0.1267 / 0.1124 | 0.1023 / 0.0846 | Cải thiện cả success và fidelity |
| Shoppers support-mixed | guidance 1 | 0.0908 | 0.1579 | 0.2369 / 0.2100 | 0.2087 / 0.2000 | Cải thiện cả success và fidelity |
| Default support-mixed | guidance 2 | 0.1850 | 0.3831 | 0.1818 / 0.1098 | 0.2520 / 0.1434 | Success tăng nhưng oversteer |

Paired seed-cluster bootstrap:

- Adult guidance 1: success `+0.0433 [0.0405, 0.0458]`, KS `-0.0244`, TV `-0.0278`.
- Shoppers guidance 1: success `+0.0671 [0.0642, 0.0717]`, KS `-0.0282`, TV `-0.0100`.
- Default guidance 2: success khoảng `+0.1981`, nhưng KS `+0.0702` và TV `+0.0336`.

=> Dense curriculum/full update không phải universal solution. Nó có thể sửa underfit nhưng cũng có thể làm condition oversteering.

### 6.4. Official CuTS baseline

| Program | Base constraint score | CuTS constraint score | Base BAC | CuTS BAC |
| --- | ---: | ---: | ---: | ---: |
| implication2 | 0.9312 | 1.0000 | 0.7711 | 0.7735 |
| implication4 | 0.9850 | 1.0000 | 0.7738 | 0.7762 |
| implication5 | 0.5901 | 1.0000 | 0.7717 | 0.7747 |
| line_constraint1 | 0.3254 | 1.0000 | 0.7716 | 0.7692 |
| line_constraint2 | 0.4061 | 1.0000 | 0.7732 | 0.7718 |

CuTS là baseline rất mạnh về validity. Hướng 1 chỉ có thể thắng bằng một trong các trục:

- không cần per-program fine-tuning;
- phục vụ nhiều unseen requests bằng cùng model;
- giảm total adaptation + sampling cost;
- giữ conditional fidelity/coverage tốt hơn.

## 7. Những hypothesis đã bị bác bỏ hoặc thu hẹp

### 7.1. “Projection-only là đủ” — bị bác bỏ

Adult và Shoppers cho thấy dense/full variants có thể đồng thời tăng direct success và giảm KS/TV so với projection-only. Tuy nhiên kết quả Default cho thấy full adaptation cũng có thể oversteer.

### 7.2. “Guidance càng mạnh càng tốt” — bị bác bỏ

Guidance 4 thường tăng satisfaction nhưng làm conditional KS/TV xấu rõ. Cần đánh giá frontier thay vì chọn một guidance scale theo validity.

### 7.3. “Foundation embedding tự động tốt hơn typed structure” — chưa được ủng hộ

Adult held-out compositions, guidance 2:

- structured condition satisfaction khoảng `0.608`;
- text embedding khoảng `0.534`;
- hybrid khoảng `0.555`.

Foundation encoder chỉ có ý nghĩa nếu giúp held-out schema/condition family hoặc giảm adaptation cost.

## 8. Khó khăn hiện tại

### 8.1. Benchmark chưa đủ rộng

End-to-end generator results hiện tập trung vào Adult, Default và Shoppers. Seven-dataset audit không đồng nghĩa seven-dataset generator benchmark.

### 8.2. Rare support không tự động tạo semantic difficulty

Một condition hiếm có thể chỉ là tail threshold. Reviewer có thể xem đây là conditional generation thông thường nếu không có unseen composition, schema transfer hoặc adaptation-cost contribution.

### 8.3. Fidelity reference có sample size nhỏ

Ở condition rất hiếm, held-out real cohort nhỏ làm KS/TV có variance lớn. Cần report cohort size, confidence intervals và có thể dùng repeated splits.

### 8.4. Reusable chưa đồng nghĩa cross-table

Hiện adapter được train riêng cho từng table. Nếu full backbone phải cập nhật nhiều, parameter-efficient/foundation claim yếu đi.

### 8.5. Baseline accounting

Phải tách rõ:

- backbone pretraining cost;
- per-table adaptation;
- per-condition adaptation;
- sampling cost;
- LLM compilation cost;
- verifier/repair cost.

## 9. Hướng tiếp theo

### 9.1. Chốt bài toán chính

Định vị tốt nhất hiện tại:

> Amortized rare conditional generation under a validity–fidelity–cost objective.

Không nên định vị chỉ là “inject user condition vào TabDiff”.

### 9.2. Mở rộng benchmark

- Hoàn tất Diabetes;
- thêm 2–4 datasets từ CuTS/CTSyn/Harpoon nếu preprocessing tương thích;
- dùng condition families có provenance từ dataset documentation;
- leave-one-threshold, leave-one-composition và leave-one-condition-family out;
- sau đó mới làm leave-one-table out.

### 9.3. Shared condition mechanism

Thử ba mức:

1. shared typed condition encoder, table-specific diffusion backbones;
2. shared hypernetwork/adapter sinh condition projection theo schema;
3. shared diffusion backbone với table-specific tokenizer/normalization.

Mốc quyết định: cùng một parameter set phải generalize sang condition family hoặc schema chưa thấy.

### 9.4. Baselines cần hoàn thiện

- rejection sampling và best-of-N ở matched denoiser budget;
- official CuTS trên nhiều programs/datasets;
- official hoặc gần-official Harpoon/CTDF khi code cho phép;
- final projection/post-repair;
- unconditional + verifier;
- per-condition fine-tuning upper bound.

### 9.5. Metrics còn thiếu

- conditional downstream utility;
- rare-mode recall/coverage;
- diversity và duplicate rate;
- privacy/membership risk;
- calibration theo rarity;
- wall-clock, GPU hours và total amortized serving cost.

## 10. Đánh giá khả năng thành paper độc lập

Hướng này có thể thành một paper độc lập nếu contribution được chốt ở ba điểm:

1. benchmark/protocol cho unseen rare conditions với matched cost;
2. reusable condition mechanism giảm rejection/per-condition adaptation cost;
3. cross-table hoặc ít nhất cross-condition-family generalization.

Evidence hiện tại đủ để tiếp tục project nhưng chưa đủ cho claim state of the art. Phần mạnh nhất là eight-seed sample-efficiency result; phần yếu nhất là benchmark breadth và shared-model generalization.

Nếu không đạt cross-table transfer, work vẫn có thể thành empirical/method paper hẹp hơn về rare conditional generation, nhưng sẽ khó định vị như foundation model.

## 11. Result files

- `outputs/rejection_frontier_aggregate_1045512/RESULTS.md`
- `outputs/rare_support_statistical_audit_1045512_1045513/RESULTS.md`
- `outputs/adapter_variant_frontier_aggregate_1048326/RESULTS.md`
- `outputs/adapter_variant_statistical_audit_1048595/RESULTS.md`
- `outputs/cuts_official_resample_audit_1045748/RESULTS.md`
- `outputs/hybrid_projection_aggregate_1044393/RESULTS.md`
- `outputs/ctdf_style_aggregate_1044372/RESULTS.md`
- `outputs/harpoon_style_aggregate_1044475/RESULTS.md`
