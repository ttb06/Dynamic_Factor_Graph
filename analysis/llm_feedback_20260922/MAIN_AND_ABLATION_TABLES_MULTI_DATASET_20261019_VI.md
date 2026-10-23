# Tổng hợp các bảng main và ablation trên nhiều dataset

Ngày tổng hợp: 2026-10-19

## 0. Quy tắc sử dụng

- **Published baseline:** method đã công bố, chạy bằng official code hoặc implementation được ghi rõ.
- **Adapted published operator:** thành phần từ paper được đặt vào TabDiff; không gọi là full official reproduction.
- **Ablation/control:** final-only repair, frozen graph, shuffled graph, Harpoon-style proxy, CDD-style projection và các lịch guidance nội bộ.
- News báo Shape/Trend dưới dạng **score** nên cao hơn tốt hơn. Các dataset còn lại báo Shape/Trend dưới dạng **error** nên thấp hơn tốt hơn.
- Chỉ so trực tiếp các hàng dùng cùng split, evaluator và paired sample. Số paper-reported không trộn vào local ranking.
- Detection bị loại khỏi bảng chính vì một số run bị saturation hoặc detector chưa hội tụ ổn định.

# I. Các bảng main

## Main Table 1 — Public hard constraints trên News

Năm inequality lấy từ public constraint file của ConstrainedDGM. Tất cả các hàng dùng cùng TabDiff sample và evaluator. `ConstrainedDGM layer` dùng feasibility operator đã công bố nhưng đặt lên TabDiff, nên là adapted operator chứ không phải full ConstrainedDGM reproduction.

| Method | Status | Shape ↑ | Trend ↑ | Alpha ↑ | Beta ↑ | R² ↑ | NRMSE ↓ | Violation ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TabDiff | Published backbone | **96.6554** | 89.1527 | 97.1327 | **42.8825** | -0.0254 | 1.0126 | 25.8789% |
| ConstrainedDGM feasibility layer | Adapted published operator | 96.2996 | 90.5950 | **97.1344** | 42.8988 | -0.0293 | 1.0146 | **0.0000%** |
| **Dynamic middle + certifier** | Proposed | 96.5864 | **90.6927** | 97.1024 | 42.8711 | **-0.0054** | **1.0027** | **0.0000%** |

Điểm chính:

- Proposed đạt cùng zero violation như feasibility layer.
- So với ConstrainedDGM layer, Trend tăng `0,0977`, R² tăng `0,0239`, NRMSE giảm `0,0119` và Shape tăng `0,2868`.
- So với TabDiff, dependency Trend và utility tốt hơn, nhưng Shape/Alpha/Beta không đồng loạt thắng.
- Đây là single-seed main table; chưa đủ cho significance claim.

## Main Table 2 — Paired TabDiff comparison trên mixed-type datasets

Shape/Trend ở bảng này là error. Held-out relation error chỉ là metric bổ trợ; selection chính vẫn dựa trên Shape, Trend và utility.

| Dataset | Seeds | Method | Shape ↓ | Trend ↓ | Alpha ↑ | Beta ↑ | Utility AUC ↑ | Held-out relation error ↓ |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Default | 3 | TabDiff | 1.9833 | 2.2165 | 98.4736 | **52.5109** | **0.7230** | 0.0309 |
| Default | 3 | **Dynamic graph** | **1.9591** | **2.2116** | **98.5218** | 52.4436 | 0.7228 | **0.0120** |
| Shoppers | 1 standard-eval seed | TabDiff | **1.3769** | **2.2975** | **97.4096** | 48.9681 | **0.9153** | 0.01383 |
| Shoppers | 1 standard-eval seed | Progressive controller | **1.3769** | **2.2975** | **97.4096** | 48.9681 | **0.9153** | 0.01383 |
| Magic | 1 | TabDiff | **1.6857** | 1.3771 | **98.9840** | 47.3242 | 0.9181 | 0.02971 |
| Magic | 1 | **Dynamic graph `s100_mid`** | **1.6857** | **1.3755** | **98.9840** | **47.3275** | **0.9217** | **0.02925** |
| Diabetes | 1 | TabDiff | 1.8718 | 7.4588 | 91.1556 | 54.4710 | 0.6500 | **0.01048** |
| Diabetes | 1 | **Dynamic graph `s100_mid`** | **1.8711** | **7.3646** | **91.1572** | **54.4873** | **0.6504** | **0.01048** |
| Adult | 1 | TabDiff | **1.1315** | **2.3819** | **98.2950** | **48.7907** | **0.8969** | **0.03374** |
| Adult | 1 | Dynamic graph | **1.1315** | **2.3819** | **98.2950** | **48.7907** | **0.8969** | **0.03374** |

Diễn giải:

- **Default:** semantic-profile gain mạnh, common metrics gần như giữ nguyên.
- **Shoppers:** controller chủ động abstain vì train-only precursor không cấp online authority; output bằng TabDiff.
- **Magic:** gain nhỏ nhưng cùng chiều trên Trend, Beta, AUC và relation error.
- **Diabetes:** một operating point cải thiện Trend khoảng `1,26%`; các setting khác không ổn định.
- **Adult:** no-op; graph/action layer hiện tại không tạo intervention hữu hiệu.

Default trong bảng dùng comparison graph 8-edge. Progressive 7-edge mới nhất được báo ở ablation vì chưa rerun cùng đầy đủ published baselines.

## Main Table 3 — Published generator baselines chạy local

Tất cả dùng cùng repository split, 4.096 synthetic rows và metric implementation trong từng dataset. Đây là one-run context, không phải paired significance test.

| Dataset | Published model | Shape ↓ | Trend ↓ | Alpha ↑ | Beta ↑ | Utility AUC ↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Default | TabDiff | 2.3851 | 2.5518 | 96.2240 | **53.3024** | 0.7175 |
| Default | **TabSyn** | **1.4810** | **1.6499** | **99.2198** | 47.7816 | 0.7127 |
| Default | ForestDiffusion | 8.9813 | 6.1003 | 94.4882 | 47.8939 | 0.6968 |
| Default | CTGAN | 15.2233 | 17.8281 | 64.4922 | 21.3704 | 0.6695 |
| Default | TVAE | 10.8799 | 6.5209 | 82.3844 | 25.9066 | **0.7244** |
| Shoppers | TabDiff | **1.6044** | 2.5302 | 96.3738 | **50.3320** | 0.8978 |
| Shoppers | **TabSyn** | 1.6652 | **1.9929** | **99.0039** | 46.3363 | 0.8817 |
| Shoppers | ForestDiffusion | 14.5687 | 4.6032 | 91.0807 | 49.7168 | **0.8992** |
| Shoppers | CTGAN | 14.2642 | 15.4354 | 83.8363 | 31.6911 | 0.8013 |
| Shoppers | TVAE | 16.6945 | 9.4195 | 53.8737 | 18.1136 | 0.8773 |
| Adult | TabDiff | **1.5066** | **2.7409** | **97.7962** | **50.1660** | **0.8975** |
| Adult | CTGAN | 13.5544 | 14.8995 | 75.1855 | 30.8415 | 0.8793 |
| Adult | TVAE | 9.9257 | 9.1408 | 94.3734 | 39.3164 | 0.8674 |
| Magic | TabDiff, paired sample | **1.6857** | **1.3771** | **98.9840** | 47.3242 | **0.9181** |
| Magic | ForestDiffusion | 3.3541 | 3.6141 | 98.1478 | **54.0137** | 0.9092 |
| Magic | CTGAN | 10.4819 | 9.7758 | 78.8477 | 14.1016 | 0.8310 |
| Magic | TVAE | 9.6239 | 8.7282 | 87.2958 | 34.0674 | 0.8632 |
| Diabetes | TabDiff, paired sample | **1.8718** | **7.4588** | **91.1556** | **54.4710** | 0.6500 |
| Diabetes | TVAE | 11.0856 | 24.4204 | 44.4580 | 49.3490 | **0.6475** |

Main-paper recommendation:

- Giữ TabDiff, TabSyn, ForestDiffusion và TabDDPM khi run hoàn tất.
- CTGAN/TVAE là published methods nhưng nên chuyển xuống appendix vì quá yếu/cũ.
- Không ghép các paper-reported SCTab-Diff numbers vào bảng này.

## Main Table 4 — Official LCATS audit trên Insurance

LCATS là published post-generation method. Nó chạy trên Insurance, không so trực tiếp với các dòng Default/News.

| Candidate | Any violation ↓ | Numeric KS ↓ | Cat. TV ↓ | Corr. MAE ↓ | TSTR NRMSE ↓ | TSTR R² ↑ | Row edit ↓ | Cell edit ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TabSyn original | 0.9269 | 0.0370 | **0.0066** | 0.0118 | **0.5254** | **0.7239** | — | — |
| LCATS aligned | **0.0000** | **0.0299** | **0.0066** | **0.0115** | 0.5335 | 0.7153 | 1.0000 | 0.2849 |

LCATS đạt zero measured violations và cải thiện Numeric KS, nhưng giảm downstream utility, sửa 100% rows và 28,49% cells. Đây là published evidence quan trọng cho trade-off của endpoint repair.

# II. Các bảng ablation

## Ablation Table 1 — Progressive timing trên Default

Ba unseen seeds `20261447–20261449`, graph/calibration được đóng băng trước evaluation.

| Metric | Strong final-only control | Progressive trajectory | Relative change | Paired wins |
| --- | ---: | ---: | ---: | ---: |
| Actionable held-out error ↓ | 0.006323 | **0.005097** | **19,38% tốt hơn** | **3/3** |
| Full held-out error ↓ | 0.011192 | **0.010293** | **8,03% tốt hơn** | **3/3** |
| Shape error ↓ | 2.041838 | **2.038613** | 0,16% tốt hơn | 1/3 |
| Trend error ↓ | 1.971839 | **1.961033** | 0,55% tốt hơn | 2/3 |
| XGBoost AUC ↑ | **0.731765** | 0.730041 | 0,24% kém hơn | 1/3 |

Đây là ablation trực tiếp nhất cho claim trajectory: cùng graph, actuator và calibration; chỉ khác can thiệp ở endpoint hay trong reverse process.

## Ablation Table 2 — Topology và evidence gate trên Default

Một seed cố định, cùng checkpoint, graph và evaluator.

| Controller | Actionable error ↓ | Full error ↓ | Online actions | Residual actions |
| --- | ---: | ---: | ---: | ---: |
| Strong final-only | 0.007035 | 0.009833 | 0 | 55 |
| Fixed-streak dynamic | 0.005898 | 0.009133 | 94 | 0 |
| Event-triggered + endpoint repair | 0.006705 | 0.009629 | 30 | 46 |
| **Progressive dynamic** | **0.005652** | **0.009027** | 76 | 0 |
| Shuffled row topology + repair | 0.007035 | 0.009833 | 60 | 53 |

Shuffled topology thực hiện 60 online actions nhưng không hơn final-only. Vì vậy gain phụ thuộc row-edge assignment và timing, không chỉ action count.

## Ablation Table 3 — Action/timing variants trên News

| Variant | Trend ↑ | R² ↑ | NRMSE ↓ | Online violation ↓ | Online actions ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Fixed middle, balanced L2** | **90.6927** | **-0.0054** | **1.0027** | 2.16% | 2,389 |
| Readability-routed switch | 90.6385 | -0.0141 | 1.0070 | 1.62% | 2,282 |
| Per-edge reliability routing | 90.5499 | -0.0209 | 1.0104 | **1.44%** | **1,834** |
| Primal–dual conservative | 90.4077 | -0.0162 | 1.0081 | 1.99% | 3,274 |
| Primal–dual responsive | 90.4152 | -0.0178 | 1.0088 | 1.78% | 3,793 |

Instantaneous middle gate dùng 2.696 actions và còn 2,66% online violation. Trajectory gate giảm action khoảng 11,4% và residual violation khoảng 18,8%, nhưng fixed balanced-L2 vẫn có standard metric tốt nhất trong sweep này.

## Ablation Table 4 — Final-only versus trajectory trên News

| Method | Shape ↑ | Trend ↑ | R² ↑ | NRMSE ↓ | Violation ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| TabDiff | **96.6554** | 89.1527 | -0.0254 | 1.0126 | 25.8789% |
| Strong final-only repair | 96.5951 | 90.2825 | -0.0162 | 1.0081 | **0.0000%** |
| ConstrainedDGM final layer | 96.2996 | 90.5950 | -0.0293 | 1.0146 | **0.0000%** |
| **Dynamic middle + certifier** | 96.5864 | **90.6927** | **-0.0054** | **1.0027** | **0.0000%** |

Final-only rows trong bảng này là timing/action ablations; chỉ ConstrainedDGM operator có nguồn từ published method.

## Ablation Table 5 — Strength và propagation trên Magic

Các số là paired delta `DynamicGraph − TabDiff`; với Shape/Trend/error, số âm là tốt hơn. AUC/Beta dương là tốt hơn.

| Variant | Projection/factor events | ΔShape ↓ | ΔTrend ↓ | ΔBeta ↑ | ΔAUC ↑ | ΔHeld-out error ↓ | Kết luận |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `s050_mid` | — | 0.0000 | +0.0034 | 0.0000 | +0.0005 | 0.0000 | Gần no-op; Trend xấu nhẹ |
| **`s100_mid`** | 479 projection | 0.0000 | **-0.0017** | +0.0033 | **+0.0036** | **-0.00046** | Operating point cân bằng nhất |
| `s100_broad` | — | +0.0022 | -0.0035 | -0.0488 | -0.0016 | +0.00105 | Trade-off không tốt |
| Positive copula factor | 31.376 factor-cell + 373 projection | +0.0000 | -0.0020 | **+0.0537** | +0.0028 | +0.00525 | Propagation quá rộng; semantic error xấu |

Positive messages thực sự active, nhưng số update rất lớn và gain không tương xứng. Branch này chưa được giữ làm main.

## Ablation Table 6 — Guidance strength trên Diabetes

| Variant | Projection events | ΔShape ↓ | ΔTrend ↓ | ΔBeta ↑ | ΔAUC ↑ | ΔHeld-out error ↓ | Kết luận |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `s050_mid` | — | 0.0000 | +0.6546 | 0.0000 | 0.0000 | 0.0000 | Xấu rõ về Trend |
| **`s100_mid`** | 29 | **-0.0007** | **-0.0942** | **+0.0163** | **+0.0003** | 0.0000 | Chỉ setting có tín hiệu dương |
| `s100_broad` | — | 0.0000 | +0.2533 | 0.0000 | 0.0000 | 0.0000 | Broad interval không hiệu quả |
| `s150_mid` | — | 0.0000 | +0.5923 | 0.0000 | 0.0000 | 0.0000 | Strength cao làm xấu Trend |

Evidence trên Diabetes chưa đủ cho generalization claim: graph chỉ có một actionable edge và kết quả nhạy với strength/window.

## Ablation Table 7 — Có cần LLM để tạo graph không? Default

Graph no-LLM được mine/verify hoàn toàn trên train, không API. Nó trùng 7/8 executable programs với headline graph.

| Method | Shape error ↓ | Trend error ↓ | Alpha ↑ | AUC ↑ | Data-graph error ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| TabDiff | 1.9833 | **2.2165** | 98.4736 | 0.7230 | 0.03923 |
| Data graph, fixed 3-pass | 2.0700 | 2.3941 | **98.6345** | **0.7271** | **0.02331** |
| **Data graph, dual middle** | **1.9821** | 2.2324 | 98.6008 | 0.7247 | 0.02958 |
| Data-graph final-only repair | 1.9521 | 2.2021 | 98.5017 | 0.7247 | 0.03471 |

Kết luận: LLM chưa phải nguồn của headline gain trên Default. Main graph có thể được tạo gần như hoàn toàn bằng train-only statistical compiler. LLM nên là optional proposal/semantic filter cho external relations.

## Ablation Table 8 — LLM contribution trên News

### Discovery coverage

| Graph source | Constraint tìm lại | Recall trên independent catalog |
| --- | ---: | ---: |
| Typed statistical compiler | 9/14 | 64,3% |
| LLM semantic additions | +2 | +14,3 điểm % |
| Verified combined pipeline | 11/14 | 78,6% |

### Soft-factor injection, delta so với cùng route không có factor

| LLM-filtered factor strength | ΔShape ↑ | ΔTrend ↑ | ΔAlpha ↑ | ΔBeta ↑ | ΔR² ↑ | ΔNRMSE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.02 | -0.024 | +0.120 | +0.125 | +0.101 | -0.0090 | +0.0044 |
| 0.05 | -0.040 | **+0.230** | +0.169 | -0.052 | -0.0057 | +0.0028 |
| 0.10 | -0.090 | +0.111 | **+0.206** | +0.039 | **+0.0082** | **-0.0040** |

LLM tăng discovery recall nhưng direct soft messages chưa tạo gain ổn định. Các factor active trên 45–53% row-factor opportunities, tức vấn đề là activation quá rộng chứ không phải signal không được dùng.

## Ablation Table 9 — Shoppers abstention

| Variant | Shape error ↓ | Trend error ↓ | AUC ↑ | Held-out error ↓ | Online actions |
| --- | ---: | ---: | ---: | ---: | ---: |
| TabDiff | **1.3769** | **2.2975** | **0.9153** | 0.01383 | 0 |
| Progressive online | **1.3769** | **2.2975** | **0.9153** | 0.01383 | 0 |
| Online + final-only repair | 1.3837 | 2.3757 | 0.9145 | **0.00947** | 92 residual |

Train-only precursor total bằng zero nên controller abstain. Đây là safety behavior hợp lệ: không tạo gain, nhưng tránh can thiệp trajectory khi calibration không ủng hộ.

# III. Published protocols chưa thể ghép vào main ranking

## Official Harpoon runs

Các số dùng official Harpoon evaluator và inference-time request, không cùng graph/metric với proposed method.

| Dataset | Request | Violation field ↓ | Alpha ↑ | KS ↑ | Detection ↑ | XGB utility ↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Adult | range | 28.0740 | 0.9220 | 0.9519 | 0.8684 | 0.9498 |
| Default | range | 4.8079 | 0.9003 | 0.9417 | 0.7988 | 0.7809 |
| Default | category | 17.6471 | 0.9478 | 0.9363 | 0.9284 | 0.7697 |
| Shoppers | category | **0.0000** | 0.5999 | 0.8974 | 0.8484 | 0.8741 |

Không dùng bảng này để nói proposed vượt Harpoon; nó chỉ chứng minh official pipeline đã được chạy.

## Baseline đang thiếu hoặc chưa hợp lệ

| Method/dataset | Tình trạng | Cách xử lý |
| --- | --- | --- |
| TabSyn Adult | Đang chờ A40 | Bổ sung vào published generator table khi xong |
| TabSyn Magic | Đang chờ A40 | Bổ sung khi xong |
| TabDDPM Magic | Đang chạy | Bổ sung khi evaluator hoàn tất |
| TabDDPM Adult | Lỗi CUDA kernel trên V100 | Rerun trên A40/L4 tương thích |
| ForestDiffusion Adult | Đang chạy | Chờ output |
| ForestDiffusion Diabetes | GPU OOM | Giảm K/batch hoặc chạy GPU RAM lớn hơn |
| SCTab-Diff | Chỉ có paper-reported context | Không ghép vào local ranking |

# IV. Bảng nào nên đưa vào paper

## Main paper

1. **Main Table 1:** News public hard constraints — TabDiff, published constraint operator và proposed.
2. **Main Table 2:** paired TabDiff versus proposed trên Default, Magic, Diabetes, Shoppers; Adult có thể để appendix vì no-op.
3. **Main Table 3 rút gọn:** TabDiff, TabSyn, ForestDiffusion, TabDDPM và proposed trên các dataset có đủ local results.
4. **Ablation Tables 1–4:** timing, topology và action geometry.
5. **Ablation Tables 7–8:** graph source và contribution thực của LLM.

## Appendix

- CTGAN/TVAE;
- full strength/window sweep;
- Harpoon official-protocol table;
- LCATS Insurance audit;
- Adult no-op và Shoppers abstention;
- per-seed results, action counts, edit cost và runtime.

# V. Claim hiện được hỗ trợ

> Với constraint graph đã được kiểm chứng, row- và timestep-dependent activation trong reverse diffusion có thể đạt external-constraint feasibility hoặc conditional relation fidelity tốt hơn endpoint-only controls tại một số operating points, trong khi phần lớn common fidelity được giữ. Lợi ích hiện mạnh nhất trên News và Default, nhỏ trên Magic, không ổn định trên Diabetes, và controller abstain/no-op trên Shoppers/Adult.

Chưa được hỗ trợ:

- SOTA chung về synthetic tabular fidelity;
- thắng official Harpoon hoặc SCTab-Diff trên cùng protocol;
- LLM là nguồn chính của gain;
- trajectory guidance luôn tốt hơn endpoint repair trên mọi dataset.
