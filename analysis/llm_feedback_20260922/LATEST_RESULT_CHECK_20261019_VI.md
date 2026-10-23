# Kiểm tra kết quả mới nhất

Ngày kiểm tra: 2026-10-19

## 1. Kết luận nhanh

- **Default vẫn là dataset mạnh nhất cho contribution hiện tại.** Progressive controller đã thắng matched final-only repair trên held-out graph metrics ở 3/3 seeds, trong khi giữ Shape/Trend gần như không đổi.
- **Magic có tín hiệu dương nhỏ.** Cấu hình `s100_mid` cải thiện Trend, Beta, AUC và held-out relation error so với paired TabDiff, nhưng biên độ nhỏ.
- **Diabetes chưa ổn định.** Chỉ `s100_mid` cải thiện Trend/AUC nhẹ; ba cấu hình còn lại làm Trend xấu hơn hoặc không thay đổi.
- **Adult là no-op.** Graph chỉ tạo khoảng một projection event và thay đổi bị triệt tiêu sau decode/type handling; cả bốn nhánh cho kết quả giống nhau.
- **LCATS official artifact audit là một negative control quan trọng:** đạt zero measured violation nhưng sửa 100% rows, 28,49% cells và làm TSTR utility xấu đi.

## 2. Method mới trên Magic

### Cấu hình tốt nhất hiện tại: `s100_mid`

| Method | Shape ↓ | Trend ↓ | Alpha ↑ | Beta ↑ | XGB AUC ↑ | Held-out error ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TabDiff | 1.685722 | 1.377122 | 98.984038 | 47.324219 | 0.918078 | 0.029706 |
| DynamicGraph | **1.685722** | **1.375457** | **98.984038** | **47.327474** | **0.921652** | **0.029251** |

Paired deltas:

- Trend error giảm khoảng **0,12%**.
- XGBoost AUC tăng **0,003574**.
- Held-out relation error giảm khoảng **1,53%**.
- Shape và Alpha không thay đổi; Beta tăng `0,003255` điểm.

Mechanism thực sự active: `479` projection events. Đây là tín hiệu dương, nhưng chưa đủ lớn để làm main result độc lập.

### Positive-factor branch

Positive factor truyền message cả trên satisfied edges và tạo `31.376` factor-cell events. Nó tăng Beta `+0,0537`, AUC `+0,00284` và cải thiện Trend khoảng `0,17%`, nhưng held-out relation error xấu đi khoảng **19,1%**. Vì vậy dense propagation hiện chưa được giữ làm main mechanism; nó truyền quá rộng và thiếu benefit gate.

### Final-only repair trên Magic

Final-only repair đưa raw macro violation từ khoảng `0,139` về `0`, nhưng held-out relation error tăng từ khoảng `0,030` lên `0,124–0,158`, đồng thời Shape thường xấu hơn. Đây là bằng chứng trực tiếp rằng **zero satisfaction không phải mục tiêu đúng** khi real data có legitimate violations.

## 3. Method mới trên Diabetes

### Cấu hình duy nhất có tín hiệu dương: `s100_mid`

| Method | Shape ↓ | Trend ↓ | Alpha ↑ | Beta ↑ | XGB AUC ↑ | Held-out error ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TabDiff | 1.871783 | 7.458820 | 91.155599 | 54.471029 | 0.650048 | 0.010477 |
| DynamicGraph | **1.871124** | **7.364649** | **91.157227** | **54.487305** | **0.650354** | **0.010477** |

- Trend error giảm khoảng **1,26%**.
- Shape giảm `0,000659`; Beta tăng `0,0163`; AUC tăng `0,000306`.
- Held-out relation error không đổi.
- Chỉ có `29` projection events trên một edge, nên evidence mechanism còn yếu.

Các cấu hình `s050_mid`, `s100_broad` và `s150_mid` làm Trend xấu hơn khoảng 1,9–8,0% hoặc không tạo thay đổi có ích. Do đó Diabetes hiện là mixed/unstable result, không phải confirmation dataset.

## 4. Adult

Cả bốn cấu hình trả lại đúng kết quả TabDiff trên hầu hết metric. Trace cho thấy chỉ khoảng một projection event và không có thay đổi cuối đáng kể. Graph chỉ có hai numeric–literal edges; evidence hiện tại chưa tách được correction bị triệt tiêu sau decode/type handling hay không vượt stability/persistence gate.

Kết luận: không dùng Adult làm evidence cho method hiện tại. Nếu tiếp tục dataset này, cần graph/action layer phù hợp categorical dependencies; tăng strength không giải quyết được no-op.

## 5. Published generator baselines đã hoàn tất

### Adult

| Published method | Shape ↓ | Trend ↓ | Alpha ↑ | Beta ↑ | XGB AUC ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| TabDiff | **1.5066** | **2.7409** | **97.7962** | **50.1660** | **0.8975** |
| CTGAN-SDV | 13.5544 | 14.8995 | 75.1855 | 30.8415 | 0.8793 |
| TVAE-SDV | 9.9257 | 9.1408 | 94.3734 | 39.3164 | 0.8674 |

TabDiff vượt rõ CTGAN/TVAE về fidelity, coverage và utility. ForestDiffusion đang chạy; TabSyn đang chờ GPU.

### Magic

| Published method | Shape ↓ | Trend ↓ | Alpha ↑ | Beta ↑ | XGB AUC ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| ForestDiffusion official | 3.3541 | 3.6141 | 98.1478 | **54.0137** | 0.9092 |
| CTGAN-SDV | 10.4819 | 9.7758 | 78.8477 | 14.1016 | 0.8310 |
| TVAE-SDV | 9.6239 | 8.7282 | 87.2958 | 34.0674 | 0.8632 |

Paired TabDiff samples của method có Shape khoảng `1,37–1,69`, Trend `1,17–4,79` và AUC `0,918–0,929`; nhìn chung mạnh hơn ba baseline trên, ngoại trừ ForestDiffusion có Beta coverage cao hơn. TabSyn đang chờ; TabDDPM đang chạy.

### Diabetes

TVAE đạt Shape `11,0856`, Trend `24,4204`, AUC `0,6475`; TabDiff đạt fidelity tốt hơn nhiều và AUC tương đương. ForestDiffusion thất bại do GPU OOM; CTGAN còn đang chạy.

## 6. LCATS official artifact audit — Insurance

| Candidate | Any violation ↓ | Numeric KS ↓ | Corr. MAE ↓ | TSTR NRMSE ↓ | TSTR R2 ↑ | Row edit ↓ | Cell edit ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TabSyn original | 0.9269 | 0.0370 | 0.0118 | **0.5254** | **0.7239** | — | — |
| LCATS aligned | **0.0000** | **0.0299** | **0.0115** | 0.5335 | 0.7153 | 1.0000 | 0.2849 |

LCATS cải thiện Numeric KS khoảng **19,2%** và đưa measured violation về zero. Tuy nhiên TSTR R2 giảm `0,0086`, NRMSE tăng khoảng `1,54%`, mọi row đều bị sửa và 28,49% cells thay đổi. Vì LCATS chạy trên Insurance, số này không được ghép trực tiếp với Default/Magic; nó chỉ là published post-repair baseline evidence.

## 7. Lỗi và job còn chạy

- `1098680_0`, TabDDPM Adult: lỗi do environment PyTorch CUDA 13 không chứa kernel cho V100 compute capability 7.0; đây là lỗi scheduling/environment, không phải model failure.
- `1098677_1`, ForestDiffusion Diabetes: GPU OOM; yêu cầu khoảng 3,35 GB trong khi chỉ còn khoảng 1,82 GB.
- `1098641`, Default/Shoppers literal-dynamic sweep: đang chờ A40.
- `1098643`, TabSyn Adult: đang chờ A40.
- `1098678`, TabSyn Magic: đang chờ A40.
- `1098680_1`, TabDDPM Magic: đang chạy.
- `1098706`, official LCATS phase 6 refresh: đang chờ A40. Independent read-only audit `1098709` đã hoàn tất.

Các giá trị Detection bằng đúng `1.0` ở một số Adult/Magic runs có dấu hiệu saturation; không dùng chúng làm claim cho tới khi detector được kiểm tra lại.

## 8. Quyết định hiện tại

1. Giữ **Default** làm main dataset và progressive dynamic activation làm main mechanism.
2. Giữ **Magic `s100_mid`** làm preliminary transfer result; cần replicate cùng config/seed protocol trước khi report chính.
3. Diabetes chỉ dùng như limitation/mixed result cho tới khi có graph nhiều edge hơn.
4. Bỏ Adult khỏi main result của phiên bản actuator hiện tại.
5. Không tối ưu raw satisfaction trên graph tự khám phá. Main selection phải dựa trên Shape, Trend, utility và held-out relation profile.
6. Published baseline table chỉ gồm official/published methods; final repair, Harpoon-style proxy, CDD-style adaptation, frozen và shuffled graph nằm ở ablation.
