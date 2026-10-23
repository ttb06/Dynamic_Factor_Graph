# Baseline LLM đã công bố cho dynamic graph

Ngày cập nhật: 2026-10-16.

## Phạm vi

Chỉ xem một phương pháp là **baseline LLM** khi LLM là một thành phần thuật toán có ý nghĩa: generator, graph inducer, graph-guided model, hoặc agent khám phá/hiệu chỉnh constraint. Một prompt tự viết để sửa dòng cuối không được tính là published baseline.

| Phương pháp | Công bố | Vai trò thực sự của LLM | Quan hệ với bài hiện tại | Trạng thái local |
| --- | --- | --- | --- | --- |
| SAGE | ACL 2026 main | LLM generator; pseudo-feature graph phụ thuộc giá trị chọn context hoặc sửa logits | Baseline gần nhất cho graph thay đổi theo row/generation step, nhưng không thay đổi theo diffusion time | Đã submit official code trên Adult, GPT-2 và Qwen3 |
| LCATS | KDD 2026 Research | Agent khám phá constraint thực thi được; TabSyn sinh dữ liệu; diagnosis và weighted repair | Baseline trực tiếp cho LLM discovery và post-generation repair | Đã có một run cũ; đang rerun full bằng release hiện tại và UnoRouter mới |
| SPADA | EMNLP 2025 main | LLM tạo sparse dependency graph; KDE/conditional flow sinh dữ liệu theo DAG | Kiểm tra graph do LLM tạo có đủ mạnh mà không cần diffusion-time activation hay không | Đã submit official graph/code trên Adult |
| GraDe | Findings EMNLP 2025 | LLM generator với graph-guided attention và FD regularization | Baseline static/learned graph bên trong LLM | Code đã tải; chưa đưa vào batch đầu vì cần môi trường Transformers cũ riêng |
| GReaT | ICLR 2023 | LLM autoregressive trực tiếp sinh row | Đối chứng LLM generator không có graph | Đã submit official code trên Adult |

SCTab-Diff không nằm trong bảng “đã publish” vì hiện là preprint. Các proxy LLM repair tự triển khai cũng không nằm trong bảng này.

## Kết quả LCATS đã có từ lần chạy trước

Đây là official-code run trên Insurance, được đánh giá lại bằng evaluator độc lập. Nó không phải số copy từ paper.

| Phương pháp | Violation theo row ↓ | Numeric KS ↓ | Categorical TV ↓ | Propensity AUC, gần 0.5 tốt | TSTR R² ↑ | Tỷ lệ ô bị sửa ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TabSyn trước LCATS | 0.0822 | 0.0359 | 0.0233 | 0.5071 | 0.7212 | 0 |
| LCATS aligned | 0.0017 | 0.0318 | 0.0233 | 0.5070 | 0.7224 | 0.2884 |

Diễn giải: LCATS loại bỏ khoảng 98% số row vi phạm tương đối và không làm giảm các metric fidelity/utility ở run này. Tuy nhiên nó chỉnh 28.84% số ô và mọi row đều có ít nhất một thay đổi; đây là lý do cần so sánh với dynamic middle-step guidance theo **edit/actuation budget tương đương**, không chỉ so violation cuối.

Nguồn local: `outputs/lcats_official_output_audit_1052790/RESULTS.md`.

## Protocol đang chạy

Tất cả phương pháp trên Adult sẽ được đánh giá bằng cùng train/test split và evaluator:

- Shape error và Trend error: fidelity biên và cặp cột, thấp hơn tốt hơn.
- Alpha precision và Beta recall: chất lượng và coverage trong không gian đã mã hóa.
- Detection score: khả năng phân biệt real/synthetic.
- TSTR XGBoost AUC: downstream utility.
- DCR: kiểm tra proximity/privacy, đặc biệt quan trọng vì SAGE dùng 70% feature-value từ một row train làm prefix theo released protocol.
- Violation trên evaluation graph độc lập: metric bổ sung, không thay thế fidelity và utility.
- Runtime, sample count và mức độ conditioning/copying phải được báo cùng kết quả.

## Job đã submit

| Job | Nội dung |
| --- | --- |
| 1085128 | Setup môi trường SPADA, GReaT và LCATS evaluation |
| 1085143 | LCATS full: discovery → TabSyn → diagnosis → alignment, model `gpt-5.6-luna` qua UnoRouter mới |
| 1085144 | LCATS reflexion + independent audit |
| 1085141 | SAGE Adult: GPT-2 và Qwen3-0.6B; Feature Selector và Logit Correction |
| 1085131 | SPADA Adult với graph response do tác giả phát hành |
| 1085137 | GReaT Adult với DistilGPT-2 |
| 1085142 | Common-metric evaluation cho các baseline Adult |

## Cách dùng trong paper

Bảng chính không nên trộn số paper với số local. Một bảng chính hợp lệ nên chứa TabDiff, Harpoon-style proxy, dynamic middle graph, SAGE, SPADA và GReaT **chỉ khi tất cả là local rerun trên cùng split/evaluator**. LCATS nên nằm ở bảng mechanism/post-repair trên Insurance hoặc một dataset LCATS hỗ trợ. Số từ paper chỉ dùng ở bảng tham khảo riêng và ghi rõ “reported by authors”.

Claim hợp lý nếu kết quả thuận lợi: dynamic graph của bài giữ được fidelity/utility của backbone tốt hơn các LLM generator/graph generator, đồng thời đạt validity tương đương post-repair với ít thay đổi hơn. Không nên claim “beat SAGE” chỉ từ violation vì SAGE và bài hiện tại có generator, conditioning và sampling protocol khác nhau.
