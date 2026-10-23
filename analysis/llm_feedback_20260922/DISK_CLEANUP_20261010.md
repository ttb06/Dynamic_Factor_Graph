# Disk cleanup — 2026-10-10

Phạm vi cleanup chỉ nằm trong `analysis/llm_feedback_20260922/outputs`.

Đã giữ lại:

- checkpoint EMA cuối cùng của mỗi run;
- checkpoint nền TabDiff cần cho inference;
- toàn bộ config, metric, bảng tổng hợp, log và graph artifact;
- các checkpoint EMA đang tạo nên kết quả chính (`full_continue`, `full_calibrated`, `robust_finetune`).

Đã xóa:

- 68 checkpoint `model_epoch_*.pt` khi cùng thư mục có đúng checkpoint `ema_model_epoch_*.pt` tương ứng (2,904,214,509 byte);
- checkpoint EMA trung gian trong các run nhiều mốc, chỉ giữ checkpoint EMA có epoch lớn nhất;
- cache Python nếu phát sinh.

Các artifact bị xóa không cần cho inference hoặc các bảng kết quả hiện tại. Chúng chỉ có thể phục hồi bằng cách train lại run tương ứng.
