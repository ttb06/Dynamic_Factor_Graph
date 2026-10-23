# Đề xuất nghiên cứu hợp nhất: sinh dữ liệu bảng theo hợp đồng

**Cập nhật:** 28/09/2026  
**Quy ước:** “đã chạy” là kết quả đã có tệp đầu ra; “đang chạy” là tác vụ đã gửi lên Slurm; “suy luận” là diễn giải chưa được kiểm chứng trực tiếp.

> Bản cập nhật mới hơn, dùng thuật ngữ `condition`, `guidance schedule`, `backbone`, `adapter` và mô tả các jobs ngày 29/09, nằm tại [`RESEARCH_UPDATE_20260929.md`](RESEARCH_UPDATE_20260929.md).

## 1. Kết luận trước

Tên bài toán nên dùng:

> **Sinh dữ liệu bảng theo hợp đồng dưới dịch chuyển cấu trúc bảng và yêu cầu người dùng.**

Một yêu cầu bằng ngôn ngữ tự nhiên được biên dịch thành hợp đồng có kiểu. Bộ sinh dữ liệu nhận lược đồ bảng và hợp đồng này, sau đó tạo ra một nhóm bản ghi vừa thỏa yêu cầu vừa giữ phân phối của nhóm thật tương ứng. Bộ kiểm chứng độc lập xác nhận kết quả và chỉ sửa phần còn vi phạm.

Điểm mới không còn nằm ở việc “dùng LLM trong diffusion”. Điểm mới có thể nằm ở giao của bốn yêu cầu:

1. yêu cầu thay đổi giữa các lần sinh và có thể chưa xuất hiện khi huấn luyện;
2. một bộ sinh được dùng chung qua nhiều bảng;
3. yêu cầu có độ phổ biến thấp, phép kết hợp và ngoại lệ;
4. kết quả được so sánh công bằng với lọc mẫu, tinh chỉnh riêng, dẫn hướng khi lấy mẫu và sửa sau sinh.

Kết quả mới nhất làm thay đổi đánh giá dự án:

- Tập yêu cầu cũ quá dễ: sinh không điều kiện rồi lọc chỉ cần trung bình khoảng 3–7 lượt sinh cho một mẫu hợp lệ và thường giữ phân phối tốt hơn sinh có điều kiện.
- Trên tám lần lặp độc lập, ở tầng yêu cầu hỗn hợp hiếm nhất đang xét, sinh có điều kiện mức nhẹ nhanh hơn lọc mẫu `4,70 [4,37; 4,98]` lần trên Default và `7,28 [6,97; 7,62]` lần trên Shoppers. Đây là tín hiệu ổn định về hiệu suất lấy mẫu, không còn là quan sát ba seed.
- Độ trung thực chưa thắng đồng đều: Default có KS và TV xấu hơn lọc; Shoppers ở tầng hỗn hợp hiếm nhất có TV tốt hơn `0,015 [0,012; 0,017]`, nhưng KS xấu hơn nhẹ `0,003 [0,000; 0,007]`. Luận điểm đúng là đường biên chi phí–độ trung thực.
- Chọn đúng hàng áp dụng bảo vệ hoàn toàn hàng ngoài phạm vi và hàng ngoại lệ. Tuy nhiên, với bất đẳng thức một chiều đơn giản, sửa tối thiểu sau sinh vẫn tốt hơn tác động trong diffusion về độ biến dạng.
- Bộ kiểm tra kiểu dựa trên lược đồ sửa hoàn toàn lỗi nghiêm trọng do LLM biến mã phân loại dạng chuỗi thành số.
- CuTS, ICML 2024, đã xác lập bài toán tùy chỉnh dữ liệu bảng bằng biểu thức khai báo. Vì vậy dự án không cần và không được trình bày như đang tạo một bài toán hoàn toàn mới.
- Mã chính thức CuTS trên Adult đạt điểm ràng buộc `1,0` ở năm chương trình được kiểm tra, với TV biên thay đổi rất nhỏ, nhưng cần gần hai giờ tinh chỉnh cho toàn bộ lô chương trình. Vì vậy mục tiêu không phải thắng CuTS về tính hợp lệ khi cho phép tinh chỉnh riêng, mà là thực thi nhiều hợp đồng mới theo kiểu dùng lại tham số.

Đánh giá hiện tại:

- đủ cơ sở để tiếp tục thành dự án nghiên cứu;
- chưa đủ để tuyên bố kết quả tốt nhất hiện nay, vì chưa có mô hình liên bảng thật sự và chưa tái lập Harpoon/CTDF trên cùng giao thức;
- hướng riêng từng bảng có trần đóng góp vừa phải;
- hướng liên bảng có tiềm năng cao hơn nhưng cần một bộ sinh dùng chung và đối chứng CTSyn/CuTS.

## 2. Ba bài toán phải tách riêng

### 2.1 Sinh nhóm bản ghi theo yêu cầu

Ví dụ: sinh các bản ghi có `AGE >= 45` và `MARRIAGE = 1`.

Mục tiêu là lấy mẫu từ phân phối có điều kiện:

`p_data(x | C)`

Đối chứng quan trọng nhất là sinh không điều kiện rồi lọc. Nếu điều kiện phổ biến, lọc mẫu gần như là lời giải chuẩn và rất khó đánh bại về độ trung thực.

### 2.2 Bảo đảm quy tắc toàn vẹn

Ví dụ: nếu số trang quản trị bằng 0 thì thời lượng tương ứng cũng bằng 0.

Mục tiêu là loại vi phạm trong khi thay đổi dữ liệu ít nhất. Phép chiếu và sửa sau sinh có lợi thế tự nhiên. Tác động trong diffusion chỉ đáng dùng nếu nó bảo toàn quan hệ phụ thuộc hoặc hiệu quả tác vụ tốt hơn rõ ràng.

### 2.3 Thực thi chính sách ở mức toàn bảng

Ví dụ: kiểm soát tỷ lệ nhóm, cân bằng công bằng hoặc một thống kê toàn bộ tập dữ liệu.

Đây là bài toán khác vì mỗi hàng không thể được đánh giá độc lập. CuTS đã hỗ trợ nhiều biểu thức thống kê loại này. Dự án hiện tại chưa có bằng chứng đủ mạnh cho nhánh này.

### 2.4 Lựa chọn cho bài chính

Bài chính nên tập trung vào **sinh nhóm bản ghi theo yêu cầu chưa thấy**, đặc biệt khi nhóm đó hiếm. Quy tắc toàn vẹn được giữ làm nhánh chứng nhận và phân tích phản ví dụ. Chính sách toàn bảng là mở rộng sau, không nên đưa vào luận điểm đầu tiên.

## 3. Vì sao bài toán có ý nghĩa

Một bộ sinh dữ liệu thông thường chỉ học phân phối của một bảng. Khi yêu cầu thay đổi, người dùng có bốn lựa chọn:

1. sinh nhiều rồi lọc;
2. tinh chỉnh một mô hình riêng;
3. ép ràng buộc khi lấy mẫu;
4. sửa dữ liệu sau khi sinh.

Mỗi lựa chọn chỉ trở thành vấn đề trong một vùng cụ thể:

- lọc mẫu tốn kém khi nhóm yêu cầu hiếm;
- tinh chỉnh riêng không phù hợp khi có nhiều yêu cầu thay đổi;
- dẫn hướng mạnh có thể làm méo các cột không liên quan;
- sửa sau sinh có thể phá quan hệ giữa các cột hoặc xóa nhóm hiếm hợp lệ.

Vì vậy, tính cấp thiết của dự án không nên được khẳng định bằng ví dụ định tính. Nó phải được chứng minh bằng một đường biên thực nghiệm theo độ phổ biến của yêu cầu:

> Khi yêu cầu ngày càng hiếm, từ điểm nào chi phí lọc vượt lợi ích của sinh có điều kiện, và sinh có điều kiện có giữ phân phối tốt hơn sửa cưỡng bức hay không?

Đây là câu hỏi đo được, có đối chứng rõ và không phụ thuộc vào việc tự tuyên bố một ứng dụng mới.

## 4. Ranh giới với các công trình liên quan

| Công trình | Phần đã giải quyết | Phần chưa giải quyết trực tiếp | Hệ quả cho dự án |
| --- | --- | --- | --- |
| [TabDiff, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5c882988ce5fac487974ee4f415b96a9-Abstract-Conference.html) | diffusion cho dữ liệu số và phân loại; lịch nhiễu theo đặc trưng; điền cột thiếu có điều kiện | yêu cầu khai báo chưa thấy và chuyển giao liên bảng | mô hình lõi hiện tại, không phải điểm mới |
| [TabSyn, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/e9750610639c3e7a849cff746bf60dbd-Abstract-Conference.html) | đưa kiểu hỗn hợp vào không gian ẩn liên tục rồi diffusion | hợp đồng thay đổi và chuyển giao liên bảng | mô hình lõi thay thế |
| [CuTS, ICML 2024](https://proceedings.mlr.press/v235/vero24a.html) | tùy chỉnh dữ liệu bằng biểu thức logic/thống kê; tinh chỉnh theo yêu cầu | không nhằm chuyển giao sang yêu cầu và bảng mới mà không tinh chỉnh | đối chứng trực tiếp nhất cho sinh dữ liệu tùy chỉnh |
| [Harpoon, ICLR 2026](https://arxiv.org/html/2602.07875) | dẫn hướng khả vi lúc lấy mẫu cho ràng buộc tùy ý | yêu cầu ngôn ngữ, chuyển giao liên bảng và ngoại lệ bất định | đối chứng dẫn hướng bắt buộc |
| CTDF, ICAIF 2025 | toán tử khả thi ở từng bước cho số và phân loại | yêu cầu mới, bộ biên dịch và chuyển giao bảng | bác bỏ luận điểm “đầu tiên áp ràng buộc trong khử nhiễu” |
| [SCTab-Diff, bản thảo 2026](https://arxiv.org/html/2609.16069) | tri thức do LLM trích xuất điều kiện hóa quá trình khử nhiễu ngược | tri thức chủ yếu tĩnh theo bảng | bác bỏ luận điểm “đầu tiên dùng tri thức LLM trong khử nhiễu” |
| [LCATS, KDD 2026](https://doi.org/10.1145/3770855.3818181) | tìm ràng buộc, sinh bằng TabSyn, chẩn đoán và sửa | không tập trung vào sinh trực tiếp nhóm bản ghi theo yêu cầu mới | đối chứng sửa sau sinh |
| [Zhao và cộng sự, bản thảo 2026](https://arxiv.org/abs/2608.15109) | LLM tạo quy tắc thực thi được, kiểm chứng bằng phản ví dụ và sửa đầu ra | không tập trung vào bộ sinh chung theo yêu cầu | bộ biên dịch và kiểm chứng không đủ làm điểm mới |
| [CTSyn, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/778055b859a3731bbe0d92fa61655e80-Abstract-Conference.html) | mô hình nền tảng diffusion liên bảng, điều kiện hóa theo lược đồ | không điều kiện hóa theo hợp đồng người dùng chưa thấy | mô hình lõi và đối chứng chính cho hướng mô hình nền tảng |
| [PromptLoop, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Lee_PromptLoop_Plug-and-Play_Prompt_Refinement_via_Latent_Feedback_for_Diffusion_Model_CVPR_2026_paper.html) và các công trình ảnh/video | mô hình nền tảng đọc bản xem trước rồi điều chỉnh quá trình diffusion | không dành cho bảng nhưng đã bao phủ ý tưởng vòng phản hồi | không nhận là điểm mới chỉ vì LLM nhìn `x̂0` |

### 4.1 Mối đe dọa mới từ CuTS

CuTS quan trọng vì nó đã định nghĩa rõ giao diện biểu thức khai báo, gồm quy tắc theo hàng, mệnh đề kéo theo, thống kê và công bằng. CuTS tiền huấn luyện mô hình lõi theo từng bảng rồi tinh chỉnh bằng hàm mất mát khả vi cho từng chương trình. Với mệnh đề kéo theo và ràng buộc theo hàng, mã chính thức còn kết hợp lọc mẫu khi sinh.

Vì vậy không được tuyên bố:

- lần đầu cho phép người dùng tùy chỉnh dữ liệu bảng tổng hợp;
- lần đầu dùng chương trình ràng buộc tổng quát;
- lần đầu kết hợp yêu cầu logic và thống kê.

Khoảng trống so với CuTS phải là:

- không tinh chỉnh riêng theo từng yêu cầu;
- chuyển giao sang tham số và phép kết hợp chưa thấy;
- dùng một mô hình lõi chung qua nhiều bảng;
- kiểm chứng kiểu dữ liệu và ngữ nghĩa của yêu cầu ngôn ngữ;
- đo rõ chi phí so với lọc mẫu của chính CuTS.

### 4.2 Phân loại đúng vị trí của tín hiệu điều khiển

| Nhóm phương pháp | Công trình tiêu biểu | Tín hiệu xuất hiện ở đâu | Điều dự án không nên lặp lại |
| --- | --- | --- | --- |
| Chuẩn bị trước khi sinh | [LLM-TabLogic](https://arxiv.org/html/2503.02161) | LLM khám phá quan hệ từ siêu dữ liệu và chọn biểu diễn trước diffusion | chỉ dùng LLM tạo tri thức tĩnh hoặc chọn biểu diễn |
| Điều kiện tĩnh trong khử nhiễu | [SCTab-Diff](https://arxiv.org/html/2609.16069) | tri thức được trích xuất trước, nhưng hàm điểm vẫn phụ thuộc trạng thái hàng ở từng bước | gọi một tri thức tiên nghiệm “động” chỉ vì tác động thay đổi theo trạng thái |
| Dẫn hướng khả vi trong khi lấy mẫu | Harpoon và CTDF | hàm mất mát hoặc toán tử khả thi tác động tại các bước khử nhiễu | tuyên bố mới từ việc ép quy tắc trong quá trình khử nhiễu ngược |
| Mô hình lớn đánh giá bản xem trước | [PromptLoop](https://openaccess.thecvf.com/content/CVPR2026/html/Lee_PromptLoop_Plug-and-Play_Prompt_Refinement_via_Latent_Feedback_for_Diffusion_Model_CVPR_2026_paper.html), [PPAD](https://arxiv.org/html/2505.20053), [Free²Guide](https://openaccess.thecvf.com/content/ICCV2025/html/Kim_Free2Guide_Training-Free_Text-to-Video_Alignment_using_Image_LVLM_ICCV_2025_paper.html), [TTGen](https://cvpr.thecvf.com/virtual/2025/poster/35755) | xem dự đoán sạch, chấm điểm, sửa chỉ dẫn, chọn nhánh hoặc quay lui | tuyên bố mới chỉ từ vòng phản hồi ở giữa quá trình sinh |
| Khám phá và kiểm chứng quy tắc bằng LLM | Zhao cùng cộng sự và EVS, hội thảo ICML 2026 | tạo giả thuyết thực thi được, kiểm chứng trên dữ liệu, loại quy tắc làm giảm hiệu quả tác vụ | xem việc LLM biên dịch và kiểm chứng quy tắc là đủ mới |
| Sửa sau sinh | LCATS và Zhao cùng cộng sự | chẩn đoán mẫu cuối rồi sửa bằng quy tắc thực thi được | chỉ chuyển LLM từ giữa quá trình sang cuối quá trình |
| Tinh chỉnh theo chương trình | CuTS | tối ưu mô hình riêng theo chương trình ràng buộc, sau đó có thể lọc | trình bày tùy chỉnh theo yêu cầu như một bài toán mới |
| Diffusion rời rạc có ràng buộc | [Constrained Discrete Diffusion, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/124c1cc9f7750f512ed81cdbf2fd5bf5-Abstract-Conference.html) | ràng buộc đi trực tiếp vào bước khử nhiễu rời rạc | bỏ qua đối chứng kỹ thuật cho cột phân loại |
| Quan hệ phụ thuộc giá trị và văn bản dài | [SAGE, ACL 2026](https://aclanthology.org/2026.acl-long.174/), [AFT-Tab, ACL 2026](https://aclanthology.org/2026.acl-long.209/), [TabDLM, bản thảo 2026](https://arxiv.org/html/2602.22586) | mô hình hóa phụ thuộc theo giá trị hoặc quan hệ văn bản–cột cấu trúc | mở nhánh văn bản dài trước khi có bộ dữ liệu và mô hình lõi phù hợp |

Điểm cần phân biệt là **đặc tả ràng buộc** và **phản ứng phụ thuộc trạng thái**. Harpoon đã có hàm mất mát cố định nhưng gradient thay đổi theo `x̂0,t`; SCTab-Diff có tri thức cố định nhưng hàm điểm phụ thuộc hàng hiện tại. Vì vậy “tín hiệu thay đổi theo hàng hoặc bước thời gian” vẫn chưa đủ mới. Khoảng trống chặt hơn là hợp đồng chưa thấy, chuyển giao liên bảng, phạm vi áp dụng và ngoại lệ, cùng chứng nhận độc lập dưới ngân sách thực tế.

Các nguồn hội nghị chính được dùng làm mốc kỹ thuật mạnh. Bản thảo và bài hội thảo như SCTab-Diff, Zhao, PPAD và TTGen được xem là mối đe dọa ý tưởng hoặc đối chứng gần, không được trình bày như kết quả đã được nhận ở hội nghị chính.

## 5. Những gì đã chạy và ý nghĩa

### 5.1 Kiểm toán dữ liệu và quan hệ

Ba bảng Adult, Default và Shoppers tạo ra 788 quan hệ ứng viên:

- 80 quan hệ thực nghiệm không có phản ví dụ trong tập huấn luyện;
- 95 quan hệ gần cứng;
- 334 quan hệ xác suất;
- 85 quan hệ nhạy cảm chỉ nên dùng để kiểm toán;
- 132 quan hệ thiếu số mẫu.

Phần lớn quan hệ khai phá từ dữ liệu không được phép biến thẳng thành quy tắc cứng. Ví dụ `Own-child -> age < 30` có gần 19,7% ngoại lệ hợp lệ. Đây là bằng chứng cho việc phải biểu diễn phạm vi áp dụng, ngoại lệ và độ chắc chắn.

### 5.2 Mẫu trung gian của diffusion

Phân tích quỹ đạo khử nhiễu trước đó cho thấy dự đoán bản ghi sạch trở nên đọc được dần theo thời gian, nhưng cột số và cột phân loại ổn định không đồng bộ. Bản xem trước quá sớm làm bộ kiểm chứng sai. Tuy nhiên, chưa có bằng chứng rằng đọc bản xem trước bằng LLM mang lại lợi ích vượt lịch cố định hoặc bộ kiểm chứng được viết bằng mã.

Kết luận: thăm dò quỹ đạo có giá trị phân tích, không đủ làm phương pháp chính.

### 5.3 Vai trò phù hợp của LLM

Biên dịch một lần từ ngôn ngữ tự nhiên sang DSL rồi thực thi xác định tốt hơn hỏi LLM theo từng hàng:

- trên benchmark nhỏ, hai model đều đạt tương đương thực thi 1,0 sau biên dịch;
- đánh giá trực tiếp từng hàng tốn khoảng hai mươi lần token;
- benchmark mở rộng cho thấy câu có ngoại lệ và công thức chính sách vẫn khó;
- chỉ dẫn tách `phạm vi / ngoại lệ / yêu cầu / cách sửa` tăng rõ độ ổn định.

Kiểm toán mới trên Default và Shoppers cho thấy lỗi lớn nhất là sai kiểu dữ liệu. Một lần chạy GPT-5.4-mini biến mã phân loại dạng chuỗi thành số, làm độ đúng giảm xuống 0,5. Bộ kiểm tra kiểu dựa trên lược đồ sửa độ đúng trung bình của cấu hình đó từ 0,8333 lên 1,0; đồng thuận trên lưới phản ví dụ từ 0,9597 lên 1,0.

Kết luận: kiến trúc nên là:

`LLM biên dịch -> kiểm tra kiểu -> kiểm tra phản ví dụ -> bộ thực thi xác định`

Không nên là:

`LLM đọc từng hàng -> LLM quyết định có sửa hay không`.

### 5.4 Sinh có điều kiện theo hợp đồng

Trên Adult, Default và Shoppers, biểu diễn có cấu trúc tốt hơn mã yêu cầu rời rạc khi kiểm tra ngưỡng và phép kết hợp chưa thấy. Kết quả kiểu hỗn hợp mạnh nhất nằm ở Shoppers:

- mức thỏa yêu cầu tăng khoảng 0,343 so với mã yêu cầu;
- KS số giảm khoảng 0,118;
- TV phân loại giảm khoảng 0,110;
- sau cùng một certifier, tỷ lệ cell cần sửa giảm khoảng 0,342.

Default yếu hơn: một số chỉ số cải thiện nhưng TV có khoảng tin cậy chứa 0 và số lần thắng đồng thời thấp hơn. Điều này cho thấy hiệu quả phụ thuộc độ mạnh của quan hệ giữa cột và độ phổ biến của nhóm cần sinh.

### 5.5 Tập hàng áp dụng

Lô 18 tác vụ mới đã hoàn tất trên Default và Shoppers, ba lần lặp độc lập, ba mức tác động.

Kết quả chính:

- tác động toàn cục làm lệch hàng ngoài phạm vi;
- chọn hàng từ bản xem trước đưa độ lệch hàng ngoài phạm vi và hàng ngoại lệ về 0 trong thiết kế ghép cặp;
- cổng chọn hàng bị xáo trộn chỉ giữ một phần nhỏ lợi ích, nên lựa chọn đúng hàng có tác dụng nhân quả;
- tác động mạnh làm cột đích bị đẩy quá xa.

Ví dụ Shoppers:

| Cách xử lý | Hệ số | Vi phạm | Lệch hàng áp dụng | KS | Tỷ lệ sửa |
| --- | ---: | ---: | ---: | ---: | ---: |
| không tác động | 4 | 0,3896 | 0 | 0 | 0 |
| chọn đúng hàng | 4 | 0,0233 | 5,0107 | 0,0259 | 0 |
| chọn đúng hàng rồi chứng nhận | 4 | 0 | 5,0135 | 0,0259 | 0,0080 |
| sửa tối thiểu sau sinh | 4 | 0 | 0,0915 | 0,0072 | 0,0716 |

Diễn giải:

- tập hàng tác động giải quyết đúng vấn đề rò tác động sang hàng không áp dụng;
- nhưng với một ngưỡng số đơn giản, sửa tối thiểu gần như là nghiệm tối ưu theo độ dịch chuyển;
- vì vậy tập hàng tác động không nên là đóng góp phương pháp chính cho quy tắc cứng đơn giản;
- nó chỉ còn đáng nghiên cứu khi thay đổi một cột đòi hỏi cập nhật đồng bộ nhiều cột liên quan.

### 5.6 Lọc mẫu so với sinh có điều kiện

Mỗi lần chạy dùng 65.536 mẫu sinh không điều kiện và lấy tập kiểm tra làm phân phối tham chiếu.

| Bảng | Loại yêu cầu | Tỷ lệ thỏa tự nhiên | Số lượt sinh trung bình cho một mẫu hợp lệ | KS sau lọc | KS của mẫu hợp lệ từ sinh có điều kiện, hệ số 1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Default | số | 0,4280 | 2,999 | 0,0566 | 0,0833 |
| Default | hỗn hợp | 0,3733 | 3,313 | 0,0511 | 0,0744 |
| Shoppers | số | 0,3601 | 3,689 | 0,0745 | 0,0948 |
| Shoppers | hỗn hợp | 0,2459 | 7,367 | 0,0769 | 0,1001 |

Kết luận mạnh:

- tập yêu cầu cũ không đủ khó;
- lọc mẫu đạt hợp lệ tuyệt đối và độ trung thực tốt hơn;
- các kết quả trước đây so với chọn tốt nhất trong 2 hoặc 4 nhánh chưa đủ để bác bỏ lọc từ một tập mẫu lớn;
- không được dùng tập yêu cầu này làm bằng chứng chính cho sự cần thiết của phương pháp.

Kết quả đầy đủ tám lần lặp của tập yêu cầu hiếm cho thấy hiệu quả tăng đơn điệu khi hỗ trợ giảm. Bảng dưới dùng sinh có điều kiện hệ số 1 và lọc mẫu từ cùng mô hình lõi.

| Bảng | Loại yêu cầu | Tầng | Hệ số tăng tốc lấy mẫu [95%] | Chênh lệch xác suất thành công [95%] | Chênh lệch KS [95%] | Chênh lệch TV [95%] |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Default | hỗn hợp | phân vị 75% | 2,06 [2,01; 2,12] | 0,133 [0,126; 0,139] | 0,028 [0,026; 0,030] | 0,021 [0,019; 0,023] |
| Default | hỗn hợp | phân vị 97,5% | 4,70 [4,37; 4,98] | 0,079 [0,069; 0,086] | 0,011 [0,005; 0,018] | 0,029 [0,024; 0,034] |
| Shoppers | hỗn hợp | phân vị 75% | 2,43 [2,39; 2,49] | 0,150 [0,148; 0,152] | 0,039 [0,037; 0,041] | 0,013 [0,011; 0,015] |
| Shoppers | hỗn hợp | phân vị 90% | 4,19 [4,09; 4,30] | 0,105 [0,104; 0,107] | -0,001 [-0,004; 0,002] | -0,003 [-0,006; -0,001] |
| Shoppers | hỗn hợp | phân vị 97,5% | 7,28 [6,97; 7,62] | 0,071 [0,066; 0,076] | 0,003 [0,000; 0,007] | -0,015 [-0,017; -0,012] |

Quy ước: hệ số tăng tốc lớn hơn 1 có lợi cho sinh có điều kiện; chênh lệch KS/TV nhỏ hơn 0 có lợi cho sinh có điều kiện. Khoảng tin cậy được bootstrap theo khối seed–hợp đồng, không coi từng hàng sinh là một quan sát độc lập.

Kết luận:

- lợi thế về hiệu suất lấy mẫu đã ổn định qua tám seed và tăng khi yêu cầu hiếm hơn;
- lợi thế về độ trung thực chỉ xuất hiện cục bộ trên Shoppers, chưa giữ trên Default;
- điều cần cải thiện bằng phương pháp không phải tính hợp lệ tuyệt đối, mà là giảm sai khác phân phối ở cùng ngân sách sinh;
- lịch dẫn hướng theo thời điểm và phạm vi tác động theo cột đang được kiểm tra trực tiếp để giải quyết điểm yếu này.

### 5.7 Đối chứng CuTS chính thức

Một lần chạy mã chính thức trên Adult cho kết quả sau:

| Chương trình | Điểm nền | Điểm CuTS | TV nền | TV CuTS | BAC nền | BAC CuTS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| implication2 | 0,9280 | 1,0000 | 0,0001 | 0,0001 | 0,7735 | 0,7753 |
| implication4 | 0,9851 | 1,0000 | 0,0001 | 0,0003 | 0,7691 | 0,7742 |
| implication5 | 0,5894 | 1,0000 | 0,0001 | 0,0002 | 0,7730 | 0,7756 |
| line_constraint1 | 0,3256 | 1,0000 | 0,0001 | 0,0021 | 0,7756 | 0,7742 |
| line_constraint2 | 0,4030 | 1,0000 | 0,0001 | 0,0002 | 0,7691 | 0,7722 |

CuTS mất khoảng 1 giờ 57 phút cho lô có tinh chỉnh, trong khi đánh giá checkpoint nền mất 4 phút 30 giây. Với mệnh đề kéo theo và ràng buộc theo hàng, mã chính thức còn dùng lọc mẫu. Đây là một đối chứng rất mạnh: dự án chỉ có lợi thế nếu amortize được chi phí qua nhiều hợp đồng mới hoặc bảng mới. Năm lần lấy mẫu trên cùng lần tinh chỉnh đã được gửi để đo phương sai lấy mẫu; chúng không thay thế seed huấn luyện độc lập.

## 6. Trạng thái các giả thuyết

| Giả thuyết | Trạng thái | Bằng chứng hiện tại | Quyết định |
| --- | --- | --- | --- |
| Mẫu trung gian đủ đọc được ở một số bước | được ủng hộ một phần | độ ổn định tăng theo thời gian nhưng không đồng đều giữa kiểu cột | giữ làm phân tích, không làm luận điểm chính |
| Phản hồi động xử lý lỗi mà tri thức tĩnh bỏ sót | chưa đủ bằng chứng | tập hàng tác động có ích, nhưng bản xem trước không chứng minh LLM cần thiết | chưa đầu tư lớn |
| Tác động trong diffusion tốt hơn sửa sau sinh | bị bác bỏ cho ngưỡng đơn giản | sửa sau sinh đạt vi phạm 0 với biến dạng thấp hơn nhiều | bỏ bối cảnh này khỏi bài chính |
| LLM bổ sung cho bộ máy quy tắc | bị bác bỏ khi quy tắc đã thực thi được | bộ thực thi DSL và bộ phân tích mẫu thắng đánh giá trực tiếp | LLM chỉ làm bộ biên dịch cho ngôn ngữ tự do |
| Biên dịch một lần đáng tin và tiết kiệm | được ủng hộ | ít token hơn, bộ kiểm tra kiểu sửa lỗi nghiêm trọng | giữ làm mô-đun giao diện |
| Biểu diễn hợp đồng chuyển giao sang phép kết hợp mới | được ủng hộ trong từng bảng | lặp lại trên ba bộ dữ liệu, mạnh nhất ở Shoppers | giữ làm bằng chứng khả thi |
| Sinh có điều kiện thắng lọc mẫu trên tập yêu cầu thường | bị bác bỏ | lọc rẻ và trung thực hơn | dùng các tầng độ hiếm làm thí nghiệm chính |
| Sinh có điều kiện có lợi ở nhóm hiếm | được ủng hộ chắc về hiệu suất lấy mẫu | tám seed: tăng tốc 4,70 lần ở Default và 7,28 lần ở Shoppers tại tầng hỗn hợp hiếm nhất | giữ làm luận điểm thực nghiệm chính |
| Lợi ích tăng đều khi độ phổ biến giảm | được ủng hộ | hệ số tăng tốc đi từ 2,06 lên 4,70 ở Default và từ 2,43 lên 7,28 ở Shoppers | dùng đường cong hỗ trợ–chi phí thay vì một ngưỡng tùy ý |
| Chỉ cập nhật phép chiếu điều kiện giữ phân phối tốt hơn cập nhật lõi | đang chạy | smoke test đã hoàn tất; ma trận ba bảng, sáu biến thể, ba seed đã gửi | chọn cấu hình theo đường biên chi phí–độ trung thực, không theo validity đơn lẻ |
| Dẫn hướng mạnh sớm rồi giảm dần về cuối cải thiện đường biên | đang chạy | tín hiệu sớm trên Default cho thấy giảm KS khi nhả dẫn hướng, nhưng chưa đủ ba seed | chờ tổng hợp 216 cấu hình trước khi giữ hoặc loại |
| Chỉ truyền tác động vào cột đích giảm lệch cột ngoài yêu cầu | đang chạy thử | đã có bằng chứng tương tự trên quy tắc kéo theo; smoke test cho yêu cầu cohort đã gửi | nếu thành công, kết hợp với lịch theo thời điểm thành đóng góp phương pháp |
| Một mô hình dùng chung chuyển giao qua bảng và hợp đồng | chưa thử | hiện vẫn huấn luyện adapter riêng | mục tiêu dài hạn chính |
| Nhánh văn bản dài | chưa thử | bộ dữ liệu hiện tại không phù hợp | chưa ưu tiên |

## 7. Phương pháp đề xuất sau khi thu hẹp

### 7.1 Luồng xử lý

1. Người dùng mô tả nhóm bản ghi cần sinh bằng ngôn ngữ tự nhiên.
2. LLM biên dịch yêu cầu sang hợp đồng có kiểu.
3. Bộ kiểm tra lược đồ xác nhận tên cột, kiểu, miền giá trị và phạm vi có thể sửa.
4. Bộ kiểm tra phản ví dụ thử hợp đồng trên các hàng sinh nhân tạo và trường hợp biên.
5. Bộ mã hóa hợp đồng tạo biểu diễn dùng chung cho nhiều bảng.
6. Bộ sinh liên bảng nhận biểu diễn lược đồ và hợp đồng để tạo nhóm bản ghi.
7. Bộ kiểm chứng độc lập đánh giá hợp lệ.
8. Nếu cần, bộ sửa tối thiểu chỉ xử lý phần dư.

### 7.2 Hai tầng triển khai

**Tầng gần:** giữ TabDiff riêng từng bảng để nghiên cứu quan hệ giữa độ hiếm, chi phí lọc và độ trung thực. Đây là nơi các tác vụ hiện tại hoạt động.

**Tầng chính:** tích hợp bộ mã hóa hợp đồng vào CTSyn hoặc một mô hình lõi liên bảng tương đương. Một bộ tham số dùng chung được kiểm tra trên bảng chưa thấy và hợp đồng chưa thấy.

### 7.3 LLM nằm ở đâu

LLM không cần xuất hiện trong từng bước diffusion. Vai trò hợp lý là:

- hiểu cách diễn đạt tự do;
- ánh xạ tên gọi miền sang cột;
- tách phạm vi, ngoại lệ và yêu cầu;
- tạo DSL có kiểu;
- trả về độ tin cậy để hệ thống từ chối khi mơ hồ.

Mọi quyết định có thể kiểm tra bằng code phải do bộ thực thi xác định đảm nhiệm.

### 7.4 Đóng góp phương pháp đang được kiểm tra

Phương pháp gần hạn không nên là thêm một vòng gọi LLM ở mọi bước. Phần có cơ sở hơn gồm ba thành phần nhỏ nhưng gắn trực tiếp với các thất bại đã đo:

1. **Bộ điều hợp dùng lại cho nhiều hợp đồng.** TabDiff được giữ làm mô hình lõi. Hợp đồng có kiểu đi qua một phép chiếu điều kiện. Một bộ điều hợp được huấn luyện một lần trên các hợp đồng nguyên tử rồi dùng lại cho ngưỡng và phép hội chưa thấy.
2. **Chương trình huấn luyện phủ liên tục miền tham số.** Thay vì chỉ học ba ngưỡng thủ công trên mỗi cột, biến thể mới dùng các phân vị 10–90% của tập huấn luyện. Đây là phép kiểm tra trực tiếp xem thất bại ngoại suy trước đó đến từ dữ liệu hợp đồng thưa hay từ biểu diễn không đủ tốt.
3. **Dẫn hướng có lịch và giới hạn phạm vi.** Cường độ điều kiện mạnh ở phần đầu rồi giảm về cuối để quỹ đạo quay về miền dữ liệu; hiệu số conditional–unconditional chỉ được truyền vào cột đích của hợp đồng. Mục tiêu là giữ lợi thế tạo nhóm hiếm nhưng giảm KS/TV trên các cột không liên quan.

Ma trận ablation bắt buộc tách rõ:

- chỉ cập nhật phép chiếu điều kiện;
- cập nhật lõi với learning rate rất nhỏ;
- cập nhật lõi theo cấu hình hiện tại;
- chương trình ngưỡng thưa và dày;
- dropout điều kiện 0,1 và 0,3;
- dẫn hướng cố định, giảm theo bậc, giảm tuyến tính, tăng tuyến tính và xung giữa;
- tác động toàn bộ đầu ra và chỉ cột đích.

Đây vẫn là phương pháp theo từng bảng. Nó chỉ trở thành bước hướng tới mô hình nền tảng nếu bộ mã hóa hợp đồng và phần tham số điều khiển được dùng chung thật sự qua bảng. Không được gọi ba bộ điều hợp riêng là chuyển giao liên bảng.

## 8. Benchmark đề xuất

### 8.1 Không xây lại từ số không

- dùng bộ dữ liệu và phép chia của CTSyn cho phần liên bảng;
- dùng các chương trình ràng buộc của CuTS cho phần tùy chỉnh;
- dùng rule families của Harpoon, CTDF, SCTab-Diff và LCATS;
- giữ metric chuẩn của TabDiff/CTSyn;
- bổ sung phép chia theo hợp đồng và độ hiếm.

### 8.2 Bốn trục đánh giá

1. **Bảng:** bảng đã thấy và bảng chưa thấy.
2. **Hợp đồng:** tham số đã thấy, tham số mới, phép kết hợp mới, ngoại lệ mới.
3. **Độ phổ biến:** trên 20%, 5–20%, 1–5%, dưới 1%.
4. **Nguồn yêu cầu:** template, chương trình từ related work, data dictionary và người viết độc lập.

### 8.3 Đối chứng bắt buộc

#### Đối chứng lấy mẫu

- sinh không điều kiện;
- sinh rồi lọc;
- best-of-N;
- huấn luyện riêng theo từng yêu cầu.

#### Đối chứng ràng buộc

- sửa tối thiểu sau sinh;
- sửa theo LCATS/Zhao;
- Harpoon;
- CTDF;
- CuTS.

#### Đối chứng liên bảng

- CTSyn chỉ điều kiện hóa theo lược đồ;
- CTSyn cộng lọc mẫu;
- CTSyn cộng sửa;
- CTSyn tinh chỉnh riêng theo hợp đồng;
- TabDiff riêng từng bảng.

#### Đối chứng biểu diễn

- mã yêu cầu rời rạc;
- DSL oracle;
- parser template;
- bộ biên dịch LLM không kiểm tra kiểu;
- bộ biên dịch LLM có kiểm tra kiểu và phản ví dụ.

### 8.4 Chỉ số

- tỷ lệ thỏa hợp đồng bằng verifier độc lập;
- KS cho cột số và TV cho cột phân loại;
- quan hệ phụ thuộc giữa cột và độ trung thực theo nhóm;
- tác vụ sử dụng dữ liệu sau sinh;
- bao phủ nhóm hiếm;
- số lượt sinh cho một mẫu hợp lệ;
- số cell phải sửa và độ lớn thay đổi;
- thời gian huấn luyện riêng theo hợp đồng;
- chi phí LLM, thời gian chạy và bộ nhớ;
- tính riêng tư nếu tiến tới bản nộp đầy đủ.

## 9. Luận điểm đóng góp có thể bảo vệ

### Luận điểm 1: giao thức đánh giá

Các bộ đánh giá hiện tại thường giữ yêu cầu cố định, tinh chỉnh riêng hoặc chỉ đo ràng buộc cuối. Dự án tách đồng thời:

- bảng chưa thấy;
- hợp đồng chưa thấy;
- độ hiếm của nhóm cần sinh;
- chi phí lọc/sửa.

### Luận điểm 2: biểu diễn hợp đồng chuyển giao được

Một hợp đồng có kiểu cho phép tổng hợp ngưỡng, giá trị phân loại, phép hội và ngoại lệ mới tốt hơn mã yêu cầu rời rạc.

### Luận điểm 3: bộ sinh liên bảng theo hợp đồng

Một mô hình lõi dùng chung giảm số lượt sinh hoặc thời gian tinh chỉnh trong vùng nhóm hiếm mà không làm giảm đáng kể độ trung thực và hiệu quả tác vụ.

### Luận điểm 4: giao diện LLM có kiểm chứng

Biên dịch một lần, kiểm tra kiểu và kiểm tra phản ví dụ đáng tin hơn đánh giá trực tiếp từng hàng, đồng thời rẻ hơn nhiều.

Luận điểm 3 là phần chưa có bằng chứng. Nếu không đạt, dự án chỉ còn đóng góp về giao thức đánh giá và giao diện yêu cầu.

## 10. Những điều không được tuyên bố

- không nói lần đầu dùng LLM trong diffusion;
- không nói lần đầu đưa ràng buộc vào diffusion cho dữ liệu bảng;
- không nói lần đầu cho phép tùy chỉnh dữ liệu bảng;
- không nói lần đầu có mô hình nền tảng sinh bảng;
- không nói tốt hơn CuTS, CTSyn, Harpoon hoặc CTDF khi mới dùng đối chứng xấp xỉ;
- không xem tương quan khai phá từ tập huấn luyện là quy luật miền;
- không xem ba bộ điều hợp riêng là chuyển giao liên bảng;
- không xem tỷ lệ hợp lệ bằng 1 sau phép chiếu là bằng chứng bộ sinh đã học ngữ nghĩa.

## 11. Điều kiện dừng hoặc đổi hướng

Chuyển sang bài phân tích và đánh giá nếu xảy ra một trong các trường hợp:

1. lọc mẫu vẫn rẻ hơn và trung thực hơn ở nhóm có độ phổ biến dưới 1–5%;
2. CuTS tinh chỉnh nhanh và tốt hơn rõ trên mọi yêu cầu;
3. mô hình dùng chung không hơn bộ điều hợp riêng trên bảng chưa thấy;
4. lợi ích biến mất khi đổi tên cột hoặc bỏ mô tả quen thuộc;
5. sinh có điều kiện làm mất nhóm hiếm hoặc giảm hiệu quả tác vụ;
6. hợp đồng từ nguồn độc lập không được bộ biên dịch xử lý ổn định.

Tiếp tục theo hướng phương pháp nếu đạt:

1. giảm rõ số lượt sinh ở nhóm hiếm;
2. giữ KS/TV gần lọc mẫu và tốt hơn sửa sau sinh;
3. giữ dấu trên ít nhất ba miền dữ liệu;
4. chuyển giao đồng thời sang bảng và hợp đồng mới;
5. lợi ích còn sau khi chuẩn hóa chi phí huấn luyện và lấy mẫu.

## 12. Các tác vụ Slurm mới

### Đã hoàn tất

- `1044553`: kiểm tra kiểu và lưới phản ví dụ cho bộ biên dịch.
- `1044574–1044591`: 18 tác vụ tập hàng áp dụng, hai bảng × ba lần lặp × ba mức tác động.
- `1044592`: tổng hợp tập hàng áp dụng.
- `1044593–1044604`: lọc mẫu so với sinh có điều kiện trên tập yêu cầu số và hỗn hợp.
- `1044605`: tổng hợp lô yêu cầu thường.
- `1044670–1044681`: quét theo độ phổ biến, gồm các phân vị đến 99% và phép kết hợp hiếm.
- `1044682`: tổng hợp đầy đủ ba lần lặp; đây là nguồn của các số trong Mục 5.6.
- `1044699`: dựng môi trường và tải checkpoint CuTS chính thức, đã hoàn tất.
- `1044704–1044744`: mở rộng quét nhóm hiếm lên tám lần lặp và tổng hợp, đã hoàn tất.
- `1044746–1044747`: CuTS chính thức trên Adult sau bản vá đánh giá, đã hoàn tất.
- `1045512–1045513`: tổng hợp lại đúng tám seed và bootstrap theo khối, đã hoàn tất; đây là nguồn của bảng mới ở Mục 5.6.
- `1045521`: smoke test bộ điều hợp chỉ cập nhật phép chiếu với chương trình ngưỡng dày, đã hoàn tất.

### Đã gửi và đang chạy/chờ

- `1045518`, `1045525–1045739`: 216 phép thử lịch dẫn hướng, gồm hai bảng, hai loại hợp đồng hiếm, ba seed và 18 lịch/cường độ.
- `1045740`: tổng hợp đường biên Pareto theo số lần gọi bộ khử nhiễu và KS/TV.
- `1045746–1045748`: CuTS với năm lần lấy mẫu trên cùng lần tinh chỉnh và tác vụ tổng hợp.
- `1045755–1045916`: huấn luyện và đánh giá sáu biến thể bộ điều hợp trên Adult, Default và Shoppers; ba seed; hai tập hợp đồng. Tổng cộng 53 lần huấn luyện mới, 108 lần đánh giá và một tác vụ tổng hợp; smoke test `1045521` được tái sử dụng làm lần huấn luyện còn lại.
- `1045923`, `1045925–1045996`: smoke test và 71 lần chạy xác nhận cho dẫn hướng chỉ tác động cột đích, cùng một tác vụ tổng hợp. Lưới chỉ giữ sáu lịch đại diện thay vì lặp lại toàn bộ 18 cấu hình.

### Lỗi hạ tầng đã xử lý

- `1044686` thất bại vì chương trình chia dữ liệu của CuTS không hỗ trợ Adult; Adult đã có sẵn tập huấn luyện và kiểm tra chính thức. Bước thừa này được bỏ và `1044699` hoàn tất.
- `1044700` thất bại vì `classifier_name` chưa được gán trong bộ đánh giá hiệu quả tác vụ của mã CuTS chính thức. Bản vá tối thiểu chọn bộ phân loại đầu tiên trong cấu hình; cấu hình mặc định của bài báo là XGBoost. Không thay đổi thuật toán sinh hoặc huấn luyện.
- `1044701` khởi động trước khi bản vá được nạp nên thất bại ở cùng lỗi khi tới bước đánh giá. `1044747` là lần chạy sạch sau bản vá.
- `1044687–1044688` không chạy do phụ thuộc vào lần dựng môi trường thất bại đầu tiên; chúng không được tính là kết quả thực nghiệm.

### Vị trí kết quả

- tập hàng áp dụng: `outputs/cross_active_aggregate_1044592/`;
- lọc mẫu: `outputs/rejection_frontier_aggregate_1044605/`;
- quét độ hiếm ba lần lặp: `outputs/rejection_frontier_aggregate_1044682/`;
- bản tổng hợp tách theo từng phân vị: `outputs/rejection_frontier_aggregate_1044745/`;
- khoảng bất định ba lần lặp sau khi hoàn tất: `outputs/rare_support_statistical_audit_1044745_1044748/`;
- quét độ hiếm tám lần lặp sau khi hoàn tất: `outputs/rejection_frontier_aggregate_1044744/`;
- khoảng bất định tám lần lặp đã sửa lỗi ghép seed: `outputs/rare_support_statistical_audit_1045512_1045513/`;
- CuTS chính thức: `outputs/cuts_official_result_audit_1045524/`;
- dẫn hướng theo thời điểm sau khi tổng hợp: `outputs/temporal_contract_guidance_aggregate_1045740/`;
- ablation bộ điều hợp sau khi tổng hợp: `outputs/adapter_variant_frontier_aggregate_1045916/`;
- dẫn hướng giới hạn ở cột đích sau khi tổng hợp: thư mục `outputs/temporal_contract_guidance_aggregate_1045996/`;
- CuTS ghi kết quả theo cấu trúc chính thức trong `external/cuts/experiment_data/` và nhật ký trong thư mục `analysis/llm_feedback_20260922/logs/`.

## 13. Kế hoạch nghiên cứu sau lô hiện tại

### Giai đoạn 1: xác định vùng bài toán có ý nghĩa

Mục tiêu là tìm điểm chuyển giữa lọc mẫu, sinh có điều kiện và sửa. Không mở rộng mô hình trước khi có kết quả này.

### Giai đoạn 2: tái lập trung thực đối chứng CuTS

So sánh cùng ràng buộc, cùng bộ dữ liệu và cùng ngân sách. Cần tách chi phí tiền huấn luyện mô hình lõi, tinh chỉnh theo yêu cầu và thời gian sinh.

### Giai đoạn 3: thử nghiệm nhỏ dùng chung qua bảng

Không cần tiền huấn luyện rất lớn ngay. Trước tiên dùng 5–8 bảng, giữ lại một bảng và một họ hợp đồng, so sánh:

- bộ mã hóa riêng từng bảng;
- bộ mã hóa dùng chung;
- chỉ dùng lược đồ;
- dùng lược đồ cộng hợp đồng.

Nếu bộ mã hóa dùng chung không chuyển giao, dừng hướng mô hình nền tảng sớm.

### Giai đoạn 4: tích hợp CTSyn hoặc mô hình lõi liên bảng

Chỉ thực hiện sau khi giai đoạn 3 có tín hiệu. Mục tiêu không phải đánh bại CTSyn về sinh không điều kiện, mà giữ chất lượng nền của CTSyn trong khi thêm khả năng điều khiển theo hợp đồng.

Trang công bố chính thức CTSyn hiện cung cấp bài báo và phụ lục nhưng không liệt kê mã nguồn chính thức. Do đó cần dự phòng hai đường: xin mã/checkpoint từ tác giả, hoặc tái tạo mô hình lõi liên bảng tối thiểu từ mô tả phương pháp. Không nên để rủi ro hạ tầng này chặn ablation theo từng bảng đang chạy.

## 14. Đánh giá khả năng thành bài ICDE

Hướng này phù hợp ICDE nếu nhấn mạnh:

- ngữ nghĩa dữ liệu;
- chất lượng dữ liệu;
- thực thi hợp đồng;
- dữ liệu không đồng nhất;
- mô hình nền tảng cho kỹ nghệ dữ liệu;
- chi phí và khả năng tái lập.

Phiên bản đủ mạnh cho bài đầy đủ cần:

- tái lập trung thực CuTS và CTSyn;
- ít nhất 5–8 bảng cho thử nghiệm nhỏ, sau đó mở rộng;
- hợp đồng có nguồn độc lập;
- đánh giá hiệu quả tác vụ, bao phủ nhóm hiếm và tính riêng tư;
- một mô hình dùng chung thật sự.

### 14.1 Đã đạt mức kết quả tốt nhất hiện nay chưa?

**Chưa.** Hiện tại chưa đủ cơ sở khoa học để dùng cụm “kết quả tốt nhất hiện nay”, vì:

| Thành phần | Bằng chứng hiện có | Phần còn thiếu |
| --- | --- | --- |
| Hiệu suất tạo nhóm hiếm | tám seed, tăng tốc 4,70–7,28 lần ở tầng hỗn hợp hiếm nhất | cần Adult/thêm miền, thời gian thực và ngân sách số lần gọi bộ khử nhiễu |
| Độ trung thực | có khoảng tin cậy; Shoppers thắng TV cục bộ nhưng Default vẫn kém lọc | cần cải thiện KS/TV nhất quán, chỉ số joint và hiệu quả tác vụ |
| Ràng buộc trong diffusion | có đối chứng xấp xỉ Harpoon/CTDF | cần chạy mã chính thức hoặc tái lập trung thực cùng ràng buộc |
| Tùy chỉnh theo yêu cầu | đã có biểu diễn hợp đồng; CuTS chính thức đạt ràng buộc 1,0 ở năm chương trình | cần giao thức chung giữa CuTS và phương pháp mới, nhiều seed huấn luyện và chi phí cho mỗi hợp đồng |
| Chuyển giao liên bảng | mới chỉ kiểm tra cùng ý tưởng trên nhiều bảng riêng biệt | chưa có một mô hình dùng chung; CTSyn chưa được tích hợp |
| Vai trò của LLM | biên dịch một lần, kiểm tra kiểu có kết quả tốt | chưa có bộ yêu cầu do người viết độc lập và đánh giá mù |

Giá trị hiện tại là đã xác định được một vùng bài toán có tín hiệu: yêu cầu hiếm khoảng dưới 1–2%, nơi lọc mẫu tốn từ hàng chục đến hơn một trăm lượt sinh và sinh có điều kiện có thể giảm chi phí nhiều lần. Đây là tiền đề tốt cho một dự án, nhưng chưa phải kết quả đủ để xếp hạng với CuTS, CTSyn, Harpoon hoặc CTDF.

### 14.2 Bài toán có đủ mạnh để thành dự án ICDE không?

Có, nếu phần chính là **chuyển giao đồng thời qua bảng và hợp đồng**, không phải chỉ “người dùng nhập điều kiện”. CuTS đã làm cho bài toán tùy chỉnh có tính chính danh; CTSyn đã làm cho mô hình sinh liên bảng có tính chính danh. Phần giao giữa hai hướng này tạo một câu hỏi tự nhiên, không cần tự dựng nhu cầu mới:

- CuTS phải tinh chỉnh theo chương trình và bảng;
- CTSyn hiểu lược đồ nhưng chưa thực thi hợp đồng người dùng chưa thấy;
- lọc mẫu suy giảm theo xác suất của nhóm cần sinh;
- sửa sau sinh có thể đạt hợp lệ nhưng không chứng minh đã lấy mẫu đúng từ phân phối có điều kiện.

Đóng góp có tầm ảnh hưởng chỉ xuất hiện nếu mô hình dùng chung thực sự giảm chi phí thích nghi trên bảng hoặc hợp đồng mới mà vẫn giữ phân phối. Nếu chỉ dừng ở TabDiff riêng từng bảng, dự án vẫn có thể thành một bài phân tích tốt, nhưng khó đủ mạnh cho bài ICDE đầy đủ.

Ước lượng:

- 2 tuần để chốt vùng độ hiếm và CuTS;
- 3–4 tuần cho thử nghiệm nhỏ liên bảng;
- 4–6 tuần cho tích hợp mô hình nền tảng, đối chứng và đánh giá rộng;
- tổng khoảng 9–13 tuần nếu môi trường CTSyn/CuTS ổn định.

## 15. Quyết định hiện tại

Dự án không còn được xem là một bài toán mới hoàn toàn. Nó là phần giao chưa được kiểm tra đầy đủ giữa:

- sinh dữ liệu bảng tùy chỉnh của CuTS;
- diffusion có ràng buộc của Harpoon/CTDF;
- tri thức ngữ nghĩa tiên nghiệm của SCTab-Diff;
- sinh dữ liệu bằng mô hình nền tảng liên bảng của CTSyn.

Câu hỏi trung tâm được đề xuất là:

> Một bộ sinh dữ liệu bảng dùng chung có thể thực hiện hợp đồng chưa thấy trên bảng chưa thấy, đặc biệt với nhóm hiếm, với chi phí thấp hơn lọc hoặc tinh chỉnh riêng và độ trung thực tốt hơn sửa cưỡng bức hay không?

Đây là câu hỏi đủ rõ để phản biện, có đối chứng mạnh, có điều kiện bác bỏ và không phụ thuộc vào việc gọi LLM ở mỗi bước diffusion.
