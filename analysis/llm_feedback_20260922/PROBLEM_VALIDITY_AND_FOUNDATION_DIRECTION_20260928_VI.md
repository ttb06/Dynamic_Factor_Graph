# Đánh giá tính chính đáng của bài toán và hướng mô hình nền tảng

> **Bản đề xuất hợp nhất mới hơn:** xem `DE_XUAT_NGHIEN_CUU_HOP_NHAT_20260928_VI.md`. Bản mới đã bổ sung CuTS, kết quả tập hàng áp dụng, đối chứng lọc mẫu và các job sweep theo độ hiếm.

**Ngày chốt phân tích:** 28/09/2026  
**Phạm vi:** đánh giá bài toán trước khi mở rộng thêm thực nghiệm; không dùng kết quả đang chờ để thay đổi kết luận.

## 1. Kết luận ngắn

Hướng hiện tại có thể gọi chính xác là **sinh dữ liệu bảng theo yêu cầu người dùng**, nhưng cách gọi này còn quá rộng. Phần đã được hiện thực và kiểm chứng là:

> Biên dịch yêu cầu chưa thấy thành một hợp đồng có kiểu; dùng hợp đồng làm điều kiện cho diffusion sinh bảng mixed-type; sau đó dùng bộ kiểm chứng độc lập để bảo đảm phần ràng buộc còn vi phạm.

Đây **chưa phải mô hình nền tảng sinh bảng**: mỗi dataset hiện vẫn có checkpoint hoặc adapter riêng. Các thử nghiệm trên Adult, Default và Shoppers mới chứng minh hiện tượng lặp lại trên nhiều bảng, chưa chứng minh một mô hình chung chuyển giao sang bảng mới.

Ba lo ngại về số lượng đối chứng, benchmark tự xây và mức cấp thiết của bài toán đều hợp lý. Nếu giữ cách mô tả “người dùng nhập điều kiện bất kỳ rồi mô hình sinh dữ liệu”, dự án chưa đủ mạnh để đặt novelty vào bản thân bài toán. Tuy nhiên, không cần trình bày đây là một bài toán hoàn toàn mới. Có thể đặt nó trong hai lĩnh vực đã có:

1. sinh dữ liệu bảng có điều kiện;
2. sinh dữ liệu bảng có ràng buộc và bảo đảm tính toàn vẹn.

Khoảng trống nên thu hẹp thành:

> **Sinh dữ liệu bảng liên bảng dưới dịch chuyển đồng thời của schema và hợp đồng:** một mô hình đã tiền huấn luyện phải tiếp nhận một bảng mới và một chính sách/yêu cầu chưa thấy, sinh đúng phân phối con được yêu cầu, đồng thời giảm chi phí loại bỏ hoặc sửa mẫu và cung cấp chứng nhận bằng bộ kiểm chứng độc lập.

Hướng này có tiềm năng tốt hơn về mặt nền tảng và liên bảng. Nó cũng khó hơn đáng kể. Kết quả hiện tại là bằng chứng khả thi cho mô-đun điều kiện, chưa phải bằng chứng cho claim liên bảng.

## 2. Cần tách ba bài toán đang bị gộp chung

| Bài toán | Ví dụ | Phân phối đích | Đối chứng đơn giản nhưng mạnh |
| --- | --- | --- | --- |
| Sinh cohort theo truy vấn | `age >= 50 AND education = Masters` | Phân phối thật có điều kiện `p_data(x | C)` | lọc/rejection sampling, best-of-N, mô hình conditional |
| Bảo đảm quy tắc toàn vẹn | `count = 0 -> duration = 0` | Phân phối gốc sau khi loại vi phạm quy tắc hợp lệ | phép chiếu, CTDF, Harpoon, sửa sau sinh |
| Thực thi chính sách ở mức tập dữ liệu | tỷ lệ nhóm, quota, fairness, tương quan đích | phân phối toàn bảng thỏa thống kê toàn cục | tối ưu/phép chiếu theo batch, reweighting, iterative proportional fitting |

Các thử nghiệm gần đây chủ yếu giải bài toán thứ nhất, rồi dùng certifier của bài toán thứ hai. Điều này hữu ích, nhưng không được gọi chung là “dynamic semantic constraints” nếu chưa nói rõ constraint động theo nghĩa nào.

Rủi ro lớn nhất của bài toán cohort là: nếu điều kiện có support đủ lớn, lấy nhiều mẫu unconditional rồi lọc cho phân phối có điều kiện rất đúng mà không cần phương pháp mới. Khi đó đóng góp chỉ tồn tại ở vùng điều kiện hiếm, điều kiện kết hợp, ngân sách lấy mẫu hạn chế, hoặc cần chuyển giao sang yêu cầu chưa thấy.

Rủi ro lớn nhất của bài toán integrity là: nếu quy tắc thực thi được và thay đổi ít cell, sửa sau sinh có thể đạt 100% hợp lệ với chi phí thấp. Khi đó mô hình conditional chỉ có giá trị nếu giảm rõ biến dạng, số cell phải sửa, mất nhóm hiếm hoặc chi phí chạy.

Vì vậy bài chính không nên gộp cả ba loại thành một điểm số. Cần chọn một mục tiêu chính, còn hai loại kia là mở rộng.

## 3. Bằng chứng hiện có thực sự nói được gì

### 3.1 Đã có tín hiệu tốt

- Biểu diễn hợp đồng có cấu trúc phản ứng với threshold và phép kết hợp chưa thấy; mã yêu cầu rời rạc không làm được điều này.
- Hiệu ứng lặp lại trên Adult, Default và Shoppers, nên không phải hiện tượng riêng của một bảng.
- Trên Shoppers mixed composition, điều kiện có cấu trúc cải thiện đồng thời mức thỏa yêu cầu và fidelity; sau cùng một certifier, nó cần ít sửa hơn khởi đầu unconditional.
- Biên dịch một lần sang DSL rồi thực thi xác định đáng tin và rẻ hơn nhiều so với hỏi LLM cho từng hàng.
- Cơ chế active-set có bằng chứng nhân quả sơ bộ: can thiệp đúng hàng giúp giữ các hàng không thuộc phạm vi gần baseline hơn can thiệp toàn cục.

### 3.2 Chưa được chứng minh

- Chưa có mô hình duy nhất học qua nhiều bảng; vì vậy chưa có foundation transfer.
- Chưa có benchmark chính sách lấy độc lập từ tài liệu miền hoặc người dùng bên ngoài.
- Chưa có tái lập faithful của CTDF, Harpoon, CTSyn, SCTab-Diff và LCATS trên cùng giao thức.
- Chưa chứng minh tốt hơn rejection sampling trong vùng support thấp với ngân sách được chuẩn hóa.
- Chưa chứng minh utility, privacy và rare-mode coverage được giữ khi sinh cohort.
- Chưa có lý do khoa học để gọi LLM ở nhiều diffusion step. Kết quả hiện tại ngược lại ủng hộ compile-once.

Do đó, kết luận đúng hiện tại là **bằng chứng khả thi mạnh**, chưa phải kết quả tốt nhất hiện nay.

## 4. Lo ngại về baseline: đúng, nhưng có thể xử lý

Vấn đề không phải thiếu công trình để so sánh; vấn đề là phải ánh xạ đúng từng đối chứng vào cùng nhiệm vụ.

### Nhóm bắt buộc cho bài toán theo yêu cầu

1. unconditional generator + rejection sampling;
2. best-of-N với cùng ngân sách denoiser;
3. mô hình conditional theo cột hoặc category đã biết;
4. huấn luyện/fine-tune riêng cho từng hợp đồng như một oracle đắt tiền;
5. request-ID condition để kiểm tra lợi ích của cấu trúc hợp đồng.

### Nhóm bắt buộc cho bài toán có ràng buộc

1. phép chiếu ở bước cuối;
2. sửa sau sinh kiểu LCATS/Zhao;
3. Harpoon với guidance khả vi;
4. CTDF với feasibility operator mixed-type;
5. prior tĩnh trong denoising kiểu SCTab-Diff.

### Nhóm bắt buộc nếu claim mô hình nền tảng

1. CTSyn chỉ condition theo schema;
2. CTSyn + lọc/sửa sau sinh;
3. CTSyn fine-tune theo từng yêu cầu;
4. TabDiff riêng từng bảng;
5. mô hình chung có schema nhưng không có hợp đồng.

Với danh sách này, dự án có đủ đối chứng để định vị. Điểm yếu hiện tại là nhiều baseline mới là proxy, không phải thiếu baseline trên thị trường.

## 5. Lo ngại benchmark tự đề xuất: đây là rủi ro lớn nhất

Benchmark tự xây không tự động làm công trình yếu, nhưng một benchmark chỉ gồm các rule do chính tác giả tạo ra rất dễ bị xem là phục vụ phương pháp. Cách giảm rủi ro là không xây lại từ số không.

### Nền benchmark nên tái sử dụng

- dùng dataset, split và metric của CTSyn cho đánh giá liên bảng;
- dùng dataset/rule families của Harpoon, CTDF, SCTab-Diff và LCATS cho constrained synthesis;
- giữ metric fidelity, utility và diversity chuẩn của TabDiff/CTSyn;
- thêm một trục đánh giá mới là hợp đồng chưa thấy, thay vì thay toàn bộ benchmark.

### Nguồn hợp đồng nên có xuất xứ độc lập

1. điều kiện lọc có thể thực thi, sinh tự động nhưng khóa trước khi huấn luyện;
2. quy tắc từ data dictionary, metadata hoặc tài liệu chính thức của dataset;
3. quy tắc đã dùng trong related work;
4. một tập nhỏ yêu cầu do người không phát triển phương pháp viết;
5. paraphrase và composition giữ lại hoàn toàn cho test.

### Giao thức chống thiên lệch

- khóa catalog, split và metric trước khi chỉnh phương pháp;
- tách bảng đã thấy/chưa thấy và hợp đồng đã thấy/chưa thấy;
- có cùng một hàng dưới nhiều yêu cầu khác nhau;
- báo cáo cả điều kiện có support cao, thấp và ngoài support;
- không dùng chính LLM/compiler đã tạo rule làm thước đo duy nhất;
- công bố trường hợp không áp dụng, ngoại lệ và mâu thuẫn, không chỉ vi phạm.

Nếu làm như vậy, contribution không phải “chúng tôi tự tạo một benchmark mới”, mà là “chúng tôi mở rộng giao thức chuẩn để đo một failure mode chưa được tách ra”. Cách này dễ bảo vệ hơn.

## 6. Bài toán có đủ cấp thiết không?

### Cách mô tả yếu

> Cho phép người dùng nhập điều kiện tùy ý để sinh dữ liệu.

Cách này quá gần conditional generation thông thường, không xác định phân phối đích và không giải thích vì sao lọc hoặc sửa là chưa đủ.

### Cách mô tả có thể bảo vệ

> Một synthesizer được tái sử dụng qua nhiều bảng phải đáp ứng các yêu cầu thay đổi theo tác vụ mà không huấn luyện lại toàn bộ mô hình. Lọc mẫu trở nên tốn kém với cohort hiếm; phép chiếu và sửa có thể phá quan hệ giữa các cột hoặc xóa ngoại lệ hợp lệ. Vấn đề là sinh đúng phân phối có điều kiện dưới dịch chuyển schema và chính sách, với bộ kiểm chứng độc lập và chi phí hữu hạn.

Mức cấp thiết chỉ thuyết phục nếu thực nghiệm chứng minh đủ ba điểm:

1. rejection/best-of-N thất bại về chi phí ở cohort hiếm;
2. projection/repair đạt validity nhưng gây biến dạng hoặc mất mode;
3. fine-tune riêng từng policy quá chậm hoặc không chuyển giao sang policy mới.

Nếu không tìm được vùng mà cả ba hiện tượng này cùng tồn tại, bài toán không đủ cấp thiết để làm contribution chính.

## 7. Hướng mô hình nền tảng/liên bảng

CTSyn đã là một diffusion foundation model cho sinh dữ liệu liên bảng: nó học không gian latent chung, condition trên schema và tiền huấn luyện qua nhiều bảng. Vì vậy không thể claim “mô hình nền tảng đầu tiên cho sinh dữ liệu bảng”. Khoảng trống hợp lý hơn là:

> CTSyn thích nghi theo schema; các phương pháp constraint nhận rule executable hoặc sửa trên từng bảng. Chưa có đánh giá thuyết phục cho một synthesizer chung phải chuyển giao đồng thời sang **schema mới và hợp đồng người dùng mới**, đặc biệt với composition, exception và support thấp.

### Định nghĩa nhiệm vụ đề xuất

Đầu vào gồm schema `S`, một mẫu dữ liệu hoặc statistics được phép từ bảng `D`, và hợp đồng `C`. Mô hình sinh `X ~ p(x | D, S, C)` sao cho:

1. bộ kiểm chứng độc lập xác nhận `C`;
2. phân phối của `X` gần cohort thật thỏa `C` khi cohort đó tồn tại;
3. các cột không bị yêu cầu vẫn giữ dependency và utility;
4. chi phí tốt hơn lọc, sửa hoặc fine-tune riêng;
5. kết quả giữ được khi `S` hoặc `C` chưa xuất hiện lúc huấn luyện.

### Điều kiện để được gọi là mô hình nền tảng

- có tham số dùng chung qua nhiều bảng;
- tiền huấn luyện trên một tập bảng đủ rộng;
- đánh giá leave-one-table-out hoặc trên tập downstream không trùng;
- có ít-shot hoặc zero-shot adaptation;
- kiểm soát trùng lặp tên cột và metadata để tránh chuyển giao giả.

Chạy cùng một phương pháp trên ba dataset riêng không đáp ứng các điều kiện này.

## 8. Ba lựa chọn dự án

| Lựa chọn | Claim trung tâm | Ưu điểm | Rủi ro | Đánh giá |
| --- | --- | --- | --- | --- |
| A. Contract-conditioned TabDiff riêng từng bảng | chuyển giao sang threshold/composition chưa thấy, giảm correction burden | gần code hiện tại, hoàn thành nhanh | novelty vừa; rejection/repair mạnh | phù hợp làm bản an toàn hoặc EAB |
| B. Policy-conditioned cross-table synthesizer | chuyển giao đồng thời qua bảng và hợp đồng | khoảng trống rõ hơn, gắn mô hình nền tảng | cần CTSyn/shared pretraining và benchmark rộng | hướng chính được khuyến nghị |
| C. LLM feedback ở nhiều denoising step | dùng preview để cập nhật constraint/action | gần ý tưởng ban đầu | novelty đông, chi phí lớn, kết quả hiện tại không ủng hộ | không nên làm claim chính |

Khuyến nghị là dùng A làm pilot và bộ ablation cho B, không bỏ kết quả hiện có. C chỉ giữ như một nhánh kiểm tra xem preview có thông tin tăng thêm hay không.

## 9. Claim có thể nhắm tới

Không nên claim:

- lần đầu dùng LLM trong diffusion;
- lần đầu đưa constraint vào tabular diffusion;
- lần đầu có foundation model sinh bảng;
- bảo đảm semantic validity cho yêu cầu tùy ý;
- tốt nhất hiện nay khi chưa có benchmark chung.

Có thể nhắm tới, nếu thực nghiệm hoàn thiện:

1. giao thức đầu tiên hoặc đầu tiên được kiểm soát chặt cho **joint schema–contract shift** trong cross-table synthesis;
2. mô-đun hợp đồng có kiểu cho phép một generator chung tổng hợp threshold, category, conjunction và exception chưa thấy;
3. conditional warm start giảm chi phí lọc/sửa và biến dạng so với cùng certifier;
4. phân tích khi nào contract conditioning có lợi theo support, dependency strength và loại schema;
5. cơ chế từ chối/certification làm cho LLM compiler an toàn hơn direct row judgement.

Claim (1) cần literature audit rộng hơn trước khi dùng từ “đầu tiên”. Các claim còn lại có thể kiểm chứng trực tiếp.

## 10. Tiêu chí quyết định trước khi đầu tư lớn

Chỉ nâng thành dự án mô hình nền tảng nếu một pilot liên bảng đạt đồng thời:

1. một mô hình dùng chung tốt hơn CTSyn schema-only + rejection/repair trên bảng chưa thấy;
2. lợi ích tồn tại trên hợp đồng chưa thấy, không chỉ paraphrase;
3. cải thiện trên ít nhất ba miền dữ liệu và không phụ thuộc tên cột trùng nhau;
4. giảm ít nhất một đại lượng có ý nghĩa: số trajectory, số cell sửa, distortion hoặc thời gian fine-tune;
5. utility và rare-mode coverage không giảm đáng kể;
6. oracle DSL và template parser vẫn được báo cáo, để lợi ích của LLM không bị phóng đại.

Nếu thất bại ở (1) hoặc (2), quay về lựa chọn A. Nếu repair/rejection Pareto-dominate ở hầu hết điều kiện, chuyển dự án sang benchmark/analysis về giới hạn của controllable synthesis thay vì cố tạo method paper.

## 11. Ước lượng thực tế

- **2 tuần đầu:** tái sử dụng giao thức CTSyn, khóa taxomony hợp đồng và chạy baseline tối thiểu trên 5–8 bảng. Đây là phép thử quyết định, chưa cần tiền huấn luyện lớn.
- **4–6 tuần tiếp:** tích hợp contract encoder vào CTSyn hoặc một backbone chia sẻ nhẹ hơn; đánh giá bảng và hợp đồng chưa thấy.
- **3–5 tuần cuối:** faithful baselines, utility/privacy/rare modes, ablation, thống kê và viết bài.

Tổng tối thiểu cho hướng foundation nghiêm túc là khoảng **9–13 tuần** nếu code CTSyn chạy ổn. Hướng riêng từng bảng có thể đóng trong **4–6 tuần**, nhưng trần novelty thấp hơn.

## 12. Quyết định cuối cùng

Các lo ngại của người nghiên cứu là đúng và cần xử lý trước khi mở rộng thí nghiệm. Dự án hiện tại **đủ tốt để làm pilot**, nhưng chưa đủ để khẳng định một bài toán mới độc lập hoặc trạng thái tốt nhất hiện nay.

Hướng đáng đầu tư nhất không phải “user condition generation” chung chung, mà là:

> **Sinh dữ liệu bảng theo hợp đồng dưới dịch chuyển schema và chính sách, dựa trên một backbone liên bảng, với kiểm chứng độc lập và so sánh chi phí công bằng với lọc, guidance và repair.**

Framing này bám vào hai bài toán đã tồn tại, dùng CTSyn làm mốc foundation, dùng Harpoon/CTDF/LCATS làm mốc constraint, và chỉ bổ sung một khoảng trống giao nhau có thể đo được. Nhờ đó dự án không phải dựa toàn bộ novelty vào một benchmark do chính mình đề xuất.
