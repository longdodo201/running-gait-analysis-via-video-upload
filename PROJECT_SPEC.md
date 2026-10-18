# PROJECT_SPEC.md — Đặc tả dự án

## 1. Tổng quan
**Tên:** Vision-Based Running Gait Analysis System  
**Loại:** Dự án cá nhân Computer Vision / video analytics  
**Mục tiêu:** Phân tích dáng chạy từ video tải lên, xử lý offline và trả về kết quả dễ hiểu cho người chạy bộ.

Đây là công cụ phân tích và giáo dục, không phải thiết bị y tế hay hệ thống chẩn đoán chấn thương.

## 2. Vấn đề cần giải quyết
Các ứng dụng kiểm tra realtime thường tập trung vào phản hồi tức thời. Dự án này tập trung vào **phân tích sau khi quay**: người dùng có thể xem lại toàn bộ clip, xem các chỉ số tổng hợp, quan sát những thời điểm có thể cần xem lại và tải báo cáo để so sánh giữa các lần chạy.

Giá trị dự kiến:
- Tóm tắt một số đặc trưng chuyển động quan sát được.
- Cho người dùng xem keypoints/skeleton và biểu đồ theo thời gian.
- Hỗ trợ so sánh các lần quay có điều kiện tương tự.
- Làm nền tảng học tập về computer vision, xử lý tín hiệu và đánh giá mô hình.

Không hứa hẹn rằng chỉ số tự động sẽ tự nó cải thiện hiệu suất hoặc giảm nguy cơ chấn thương.

## 3. Phạm vi MVP
### 3.1 Input
- Một video chạy bộ do người dùng tải lên.
- Ưu tiên góc quay ngang (side-on), toàn thân và cả hai chân nhìn thấy phần lớn thời gian.
- Video có FPS, độ dài và độ phân giải có thể đọc được.
- Hệ thống hiển thị thông tin video và cảnh báo khi chất lượng đầu vào không phù hợp.

### 3.2 Pipeline
1. Đọc và kiểm tra file video.
2. Trích xuất frame và timestamp.
3. Ước lượng pose bằng model pretrained chạy local, ưu tiên MediaPipe Pose Landmarker sau khi xác minh API/giấy phép.
4. Chuẩn hóa/lưu landmarks, visibility/presence hoặc confidence nếu model cung cấp.
5. Kiểm tra chất lượng landmark và xử lý chuỗi thời gian.
6. Ước lượng sự kiện chu kỳ dáng chạy nếu dữ liệu đủ chất lượng.
7. Tính metric có định nghĩa rõ ràng.
8. Tạo video overlay, biểu đồ và báo cáo.
9. Hiển thị kết quả trên giao diện web local và cho tải file đầu ra.

### 3.3 Output tối thiểu
- Video MP4 có overlay skeleton/landmarks khi khả thi.
- Tóm tắt chất lượng input và cảnh báo.
- Biểu đồ chuỗi thời gian cho một số tín hiệu/metric có ý nghĩa.
- `metrics.json` và/hoặc CSV.
- Báo cáo ngắn hoặc trang kết quả trong giao diện.

## 4. Metric dự kiến
Chỉ hiển thị metric khi có đủ dữ liệu và ghi rõ đơn vị, định nghĩa cùng giới hạn.

| Metric | Định nghĩa MVP | Điều kiện / giới hạn |
|---|---|---|
| Cadence | Số bước ước lượng mỗi phút, dựa trên sự kiện bước được phát hiện | Phụ thuộc độ đúng của event detector và quy ước đếm bước |
| Step time | Khoảng thời gian giữa hai sự kiện bước liên tiếp theo quy ước đã định nghĩa | Chỉ tính khi event sequence đủ tin cậy |
| Knee angle | Góc hình học tại gối từ hip-knee-ankle landmarks trong mỗi frame | Là góc 2D từ camera đơn, nhạy với góc quay và sai số keypoint |
| Knee ROM | Khoảng biến thiên góc gối trong một chu kỳ hoặc khoảng phân tích được định nghĩa | Chỉ so sánh khi quy trình phân đoạn nhất quán |
| Left/right comparison | So sánh metric hai bên khi cả hai bên quan sát được đủ tốt | Góc quay ngang có thể khiến một bên bị che khuất; không ép ra kết quả |
| Contact time | Chỉ là metric thử nghiệm nếu chưa được xác thực bằng phương pháp phù hợp | Không suy ra thời gian tiếp đất chính xác chỉ từ vị trí cổ chân |

Các metric được hỗ trợ có thể thay đổi theo kết quả kiểm thử. Nếu metric không đáng tin cậy, hiển thị “không đủ dữ liệu” thay vì một con số gây hiểu lầm.

## 5. Chất lượng video
Đánh giá heuristic, không phải xác suất chính xác:
- Người chạy có nằm trong khung hình không.
- Hông, gối, cổ chân/bàn chân có quan sát được không.
- Tỷ lệ frame có landmark hợp lệ.
- Mức độ che khuất và độ ổn định của keypoint.
- FPS, thời lượng, độ phân giải.
- Góc quay có phù hợp với phân tích 2D không.

Giao diện cần giải thích vì sao clip bị đánh giá chất lượng thấp và đưa ra hướng dẫn quay lại.

## 6. Yêu cầu chức năng
- **FR-01:** Upload video và kiểm tra định dạng/khả năng đọc.
- **FR-02:** Cho xem metadata video và thông báo lỗi đầu vào.
- **FR-03:** Chạy pose estimation offline bằng model pretrained đã cấu hình.
- **FR-04:** Lưu kết quả landmarks theo frame/timestamp.
- **FR-05:** Tính metric qua các hàm độc lập, có thể unit-test.
- **FR-06:** Tạo video overlay.
- **FR-07:** Hiển thị metric, biểu đồ và cảnh báo chất lượng.
- **FR-08:** Cho tải JSON/CSV và video kết quả.
- **FR-09:** Hiển thị các giới hạn và không đưa ra kết luận y tế.
- **FR-10:** Có chế độ chạy test và script đánh giá tách biệt với ứng dụng.

## 7. Yêu cầu phi chức năng
- Chạy được trên máy cá nhân phổ thông ở mức phù hợp với kích thước video.
- Unit tests không cần mạng.
- Có log lỗi đủ để debug nhưng không ghi nội dung nhạy cảm không cần thiết.
- Xử lý file tạm và thư mục output có quy ước rõ ràng.
- Tách biệt pipeline phân tích khỏi giao diện Streamlit.
- Kết quả có thể tái lập khi dùng cùng video, cấu hình và phiên bản model/thư viện.

## 8. Ngoài phạm vi MVP
- Realtime coaching.
- Huấn luyện pose-estimation model.
- Phân loại dáng chạy “đúng/sai” tổng quát.
- Chẩn đoán hoặc dự đoán chấn thương.
- Đo lực phản lực mặt đất, áp lực bàn chân hay các biến sinh cơ học cần cảm biến chuyên dụng.
- Khẳng định độ chính xác tương đương motion capture.
- So sánh MediaPipe và ViTPose như một yêu cầu bắt buộc của MVP.

## 9. Giả định và quyết định còn mở
- Video mặc định là một người chạy, quay ngang.
- Phiên bản đầu tiên dùng model pretrained local; MediaPipe là lựa chọn ưu tiên cần xác minh khả năng/API/giấy phép tại thời điểm triển khai.
- Đơn vị thời gian lấy từ timestamp/FPS video; phải xử lý variable frame rate nếu hỗ trợ.
- Cần chủ dự án chốt giấy phép repo, kích thước video tối đa và nền tảng OS mục tiêu trước khi phát hành.
