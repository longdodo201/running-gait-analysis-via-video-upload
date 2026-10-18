# AGENTS.md — Quy tắc làm việc cho AI Coding Agent

## 1. Mục tiêu làm việc
Xây dựng một MVP có thể chạy được cho **Vision-Based Running Gait Analysis System**: người dùng tải lên video chạy bộ quay ngang, hệ thống phân tích offline, rồi xuất video có chú thích, các chỉ số dáng chạy, biểu đồ và dữ liệu có cấu trúc.

Ưu tiên theo thứ tự:
1. Tính đúng đắn và tính trung thực của phân tích.
2. Pipeline end-to-end chạy được trên máy cá nhân.
3. Khả năng kiểm thử, tái lập và giải thích kết quả.
4. Giao diện và trải nghiệm người dùng.
5. Tính năng nâng cao.

## 2. Nguyên tắc kỹ thuật
- Dùng Python 3.11 làm phiên bản mục tiêu ban đầu.
- Ưu tiên suy luận cục bộ với model pose pretrained; mặc định khảo sát MediaPipe Pose Landmarker trước.
- Không huấn luyện model mới trong MVP. Không thêm dịch vụ/API trả phí hoặc yêu cầu API key nếu chưa có phê duyệt rõ ràng.
- Không tải model hoặc gọi mạng ngầm trong unit tests.
- Giữ các module nhỏ, một trách nhiệm rõ ràng; tránh gom toàn bộ pipeline vào `app.py`.
- Dùng type hints cho các hàm chính, docstring cho logic không hiển nhiên, tên biến thể hiện ý nghĩa.
- Định nghĩa rõ đơn vị, hệ tọa độ, kiểu dữ liệu và quy ước trái/phải.
- Xử lý lỗi có thông báo hữu ích; không âm thầm bỏ qua lỗi đầu vào hoặc lỗi inference.
- Không hard-code đường dẫn máy cá nhân, secret, API key hay dữ liệu người dùng.
- Chỉ thêm dependency khi có lý do; khóa phiên bản tương thích trong `requirements.txt` hoặc tệp lock phù hợp.

## 3. Tính trung thực về kết quả
- Không tuyên bố hệ thống chẩn đoán bệnh, dự đoán chấn thương hoặc xác định dáng chạy “bình thường/bất thường” nếu chưa có nghiên cứu và dữ liệu đánh giá phù hợp.
- Phân biệt rõ:
  - **measurement/estimate**: chỉ số được tính hoặc ước lượng từ video;
  - **quality flag**: cảnh báo chất lượng input;
  - **validated metric**: chỉ số đã được đánh giá với ground truth phù hợp.
- Không coi confidence của pose model là độ chính xác sinh cơ học.
- Không tuyên bố contact time chính xác chỉ dựa trên keypoint cổ chân. Nếu chưa có phương pháp và đánh giá đủ tốt, bỏ qua hoặc ghi rõ là experimental estimate.
- Không bịa ground truth, kết quả benchmark, test pass, ảnh đầu ra hoặc số liệu đánh giá.
- Nêu rõ giới hạn của video đơn camera 2D: góc quay, phối cảnh, che khuất, FPS và sai số keypoint có thể ảnh hưởng kết quả.

## 4. Quy trình triển khai
1. Đọc `PROJECT_SPEC.md`, `ARCHITECTURE.md`, `PLAN.md`, `EVALUATION.md` và `README.md` trước khi sửa code.
2. Kiểm tra API thực tế của phiên bản thư viện được chọn và giấy phép model.
3. Triển khai theo từng giai đoạn trong `PLAN.md`; sau mỗi giai đoạn, chạy test liên quan.
4. Không viết toàn bộ dự án trong một file hoặc tạo stub giả rồi tuyên bố hoàn tất.
5. Nếu không thể thực thi code trong môi trường hiện tại, nói rõ phần nào chưa chạy/được xác minh.
6. Cập nhật tài liệu khi thay đổi kiến trúc, metric, định dạng output hoặc cách chạy.

## 5. Kiểm thử bắt buộc
- Viết unit tests cho hình học góc, xử lý chuỗi thời gian, phát hiện sự kiện và tính metric.
- Dùng dữ liệu tổng hợp để kiểm tra toán học và edge cases; không dùng dữ liệu tổng hợp để tuyên bố độ chính xác thực tế.
- Kiểm thử lỗi video rỗng, codec không hỗ trợ, FPS bất thường, thiếu landmark, confidence thấp, clip quá ngắn và người bị che khuất.
- Unit tests thông thường phải chạy offline, không tải model và không cần Internet.
- Chỉ ghi “tests passed” nếu thực sự chạy và quan sát được kết quả.

## 6. Quy tắc Git và dữ liệu
- Không commit video cá nhân, file output lớn, cache, secrets hoặc model weights nếu chưa được quyết định rõ.
- Có `.gitignore` cho môi trường ảo, cache, `outputs/`, file tạm và secrets.
- Thêm dữ liệu mẫu chỉ khi có quyền sử dụng và giấy phép rõ ràng.
- Không tự tạo giấy phép dự án hoặc dữ liệu bên thứ ba; đánh dấu mục cần chủ dự án quyết định.

## 7. Definition of Done
Một phần việc chỉ hoàn tất khi:
- Có code thực thi được, không chỉ là placeholder.
- Có test phù hợp và kết quả test trung thực.
- Có thông báo lỗi và giới hạn đầu vào cần thiết.
- Tài liệu liên quan được cập nhật.
- Kết quả không được mô tả vượt quá bằng chứng hiện có.
