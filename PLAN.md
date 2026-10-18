# PLAN.md — Lộ trình triển khai

## Nguyên tắc
Triển khai theo lát cắt end-to-end nhỏ, kiểm thử từng phần. Không bắt đầu bằng dashboard phức tạp hoặc nhiều metric chưa được xác thực. Mỗi giai đoạn cần có tiêu chí hoàn tất cụ thể.

## Phase 0 — Khảo sát và chốt quyết định
**Việc làm**
- Xác minh môi trường Python/OS và khả năng chạy model local.
- Kiểm tra API, phiên bản và giấy phép của MediaPipe Pose Landmarker cùng model asset.
- Chốt giới hạn video, định dạng được hỗ trợ và cấu hình mặc định.
- Tạo `.gitignore`, dependency file và cấu trúc thư mục.

**Hoàn tất khi**
- Có quyết định công nghệ được ghi lại.
- Có lệnh cài đặt môi trường tái lập.
- Không phụ thuộc API key hoặc dịch vụ trả phí ngoài ý muốn.

## Phase 1 — Video I/O và dữ liệu landmarks
**Việc làm**
- Đọc video, metadata, frame và timestamp.
- Tạo adapter cho pose estimator.
- Lưu kết quả landmark theo schema thống nhất.
- Thêm một script CLI tối thiểu để chạy một video.

**Hoàn tất khi**
- Đọc được video mẫu hợp lệ.
- Lỗi file/codec được báo rõ.
- Có output landmarks tái sử dụng được.
- Test schema và video I/O cơ bản chạy được.

## Phase 2 — Quality assessment và hình học
**Việc làm**
- Kiểm tra độ bao phủ landmark, missing/low-visibility frames.
- Viết hàm góc 2D từ ba điểm.
- Tính knee angle theo frame và đánh dấu frame không hợp lệ.
- Viết tests cho các trường hợp chuẩn, biên và dữ liệu thiếu.

**Hoàn tất khi**
- Công thức góc được kiểm tra bằng ví dụ toán học.
- Metric không được tính từ landmark không đủ chất lượng.
- Quality warnings có thể giải thích.

## Phase 3 — Signal processing và gait events
**Việc làm**
- Chọn trajectory/feature phù hợp cho clip quay ngang.
- Thử smoothing và xử lý missing segments.
- Xây dựng bộ phát hiện candidate foot-strike/step events với ngưỡng và khoảng cách tối thiểu.
- Xuất event timestamp, side, confidence/flags.
- Xây dựng định dạng annotation thủ công.

**Hoàn tất khi**
- Có synthetic tests cho timestamp, khoảng cách sự kiện và edge cases.
- Có thể chạy evaluation trên video đã annotate.
- Event detector không tuyên bố validated nếu chưa có đánh giá thực nghiệm.

## Phase 4 — Metric và đánh giá ban đầu
**Việc làm**
- Tính cadence và step time từ event sequence hợp lệ.
- Tính knee angle và ROM với định nghĩa chu kỳ rõ ràng.
- Đánh giá event detector và metric trên tập video đã annotate.
- Ghi rõ giới hạn ground truth.

**Hoàn tất khi**
- Mỗi metric có định nghĩa, đơn vị, điều kiện xuất kết quả.
- Có báo cáo kết quả đánh giá thật hoặc nêu rõ chưa có dữ liệu đủ.
- Metric không đáng tin cậy được đánh dấu experimental/unavailable.

## Phase 5 — Visualization và report
**Việc làm**
- Render skeleton/landmarks và thông tin frame/event.
- Tạo plot tín hiệu và metric theo thời gian.
- Xuất JSON/CSV với metadata, warning và phương pháp tính.
- Kiểm thử artifact đầu ra.

**Hoàn tất khi**
- Video overlay mở được.
- JSON/CSV hợp lệ và nhất quán.
- Các plot không che giấu khoảng dữ liệu thiếu.

## Phase 6 — Streamlit UI
**Việc làm**
- Upload video.
- Hiển thị metadata, quality summary và cảnh báo.
- Cho chạy phân tích, hiển thị tiến độ ở mức phù hợp.
- Trình bày video kết quả, metric, chart và nút tải output.
- Thông báo rõ khi clip không phù hợp.

**Hoàn tất khi**
- Người dùng có thể đi từ upload đến output trên máy local.
- UI gọi pipeline module, không chứa logic tính toán.
- Luồng lỗi hiển thị thông báo dễ hiểu.

## Phase 7 — Tích hợp, tài liệu và phát hành MVP
**Việc làm**
- Chạy test suite và sửa lỗi.
- Viết README, methodology, evaluation.
- Thử nghiệm end-to-end với video có quyền sử dụng.
- Ghi lại môi trường và hạn chế đã biết.

**Hoàn tất khi**
- Cài đặt và chạy theo README được xác minh.
- Unit/integration tests đã chạy và kết quả được ghi đúng.
- Có một quy trình demo tái lập.
- Không tuyên bố benchmark chưa thực hiện.

## Phase 8 — Nâng cao (không bắt buộc)
Chỉ xem xét sau khi MVP ổn định:
- So sánh MediaPipe với pose estimator khác.
- Cải thiện event detection bằng dữ liệu annotate lớn hơn.
- Hỗ trợ góc quay khác hoặc nhiều người.
- So sánh tiến bộ giữa các buổi quay có điều kiện tương tự.
- Thêm classifier chỉ khi có ground truth đáng tin cậy và câu hỏi nghiên cứu rõ.

## Rủi ro chính
1. **Pose jitter/occlusion:** làm sai trajectory và góc khớp.
2. **Foot-strike detection:** không thể mặc định rằng local minimum là foot strike.
3. **Video khác nhau:** góc máy, FPS, khoảng cách, quần áo và ánh sáng ảnh hưởng kết quả.
4. **Metric quá tự tin:** con số trông chính xác nhưng không có ground truth xác nhận.
5. **Scope creep:** thêm UI/metric trước khi pipeline cơ bản được đánh giá.

Giảm thiểu bằng cách làm MVP nhỏ, giữ warning, đánh giá bằng annotations và ghi rõ giới hạn.
