# Vision-Based Running Gait Analysis System

Ứng dụng phân tích dáng chạy từ video tải lên. MVP hướng tới video một người chạy quay ngang, xử lý offline trên máy cá nhân, sau đó xuất video có overlay, một số metric ước lượng, biểu đồ và dữ liệu JSON/CSV.

> **Lưu ý:** Đây là công cụ học tập/quan sát chuyển động, không phải công cụ chẩn đoán y tế. Chỉ số từ video đơn camera chịu ảnh hưởng của góc quay, phối cảnh, che khuất, FPS và sai số pose estimation.

## Tính năng dự kiến của MVP
- Đọc và kiểm tra video.
- Trích xuất pose bằng model pretrained chạy local.
- Đánh giá chất lượng landmark và cảnh báo clip không phù hợp.
- Ước lượng cadence/step time khi phát hiện event đủ tin cậy.
- Tính góc gối 2D và knee ROM theo định nghĩa đã ghi.
- Xuất video overlay, chart, JSON/CSV.
- Giao diện local bằng Streamlit.

Các metric có thể bị ẩn hoặc đánh dấu experimental nếu chưa đủ dữ liệu để tin cậy. Contact time không được xem là phép đo chính xác chỉ từ keypoint cổ chân.

## Yêu cầu môi trường
- Windows 64-bit, Python 3.11, suy luận local trên CPU.
- MediaPipe Pose Landmarker Full float16 v1 là model candidate đầu tiên.
- MP4, giới hạn đầu vào ứng dụng dự kiến 100 MB và 60 giây; đây không phải giới hạn của MediaPipe.
- Các dependency trực tiếp đã được pin trong `requirements.txt`.

Xem [hướng dẫn cài đặt và xác minh Phase 0](docs/PHASE0_SETUP.md) để tạo môi trường, tải model và kiểm tra SHA-256. Không đặt API key hoặc secret vào repository.

## Chạy ứng dụng
Khi `app.py` đã được triển khai:
```bash
streamlit run app.py
```

Mở địa chỉ local mà Streamlit in ra terminal, tải video lên và làm theo hướng dẫn trên màn hình.

## Chạy kiểm thử
```bash
pytest -q
```

Unit tests thông thường phải chạy offline và không tự tải model. Smoke test model/video có thể được cung cấp riêng trong `scripts/`.

## Đánh giá event detector
Khi đã có annotation thủ công:
```bash
python scripts/evaluate.py --help
```

Dùng schema và quy trình được mô tả trong `EVALUATION.md`. Không diễn giải nhãn thủ công từ video thành ground truth cấp phòng thí nghiệm.

## Đầu ra
Tên và cấu trúc cụ thể sẽ được chốt trong quá trình triển khai. Dự kiến gồm:
- video overlay `.mp4`
- `metrics.json`
- bảng dữ liệu `.csv`
- các biểu đồ
- cảnh báo chất lượng và metadata

Không nên commit video cá nhân, file output lớn hoặc model weights vào Git nếu chưa có quyết định rõ.

## Cấu trúc dự án
Xem `ARCHITECTURE.md` để biết luồng dữ liệu và trách nhiệm từng module; xem `PLAN.md` để biết các giai đoạn triển khai; xem `AGENTS.md` để biết quy tắc dành cho coding agent.

## Giới hạn đã biết
- Ước lượng pose có thể nhiễu hoặc mất landmark khi chân bị che.
- Góc 2D không phải phép đo 3D sinh cơ học.
- Foot-strike detection cần đánh giá trên dữ liệu được annotate.
- Cadence/step time phụ thuộc chất lượng event detection.
- Không có cơ sở để đưa ra chẩn đoán hoặc dự đoán chấn thương.

## Trạng thái dự án
Repository hiện có tài liệu đặc tả và nền tảng cài đặt Phase 0; pipeline phân tích và ứng dụng chưa được triển khai. Kiểm tra import/model/inference đã chạy trên Windows 11 x64 với Python 3.12.10; môi trường Python 3.11 chưa được chạy trên host này. Không coi tài liệu setup là bằng chứng ứng dụng end-to-end đã chạy.
