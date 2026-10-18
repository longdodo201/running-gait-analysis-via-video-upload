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
- Python 3.11 (mục tiêu ban đầu).
- FFmpeg nếu cần cho codec/chuyển đổi video.
- Model asset được tải theo hướng dẫn riêng sau khi xác minh phiên bản, nguồn và giấy phép.
- Bộ nhớ/CPU/GPU tùy kích thước video và model.

Phiên bản thư viện chính xác sẽ được khóa sau Phase 0 trong `PLAN.md`; không cài đặt bằng một danh sách version chưa được kiểm tra.

## Cài đặt (sau khi repository được triển khai)
```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.venv\\Scripts\\Activate.ps1
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Sau khi `requirements.txt` đã được xác minh:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Tải/đặt model asset theo hướng dẫn do dự án cung cấp. Không đặt API key hoặc secret vào repository.

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
Đây là tài liệu nền tảng/specification. Chỉ đánh dấu tính năng “đã triển khai” sau khi code thực sự tồn tại và đã được kiểm thử. Không coi README này là bằng chứng ứng dụng đã chạy.
