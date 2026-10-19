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

## Phase 1: trích xuất landmark từ video local

Phase 1 hiện nhận file MP4 local, kiểm tra khả năng giải mã/metadata và giới hạn ứng dụng (mặc định 100 MiB, 60 giây), sau đó xử lý tuần tự từng frame bằng MediaPipe Pose Landmarker chạy CPU. JSON được ghi vào `outputs/landmarks.json`; có thể yêu cầu thêm video MP4 đã vẽ skeleton. Pipeline không phụ thuộc Streamlit.

Sau khi cài dependency và tải model theo [docs/PHASE0_SETUP.md](docs/PHASE0_SETUP.md), chạy từ thư mục gốc:

```powershell
python scripts\process_video.py "data\samples\my_run.mp4" --model "models\pose_landmarker_full.task" --output "outputs\my_run_landmarks.json" --annotated-video "outputs\my_run_annotated.mp4"
```

`--annotated-video` là tùy chọn. Dùng `--max-bytes 0` hoặc `--max-seconds 0` để tắt từng giới hạn; các giới hạn này thuộc cấu hình ứng dụng và không phải giới hạn MediaPipe. Không có video mẫu nào được bao gồm trong repository.

JSON dùng `schema_version` `1.0`, metadata video, quy ước timestamp và danh sách `frames`. Mỗi frame chứa `frame_index`, `timestamp_seconds` (null nếu FPS nguồn không hợp lệ) và `poses`; mỗi pose có map `landmarks` theo tên 33 điểm MediaPipe. Mỗi landmark chứa `x`, `y`, `z` đã chuẩn hóa theo đầu ra model, cùng `visibility` và `presence` nếu API cung cấp. `z` là tọa độ tương đối của model, không phải độ sâu đo theo mét. Đây là dữ liệu pose thô; Phase 1 không tính gait event hay metric.

## Chạy kiểm thử
```bash
python -m unittest discover -s tests -v
```

Tests Phase 1 dùng fixture/mocks, chạy offline và không cần model hoặc video chạy thật. Manual smoke test riêng: chuẩn bị một MP4 ngắn có quyền sử dụng, tải và xác minh model theo Phase 0, rồi chạy lệnh CLI ở trên; xác nhận số frame/pose và mở overlay đầu ra nếu bật. Không có smoke test video thật nào được khẳng định trong repository này.

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
Phase 0 setup và Phase 1 video/landmark pipeline đã được triển khai. Phase 1 chưa xác thực chất lượng pose trên video chạy thật; Phase 2 trở đi (quality assessment, gait events, metrics, charts/reporting và Streamlit UI) chưa được triển khai. Không diễn giải độ tin cậy của model thành độ chính xác sinh cơ học hoặc kết luận y tế.
