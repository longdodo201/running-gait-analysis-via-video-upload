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

Unit/integration tests dùng fixture/mocks, chạy offline và không cần model hoặc video thật. Smoke test local tùy chọn: chuẩn bị MP4 có quyền sử dụng, tải và xác minh model theo Phase 0, rồi chạy CLI hoặc UI; kiểm tra video overlay cùng các artifact. Test tự động không chạy inference thật.

## Đánh giá event detector
Quy trình đánh giá candidate và annotation thủ công ban đầu được mô tả trong [docs/PHASE2_5_EVENT_EVALUATION.md](docs/PHASE2_5_EVENT_EVALUATION.md). Ground Truth event-level hiện có trong `outputs/foot_strike_ground_truth.csv`; so sánh hai thuật toán, góc gối tại các frame đã gán nhãn và cadence được mô tả trong [docs/PHASE3_GROUND_TRUTH_EVALUATION.md](docs/PHASE3_GROUND_TRUTH_EVALUATION.md). Nhãn video là reference quan sát, không phải ground truth cấp phòng thí nghiệm.

## Ứng dụng local sau khi phân tích

Từ thư mục gốc, cài dependencies theo [docs/PHASE0_SETUP.md](docs/PHASE0_SETUP.md), tải model và khởi chạy:

```powershell
streamlit run app.py
```

Chọn một MP4 side-view rồi nhấn **Analyze video**. Quá trình chạy cục bộ; giới hạn mặc định là 100 MiB và 60 giây. Mỗi lượt chạy được lưu trong thư mục mới dưới `outputs/streamlit_runs/`; file upload tạm được xóa sau phân tích. Cần có `models/pose_landmarker_full.task` tại repository root.

UI hiển thị skeleton overlay, biểu đồ góc gối 2D và độ bao phủ dữ liệu, đồng thời cho tải các artifact. Với upload tên `running_gait_sideview.mp4`, app dùng Ground Truth CSV hiện có nếu frame/timestamp khớp landmarks đã trích xuất. Dashboard khi đó hiển thị kết quả event evaluation, knee angle tại Ground Truth events và cadence; heel-event predictions vẫn được giữ riêng và không được gọi là confirmed foot-strikes. Video khác không dùng nhãn của clip này.

## Đầu ra
Mỗi lượt phân tích trong UI tạo một thư mục mới với video overlay gốc `.mp4`, bản H.264 chuyển mã cục bộ để phát trong browser khi encoder sẵn có, landmark JSON, CSV góc gối thô và đã làm mượt, SVG biểu đồ góc gối, cùng CSV/SVG candidate heel-event. Khi đúng video có Ground Truth, pipeline bổ sung CSV/JSON evaluation, match event, knee-angle-at-event, cadence CSV/SVG, summary JSON và báo cáo Markdown trong cùng thư mục. Nếu chuyển mã thất bại, video gốc vẫn được giữ để tải xuống; khả năng phát MP4V phụ thuộc trình duyệt. Cadence lấy từ nhãn Ground Truth và là số bước quan sát được trên thời lượng clip; không suy ra từ candidate.

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
Các module Phase 1–3 và UI MVP local đã được triển khai trong working tree hiện tại. Có artifact phân tích và manual review cục bộ, nhưng điều đó không xác nhận độ chính xác tổng quát. Heel-event detection vẫn là thử nghiệm; cadence, step time, góc gối tại tiếp đất và knee ROM chưa được cung cấp trong UI. Tọa độ 2D chịu ảnh hưởng của góc camera, phối cảnh, che khuất, ánh sáng, FPS và sai số landmark. Không diễn giải confidence của model thành độ chính xác sinh cơ học hoặc kết luận y tế.
