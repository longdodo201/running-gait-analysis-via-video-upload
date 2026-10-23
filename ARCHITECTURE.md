# ARCHITECTURE.md — Kiến trúc và luồng dữ liệu

## 1. Nguyên tắc kiến trúc
- UI chỉ điều phối input/output, không chứa thuật toán metric.
- Mỗi bước pipeline có input/output rõ ràng và dễ kiểm thử.
- Dùng model pretrained; không huấn luyện trong MVP.
- Giữ dữ liệu thô (landmarks) riêng với metric đã tính để có thể đánh giá/debug.
- Nếu bước nào thất bại hoặc chất lượng thấp, phát cảnh báo có cấu trúc thay vì tiếp tục tạo kết quả giả chính xác.

## 2. Luồng tổng thể
```text
Video upload
    |
    v
Video I/O + Validation
    |  VideoMetadata, frames, timestamps
    v
Pose Estimator (pretrained, local)
    |  Landmarks per frame + visibility/confidence
    v
Landmark Quality / Normalization
    |  Validated landmark sequence + quality report
    v
Signal Processing
    |  Smoothed trajectories / valid segments
    v
Gait Event Detection
    |  Candidate foot-strike/step events + confidence/reasons
    v
Metric Calculation
    |  Cadence, step time, knee angle/ROM, optional comparisons
    v
Visualization + Report
    |  Annotated video, plots, JSON/CSV, warnings
    v
Streamlit UI + Downloads
```

## 3. Cấu trúc thư mục dự kiến
```text
running-gait-analysis/
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── AGENTS.md
├── PROJECT_SPEC.md
├── ARCHITECTURE.md
├── PLAN.md
├── EVALUATION.md
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── video_io.py
│   ├── pose_estimator.py
│   ├── landmarks.py
│   ├── quality.py
│   ├── signal_processing.py
│   ├── foot_strike.py
│   ├── gait_metrics.py
│   ├── analysis_pipeline.py
│   ├── visualization.py
│   ├── report.py
│   └── utils.py
├── tests/
│   ├── test_angles.py
│   ├── test_signal_processing.py
│   ├── test_foot_strike.py
│   ├── test_gait_metrics.py
│   ├── test_quality.py
│   └── test_pipeline.py
├── scripts/
│   ├── download_model.py
│   └── evaluate.py
├── data/
│   ├── samples/
│   └── annotations/
├── outputs/
└── docs/
    ├── methodology.md
    └── evaluation.md
```

## 4. Trách nhiệm module
- `app.py`: giao diện Streamlit, nhận video, gọi pipeline, trình bày kết quả/tải file.
- `config.py`: cấu hình có thể điều chỉnh, giới hạn input, tham số xử lý; không chứa secrets.
- `src/video_io.py`: đọc metadata, giải mã frame, timestamps, ghi video; kiểm tra lỗi codec.
- `src/pose_estimator.py`: adapter bao quanh API pose model; không để API của thư viện rò rỉ khắp codebase.
- `src/landmarks.py`: quy ước tên keypoint, chuẩn hóa định dạng, tính góc hình học cơ bản.
- `src/quality.py`: đánh giá độ bao phủ/visibility/occlusion và sinh warning.
- `src/signal_processing.py`: lọc nhiễu và xử lý chuỗi thời gian có xét timestamp.
- `src/foot_strike.py`: tạo candidate event từ trajectory; trả cả thời điểm, bên, confidence và lý do/flag.
- `src/foot_strike_analysis.py`: đọc Ground Truth event CSV, ghép prediction một-một, tính event-level knee angles và cadence; detector contact-zone riêng dùng foot-index/heel vào vùng gần belt với vertical speed thấp.
- `src/gait_metrics.py`: tính cadence, step time, knee angle/ROM và metric khác từ input đã kiểm tra.
- `src/analysis_pipeline.py`: điều phối pipeline; không vẽ UI.
- `src/visualization.py`: overlay skeleton, đánh dấu event và tạo plot.
- `src/report.py`: chuẩn hóa output JSON/CSV và nội dung báo cáo.
- `scripts/evaluate.py`: chạy đánh giá trên annotations, xuất metric evaluation.
- `tests/`: unit/integration tests; các test toán học phải độc lập với model và mạng.

## 5. Hợp đồng dữ liệu đề xuất
Định dạng nội bộ có thể là dataclass hoặc typed models, nhưng phải thống nhất và ghi trong code.

### VideoMetadata
- `width`, `height`: pixel.
- `fps`: float nếu đọc được.
- `frame_count`: integer nếu đáng tin cậy.
- `duration_seconds`: float nếu tính được.
- `timestamps_seconds`: thời điểm từng frame nếu có; không mặc định rằng mọi video đều constant-frame-rate nếu chưa xác minh.

### LandmarkFrame
- `frame_index`: integer.
- `timestamp_seconds`: float.
- `landmarks`: map từ tên keypoint sang tọa độ.
- `visibility/presence/confidence`: trường tùy model; phải ghi rõ ý nghĩa, không đổi tên thành “accuracy”.

### GaitEvent
- `event_type`: ví dụ `foot_strike_candidate`.
- `side`: `left`, `right`, hoặc `unknown`.
- `timestamp_seconds`: thời điểm ước lượng.
- `confidence`: điểm heuristic/model nếu có, không phải xác suất đã hiệu chuẩn trừ khi được hiệu chuẩn.
- `flags`: danh sách lý do cảnh báo hoặc loại trừ.

### AnalysisResult
- metadata video.
- quality summary và warnings.
- landmarks hoặc tham chiếu tới file landmarks.
- events.
- metrics kèm đơn vị, phương pháp, trạng thái/quality flag.
- đường dẫn hoặc nội dung các artifacts đã sinh.

## 6. Luồng lỗi
- File không đọc được: dừng sớm và giải thích định dạng/codec.
- Không có pose hợp lệ: không tính metric; trả thông báo quay lại.
- Visibility thấp hoặc thiếu chân: chỉ trả metric có đủ dữ liệu, còn lại đánh dấu unavailable.
- Event detector không tìm được chuỗi hợp lý: không xuất cadence/step time giả.
- Render thất bại: giữ lại dữ liệu phân tích nếu an toàn, báo rõ artifact nào chưa tạo.
- Người dùng upload clip quá lớn: thông báo giới hạn hoặc yêu cầu clip ngắn hơn; không crash.

## 7. Ghi chú thiết kế
- Phải lưu timestamps, không chỉ frame index, để metric theo thời gian có thể xử lý đúng FPS.
- Bộ lọc tín hiệu phải xét missing values/occlusion; tránh nội suy qua khoảng mất dữ liệu dài.
- Phải có quy ước trái/phải nhất quán, đặc biệt với video quay ngang và phía chân bị che khuất.
- Event detection là phần có độ bất định cao nhất; giữ kết quả candidate và warning để phục vụ đánh giá.
- Ground Truth evaluation chỉ chạy khi caller cung cấp nhãn cho đúng clip; pipeline không gán nhãn hoặc sửa CSV tự động.
- Dashboard có thể truyền cache cục bộ cho artifacts pose extraction. Cache phải khớp hash video/model, metadata, runtime/schema và extraction settings; các bước gait analysis và evaluation luôn chạy lại.
- Dùng video gốc và kết quả pose để overlay; không thay đổi âm thầm FPS/kích thước khi xuất.
