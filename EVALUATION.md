# EVALUATION.md — Đánh giá, ground truth và kiểm thử

## 1. Mục đích
Đánh giá riêng từng tầng:
1. Tính đúng của phép toán.
2. Chất lượng pose/landmark trên video mục tiêu.
3. Độ đúng của phát hiện gait events.
4. Độ đúng của metric được suy ra.
5. Độ ổn định end-to-end.

Không gộp các tầng này thành một tuyên bố chung kiểu “model chính xác X%”.

## 2. Ground truth
### 2.1 Event annotations
MVP có thể bắt đầu bằng việc gán nhãn thủ công các sự kiện bước/foot-strike trên video. Mỗi annotation nên có:
- `video_id`
- `event_type`
- `side` nếu quan sát được
- `timestamp_seconds` hoặc `frame_index` kèm FPS/timestamp mapping
- `annotator_id` (mã không định danh)
- `visibility`: rõ / không chắc / bị che khuất
- `notes` tùy chọn

Ví dụ schema JSON:
```json
{
  "video_id": "sample_001",
  "fps": 30.0,
  "events": [
    {
      "event_type": "foot_strike",
      "side": "left",
      "timestamp_seconds": 1.24,
      "visibility": "clear"
    }
  ]
}
```

Quy ước gán nhãn phải được viết trước khi annotate: định nghĩa frame/timestamp của event, cách xử lý event mơ hồ, và cách xác định bên chân. Nếu nhiều người gán nhãn, nên kiểm tra độ đồng thuận trên một phần dữ liệu.

### 2.2 Giới hạn ground truth
- Nhãn thủ công từ video là tham chiếu quan sát, không tương đương force plate hoặc cảm biến tiếp xúc.
- Nếu mục tiêu là xác thực contact time hay sự kiện tiếp đất chính xác, cần ground truth phù hợp hơn và đồng bộ hóa.
- Không dùng video tổng hợp hoặc dữ liệu tự tạo để tuyên bố độ chính xác trên người thật.
- Không tạo nhãn giả để làm đẹp kết quả.

## 3. Chia dữ liệu
- Tách theo video/người chạy, không chia frame ngẫu nhiên giữa train/test.
- Nếu dùng nhiều clip của cùng người, tránh để clip gần như trùng lặp xuất hiện ở cả development và test.
- Lưu metadata về góc quay, FPS, chất lượng và điều kiện ghi hình.
- Tập test cuối cùng không dùng để chỉnh ngưỡng liên tục.

Vì MVP không huấn luyện model, tập validation vẫn cần để điều chỉnh event detector; tập test độc lập hữu ích để ước lượng khả năng tổng quát hóa.

## 4. Chỉ số đánh giá
### 4.1 Event detection
Ghép event dự đoán với ground truth theo một tolerance thời gian được định trước. Một ground-truth event chỉ được ghép tối đa một lần.
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = 2 × Precision × Recall / (Precision + Recall)
- Event timing error: sai số tuyệt đối giữa timestamp dự đoán và timestamp tham chiếu; báo cáo median và mean/MAE nếu phù hợp.

Báo cáo rõ tolerance dùng để tính TP, ví dụ một ngưỡng thời gian đã định trước. Không chọn tolerance sau khi xem kết quả chỉ để cải thiện điểm.

### 4.2 Gait metrics
- Cadence MAE nếu ground truth cadence được định nghĩa và gán nhãn độc lập.
- Step-time MAE nếu có event references phù hợp.
- Với knee angle/ROM, video đơn camera chỉ cho góc hình học 2D; cần phương pháp tham chiếu phù hợp để tuyên bố sai số đo. Nếu không có tham chiếu, chỉ báo cáo tính nhất quán và giới hạn, không gọi đó là accuracy đã xác thực.

### 4.3 Pose và quality
Nếu không có nhãn keypoint độc lập, không tuyên bố độ chính xác pose thực nghiệm. Có thể báo cáo:
- Tỷ lệ frame có landmark cần thiết.
- Tỷ lệ frame bị đánh dấu low-quality.
- Số đoạn mất dữ liệu.
Các số này đo được mức bao phủ, không trực tiếp đo độ đúng vị trí.

## 5. Unit tests và integration tests
- Unit tests: góc 2D, xử lý timestamps, smoothing, ngưỡng sự kiện, khoảng cách tối thiểu giữa events, cadence và step time.
- Edge cases: chuỗi rỗng, NaN, thiếu keypoint, timestamp trùng/không tăng, FPS bất thường, video quá ngắn, không có event, event quá gần nhau.
- Integration tests: pipeline với fixture nhỏ hoặc mock pose estimator; không cần tải model/network.
- Smoke test: chạy model local và render video mẫu trong môi trường đã cài asset. Ghi riêng kết quả vì test này có thể tốn tài nguyên.

## 6. Báo cáo đánh giá
Báo cáo phải ghi:
- Commit/version code.
- Phiên bản Python, dependency và model asset.
- Mô tả dataset, số video, số người nếu biết, điều kiện quay.
- Quy tắc annotation và tolerance.
- Metric cùng số lượng mẫu/sự kiện.
- Kết quả và trường hợp thất bại.
- Giới hạn, bias có thể có và điều gì chưa được kiểm chứng.

Không ghi kết quả dự kiến như kết quả thật. Nếu chưa có dataset, ghi “chưa đánh giá thực nghiệm — cần dữ liệu annotated”, đồng thời cung cấp script và template để chạy sau.
