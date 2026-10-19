# Phase 1: Hướng dẫn chạy video mới trên Windows

Tài liệu này hướng dẫn chạy pipeline Phase 1 từ VS Code trên Windows để tạo:
- File JSON chứa landmarks MediaPipe theo từng frame.
- Video MP4 có overlay skeleton (tùy chọn).

Phase 1 chỉ trích xuất pose landmarks. Nó **chưa** tính cadence, step time, góc gối/knee ROM, không đánh giá y tế và không kết luận dáng chạy đúng/sai.

## 1. Chuẩn bị một lần

### 1.1. Mở đúng repository

1. Mở GitHub Desktop và chọn repository `running-gait-analysis-via-video-upload`.
2. Chọn branch `main`, nhấn **Fetch origin**; nếu xuất hiện **Pull origin**, nhấn để cập nhật.
3. Chọn **Repository → Show in Explorer**.
4. Mở VS Code → **File → Open Folder** → chọn thư mục gốc repository. Thư mục này cần có `requirements.txt`, `scripts`, `src`, `tests` và `docs`.

### 1.2. Mở Terminal

Trong VS Code chọn **Terminal → New Terminal**. Kiểm tra thư mục hiện tại là thư mục gốc repository, nơi có `requirements.txt`.

Các lệnh trong tài liệu dùng cú pháp **Command Prompt (CMD)** trừ khi ghi rõ PowerShell. Nếu Terminal đang là PowerShell, các lệnh `python`, `pip` và lệnh chạy pipeline vẫn dùng được; lệnh kích hoạt môi trường có cú pháp khác (xem mục 1.3).

### 1.3. Tạo môi trường Python riêng

Kiểm tra Python:

```cmd
python --version
```

Máy đã được smoke test với Python 3.12.10. Dùng Python 3.12 đang cài trên máy; môi trường mục tiêu ban đầu của dự án là Python 3.11, nên nếu có vấn đề tương thích hãy ghi lại phiên bản và lỗi thay vì tự ý đổi phiên bản dependency.

Nếu `py` không được nhận diện nhưng `python --version` hoạt động, dùng `python` cho các lệnh bên dưới.

Do Windows từng gặp lỗi quyền truy cập/đường dẫn khi tạo môi trường ngay trong repository, hướng dẫn này đặt virtual environment **bên ngoài repository**:

```cmd
python -m venv "%USERPROFILE%\.venvs\rga-phase1-312"
```

Nếu thư mục môi trường đã tồn tại, không cần tạo lại. Kích hoạt trong **CMD**:

```cmd
"%USERPROFILE%\.venvs\rga-phase1-312\Scripts\activate.bat"
```

Nếu Terminal là **PowerShell**, kích hoạt bằng:

```powershell
& "$env:USERPROFILE\.venvs\rga-phase1-312\Scripts\Activate.ps1"
```

Khi kích hoạt thành công, đầu dòng Terminal sẽ có `(rga-phase1-312)`. Kiểm tra:

```cmd
python --version
python -m pip --version
```

Cài dependency (chỉ cần làm khi thiết lập môi trường lần đầu hoặc khi `requirements.txt` thay đổi):

```cmd
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

Kết quả mong đợi của `pip check` là `No broken requirements found.`

> Nếu môi trường đã cài dependency và chạy được rồi, các lần chạy video tiếp theo không cần cài lại thư viện.

## 2. Chuẩn bị model MediaPipe

Pipeline cần file model `pose_landmarker_full.task`. Model không được lưu trong Git repository vì là asset tải về riêng.

Trong lần smoke test trước, model nằm tại:

```text
C:\Users\Admin\Documents\Codex\2026-10-18\github-plugin-github-openai-curated-remote-2\work\phase0-validation\assets\pose_landmarker_full.task
```

Đây là đường dẫn riêng của máy tại thời điểm kiểm thử, không phải đường dẫn chung cho mọi máy. Nếu file đã được chuyển đi, hãy tìm `pose_landmarker_full.task` trong File Explorer và dùng **Copy as path** để lấy đúng đường dẫn hiện tại. Không cần tải lại nếu file model hiện có vẫn là bản đã xác minh.

Có thể đặt model tại `models\pose_landmarker_full.task` trong repository nếu muốn dùng đường dẫn ngắn, nhưng thư mục `models` và file model không bắt buộc phải được commit lên GitHub.

Xác minh model trước khi chạy (thay đường dẫn ví dụ bằng đường dẫn thực tế của bạn):

```cmd
python scripts\verify_environment.py --model "C:\path\to\pose_landmarker_full.task"
```

Thành công khi có dòng tương tự `Pose Landmarker initialized with CPU delegate.` Các cảnh báo TensorFlow Lite về feedback manager có thể xuất hiện và không nhất thiết là lỗi. Nếu có lỗi `Model file not found`, kiểm tra đường dẫn và dấu ngoặc kép.

## 3. Chạy một video mới (mỗi lần sử dụng)

### 3.1. Lấy đường dẫn video

1. Lưu video chạy bộ ở một vị trí trên máy, ví dụ thư mục Windows `Videos`.
2. Trong File Explorer, nhấp chuột phải vào video → **Copy as path**.
3. Đường dẫn thường có dạng `"C:\Users\YourName\Videos\my_run.mp4"`. Đây chỉ là ví dụ; hãy dùng đường dẫn thật của bạn.

Video cần là MP4 có thể giải mã được bằng OpenCV. Nên chọn video quay ngang, thấy toàn thân người chạy và có ít che khuất. Các giới hạn ứng dụng mặc định là 100 MiB và 60 giây; đây là giới hạn cấu hình của dự án, không phải giới hạn của MediaPipe.

### 3.2. Chạy pipeline để tạo cả JSON và skeleton video

Từ thư mục gốc repository, khi môi trường đã kích hoạt, chạy lệnh dưới đây. Thay cả hai đường dẫn mẫu bằng đường dẫn thực tế:

```cmd
python scripts\process_video.py "C:\path\to\my_run.mp4" --model "C:\path\to\pose_landmarker_full.task" --output "outputs\my_run_landmarks.json" --annotated-video "outputs\my_run_annotated.mp4"
```

Ví dụ dùng video `running_gait_sideview2.mp4` trên máy đã smoke test:

```cmd
python scripts\process_video.py "C:\Users\Admin\Videos\running_gait_sideview2.mp4" --model "C:\Users\Admin\Documents\Codex\2026-10-18\github-plugin-github-openai-curated-remote-2\work\phase0-validation\assets\pose_landmarker_full.task" --output "outputs\running_gait_sideview2_landmarks.json" --annotated-video "outputs\running_gait_sideview2_annotated.mp4"
```

Pipeline tự tạo thư mục đầu ra nếu cần. Khi hoàn tất, Terminal sẽ in số frame đã lưu và số pose phát hiện được.

### 3.3. Chỉ tạo JSON (không tạo video skeleton)

Nếu chỉ cần JSON, bỏ tham số `--annotated-video`:

```cmd
python scripts\process_video.py "C:\path\to\my_run.mp4" --model "C:\path\to\pose_landmarker_full.task" --output "outputs\my_run_landmarks.json"
```

### 3.4. Chạy nhiều video

Mỗi video nên dùng tên đầu ra riêng để không ghi đè kết quả trước đó. Ví dụ, với `run_a.mp4` và `run_b.mp4`, đặt lần lượt:
- `outputs\run_a_landmarks.json` và `outputs\run_a_annotated.mp4`
- `outputs\run_b_landmarks.json` và `outputs\run_b_annotated.mp4`

Nếu chạy lại cùng tên đầu ra, kết quả cũ có thể bị thay thế.

## 4. Tìm và kiểm tra kết quả

Các đường dẫn đầu ra là **đường dẫn tương đối với thư mục gốc repository**:
- `outputs\<ten_video>_landmarks.json`: metadata video, timestamp và danh sách frame; mỗi pose có 33 landmarks.
- `outputs\<ten_video>_annotated.mp4`: video có vẽ skeleton, nếu dùng `--annotated-video`.

Có thể xem file ở cả hai nơi:
- Trong VS Code: mở thư mục `outputs` ở Explorer bên trái.
- Trong Windows File Explorer: mở thư mục repository → `outputs`.

Nên mở MP4 trước để xem skeleton có bám theo người chạy hay không. Mở JSON trong VS Code để kiểm tra dữ liệu. Một pose được phát hiện ở mỗi frame không tự chứng minh mọi tọa độ landmark đều chính xác.

## 5. Các trường hợp lỗi thường gặp

| Thông báo/tình huống | Cách xử lý |
|---|---|
| `'py' is not recognized` | Dùng `python` nếu `python --version` hiển thị phiên bản hợp lệ. |
| `Permission denied` khi tạo `.venv` trong repository | Dùng môi trường bên ngoài repository như mục 1.3; không cần xóa thư mục `.venv` cũ ngay. |
| `$env:USERPROFILE` bị hiểu như một tên thư mục | Terminal đang là CMD nhưng đã dùng cú pháp PowerShell. Trong CMD dùng `%USERPROFILE%`; trong PowerShell dùng `$env:USERPROFILE`. |
| `Model file not found` | Dùng **Copy as path** trong File Explorer và truyền đường dẫn chính xác vào `--model`. |
| `No module named ...` | Kiểm tra đầu dòng Terminal có `(rga-phase1-312)`, rồi cài lại bằng `python -m pip install -r requirements.txt`. |
| `pip check` báo dependency conflict | Lưu lại toàn bộ thông báo; không tự ý nâng/hạ phiên bản gói riêng lẻ. |
| Video không mở/không đọc được | Kiểm tra đúng đường dẫn, phần mở rộng MP4 và khả năng giải mã codec. Thử một video MP4 khác ngắn hơn. |
| Video vượt 100 MiB hoặc 60 giây | Dùng clip ngắn hơn hoặc xem hướng dẫn CLI trong `docs/PHASE1_PIPELINE.md` về `--max-bytes 0` / `--max-seconds 0` để tắt giới hạn tương ứng khi thật sự cần. |
| Không phát hiện người chạy | Kiểm tra người có xuất hiện rõ, toàn thân trong khung hình, đủ sáng và ít bị che khuất không. Pipeline có thể báo lỗi nếu toàn video không phát hiện được pose. |
| Có cảnh báo TensorFlow Lite nhưng vẫn khởi tạo CPU | Cảnh báo feedback-manager thường không nghiêm trọng nếu lệnh tiếp tục chạy; hãy dựa vào kết quả và mã lỗi cuối cùng. |
| Không thấy file đầu ra | Đợi lệnh kết thúc, xem thông báo cuối Terminal và kiểm tra thư mục `outputs` bên trong đúng repository đang mở. |

## 6. Kiểm thử và an toàn dữ liệu

Chạy unit tests từ thư mục gốc:

```cmd
python -m unittest discover -s tests -v
```

Các unit tests dùng fixture/mock; chúng không thay thế kiểm tra trực quan trên video thật. Sau mỗi video mới, nên xem skeleton và kiểm tra số frame/pose trong thông báo Terminal.

**Không commit video cá nhân, JSON kết quả, video skeleton hoặc model vào GitHub.** Thư mục `outputs/`, môi trường `.venv/` và file model đã được thêm vào `.gitignore` trong repository hiện tại. Nếu GitHub Desktop hiện các file này là thay đổi cần commit, dừng lại và kiểm tra `.gitignore` trước.

## 7. Giới hạn Phase 1

- Đầu ra là pose landmarks thô, không phải kết quả phân tích dáng chạy.
- Chưa có gait-event detector, cadence, step time, góc gối/knee ROM, biểu đồ hay báo cáo.
- Tọa độ 2D chịu ảnh hưởng của góc camera, phối cảnh, che khuất, ánh sáng, độ phân giải và nhiễu landmark.
- Timestamp dựa trên FPS nguồn khi FPS hợp lệ; video variable-frame-rate không được khôi phục timestamp chính xác từ metadata OpenCV.
- Thành công khi chạy pipeline chỉ xác nhận chương trình xử lý được clip đó; không chứng minh độ chính xác sinh cơ học hoặc khả năng chẩn đoán.

Sau khi kiểm tra đầu ra trên một vài video, có thể chuyển sang Phase 2 để xây dựng đặc trưng và metric với quy tắc đánh giá rõ ràng.
