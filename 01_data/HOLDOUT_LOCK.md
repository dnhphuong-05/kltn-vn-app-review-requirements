# ⚠ HOLD-OUT LOCK

`processed/shopee_holdout.csv` và `processed/lazada_external_test.csv` là tập **khóa**.

Theo amendment v1.7, các tập khóa gồm: **Shopee Hold-out 1.000** (400 pilot + 600), **focused FEATURE/UX set ~150**
(báo cáo tách biệt) và **Lazada external test 500**. Danh sách review_id: `01_data/manifests/splits_manifest_T0_2026-10-06.json`.
Các file gán nhãn của những tập này (`01_data/gold/annotations/`, `07_human_eval/pilot/`) cũng thuộc diện khóa.

KHÔNG dùng cho: training, tuning hyperparameter, chọn ngưỡng, tuning prompt, chọn mô hình,
hay bất kỳ quyết định thiết kế nào. Chỉ mở **một lần** sau khi toàn bộ tuning trên Validation
đã chốt và config/prompt đã đóng băng (`03_configs/frozen/`, `04_prompts/`).

Ghi lại ngày mở và commit hash tại đây:

| tập | ngày mở | commit | người mở |
|---|---|---|---|
| Shopee Hold-out (1.000) | 26/10/2026 | a79b305864bb817c31046747887655f36edc6988 | dnhphuong-05 |
| Focused FEATURE/UX set (150) | 26/10/2026 | a79b305864bb817c31046747887655f36edc6988 | dnhphuong-05 |
| Lazada external test (500) | 26/10/2026 | a79b305864bb817c31046747887655f36edc6988 | dnhphuong-05 |

**Ghi chú lần mở (GĐ10)**

- Ba tập được mở MỘT lần duy nhất, ngày 26/10/2026, trong notebook `10_rq1_evaluate_compare.ipynb` (nơi duy nhất đọc `gold_label`).
  Không mở lại; mọi phân tích sau đó chỉ dùng các file `gd10_*` đã lưu trong `05_experiments/rq1_classification/`.
- Mô hình neural được train lại từ config frozen v2.1 (`03_configs/frozen/`) trong một phiên Kaggle ngày 25/10/2026 (notebook 10a,
  cùng commit trên). Phiên đó chỉ nhận `review_id` + `text` của các tập khóa, không có nhãn. LLM dùng prompt v1 đã khóa (10b).
- Cổng Bước 0 đạt trước khi dự đoán tập khóa: Macro-F1 Validation của 5 cấu hình train lại lệch số GĐ6/7/9 dưới 0,05 điểm
  (lớn nhất 0,048), recall ACTIONABLE seed 42 vẫn ≥ 0,80 (ViSoBERT 0,817; Bi-LSTM 0,873). Xem `gd10_gate_validation.csv`, `gd10_gate_recall.csv`.
- Khi chấm điểm, `gd10_open_info.json` ghi `uncommitted_changes: true`. Các thay đổi chưa commit lúc đó chỉ là output của notebook
  10 và 10b; `02_src/` và `03_configs/frozen/` trùng commit trên.

**Lần đọc lại nhãn sau GĐ10 (chấm lại, không tuning)**

| ngày | notebook | commit | mục đích |
|---|---|---|---|
| 15/11/2026 | `18_paper_reanalysis.ipynb` | 2927f88af499d702a5bd8145977e8f06411aca57 (có thay đổi chưa commit) | Amendment v3.1: đọc `gold_label` của Hold-out, focused và Lazada để chấm lại các dự đoán đã khóa `gd10_*` trên tập khử trùng, chấm baseline phẳng trên Lazada, kiểm định McNemar, phân tích lỗi và biên Pareto. Không train lại, không gọi API, không đổi config, prompt hay ngưỡng, không chọn lại cấu hình. Trước khi phân tích, notebook tái tạo khớp từng ô các bảng GĐ10. |
| 15/11/2026 | `22_paper_step2_evaluate.ipynb` | 9e19ef75c14d7dd0ebc9aa95cda542a8238a303d (có thay đổi chưa commit) | Amendment v3.2 (mục 5), v3.3, v3.4, v3.5: lần chấm duy nhất cho các mô hình mới của bước 2 (TF-IDF, flat LLM, PhoBERT, flat Bi-LSTM) trên Hold-out, focused và Lazada, sau khi mọi cấu hình, ngưỡng và prompt đã khóa trên Validation; không train lại, không gọi API. Notebook được chạy hai lần trong ngày: lần thứ hai chỉ để sửa cách ghi nhãn hình Pareto, 20/21 bảng trùng từng ô với lần đầu (bảng còn lại chỉ thêm cột đánh dấu nhãn hình). |
| 15/11/2026 | `23_paper_neutral_prompt.ipynb` | 9e19ef75c14d7dd0ebc9aa95cda542a8238a303d (có thay đổi chưa commit) | Amendment v3.6: chỉ đọc nhãn Lazada (500) để chấm biến thể prompt LLM trung tính `v1n`, sau khi ba prompt v1n đã khóa vào registry; biến thể định sẵn, chạy một lần, không chỉnh trên Lazada. Không đọc nhãn Hold-out (phần Hold-out tùy chọn không chạy). |
| 15/11/2026 | `24_paper_lazada_error_analysis.ipynb` | 9e19ef75c14d7dd0ebc9aa95cda542a8238a303d (có thay đổi chưa commit) | Amendment v3.7: đọc `gold_label` và văn bản của Hold-out và Lazada để phân tích khám phá lỗi giữ nhầm OTHER của LLM (dự đoán đã lưu, nhóm từ khóa cố định trước) và tạo phiếu mã hóa thủ công mù `07_human_eval/paper_lazada_errors/`. Không gọi API, không đổi mô hình, prompt hay ngưỡng. Chạy lại cùng ngày sau khi người nghiên cứu mã hóa xong 161 dòng, để tính phân bố nhóm và đóng góp vào tỷ lệ giữ nhầm (cùng dự đoán, cùng nhãn). |
| 16/11/2026 | `23_paper_routing_calibration.ipynb` | 6030af179983c56d51257bfb175c17a5bb2f7fef | Amendment v4.1: đọc `gold_label` của Hold-out và Lazada để chấm định tuyến theo độ tin cậy (PhoBERT, ViSoBERT, TF-IDF ⇢ LLM phẳng) và hiệu chỉnh nhiệt độ tầng 1, chỉ dùng dự đoán đã lưu; θ, T và ngưỡng chọn trên Validation, lưu `paper_step4_selection.json` kèm SHA-256 trước khi đọc nhãn. Không train lại, không gọi API. Notebook tự ghi `holdout_use: 3` (lần chấm luật quyết định mới thứ ba, sau 26/10 và 15/11). Bổ sung vào nhật ký ngày 21/11/2026 vì lúc chạy chưa ghi. |
| 21/11/2026 | `25_paper_fair_controls.ipynb` | b436fad136189c47383eb98f13dfdf1539e9c940 (có thay đổi chưa commit) | Amendment v4.2: đọc `gold_label` của Hold-out, focused và Lazada để chấm đối chứng công bằng (mô hình phẳng có ngưỡng ưu tiên recall), các chính sách định tuyến (độ tin cậy, bất đồng PhoBERT–ViSoBERT, ngẫu nhiên cùng số lượt gọi, ngân sách cố định) và chấm lại trên Hold-out bỏ trùng y hệt cùng gần trùng; chỉ dùng dự đoán đã lưu, ngưỡng chọn trên Validation và lưu `paper_step5_selection.json` (SHA-256 `4c19fb8f…2bb8`) trước khi đọc nhãn. Không train lại, không gọi API. `holdout_use: 4`; mọi kết quả là phân tích khám phá. Notebook được chạy hai lần trong ngày: lần thứ hai chỉ thêm bảng hòa vốn khi đã có nhãn, cột ghi chú hòa vốn và tỷ lệ gửi LLM của hai cấu hình phẳng; mã băm lựa chọn trùng lần đầu, các bảng B1, B2 và gần trùng không đổi. |
| 21/11/2026 | `26_paper_tables.ipynb` | b436fad136189c47383eb98f13dfdf1539e9c940 (có thay đổi chưa commit) | Amendment v4.3: đọc lại `gold_label` của Hold-out, focused và Lazada chỉ để chấm lại các dự đoán đã khóa và sinh mọi bảng chính của bài báo từ một nguồn, với lần chạy chuẩn (seed 42; lần chạy duy nhất của LLM) làm đại lượng chính. Không train lại, không gọi API, không tạo dự đoán mới, không chọn hay chỉnh gì; như v3.1, đây không phải lần dùng mới để chọn luật quyết định. 42 cổng nhất quán đạt: mọi Macro-F1, CI, chênh lệch ghép cặp, p McNemar, hòa vốn và lựa chọn chi phí trùng file của bước 2, 4 và 5 (lệch tối đa dưới 1e-12). Notebook được chạy bốn lần trong ngày: lần một dừng ở bước lưu vì thiếu định nghĩa một cột, lần hai và ba chỉ đổi tên cột và mã cấu hình sang tiếng Anh, lần bốn thêm hai hình cho bài báo (đường định tuyến trên Hold-out đã đăng ký ở v4.1 kèm định tuyến ngẫu nhiên của v4.2; đường chi phí không dùng nhãn); số liệu bốn lần như nhau. |

## Sự cố ghi nhận

| ngày | mô tả |
|---|---|
| 25/10/2026 | Trong lúc kiểm tên cột các tập khóa, một lệnh sed lỗi đã in ra nhãn gold của 2 review Hold-out (1 BUG, 1 OTHER). Sự cố xảy ra SAU khi config đã đóng băng (v2.1), nên không có quyết định tuning/chọn model nào dựa trên thông tin này; model và ngưỡng đã khóa trước đó. 2/1.000 review không đủ ảnh hưởng bất kỳ metric nào. Ghi nhận để minh bạch. |

> Ghi chú 16/11/2026 (amendment v4.0): notebook được đánh số lại. Tên notebook trong các bảng trên là tên tại thời điểm chạy; bảng đối chiếu số cũ → mới ở `notebooks/README.md` (10 → 12, 18 → 13, 22 → 17, 23 → 18, 24 → 19).
