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

## Sự cố ghi nhận

| ngày | mô tả |
|---|---|
| 25/10/2026 | Trong lúc kiểm tên cột các tập khóa, một lệnh sed lỗi đã in ra nhãn gold của 2 review Hold-out (1 BUG, 1 OTHER). Sự cố xảy ra SAU khi config đã đóng băng (v2.1), nên không có quyết định tuning/chọn model nào dựa trên thông tin này; model và ngưỡng đã khóa trước đó. 2/1.000 review không đủ ảnh hưởng bất kỳ metric nào. Ghi nhận để minh bạch. |
