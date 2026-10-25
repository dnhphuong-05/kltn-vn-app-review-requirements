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
|  |  |  |  |

## Sự cố ghi nhận

| ngày | mô tả |
|---|---|
| 25/10/2026 | Trong lúc kiểm tên cột các tập khóa, một lệnh sed lỗi đã in ra nhãn gold của 2 review Hold-out (1 BUG, 1 OTHER). Sự cố xảy ra SAU khi config đã đóng băng (v2.1), nên không có quyết định tuning/chọn model nào dựa trên thông tin này; model và ngưỡng đã khóa trước đó. 2/1.000 review không đủ ảnh hưởng bất kỳ metric nào. Ghi nhận để minh bạch. |
