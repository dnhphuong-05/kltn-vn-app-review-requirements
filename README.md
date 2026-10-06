# KLTN — AI-Assisted Business Analysis: App Review → Candidate Requirement

Khóa luận (K234111362) theo phương pháp **Design Science Research**. Artifact gồm một pipeline
phân loại **cascade** hai tầng, so sánh ba chiến lược AI theo hiệu năng **và** tổng chi phí, rồi
sinh Candidate Requirement truy vết được từ app review tiếng Việt (Shopee + Lazada).

## Bề mặt chạy
- `notebooks/` — **pipeline chính**, mỗi notebook là một Kaggle kernel chạy độc lập.
- `src/` — logic tái dùng, được *import* vào notebook (không nhồi logic vào cell).

## Thứ tự quy trình (khớp đề cương)
`00_protocol` (freeze) → `01_data` → `notebooks/01–16` → kết quả về `05_experiments` →
artifact độc lập công cụ ở `06_output_data_model` → đóng gói `08_deliverables`.

## Tái lập nhanh
1. `pip install -r requirements.txt`
2. Đặt dữ liệu thô vào `01_data/raw/` (xem `01_data/HOLDOUT_LOCK.md` trước khi đụng Hold-out).
3. Chạy `notebooks/` theo thứ tự số. Chi tiết Kaggle: `notebooks/README.md`.

## Chống data leakage (bắt buộc)
- Final Hold-out **khóa** cho tới khi mọi tuning xong — xem `01_data/HOLDOUT_LOCK.md`.
- Config/prompt **đóng băng** ở `03_configs/frozen/` và `04_prompts/` (kèm hash) trước test.
- `05_experiments/` chỉ chứa thứ do notebook sinh ra (tái lập được).
