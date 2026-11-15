# KLTN — Phân loại app review thương mại điện tử tiếng Việt theo hiệu năng và chi phí

Khóa luận (K234111362). Nghiên cứu so sánh năm chiến lược phân loại (TF-IDF + bộ phân loại tuyến tính, Bi-LSTM + fastText, ViSoBERT,
PhoBERT, LLM few-shot) và ba kiến trúc (phẳng 4 lớp, cascade hai tầng, hybrid) cho review ứng dụng Shopee và Lazada, nhãn
BUG / FEATURE / UX / OTHER, đánh giá theo cả hiệu năng lẫn tổng chi phí. Phân loại toàn corpus và gom cụm được giữ lại làm ví dụ
hướng phát triển (notebook 21–22). Phạm vi theo amendment v4.0; tên đề tài mới đang chờ duyệt.

## Bề mặt chạy
- `notebooks/` — **pipeline chính**, mỗi notebook là một bước chạy độc lập (máy local hoặc Kaggle kernel); danh sách và bảng đổi số
  ở `notebooks/README.md`.
- `02_src/src/` — logic tái dùng, được *import* vào notebook (không nhồi logic vào cell).

## Thư mục
| Thư mục | Nội dung |
|---|---|
| `00_protocol/` | protocol đóng băng, guideline gán nhãn, decision tree, amendment |
| `01_data/` | manifest, gold (chỉ manifest được commit), `HOLDOUT_LOCK.md` |
| `02_src/src/` | thu thập, làm sạch, chia tập, gán nhãn, các chiến lược phân loại, phân tích bài báo, mô hình chi phí, gom cụm |
| `03_configs/` | `dev/` (tune) và `frozen/` (khóa) |
| `04_prompts/` | prompt few-shot phân loại, prompt đặt tên cụm, `prompts_registry.md` (hash) |
| `05_experiments/` | output do notebook sinh: `eda/`, `learning_curve/`, `rq1_classification/`, `rq2_topics/` (ví dụ gom cụm) |
| `07_human_eval/` | calibration, pilot, mã hóa lỗi Lazada, purity của cụm (phiếu có văn bản review không commit) |
| `08_deliverables/` | chỉ mục các gói kết quả |
| `09_thesis/` | đề cương, hình, slides, tổng quan tài liệu |

## Thứ tự quy trình
`00_protocol` (freeze) → `01_data` → `notebooks/00–20` → kết quả về `05_experiments` → chỉ mục ở `08_deliverables`.

## Tái lập nhanh
1. `pip install -r requirements.txt`
2. Đặt dữ liệu thô vào `01_data/raw/` (xem `01_data/HOLDOUT_LOCK.md` trước khi đụng Hold-out).
3. Chạy `notebooks/` theo thứ tự số. Chi tiết Kaggle: `notebooks/README.md`; các bước tái lập: `08_deliverables/D7_thesis_reproducibility/`.

## Chống data leakage (bắt buộc)
- Final Hold-out **khóa** cho tới khi mọi tuning xong — xem `01_data/HOLDOUT_LOCK.md`.
- Config/prompt **đóng băng** ở `03_configs/frozen/` và `04_prompts/` (kèm hash) trước test.
- `05_experiments/` chỉ chứa thứ do notebook sinh ra (tái lập được).
