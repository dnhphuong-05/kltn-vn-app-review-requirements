# Notebooks — pipeline chính (Kaggle)

Mỗi notebook = một Kaggle kernel chạy độc lập. Thứ tự số = thứ tự quy trình.

## Quy ước Kaggle
- **Đọc** dữ liệu: `/kaggle/input/<dataset>` (read-only). Dùng `input_dir('<dataset>')` từ `src.utils.kaggle_paths`.
- **Ghi** kết quả: `/kaggle/working/` → tải về, bỏ vào `05_experiments/` tương ứng.
- **Import `src/`**: đóng gói RIÊNG thư mục `02_src/` thành một Kaggle Dataset (vd `kltn-src`) rồi
  `sys.path.append('/kaggle/input/kltn-src')`, hoặc `pip install git+<repo>`. Chạy local thì `sys.path.append('../02_src')`.
  KHÔNG đóng gói cả thư mục repo trên máy: bản local có `01_data/raw/` (chỉ bị gitignore, vẫn nằm trên đĩa).
- **Dữ liệu trên Kaggle**: chỉ upload bản interim đã scrub `01_data/interim/{shopee,lazada}_clean_T0_2026-10-06.jsonl`
  (+ manifest). TUYỆT ĐỐI KHÔNG upload `01_data/raw/` — raw còn ~1.100 tên người dùng sót (ghi chú thực thi GĐ2).
- **GPU**: bật accelerator cho 05/06/07 (ViSoBERT, Bi-LSTM). Mỗi notebook tách riêng để vừa giới hạn thời gian.
- **Seed**: luôn `set_seed(42)` ở cell đầu. Learning curve lặp 3 seeds.
- **Hold-out**: notebook nào mở Hold-out phải tuân `01_data/HOLDOUT_LOCK.md`.

## Danh sách
| # | Notebook | GPU | RQ |
|---|---|---|---|
| 00 | smoke_test (kiểm tra môi trường) | | - |
| 01 | data_collection | | - |
| 02 | cleaning_eda | | - |
| 03 | make_splits | | - |
| 04 | annotation_agreement | | - |
| 05 | learning_curve | ✓ | RQ1 |
| 06 | rq1_tier1_actionable | ✓ | RQ1 |
| 07 | rq1_tier2_bfu | ✓ | RQ1 |
| 08 | rq1_llm_fewshot | API | RQ1 |
| 09 | rq1_flat_baseline | | RQ1 |
| 10 | rq1_evaluate_compare | | RQ1 |
| 11 | full_inference_actionable | ✓ | - |
| 12 | rq2_bertopic_dominant_type | | RQ2 |
| 13 | temporal_validation | | RQ2 |
| 14 | rq3_generation | API | RQ3 |
| 15 | rq3_evaluation | | RQ3 |
| 16 | build_output_data_model | | D6 |
