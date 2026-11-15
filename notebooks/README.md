# Notebooks — pipeline chính

Mỗi notebook là một bước chạy độc lập (máy local hoặc Kaggle kernel). Thứ tự số là thứ tự quy trình.

## Quy ước Kaggle
- **Đọc** dữ liệu: `/kaggle/input/<dataset>` (read-only). Dùng `input_dir('<dataset>')` từ `src.utils.kaggle_paths`.
- **Ghi** kết quả: `/kaggle/working/` → tải về, bỏ vào `05_experiments/` tương ứng.
- **Import `src/`**: đóng gói RIÊNG thư mục `02_src/` thành một Kaggle Dataset (vd `kltn-src`) rồi
  `sys.path.append('/kaggle/input/kltn-src')`, hoặc `pip install git+<repo>`. Chạy local thì `sys.path.append('../02_src')`.
  KHÔNG đóng gói cả thư mục repo trên máy: bản local có `01_data/raw/` (chỉ bị gitignore, vẫn nằm trên đĩa).
- **Dữ liệu trên Kaggle**: chỉ upload bản interim đã scrub `01_data/interim/{shopee,lazada}_clean_T0_2026-10-06.jsonl`
  (+ manifest). TUYỆT ĐỐI KHÔNG upload `01_data/raw/` — raw còn ~1.100 tên người dùng sót (ghi chú thực thi GĐ2).
- **GPU**: bật accelerator cho 05, 06, 07, 09, 10, 16, 21. Mỗi notebook tách riêng để vừa giới hạn thời gian.
- **Seed**: luôn `set_seed(42)` ở cell đầu. Learning curve và thực nghiệm phân loại lặp 3 seed.
- **Hold-out**: notebook nào đọc nhãn Hold-out phải tuân `01_data/HOLDOUT_LOCK.md`.

## Danh sách

**Dữ liệu và gán nhãn**

| # | Notebook | Chạy ở | Số cũ |
|---|---|---|---|
| 00 | `00_smoke_test` — kiểm tra môi trường | local | 00 |
| 01 | `01_data_collection` — thu thập review Google Play | local | 01 |
| 02 | `02_cleaning_eda` — làm sạch, phân tích mô tả | local | 02 |
| 03 | `03_make_splits` — pilot, chia Train / Validation / Hold-out / focused | local | 03 |
| 04 | `04_annotation_agreement` — calibration, độ đồng thuận | local | 04 |

**Phân loại, giai đoạn 1 (ViSoBERT, Bi-LSTM + fastText, LLM few-shot; cascade, flat ViSoBERT, hybrid)**

| # | Notebook | Chạy ở | Số cũ |
|---|---|---|---|
| 05 | `05_learning_curve` — learning curve 3 seed | Kaggle GPU | 05 |
| 06 | `06_tier1_actionable` — tầng 1 trên Validation, ngưỡng | Kaggle GPU | 06 |
| 07 | `07_tier2_bfu` — tầng 2 trên Validation | Kaggle GPU | 07 |
| 08 | `08_llm_fewshot` — LLM few-shot tầng 1 và 2 trên Validation | local, API | 08 |
| 09 | `09_flat_visobert` — ViSoBERT phẳng 4 lớp | Kaggle GPU | 09 |
| 10 | `10_final_neural_kaggle` — train lại mô hình neural, dự đoán trên các tập khóa | Kaggle GPU | 10a |
| 11 | `11_final_llm` — LLM few-shot trên các tập khóa | local, API | 10b |
| 12 | `12_holdout_scoring` — chấm Hold-out, focused set, Lazada (một lần) | local | 10 |

**Phân loại, giai đoạn 2 (bài báo; amendment v3.1–v3.8)**

| # | Notebook | Chạy ở | Số cũ |
|---|---|---|---|
| 13 | `13_reanalysis_saved_predictions` — kiểm tra trùng lặp, McNemar, phân rã lỗi trên dự đoán đã lưu (v3.1) | local | 18 |
| 14 | `14_tfidf` — TF-IDF + bộ phân loại tuyến tính (v3.2) | local, CPU | 19 |
| 15 | `15_flat_llm` — LLM few-shot phẳng 4 lớp (v3.3) | local, API | 20 |
| 16 | `16_phobert_flat_bilstm_kaggle` — PhoBERT mọi tầng và Bi-LSTM phẳng (v3.4, v3.5) | Kaggle GPU | 21 |
| 17 | `17_joint_scoring` — chấm chung năm chiến lược, 28 cấu hình, Pareto (v3.2–v3.5) | local | 22 |
| 18 | `18_neutral_prompt` — prompt LLM trung tính về ứng dụng trên Lazada (v3.6) | local, API | 23 |
| 19 | `19_lazada_error_analysis` — lỗi giữ nhầm OTHER, phiếu mã hóa thủ công (v3.7) | local | 24 |
| 20 | `20_cost_model` — mô hình tổng chi phí, hòa vốn, cấu hình rẻ nhất theo ngưỡng (v3.8) | local | 25 |

**Ví dụ hướng phát triển (ngoài phạm vi nghiên cứu phân loại)**

| # | Notebook | Chạy ở | Số cũ |
|---|---|---|---|
| 21 | `21_future_full_inference` — chọn cấu hình, phân loại toàn corpus | Kaggle GPU | 11 |
| 22 | `22_future_bertopic_clustering` — gom cụm BERTopic, loại chủ đạo, purity | Kaggle hoặc local (API khi đặt tên cụm) | 12 |

## Đổi số notebook (16/11/2026, amendment v4.0)

Notebook được đánh số lại liên tục theo thứ tự quy trình. Protocol, amendment, `01_data/HOLDOUT_LOCK.md`, các config đã khóa, các file
`*_run_info.json` và output đã lưu trong notebook giữ số cũ tại thời điểm chạy; cột "Số cũ" ở trên dùng để đối chiếu. Các notebook cũ
13 (temporal validation), 14–15 (sinh và đánh giá Candidate Requirement), 16 (mô hình dữ liệu D6), 17 (hình và bảng cho báo cáo) và 26
(bảng cho bản thảo bài báo, chưa từng commit) đã được gỡ khỏi repo; notebook cũ 13–17 còn trong lịch sử git (commit `9e19ef7`).
