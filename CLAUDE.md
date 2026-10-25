# CLAUDE.md — Quy tắc bắt buộc cho dự án này

> Claude Code tự đọc file này mỗi phiên. Đây là HỢP ĐỒNG BẮT BUỘC, bám theo
> đề cương V5 (xem `09_thesis/proposal/`). Nếu yêu cầu người dùng mâu thuẫn với
> quy tắc dưới đây, HỎI LẠI trước khi làm. Protocol đầy đủ (đã đóng băng):
> `00_protocol/research_protocol.md` ; runbook từng bước: `00_protocol/QUY_TRINH_CHUAN.md`.

## 0. Định vị
Nghiên cứu Design Science Research. Sản phẩm = artifact (pipeline cascade +
quy tắc chọn chiến lược cost-aware) biến app review tiếng Việt (Shopee + Lazada)
thành Candidate Requirement truy vết được. KHÔNG phát minh model mới (đóng góp
dạng exaptation).

## 1. Kiến trúc HAI TẦNG — cốt lõi, không tự đổi
```
Review → TẦNG 1: ACTIONABLE / NOT_ACTIONABLE     (so 3 chiến lược — RQ1 phần A)
           │ (giữ ACTIONABLE)
           ├─ TẦNG 2: BUG / FEATURE / UX          (so 3 chiến lược — RQ1 phần B)
           └─ Gom cụm TOÀN BỘ actionable → mỗi cụm gắn LOẠI CHỦ ĐẠO → sinh requirement
```
- Gom cụm chạy trên TOÀN BỘ review actionable. BUG/FEATURE/UX là THUỘC TÍNH
  gắn vào cụm/review SAU khi gom, KHÔNG phải điều kiện chia trước khi gom.
- So sánh 3 chiến lược ở CẢ HAI tầng (không chỉ tầng 1). Báo cáo per-tier,
  hybrid 3×3 (ghép chéo), và end-to-end 4 lớp.

## 2. Nhãn — MỘT bộ gold 4 lớp, suy ra hai tầng (KHÔNG gán lại)
- Gán nhãn MỘT lần theo 4 lớp: BUG / FEATURE / UX / OTHER (`01_data/gold/`).
- Tầng 1 suy ra: ACTIONABLE = BUG ∪ FEATURE ∪ UX ; NOT_ACTIONABLE = OTHER.
- Tầng 2 chỉ phân BUG/FEATURE/UX trên phần ACTIONABLE.
- DÙNG CHUNG một Gold Dataset và một Hold-out Test cho cả hai tầng.
  (Đây là quyết định của đề cương — KHÔNG tách gold/hold-out/Kappa riêng mỗi tầng.)
- Review ambiguous giữa BUG/FEATURE/UX vẫn là ACTIONABLE hợp lệ cho tầng 1.

## 3. Phương án A (Bi-LSTM) — dùng PRETRAINED fastText tiếng Việt
- Theo đề cương V5: Bi-LSTM + **pretrained fastText tiếng Việt** (static embeddings).
- KHÔNG tự huấn luyện embedding từ đầu ở phương án này. Nếu người dùng muốn đổi
  sang tự train, DỪNG và nhắc rằng việc đó mâu thuẫn đề cương, hỏi xác nhận +
  nhắc phải sửa đề cương trước.
- Ba chiến lược: A = Bi-LSTM+fastText | B = fine-tuned ViSoBERT | C = LLM few-shot.
- Baseline phẳng 4 lớp chỉ để ĐỐI CHỨNG (reference only), không đưa vào downstream.

## 4. Hold-out Test — bất khả xâm phạm (xem `01_data/HOLDOUT_LOCK.md`)
- Lấy mẫu ngẫu nhiên thuần, KHÔNG dựa nhãn AI.
- KHÔNG dùng làm ví dụ few-shot; KHÔNG vào Train/Validation; KHÔNG tune ngưỡng/
  hyperparameter/prompt/chọn model trên đó.
- Mở DUY NHẤT một lần sau khi mọi tuning đã chốt và config/prompt đã đóng băng.
- Phát hiện leakage → DỪNG, báo người dùng, không tự sửa âm thầm.

## 5. Config & Prompt — đóng băng trước đánh giá cuối
- `03_configs/frozen/` và `04_prompts/` là bản KHÓA. Ghi prompt_hash vào
  `04_prompts/prompts_registry.md`. Tuning diễn ra ở `03_configs/dev/`.
- LLM few-shot: ví dụ CHỈ lấy từ Train (Validation là nơi phát triển prompt nên không
  được dùng làm ví dụ — amendment v1.1). Chốt model + version + temperature
  trong protocol trước khi chạy test.

## 6. Metrics — hai tầng, khóa trước Hold-out
- Tầng 1: F1 + RECALL của ACTIONABLE (ưu tiên recall; có thể chỉnh ngưỡng).
- Tầng 2: Macro-F1 (3 lớp) + per-class F1 trên gold-actionable.
- End-to-end: Macro-F1 (4 lớp) trên toàn Hold-out để so baseline + nghiên cứu khác.
- Learning curve: 400→800→1200→1600→2000→… , 3 seeds; Tier1 actionable-F1, Tier2 per-class F1.

## 7. Gom cụm (BERTopic) — topic phải CÓ NGHĨA
- Chỉ gom review actionable. Chọn min_topic_size theo COHERENCE SCORE (không tùy tiện).
- Giảm outlier, gộp topic trùng, đặt tên topic bằng LLM.
- Gắn loại chủ đạo theo đa số có ngưỡng (~60%); dưới ngưỡng → "mixed", gắn cờ cho BA.
- Báo thêm type-purity. RQ2 đánh giá bằng coherence (định lượng) + kiểm tra bằng mắt.

## 8. Tính hợp lệ thời gian
- Temporal Issue Validation bằng date/app-version: ACTIVE / RECURRING /
  POSSIBLY_RESOLVED / INSUFFICIENT. Mặc định chỉ ACTIVE & RECURRING sang sinh requirement.

## 9. Xếp ưu tiên — KHÔNG dùng công thức Severity/Impact
- Chỉ dùng issue frequency (số review trong cụm / tổng actionable). BA quyết định cuối.

## 10. Sinh requirement — User Story TIẾNG ANH
- User Story + Acceptance Criteria bằng tiếng Anh (để chạy AQUSA).
- Giữ Source Traceability: mỗi requirement giữ evidence IDs (review_id nguồn) —
  giá trị lõi, backend phải kiểm tra được.

## 11. Chi phí / GPU / thời gian — đo thực tế
- Không hardcode. Đo giờ gán nhãn, train/inference time, chi phí API/token thực.
- Chi phí là tiêu chí HẠNG NHẤT ngang hiệu năng. GPU khuyến nghị, không bắt buộc.

## 12. Cấu trúc thư mục — không tự tạo thư mục lạ
```
00_protocol/        protocol đóng băng, guideline, decision tree, amendments
01_data/            raw (bất biến) / interim / processed / gold / manifests ; HOLDOUT_LOCK.md
02_src/src/         logic tái dùng (import vào notebook) — KHÔNG nhồi logic vào cell
03_configs/         dev (tune) / frozen (khóa)
04_prompts/         few-shot + requirement gen (+ prompt_hash)
05_experiments/     output do notebook sinh (tái lập) — learning_curve/rq1/rq2/rq3
06_output_data_model/  artifact D6 (CSV quan hệ) + powerbi/
07_human_eval/      phiếu chấm + kết quả
08_deliverables/    đóng gói cuối D1–D7 (copy/link)
09_thesis/          proposal + chapters + figures + references
10_docs/            timeline, biên bản, slide
```

## 13. Quy ước làm việc — trước khi hành động
- Pipeline chính nằm ở `notebooks/` (mỗi notebook = 1 Kaggle kernel). Xử lý dữ
  liệu tương tác → notebook, KHÔNG viết script rời ngoài `src/`.
- Logic đặt ở `src/`, notebook chỉ IMPORT. Không tự sửa logic `src/` mà không
  giải thích trước.
- Kaggle: đọc `/kaggle/input`, ghi `/kaggle/working` rồi mang về `05_experiments/`.
  Dùng `src.utils.kaggle_paths`. Luôn `set_seed(42)`.
- KHÔNG chạy API tốn phí trên toàn bộ dữ liệu khi chưa test mẫu nhỏ + người dùng xác nhận.
- Lỗi: DÁN NGUYÊN lỗi + giải thích tiếng Việt trước khi thử cách khác.
- Mọi thay đổi quyết định nghiên cứu phải ghi vào `00_protocol/protocol_amendments.md` (có version).

## 14. Luận văn — định dạng & cấu trúc
- Khi viết/chỉnh nội dung luận văn, BẮT BUỘC tuân theo yêu cầu định dạng + cấu trúc trong
  `09_thesis/guidelines/` (bản tóm tắt: `09_thesis/guidelines/yeu_cau_tom_tat.md`). Kiểm tra lại quy định này
  trước khi viết mỗi chương.
