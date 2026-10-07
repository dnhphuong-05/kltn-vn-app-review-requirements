# QUY TRÌNH CHUẨN — Runbook end-to-end

> Bản quy trình chi tiết mà `CLAUDE.md` và `research_protocol.md` trỏ tới.
> Mỗi giai đoạn ghi rõ: INPUT → việc làm → OUTPUT → cổng kiểm soát (gate).
> Quyết định nghiên cứu đã đóng băng; thay đổi phải ghi `protocol_amendments.md`.

## Bản đồ giai đoạn ↔ notebook
| GĐ | Notebook | Input | Output | Gate |
|---|---|---|---|---|
| 0 | — (00_protocol) | — | protocol khóa | Freeze trước khi phân tích |
| 1 | 01_data_collection | Google Play | `01_data/raw/` (JSONL) | Snapshot bất biến |
| 2 | 02_cleaning_eda | raw | `01_data/interim/` | — |
| 3 | 04_annotation_agreement (phần A: calibration) | interim (mẫu calibration) | κ report + guideline chốt | κ ≥ 0.6 → chốt guideline |
| 4 | 03_make_splits → 04_annotation_agreement (phần B) | interim + guideline đã chốt | pilot → Hold-out size → splits + manifest → `gold/` + provenance | Khóa Hold-out |

> Thứ tự GĐ3/GĐ4 đảo theo amendment v1.1: guideline phải chốt TRƯỚC pilot vì pilot cần nhãn.
> Tên file notebook giữ nguyên (03/04); thứ tự chạy là 04-phần A → 03 → 04-phần B.
| 5 | 05_learning_curve | Train/Val | curve + stopping evidence | Chốt Train size |
| 6–9 | 06..09 | Train/Val | model + prediction đã lưu | Chưa mở Hold-out |
| 10 | 10_rq1_evaluate_compare | prediction + Hold-out | bảng RQ1 | **MỞ Hold-out 1 lần** |
| 11 | 11_full_inference_actionable | corpus + model thắng | actionable reviews | — |
| 12 | 12_rq2_bertopic_dominant_type | actionable | topics + loại chủ đạo | Coherence chọn k |
| 13 | 13_temporal_validation | topics + metadata | trạng thái thời gian | — |
| 14 | 14_rq3_generation | cụm ACTIVE/RECURRING | requirements + evidence | Prompt đã khóa |
| 15 | 15_rq3_evaluation | requirements | AQUSA + human sample | — |
| 16 | 16_build_output_data_model | tất cả | `06_output_data_model/` | — |

---

## GĐ0 — Đóng băng protocol
Khóa `research_protocol.md`, `annotation_guideline_v1.md`, taxonomy, metrics,
seeds, model/version của LLM. Sau mốc này, mọi thay đổi vào `protocol_amendments.md`.

## GĐ1 — Thu thập (notebook 01)
INPUT: Google Play (Shopee main, Lazada external).
LÀM: crawl review + metadata (date, app version, rating) — CHẠY LOCAL (Kaggle tắt
internet mặc định), logic ở `src/collect/google_play.py`. Sort NEWEST (Shopee tách theo
`filter_score_with` = 1..5 rồi gộp + khử trùng theo review_id — amendment v1.4), dừng theo NGÀY
khi lô vượt cutoff 2025-10-06 (T0 = 2026-10-06 − 12 tháng, amendment v1.3), trần an toàn 120.000/app (amendment v1.2).
Hash userName, không lưu userImage, scrub PII ([NAME]/[PHONE]/[EMAIL]) trong text + reply_content
ngay khi ghi. Lưu JSONL vào `01_data/raw/{shopee,lazada}/`.
KHÔNG sửa file raw về sau. Xong upload raw lên Kaggle làm Dataset.
OUTPUT: raw snapshot + manifest (T0, version thư viện, package id, lang/country, sort,
số review, min/max review_date, checksum) vào `01_data/manifests/`.
GATE (nghiệm thu): min(review_date) ≤ 2025-10-06 cho CẢ HAI app.

## GĐ2 — Làm sạch + EDA (notebook 02)
LÀM: chuẩn hóa text, lọc trùng/spam, trích metadata, de-identify nếu có PII.
EDA: phân bố nhãn dự kiến, độ dài, review theo thời gian/phiên bản.
OUTPUT: `01_data/interim/`.

## GĐ3 — Calibration + đồng thuận + CHỐT guideline (notebook 04, phần A)
QUYẾT ĐỊNH ĐỀ CƯƠNG: gán nhãn MỘT lần theo 4 lớp (BUG/FEATURE/UX/OTHER).
Hai tầng suy ra từ đây, KHÔNG gán lại, KHÔNG tách gold/Hold-out riêng mỗi tầng.
LÀM (thứ tự bắt buộc):
1. Guideline v1 (`annotation_guideline_v1.md`).
2. Rút mẫu calibration 100–150 random từ interim Shopee; ghi review_id vào
   `manifests/` và LOẠI khỏi pool Hold-out (calibration không vào Hold-out — Q9).
3. 2 annotator gán độc lập (ẩn rating/date, không thấy prediction) theo 2 batch
   50–75 → sửa guideline sau batch 1 → đo percent agreement + Cohen κ + confusion
   theo lớp trên batch 2 → adjudicate SAU khi đo.
4. CHỐT guideline (version + hash). Calibration sau adjudicate đưa vào Train.
GATE: κ ≥ 0.6 (không là tiêu chí duy nhất — soi bất đồng FEATURE/UX) mới chốt guideline.

## GĐ4 — Pilot → Hold-out size → splits → gán nhãn diện rộng (notebook 03, rồi 04 phần B) — GATE khóa Hold-out
LÀM (thứ tự bắt buộc):
1. Pilot prevalence 300–500 review random thuần từ pool (trừ calibration), gán
   double bằng guideline ĐÃ CHỐT → ước p_rare (lớp hiếm nhất). Pilot ⊂ Hold-out (Q9).
2. Tính kích thước Hold-out: N ≈ 80 / p_rare, trần 1.000 (Q8).
3. Tạo splits: `shopee_holdout.csv` (random thuần, gồm pilot), `shopee_validation.csv`
   (300–400, stratified 4 lớp), `shopee_train.csv` (~1.200 khởi đầu, random +
   keyword-enriched), `lazada_external_test.csv` (500, random).
4. Ghi checksum + row counts + sampling rule vào `manifests/` và `sampling_manifest.md`.
5. Gán nhãn diện rộng: Hold-out + Lazada double-label toàn bộ, đo agreement TRƯỚC
   adjudication; Train/Validation annotator chính + ≥15–20% double-label audit →
   cập nhật `gold/provenance.csv`.
GATE: từ bước 3, `shopee_holdout` và `lazada_external_test` tuân `HOLDOUT_LOCK.md`.

## GĐ5 — Learning curve (notebook 05, GPU)
LÀM: Train 400→800→1200→1600→2000→… , mỗi mốc 3 seeds (42/1337/2024).
Tier 1: actionable-class F1 ; Tier 2: per-class F1. Dừng theo plateau rule
viết trước (Val cải thiện < ~1 điểm ở hai lần tăng liên tiếp, và F1 FEATURE/UX
không còn tăng thực chất). Validation CỐ ĐỊNH, không đổi theo mốc.
OUTPUT: `05_experiments/learning_curve/`.

## GĐ6–9 — RQ1, ba chiến lược, hai tầng (notebook 06–09)
- 06 Tier 1 ACTIONABLE/NOT cho A/B/C — tối ưu Recall của ACTIONABLE.
- 07 Tier 2 BUG/FEATURE/UX trên gold-actionable cho A/B/C.
- 08 Chiến lược C (LLM few-shot) — ví dụ từ Train/Val, prompt khóa (`04_prompts/`).
- 09 Baseline phẳng 4 lớp (reference only).
Chiến lược: A = Bi-LSTM + PRETRAINED fastText | B = fine-tuned ViSoBERT |
C = LLM few-shot. LƯU prediction để GĐ10 ghép chéo mà không train lại.
GATE: TUYỆT ĐỐI chưa chạm Hold-out.

## GĐ10 — Tổng hợp RQ1 (notebook 10) — GATE mở Hold-out 1 lần
LÀM: per-tier; hybrid 3×3 (ghép chéo tầng 1 model X với tầng 2 model Y từ
prediction đã lưu); end-to-end Macro-F1 4 lớp trên Hold-out; so baseline phẳng;
log time/1k + API cost + train time + hardware.
Quy tắc chọn (viết trước): loại phương án bị dominated; khi hiệu năng tương đương
trong khoảng bất định, ưu tiên TỔNG CHI PHÍ và yêu cầu triển khai thấp hơn.
GHI ngày mở Hold-out + commit hash vào `HOLDOUT_LOCK.md`.

## GĐ11 — Phân loại toàn corpus (notebook 11)
Dùng cấu hình thắng → chạy cascade trên toàn bộ corpus → tập actionable
(`01_data/processed/` hoặc experiments tùy quy mô).

## GĐ12 — Gom cụm + loại chủ đạo (notebook 12)
Chỉ gom actionable. Sentence embeddings + BERTopic. Chọn min_topic_size theo
coherence. Gắn loại chủ đạo (đa số ngưỡng ~60%; dưới ngưỡng → "mixed" + cờ).
Báo type-purity. RQ2: coherence (định lượng) + kiểm tra bằng mắt.

## GĐ13 — Temporal validation (notebook 13)
date/app-version → ACTIVE / RECURRING / POSSIBLY_RESOLVED / INSUFFICIENT.
Mặc định chỉ ACTIVE & RECURRING sang sinh requirement.

## GĐ14 — Sinh requirement (notebook 14)
Cụm ACTIVE/RECURRING → User Story + Acceptance Criteria (TIẾNG ANH, để chạy
AQUSA) + Evidence IDs (review_id nguồn). Template theo loại chủ đạo. Prompt khóa.
LƯU Ý: test mẫu nhỏ + xác nhận trước khi chạy API trên toàn bộ.

## GĐ15 — Đánh giá requirement (notebook 15)
AQUSA + rules (cấu trúc) + human semantic sample (đúng ngữ nghĩa so review nguồn).
OUTPUT: `05_experiments/rq3_requirements/` + `07_human_eval/`.

## GĐ16 — Dựng artifact D6 (notebook 16)
Xuất 5 bảng quan hệ ra `06_output_data_model/`:
reviews / topics / requirements / requirement_evidence / model_comparison.
Power BI (tùy chọn) đọc từ 5 bảng này.

---

## Nguyên tắc xuyên suốt
- Chi phí là tiêu chí hạng nhất ngang hiệu năng; đo thực tế, không hardcode.
- Logic ở `src/`, notebook chỉ import. Không script rời ngoài `src/`.
- Lỗi: dán nguyên lỗi + giải thích tiếng Việt trước khi thử cách khác.
