# Research Protocol (FROZEN) — v1.1

> Đóng băng trước khi phân tích có hỗ trợ model. Mọi thay đổi ghi ở `protocol_amendments.md`.
>
> **Trạng thái: FROZEN v1.1 — 06/10/2026.** Q1–Q19 đã chốt (xem `protocol_amendments.md` v1.1).
> Đã áp amendment **v1.2 (06/10/2026)**: quy tắc cụm mixed (Q14), trần crawl 120.000, thêm trường reply.
> Đã áp amendment **v1.3 (06/10/2026)**: T0 = 2026-10-06 (ngày crawl thực tế), cutoff 2025-10-06, 3 cửa sổ temporal theo T0 mới.
> Đã áp amendment **v1.4 (06/10/2026)**: Shopee crawl stratified theo rating; Hold-out giữ phân bố đã gom; scrub PII khi ghi raw.
> Đã áp amendment **v1.5 (07/10/2026)**: corpus Shopee gộp theo mức sao được xác nhận near-natural (xấp xỉ toàn bộ review của cửa sổ 12 tháng).
> Đã áp amendment **v1.7 (13/10/2026)**: Hold-out 1.000 (400 pilot + 600), focused FEATURE/UX set ~150 (khóa, báo cáo riêng), Validation ~350, Train ~1.200 single-label + audit 15%, Lazada 500.
> Đã áp amendment **v1.8 (23/10/2026)**: Validation cân lại bằng hoán đổi 15 FEATURE ↔ 15 OTHER với Train (Validation 22 FEATURE, Train 129).
> Đã áp amendment **v1.9 (23/10/2026)**: kết luận learning curve — dừng Train ở 1.200 (tầng 1 tăng chậm; tầng 2 chưa bão hòa nhưng pool FEATURE cạn); F1 FEATURE đo bổ sung trên focused set. Số ViSoBERT cập nhật 23/10/2026 sau khi sửa bug tokenizer (ghi chú thực thi GĐ6), kết luận giữ nguyên.
> Đã áp amendment **v2.0 (23/10/2026)**: Q5 — sàn recall tầng 1 hạ từ 0,90 xuống 0,80, quét ngưỡng 0,01–0,90 (chốt sau khi xem Validation, Hold-out vẫn khóa).
> Đã áp amendment **v2.1 (25/10/2026)**: đóng băng toàn bộ config RQ1 (cấu hình, ngưỡng, prompt hash) vào `03_configs/frozen/` trước khi mở Hold-out; GĐ10 train lại từ config này, không tune lại.
> Đã áp amendment **v2.2 (29/10/2026)**: chiến lược phân loại downstream (GĐ11, RQ2/RQ3) là cascade ViSoBERT frozen v2.1 — quyết định cost-aware đưa ra sau Hold-out, ngoài phạm vi quy tắc chọn viết trước (LLM hơn có ý nghĩa thống kê); hạn chế recall tầng 1 Hold-out 69,8 mang sang RQ2/RQ3.
> Đã áp amendment **v2.3 (29/10/2026)**: RQ2 — chỉ review ≥ 3 token nội dung định hình cụm (review ngắn gắn sau theo cosine), ngưỡng outlier theo phân vị 10 thành viên–tâm, gộp cụm theo cặp, đặt tên LLM prompt v4 (mọi cụm trong một lượt gọi, tên phân biệt được, cờ scope chỉ tham khảo).
> Đã áp amendment **v2.4 (01/11/2026)**: khóa cấu hình và kết quả RQ2 (17 cụm, khóa theo sha256 file) và chốt thủ tục manual topic purity (10 review/cụm, 2 người chấm, cờ BA < 0,60) trước khi chấm.
> Đã áp amendment **v2.5 (01/11/2026)**: pre-register luật Temporal Issue Validation (N ≥ 20; c30 ≥ 5 và R90 ≥ 0,5 → RECURRING nếu ≥ 6/12 cửa sổ 30 ngày, ngược lại ACTIVE; R90 < 0,5 → POSSIBLY_RESOLVED) trước khi tính trạng thái của cụm.
> Đã áp amendment **v2.6 (01/11/2026, cập nhật 02/11/2026)**: GĐ14 — chuyển BA cụm purity < 0,25 (9, 2, 13, 4); cụm 0 và 8 không loại mà chỉ sinh từ review nêu chức năng app; lọc evidence theo loại cho cụm 0,25–0,60 (11, 12, 14; cụm mixed 7 theo Q14); sinh bình thường 7 cụm còn lại.
> Đã áp amendment **v2.7 (02/11/2026)**: GĐ14 — sàng lọc cụm 0 và 8 bằng LLM prompt khóa v2 (QA 40 review: đồng ý 87,5%, κ = 0,75; recall app_actionable cụm 0 ước tính 0,64 — giới hạn); cụm 0 tách theo loại sau lọc, temporal tính lại theo unit; 2–3 Acceptance Criteria Given/When/Then.
> Đã áp amendment **v2.8 (02/11/2026)**: GĐ14 — prompt sinh requirement v3 là bản cuối (config frozen); không thêm bước LLM kiểm evidence; fidelity và hallucination đo bằng human sample ở GĐ15; output là Candidate Requirement cần BA duyệt.
> Đã áp amendment **v2.9 (02/11/2026)**: GĐ15 — chấm tay toàn bộ 16 requirement (thay mẫu 30–50 vì chỉ có 16) + chấm 93 review trích dẫn (evidence precision); luật QUS cài đặt lại theo AQUSA; thang chấm chốt trước khi chấm.
> Chỗ còn "pre-register sau EDA" (ngưỡng temporal) là nội dung đề cương yêu cầu chốt sau EDA
> trên non-test data — sẽ ghi bằng amendment, không phải chỗ trống.
>
> Nguồn: đề cương V5 (`09_thesis/proposal/`, bản 29/09/2026). Ký hiệu "ĐC §x" = mục trong đề cương.

## Research Questions (ĐC §4.3)
- **RQ1** — Trong kiến trúc phân tầng (Tier 1: ACTIONABLE/NOT; Tier 2: BUG/FEATURE/UX),
  ba chiến lược (Bi-LSTM+fastText | ViSoBERT fine-tune | LLM few-shot) khác nhau thế nào ở
  từng tầng về hiệu năng, thời gian, chi phí, yêu cầu triển khai?
- **RQ2** — Các review actionable có thể được gom thành issue groups nhất quán và được xác nhận
  còn phù hợp về mặt thời gian ở mức độ nào trước khi dùng làm bằng chứng để sinh Candidate Requirement?
- **RQ3** — Candidate Requirements do hệ thống sinh ra đáp ứng structural quality, semantic
  correctness và source traceability ở mức độ nào?

## Taxonomy (cascade) — một bộ gold 4 lớp, suy ra hai tầng (ĐC §5, §7.2, §7.3)
- Gán nhãn MỘT lần theo 4 lớp BUG / FEATURE / UX / OTHER (định nghĩa: `annotation_guideline_v1.md`).
- Tier 1: ACTIONABLE = BUG ∪ FEATURE ∪ UX ; NOT-ACTIONABLE = OTHER
- Tier 2: BUG / FEATURE / UX — chỉ trên phần ACTIONABLE.
- Không gán lại cho từng tầng; dùng chung một Gold Dataset và một Hold-out cho cả hai tầng.
- Review `ambiguous` giữa BUG/FEATURE/UX: là ACTIONABLE hợp lệ → dùng cho train/eval Tier 1;
  KHÔNG dùng train Tier 2 ở vòng đầu, giữ cho error analysis (ĐC §7.2).
  Hold-out: mọi dòng (kể cả ambiguous) có MỘT gold_label sau adjudication và được tính vào
  metric chính của cả hai tầng (Q11). ACTIONABLE chỉ khi review chỉ ra việc cụ thể dev xử lý được;
  không thì OTHER.
- Head phân loại: Tier 1 = 2 lớp, Tier 2 = 3 lớp. Head "Softmax 4 lớp" ở ĐC §7.3.1/§7.3.2
  chỉ áp dụng cho baseline phẳng.
- Chỉ BUG/FEATURE/UX đi tiếp downstream.

## Chiến lược RQ1 (ĐC §7.3.1–7.3.3)
- **A** — Bi-LSTM (1–2 layer) + **pretrained fastText tiếng Việt 300d** (Common Crawl + Wikipedia);
  word segmentation nhất quán; pooling mean/max hoặc final hidden; weighted CE nếu lệch lớp.
  Self-trained fastText chỉ là ablation tùy chọn (nếu làm: chỉ train trên pool không chứa test).
- **B** — Fine-tuned ViSoBERT; tokenizer gốc; LR/batch/max_len/epochs/weight decay tune trên Validation;
  weighted loss hoặc sampler chọn bằng Validation.
- **C** — LLM few-shot qua API, không cập nhật trọng số; prompt = taxonomy + decision tree +
  JSON schema + rule edge case + exemplars. Retry/parse error/refusal/malformed JSON đều log và
  tính vào runtime/cost; không sửa nhãn bằng tay.
- **Baseline phẳng 4 lớp**: ViSoBERT (mặc định theo ĐC §7.3.4) — reference only, không vào downstream.
- Input mô hình = text review. Rating/date/version chỉ là metadata, không làm feature.

## Metrics (khóa trước Hold-out) (ĐC §7.3.4, §10)
- Tier 1: F1 + **Recall của ACTIONABLE** (ưu tiên recall). Ngưỡng chỉnh trên Validation (Q5):
  trong các ngưỡng đạt **Recall_ACTIONABLE ≥ 0.80** (quét 0.01–0.90, bước 0.01; amendment v2.0, trước đó
  0.90), chọn ngưỡng có **Macro-F1 (2 lớp) cao nhất** trên lần chạy seed 42 (Q6), giữ nguyên cho các seed
  khác và cho Hold-out; báo kèm recall thấp nhất qua 3 seed.
  Không ngưỡng nào đạt 0.80 → chọn ngưỡng có recall cao nhất và báo cáo. LLM (C) không có ngưỡng
  xác suất → dùng nhãn trực tiếp.
- Tier 2: Macro-F1 (3 lớp) + per-class P/R/F1 trên **gold-actionable** (tách khỏi lỗi Tier 1).
- End-to-end: Macro-F1 (4 lớp) = (F1_BUG + F1_FEATURE + F1_UX + F1_OTHER)/4 trên toàn Hold-out;
  confusion matrix 4×4; so với baseline phẳng. Nếu cascade không vượt baseline rõ ràng → báo cáo trung thực.
- Hybrid 3×3: Tier 1 của X × Tier 2 của Y, ghép từ prediction đã lưu, không train lại;
  báo end-to-end Macro-F1 + tổng chi phí.
- Tổng chi phí một cấu hình = chi phí Tier 1 trên TOÀN BỘ review + chi phí Tier 2 CHỈ trên phần
  Tier 1 dự đoán ACTIONABLE.
- Thời gian/chi phí (đo thực, không hardcode): inference wall-clock / 1.000 review (ghi warm-up,
  batch size); training time; LLM = tokens × giá API tại ngày chạy; local = môi trường compute +
  thời gian (không ghi "chi phí 0"); hardware/GPU, peak memory nếu có; giờ gán nhãn.
- Deployment indicators (không gộp Low/Med/High): nhu cầu dữ liệu train, GPU, dung lượng model,
  phụ thuộc API, chi phí lặp lại, nhu cầu retrain, khả năng tái lập/version, privacy/data transfer.
- Uncertainty (Q15, **bắt buộc**): bootstrap trên Hold-out, ~10.000 resample (seed 42) → 95% CI cho
  Macro-F1 (Tier 2 3 lớp, end-to-end 4 lớp), F1/Recall ACTIONABLE (Tier 1) và chênh lệch giữa các cấu hình.
  "Tương đương trong khoảng bất định" = CI của chênh lệch chứa 0.
- Cross-app: chạy model đã freeze trên Lazada External Test, không tune lại; cùng bộ metric như trên.
- Early stopping / model selection trên Validation: Tier 1 theo F1 ACTIONABLE, Tier 2 theo Macro-F1 3 lớp,
  baseline phẳng theo Macro-F1 4 lớp.
- **Quy tắc chọn cấu hình downstream (viết trước):** loại phương án bị dominated rõ ràng; nếu hiệu năng
  tương đương trong khoảng bất định thì ưu tiên tổng chi phí và yêu cầu triển khai thấp hơn.
  Kết luận ở dạng "phù hợp nhất cho bối cảnh này", không phải "tốt nhất nói chung".

## Seeds / môi trường
- seed = 42 (mặc định mọi notebook, `set_seed(42)`); 3 seeds cho learning curve và cho A/B ở RQ1: 42, 1337, 2024.
- Báo cáo A/B: mean ± std qua 3 seeds; không ensemble seeds (ĐC §17).
  **Seed 42 là run chuẩn** (Q6): mọi prediction lưu cho hybrid 3×3, bootstrap CI, full inference và
  downstream lấy từ seed 42; seed 1337/2024 chỉ để báo mean ± std.
- **LLM (Q1–Q3):** OpenAI `gpt-4o-mini`, gọi bằng snapshot pin **`gpt-4o-mini-2024-07-18`**,
  **temperature = 0**. Dùng CHUNG một model cho phân loại (C), sinh requirement và đặt tên topic.
  prompt_hash: xem `04_prompts/prompts_registry.md`. Ghi kèm ngày chạy + bảng giá tại thời điểm chạy.
  Cấu hình đọc từ env (`.env.example`) qua `src.utils.llm`.
- **Sentence embedding (Q13):** `paraphrase-multilingual-mpnet-base-v2` (sentence-transformers), pin
  commit revision khi tải lần đầu và ghi vào config. Được đổi sang một Vietnamese SBERT sau kiểm tra
  định tính trên non-test data — nếu đổi phải ghi amendment kèm lý do, TRƯỚC khi khóa cấu hình BERTopic.
- Environment lock: `requirements.txt`; ghi version Python, PyTorch, Transformers, BERTopic, tokenizer.

## Dữ liệu (ĐC §5, §7.2.2, §7.2.3)
- Nguồn: Google Play. Main: Shopee (~41.7k) ; External test: Lazada (~3.0k). Số chính xác ghi ở
  `01_data/manifests/` sau GĐ1. **T0 = 2026-10-06** (ngày crawl thực tế, Q18; amendment v1.3). Độ phủ dữ liệu ≥ 12 tháng:
  dừng theo NGÀY khi vượt cutoff **2025-10-06**; nghiệm thu GĐ1 khi min(review_date) ≤ 2025-10-06 cho CẢ HAI app
  (không ép hai app cùng số lượng).
  - **Shopee (amendment v1.4):** crawl `Sort.NEWEST` tách theo `filter_score_with` = 1..5 (Google Play giới hạn
    phân trang ~40.000/lượt), mỗi mức sao dừng theo cutoff, rồi gộp + khử trùng theo `review_id`. Manifest ghi
    min/max review_date và lý do dừng của TỪNG mức sao.
  - **Lazada:** `Sort.NEWEST`, không lọc sao.
  - **PII:** scrub ngay khi ghi raw — tên sau lời chào/cảm ơn → `[NAME]`, số điện thoại → `[PHONE]`, email →
    `[EMAIL]`, áp cho `text` và `reply_content`; `userName` → sha256[:12]; không lưu `userImage`.
- **Thiết lập phân bố (amendment v1.4, cập nhật v1.5):** corpus Shopee được thu thập theo từng mức sao rồi gộp.
  Vì không mức sao nào bị chặn phân trang trước cutoff, corpus gộp xấp xỉ TOÀN BỘ review của cửa sổ 12 tháng
  và có phân bố **gần tự nhiên (near-natural)** — bằng chứng ở mục Hạn chế. Pilot, Hold-out, Validation, Train
  đều rút từ corpus đã gộp nên CÙNG phân bố; KHÔNG tái cân bằng bằng trọng số. "Random thuần" của Hold-out
  (CLAUDE.md §4) hiểu là random thuần TRÊN corpus đã gộp.
- Làm sạch tối thiểu (giữ phủ định, emoji, dấu câu, viết tắt); loại review hoàn toàn không phải tiếng Việt.

| Tập | Kích thước | Cách tạo | Gán nhãn |
|---|---|---|---|
| Random pilot (ước prevalence) | 300–500 | Random thuần từ clean corpus Shopee (corpus gộp theo mức sao, near-natural — v1.5), gán bằng guideline ĐÃ CHỐT. Prevalence dùng tính kích thước Hold-out; đồng thời (cập nhật v1.5) vì corpus near-natural, prevalence này XẤP XỈ tỷ lệ actionable trong review Google Play 12 tháng của app và được báo cáo như ước lượng KÈM khoảng tin cậy 95%. Ranh giới: đây là tỷ lệ trong REVIEW (self-selection bias — người gặp vấn đề có xu hướng viết review nhiều hơn), KHÔNG phải tỷ lệ trong toàn bộ người dùng; không suy ra "X% người dùng gặp vấn đề" | Double-label; **nằm trong Hold-out** (Q9) |
| Calibration | 100–150 (2 batch × 50–75) | Random, rời với pilot/Hold-out | 2 annotator độc lập; **không vào Hold-out**, adjudicate xong đưa vào **Train** (Q9) |
| Shopee Internal Hold-out | N ≈ m / p_rare với **m = 80**, **trần 1.000** (Q8); gồm cả pilot | Random thuần trên corpus đã gộp, giữ nguyên phân bố của corpus (near-natural, v1.5), KHÔNG tái cân bằng (Q9, v1.4); lấy TRƯỚC mọi model-assisted sampling, không dựa nhãn AI | Double-label toàn bộ; đo agreement trước adjudication |
| Shopee Validation | 300–400, cố định sớm, không đổi theo mốc learning curve | Từ phần còn lại; **stratified theo 4 lớp** (Q8), ghi sampling rule ở `sampling_manifest.md` | ≥15–20% double-label audit |
| Shopee Train | Khởi đầu ~1.200; mở rộng theo batch 400 tới ~2.000–3.000 nếu curve còn tăng | Random + keyword-assisted enrichment cho lớp hiếm (phần lớn vẫn random); ghi `sampling_source` | 1 annotator chính + ≥15–20% double-label audit |
| Lazada External Test | **500** (Q8) | Random độc lập | Double-label toàn bộ; không train/tune |

- Tổng nhãn tay planning ~2.500–4.000 (không phải quota).
- Hold-out + Lazada tuân `01_data/HOLDOUT_LOCK.md`.
- Gate agreement (Q10): **Cohen's κ ≥ 0.6** trên calibration batch 2 (4 lớp) mới chốt guideline và nhân rộng;
  κ là cổng nhưng không là tiêu chí duy nhất — soi số ca bất đồng tuyệt đối của FEATURE/UX. Có annotator
  thứ hai thật → inter-annotator agreement chuẩn (hai người gán độc lập calibration + Hold-out).
  Luôn báo cáo đồng thời percent agreement, κ và confusion giữa annotator theo lớp.

## Learning curve (ĐC §7.2.4)
- Validation cố định. Subset lồng nhau 400 → 800 → 1.200 → 1.600 → 2.000 → 2.400 → … (có thể thêm 200);
  mỗi subset đủ cả 4 lớp; 3 seeds (42/1337/2024); báo mean ± std, vẽ cả Train và Validation.
- Metric: Tier 1 = F1 ACTIONABLE ; Tier 2 = per-class F1 (BUG/FEATURE/UX) + Macro-F1 3 lớp.
- Model chạy learning curve (Q7): **ViSoBERT là chính** (mọi mốc × 3 seeds); **Bi-LSTM spot-check 2 mốc**
  để xem xu hướng có khác không. LLM few-shot không có learning curve, không dùng để quyết định "đủ dữ liệu".
- **Quy tắc dừng (viết trước):** plateau khi metric Validation cải thiện < 1 điểm % ở hai lần tăng liên tiếp,
  dải dao động giữa seeds chồng lấp đáng kể, VÀ F1 của BUG/FEATURE/UX không còn tăng thực chất.
  Macro gần phẳng nhưng FEATURE (hoặc lớp hiếm) còn tăng rõ → tiếp tục gán.
- Sau khi dừng: lưu stopping evidence (curve, per-class F1, số mẫu từng lớp) rồi mới tune RQ1 nghiêm túc.

## LLM few-shot (ĐC §7.3.3)
- Exemplars (Q4): **CHỈ từ Train** (Validation là nơi phát triển prompt nên không làm ví dụ). Tuyệt đối
  không lấy từ Hold-out / Lazada.
- **Bộ cố định** cho mọi review: Tier 1 ~4 ví dụ/lớp (ACTIONABLE/NOT), Tier 2 ~3 ví dụ/lớp (BUG/FEATURE/UX).
  Danh sách review_id exemplar + hash được chốt và ghi vào `04_prompts/prompts_registry.md` trước eval.
- Phát triển prompt trên Validation; freeze prompt + schema + exemplars + temperature trước khi mở Hold-out.

## RQ2 — Gom cụm + loại chủ đạo (ĐC §7.4, §10.2)
- Input: toàn bộ review được cấu hình thắng phân loại ACTIONABLE trên full corpus.
- BERTopic: sentence embedding → UMAP → HDBSCAN → c-TF-IDF; giảm outlier; gộp topic trùng;
  đặt tên bằng LLM (tên không đổi membership).
- Thiết kế gom cụm: gom chung toàn bộ actionable rồi gắn loại (theo CLAUDE.md §1) — phương án chính.
  Gom riêng từng loại (ĐC §7.4) chỉ là ablation tùy chọn trên non-test, cắt đầu tiên nếu trễ (Q12).
- Chọn min_topic_size (Q12): ứng viên **{15, 20, 30, 50, 75}**; tiêu chí chính **coherence C_v**, kèm
  outlier rate, số topic và phân bố kích thước làm tín hiệu phụ (loại cấu hình suy biến: quá nhiều cụm
  rất nhỏ hoặc một cụm khổng lồ).
- Loại chủ đạo: một loại ≥ 60% review trong cụm → gán loại đó; không loại nào đạt → `mixed` + cờ cho BA.
  Cụm `mixed` (Q14, sửa ở amendment v1.2): KHÔNG bỏ. Tách review trong cụm theo loại; chỉ sinh requirement
  (template của loại đó, cờ `mixed-origin`) cho loại có **≥ 3 review HOẶC ≥ 10% số review của cụm**.
  Loại dưới cả hai ngưỡng không sinh requirement, chỉ ghi chú `mixed-origin` kèm review_id để BA xem.
- Metrics: coherence; outlier rate; số topic + phân bố kích thước; type-purity; manual topic purity
  (sau khi khóa cấu hình: mỗi topic ~10 review ngẫu nhiên, 2 người chấm; purity = phù hợp / đã kiểm);
  temporal evidence coverage; valid generation rate; traceability coverage.

## Temporal Issue Validation (ĐC §7.5)
- Historical Context Window ≥ 12 tháng: [2025-10-06, 2026-10-06]. Current Evidence Window tính từ
  **T0 = 2026-10-06** (Q18; amendment v1.3):
  - last_30 = [2026-09-06, 2026-10-06]
  - last_60 = [2026-08-07, 2026-10-06]
  - last_90 = [2026-07-08, 2026-10-06]
- Trường: count_30d/60d/90d, first_seen, last_seen, first_version/last_version (nếu version đáng tin), số cửa sổ có issue.
- Trạng thái: ACTIVE / RECURRING / POSSIBLY_RESOLVED / INSUFFICIENT_EVIDENCE (định nghĩa ĐC §7.5.2).
  Mặc định chỉ ACTIVE & RECURRING sang sinh requirement. Không gọi FIXED; không suy nhân quả từ update.
- Ngưỡng cụ thể (count tối thiểu, tỷ lệ recent/historical, số cửa sổ cho RECURRING): pre-register
  SAU EDA trên non-test data, TRƯỚC khi xem kết quả downstream → ghi vào `protocol_amendments.md`.
- Xếp ưu tiên: chỉ issue frequency (số review cụm / tổng actionable) + temporal_status; không dùng Severity/Impact.

## RQ3 — Sinh + đánh giá requirement (ĐC §7.6, §7.7, §10.3)
- Output JSON schema cố định: User Story (As a / I want / so that) + 1–3 Acceptance Criteria
  Given/When/Then — **tiếng Anh**; review nguồn giữ tiếng Việt.
- Evidence IDs chỉ được nằm trong danh sách đưa vào prompt; backend validate (tồn tại, thuộc topic, đúng danh sách input).
- Prompt khóa trong `04_prompts/` + prompt_hash; test mẫu nhỏ + người dùng xác nhận trước khi chạy toàn bộ.
- Đánh giá tự động: AQUSA/QUS-rule violation; Given-When-Then validity; evidence-ID validity.
- Human sample 30–50 requirement, stratify theo BUG/FEATURE/UX và topic size; thang 1–5 cho fidelity,
  actionability, AC testability; hallucination binary hoặc 1–5; chấm dựa trên review nguồn tiếng Việt.
  2 rater → weighted kappa; 1 rater → gọi là expert validation exploratory. Không sửa output trước khi đánh giá tự động.

## Hạn chế đã biết từ thu thập dữ liệu (amendment v1.4, cập nhật v1.5)
- **Phân bố corpus Shopee — near-natural (v1.5, thay cho mô tả "stratified-by-rating, không phải phân bố tự
  nhiên" của v1.4):** corpus thu thập theo từng mức sao (`filter_score_with` 1..5) rồi gộp. Bằng chứng phân bố
  gần tự nhiên: (i) cả 5 mức sao đều dừng do đi qua cutoff (`reached_cutoff`), không mức nào bị chặn phân trang
  hay chạm trần; (ii) trên cùng khoảng 2026-01-12..2026-10-05, corpus gộp khớp 39.991/39.991 review với mẻ
  `Sort.NEWEST` 40.000 không lọc sao (9 review chỉ có ở mẻ NEWEST, khả năng bị xóa/sửa giữa hai lần crawl);
  (iii) phân bố rating trùng: 1★ 40,4% / 2★ 3,4% / 3★ 3,3% / 4★ 5,0% / 5★ 47,9%. Kết luận: corpus xấp xỉ TOÀN BỘ
  review của cửa sổ 12 tháng; mọi metric RQ1 (Tier 1, Tier 2, end-to-end, hybrid, CI) được diễn giải trên phân bố
  gần tự nhiên của Shopee trên Google Play trong cửa sổ này. Giới hạn còn lại: chỉ những review Google Play hiển
  thị tại thời điểm crawl (review đã xóa/ẩn không có mặt).
- **Lazada:** thu thập `Sort.NEWEST` không lọc sao (phân bố tự nhiên). Vì Shopee cũng near-natural, chênh lệch
  phân bố do cách thu thập không đáng kể; khác biệt rating giữa hai app (Lazada 1★ ~60%, Shopee 1★ ~40%) là đặc
  điểm thật của dữ liệu và cần lưu ý khi diễn giải kết quả cross-app.
- **app_version thiếu:** Shopee 26,3%, Lazada 34,2% (mẻ crawl 07/10, `_r2`) → temporal validation dựa chủ yếu vào
  `review_date`; app_version chỉ là bằng chứng phụ khi có.
- **Scrub PII bằng luật:** bắt tên sau lời chào/cảm ơn, số điện thoại, email; có thể sót tên ở vị trí khác →
  QA mẫu ở GĐ2 (ĐC §7.1).

### Ghi nhận chuyển sang GĐ2 (v1.5 — chưa xử lý ở GĐ1)
- Cắt corpus về đúng cửa sổ [2025-10-06 00:00, 2026-10-06 23:59:59] (+07:00): lô cuối mỗi mức sao vượt cutoff nên
  raw có review cũ hơn cutoff (Shopee 831). File raw giữ nguyên, cắt ở `interim/`.
- Ngày 06/10/2026 chỉ có một phần (review mới nhất 06/10 07:18 giờ VN) do Google Play hiển thị trễ ~1 ngày →
  ảnh hưởng nhẹ cửa sổ last_30; ghi vào báo cáo temporal.
- QA mẫu kết quả scrub PII, đặc biệt Lazada (1.940 chỗ thay trong mẻ `_r2`).

## Hạn chế ghi nhận khi tuning RQ1 (GĐ6–GĐ7, 23/10/2026)
- **Số trên Validation lạc quan:** cùng Validation dùng để chọn cấu hình, chọn ngưỡng (tầng 1) và báo kết quả;
  số khách quan là Hold-out (GĐ10).
- **Cấu hình thắng ở mép lưới, không mở rộng lưới:** tầng 2 — ViSoBERT 8 epoch / lr 3e-5, Bi-LSTM 20 epoch /
  lr 2e-3 đều là giá trị lớn nhất đã thử. Không mở rộng để tránh lạc quan hóa trên Validation nhỏ (71 review
  actionable); kết quả có thể cải thiện chút với lưới rộng hơn. Mục tiêu RQ1 là so sánh các chiến lược theo cùng
  quy trình (lưới 1 seed → chọn theo Macro-F1 → 3 seed), không tối đa F1 từng model.
- **Cỡ lưới khác nhau:** Bi-LSTM dò nhiều cấu hình hơn ViSoBERT (tầng 1: 12 so với 9; tầng 2: 24 so với 9) vì huấn
  luyện rẻ hơn; lạc quan do dò nhiều (nếu có) nghiêng về Bi-LSTM.

## Hạn chế ghi nhận ở RQ2 (GĐ12–GĐ14, 02/11/2026)
- **Purity không đồng đều giữa các cụm:** manual topic purity từ 0,05 đến 1,00 (trung bình 0,556; 9/17 cụm < 0,60). Gom cụm theo
  embedding tạo cụm theo chủ đề bề mặt (quảng cáo, giao hàng, mã) nên một cụm có thể lẫn than phiền chung chung và lẫn review trong lẫn
  ngoài phạm vi app; vì vậy sinh requirement không dùng nguyên cụm mà phải lọc review trong cụm (amendment v2.6).
- **Coherence tự động không thay thế người chấm:** Spearman giữa manual purity và C_v của cụm chỉ 0,17 (với type purity 0,49).
- **Chọn min_topic_size ở mép lưới:** C_v tăng đều theo min_topic_size, giá trị chọn (75) là lớn nhất đã thử; lưới chốt trước nên không mở rộng.
- **Một phần purity thấp do tên cụm:** tên do LLM đặt có thể hẹp hoặc mơ hồ hơn nội dung cụm (cụm 4, 13, 14); purity đo cả cụm lẫn tên.
- **Temporal không phân biệt:** với các cụm lớn và lượng review đều theo thời gian, cả 17 cụm đều RECURRING (amendment v2.5).

## Hạn chế ghi nhận ở RQ3 (GĐ14–GĐ15, 03/11/2026)
- **Tổng thể nhỏ:** chỉ 16 Candidate Requirement (16 unit, mỗi unit 1 requirement); chấm toàn bộ thay cho mẫu 30–50 (amendment v2.9);
  so sánh theo loại hay theo nguồn unit chỉ có 3–7 requirement mỗi nhóm, không kiểm định thống kê.
- **Đồng thuận thấp ở thang mức requirement:** weighted κ 0,28 (fidelity), 0,05 (actionability), 0,25 (AC testability); hallucination
  κ 0,19. Người chấm A khắt khe hơn B một cách hệ thống (fidelity thấp hơn ở 14/16 requirement), thứ hạng fidelity tương quan vừa
  (Spearman 0,61); các điểm này báo kèm từng người chấm. Phán đoán mức review trích dẫn đồng thuận cao (κ 0,78) — evidence precision 0,78
  là số đo đáng tin nhất.
- **Hallucination không loại được bằng prompt:** 5/16 requirement bị cả hai người chấm xác định có chi tiết bịa, 13/16 bị ít nhất một
  người xác định (thêm màn hình hay bước, đổi bối cảnh, diễn giải lệch nghĩa, thêm cam kết ngoài phạm vi app); một model cố định cho mọi
  bước (gpt-4o-mini, temperature 0), không dùng LLM kiểm LLM (amendment v2.8).
- **Chất lượng phụ thuộc độ thuần của cụm nguồn:** evidence precision 0,96 ở unit từ cụm bình thường so với 0,39–0,53 ở unit lọc chức năng
  app hoặc tách loại (cụm 0, 7, 8); ba requirement FEATURE đều thuộc nhóm này nên điểm theo loại bị lẫn với nguồn gốc unit.
- **Sàng lọc phạm vi bỏ sót:** recall app_actionable của cụm 0 ước tính 0,64 (QA 40 review), nên tần suất issue của cụm 0 bị đếm thấp
  (amendment v2.7).
- **Structural quality cao không bảo đảm đúng nội dung:** 15/16 User Story không lỗi QUS và 47/47 AC đúng dạng Given/When/Then, nhưng
  nhiều AC dùng từ định tính thiếu ngưỡng; output là Candidate, BA duyệt trước khi dùng.

## Kiểm soát leakage & tái lập (ĐC §11, Phụ lục A)
- Annotator không thấy prediction model. Hold-out không vào few-shot / prompt-development log.
- Manifest checksum + số dòng từng split. Mọi bảng kết quả sinh từ code.
- Sau khi mở Hold-out: không retune. Bug code → ghi incident, sửa chung, chạy lại TOÀN BỘ model cùng protocol.

---

## Các quyết định đã chốt (Q1–Q19, 06/10/2026)
Q1–Q3 LLM · Q4 few-shot · Q5 ngưỡng Tier 1 · Q6 seed chuẩn · Q7 learning curve · Q8 kích thước splits ·
Q9 dùng lại pilot/calibration · Q10 gate κ · Q11 ambiguous · Q12 gom cụm · Q13 embedding · Q14 cụm mixed ·
Q15 bootstrap · Q16/Q17 guideline (xem `annotation_guideline_v1.md`) · Q18 T0 · Q19 ngày amendment.
Chi tiết thay đổi: `protocol_amendments.md` v1.1.
