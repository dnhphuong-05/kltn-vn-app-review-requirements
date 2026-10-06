# Timeline

Protocol FROZEN v1.1 — 06/10/2026. T0 (crawl) = 2026-10-06 (amendment v1.3). Hạn nộp 22/11/2026.

| Tuần | Ngày | Việc chính | Gate |
|---|---|---|---|
| 1 | 06–12/10 | Freeze (06/10) → crawl GĐ1 06/10 (T0, phủ ≥12 tháng) → clean+EDA → calibration + kappa + chốt guideline → pilot → make splits | Freeze; khóa Hold-out |
| 2 | 13–19/10 | Gán nhãn diện rộng (train ~1200 + holdout + validation + Lazada). Song song viết src/ cho A/B/C + clustering | — |
| 3 | 20–26/10 | RQ1 trên Train/Val: learning curve → tier1 → tier2 → LLM few-shot → baseline | Chưa mở Hold-out |
| 4 | 27/10–02/11 | GĐ10 mở Hold-out 1 lần (RQ1 + hybrid 3×3 + cost + CI) → full inference → BERTopic + loại chủ đạo → temporal | Mở Hold-out |
| 5 | 03–09/11 | Sinh requirement → AQUSA + human sample → output data model → buffer | — |
| viết | 09–21/11 | Kết quả + thảo luận + hạn chế (mở đầu/tổng quan/phương pháp viết sớm từ tuần 3) | — |
| nộp | 22/11 | Nộp | — |

**Cắt nếu trễ (theo thứ tự):** gom-riêng từng loại → LLM thứ hai → dò hyperparameter rộng → Power BI.
**GIỮ:** hybrid 3×3, bootstrap CI.
