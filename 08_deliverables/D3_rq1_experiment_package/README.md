# D3_rq1_experiment_package

Thực nghiệm phân loại: learning curve, metrics hai tầng, flat / cascade / hybrid, chi phí, các phân tích bổ sung.

| Thành phần | Vị trí |
|---|---|
| Learning curve (3 seed) | `05_experiments/learning_curve/` |
| Kết quả Validation, ngưỡng tầng 1, LLM few-shot | `05_experiments/rq1_classification/` (tier1_*, tier2_*, flat_*, llm_*) |
| Hold-out giai đoạn 1 (tầng 1, tầng 2, end-to-end 4 lớp, hybrid 3×3, Lazada, CI, chi phí) | `05_experiments/rq1_classification/gd10_*.csv`, `gd10_run_info.json` |
| Giai đoạn 2 (TF-IDF, PhoBERT, flat Bi-LSTM, flat LLM, chấm chung, prompt trung tính, phân tích lỗi Lazada, tổng chi phí) | `05_experiments/rq1_classification/paper_*` |
| Phiếu mã hóa thủ công lỗi giữ nhầm OTHER | `07_human_eval/paper_lazada_errors/` (phiếu có văn bản review: không commit) |
| Config/prompt đã khóa | `03_configs/frozen/{tier1,tier2,flat}_config.yaml`, `classification.yaml`, `paper_*.yaml`; `04_prompts/classification_fewshot/` |
| Notebook | `notebooks/05`–`20` |
