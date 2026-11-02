# D3_rq1_experiment_package

Learning curve, metrics hai tầng, hybrid 3×3, time/cost, baseline.

| Thành phần | Vị trí |
|---|---|
| Learning curve (3 seed) | `05_experiments/learning_curve/` |
| Kết quả Validation, ngưỡng tầng 1, LLM few-shot | `05_experiments/rq1_classification/` (tier1_*, flat_*, llm_*) |
| Kết quả Hold-out (tầng 1, tầng 2, end-to-end 4 lớp, hybrid 3×3, Lazada, CI, chi phí) | `05_experiments/rq1_classification/gd10_*.csv`, `gd10_run_info.json` |
| Bảng so sánh dạng dài | `06_output_data_model/model_comparison.csv` |
| Config/prompt đã khóa | `03_configs/frozen/{tier1,tier2,flat}_config.yaml`, `classification.yaml`; `04_prompts/classification_fewshot/` |
| Notebook | `notebooks/05`–`10b` |
