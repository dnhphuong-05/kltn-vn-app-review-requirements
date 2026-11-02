# D5_candidate_requirements

Candidate Requirements + evidence + đánh giá chất lượng.

| Thành phần | Vị trí |
|---|---|
| 16 Candidate Requirement (User Story + 2–3 AC Given/When/Then, tiếng Anh) | `05_experiments/rq3_requirements/requirements.csv` |
| Evidence (93 review_id, thứ hạng gần tâm) | `05_experiments/rq3_requirements/requirement_evidence.csv` |
| Unit sinh, tập evidence của từng review, cụm chuyển BA | `gd14_generation_units.csv`, `gd14_evidence_pool.csv`, `gd14_ba_manual_review.csv` |
| Sàng lọc cụm 0/8 + QA | `gd14_scope_screening*.csv`, `gd14_scope_screening_qa_summary.csv`, `07_human_eval/scope_screening/README.md` |
| Log lượt gọi LLM (không văn bản review), chi phí thật | `gd14_*_calls.jsonl`, `gd14_run_info.json` |
| Đánh giá tự động (QUS theo AQUSA, GWT, evidence-ID, trùng lặp) | `gd15_*.csv`, `gd15_run_info.json` |
| Chấm tay (fidelity, actionability, AC testability, hallucination, evidence precision) | `07_human_eval/rq3_requirements/README.md`; kết quả `gd15_human_*.csv` sau khi chấm |
| Config/prompt đã khóa | `03_configs/frozen/rq3_generation_config.yaml`; `04_prompts/requirement_generation/` (hash trong `prompts_registry.md`) |
| Notebook | `notebooks/14_rq3_generation.ipynb`, `notebooks/15_rq3_evaluation.ipynb` |
