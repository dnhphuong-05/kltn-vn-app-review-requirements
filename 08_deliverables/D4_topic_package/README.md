# D4_topic_package

Ví dụ hướng phát triển: chọn cấu hình phân loại, phân loại toàn corpus, gom cụm review actionable, loại chủ đạo và purity. Phần này
nằm ngoài phạm vi nghiên cứu phân loại; kiểm tra tính hợp lệ theo thời gian, sinh Candidate Requirement và mô hình dữ liệu đầu ra
(D5, D6) đã được gỡ khỏi repo theo amendment v4.0.

| Thành phần | Vị trí |
|---|---|
| Phân loại toàn corpus bằng cascade ViSoBERT | `05_experiments/rq2_topics/gd11_*` |
| Cụm (17 + outlier), loại chủ đạo, type purity, tên LLM, từ khóa | `05_experiments/rq2_topics/topics.csv`, `review_topic_map.csv`, `gd12_*` |
| Manual topic purity (2 người chấm) | `05_experiments/rq2_topics/gd12_manual_purity*.{csv,json}`, `07_human_eval/topic_purity/README.md` |
| Hình (lưới C_v, bản đồ cụm, thành phần, từ khóa, phân cấp) | `05_experiments/rq2_topics/figures/` |
| Config đã khóa (kèm sha256 kết quả) | `03_configs/frozen/rq2_topics_config.yaml`; prompt đặt tên `04_prompts/topic_naming/` |
| Notebook | `notebooks/21_future_full_inference.ipynb`, `notebooks/22_future_bertopic_clustering.ipynb` |
