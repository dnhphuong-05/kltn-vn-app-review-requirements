# D4_topic_package

Topic model + dominant type + purity + temporal status.

| Thành phần | Vị trí |
|---|---|
| Cụm (17 + outlier), loại chủ đạo, type purity, tên LLM, từ khóa | `05_experiments/rq2_topics/topics.csv`, `review_topic_map.csv`, `gd12_*` |
| Manual topic purity (2 người chấm) | `05_experiments/rq2_topics/gd12_manual_purity*.{csv,json}`, `07_human_eval/topic_purity/README.md` |
| Temporal validation (luật v2.5, độ nhạy) | `05_experiments/rq2_topics/gd13_*` |
| Hình (lưới C_v, bản đồ cụm, thành phần, heatmap temporal) | `05_experiments/rq2_topics/figures/` |
| Config đã khóa (kèm sha256 kết quả) | `03_configs/frozen/rq2_topics_config.yaml`, `03_configs/frozen/temporal_config.yaml`; prompt đặt tên `04_prompts/topic_naming/` |
| Notebook | `notebooks/11`, `12`, `13` |
