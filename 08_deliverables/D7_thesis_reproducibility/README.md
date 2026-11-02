# D7_thesis_reproducibility

Protocol freeze, env lock, seeds, commit hash, hướng dẫn tái lập.

## Thành phần
| Thành phần | Vị trí |
|---|---|
| Protocol đóng băng + mọi amendment (có version, ngày, lý do) | `00_protocol/research_protocol.md`, `00_protocol/protocol_amendments.md` |
| Runbook từng giai đoạn | `00_protocol/QUY_TRINH_CHUAN.md` |
| Config đã khóa / prompt đã khóa + hash | `03_configs/frozen/`, `04_prompts/prompts_registry.md` |
| Môi trường | `requirements.txt`; phiên bản thư viện thực tế của từng lượt chạy trong `05_experiments/**/gd*_run_info.json` |
| Seed | `set_seed(42)` ở cell đầu mọi notebook; learning curve và RQ1 lặp 3 seed; LLM `seed=42`, temperature 0, snapshot `gpt-4o-mini-2024-07-18` |
| Commit của từng lượt chạy | trường `commit` trong `gd*_run_info.json`; mở Hold-out: `01_data/HOLDOUT_LOCK.md` |

## Tái lập
1. Dữ liệu: `01_data/raw/` không phân phối (còn tên người dùng); bản interim đã scrub dựng lại từ raw bằng notebook 02. Chỉ dùng bản
   interim trên Kaggle.
2. Chạy notebook theo thứ tự trong `00_protocol/QUY_TRINH_CHUAN.md` (05–07, 10a, 11, 12 cần GPU/Kaggle; 08, 10b, 14 gọi API).
3. Notebook có cache kết quả: khi file kết quả đã có thì không gọi API hay huấn luyện lại; xóa file kết quả để chạy lại thật.
4. Kết quả GĐ12 được khóa bằng sha256 (`03_configs/frozen/rq2_topics_config.yaml`); notebook 12, 14 dừng nếu không khớp.
5. Khóa API: `.env` (local) hoặc Kaggle Secrets; không ghi vào log.
