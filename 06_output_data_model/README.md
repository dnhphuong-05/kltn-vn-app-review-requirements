# D6 — Structured Output (độc lập công cụ)

Năm bảng quan hệ CSV (UTF-8 có BOM, mở thẳng bằng Excel hoặc bất kỳ công cụ BI nào). Dựng bằng
`notebooks/16_build_output_data_model.ipynb` (logic ở `02_src/src/output_layer/core.py`) từ kết quả đã khóa trong `05_experiments/`;
không dự đoán hay tính lại. Mô tả từng cột: `data_dictionary.csv`. Thông tin lượt dựng: `build_info.json`.

| Bảng | Một dòng là | Số dòng | Ghi chú |
|---|---|---|---|
| `reviews.csv` | một review của corpus Shopee (GĐ11) | 56.050 | **Có văn bản review → không commit** (`.gitignore`); chạy lại notebook 16 để tạo |
| `topics.csv` | một cụm GĐ12 (kể cả outlier -1) | 18 | tên, loại chủ đạo, type/manual purity, temporal, cách GĐ14 xử lý |
| `requirements.csv` | một Candidate Requirement | 16 | User Story + AC (tiếng Anh), issue frequency, cờ mixed-origin, lỗi QUS; cột `eval_*` thêm sau chấm tay GĐ15 |
| `requirement_evidence.csv` | một review được trích làm evidence | 93 | `eval_supports` thêm sau chấm tay GĐ15 |
| `model_comparison.csv` | một chiến lược × tầng (RQ1, Hold-out) | 10 | F1, recall, thời gian, chi phí, phần cứng, Lazada |

## Khóa liên kết
`reviews.topic_id → topics.topic_id` ; `requirements.topic_id → topics.topic_id` ;
`requirement_evidence.requirement_id → requirements.requirement_id` ; `requirement_evidence.review_id → reviews.review_id`.
Notebook 16 kiểm tra các ràng buộc này trước khi xuất.

## Truy vết một requirement
`requirements.requirement_id` → `requirement_evidence` (review_id, thứ hạng gần tâm) → `reviews` (văn bản, ngày, phiên bản, nhãn dự đoán,
`evidence_status`) → `topics` (cụm, purity, `generation_step`). `reviews.evidence_status` cho biết vì sao một review actionable không làm
evidence: `ba_manual` (cụm chuyển BA), `out_of_scope_app` (sàng lọc cụm 0/8), `type_mismatch`, `outlier`, `temporal_insufficient`.

## Trạng thái
`requirements.status = candidate`: requirement do hệ thống gợi ý; BA duyệt, sửa hoặc loại trước khi dùng (amendment v2.8). Xếp ưu tiên
chỉ theo `issue_frequency` và `temporal_status`; không có điểm Severity/Impact.
