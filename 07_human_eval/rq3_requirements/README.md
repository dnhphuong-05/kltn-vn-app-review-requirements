# Chấm Candidate Requirement (GĐ15, RQ3, amendment v2.9) — hướng dẫn cho người chấm

Toàn bộ **16 Candidate Requirement** do GĐ14 sinh (`05_experiments/rq3_requirements/requirements.csv`) và **93 review được trích làm
evidence**. Output không được sửa trước hoặc trong khi chấm.

| File | Nội dung |
|---|---|
| `rq3_requirement_eval_blank.csv` | Phiếu mức requirement: 16 dòng (notebook 15 tạo, không ghi đè) |
| `rq3_evidence_eval_blank.csv` | Phiếu mức review trích dẫn: 93 dòng |
| `rq3_requirement_eval_A.csv`, `rq3_evidence_eval_A.csv` | Bản đã chấm của người chấm A (B tương tự) |

Hai phiếu có văn bản review nên không commit.

## Phiếu 1 — mức review trích dẫn (`rq3_evidence_eval_*.csv`) — nên chấm TRƯỚC
Mỗi dòng: một requirement (`problem_vi`, `user_story`) và một review được trích (`text`).

**`supports`** — review này có nêu đúng vấn đề hoặc đề xuất mà requirement mô tả không?
- **1**: có; review có thể kèm ý khác, miễn vấn đề của requirement là một ý rõ trong review.
- **0**: review nói vấn đề khác, chỉ cùng chủ đề chung (ví dụ cùng "giao hàng" nhưng khác vấn đề), hoặc quá chung chung.

## Phiếu 2 — mức requirement (`rq3_requirement_eval_*.csv`)
Mỗi dòng có `user_story`, `acceptance_criteria` (mỗi AC một dòng) và `cited_reviews` (văn bản các review được trích, nhãn E#). Chấm so
với các review được trích, không so với hiểu biết riêng về ứng dụng.

| Cột | Thang | Câu hỏi | Mốc |
|---|---|---|---|
| `fidelity` | 1–5 | Requirement có nói đúng vấn đề mà các review được trích nêu không? | 5 đúng và đủ ý chính; 4 đúng, thiếu hoặc thừa nhẹ; 3 đúng một phần; 2 lệch phần lớn; 1 không liên quan |
| `actionability` | 1–5 | Đội phát triển app có thể bắt tay làm từ requirement này không? | 5 làm được ngay; 4 cần làm rõ nhỏ; 3 cần làm rõ đáng kể; 2 quá mơ hồ; 1 không phải việc của đội app |
| `ac_testability` | 1–5 | Các AC có kiểm tra được đạt/không đạt trên app không? | 5 mọi AC kiểm tra được; 4 phần lớn; 3 một nửa; 2 ít; 1 không AC nào |
| `hallucination` | 0/1 | User Story hoặc AC có thông tin không xuất hiện trong review được trích không (bối cảnh, màn hình, bước, nguyên nhân, lợi ích)? | **1** = có ít nhất một chi tiết bịa; 0 = không |
| `type_correct` | 0/1 | Loại (`type`: BUG/FEATURE/UX) có đúng với nội dung requirement không? | 1 đúng; 0 sai |
| `hallucination_note` | chữ | Ghi chi tiết bị bịa (ví dụ "AC2: 'requests support' không có trong review") | |
| `comment` | chữ | Ghi chú khác | |

Mốc chung cho `hallucination`: diễn đạt lại hoặc khái quát hóa đúng ý review KHÔNG tính là bịa; thêm tình huống, màn hình, nút, bước hay
lợi ích cụ thể mà không review nào nhắc thì tính là bịa.

## Độc lập
Mỗi người một bản sao riêng; không xem bản của người kia, không thảo luận trong lúc chấm. Chỉ một người chấm thì kết quả gọi là expert
validation exploratory.

## Đưa vào Google Sheets
1. **File → Import → Upload** từng phiếu blank → *Replace spreadsheet*, **bỏ chọn** "Convert text to numbers, dates, and formulas".
2. **File → Make a copy** cho mỗi người chấm (`_A`, `_B`).
3. **Data → Data validation → Dropdown**: `supports`, `hallucination`, `type_correct` ∈ {0, 1}; `fidelity`, `actionability`,
   `ac_testability` ∈ {1, 2, 3, 4, 5}; chọn *Reject input*. Bật *Wrap text* cho các cột văn bản.
4. Không xóa, không sắp xếp lại, không sửa `item_no`, `requirement_id`, `review_id`, `text` và các cột requirement.
5. Chấm xong: **File → Download → CSV**, đặt tên `rq3_requirement_eval_A.csv`, `rq3_evidence_eval_A.csv` (B tương tự), chép vào thư
   mục này rồi chạy lại notebook 15 (mục "Kết quả chấm tay").
