# Pilot prevalence — hướng dẫn cho người gán

Mẫu: 400 review Shopee rút **ngẫu nhiên thuần** (seed 42) từ corpus interim, sau khi loại 180 review calibration.
Không chọn theo từ khóa, không lọc review trùng hay theo số sao — mục đích là đo đúng tỷ lệ từng nhãn trong dữ liệu.
Danh sách review_id ở `01_data/manifests/pilot_manifest_T0_2026-10-06.json`.

**Pilot là một phần của Hold-out** (Q9) nên thuộc diện khóa theo `01_data/HOLDOUT_LOCK.md`: không dùng nội dung pilot để
sửa guideline, không lấy làm ví dụ few-shot, không đưa vào Train/Validation. Ca lệch được chốt theo guideline **v1.2 (FROZEN)**.

| File | Nội dung |
|---|---|
| `pilot_blank.csv` | 400 review, 5 cột `review_id, text, label, ambiguous, note` (không có số sao, ngày, phiên bản app — Q17) |
| `pilot_A.csv`, `pilot_B.csv` | Bản đã gán của hai người (tải về từ Google Sheets) |
| `pilot_disagreements_to_finalize.csv` | Sinh ra sau khi đo; hai người điền `agreed_label`, `agreed_ambiguous` cho các ca lệch |

## Quy tắc gán
- `label`: **BUG / FEATURE / UX / OTHER** theo `00_protocol/annotation_guideline_v1.md` (v1.2) và `decision_tree.md`.
- `ambiguous`: **1** khi review có ≥ 2 ý actionable thuộc lớp khác nhau mà không ý nào trội hẳn; **0** trong các trường hợp
  còn lại. Vẫn chọn một nhãn chính (ý xuất hiện trước hoặc được nhấn mạnh hơn).
- `note`: ghi ngắn lý do khi phân vân.
- **Độc lập:** mỗi người một bản sao riêng, không xem bản của người kia, không thảo luận trong lúc gán.

## Google Sheets
1. **File → Import → Upload** `pilot_blank.csv` → *Replace spreadsheet*, **bỏ chọn** "Convert text to numbers, dates, and formulas".
2. **File → Make a copy** thành `pilot_A` và `pilot_B`, chia sẻ mỗi bản cho đúng một người.
3. Cột `label`: **Data → Data validation → Dropdown** `BUG, FEATURE, UX, OTHER`, *Reject input*. Cột `ambiguous`: dropdown `0, 1`.
4. Không xóa, sắp xếp lại hay sửa `review_id` và `text`.
5. Gán xong: **File → Download → CSV**, đổi tên thành `pilot_A.csv` / `pilot_B.csv`, đặt vào thư mục này.
