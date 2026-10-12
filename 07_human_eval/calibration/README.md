# Calibration gán nhãn — hướng dẫn cho người gán

Mẫu: 150 review Shopee rút ngẫu nhiên (seed 42) từ corpus interim, chia **2 batch × 75**
(`research_protocol.md`, ĐC §7.2.1). Danh sách review_id ở `01_data/manifests/calibration_manifest_T0_2026-10-06.json`.

| File | Nội dung |
|---|---|
| `calibration_batch1_blank.csv` | Batch 1 — gán trước, dùng để thảo luận và sửa guideline |
| `calibration_batch2_blank.csv` | Batch 2 — gán SAU khi guideline đã sửa; dùng để đo agreement (cổng κ ≥ 0.6) |

Mỗi file có 5 cột: `review_id`, `text`, `label`, `ambiguous`, `note`. Không có rating, ngày, phiên bản app (cố ý ẩn
để tránh thiên lệch theo số sao — Q17). `text` là văn bản gốc đã ẩn danh hóa, giữ nguyên emoji/viết tắt.

## Quy tắc gán
- `label`: một trong **BUG / FEATURE / UX / OTHER** theo `00_protocol/annotation_guideline_v1.md` và `decision_tree.md`.
- `ambiguous`: điền **1** nếu review có hai ý actionable độc lập ngang nhau, không xác định được ý chính; để trống nếu không.
  Vẫn chọn MỘT nhãn tốt nhất ở `label`.
- `note`: ghi ngắn lý do khi phân vân (rất có ích cho buổi họp sửa guideline).
- **Độc lập:** mỗi người một bản sao riêng; không xem bản của người kia, không thảo luận trong lúc gán.
  Không tra rating/ngày của review ở nguồn khác.

## Đưa vào Google Sheets
1. Google Sheets → **File → Import → Upload** → chọn `calibration_batch1_blank.csv` → *Replace spreadsheet*,
   separator *Detect automatically*, **bỏ chọn** "Convert text to numbers, dates, and formulas".
2. Tạo HAI bản sao riêng: **File → Make a copy**, đặt tên `calibration_batch1_A` và `calibration_batch1_B`;
   chia sẻ mỗi bản cho đúng một người gán (A hoặc B). Người nghiên cứu không mở bản của người này cho người kia.
3. Trong mỗi bản: chọn cột `label` (từ dòng 2) → **Data → Data validation → Add rule → Dropdown** với 4 giá trị
   `BUG`, `FEATURE`, `UX`, `OTHER`, chọn *Reject input*. Cột `ambiguous`: dropdown một giá trị `1` (hoặc checkbox).
4. Có thể ẩn bớt cột `review_id` khi gán nhưng **không xóa, không sắp xếp lại, không sửa** `review_id` và `text`.

## Tải về khi gán xong
- **File → Download → Comma-separated values (.csv)** từ từng bản, đổi tên đúng:
  `calibration_batch1_A.csv`, `calibration_batch1_B.csv` (batch 2: `calibration_batch2_A.csv`, `calibration_batch2_B.csv`).
- Đặt vào thư mục này (`07_human_eval/calibration/`) rồi chạy phần đo agreement trong `notebooks/04_annotation_agreement.ipynb`.

## Trình tự
1. A, B gán batch 1 độc lập → tải về → đo agreement batch 1 → họp thảo luận các ca bất đồng → sửa guideline
   (ghi `00_protocol/edge_cases_log.md`, tăng version guideline).
2. A, B gán batch 2 bằng guideline đã sửa → tải về → đo agreement batch 2 = **cổng** (κ ≥ 0.6, kèm soi bất đồng FEATURE/UX).
3. Đạt cổng → chốt guideline (bước riêng, sau khi người nghiên cứu xác nhận). Chưa đạt → họp sửa tiếp và gán batch
   calibration mới (calibration lần 2).

Các review calibration KHÔNG vào Hold-out; sau adjudication được đưa vào Train (Q9).
