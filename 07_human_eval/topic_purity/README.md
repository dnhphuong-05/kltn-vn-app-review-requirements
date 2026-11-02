# Manual topic purity (RQ2) — hướng dẫn cho người chấm

Mẫu: 17 cụm × 10 review = **170 review**, rút ngẫu nhiên (seed 42) trong mỗi cụm của kết quả gom cụm đã khóa
(`03_configs/frozen/rq2_topics_config.yaml`, amendment v2.4). Không lấy review outlier.

| File | Nội dung |
|---|---|
| `topic_purity_blank.csv` | Phiếu gốc (notebook 12 tạo, không ghi đè) |
| `topic_purity_A.csv`, `topic_purity_B.csv` | Hai bản đã chấm, tải về từ Google Sheets dạng CSV |

Các cột: `item_no` (số thứ tự), `topic_id`, `topic_name` (tên cụm), `keywords` (6 từ khóa đặc trưng của cụm), `review_id`,
`text`, `relevant` (ô chấm), `note` (ghi chú). Phiếu không có nhãn BUG/FEATURE/UX hay loại chủ đạo của cụm (cố ý ẩn để người chấm đánh giá cụm, không đánh giá
bộ phân loại).

## Câu hỏi cho mỗi dòng
**Review này có nói về vấn đề mà tên cụm và từ khóa của cụm mô tả không?**

- `relevant` = **1**: review nêu đúng vấn đề đó (có thể kèm ý khác, miễn vấn đề của cụm là một ý rõ ràng trong review).
- `relevant` = **0**: review nói về vấn đề khác, hoặc chỉ than phiền chung chung không nêu rõ vấn đề (ví dụ "app tệ",
  "lỗi hoài" trong một cụm có tên cụ thể).
- `note`: ghi ngắn khi phân vân, hoặc khi thấy tên cụm sai/quá hẹp so với các review của cụm.

Chấm từng review theo tên và từ khóa của cụm đó; không so với các cụm khác. Phiếu xếp theo cụm, mỗi cụm 10 dòng liền nhau.

## Độc lập
Mỗi người một bản sao riêng; không xem bản của người kia, không thảo luận trong lúc chấm. Người nghiên cứu không mở bản của
người này cho người kia.

## Đưa vào Google Sheets
1. Google Sheets → **File → Import → Upload** → chọn `topic_purity_blank.csv` → *Replace spreadsheet*, separator *Detect
   automatically*, **bỏ chọn** "Convert text to numbers, dates, and formulas".
2. **File → Make a copy** hai lần, đặt tên `topic_purity_A` và `topic_purity_B`; chia sẻ mỗi bản cho đúng một người chấm.
3. Trong mỗi bản: chọn cột `relevant` (từ dòng 2) → **Data → Data validation → Add rule → Dropdown** với hai giá trị `1`, `0`,
   chọn *Reject input*.
4. Có thể ẩn cột `review_id` khi chấm nhưng **không xóa, không sắp xếp lại, không sửa** `item_no`, `topic_id`, `review_id`, `text`.
5. Chấm xong: **File → Download → CSV**, đặt tên `topic_purity_A.csv` / `topic_purity_B.csv` và chép vào thư mục này, rồi chạy
   lại hai cell cuối của notebook 12 (mục "Kết quả manual purity").

## Cách tính
Purity của một cụm = tỷ lệ dòng `relevant = 1`, tính riêng cho từng người rồi lấy trung bình hai người. Độ đồng thuận: % trùng
và Cohen κ trên 170 phán đoán 1/0. Cụm có purity < 0,60 được gắn cờ cho BA. Kết quả chấm dùng để đánh giá; cụm không được chỉnh
lại theo kết quả chấm.
