# Phiếu mã hóa lỗi giữ nhầm OTHER của LLM (amendment v3.7)

## Phiếu gồm những gì

`coding_sheet.csv` có 161 review mà nhãn gold là OTHER nhưng LLM (cascade hoặc phẳng, prompt v1) lại gọi là ACTIONABLE. Review lấy từ cả
Hold-out Shopee và Lazada, đã trộn ngẫu nhiên. Phiếu không ghi review thuộc ứng dụng nào và không hiện gợi ý từ khóa, để mã hóa không bị
dẫn dắt. Thông tin đó nằm trong `coding_key.csv`; **không mở file này trước khi mã hóa xong**.

`coding_sheet.csv` có văn bản review nên không commit, chỉ giữ trên máy; `coding_key.csv` không có văn bản review nên vẫn commit.

## Cần điền

| cột | cách điền |
|---|---|
| `nhóm chính` | đúng một nhóm trong codebook dưới đây, ghi y nguyên tên nhóm |
| `ranh giới (có/không)` | "có" nếu đọc lại thấy review có thể được xem là ACTIONABLE theo guideline (ca ranh giới), ngược lại "không" |
| `ghi chú` | tùy chọn; bắt buộc khi chọn "khác" |

Không sửa cột `mã dòng` và `text`, không đổi thứ tự dòng. Nhãn gold giữ nguyên, cột `ranh giới` chỉ để ghi nhận.

## Codebook

| nhóm chính | khi nào chọn |
|---|---|
| giao hàng | vận chuyển, shipper, giao trễ, giao chậm, giao sai, chưa nhận được hàng |
| người bán, sản phẩm | chất lượng hàng, hàng giả, shop gửi sai hoặc thiếu, đóng gói |
| giá, phí, voucher | giá cả, phí ship, giá trị, điều kiện hoặc hạn của voucher, xin thêm ưu đãi (không phải voucher lỗi kỹ thuật) |
| chính sách, tài khoản, đơn hàng | khóa tài khoản, "hoạt động bất thường", đơn tự hủy, hoàn tiền, trả hàng, đổi trả: kết quả do chính sách hay quy trình, dù người dùng gọi là "lỗi" |
| chăm sóc khách hàng | tổng đài, nhân viên, bộ phận hỗ trợ, phản hồi chậm |
| than phiền app chung chung | chê app hoặc nói "lỗi", "không được" mà không chỉ ra chức năng hay thành phần cụ thể, hoặc nguyên nhân rõ ràng ngoài app (mạng, thiết bị) |
| khen chung chung | khen hoặc cảm ơn, không kèm vấn đề |
| khác | không thuộc nhóm nào ở trên; ghi rõ trong `ghi chú` |

Review có nhiều ý thì chọn ý chính, tức ý xuất hiện trước hoặc được nhấn mạnh hơn, giống guideline gán nhãn.

## Lưu file

Thời gian dự kiến khoảng 45–60 phút. Nếu mở bằng Excel, lưu lại dưới dạng **CSV UTF-8** để giữ dấu tiếng Việt. Sau khi điền xong mọi
dòng, chạy lại notebook `19_lazada_error_analysis.ipynb` để có phân bố nhóm theo ứng dụng và độ khớp với nhóm gợi ý từ khóa.
