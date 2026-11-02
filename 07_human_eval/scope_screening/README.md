# QA sàng lọc phạm vi cụm 0 và 8 (GĐ14, amendment v2.6 bước 2)

`scope_screening_qa.csv`: 40 review — mỗi cụm (0 giao hàng; 8 sản phẩm/người bán) 10 review mỗi nhãn LLM, rút ngẫu nhiên (seed 42),
xáo thứ tự trong cụm. Nhãn LLM từ prompt `04_prompts/requirement_generation/scope_screening_system_v2.txt` (hash trong registry).
File có văn bản review nên không commit.

Các cột: `item_no`, `topic_id`, `review_id`, `text`, `llm_label`, `llm_reason`, `qa_label` (ô cần điền), `qa_note`.

## Câu hỏi cho mỗi dòng
**Review có nêu một lỗi hoặc một đề xuất cụ thể về CHỨC NĂNG CỦA ỨNG DỤNG không?**

- `qa_label` = **app_actionable**: chính ứng dụng hiển thị sai, thiếu chức năng, hoặc không cho thao tác; hoặc đề xuất cụ thể một
  chức năng của ứng dụng (kể cả khi review có thêm ý ngoài phạm vi).
- `qa_label` = **out_of_scope**: chỉ là vận hành giao hàng, shipper, CSKH; yêu cầu Shopee kiểm soát hay chế tài người bán, đổi chính
  sách; chất lượng hay giá sản phẩm; khen hoặc than phiền chung.
- `qa_note`: ghi ngắn khi phân vân (ví dụ: trạng thái đơn hiển thị sai do shipper cập nhật).

Có thể che cột `llm_label`, `llm_reason` khi chấm để không bị ảnh hưởng. Không sửa `item_no`, `review_id`, `text`.
