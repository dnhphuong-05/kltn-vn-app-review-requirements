# EDA — Giai đoạn 2 (mô tả thuần, chưa có nhãn)

Báo cáo được sinh từ `src/preprocessing/eda.py`. Không có ước lượng tỷ lệ actionable/BUG/FEATURE/UX ở giai đoạn này.

## 1. Đối soát số lượng

| Bước | Shopee | Lazada |
|---|---|---|
| Raw | 57,200 | 3,600 |
| Trùng review_id (loại) | 0 | 0 |
| Ngoài cửa sổ — trước 2025-10-06 (loại) | 831 | 94 |
| Ngoài cửa sổ — sau 2026-10-06 23:59 (loại) | 0 | 0 |
| Sau cắt cửa sổ | 56,369 | 3,506 |
| Text rỗng (loại) | 0 | 0 |
| Không có chữ cái — emoji/ký hiệu/số (loại) | 319 | 12 |
| Sau lọc (interim) | 56,050 | 3,494 |

## 2. Tóm tắt mô tả

| Chỉ số | Shopee | Lazada |
|---|---|---|
| Số review | 56,050 | 3,494 |
| Khoảng ngày (giờ VN) | 2025-10-06 → 2026-10-06 07:18 | 2025-10-06 → 2026-10-06 05:45 |
| Phân bố rating | 1★: 39.9% / 2★: 3.4% / 3★: 3.2% / 4★: 5.0% / 5★: 48.5% | 1★: 60.7% / 2★: 3.5% / 3★: 3.8% / 4★: 3.7% / 5★: 28.3% |
| Số từ Q25/Q50/Q75/Q90 | 2.0 / 6.0 / 16.0 / 32.0 | 4.0 / 11.0 / 25.0 / 47.0 |
| Số ký tự (trung vị) | 27.0 | 47.0 |
| Review ≤ 2 từ | 28.6% | 19.3% |
| Thiếu app_version | 26.3% | 34.0% |
| Top version | 3.65.22 (2,729), 3.61.28 (2,416), 3.81.32 (2,254), 3.77.25 (2,107), 3.66.26 (1,828) | 7.88.0 (221), 7.89.1 (116), 7.93.1 (114), 7.95.0 (111), 7.86.5 (109) |
| Có phản hồi của sàn | 60.3% | 100.0% |
| Ngôn ngữ (rule-based) | vi: 90.9% / und: 8.8% / non_vi: 0.3% | vi: 92.3% / und: 7.3% / non_vi: 0.4% |
| near_duplicate | 30.1% | 15.5% |
| Emoji/ký hiệu-only đã loại (trên số sau cắt cửa sổ) | 0.57% | 0.34% |

## 3. Nhận xét

**Shopee.** Sau khi làm sạch còn 56.050 review trong 12 tháng. Điều tôi thấy rõ nhất là đánh giá rất phân cực: 1★ và 5★ chiếm gần 90%, các mức 2–4★ chỉ khoảng 11–12%. Mỗi tháng có khoảng 3.500–6.300 review, nhiều nhất vào 01/2026. Phần lớn review rất ngắn: trung vị chỉ 6 từ, gần 29% có không quá 2 từ, và khoảng 30% review trùng y hệt nội dung với review khác, chủ yếu là những câu như "ok", "tốt", "rất tốt". Khoảng 60% review có phản hồi của sàn; 26,3% không có thông tin phiên bản app.

**Lazada.** Lazada có 3.494 review, ít hơn Shopee rất nhiều nhưng vẫn đủ cho External Test. Review ở đây tiêu cực hơn hẳn: 1★ khoảng 61%, 5★ khoảng 28%. Mỗi tháng có khoảng 190–520 review, nhiều nhất vào 12/2025. Người dùng Lazada viết dài hơn (trung vị 11 từ, khoảng 19% có không quá 2 từ) và ít trùng nội dung hơn (khoảng 15%). Sàn trả lời mọi review và thường gọi tên người dùng, đó cũng là lý do Lazada có nhiều chỗ phải ẩn danh hóa. 34,0% không có thông tin phiên bản app.

## 4. QA ẩn danh hóa (PII)

| | Shopee | Lazada |
|---|---|---|
| Mẫu reply_content soi | 50 | 50 |
| Mẫu text soi | 30 | 30 |
| Nghi sót trong mẫu (regex phụ) | 0 | 0 |
| Nghi sót toàn corpus (regex phụ) | reply_content:long_digits=3, text:long_digits=8, text:title_after_pronoun=10 | 0 |

**Hành động:** Tôi sửa luật ẩn danh hóa cho phản hồi của sàn hai lần. Lần 1 (từ regex dò): bắt thêm tên sau "Chào bạn" bắt đầu bằng chữ như "Anh", "Nha", "Vui", và phản hồi mở đầu thẳng bằng tên trước ", cảm ơn". Lần 2 (từ việc tự đọc mẫu, 3/50 phản hồi Lazada còn sót): bắt thêm tên đứng đầu trước ", chúng tôi", ", mình" hoặc câu tiếng Anh. Sau hai lần sửa, mẫu đọc lại và regex dò không còn thấy tên sót ở Lazada; các trường hợp regex còn báo ở Shopee tôi đã kiểm tra từng cái và đều là báo nhầm. File thô giữ nguyên và không đưa lên Git; từ đây tôi chỉ dùng dữ liệu trung gian.

## 5. Hạn chế

- Ngày 06/10/2026 chỉ có một phần (Google Play hiển thị review trễ khoảng 1 ngày); không bù dữ liệu. Ảnh hưởng nhẹ tới cửa sổ last_30.
- Ngôn ngữ được gắn bằng luật (dấu tiếng Việt, chữ ngoài hệ Latin, từ điển tiếng Việt không dấu); review rất ngắn không dấu được để `und`. Review không phải tiếng Việt chỉ được đánh dấu, chưa loại.
- near_duplicate = text_clean trùng y hệt; phần lớn là câu ngắn chung chung ("tốt", "ok"); chỉ đánh dấu, không xóa.

## 6. Hình

- `eda_rating_distribution.png`
- `eda_length_words.png`
- `eda_monthly_volume.png`
- `eda_daily_tail.png`
- `eda_app_version_top.png`
