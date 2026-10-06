# Annotation Guideline — v1

> **Trạng thái: DRAFT (quy tắc đã chốt 06/10/2026)** — nội dung guideline FREEZE sau calibration batch 2
> đạt gate κ ≥ 0.6 và TRƯỚC khi gán pilot/Hold-out (ĐC §7.2.1 bước 5).
> Sau freeze, mọi thay đổi tăng version (v1.1, v2…), ghi `edge_cases_log.md` + `protocol_amendments.md`,
> và KHÔNG áp dụng ngầm cho phần đã gán xong.
> Nguồn: đề cương V5 §7.2. Ký hiệu "ĐC §x" = mục trong đề cương.

## Nhãn (gán MỘT lần, 4 lớp)
| Nhãn | Định nghĩa vận hành (ĐC §7.2) |
|---|---|
| **BUG** | Chức năng/hành vi được kỳ vọng nhưng bị lỗi: crash, đứng, không hoàn thành thao tác, sai kết quả, không tải, không đăng nhập… |
| **FEATURE** | Đề xuất chức năng mới HOẶC thay đổi/cải tiến cụ thể đối với chức năng hiện có. |
| **UX** | Chức năng về cơ bản vẫn hoạt động nhưng cách trình bày/tương tác gây khó hiểu, khó thao tác hoặc bất tiện cụ thể. |
| **OTHER** | Khen/chê chung chung; không actionable; vấn đề logistics/service không chỉ ra vấn đề của app; spam/không liên quan. |

Hai tầng được SUY RA, annotator không gán riêng:
- Tier 1: ACTIONABLE = BUG ∪ FEATURE ∪ UX ; NOT_ACTIONABLE = OTHER.
- Tier 2: BUG / FEATURE / UX trên phần ACTIONABLE.

## Cây quyết định (đi theo thứ tự, ưu tiên hơn đọc 4 định nghĩa riêng lẻ — sơ đồ: `decision_tree.md`)
1. Review có nói về chính app/phần mềm không? Không → OTHER.
2. Có chỉ ra điều cụ thể mà dev xử lý được không? Không → OTHER.
3. Có biểu hiện malfunction / lỗi? → BUG.
4. Có yêu cầu thêm/đổi chức năng chưa có? → FEATURE.
5. Chức năng có rồi nhưng khó dùng / rối / gây friction cụ thể? → UX.
6. Còn lại → OTHER.

## Multi-intent
- KHÔNG áp thứ tự ưu tiên máy móc BUG > FEATURE > UX.
- Xác định "main actionable intent" theo mệnh đề chính và yêu cầu rõ nhất.
- Hai intent độc lập ngang nhau, không xác định được intent chính → vẫn gán một nhãn tốt nhất VÀ
  đặt `ambiguous_flag = 1`. Dòng ambiguous giữa BUG/FEATURE/UX vẫn là ACTIONABLE hợp lệ ở Tier 1;
  không dùng train Tier 2 vòng đầu; giữ cho error analysis, có thể adjudicate sau.
- Phân vân giữa một nhãn actionable và OTHER (Q11): ACTIONABLE CHỈ KHI review chỉ ra một việc cụ thể
  mà dev xử lý được; không thì OTHER.
- Hold-out (Q11): mọi dòng — kể cả `ambiguous_flag = 1` — có MỘT gold_label sau adjudication và được
  tính vào metric chính của cả hai tầng.

## Edge cases — quy tắc (chốt 06/10/2026, Q16)
| Tình huống | Nhãn | Nguồn |
|---|---|---|
| Logistics / giao trễ / shipper / người bán — không do app | OTHER | ĐC §7.2; Q16 |
| Tracking đơn hiển thị sai / không cập nhật TRONG app | BUG | Q16 |
| Voucher bị lỗi, không áp dụng được | BUG | Q16 |
| Xin thêm loại voucher mới | FEATURE | Q16 |
| Than phiền giá / phí / chính sách (không phải lỗi app) | OTHER | Q16 |
| OTP / đăng nhập bị lỗi (không nhận OTP, không đăng nhập được) | BUG | Q16 |
| Xin cách đăng nhập mới | FEATURE | Q16 |
| Ranh giới FEATURE vs UX: chức năng CHƯA CÓ | FEATURE | Q16 |
| Ranh giới FEATURE vs UX: chức năng ĐÃ CÓ nhưng khó dùng / rối | UX | Q16 |
| Chê chung ("app khó dùng", "rối quá") không chỉ ra chức năng/điểm cụ thể | OTHER (bước 2) | Cây quyết định; Q11 |

## Quy trình gán (ĐC §7.2.1; thứ tự theo `QUY_TRINH_CHUAN.md` v1.1)
1. Calibration batch 1 (~50–75): 2 annotator độc lập, không thảo luận khi gán → so bất đồng → sửa guideline.
2. Calibration batch 2 (~50–75) với guideline đã sửa → đo percent agreement + Cohen's κ + confusion theo lớp.
   **Gate (Q10): κ ≥ 0.6** mới chốt guideline. κ là cổng nhưng không là tiêu chí duy nhất — soi số ca
   bất đồng tuyệt đối giữa FEATURE/UX. Đây là inter-annotator agreement chuẩn với annotator thứ hai thật
   (hai người gán độc lập calibration + Hold-out).
3. Freeze guideline → pilot prevalence (thuộc Hold-out) → gán Hold-out (double-label toàn bộ, agreement đo
   TRƯỚC adjudication).
4. Train/Validation: annotator chính + ≥15–20% double-label audit. Calibration sau adjudicate đưa vào Train.
- Annotator KHÔNG thấy prediction của model.
- **Ẩn rating và date** khi gán (Q17); hai trường này vẫn giữ trong metadata/provenance, chỉ không hiển thị
  trong file gán nhãn.
- Mọi ca khó mới → ghi `edge_cases_log.md` kèm guideline_version.

## Cột file gán nhãn / provenance (ĐC §7.2.1 bước 8, Phụ lục B)
`review_id, app, content, raw_text_hash, rating, date, app_version, sampling_source (random|keyword_enriched),
split (train|validation|internal_holdout|external_test), label_a, label_b, agreement_flag, gold_label,
ambiguous_flag, guideline_version`

## Ví dụ (mục tiêu ≥10–20 / nhãn, gồm edge cases: voucher/phí, logistics, tracking, OTP, multi-intent, UX chung)
Các dòng dưới là ví dụ minh họa có sẵn trong đề cương (không phải review thật). Bổ sung ví dụ thật
từ batch calibration sau khi adjudicate: `<ĐIỀN: sau calibration>`.
| text | label | ghi chú |
|---|---|---|
| Cập nhật xong bấm thanh toán thì app đứng luôn. | BUG | Malfunction (ĐC §7.2) |
| Cập nhật bản mới xong thanh toán ShopeePay cứ bị đứng, bấm lại mấy lần vẫn không được. | BUG | Ví dụ end-to-end (ĐC §16) |
| Nên cho phép lọc cùng lúc nhiều mức giá. | FEATURE | Đổi/cải tiến chức năng có sẵn (ĐC §7.2) |
| Nút xóa và nút mua sát nhau nên dễ bấm nhầm. | UX | Chạy được nhưng friction cụ thể (ĐC §7.2) |
| Giao hàng chậm quá. | OTHER | Logistics, không chỉ ra vấn đề app (ĐC §7.2) |
| App tuyệt vời. | OTHER | Khen chung (ĐC §7.2) |
