# Annotation Guideline — v1.2

> **Trạng thái: FROZEN v1.2 — 12/10/2026.** Đóng băng sau calibration batch 2 đạt cổng (κ = 0,785 ≥ 0.6) và
> TRƯỚC khi gán pilot/Hold-out (ĐC §7.2.1 bước 5).
> Sau freeze, mọi thay đổi tăng version (v1.1, v2…), ghi `edge_cases_log.md` + `protocol_amendments.md`,
> và KHÔNG áp dụng ngầm cho phần đã gán xong.
> Nguồn: đề cương V5 §7.2. Ký hiệu "ĐC §x" = mục trong đề cương.
>
> **Lịch sử phiên bản**
> - v1 (06/10/2026): định nghĩa 4 nhãn, cây quyết định, edge cases Q16.
> - **v1.1 (08/10/2026)**: hiệu chỉnh sau calibration batch 1 (75 review, κ = 0,671, 9 ca bất đồng) — thêm quy tắc
>   phân xử R1–R5 và quy tắc `ambiguous`; định nghĩa 4 nhãn (ĐC §7.2) giữ nguyên. Chi tiết: `edge_cases_log.md`.
>   Batch 2 được gán bằng v1.1.
> - **v1.2 (08/10/2026)**: batch 2 đạt cổng (κ = 0,785) nhưng 7/8 ca lệch dồn quanh BUG (một người gán "malfunction"
>   cho thao tác app không hỗ trợ hoặc kết quả do chính sách/logistics) → thêm R6 (ranh giới BUG), bổ sung R3
>   ("không cho tắt / bắt buộc" → FEATURE); 12/10/2026 làm rõ R6 cho hệ thống chống gian lận ("hoạt động bất thường"
>   → OTHER). Chốt theo phương án (a): không gán thêm batch; R6 được kiểm lại qua agreement đo trên Hold-out.
>   Ví dụ thật lấy từ calibration batch 1 + batch 2 + bộ bổ sung FEATURE–UX.

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
- **Quy tắc `ambiguous` (v1.1):** đánh `ambiguous = 1` khi review có **≥ 2 intent actionable thuộc LỚP KHÁC NHAU**
  (vd vừa BUG vừa FEATURE) mà không ý nào trội hẳn. VẪN phải chọn một label chính: intent **xuất hiện trước**, hoặc
  intent được **nhấn mạnh hơn**. Nếu một ý trội rõ → `ambiguous = 0`. Không bao giờ để trống label, kể cả khi
  `ambiguous = 1`. Nhiều ý nhưng cùng một lớp (vd hai lỗi khác nhau) → `ambiguous = 0`.
- Dòng ambiguous giữa BUG/FEATURE/UX vẫn là ACTIONABLE hợp lệ ở Tier 1; không dùng train Tier 2 vòng đầu;
  giữ cho error analysis, có thể adjudicate sau.
- Phân vân giữa một nhãn actionable và OTHER (Q11): ACTIONABLE CHỈ KHI review chỉ ra một việc cụ thể
  mà dev xử lý được; không thì OTHER.
- Hold-out (Q11): mọi dòng — kể cả `ambiguous_flag = 1` — có MỘT gold_label sau adjudication và được
  tính vào metric chính của cả hai tầng.

## Edge cases — quy tắc (Q16 chốt 06/10/2026; R1–R5 bổ sung v1.1, 08/10/2026)
Các quy tắc R là quy tắc **phân xử** cho ca khó; chúng không định nghĩa lại 4 nhãn của ĐC §7.2.

| Tình huống | Nhãn | Nguồn |
|---|---|---|
| Logistics / giao trễ / shipper / người bán — không do app, không kèm yêu cầu chức năng | OTHER | ĐC §7.2; Q16; R4 |
| Tracking đơn hiển thị sai / không cập nhật TRONG app | BUG | Q16 |
| Voucher sai KỸ THUẬT: không áp được, hiển thị hoặc trừ sai số tiền | BUG | Q16; R1 |
| Voucher về GIÁ TRỊ / ĐIỀU KIỆN / THỜI HẠN (hết hạn, giá trị bị giảm, điều kiện khó) | OTHER | R1 |
| Xin THÊM / NHIỀU giảm giá, ưu đãi (xin ưu đãi, không phải chức năng app) | OTHER | R1 |
| Xin một LOẠI voucher / cơ chế ưu đãi mới trong app | FEATURE | Q16 |
| Than phiền giá / phí / chính sách (không phải lỗi app) | OTHER | Q16 |
| OTP / đăng nhập bị lỗi (không nhận OTP, không đăng nhập được) | BUG | Q16 |
| Xin cách đăng nhập mới | FEATURE | Q16 |
| Than phiền một THÀNH PHẦN UI cụ thể gây phiền (quảng cáo, popup, thông báo); nêu đúng thành phần là đủ cụ thể, vd "quảng cáo nhiều quá" | UX | R2 |
| Ranh giới FEATURE vs UX: khả năng CHƯA CÓ, hoặc đề xuất thay đổi/cải tiến cụ thể trên chức năng đã có (vd lọc nhiều mức giá) | FEATURE | ĐC §7.2; Q16 |
| Hành vi ĐÃ CÓ gây phiền — CHỈ than phiền cách nó vận hành, không hàm ý thiếu một tùy chọn cụ thể | UX | Q16; R3 |
| Hành vi ĐÃ CÓ gây phiền — KÈM đề xuất thay đổi / thêm tùy chọn cụ thể (vd "cho tùy chọn tắt auto-play", "tắt tự vào Live dùm") | FEATURE | R3 |
| "Không cho tắt / bắt buộc / không cho chọn X" — luôn FEATURE, bất kể giọng than phiền (vd "bắt dùng Face ID, không cho tắt") | FEATURE | R3 bổ sung (v1.2) |
| **Nguyên tắc ranh giới BUG:** "app LÀM SAI điều lẽ ra làm được" → BUG; "app KHÔNG LÀM ĐƯỢC điều tôi muốn" → FEATURE | — | R6 |
| Chức năng CÓ tồn tại nhưng chạy sai / không chạy: crash, lag, nút không ăn, thanh toán báo lỗi, hiển thị sai | BUG | R6 |
| App thiết kế KHÔNG hỗ trợ một thao tác ("không cho hủy đơn", "không hỗ trợ đổi sđt", "bắt buộc Face ID", "không cho tắt Live") | FEATURE | R6 |
| Kết quả do chính sách / hệ thống chống gian lận / logistics / người bán (khóa tài khoản, đơn tự hủy hoặc không nhận xu vì "hoạt động bất thường", đơn không về, shop chưa nhận hàng trả, không cho trả hàng) — dù người dùng gọi là "lỗi" | OTHER | R6 (làm rõ 12/10/2026) |
| …trừ khi app HIỂN THỊ SAI trạng thái (tracking báo đã giao mà chưa) | BUG | Q16; R6 |
| YÊU CẦU một chức năng cụ thể trong app, dù lý do là logistics/giá (vd "trả lại lựa chọn đơn vị vận chuyển vì GHN lâu") | Theo yêu cầu (FEATURE/UX) | R4 |
| Nhắc "lỗi"/"không được" nhưng nguyên nhân rõ ràng ngoài app (pin, mạng, thiết bị) | OTHER | R5 |
| Nhắc "lỗi"/"không được" nhưng mơ hồ, không xác định được nguyên nhân hay yêu cầu | OTHER (mặc định bảo thủ, bước 2) | R5 |
| Chê chung ("app khó dùng", "rối quá") không chỉ ra chức năng/thành phần cụ thể | OTHER (bước 2) | Cây quyết định; Q11 |

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

## Ví dụ (≥ 10 / nhãn, review thật từ calibration, nhãn = đồng thuận A/B hoặc đã chốt trong `edge_cases_log.md`)
Nguồn: calibration batch 1, batch 2 và bộ bổ sung FEATURE–UX. Văn bản giữ nguyên cách viết của người dùng (đã ẩn danh hóa),
review dài được cắt bớt ("…"). Không dùng làm ví dụ: các review đang chờ chốt nhãn trong
`07_human_eval/calibration/disagreements_to_finalize.csv`. Sáu ví dụ minh họa của đề cương (ĐC §7.2, §16) được giữ ở cuối.

### BUG (10)
| review_id | text | ghi chú |
|---|---|---|
| 3966135b | tìm kiếm trên shoppee bị cái gì mà ko ra v cứ ra mấy cái nhảm gì đâu ko | tìm kiếm trả kết quả sai |
| b4882cc5 | Không đăng nhập đc | không đăng nhập được (Q16) |
| c1feab98 | k tải shoppe dc | không tải được app |
| ea9bac97 | lag | lag |
| 986554df | muốn xoá sản phẩm trong giỏ hàng cứ bị lỗi vài xoá ko đc chỉ cần cạp nhật vấn đề về lỗi giỏ hàng nữa là quá OK luôn | xóa sản phẩm trong giỏ bị lỗi |
| 8cac906e | đặt có cái balo thôi mà bắt xác nhận hoài, lag, tóm lại cập nhật xong như củ ca.c | lag, bắt xác nhận lặp lại |
| fcfd1b2e | app siêu tệ, đang nói chuyện với cskh tự dưng bị văng ra rồi không tìm thấy mục trò chuyện đó nữa, vào lại phần cskh thì màn hình trắng xóa | văng khỏi chat, màn hình trắng |
| 70b45598 | có mã mà nó ko cho đặt | có mã nhưng không đặt được (R1: voucher không áp được) |
| ff90f91f | Rất tệ mk đã áp mã thời hạn sử sụng đến 25/10 ms hết mà h mk áp thì ko đc liên tục bảo là đã hết lượt sử dụng ko những vậy lại còn hay lắc đơ | mã còn hạn nhưng báo hết lượt + đơ |
| 166cf0de | tủ nhận hàng và điểm nhận hàng là gì vậy nhập địa chỉ rồi nhưng mà không đặt hàng được cứ bảo không tìm thấy dù có đổi tài khoản hay điện thoại vẫn kh … | không đặt được, báo không tìm thấy địa chỉ (ambiguous = 1) |

### FEATURE (16)
| review_id | text | ghi chú |
|---|---|---|
| b55fae39 | app nên để số điện thoại và tên shipper lên tiến độ giao hàng để người dùng còn liên hệ chứ đơn hàng đến chạm giao hàng mà 1 tuần lễ ko thấy shipper g … | hiển thị sđt/tên shipper |
| b509d88b | App nên thêm tên, số liên lạc của shipper để khách hàng có thể liên hệ khi cần | thêm liên lạc shipper |
| b2467038 | mong shopee thêm chế độ giao diện tối, đặc biệt cho các ôg sử dụng trong bóng rối | giao diện tối |
| 332b3389 | app có thể bổ sung darkmode hình nền màu đen được không, màu trắng chói mắt quá | darkmode |
| 2c1ee7ca | Shoppe nên thêm chức năng lọc các nhận xét mới nhất và nên bỏ cái vừa vào shoppe là vô phần video/livestream (Rất bất tiện!) | lọc nhận xét mới nhất |
| fb74d347 | Đề nghị Shopee nên có nút báo cáo video bởi vì có rất nhiều video cắt ghép mang nội dung độc hại, nhảm nhí; quảng cáo thì quá trớn, quá tâng bốc sản p … | nút báo cáo video |
| a455751b | Nói thật thì mik rất là thích shopee gia đình mik ko khá giả đâu mà mik cx chx đủ lớn để đi làm rất là bt ơn shopee đã tạo công ăn vc làm cho mik và n … | bộ phận xử lý report, hình phạt cấm chat |
| ebdf6afe | chăm sóc khách hàng của bạn trả lời chậm , cái gì cũng ko biết , không đáp ứng được nhu cầu của khách , không hỗ trỡ cũng như không giải quyết được gì … | hiển thị lý do vi phạm trong Shopee Live |
| 093b366a | mua hàng nhưng bắt nhận hàng ở tủ hoặc nhận chủ động khó tìm chỗ nhận và còn xa nx khu vực tôi ko có , khó chịu khi đặt hàng rồi lại bị tự động hủy ở … | cho giao tận nơi thay vì bắt nhận ở tủ (ambiguous = 1) |
| dad71cb2 | Tắt chức năng vào app là nhảy vô video với live dùm cái. Mất thời gian, phiền, chậm, lag đặc biệt máy yếu. | R3: đề xuất tắt tự vào Live/video |
| fb0c4bbc | mỗi lần vào là vào thẳng Live & video , tôi không có nhu cầu thì có tắt được không ? với nút back của điện thoại trong rất nhiều trường hợp không có t … | R3 + ambiguous = 1: nhiều ý khác lớp, lấy ý đầu |
| 93d24e85 | trả lại cái lựa chọn đơn vị vận chuyển đi tại đơn vị vận chuyển GHN giao rất lâu nếu so với các đơn vị khác trễ tận 2 3 ngày có lúc trễ tận 7 ngày cho … | R4: yêu cầu chức năng, lý do logistics |
| 92402354 | t lỡ tay nhấn đặt shopee food và không được chủ động hủy đơn đặt lại. 1 sao không nói nhiều | R6: không cho chủ động hủy đơn |
| e63ed42d | không hỗ trợ đổi sđt | R6: không hỗ trợ đổi sđt |
| 7447906e | không cài face id mà thanh toán bắt dùng face id. cũng không cho tắt😡 | R6 + R3: bắt buộc Face ID, không cho tắt |
| 7825c8f5 | bắt ép coi live và video khi mở app, lướt mua hàng tới đâu cũng hiện video gây khó chịu. không cho tắt và không có cách liên hệ hỗ trợ. cực kỳ phiền | R3: bắt xem Live, không cho tắt |

### UX (12)
| review_id | text | ghi chú |
|---|---|---|
| 5d4a9ae9 | QC nhiều quá | R2: quảng cáo |
| 42acd7bc | đi đâu cũng thấy quảng cáo | R2: quảng cáo |
| 79bd9683 | quảng cáo nhiều vcl | R2: quảng cáo |
| 8e9ea2d4 | bớt quản cáo lại đang kiểm tra 15p tự nhiên hiện rồi xong luôn | R2: quảng cáo bật lên giữa chừng |
| 4316ea7d | quảng cáo lừa người td dấu x để mở app | R2: quảng cáo dấu X giả |
| 3851f20a | thông báo thì nhiều, đang ở bên gg thì bị chuyển sang shoppe liên tục | R2: thông báo, tự chuyển sang app |
| 5dd43325 | mẹ nó tối ngày dẫn vô live rồi giảm âm lượng | R3: chỉ than phiền tự vào Live, giảm âm lượng |
| 45484d4e | Chào đội ngũ hỗ trợ, mk muốn phản ánh vấn đề về quảng cáo trong game. Gần đây mình gặp quảng cáo Shopee khá thường xuyên. Sau khi xem xong, quảng cáo … | R2: quảng cáo tự chuyển sang app |
| 6f0b537c | đang dùng app khác tự nhiên quảng cáo shoppee một phát là y như rằng nó thoát app đang dùng luôn làm phải đăng nhập lại để vào. Rất bực mình 😡😡😡.Nhưng … | R2: quảng cáo làm thoát app đang dùng |
| c7626350 | qc bên app khác cực kì khó chịu, bắt ấn vào chuyển sang app mới chịu cho tắt qc, nói chung shoppe qc ncc | R2: phải bấm vào quảng cáo mới tắt được |
| 8ebf8a04 | quảng cáo shope cứ nhấp bỏ qua qc lại vô trang tải shope. phiền! | R2: bỏ qua quảng cáo vẫn vào trang tải |
| 7786d2b5 | hiển thị voucher lum la, cả voucher sinh nhật của ngta cũng tính là giá sản phẩm làm khách hàng tưởng nhầm, chưa kể giá áp dụng ở trang chủ là của sho … | hiển thị voucher/giá gây hiểu nhầm (ambiguous = 1, ý đầu là UX) |

### OTHER (20)
| review_id | text | ghi chú |
|---|---|---|
| bf17da8f | tốt | khen chung |
| a8d6b536 | very good | khen chung |
| 28882499 | OK | chung chung |
| 7af14850 | ứng dụng siêu tệ | chê chung |
| 3f839403 | app rác | chê chung |
| 81cef14d | cập nhật shopee càng nhìn càng rối khó dùng. | chê chung, không chỉ ra chức năng/thành phần |
| 03f1c809 | giao lâu | logistics |
| 7515a405 | shipper tự ý hủy đơn hàng của khách, | logistics: shipper |
| 87de4734 | ko cho hoàn hàng | R6: chính sách trả hàng |
| 4958811e | chính sách trả hàng cực kỳ tệ. người bán gửi sai hàng quay video mở hàng nói giả mạo ngụy tạo bằng chứng không cho trả hàng. quá tệ | R6: chính sách trả hàng / người bán |
| 05a679a1 | ko làm j mà bị vi phạm chính sách shoppe là sao? | R6: chính sách |
| e07c1b0f | xu điểm danh ngày càng ít , acc mới đc giảm nhiều hơn , cho dù là hạng kim cương cx kh đc mã giảm nhiều | R1: giá trị ưu đãi/xu |
| 954843cf | quá tuyệt vời mong có nhiều voucher | R1: xin thêm voucher |
| 29248bd9 | mong thêm nhiều giảm giá | R1: xin thêm giảm giá |
| 42a8462c | mk vừa lập acc lưu đc viu chờ 80k xg e nó ghi là 3/8 hết hạn r mk để đấy chx đc 3 ngày viu chờ ko còn nó lại cho mk viu chờ 60k thôi bực ghê | R1: voucher hết hạn/giảm giá trị |
| aa4bf2af | tôi thực sự hỏi chấm shopee tôi đã làm bị khóa tài khoản trong khi tui đã đăng ký shopee vip rồi mà vẫn bị khóa là sau .tui có chứng cứ đoành hoàng | R6: khóa tài khoản |
| 99aa591f | Ứng dụng SV mé nó đặt hàng méo về | R6: đơn không về |
| 8e0c8551 | đang mua thì bị hết pin, bị thu dth, mong nhà phát hành sửa lỗi | R5: nguyên nhân ngoài app |
| 9e89d23d | bỏ thêm tủ đồ đi nhé, không đặt hàng dc | R5: mơ hồ |
| fde373b1 | Làm Affiliate Marketing (Tiếp thị liên kết): Bạn gắn link các sản phẩm từ Zalo Mini App, Shopee, Lazada,... vào video hoặc phần bình luận. Khi khách h … | không liên quan (quảng cáo affiliate) |

### Ví dụ minh họa của đề cương (không phải review thật)
| text | label | ghi chú |
|---|---|---|
| Cập nhật xong bấm thanh toán thì app đứng luôn. | BUG | Malfunction (ĐC §7.2) |
| Cập nhật bản mới xong thanh toán ShopeePay cứ bị đứng, bấm lại mấy lần vẫn không được. | BUG | Ví dụ end-to-end (ĐC §16) |
| Nên cho phép lọc cùng lúc nhiều mức giá. | FEATURE | Đổi/cải tiến chức năng có sẵn (ĐC §7.2) |
| Nút xóa và nút mua sát nhau nên dễ bấm nhầm. | UX | Chạy được nhưng friction cụ thể (ĐC §7.2) |
| Giao hàng chậm quá. | OTHER | Logistics, không chỉ ra vấn đề app (ĐC §7.2) |
| App tuyệt vời. | OTHER | Khen chung (ĐC §7.2) |
