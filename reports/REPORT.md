# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ            | Nghĩa ngắn                                                    |
| ------------- | ------------------------------------------------------------- |
| Prediction    | Kết quả model vừa đoán ra; chưa phải nhãn đúng                |
| Ground truth  | Nhãn do người gán theo quy tắc                                |
| Taxonomy      | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint    | File model `.pt` đã huấn luyện sẵn                            |
| Guideline     | Quy tắc gán nhãn do người viết                                |
| Annotator     | Người gán nhãn                                                |
| Reviewer / QC | Người kiểm tra nhãn                                           |
| Escalation    | Ca khó → hỏi Lab Coach, không tự bịa                          |
| Model score   | Số “tin” của model; không phải điểm chất lượng nhãn người     |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1.  Chép record hạng 1: `class_id`=468 ; `class_name`=`cab` ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=`ImageNet-1K`

2.  Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
    Report này nói về **cả bức ảnh** không phải về mội **vật thể riêng biệt** bởi vì:
    "task": "image_classification": Xác định nhãn phân loại cho toàn bộ ảnh.

        "class_name": "cab": Mô hình dự đoán ảnh thuộc lớp cab.

        "score": 0.510915: Độ tin cậy của dự đoán khoảng 51,09%.

        Không có thông tin tọa độ bounding box như x, y, width, height để xác định vị trí của một vật thể riêng lẻ.

3.  Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
    Model không tự đặt tên lớp khi dự đoán. Model trả về xác suất theo các class ID đã học; hệ thống sử dụng class list đi kèm checkpoint hoặc taxonomy để ánh xạ ID thành tên lớp.

4.  Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
    class_id: Mã định danh lớp, giúp máy tính xử lý và ánh xạ nhãn.
    class_name: Tên lớp, giúp con người dễ đọc và hiểu kết quả.
    taxonomy_name: Xác định hệ thống phân loại, tránh nhầm lẫn giữa các bộ nhãn khác nhau.
    Cần ghi cả 3 trường trên để: - Tránh nhầm lẫn khi nhiều taxonomy có cùng tên lớp nhưng ID hoặc ý nghĩa khác nhau. - Giúp đối chiếu kết quả giữa các model và dataset. - Dễ dàng chuyển đổi, thống kê và đánh giá dữ liệu. - Đảm bảo khả năng truy xuất nguồn gốc và tính nhất quán của dữ liệu.

5.  Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
    Guideline cần quy định rõ tiêu chí chọn lớp đại diện cho toàn bộ ảnh: - Chọn chủ thể chính hoặc đối tượng nổi bật nhất. - Ưu tiên đối tượng chiếm diện tích lớn hoặc thu hút sự chú ý nhất. - Nếu có nhiều đối tượng, cần quy định thứ tự ưu tiên để chọn một lớp. - Nếu các đối tượng có mức độ nổi bật tương đương, cần có quy tắc xử lý trường hợp hòa.
    Ví dụ: Ảnh giao thông có nhiều xe buýt, ô tô và taxi. Người gán nhãn cần xác định phương tiện đại diện theo guideline, không tự chọn tùy ý.
    Kết luận: Guideline phải thống nhất tiêu chí chọn một lớp đại diện cho toàn bộ ảnh, tránh mỗi người gán nhãn một kiểu.

6.  Vì sao không được lấy `score` của model làm ground truth?
    Không được lấy score của model làm ground truth vì: - score chỉ thể hiện mức độ tin cậy của dự đoán, không đảm bảo dự đoán đúng. - Model có thể dự đoán sai dù score cao. - Score có thể không phản ánh chính xác xác suất thực tế. - Ground truth phải dựa trên dữ liệu được con người gán nhãn và kiểm chứng theo guideline.
    Kết luận: score là kết quả dự đoán của model, còn ground truth là nhãn chuẩn được xác minh, không được đánh đồng hai giá trị này.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=`person` ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
   Góc trên-trái: (385.33, 69.24).
   Góc dưới-phải: (498.92, 348.92).
   Chiều rộng: 113.58 pixel.
   Chiều cao: 279.68 pixel.

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
   Ngưỡng 0.60: Có 6 prediction đạt score từ 0.60 trở lên.
   Ngưỡng 0.20: JSON hiện tại chỉ chứa kết quả từ score_threshold = 0.35, nên không có các prediction nằm trong khoảng 0.20 đến dưới 0.35.

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
   Reviewer cần: - Kiểm tra các bounding box có score thấp vì dễ chứa dự đoán sai (false positive). - Loại bỏ các box không tương ứng với vật thể thực tế. - Kiểm tra và xử lý các box trùng lặp. - Xác nhận class và vị trí bounding box có chính xác không.
   Kết luận: Hạ threshold giúp phát hiện nhiều vật thể hơn, giảm bỏ sót (false negative), nhưng làm tăng số dự đoán sai và khối lượng công việc review.

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
   Quy tắc: Bounding box phải bao sát toàn bộ phần vật thể nhìn thấy được, không bao gồm quá nhiều background. - Mỗi vật thể được gán một bounding box riêng. - Box phải bao trọn vật thể, không bị thiếu phần nhìn thấy được. - Không tự ý mở rộng box để bao gồm phần bị che khuất. - Box phải xác định đúng class theo taxonomy.
   Kết luận: Bounding box cần chính xác, nhất quán và chỉ bao quanh phần vật thể thực tế nhìn thấy.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
   Ví dụ: Người ở mép trái ảnh bị cắt khỏi khung hình, chỉ nhìn thấy một phần cơ thể.
   Cách xử lý: Đưa lên hỏi Lab Coach (escalation).
   Lý do: Cần xác nhận cách xử lý vật thể bị cắt mép, đặc biệt khi chỉ nhìn thấy một phần nhỏ. Không nên tự suy đoán quy tắc khi guideline chưa quy định rõ.
   Kết luận: Những trường hợp chưa có quy định thống nhất cần escalation để đảm bảo tất cả annotator gán nhãn nhất quán.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=`kitchen-001` ; `class_name`=`person` ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`=[
   [
   446.0,
   70.0
   ],
   [
   445.0,
   71.0
   ],
   [
   444.0,
   71.0
   ],
   [
   443.0,
   72.0
   ],
   [
   442.0,
   72.0
   ],
   ]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
   Box (Bounding Box): Xác định vị trí và kích thước vật thể bằng hình chữ nhật bao quanh vật thể (tọa độ x, y, width, height).
   Polygon: Xác định đường viền vật thể bằng nhiều điểm tọa độ, mô tả chính xác hình dạng và diện tích thực tế của vật thể.

3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
   instance_id dùng để định danh duy nhất một đối tượng cụ thể trong ảnh/sample. - Không phải Class ID: Không biểu thị loại đối tượng. - Không phải Tracking ID: Không theo dõi đối tượng qua nhiều frame video.
   Ví dụ: 3 người cùng có class_id = 0 nhưng có instance_id khác nhau.

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
   Quy tắc: Vẽ mask bám sát đường viền thực tế của vật thể, bao phủ toàn bộ phần nhìn thấy được, không bao gồm nền và không tự ý vẽ phần bị che khuất.

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
   Trên ảnh kitchen: dining table 0.58 có mask lớn chồng lên các bowl (0.70, 0.74) và oven 0.49, còn person 0.43 ở mép trái bị che và dính vào bát, nên biên không rõ. Với quy tắc chung (vẽ theo phần nhìn thấy) thì dùng guideline; còn ca không chắc đó là vật nào hoặc biên không phân biệt được thì escalation lên Lab Coach.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ                | Ground truth trông như thế nào? (đơn vị nhãn)            | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON                                                                                        | Annotator làm gì?                                                              | Reviewer xem gì?                                                               |
| --------------------- | -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| Phân loại ảnh         | Một lớp cho cả ảnh (class_id, class_name, taxonomy_name) | Ảnh traffic có nhiều xe buýt và ô tô; top-1 cab chỉ 0.510915, minibus 0.164284, police_van 0.085848 nên khó chọn một nhãn                 | Chọn một lớp từ taxonomy theo tiêu chí guideline (vật lớn nhất, gần giữa nhất) | Nhãn có nằm trong taxonomy và đúng tiêu chí không; không dùng score làm căn cứ |
| Phát hiện vật thể     | Một hộp bbox_xyxy (pixel) và một lớp cho mỗi vật         | Ảnh kitchen: person 0.61 bị cắt ở mép ảnh; ngưỡng 0.20 cho 17 hộp nhưng 0.60 chỉ 6 hộp, nhiều hộp yếu (spoon, bottle)                     | Vẽ hộp ôm sát phần nhìn thấy của từng vật                                      | Hộp có sát không, có thiếu hoặc thừa vật, có sai lớp không                     |
| Instance segmentation | Một polygon polygon_xy và một instance_id cho mỗi vật    | Ảnh kitchen: dining table 0.58 chồng lên bowl 0.70 và 0.74; person 0.43 bị che; traffic-001 (bus) có 120 điểm polygon nên vẽ tay tốn công | Vẽ biên theo vật, tách riêng từng instance                                     | Biên mask có khớp vật không, các instance chồng nhau có bị lẫn không           |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
   Chỉ sử dụng 3 ảnh COCO công khai, không dùng dữ liệu cá nhân hoặc thông tin định danh trong báo cáo.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
   Dừng xử lý và báo ngay cho Lab Coach.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [*] `REPORT.md` (file này đã điền)
- [*] `day1_lab_outputs/classification_predictions.json`
- [*] `day1_lab_outputs/detection_predictions.json`
- [*] `day1_lab_outputs/segmentation_predictions.json`
- [*] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [*] `day1_lab_outputs/visuals/classification_top5.png`
- [*] `day1_lab_outputs/visuals/detection_predictions.png`
- [*] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [] Ô validation cuối notebook báo `PASS`
- [] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [] Đã nộp **link repository** trên LMS
