# PROJECT SPECIFICATION: AI WASTE CLASSIFICATION AGENT

## 1. Tổng quan Dự án (Project Overview)

* **Tên dự án:** Waste Classification & Recycling Advisory System (Tác tử AI phân loại rác & Gợi ý tái chế).
* **Mục tiêu:** Xây dựng hệ thống tác tử thông minh tiếp nhận ảnh rác thải, phân loại chính xác nhóm rác, đồng thời liên kết cơ sở tri thức luật/quy chuẩn để hướng dẫn bỏ đúng thùng rác và cách thức xử lý/tái chế tại nguồn.
* **Đối tượng thực hiện:** Nhóm 5 người.

---

## 2. Nền tảng Lý thuyết môn học (HUST AI Framework Alignment)

| Thành phần PEAS | Mô tả chi tiết | Tiêu chí & Thông số cụ thể |
| :--- | :--- | :--- |
| **P**<br>*(Performance Measure)*<br>**Độ đo hiệu năng** | • Tối đa hóa độ chính xác nhận diện tổng thể.<br>• Phạt nặng (*Zero-tolerance*) lỗi bỏ sót rác nguy hại sang nhóm rác sinh hoạt thông thường.<br>• Tối ưu thời gian đáp ứng thời gian thực.<br>• Đảm bảo độ tin cậy trước khi đưa khuyến nghị.<br>• Tuân thủ 100% quy chuẩn phân loại rác tại nguồn. | • $\text{Accuracy} \ge 90\%$, $\text{Precision}/\text{Recall}$ cân bằng.<br>• Hàm phạt cực đại với lỗi False Negative ở nhóm Pin/Ắc quy/Chất độc hại.<br>• Thời gian phản hồi tác tử $\le 1.5$ giây.<br>• Ngưỡng tin cậy $\ge 0.6$ (dưới ngưỡng sẽ kích hoạt trạng thái yêu cầu chụp lại). |
| **E**<br>*(Environment)*<br>**Môi trường** | • Vật thể rác thải sinh hoạt, công cộng đa dạng hình dạng, dơ bẩn, móp méo, che khuất một phần.<br>• Điều kiện chụp: ánh sáng ngẫu nhiên, nền bàn, sàn hoặc khay nhận rác.<br>• Hệ thống 4 nhóm thùng rác quy chuẩn tại nguồn. | • **Nhóm rác:** Nhựa, Giấy, Kim loại, Thủy tinh, Hữu cơ, Nguy hại.<br>• **Thùng rác đích:**<br>  - 🟡 *Tái chế* (Chai nhựa, lon, giấy sạch)<br>  - 🟢 *Hữu cơ* (Thức ăn thừa, lá cây)<br>  - ⚪ *Rác còn lại / Vô cơ* (Túi bẩn, hộp xốp)<br>  - 🔴 *Nguy hại* (Pin, ắc quy, bóng đèn) |
| **A**<br>*(Actuators)*<br>**Cơ cấu tác động** | • Giao diện tương tác người dùng (Streamlit Dashboard).<br>• Cơ chế phát tín hiệu thị giác phân loại.<br>• Bộ sinh khuyến nghị quy trình xử lý tại nguồn.<br>• Cơ chế cảnh báo nguy hiểm khẩn cấp. | • Hiển thị Bounding Box quanh vật thể kèm nhãn & điểm Confidence (%).<br>• Thẻ chỉ thị đổi màu theo màu thùng rác quy chuẩn.<br>• Văn bản hướng dẫn chi tiết các bước sơ chế (vd: súc rửa, ép xẹp, bọc kín).<br>• Hộp cảnh báo nổi bật (*Alert popup*) đối với rác độc hại/cháy nổ. |
| **S**<br>*(Sensors)*<br>**Cảm biến** | • Thiết bị thu nhận tín hiệu hình ảnh từ môi trường thực tế.<br>• Cổng nhận dữ liệu số từ người dùng.<br>• Bộ tiền xử lý ma trận điểm ảnh chuẩn hóa. | • Webcam / Camera tích hợp (luồng video frames trực tiếp).<br>• Module upload file ảnh kỹ thuật số (`.jpg`, `.jpeg`, `.png`).<br>• Bộ tiền xử lý chuẩn hóa kích thước $640 \times 640$, không gian màu RGB và dải giá trị pixel $[0, 1]$. |

#### Bảng Đặc tính Môi trường (Russell & Norvig Framework):
| Đặc tính môi trường | Đánh giá | Giải thích học thuật trong bài toán |
| :--- | :--- | :--- |
| **Observability** *(Khả năng quan sát)* | **Partially Observable** | Camera 2D chỉ thấy bề mặt bên ngoài ở một góc chụp; không quan sát được mặt sau, trọng lượng hoặc bên trong bao bì kín. |
| **Agents** *(Số lượng tác tử)* | **Single-Agent** | Tác tử hoạt động độc lập để phân loại ảnh rác (hỗ trợ người dùng - Human-in-the-loop). |
| **Determinism** *(Tính tiền định)* | **Stochastic** | Điều kiện môi trường có yếu tố ngẫu nhiên (góc chụp, độ phản quang, biến dạng cơ học của rác). |
| **Episodic / Sequential** *(Giai đoạn)* | **Episodic** | Mỗi lần đưa ảnh rác vào là một phiên xử lý độc lập, không phụ thuộc vào kết quả của các lần chụp trước đó. |
| **Dynamism** *(Tính biến đổi)* | **Static** | Ảnh tĩnh tải lên không thay đổi trong suốt thời gian tác tử phân tích và suy luận luật. *(Semi-dynamic nếu xử lý video stream liên tục)*. |
| **Continuity** *(Tính liên tục)* | **Continuous / Discrete** | Không gian cảm biến là **Liên tục** (điểm ảnh, toạ độ bounding box); không gian quyết định là **Rời rạc** (tập nhãn rác và 4 loại thùng). |
| **Known / Unknown** *(Mức độ hiểu biết)* | **Known** | Quy chuẩn phân loại rác và tập luật xử lý đã được định nghĩa tường minh trong cơ sở tri thức. |

* **Mô hình Học máy $(T, P, E)$:**
  * **T (Task):** Phân loại ảnh rác vào các lớp định danh (Nhựa, Giấy, Kim loại, Thủy tinh, Hữu cơ, Nguy hại).
  * **P (Performance):** Accuracy, Precision, Recall, Confusion Matrix trên tập kiểm thử.
  * **E (Experience):** Tập dữ liệu ảnh có gán nhãn được chia tỷ lệ chuẩn **80% Train – 20% Test** (kiểm soát hiện tượng Overfitting).

* **Cơ sở tri thức (Knowledge Base & Rules):**
  * Sử dụng cấu trúc luật sản xuất $\text{IF... THEN...}$ và biểu diễn thuộc tính $\text{Prop}(\text{Đối tượng}, \text{Thuộc tính}, \text{Giá trị})$ để ánh xạ từ kết quả Computer Vision sang hành động phân loại/tái chế.

---

## 3. Kiến trúc Hệ thống & Luồng Dữ liệu (System Architecture)

```text
[Input Image (User/Webcam)] 
         │
         ▼
[Computer Vision: YOLOv8 / Pre-trained CNN]
         │ (Tên nhãn rác: 'plastic_bottle', 'battery',...)
         ▼
[Knowledge Base / Rule Engine: Python dict/rules]
         │ (Thùng rác, Độ độc hại, Quy trình xử lý)
         ▼
[UI Layer: Streamlit Dashboard (Local Offline Web App)]
```

* **Công nghệ sử dụng:**
  * Ngôn ngữ: `Python 3.10+`
  * Thị giác máy tính: `Ultralytics (YOLOv8)` hoặc `TensorFlow/Keras`
  * Giao diện: `Streamlit`
  * Cơ sở tri thức: Module Python rule-based (dictionary/functions) hoặc `Owlready2`
  * Nguồn dữ liệu: Open Datasets (**TACO Dataset**, **Trash-ICRA19**, hoặc **Garbage Classification** trên Kaggle).

---

## 4. Phân công Vai trò (Team Roles)

* **TV1 (PM & System Architect):** Thiết lập kiến trúc PEAS, $(T, P, E)$, cấu trúc thư mục và liên kết các module.
* **TV2 (Data Engineer):** Tải Open Dataset, làm sạch ảnh, chia folder `train/` (80%) và `val/test/` (20%).
* **TV3 (ML Engineer):** Viết script train/fine-tune mô hình và xuất file trọng số (`best.pt` hoặc `.h5`).
* **TV4 (Knowledge Engineer):** Xây dựng module tra cứu luật `waste_rules.py` (nhập nhãn $\rightarrow$ trả về thùng rác, hướng dẫn).
* **TV5 (UI/UX & Testing):** Viết giao diện `app.py` bằng Streamlit, tích hợp mô hình và thực hiện test kịch bản demo.

---

## 5. Cấu trúc Thư mục Dự án Dự kiến (Target Directory Structure)

```text
waste_ai_project/
│
├── data/
│   ├── train/               # 80% ảnh gán nhãn
│   └── test/                # 20% ảnh kiểm thử độc lập
│
├── models/
│   └── best.pt              # Trọng số mô hình sau khi train
│
├── knowledge/
│   └── waste_rules.py       # Cơ sở tri thức và tập luật gợi ý tái chế
│
├── train.py                 # Script huấn luyện / đánh giá mô hình
├── app.py                   # Ứng dụng giao diện Web Streamlit
├── requirements.txt         # Danh sách thư viện (ultralytics, streamlit, opencv,...)
└── PROJECT_CONTEXT.md       # File bối cảnh dự án này
```
