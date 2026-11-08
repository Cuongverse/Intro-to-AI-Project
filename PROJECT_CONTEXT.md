# PROJECT SPECIFICATION: CONTEXT-AWARE AI WASTE SORTING & ADVISORY AGENT

## 1. Tổng quan Dự án & Nền tảng Nghiên cứu (Project Overview & Academic Grounding)

* **Tên dự án:** Context-Aware Waste Detection & Ontological Recycling Advisory System *(Tác tử AI Nhận diện Rác thải & Tư vấn Tái chế dựa trên Bản thể học Ngữ cảnh)*.
* **Bối cảnh ứng dụng:** Hệ thống Trợ lý Thị giác tại Trạm phân loại rác thông minh (*Smart Sorting Station*), kết hợp nhận diện thực thể vật chất qua camera và suy luận ngữ cảnh (*Human-in-the-loop*) để phân luồng tái chế và xử lý rác nguy hại tại nguồn.
* **Quy mô thực hiện:** Nhóm 5 sinh viên – Học phần Nhập môn Trí tuệ Nhân tạo (Đại học Bách khoa Hà Nội - HUST).
* **Cơ sở hai bài báo khoa học nền tảng (2025):**
  1. **Tầng Cảm nhận Thị giác (Perception Layer):** Dipo, M. H. et al. (2025). *"Real-Time Waste Detection and Classification Using YOLOv12-Based Deep Learning Model"*, Digital 2025, 5(2), 19. Kế thừa kiến trúc **YOLOv12n** (tích hợp khối *Area Attention $A^2$*, *R-ELAN* và *C3k2*) cùng bộ dữ liệu thực tế 7.288 ảnh của nhóm tác giả.
  2. **Tầng Biểu diễn Tri thức & Suy luận (Reasoning Layer):** Milivinti, E. (2025). *"The Interoperability Challenge in Waste Management: An Ontological Approach"*, Proceedings of FOIS 2025 Satellite Events, CEUR-WS Vol-4111. Ứng dụng khung **Basic Formal Ontology (BFO 2020)** để định nghĩa "rác" là một **Vai trò phụ thuộc ngữ cảnh (*Context-dependent Role - BFO:0000023*)** thay vì thuộc tính vật lý cố định.

---

## 2. Nền tảng Lý thuyết Môn học (HUST AI Framework Alignment)

### 2.1. Mô hình Tác tử PEAS

| Thành phần PEAS | Mô tả chi tiết trong Hệ thống | Tiêu chí & Kết quả Thực nghiệm Đạt được |
| :--- | :--- | :--- |
| **P**<br>*(Performance Measure)*<br>**Độ đo hiệu năng** | • Tối đa hóa độ chính xác định vị và phân loại thực thể vật chất.<br>• Đảm bảo tính nhất quán ngữ nghĩa khi suy diễn vai trò rác theo ngữ cảnh.<br>• Phạt nặng (*Zero-tolerance*) lỗi bỏ sót rác nguy hại/nhiễm hóa chất vào luồng tái chế hoặc sinh hoạt thông thường.<br>• Đảm bảo độ trễ thời gian thực và kiểm soát ngưỡng tin cậy cảm biến. | • **Mục tiêu Thị giác:** $\text{mAP@0.5} \ge 0.78$, $\text{F1-Score} \ge 0.75$.<br>• **Kết quả Thực tế (Epoch 99/100):** **$\text{mAP@0.5} = 86.12\%$**, **$\text{F1-Score} = 84.31\%$**, $\text{Precision} = 86.01\%$, $\text{Recall} = 82.68\%$.<br>• **Suy luận BFO:** Độ chính xác gán nhãn `BFORole` đạt $100\%$ theo bảng chân trị Logic vị từ.<br>• **Độ trễ (Latency):** $< 500\text{ ms}$/ảnh trên CPU máy cá nhân.<br>• **Cổng an toàn Cảm biến (Sensor Gate):** $\text{Confidence} \ge 0.60$ (dưới $60\%$ tự động kích hoạt `UncertainObservationRole` yêu cầu chụp lại). |
| **E**<br>*(Environment)*<br>**Môi trường** | • Trạm thu gom rác tại nguồn (trường học, tòa nhà): ánh sáng đa dạng, vật thể móp méo, trong suốt hoặc chồng lấp.<br>• Không gian trạng thái kết hợp giữa thực thể vật lý và điều kiện ngữ cảnh (độ nhiễm bẩn bề mặt, hạ tầng thu hồi tại chỗ). | • **6 Thực thể Vật chất (`MaterialCategory` - BFO:0000040):** `Plastic`, `Metal_Battery`, `Paper`, `Glass`, `Organic`, `Medical`.<br>• **4 Luồng Thùng rác Đích Quy chuẩn:**<br>  - 🟡 **Thùng Vàng:** *Tài nguyên Tái chế* (`RecyclableResourceRole`)<br>  - 🟢 **Thùng Xanh Lá:** *Phân hủy Sinh học* (`OrganicCompostRole`)<br>  - ⚪ **Thùng Xám/Trắng:** *Rác Sinh hoạt Còn lại* (`ResidualWasteRole`)<br>  - 🔴 **Thùng Đỏ:** *Rác Nguy hại Đặc biệt* (`HazardousWasteRole`) |
| **A**<br>*(Actuators)*<br>**Cơ cấu tác động** | • Giao diện tương tác Streamlit (`app.py`).<br>• Thẻ Phán quyết Bản thể học (*BFO Reasoning Cards*).<br>• Hệ thống cảnh báo an toàn khẩn cấp và hướng dẫn sơ chế tại nguồn. | • Vẽ Bounding Box đổi màu động theo `BFORole` + Nhãn thực thể + Độ tin cậy ($\%$).<br>• Xuất chuỗi suy diễn Logic vị từ (*First-Order Logic Trace*): `MaterialEntity` $\land$ `Disposition` $\land$ `Context` $\rightarrow$ `BFORole`.<br>• Kiểm tra 3 điều kiện định nghĩa rác $(a, b, c)$ của Milivinti (2025) và xuất quy trình sơ chế từng bước. |
| **S**<br>*(Sensors)*<br>**Cảm biến** | • Cảm biến hình ảnh kỹ thuật số 2D.<br>• Bộ chọn ngữ cảnh (*Context Selectors*) thu nhận trạng thái bề mặt và hạ tầng để bù đắp cho đặc tính *Partially Observable* của camera 2D. | • Camera/Webcam hoặc cổng tải file ảnh (`.jpg`, `.jpeg`, `.png`, `.webp`) chuẩn hóa về ma trận $640 \times 640$.<br>• Cảm biến ngữ cảnh: Tình trạng vật lý (`CLEAN_INTACT`, `FOOD_GREASE_SOILED`, `CHEMICAL_CONTAMINATED`, `BROKEN_SHARP`) và Hạ tầng (`FULL_RECYCLING_FACILITY`, `BASIC_MUNICIPAL_ONLY`). |

### 2.2. Đặc tính Môi trường (Russell & Norvig Framework)

| Đặc tính Môi trường | Phân loại | Giải thích Học thuật trong Bài toán Phân loại Rác BFO |
| :--- | :--- | :--- |
| **Observability** | **Partially Observable** | Camera 2D chỉ thu nhận được đặc trưng quang học bề mặt ở một góc nhìn; không thể tự biết bên trong chai nhựa có tồn dư hóa chất độc hại hay tại địa phương có nhà máy tái chế hay không $\rightarrow$ Bắt buộc phải kết hợp **Context Ingestion** và **BFO Reasoning Engine**. |
| **Agents** | **Single-Agent** | Tác tử AI hoạt động độc lập như một hệ chuyên gia tư vấn hỗ trợ người dùng (*Human-in-the-loop*). |
| **Determinism** | **Stochastic** | Nhiễu ngẫu nhiên từ cường độ sáng, góc nghiêng, độ trong suốt của vật liệu và sự biến dạng cơ học của rác. |
| **Episodic / Sequential** | **Episodic** | Mỗi lượt nhận diện và suy diễn trên một bức ảnh tại trạm thu gom là một phiên quyết định độc lập. |
| **Dynamism** | **Static / Semi-dynamic** | Ảnh đầu vào tĩnh trong suốt quá trình suy luận, nhưng ngữ cảnh môi trường (hạ tầng, trạng thái nhiễm bẩn) có thể thay đổi linh hoạt làm biến đổi trực tiếp Vai trò (`Role`) của thực thể. |
| **Continuity** | **Continuous & Discrete** | Tọa độ không gian $(x_c, y_c, w, h)$ và điểm độ tin cậy là **Liên tục**; tập 6 Thực thể vật chất và 4 Vai trò thùng rác đích là **Rời rạc**. |

---

## 3. Thiết kế Chi tiết Hai Tầng Cốt lõi (Core Technical Design)

### 3.1. Tầng Thị giác & Chuẩn hóa Dữ liệu (Dựa trên Paper 1 - Dipo et al., 2025)

Bộ dữ liệu gốc từ kho lưu trữ của *Dipo et al. (2025)* gồm **7.288 ảnh** (`5.121` train, `1.082` valid, `1.085` test) với tổng cộng **23.298 vật thể (bounding boxes)**. Trong dữ liệu xuất từ Roboflow của tác giả, các nhãn bị phân mảnh thành **16 lớp (`nc = 16`)** chồng chéo cấp độ ngữ nghĩa và trộn lẫn định dạng đa giác (*Polygon segmentation*) với hộp chữ nhật (*Bounding Box*).

Module `scripts/remap_labels.py` thực hiện hai bước tiền xử lý cốt lõi:
1. **Chuẩn hóa Hình học ($O(N)$):** Quy đổi toàn bộ các nhãn đa giác ($2k$ tọa độ đỉnh) về đúng chuẩn hộp giới hạn 5 cột của YOLO (`<new_class_id> <x_center> <y_center> <width> <height>`) bằng cực trị bao đóng $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$.
2. **Quy hoạch Ngữ nghĩa về 6 Thực thể Vật chất (`BFO:0000040`):**

| New ID | Tên Thực thể Chuẩn (`MaterialCategory`) | Các Index & Nhãn gốc từ Roboflow (16 classes) được gộp vào | Số lượng Mẫu (Boxes) | Tỷ lệ (%) | Mã EWC Tham chiếu (Paper 2) |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **`0`** | **`Plastic`** | `[6] HDPE Plastic` (796), `[10] PET Bottle` (793), `[12] Single-Use-Plastic` (4565), `[13] Single-layer Plastic` (9851), `[14] Squeeze Tube` (1655) | **17.660** | $75.80\%$ | `EWC 15 01 02 / 20 01 39` |
| **`1`** | **`Metal_Battery`** | `[0] Battery` (2291) | **2.291** | $9.83\%$ | `EWC 20 01 33* / 16 06 01*` |
| **`2`** | **`Paper`** | `[2] Cardboard Paper` (200), `[9] Newspaper` (110), `[11] Paper` (1661) | **1.971** | $8.46\%$ | `EWC 15 01 01 / 20 01 01` |
| **`3`** | **`Glass`** | `[3] Glass` (158), `[4] Glass-Bottle` (279) | **437** | $1.88\%$ | `EWC 15 01 07 / 20 01 02` |
| **`4`** | **`Organic`** | `[1] BiologicalWaste` (114) | **114** | $0.49\%$ | `EWC 20 01 08` |
| **`5`** | **`Medical`** | `[5] Glove` (136), `[7] Mask` (229), `[8] Medicine` (212), `[15] syringe` (248) | **825** | $3.54\%$ | `EWC 18 01 01 / 18 01 03*` |
| **Tổng** | **6 Lớp BFO** | **16 Lớp gốc** | **23.298** | **$100\%$** | — |

### 3.2. Kết quả Thực nghiệm Huấn luyện YOLOv12n (100 Epochs trên GPU NVIDIA Tesla T4)

Mô hình `YOLOv12n` (`2.55M` tham số, `159` layers fused, `6.3` GFLOPs) được huấn luyện trong 100 epochs với `batch = 16`, `imgsz = 640`, bộ tối ưu hóa `SGD (lr0 = 0.01, weight_decay = 0.0005)`. Việc khử nhiễu nội lớp bằng cách gom 16 nhãn phân mảnh về 6 Thực thể Vật chất BFO đã giúp mô hình hội tụ vượt trội so với kết quả gốc trong bài báo của *Dipo et al. (2025)*:

| Chỉ số Đánh giá (Metrics) | Bài báo gốc *Dipo et al. (2025)* (16 lớp) | Mục tiêu PEAS | Kết quả Thực nghiệm của Nhóm (6 lớp BFO - Best Epoch 99) | Độ tăng trưởng ($\Delta$) |
| :--- | :---: | :---: | :---: | :---: |
| **Mean Precision ($P$)** | $73.00\%$ | Cân bằng $P/R$ | **$86.01\%$ (`0.8601`)** | **+$13.01\%$** vs Bài báo |
| **Mean Recall ($R$)** | $77.00\%$ | Cân bằng $P/R$ | **$82.68\%$ (`0.8268`)** | **+$5.68\%$** vs Bài báo |
| **F1-Score** | $75.00\%$ | $\ge 75.00\%$ | **$84.31\%$ (`0.8431`)** | **+$9.31\%$** vs PEAS |
| **mAP@0.5** | $78.00\%$ | $\ge 78.00\%$ | **$86.12\%$ (`0.8612`)** | **+$8.12\%$** vs PEAS |

* **Độ chính xác nhận diện theo từng lớp trên Ma trận nhầm lẫn chuẩn hóa (`confusion_matrix_normalized.png`):** `Organic` ($100\%$), `Plastic` ($99\%$), `Paper` ($97\%$), `Metal_Battery` ($96\%$), `Medical` ($91\%$), `Glass` ($58\%$).

### 3.3. Tầng Bản thể học & Luật Suy diễn Vị từ (Dựa trên Paper 2 - Milivinti, 2025)

Theo *Milivinti (2025)* (Section 4.2, trang 9), một thực thể vật chất độc lập (`MaterialEntity` - $\text{BFO:0000040}$) chỉ nhận vai trò là **Rác (`Waste Role` - $\text{BFO:0000023}$)** trong một ngữ cảnh không-thời gian khi thỏa mãn đồng thời 3 điều kiện:
1. **Điều kiện $(a)$:** Công năng thiết kế ban đầu (`BFOFunction` - $\text{BFO:0000034}$) không còn được thực hiện.
2. **Điều kiện $(b)$:** Tồn tại ý định thải bỏ từ người sở hữu (*Intention of disposal*).
3. **Điều kiện $(c)$:** Không tồn tại phương án khả thi về mặt kinh tế hoặc kỹ thuật để hiện thực hóa các đặc tính cố hữu (`Disposition` - $\text{BFO:0000016}$) của thực thể trong ngữ cảnh hiện tại.

**Ma trận Suy diễn Vai trò (`BFOReasoningEngine.evaluate()`) trong `knowledge/waste_rules.py`:**

| Luật | Thực thể (`MaterialCategory`) | Đặc tính Cố hữu (`Disposition`) | Ngữ cảnh Quan sát (`ObservationContext`) | Vai trò được gán (`BFORole`) | Phân luồng Thùng rác Đích |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **Luật 0** | *Mọi thực thể* | *Mọi đặc tính* | $\text{Confidence} < 0.60$ | **`UncertainObservationRole`** | 🔵 **Cảnh báo Độ tin cậy thấp:** Yêu cầu đặt ra vùng đủ sáng và chụp lại. |
| **Luật 1** | `Metal_Battery` / `Medical` / *Bất kỳ* | `ToxicityHazard` / `InfectiousBiohazard` | Hoặc `CHEMICAL_CONTAMINATED` | **`HazardousWasteRole`** | 🔴 **Thùng Đỏ (Nguy hại):** Cô lập khẩn cấp (*Zero-tolerance*). |
| **Luật 2** | `Glass` | `MechanicalRecyclability` | `BROKEN_SHARP` (Vỡ sắc nhọn) | **`ResidualWasteRole`** | ⚪ **Thùng Xám/Trắng (Kèm Cảnh báo Đỏ):** Bọc kín nhiều lớp giấy dày chống đứt tay trước khi bỏ. |
| **Luật 3** | `Organic` | `Biodegradability` | Không nhiễm hóa chất độc hại | **`OrganicCompostRole`** | 🟢 **Thùng Xanh Lá (Hữu cơ):** Chắt bỏ nước thừa để ủ phân vi sinh. |
| **Luật 4A** | `Plastic` / `Paper` / `Glass` | `MechanicalRecyclability` | `CLEAN_INTACT` $\land$ `FULL_RECYCLING_FACILITY` | **`RecyclableResourceRole`** | 🟡 **Thùng Vàng (Tái chế):** Bác bỏ Điều kiện $(c)$ $\rightarrow$ Thu hồi làm nguyên liệu thứ cấp. |
| **Luật 4B** | `Plastic` / `Paper` / `Glass` | `MechanicalRecyclability` | `FOOD_GREASE_SOILED` $\lor$ `BASIC_MUNICIPAL_ONLY` | **`ResidualWasteRole`** | ⚪ **Thùng Xám/Trắng (Rác còn lại):** Thỏa mãn Điều kiện $(c)$ do chi phí làm sạch quá cao hoặc thiếu hạ tầng. |

---

## 4. Cấu trúc Thư mục Chuẩn của Dự án (Repository Architecture)

```text
Intro-to-AI-Project/
├── knowledge/
│   ├── waste_bfo.owl               # Tệp đặc tả Bản thể học chuẩn OWL 2 RDF/XML (tương thích Protégé & Owlready2)
│   └── waste_rules.py              # Động cơ suy diễn BFO OOP (MaterialEntity, Disposition, BFORole, Logic Trace)
├── models/
│   ├── best.pt                     # Trọng số YOLOv12n đã huấn luyện (100 epochs, mAP@0.5 = 86.12%, ~5.3 MB)
│   └── metrics/                    # Bộ biểu đồ & dữ liệu kiểm định thực nghiệm trích xuất từ Tesla T4
│       ├── confusion_matrix.png
│       ├── confusion_matrix_normalized.png
│       ├── labels.jpg
│       ├── results.csv
│       └── results.png
├── notebooks/
│   └── train_yolov12_colab.ipynb   # Notebook thực thi toàn bộ pipeline tải dữ liệu, gộp nhãn và train trên Colab
├── scripts/
│   └── remap_labels.py             # Thuật toán quy đổi 16 nhãn gốc -> 6 lớp BFO và chuẩn hóa Polygon -> BBox
├── app.py                          # Giao diện Web Streamlit 3 Tabs (Live Agent, Metrics Benchmark, Ontology Inspector)
├── train.py                        # Script CLI huấn luyện và đánh giá độc lập bằng dòng lệnh
├── requirements.txt                # Danh sách thư viện chạy Local (ultralytics, streamlit, opencv-headless, owlready2)
├── .gitignore                      # Chặn đẩy data/, venv/, __pycache__/ lên kho chứa GitHub
├── CONTRIBUTING.md                 # Quy ước phân nhánh Git và quy trình đóng góp mã nguồn của 5 thành viên
├── PROJECT_CONTEXT.md              # Tài liệu đặc tả kỹ thuật và cơ sở học thuật của dự án (File hiện tại)
└── README.md                       # Tài liệu tổng quan, kết quả thực nghiệm và hướng dẫn vận hành