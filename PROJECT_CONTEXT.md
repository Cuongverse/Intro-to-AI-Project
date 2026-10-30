# PROJECT SPECIFICATION: CONTEXT-AWARE AI WASTE SORTING & ADVISORY AGENT

## 1. Tổng quan Dự án & Nền tảng Nghiên cứu (Project Overview & Academic Grounding)

* **Tên dự án:** Context-Aware Waste Detection & Ontological Recycling Advisory System (Tác tử AI nhận diện rác thải và Tư vấn tái chế dựa trên Bản thể học Ngữ cảnh).
* **Bối cảnh ứng dụng:** Hệ thống Trợ lý Thị giác tại Trạm phân loại rác thông minh (Smart Sorting Station), kết hợp nhận diện thực thể vật lý qua camera và suy luận ngữ cảnh (Human-in-the-loop) để phân luồng tái chế chính xác.
* **Quy mô thực hiện:** Nhóm 5 sinh viên – Học phần Nhập môn Trí tuệ Nhân tạo (HUST).
* **Cơ sở bài báo khoa học nền tảng (2025):**
  1. **Thị giác máy tính (Perception Layer):** Dipo, M.H. et al. (2025). *"Real-Time Waste Detection and Classification Using YOLOv12-Based Deep Learning Model"*, Digital 2025, 5, 19. Kế thừa kiến trúc **YOLOv12** (tích hợp khối *R-ELAN* và *Area Attention*) cùng bộ dữ liệu thực tế của nhóm tác giả.
  2. **Biểu diễn tri thức & Suy luận (Reasoning Layer):** Milivinti, E. (2025). *"The Interoperability Challenge in Waste Management: An Ontological Approach"*, FOIS 2025 / CEUR-WS Vol-4111. Ứng dụng khung **Basic Formal Ontology (BFO)** để định nghĩa "rác" là một **Vai trò phụ thuộc ngữ cảnh (*Context-dependent Role*)** thay vì thuộc tính vật lý cố định.

---

## 2. Nền tảng Lý thuyết Môn học (HUST AI Framework Alignment)

### 2.1. Mô hình Tác tử PEAS

| Thành phần PEAS | Mô tả chi tiết trong Hệ thống | Tiêu chí & Thông số Kỹ thuật |
| :--- | :--- | :--- |
| **P**<br>*(Performance Measure)*<br>**Độ đo hiệu năng** | • Tối đa hóa độ chính xác định vị và nhận diện thực thể vật liệu.<br>• Đảm bảo tính nhất quán ngữ nghĩa khi suy diễn vai trò rác theo ngữ cảnh.<br>• Phạt nặng (*Zero-tolerance*) lỗi phân luồng nhầm rác nguy hại/nhiễm bẩn vào luồng tái chế sạch.<br>• Tối ưu tốc độ suy luận thời gian thực. | • Thị giác: $\text{mAP@0.5} \ge 0.78$, $\text{F1-Score} \ge 0.75$.<br>• Suy luận: Độ chính xác gán nhãn `Role` đạt $100\%$ theo bảng chân trị BFO.<br>• Thời gian phản hồi tổng thể $\le 0.5$ giây/ảnh trên CPU.<br>• Ngưỡng độ tin cậy (*Confidence threshold*) $\ge 0.50$. |
| **E**<br>*(Environment)*<br>**Môi trường** | • Trạm thu gom rác thực tế tại trường học/tòa nhà: ánh sáng thay đổi, vật thể móp méo, chồng lấp.<br>• Không gian trạng thái kết hợp giữa hình thái vật lý và điều kiện ngữ cảnh (độ sạch, hạ tầng thu gom). | • **6 Thực thể Vật liệu (Material Entities):** `Plastic`, `Metal_Battery`, `Paper`, `Glass`, `Organic`, `Medical`.<br>• **4 Phân luồng đích (Target Streams):**<br>  - 🔵 *Tài nguyên Tái chế* (`Recyclable Resource`)<br>  - 🟢 *Phân hủy Sinh học* (`Compostable / Organic`)<br>  - ⚫ *Rác Còn lại / Xử lý nhiệt* (`Residual Waste`)<br>  - 🔴 *Rác Nguy hại / Y tế* (`Hazardous / Biohazard Waste`) |
| **A**<br>*(Actuators)*<br>**Cơ cấu tác động** | • Giao diện hiển thị trực quan Streamlit.<br>• Bộ giải thích suy luận Bản thể học (*Explainable Ontology Card*).<br>• Hệ thống cảnh báo an toàn và hướng dẫn sơ chế tại nguồn. | • Vẽ Bounding Box + Nhãn thực thể vật lý + Độ tin cậy ($\%$).<br>• Hiển thị chuỗi suy diễn: `Material Entity` $\rightarrow$ `Disposition` $\rightarrow$ `Context` $\rightarrow$ `Assigned Role`.<br>• Xuất chỉ dẫn hành động cụ thể (ví dụ: tháo nắp, tráng sạch hoặc cách ly khẩn cấp). |
| **S**<br>*(Sensors)*<br>**Cảm biến** | • Cảm biến hình ảnh kỹ thuật số.<br>• Giao diện thu nhận tín hiệu ngữ cảnh (*Human-in-the-loop Context Input*) bù đắp cho phần khuất mà camera 2D không thấy được. | • Camera/Webcam hoặc cổng tải file ảnh (`.jpg`, `.png`) chuẩn hóa về kích thước $640 \times 640$.<br>• Nút chọn tình trạng vật lý (Sạch / Bẩn dầu mỡ / Nhiễm hóa chất) và trạng thái hạ tầng tại chỗ. |

### 2.2. Đặc tính Môi trường (Russell & Norvig Framework)
* **Partially Observable (Quan sát được một phần):** Camera 2D chỉ nhìn thấy bề mặt ngoài của vật thể, không thể tự biết bên trong vỏ chai còn tồn dư hóa chất độc hại hay không $\rightarrow$ Đây chính là lý do học thuật bắt buộc phải có **Tầng nhập ngữ cảnh (Context Ingestion)** và **BFO Ontology**.
* **Single-Agent & Human-in-the-Loop:** Tác tử AI đóng vai trò hệ chuyên gia tư vấn tương tác trực tiếp với người phân loại.
* **Stochastic & Episodic:** Ảnh đầu vào có nhiễu ngẫu nhiên về góc chụp và độ sáng; mỗi lượt phân loại một lô rác là một phiên xử lý độc lập.

---

## 3. Thiết kế Chi tiết Hai Tầng Cốt lõi (Core Technical Design)

### 3.1. Tầng Thị giác & Tiền xử lý Dữ liệu (Dựa trên Paper 1 - YOLOv12)
Bộ dữ liệu gốc trên Roboflow (7.288 ảnh) bị phân mảnh thành 18 lớp không đồng nhất và chứa nhãn nhiễu. Nhóm thực hiện chuẩn hóa bằng script `scripts/remap_labels.py` theo bảng ánh xạ sau:

| Mã lớp mới (`class_id`) | Tên Thực thể Chuẩn (`Material Entity`) | Các lớp gốc từ Roboflow (18 classes) được gộp vào | Đặc tính Cố hữu (`BFO Disposition`) |
| :---: | :--- | :--- | :--- |
| **`0`** | **`Plastic`** | `PET Bottle`, `HDPE Plastic`, `Single-Use-Plastic`, `Single-layer Plastic`, `Squeeze Tube`, `Red Bottle` | `MechanicalRecyclability` |
| **`1`** | **`Metal_Battery`** | `Battery` | `HeavyMetalToxicity`, `FireHazard`, `MaterialRecovery` |
| **`2`** | **`Paper`** | `Paper`, `Cardboard Paper`, `Newspaper` | `CelluloseRecyclability`, `Biodegradability` |
| **`3`** | **`Glass`** | `Glass`, `Glass-Bottle` | `InfiniteRecyclability`, `SharpPhysicalHazard` (nếu vỡ) |
| **`4`** | **`Organic`** | `BiologicalWaste` | `Compostability` |
| **`5`** | **`Medical`** | `Mask`, `Glove`, `Medicine`, `syringe` | `BiohazardInfectious`, `ChemicalToxicity` |
| *Loại bỏ* | *Bỏ qua (Ignore)* | `UNIT BOX` (Nhãn lỗi không phải thực thể rác) | — |

### 3.2. Tầng Bản thể học & Luật Suy diễn (Dựa trên Paper 2 - BFO Framework)
Theo *Milivinti (2025)*, một thực thể vật chất chỉ nhận vai trò là **Rác (`Waste Role`)** khi thỏa mãn 3 điều kiện:
1. **(a)** Công năng thiết kế ban đầu (*Function*) không còn được thực hiện.
2. **(b)** Có ý định loại bỏ (*Intention of disposal*).
3. **(c)** Không có phương án khả thi về mặt kinh tế/kỹ thuật để hiện thực hóa các đặc tính hữu ích (*Dispositions*) của vật thể trong ngữ cảnh hiện tại.

**Bảng quy tắc suy diễn Vai trò (`Role Assignment Matrix`) trong `knowledge/waste_rules.py`:**

| Thực thể (`MaterialEntity`) | Đặc tính (`Disposition`) | Ngữ cảnh Vật lý (`ContextState`) | Vai trò được gán (`BFO Role`) | Phân luồng Thùng rác & Hành động |
| :--- | :--- | :--- | :--- | :--- |
| `Plastic` / `Paper` | `Recyclability` | `Clean_Intact` (Sạch) | **`RecyclableResourceRole`** | 🔵 **Thùng Tái chế:** Đổ hết nước, bóp dẹp để tối ưu thể tích. |
| `Plastic` / `Paper` | `Recyclability` | `Food_Soiled` (Dính dầu mỡ/thức ăn) | **`ResidualWasteRole`** | ⚫ **Thùng Rác còn lại:** Dầu mỡ làm hỏng bột giấy/hạt nhựa, không tái chế kinh tế được (Điều kiện c). |
| `Plastic` / `Glass` | `Recyclability` | `Chemical_Contaminated` (Nhiễm hóa chất) | **`HazardousWasteRole`** | 🔴 **Thùng Nguy hại:** Bọc kín, cách ly hoàn toàn khỏi luồng tái chế. |
| `Glass` | `InfiniteRecyclability` | `Broken_Shattered` (Đã vỡ vụn) | **`PhysicalHazardWasteRole`** | 🟠 **Cảnh báo An toàn:** Bọc nhiều lớp giấy dày trước khi bỏ để tránh gây thương tích cho công nhân. |
| `Metal_Battery` | `HeavyMetalToxicity` | *Mọi trạng thái thải bỏ* | **`HazardousEWasteRole`** | 🔴 **Điểm Thu hồi Pin/Điện tử:** Tuyệt đối không bỏ thùng rác gia đình. |
| `Medical` | `BiohazardInfectious` | *Mọi trạng thái thải bỏ* | **`BiohazardMedicalRole`** | 🔴 **Thùng Y tế / Nguy hại sinh học:** Khử khuẩn, bỏ túi kín theo quy chuẩn y tế. |
| `Organic` | `Compostability` | `Uncontaminated` | **`BiomassResourceRole`** | 🟢 **Thùng Hữu cơ:** Dùng ủ phân compost hoặc làm khí sinh học. |

---

## 4. Kế hoạch Thực hiện & Phân bổ Môi trường (Execution Plan)

1. **Môi trường Đám mây (Google Colab):**
   * Dành riêng cho việc tải bộ dữ liệu 7.3k ảnh, chạy `scripts/remap_labels.py` và huấn luyện mô hình YOLOv12 trên GPU NVIDIA T4.
   * Đầu ra thu hoạch: Tệp `best.pt` và bộ ảnh biểu đồ đánh giá (`confusion_matrix.png`, `results.png`).
2. **Môi trường Máy trạm Cá nhân (Local VS Code):**
   * Chỉ lưu mã nguồn nhẹ (`app.py`, `knowledge/waste_rules.py`, `scripts/remap_labels.py`) và tệp trọng số `best.pt` đã huấn luyện xong.
   * Chạy giao diện Web Streamlit phục vụ demo trực tiếp trên lớp.