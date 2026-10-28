# 🤝 Hướng Dẫn Đóng Góp Dự Án (CONTRIBUTING GUIDE)

Chào mừng các thành viên đến với dự án **Waste Classification & Recycling Advisory System**!

Tài liệu này được thiết kế riêng cho nhóm 5 thành viên (kể cả các bạn mới bắt đầu làm quen với Git/Python) nhằm giúp mọi người phối hợp nhịp nhàng, hạn chế tối đa xung đột mã nguồn (*merge conflicts*) và đảm bảo tiến độ hoàn thành đồ án môn **Nhập môn Trí tuệ Nhân tạo - HUST**.

---

## 🎯 1. Phân Công Vai Trò & Phạm Vi Phụ Trách

Mỗi thành viên chịu trách nhiệm chính ở một khu vực (module) cụ thể để tránh chỉnh sửa đè lên code của nhau:

| Thành viên | Vai trò | Khu vực / Tệp phụ trách chính |
| :--- | :--- | :--- |
| **TV1** | **PM & Architect** | Quản lý repo, `PROJECT_CONTEXT.md`, `README.md`, `requirements.txt`, kết nối các luồng |
| **TV2** | **Data Engineer** | Thư mục `data/`, kịch bản cào/chia dữ liệu `scripts/prepare_data.py` |
| **TV3** | **ML Engineer** | `train.py`, đánh giá mô hình, thư mục `models/` (xuất file cấu hình/weights) |
| **TV4** | **Knowledge Engineer** | `knowledge/waste_rules.py`, xây dựng cơ sở tri thức và hệ luật `IF...THEN` |
| **TV5** | **UI/UX & Tester** | `app.py`, cấu hình giao diện Streamlit, test kịch bản người dùng |

---

## 🌿 2. Quy Trình Làm Việc Với Git & GitHub (Git Workflow)

### 2.1. Cấu trúc Nhánh (Branching Model)
* `main`: Nhánh chạy ổn định nhất, chỉ dùng để nộp bài, báo cáo và chạy demo sản phẩm cuối cùng.
* `dev`: Nhánh tích hợp chung của cả nhóm. Mọi tính năng sau khi kiểm tra xong sẽ được gộp vào đây.
* `feature/<tên-tính-năng>`: Nhánh làm việc riêng của từng thành viên cho từng tính năng cụ thể.

**Ví dụ đặt tên nhánh chuẩn:**
* TV2: `feature/dataset-cleaning`
* TV3: `feature/yolov8-training-pipeline`
* TV4: `feature/recycling-rules-engine`
* TV5: `feature/streamlit-dashboard`
* Sửa lỗi: `fix/webcam-stream-bug`

---

### 2.2. Hướng Dẫn Các Bước Làm Việc Hàng Ngày (Step-by-Step Cho Người Mới)

#### Bước 1: Đồng bộ mã nguồn mới nhất từ nhánh `dev`
Trước khi bắt đầu code tính năng mới, hãy luôn cập nhật code mới nhất từ nhóm:
```bash
git checkout dev
git pull origin dev
```

#### Bước 2: Tạo nhánh tính năng mới từ `dev`
```bash
git checkout -b feature/ten-tinh-nang-cua-ban
```

#### Bước 3: Viết code và kiểm tra cục bộ
Hãy đảm bảo code chạy được trên máy của bạn và không làm gãy các module khác.

#### Bước 4: Lưu thay đổi (Commit)
Đặt thông điệp commit rõ ràng, mang tính mô tả:
```bash
git add .
git commit -m "feat(rules): them quy tac xu ly pin va rac thai nguy hai"
```

#### Bước 5: Đẩy nhánh lên GitHub
```bash
git push -u origin feature/ten-tinh-nang-cua-ban
```

#### Bước 6: Tạo Pull Request (PR)
1. Truy cập vào GitHub của dự án.
2. Bạn sẽ thấy thông báo tạo Pull Request từ nhánh vừa đẩy.
3. Chọn gộp nhánh vào **`dev`** (KHÔNG gộp trực tiếp vào `main`).
4. Gán tag TV1 (hoặc các bạn liên quan) để cùng review trước khi nhấn Merge.

---

## ✍️ 3. Quy Ước Đặt Tên Commit (Commit Convention)

Áp dụng chuẩn Conventional Commits ngắn gọn:

* `feat`: Thêm tính năng mới (Ví dụ: `feat(ui): them bo upload anh qua webcam`)
* `fix`: Sửa lỗi (Ví dụ: `fix(model): sua loi kich thuoc anh dau vao`)
* `docs`: Cập nhật tài liệu (Ví dụ: `docs: cap nhat huong dan cai dat`)
* `refactor`: Tối ưu/cấu trúc lại code nhưng không đổi logic (Ví dụ: `refactor(rules): tach tu dien quy chuan ra file rieng`)
* `test`: Thêm hoặc chỉnh sửa test case (Ví dụ: `test: them test case cho ham tra cuu luat`)
* `chore`: Các việc vặt (Ví dụ: `chore: them thu vien streamlit vao requirements.txt`)

---

## ⚠️ 4. Quy Tắc Vàng Về Dữ Liệu & File Nặng (BẮT BUỘC)

> [!CAUTION]
> **Tuyệt đối KHÔNG commit các tệp sau lên GitHub:**
> 1. Thư mục dữ liệu ảnh: `data/train/`, `data/test/`, `datasets/`
> 2. File nén: `*.zip`, `*.tar.gz`, `*.rar`
> 3. Trọng số mô hình có dung lượng lớn: `*.pt`, `*.h5`, `*.onnx` (file > 50MB)
> 4. Môi trường ảo: `venv/`, `.venv/`

### Cách quản lý file nặng:
* Tệp `.gitignore` đã được cấu hình sẵn để tự động bỏ qua các file này.
* **Với Dataset:** Chia sẻ qua link Google Drive hoặc tải trực tiếp từ Kaggle/TACO.
* **Với Model Weights:** File `best.pt` sau khi train thành công sẽ được tải lên Google Drive của nhóm hoặc đính kèm vào phần **Releases** trên GitHub.

---

## 🐍 5. Tiêu Chuẩn Code (Coding Standards)

1. **Chuẩn mã nguồn Python:**
   * Viết code rõ ràng, tuân thủ tinh thần PEP 8.
   * Đặt tên biến, hàm bằng tiếng Anh theo kiểu `snake_case` (ví dụ: `predict_waste()`, `get_recycling_advice()`).
   * Tên class đặt theo kiểu `CamelCase` (ví dụ: `WasteClassifier`, `RuleEngine`).

2. **Ghi chú & Chú thích (Comments):**
   * Vì nhóm có thành viên mới, mỗi hàm quan trọng bắt buộc phải có **docstring** ngắn giải thích:
     * Đầu vào (Args) là gì?
     * Đầu ra (Returns) là gì?
     * Ví dụ:
       ```python
       def get_waste_info(label: str) -> dict:
           """
           Tra cứu cơ sở tri thức theo nhãn phân loại từ mô hình.
           
           Args:
               label (str): Tên nhãn rác (vd: 'plastic_bottle', 'battery')
           Returns:
               dict: Chứa tên nhóm, màu thùng rác, mức độ độc hại và hướng dẫn tái chế.
           """
           pass
       ```

---

## 📋 6. Kế Hoạch 4 Tuần & Checklist Nghiệm Thu (Definition of Done)

* [ ] **Tuần 1:**
  * TV1 hoàn thành khung repo, PEAS, $(T, P, E)$.
  * TV2 thu thập được ít nhất 1.000 - 3.000 ảnh gán nhãn, chia 80% train / 20% test.
  * Nhóm cài đặt được môi trường ảo và chạy thử code mẫu.

* [ ] **Tuần 2:**
  * TV3 chạy thành công script `train.py` cho ra mô hình Baseline ban đầu.
  * TV4 hoàn thành khung hàm `waste_rules.py` với đầy đủ các nhóm rác cơ bản.

* [ ] **Tuần 3:**
  * TV3 tinh chỉnh siêu tham số, cải thiện Accuracy $\ge 85\%$.
  * TV5 dựng xong giao diện Streamlit cơ bản nhận ảnh và hiển thị thông tin.
  * Ghép nối: Model $\rightarrow$ Rules $\rightarrow$ UI.

* [ ] **Tuần 4:**
  * Đạt chỉ tiêu PEAS: Accuracy $\ge 90\%$, phản hồi $< 1.5$s.
  * Cả nhóm kiểm thử chéo, chuẩn bị Slide thuyết trình, Báo cáo đồ án và Video Demo.

---

## 💬 7. Kênh Liên Lạc & Giải Quyết Vấn Đề

* Nếu gặp lỗi kỹ thuật hoặc xung đột code khi `git pull`: **Không tự ý force push (`git push -f`)**. Hãy thông báo ngay lên nhóm Zalo/Discord của nhóm để cùng giải quyết.
* Khi có ý tưởng mới về cơ sở tri thức hay giao diện, hãy bàn bạc thống nhất với TV1 và cả nhóm trước khi triển khai.

