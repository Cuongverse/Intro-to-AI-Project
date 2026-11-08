"""
File: app.py
Mục đích: Giao diện Tác tử AI Nhận diện Rác thải & Tư vấn Tái chế dựa trên Bản thể học Ngữ cảnh
          (Context-Aware Waste Detection & Ontological Recycling Advisory System).
          Kết nối Tầng Nhận diện YOLOv12 (models/best.pt - Dipo et al., 2025)
          với Tầng Suy luận BFO Ontology (knowledge/waste_rules.py - Milivinti, 2025).
Môi trường chạy: Máy tính cá nhân (streamlit run app.py).
"""

from __future__ import annotations
import csv
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO

from knowledge.waste_rules import (
    AdvisoryDecision,
    BFOReasoningEngine,
    BFORole,
    ContaminationState,
    InfrastructureContext,
    MATERIAL_KNOWLEDGE_BASE,
    ObservationContext,
)


# =====================================================================
# 1. CẤU HÌNH TRANG & ĐƯỜNG DẪN HỆ THỐNG (SYSTEM PATHS & CONFIG)
# =====================================================================

st.set_page_config(
    page_title="HUST AI Agent | YOLOv12 & BFO Waste Advisory",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_WEIGHTS_PATH = PROJECT_ROOT / "models" / "best.pt"
METRICS_DIR = PROJECT_ROOT / "models" / "metrics"
OWL_FILE_PATH = PROJECT_ROOT / "knowledge" / "waste_bfo.owl"

# Bảng màu BGR (dùng cho OpenCV vẽ Bounding Box theo đúng màu Thùng rác đích)
ROLE_BGR_COLORS: Dict[BFORole, Tuple[int, int, int]] = {
    BFORole.RECYCLABLE_RESOURCE: (11, 158, 245),   # Vàng cam (#F59E0B)
    BFORole.ORGANIC_COMPOST: (129, 185, 16),       # Xanh lá (#10B981)
    BFORole.RESIDUAL_WASTE: (128, 114, 107),       # Xám (#6B7280)
    BFORole.HAZARDOUS_WASTE: (68, 68, 239),        # Đỏ cảnh báo (#EF4444)
    BFORole.UNCERTAIN_ENTITY: (246, 130, 59),      # Xanh dương (#3B82F6)
}


# =====================================================================
# 2. KHỞI TẠO SINGLETON CHO MODEL YOLOv12 & BFO ENGINE (CACHE RESOURCE)
# =====================================================================

@st.cache_resource(show_spinner="Đang nạp trọng số YOLOv12 (models/best.pt) và BFO Engine vào RAM...")
def load_agent_core() -> Tuple[YOLO, BFOReasoningEngine, str]:
    """
    Hàm Singleton nạp mô hình YOLOv12 và BFO Reasoning Engine đúng 1 lần duy nhất.
    Đảm bảo thời gian phản hồi suy luận < 0.5s trên CPU laptop cá nhân theo chuẩn PEAS.
    """
    if MODEL_WEIGHTS_PATH.exists():
        yolo_model = YOLO(str(MODEL_WEIGHTS_PATH))
        model_status = f"Đã nạp trọng số chuẩn: models/best.pt ({MODEL_WEIGHTS_PATH.stat().st_size / (1024*1024):.2f} MB)"
    else:
        yolo_model = YOLO("yolo12n.pt")
        model_status = "CẢNH BÁO: Chưa tìm thấy models/best.pt, đang dùng tạm yolo12n.pt gốc!"

    bfo_engine = BFOReasoningEngine(owl_path=OWL_FILE_PATH)
    return yolo_model, bfo_engine, model_status


# =====================================================================
# 3. CÁC HÀM XỬ LÝ ẢNH, VẼ BOUNDING BOX VÀ ĐỌC METRICS
# =====================================================================

def run_perception_and_reasoning(
    image_pil: Image.Image,
    model: YOLO,
    engine: BFOReasoningEngine,
    context: ObservationContext,
    iou_threshold: float = 0.45,
    min_det_conf: float = 0.25,
) -> Tuple[np.ndarray, List[Tuple[AdvisoryDecision, List[int]]], float]:
    """
    Thực thi toàn bộ chu trình Tác tử (Sense -> Perceive -> Reason -> Act):
      1. Chuẩn hóa ảnh và chạy suy luận YOLOv12.
      2. Đưa từng thực thể phát hiện được qua BFOReasoningEngine cùng với ObservationContext.
      3. Vẽ Bounding Box đổi màu theo Vai trò BFO (Role) lên ảnh.
    """
    img_rgb = np.array(image_pil.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    t_start = time.perf_counter()
    results = model.predict(
        source=img_bgr,
        imgsz=640,
        conf=min_det_conf,
        iou=iou_threshold,
        verbose=False,
    )
    t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    annotated_bgr = img_bgr.copy()
    decisions_with_boxes: List[Tuple[AdvisoryDecision, List[int]]] = []

    if results and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        names_map = results[0].names

        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            raw_label = names_map.get(cls_id, "Plastic")

            # Suy diễn vai trò BFO dựa trên Thực thể + Ngữ cảnh
            decision = engine.evaluate(
                detected_label=raw_label,
                confidence=conf,
                context=context,
            )
            decisions_with_boxes.append((decision, xyxy))

            # Vẽ Bounding Box với màu sắc tương ứng với Vai trò (Role) sau suy diễn
            x1, y1, x2, y2 = xyxy
            box_color = ROLE_BGR_COLORS.get(decision.assigned_role, (0, 200, 255))
            cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), box_color, 3)

            tag_text = f"[{ i + 1 }] {decision.material_category} ({conf:.0%})"
            (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            y_text_bg = max(y1 - th - 10, 0)
            cv2.rectangle(annotated_bgr, (x1, y_text_bg), (x1 + tw + 10, y_text_bg + th + 10), box_color, -1)
            cv2.putText(
                annotated_bgr,
                tag_text,
                (x1 + 5, y_text_bg + th + 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return annotated_rgb, decisions_with_boxes, t_elapsed_ms


def load_final_epoch_metrics(csv_path: Path) -> Optional[Dict[str, float]]:
    """Đọc dòng cuối cùng (hoặc epoch có mAP50 cao nhất) từ models/metrics/results.csv."""
    if not csv_path.exists():
        return None
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            if not reader:
                return None
            # Tìm dòng có metrics/mAP50(B) cao nhất (tương ứng với trọng số best.pt)
            best_row = max(
                reader,
                key=lambda r: float(r.get("metrics/mAP50(B)", r.get("       metrics/mAP50(B)", 0.0)) or 0.0)
            )
            clean_row = {k.strip(): float(v.strip()) for k, v in best_row.items() if v.strip()}
            p = clean_row.get("metrics/precision(B)", 0.0)
            r = clean_row.get("metrics/recall(B)", 0.0)
            map50 = clean_row.get("metrics/mAP50(B)", 0.0)
            map50_95 = clean_row.get("metrics/mAP50-95(B)", 0.0)
            f1 = (2 * p * r) / (p + r + 1e-8)
            return {
                "epoch": int(clean_row.get("epoch", 100)),
                "precision": p,
                "recall": r,
                "f1": f1,
                "map50": map50,
                "map50_95": map50_95,
            }
    except Exception:
        return None


# =====================================================================
# 4. RENDER GIAO DIỆN THANH BÊN (SIDEBAR - CẢM BIẾN NGỮ CẢNH BFO)
# =====================================================================

yolo_model, bfo_engine, model_status_msg = load_agent_core()

with st.sidebar:
    st.markdown("## ⚙️ Cảm biến Ngữ cảnh BFO")
    st.caption("Thiết lập thông số môi trường *(Partially Observable Environment)* theo Milivinti (2025):")

    contamination_labels = {
        ContaminationState.CLEAN_INTACT: "✨ Sạch sẽ & Nguyên vẹn (Clean & Intact)",
        ContaminationState.FOOD_GREASE_SOILED: "🍔 Bám bẩn dầu mỡ / Thức ăn thừa (Food/Grease Soiled)",
        ContaminationState.CHEMICAL_CONTAMINATED: "☣️ Nhiễm hóa chất độc hại / Dịch y tế (Chemical/Bio Hazard)",
        ContaminationState.BROKEN_SHARP: "🔨 Vỡ vụn sắc nhọn / Biến dạng rò rỉ (Broken & Sharp)",
    }

    selected_contamination = st.radio(
        "1. Tình trạng vật lý bề mặt vật thể:",
        options=list(contamination_labels.keys()),
        format_func=lambda x: contamination_labels[x],
        index=0,
    )

    infra_labels = {
        InfrastructureContext.FULL_RECYCLING_FACILITY: "🏭 Có hạ tầng thu hồi & tái chế tại chỗ (Full Recycling)",
        InfrastructureContext.BASIC_MUNICIPAL_ONLY: "🗑️ Chỉ có thu gom rác sinh hoạt cơ bản (Basic Municipal)",
    }

    selected_infra = st.radio(
        "2. Điều kiện hạ tầng kỹ thuật - kinh tế (Điều kiện c):",
        options=list(infra_labels.keys()),
        format_func=lambda x: infra_labels[x],
        index=0,
    )

    st.divider()
    st.markdown("### 🎯 Tham số Tiên đề BFO (Section 4.2)")
    cond_a_unfulfilled = st.checkbox(
        "Điều kiện (a): Vật thể đã hết công năng thiết kế ban đầu",
        value=True,
        disabled=True,
    )
    cond_b_disposal = st.checkbox(
        "Điều kiện (b): Người sở hữu có ý định thải bỏ tại trạm",
        value=True,
        disabled=True,
    )

    st.divider()
    st.markdown("### 🔍 Tinh chỉnh Cảm biến Thị giác")
    min_conf_slider = st.slider(
        "Ngưỡng phát hiện YOLO tối thiểu (Detection Floor):",
        min_value=0.15,
        max_value=0.90,
        value=0.25,
        step=0.05,
        help="Các vật thể có độ tin cậy >= ngưỡng này nhưng < 0.60 sẽ được bắt lại để kích hoạt cảnh báo yêu cầu chụp lại theo chuẩn PEAS.",
    )
    iou_slider = st.slider(
        "Ngưỡng chống chồng lấp NMS (IoU Threshold):",
        min_value=0.20,
        max_value=0.80,
        value=0.45,
        step=0.05,
    )

    st.divider()
    st.success(model_status_msg)
    st.info(f"OWL Schema: {'Đã kết nối waste_bfo.owl' if bfo_engine.owl_loaded else 'Chế độ Rule-based độc lập'}")

current_context = ObservationContext(
    contamination=selected_contamination,
    infrastructure=selected_infra,
    designed_function_fulfilled=not cond_a_unfulfilled,
    disposal_intention=cond_b_disposal,
)


# =====================================================================
# 5. TIÊU ĐỀ CHÍNH & HỆ THỐNG 3 TABS
# =====================================================================

st.title("♻️ Tác tử AI Nhận diện Rác thải & Tư vấn Tái chế dựa trên Bản thể học Ngữ cảnh")
st.markdown(
    "**Học phần:** Nhập môn Trí tuệ Nhân tạo (HUST) | "
    "**Perception:** `YOLOv12n` *(Dipo et al., 2025)* | "
    "**Reasoning:** `Basic Formal Ontology - BFO` *(Milivinti, 2025)*"
)

tab_agent, tab_metrics, tab_ontology = st.tabs([
    "🚀 Trạm Phân loại & Tư vấn BFO (Live Agent)",
    "📊 Kiểm định Thực nghiệm YOLOv12 (Metrics & Benchmark)",
    "🧠 Cơ sở Tri thức Bản thể học (BFO Knowledge Base)",
])


# ---------------------------------------------------------------------
# TAB 1: TRẠM PHÂN LOẠI & SUY LUẬN BẢN THỂ HỌC BFO THỜI GIAN THỰC
# ---------------------------------------------------------------------
with tab_agent:
    input_mode = st.radio(
        "Chọn nguồn tín hiệu Cảm biến hình ảnh (Sensors):",
        options=["📁 Tải ảnh kỹ thuật số (.jpg, .png)", "📷 Chụp trực tiếp từ Webcam trạm thu gom"],
        horizontal=True,
    )

    input_image_pil: Optional[Image.Image] = None

    if input_mode.startswith("📁"):
        uploaded_file = st.file_uploader(
            "Tải lên ảnh chứa một hoặc nhiều vật thể rác (Nhựa, Pin, Giấy, Thủy tinh, Hữu cơ, Y tế):",
            type=["jpg", "jpeg", "png", "webp"],
        )
        if uploaded_file is not None:
            input_image_pil = Image.open(uploaded_file)
    else:
        camera_photo = st.camera_input("Đặt vật thể rác trước camera và nhấn chụp:")
        if camera_photo is not None:
            input_image_pil = Image.open(camera_photo)

    if input_image_pil is not None:
        annotated_img, detections, latency_ms = run_perception_and_reasoning(
            image_pil=input_image_pil,
            model=yolo_model,
            engine=bfo_engine,
            context=current_context,
            iou_threshold=iou_slider,
            min_det_conf=min_conf_slider,
        )

        # Hàng chỉ số tổng quan PEAS
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Số thực thể phát hiện", f"{len(detections)} vật thể")
        kpi2.metric(
            "Thời gian phản hồi (Latency)",
            f"{latency_ms:.1f} ms",
            delta="Đạt chuẩn PEAS (< 500 ms)" if latency_ms < 500 else "Trên 500 ms",
        )
        hazard_count = sum(1 for d, _ in detections if d.is_hazardous_alert)
        kpi3.metric("Cảnh báo Nguy hại (Zero-tolerance)", f"{hazard_count} mục")
        retake_count = sum(1 for d, _ in detections if d.requires_retake)
        kpi4.metric("Cảnh báo Độ tin cậy thấp (<60%)", f"{retake_count} mục")

        st.divider()

        col_vis, col_reason = st.columns([1.05, 1.25], gap="large")

        with col_vis:
            st.subheader("👁️ Tầng Cảm nhận Thị giác (YOLOv12 Output)")
            st.image(
                annotated_img,
                caption=f"Ảnh đã chuẩn hóa và khoanh vùng thực thể vật chất (Thời gian suy luận: {latency_ms:.1f} ms)",
                use_container_width=True,
            )
            st.caption(
                "💡 **Thử nghiệm tính năng động của BFO:** Hãy giữ nguyên bức ảnh này và bấm đổi **Tình trạng vật lý** "
                "ở thanh công cụ bên trái từ *Sạch sẽ* sang *Bám bẩn dầu mỡ* hoặc *Nhiễm hóa chất*. "
                "Bạn sẽ thấy màu khung Bounding Box và Thùng rác đích tự động chuyển hóa mà không cần gán nhãn lại!"
            )

        with col_reason:
            st.subheader("🧠 Thẻ Phán quyết Bản thể học (BFO Reasoning Cards)")

            if not detections:
                st.warning(
                    "Không phát hiện được vật thể nào vượt qua ngưỡng lọc hiện tại. "
                    "Hãy thử giảm nhẹ thanh trượt 'Ngưỡng phát hiện YOLO tối thiểu' ở cột bên trái hoặc chụp ảnh rõ nét hơn."
                )
            else:
                for idx, (dec, bbox_coords) in enumerate(detections, start=1):
                    # Cảnh báo khẩn cấp nếu là Rác Nguy Hại (Zero-tolerance PEAS)
                    if dec.is_hazardous_alert:
                        st.error(
                            f"🚨 **CẢNH BÁO AN TOÀN CẤP ĐỘ CAO (Vật thể #{idx} - {dec.material_category}):** "
                            f"Phát hiện đặc tính nguy hại độc성/sinh học hoặc vật sắc nhọn! Tuyệt đối cô lập khỏi luồng rác thông thường."
                        )
                    elif dec.requires_retake:
                        st.warning(
                            f"⚠️ **YÊU CẦU CHỤP LẠI (Vật thể #{idx}):** Độ tin cậy cảm biến chỉ đạt **{dec.confidence:.1%}** "
                            f"(dưới ngưỡng an toàn PEAS 60%). Chưa đưa ra khuyến nghị vứt rác tự động."
                        )

                    # Khung hiển thị Thùng rác đích với màu chuẩn
                    st.markdown(
                        f"""
                        <div style="border-left: 8px solid {dec.target_bin_color_hex}; background-color: rgba(128,128,128,0.08);
                                    padding: 14px 18px; border-radius: 8px; margin-bottom: 12px;">
                            <h4 style="margin: 0; color: {dec.target_bin_color_hex};">
                                Vật thể #{idx}: {dec.target_bin_badge} — {dec.target_bin_name}
                            </h4>
                            <p style="margin: 6px 0 0 0; font-size: 0.95rem;">
                                <b>Thực thể Vật chất (YOLOv12):</b> <code>{dec.material_category}</code> ({dec.vietnamese_material_name})
                                &nbsp;|&nbsp; <b>Confidence:</b> <code>{dec.confidence:.1%}</code>
                                &nbsp;|&nbsp; <b>Tọa độ Box:</b> <code>{bbox_coords}</code>
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    with st.expander(f"🔎 Giải phẫu Chuỗi Suy diễn BFO & Hướng dẫn Sơ chế (Vật thể #{idx})", expanded=True):
                        st.markdown(f"- **Định danh Thực thể (`BFO:0000040`):** `{dec.bfo_entity_iri}`")
                        st.markdown(f"- **Công năng thiết kế (`BFO:0000034`):** {dec.designed_function}")
                        st.markdown(
                            "- **Đặc tính cố hữu (`BFO:0000016`):** "
                            + ", ".join(f"`{d}`" for d in dec.inherent_dispositions)
                        )
                        st.markdown(f"- **Vai trò được gán theo ngữ cảnh (`BFO:0000023`):** **`{dec.assigned_role.value}`**")

                        st.markdown("**Biểu thức Suy diễn Logic Vị từ (First-Order Logic Trace):**")
                        st.code(dec.fol_inference_trace, language="prolog")

                        st.markdown("**Kiểm tra 3 Điều kiện Định nghĩa Rác của Milivinti (2025 - Section 4.2):**")
                        for k_cond, v_cond in dec.bfo_conditions_check.items():
                            st.markdown(f"  - *{k_cond}:* **{v_cond}**")

                        st.markdown("**🛠️ Quy trình Sơ chế tại nguồn trước khi bỏ vào thùng (Actuator Advisory):**")
                        for step_i, step_text in enumerate(dec.preprocessing_steps, start=1):
                            st.markdown(f"  {step_i}. {step_text}")
    else:
        st.info("👆 Hãy tải lên một bức ảnh rác thải hoặc bật Webcam ở phía trên để bắt đầu phiên làm việc của Tác tử.")


# ---------------------------------------------------------------------
# TAB 2: KIỂM ĐỊNH THỰC NGHIỆM YOLOv12 (ĐỌC TỪ models/metrics/)
# ---------------------------------------------------------------------
with tab_metrics:
    st.subheader("📊 Kết quả Huấn luyện & Kiểm định Thực nghiệm trên GPU Tesla T4 (100 Epochs)")
    st.markdown(
        "Dữ liệu dưới đây được trích xuất trực tiếp từ thư mục `models/metrics/` sau khi chuẩn hóa bộ dữ liệu "
        "từ 16 lớp phân mảnh về **6 lớp Thực thể Vật chất BFO** và huấn luyện mô hình `YOLOv12n`."
    )

    csv_metrics = load_final_epoch_metrics(METRICS_DIR / "results.csv")
    if csv_metrics:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Best Epoch", f"{csv_metrics['epoch']}/100")
        m2.metric("Mean Precision (P)", f"{csv_metrics['precision']:.4f}", delta=f"{csv_metrics['precision'] - 0.73:+.4f} vs Paper")
        m3.metric("Mean Recall (R)", f"{csv_metrics['recall']:.4f}", delta=f"{csv_metrics['recall'] - 0.77:+.4f} vs Paper")
        m4.metric("F1-Score", f"{csv_metrics['f1']:.4f}", delta=f"{csv_metrics['f1'] - 0.75:+.4f} vs Mục tiêu PEAS (0.75)")
        m5.metric("mAP@0.5", f"{csv_metrics['map50']:.4f}", delta=f"{csv_metrics['map50'] - 0.78:+.4f} vs Mục tiêu PEAS (0.78)")

    st.divider()

    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.markdown("#### 1. Đồ thị Hội tụ Hàm mất mát & Độ chính xác (`results.png`)")
        results_png = METRICS_DIR / "results.png"
        if results_png.exists():
            st.image(str(results_png), caption="Biểu đồ box_loss, cls_loss, dfl_loss và mAP qua 100 epochs", use_container_width=True)
        else:
            st.warning("Chưa tìm thấy file models/metrics/results.png")

    with col_chart2:
        st.markdown("#### 2. Ma trận Nhầm lẫn Chuẩn hóa 6 Lớp BFO (`confusion_matrix_normalized.png`)")
        cm_norm_png = METRICS_DIR / "confusion_matrix_normalized.png"
        if cm_norm_png.exists():
            st.image(str(cm_norm_png), caption="Ma trận nhầm lẫn chuẩn hóa trên 6 Thực thể Vật chất", use_container_width=True)
        else:
            st.warning("Chưa tìm thấy file models/metrics/confusion_matrix_normalized.png")

    col_chart3, col_chart4 = st.columns(2)
    with col_chart3:
        st.markdown("#### 3. Ma trận Nhầm lẫn Số lượng Tuyệt đối (`confusion_matrix.png`)")
        cm_png = METRICS_DIR / "confusion_matrix.png"
        if cm_png.exists():
            st.image(str(cm_png), use_container_width=True)

    with col_chart4:
        st.markdown("#### 4. Phân bố Tập dữ liệu sau khi Gộp nhãn (`labels.jpg`)")
        labels_jpg = METRICS_DIR / "labels.jpg"
        if labels_jpg.exists():
            st.image(str(labels_jpg), use_container_width=True)


# ---------------------------------------------------------------------
# TAB 3: CƠ SỞ TRI THỨC BẢN THỂ HỌC BFO (ONTOLOGY INSPECTOR)
# ---------------------------------------------------------------------
with tab_ontology:
    st.subheader("🧠 Cấu trúc Bản thể học BFO (Milivinti, 2025 - FOIS Proceedings)")
    st.markdown(
        "Khác với cách tiếp cận truyền thống coi *'rác'* là thuộc tính cố hữu của vật thể, khung **Basic Formal Ontology (BFO)** "
        "tách biệt rõ **Thực thể Vật chất (`BFO:0000040`)**, **Công năng thiết kế (`BFO:0000034`)**, "
        "**Đặc tính cố hữu (`BFO:0000016`)** và **Vai trò phụ thuộc ngữ cảnh (`BFO:0000023`)**."
    )

    ontology_table_rows = []
    for cat, prof in MATERIAL_KNOWLEDGE_BASE.items():
        ontology_table_rows.append({
            "Class YOLOv12": cat.value,
            "Tên Tiếng Việt": prof.vietnamese_name,
            "Định danh BFO (IRI)": prof.bfo_iri,
            "Mã EWC (Châu Âu)": prof.ewc_reference_code,
            "Công năng Thiết kế (BFO:0000034)": prof.designed_function.value,
            "Đặc tính Cố hữu (BFO:0000016)": " | ".join(d.name for d in prof.inherent_dispositions),
        })

    st.table(ontology_table_rows)

    if OWL_FILE_PATH.exists():
        with st.expander("📄 Xem mã nguồn Tệp Bản thể học chuẩn OWL 2 RDF/XML (`knowledge/waste_bfo.owl`)"):
            owl_content = OWL_FILE_PATH.read_text(encoding="utf-8")
            st.code(owl_content, language="xml")