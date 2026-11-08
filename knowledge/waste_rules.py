"""
File: knowledge/waste_rules.py
Mục đích: Hiện thực hóa Tầng Suy luận Bản thể học Ngữ cảnh (BFO Ontology Reasoning Engine)
          kết nối đầu ra của YOLOv12 (Dipo et al., 2025) với khung lý thuyết Basic Formal Ontology
          của Elena Milivinti (2025) và tiêu chuẩn PEAS học phần Nhập môn TTNT (HUST).
Môi trường chạy: Máy tính cá nhân (Local Python 3.10+).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# =====================================================================
# 1. ĐỊNH NGHĨA CÁC CẤP ĐỘ THỰC THỂ BFO (BFO 2020 TAXONOMY ENUMERATIONS)
# =====================================================================

class MaterialCategory(str, Enum):
    """
    6 Thực thể Vật chất độc lập (Independent Continuant - BFO:0000040)
    khớp 100% với 6 class đầu ra của mô hình YOLOv12 (models/best.pt).
    """
    PLASTIC = "Plastic"
    METAL_BATTERY = "Metal_Battery"
    PAPER = "Paper"
    GLASS = "Glass"
    ORGANIC = "Organic"
    MEDICAL = "Medical"


class BFOFunction(str, Enum):
    """
    Công năng thiết kế ban đầu của vật thể (Function - BFO:0000034).
    """
    PACKAGING_CONTAINMENT = "Containment & Packaging Function (BFO:0000034)"
    ENERGY_STORAGE = "Electrochemical Energy Storage Function (BFO:0000034)"
    INFORMATION_CARRIER = "Print & Fiber Structural Function (BFO:0000034)"
    BIOLOGICAL_NUTRITION = "Organic / Biological Matter Function (BFO:0000034)"
    CLINICAL_PROTECTION = "Medical & Sanitary Protection Function (BFO:0000034)"


class Disposition(str, Enum):
    """
    Đặc tính vật lý/hóa học cố hữu của vật liệu (Disposition - BFO:0000016).
    Không thay đổi dù vật thể đang được sử dụng hay bị vứt bỏ.
    """
    MECHANICAL_RECYCLABILITY = "MechanicalRecyclability (Có khả năng tái chế cơ/nhiệt)"
    BIODEGRADABILITY = "Biodegradability (Có khả năng phân hủy sinh học)"
    TOXICITY_HAZARD = "ToxicityHazard (Độc hại hóa học / Nguy cơ rò rỉ kim loại nặng)"
    INFECTIOUS_BIOHAZARD = "InfectiousBiohazard (Nguy cơ lây nhiễm mầm bệnh sinh học)"
    SHARP_PHYSICAL_HAZARD = "SharpPhysicalHazard (Nguy cơ gây sát thương cơ học sắc nhọn)"
    INERT_STABILITY = "InertStability (Trơ hóa học, bền vững vật lý)"


class BFORole(str, Enum):
    """
    Vai trò động phụ thuộc ngữ cảnh (Role - BFO:0000023).
    Được suy diễn tại thời điểm phân loại dựa trên 3 điều kiện (a, b, c) của Milivinti (2025).
    """
    RECYCLABLE_RESOURCE = "RecyclableResourceRole (Tài nguyên Tái chế)"
    ORGANIC_COMPOST = "OrganicCompostRole (Nguyên liệu Phân hủy Sinh học)"
    RESIDUAL_WASTE = "ResidualWasteRole (Rác thải Sinh hoạt Còn lại)"
    HAZARDOUS_WASTE = "HazardousWasteRole (Rác thải Nguy hại Đặc biệt)"
    UNCERTAIN_ENTITY = "UncertainObservationRole (Quan sát Chưa đủ Độ tin cậy)"


# =====================================================================
# 2. ĐỊNH NGHĨA NGỮ CẢNH KHÔNG-THỜI GIAN (SPATIOTEMPORAL CONTEXT)
# =====================================================================

class ContaminationState(str, Enum):
    """Tình trạng vật lý bề mặt của vật thể tại trạm thu gom (Sensors phụ trợ / User Input)."""
    CLEAN_INTACT = "CLEAN_INTACT"               # Sạch sẽ, khô ráo, không lẫn tạp chất
    FOOD_GREASE_SOILED = "FOOD_GREASE_SOILED"   # Bám bẩn dầu mỡ, thức ăn thừa khó tách rời
    CHEMICAL_CONTAMINATED = "CHEMICAL_CONTAMINATED"  # Nhiễm hóa chất độc hại, dung môi, dịch y tế
    BROKEN_SHARP = "BROKEN_SHARP"               # Vỡ vụn sắc nhọn hoặc biến dạng rò rỉ


class InfrastructureContext(str, Enum):
    """Điều kiện hạ tầng kỹ thuật - kinh tế tại điểm thu gom (Điều kiện c - Milivinti, 2025)."""
    FULL_RECYCLING_FACILITY = "FULL_RECYCLING_FACILITY"     # Có chuỗi thu hồi tái chế chuyên biệt
    BASIC_MUNICIPAL_ONLY = "BASIC_MUNICIPAL_ONLY"           # Chỉ có thu gom rác sinh hoạt & nguy hại cơ bản


@dataclass(frozen=True)
class ObservationContext:
    """
    Đóng gói toàn bộ thông tin ngữ cảnh tại thời điểm tác tử quan sát vật thể.
    Theo Milivinti (2025) Section 4.2:
      (a) designed_function_fulfilled: Công năng thiết kế còn được đáp ứng không? (Mặc định False khi đưa vào trạm rác)
      (b) disposal_intention: Người cầm có ý định thải bỏ không? (Mặc định True tại trạm rác)
      (c) contamination & infrastructure: Quyết định tính khả thi kinh tế/kỹ thuật để hiện thực hóa Disposition.
    """
    contamination: ContaminationState = ContaminationState.CLEAN_INTACT
    infrastructure: InfrastructureContext = InfrastructureContext.FULL_RECYCLING_FACILITY
    designed_function_fulfilled: bool = False
    disposal_intention: bool = True


# =====================================================================
# 3. CẤU TRÚC THỰC THỂ VẬT CHẤT & KẾT QUẢ PHÁN QUYẾT (DATA STRUCTURES)
# =====================================================================

@dataclass(frozen=True)
class MaterialEntityProfile:
    """Biểu diễn một lớp Thực thể Vật chất (BFO:0000040) trong Cơ sở Tri thức."""
    category: MaterialCategory
    bfo_iri: str
    vietnamese_name: str
    designed_function: BFOFunction
    inherent_dispositions: Tuple[Disposition, ...]
    ewc_reference_code: str  # Mã danh mục chất thải châu Âu (European Waste Catalogue - Table 1 Paper 2)


@dataclass
class AdvisoryDecision:
    """
    Đối tượng đầu ra của Tầng Suy luận (Actuator Payload) gửi tới giao diện Streamlit (app.py).
    """
    material_category: str
    vietnamese_material_name: str
    confidence: float
    bfo_entity_iri: str
    designed_function: str
    inherent_dispositions: List[str]
    assigned_role: BFORole
    target_bin_name: str
    target_bin_color_hex: str
    target_bin_badge: str
    is_hazardous_alert: bool
    requires_retake: bool
    fol_inference_trace: str       # Chuỗi biểu thức Logic vị từ giải thích lý do suy diễn
    bfo_conditions_check: Dict[str, str]
    preprocessing_steps: List[str]


# =====================================================================
# 4. CƠ SỞ TRI THỨC BẢN THỂ HỌC (ONTOLOGICAL KNOWLEDGE BASE REGISTRY)
# =====================================================================

MATERIAL_KNOWLEDGE_BASE: Dict[MaterialCategory, MaterialEntityProfile] = {
    MaterialCategory.PLASTIC: MaterialEntityProfile(
        category=MaterialCategory.PLASTIC,
        bfo_iri="BFO:0000040/PlasticArtifact",
        vietnamese_name="Nhựa & Bao bì Polymer (PET, HDPE, Túi nilon)",
        designed_function=BFOFunction.PACKAGING_CONTAINMENT,
        inherent_dispositions=(
            Disposition.MECHANICAL_RECYCLABILITY,
            Disposition.INERT_STABILITY,
        ),
        ewc_reference_code="EWC 15 01 02 / 20 01 39"
    ),
    MaterialCategory.METAL_BATTERY: MaterialEntityProfile(
        category=MaterialCategory.METAL_BATTERY,
        bfo_iri="BFO:0000040/ElectrochemicalBatteryCell",
        vietnamese_name="Pin, Ắc quy & Linh kiện Kim loại Nguy hại",
        designed_function=BFOFunction.ENERGY_STORAGE,
        inherent_dispositions=(
            Disposition.TOXICITY_HAZARD,
            Disposition.MECHANICAL_RECYCLABILITY,
        ),
        ewc_reference_code="EWC 20 01 33* / 16 06 01*"
    ),
    MaterialCategory.PAPER: MaterialEntityProfile(
        category=MaterialCategory.PAPER,
        bfo_iri="BFO:0000040/CellulosePaperProduct",
        vietnamese_name="Giấy, Báo chí & Bìa Carton",
        designed_function=BFOFunction.INFORMATION_CARRIER,
        inherent_dispositions=(
            Disposition.MECHANICAL_RECYCLABILITY,
            Disposition.BIODEGRADABILITY,
        ),
        ewc_reference_code="EWC 15 01 01 / 20 01 01"
    ),
    MaterialCategory.GLASS: MaterialEntityProfile(
        category=MaterialCategory.GLASS,
        bfo_iri="BFO:0000040/SilicateGlassContainer",
        vietnamese_name="Thủy tinh & Chai lọ Kính",
        designed_function=BFOFunction.PACKAGING_CONTAINMENT,
        inherent_dispositions=(
            Disposition.MECHANICAL_RECYCLABILITY,
            Disposition.INERT_STABILITY,
        ),
        ewc_reference_code="EWC 15 01 07 / 20 01 02"
    ),
    MaterialCategory.ORGANIC: MaterialEntityProfile(
        category=MaterialCategory.ORGANIC,
        bfo_iri="BFO:0000040/BiologicalOrganicMatter",
        vietnamese_name="Rác Hữu cơ & Phế phẩm Sinh học",
        designed_function=BFOFunction.BIOLOGICAL_NUTRITION,
        inherent_dispositions=(
            Disposition.BIODEGRADABILITY,
        ),
        ewc_reference_code="EWC 20 01 08"
    ),
    MaterialCategory.MEDICAL: MaterialEntityProfile(
        category=MaterialCategory.MEDICAL,
        bfo_iri="BFO:0000040/ClinicalMedicalArtifact",
        vietnamese_name="Rác thải Y tế (Kim tiêm, Khẩu trang, Găng tay, Thuốc)",
        designed_function=BFOFunction.CLINICAL_PROTECTION,
        inherent_dispositions=(
            Disposition.INFECTIOUS_BIOHAZARD,
            Disposition.TOXICITY_HAZARD,
            Disposition.SHARP_PHYSICAL_HAZARD,
        ),
        ewc_reference_code="EWC 18 01 01 / 18 01 03* / 20 01 31*"
    ),
}


# Quy chuẩn 4 loại thùng rác tại nguồn (Mục 2 PEAS - PROJECT_CONTEXT.md)
BIN_SPECIFICATIONS = {
    BFORole.RECYCLABLE_RESOURCE: {
        "name": "THÙNG VÀNG - RÁC TÁI CHẾ (RECYCLABLE)",
        "color_hex": "#F59E0B",
        "badge": "🟡 Thùng Tái Chế"
    },
    BFORole.ORGANIC_COMPOST: {
        "name": "THÙNG XANH LÁ - RÁC HỮU CƠ (ORGANIC)",
        "color_hex": "#10B981",
        "badge": "🟢 Thùng Hữu Cơ"
    },
    BFORole.RESIDUAL_WASTE: {
        "name": "THÙNG XÁM/TRẮNG - RÁC CÒN LẠI (RESIDUAL WASTE)",
        "color_hex": "#6B7280",
        "badge": "⚪ Thùng Rác Còn Lại"
    },
    BFORole.HAZARDOUS_WASTE: {
        "name": "THÙNG ĐỎ - RÁC NGUY HẠI ĐẶC BIỆT (HAZARDOUS)",
        "color_hex": "#EF4444",
        "badge": "🔴 Thùng Nguy Hại"
    },
    BFORole.UNCERTAIN_ENTITY: {
        "name": "CẢNH BÁO ĐỘ TIN CẬY THẤP - YÊU CẦU CHỤP LẠI",
        "color_hex": "#3B82F6",
        "badge": "🔵 Cần Xác Minh Lại"
    },
}


# =====================================================================
# 5. ĐỘNG CƠ SUY DIỄN BẢN THỂ HỌC BFO (BFO ONTOLOGICAL REASONING ENGINE)
# =====================================================================

class BFOReasoningEngine:
    """
    Động cơ suy diễn kết hợp kết quả nhận diện từ YOLOv12 và Ngữ cảnh (ObservationContext)
    để gán Vai trò (BFORole) theo đúng định nghĩa hình thức của Milivinti (2025).
    """

    CONFIDENCE_THRESHOLD: float = 0.60  # Tiêu chuẩn PEAS trong PROJECT_CONTEXT.md

    def __init__(self, owl_path: Optional[Path] = None) -> None:
        self.owl_path = owl_path or Path("knowledge/waste_bfo.owl")
        self.owl_loaded = self.owl_path.exists()

    def evaluate(
        self,
        detected_label: str,
        confidence: float,
        context: ObservationContext = ObservationContext()
    ) -> AdvisoryDecision:
        """
        Hàm suy diễn trung tâm:
        Input:
          - detected_label: Chuỗi tên lớp xuất ra từ YOLOv12 ('Plastic', 'Metal_Battery',...)
          - confidence: Điểm độ tin cậy trong khoảng [0.0, 1.0]
          - context: Đối tượng ngữ cảnh chứa trạng thái nhiễm bẩn và hạ tầng thu gom
        Output:
          - AdvisoryDecision: Phán quyết đầy đủ cho tầng Actuator (UI Streamlit)
        """
        # Chuẩn hóa nhãn đầu vào về Enum MaterialCategory
        try:
            category = MaterialCategory(detected_label)
        except ValueError:
            # Hỗ trợ tương thích ngược nếu model trả về tên viết thường hoặc biến thể
            normalized_map = {
                "plastic": MaterialCategory.PLASTIC,
                "metal_battery": MaterialCategory.METAL_BATTERY,
                "battery": MaterialCategory.METAL_BATTERY,
                "paper": MaterialCategory.PAPER,
                "glass": MaterialCategory.GLASS,
                "organic": MaterialCategory.ORGANIC,
                "medical": MaterialCategory.MEDICAL,
            }
            category = normalized_map.get(detected_label.lower(), MaterialCategory.PLASTIC)

        profile = MATERIAL_KNOWLEDGE_BASE[category]
        dispositions = profile.inherent_dispositions

        # Kiểm tra 3 điều kiện nền tảng (a, b, c) của Milivinti (2025) - Section 4.2
        cond_a = not context.designed_function_fulfilled
        cond_b = context.disposal_intention

        # -----------------------------------------------------------------
        # LUẬT 0: KIỂM TRA NGƯỠNG TIN CẬY CẢM BIẾN (PEAS SENSOR GATE < 0.60)
        # -----------------------------------------------------------------
        if confidence < self.CONFIDENCE_THRESHOLD:
            role = BFORole.UNCERTAIN_ENTITY
            bin_info = BIN_SPECIFICATIONS[role]
            fol_trace = (
                f"Confidence({category.value}, {confidence:.2f}) < {self.CONFIDENCE_THRESHOLD:.2f} "
                f"-> Trigger(UncertainObservationRole)"
            )
            return AdvisoryDecision(
                material_category=category.value,
                vietnamese_material_name=profile.vietnamese_name,
                confidence=confidence,
                bfo_entity_iri=profile.bfo_iri,
                designed_function=profile.designed_function.value,
                inherent_dispositions=[d.value for d in dispositions],
                assigned_role=role,
                target_bin_name=bin_info["name"],
                target_bin_color_hex=bin_info["color_hex"],
                target_bin_badge=bin_info["badge"],
                is_hazardous_alert=False,
                requires_retake=True,
                fol_inference_trace=fol_trace,
                bfo_conditions_check={
                    "Condition (a) [Function Unfulfilled]": str(cond_a),
                    "Condition (b) [Disposal Intention]": str(cond_b),
                    "Sensor Gate [Confidence >= 0.60]": f"FAILED ({confidence:.1%})",
                },
                preprocessing_steps=[
                    "Độ tin cậy thị giác dưới ngưỡng an toàn PEAS (60%).",
                    "Đặt vật thể ra vùng đủ sáng, tách rời các vật thể bị chồng lấp và chụp lại."
                ]
            )

        # -----------------------------------------------------------------
        # LUẬT 1: ZERO-TOLERANCE VỚI ĐẶC TÍNH ĐỘC HẠI CỐ HỮU HOẶC NHIỄM HÓA CHẤT
        # (Pin/Ắc quy, Rác Y tế, hoặc bất kỳ vật liệu nào bị nhiễm hóa chất/dịch y tế)
        # -----------------------------------------------------------------
        has_inherent_hazard = (
            Disposition.TOXICITY_HAZARD in dispositions
            or Disposition.INFECTIOUS_BIOHAZARD in dispositions
        )
        is_chemically_contaminated = (
            context.contamination == ContaminationState.CHEMICAL_CONTAMINATED
        )

        if has_inherent_hazard or is_chemically_contaminated:
            role = BFORole.HAZARDOUS_WASTE
            bin_info = BIN_SPECIFICATIONS[role]
            reason_pred = (
                "HasDisposition(x, Toxicity/Biohazard)"
                if has_inherent_hazard
                else "Context(x, ChemicalContaminated)"
            )
            fol_trace = (
                f"MaterialEntity({category.value}) ∧ {reason_pred} "
                f"-> HasRole(x, HazardousWasteRole) [Zero-Tolerance Rule]"
            )
            steps = self._get_preprocessing_instructions(category, role, context)
            return self._build_decision(
                profile, confidence, role, bin_info,
                is_hazardous=True,
                fol_trace=fol_trace,
                cond_a=cond_a, cond_b=cond_b,
                cond_c_status="Bắt buộc cô lập an toàn sinh học/hóa học (EWC Hazardous*)",
                steps=steps
            )

        # -----------------------------------------------------------------
        # LUẬT 2: THỦY TINH VỠ SẮC NHỌN (XỬ LÝ AN TOÀN CƠ HỌC ĐẶC THÙ)
        # -----------------------------------------------------------------
        if category == MaterialCategory.GLASS and context.contamination == ContaminationState.BROKEN_SHARP:
            role = BFORole.RESIDUAL_WASTE
            bin_info = BIN_SPECIFICATIONS[role]
            fol_trace = (
                f"MaterialEntity(Glass) ∧ Context(x, BrokenSharp) "
                f"-> HasRole(x, ResidualWasteRole) [Safety Isolation]"
            )
            steps = [
                "CẢNH BÁO AN TOÀN: Thủy tinh đã vỡ có nguy cơ gây đứt tay công nhân thu gom.",
                "Bọc kín các mảnh vỡ trong nhiều lớp giấy báo dày hoặc hộp carton cứng.",
                "Dán băng dính bên ngoài và ghi rõ 'MẢNH THỦY TINH VỠ' trước khi bỏ vào Thùng Rác Còn Lại."
            ]
            return self._build_decision(
                profile, confidence, role, bin_info,
                is_hazardous=True,
                fol_trace=fol_trace,
                cond_a=cond_a, cond_b=cond_b,
                cond_c_status="Mảnh vỡ nhỏ lẫn tạp chất không khả thi để phân loại quang học tại trạm",
                steps=steps
            )

        # -----------------------------------------------------------------
        # LUẬT 3: HIỆN THỰC HÓA ĐẶC TÍNH PHÂN HỦY SINH HỌC (ORGANIC & PAPER BẨN THỨC ĂN)
        # -----------------------------------------------------------------
        if category == MaterialCategory.ORGANIC:
            role = BFORole.ORGANIC_COMPOST
            bin_info = BIN_SPECIFICATIONS[role]
            fol_trace = (
                "MaterialEntity(Organic) ∧ HasDisposition(x, Biodegradability) ∧ ¬Contaminated(Chemical) "
                "-> HasRole(x, OrganicCompostRole)"
            )
            steps = self._get_preprocessing_instructions(category, role, context)
            return self._build_decision(
                profile, confidence, role, bin_info,
                is_hazardous=False,
                fol_trace=fol_trace,
                cond_a=cond_a, cond_b=cond_b,
                cond_c_status="Khả thi hiện thực hóa quá trình ủ phân vi sinh (Composting Process)",
                steps=steps
            )

        # -----------------------------------------------------------------
        # LUẬT 4: ĐÁNH GIÁ ĐIỀU KIỆN (C) CỦA MILIVINTI CHO VẬT LIỆU CÓ TÍNH TÁI CHẾ
        # (Plastic, Paper, Glass)
        # -----------------------------------------------------------------
        if Disposition.MECHANICAL_RECYCLABILITY in dispositions:
            is_clean = (context.contamination == ContaminationState.CLEAN_INTACT)
            has_infra = (context.infrastructure == InfrastructureContext.FULL_RECYCLING_FACILITY)

            # Trường hợp 4A: Sạch + Có hạ tầng tái chế -> Điều kiện (c) bị bác bỏ -> Là TÀI NGUYÊN TÁI CHẾ
            if is_clean and has_infra:
                role = BFORole.RECYCLABLE_RESOURCE
                bin_info = BIN_SPECIFICATIONS[role]
                fol_trace = (
                    f"MaterialEntity({category.value}) ∧ HasDisposition(x, MechanicalRecyclability) "
                    f"∧ Context(x, CleanIntact) ∧ Infra(FullRecycling) -> HasRole(x, RecyclableResourceRole)"
                )
                steps = self._get_preprocessing_instructions(category, role, context)
                return self._build_decision(
                    profile, confidence, role, bin_info,
                    is_hazardous=False,
                    fol_trace=fol_trace,
                    cond_a=cond_a, cond_b=cond_b,
                    cond_c_status="Thỏa mãn kinh tế tuần hoàn: Hiện thực hóa Disposition tại quy trình Recycling (BFO:0000015)",
                    steps=steps
                )

            # Trường hợp 4B: Bám bẩn dầu mỡ thực phẩm hoặc Không có hạ tầng tái chế
            # -> Thỏa mãn cả 3 điều kiện (a, b, c) của Milivinti (2025) -> Chuyển thành RÁC CÒN LẠI
            role = BFORole.RESIDUAL_WASTE
            bin_info = BIN_SPECIFICATIONS[role]
            degradation_cause = (
                "Context(x, FoodGreaseSoiled)" if not is_clean else "¬Infra(FullRecycling)"
            )
            fol_trace = (
                f"MaterialEntity({category.value}) ∧ HasDisposition(x, MechanicalRecyclability) "
                f"∧ {degradation_cause} -> HasRole(x, ResidualWasteRole) [Milivinti Condition (c) Met]"
            )
            steps = self._get_preprocessing_instructions(category, role, context)
            return self._build_decision(
                profile, confidence, role, bin_info,
                is_hazardous=False,
                fol_trace=fol_trace,
                cond_a=cond_a, cond_b=cond_b,
                cond_c_status=(
                    "Không khả thi về kinh tế/kỹ thuật để tái chế do nhiễm bẩn dầu mỡ hoặc thiếu hạ tầng thu hồi"
                ),
                steps=steps
            )

        # Nhánh dự phòng mặc định
        role = BFORole.RESIDUAL_WASTE
        bin_info = BIN_SPECIFICATIONS[role]
        return self._build_decision(
            profile, confidence, role, bin_info,
            is_hazardous=False,
            fol_trace=f"MaterialEntity({category.value}) -> HasRole(x, ResidualWasteRole)",
            cond_a=cond_a, cond_b=cond_b,
            cond_c_status="Xử lý theo luồng rác sinh hoạt thông thường",
            steps=["Bỏ vào thùng rác sinh hoạt còn lại."]
        )

    def _build_decision(
        self,
        profile: MaterialEntityProfile,
        confidence: float,
        role: BFORole,
        bin_info: Dict[str, str],
        is_hazardous: bool,
        fol_trace: str,
        cond_a: bool,
        cond_b: bool,
        cond_c_status: str,
        steps: List[str]
    ) -> AdvisoryDecision:
        return AdvisoryDecision(
            material_category=profile.category.value,
            vietnamese_material_name=profile.vietnamese_name,
            confidence=confidence,
            bfo_entity_iri=f"{profile.bfo_iri} ({profile.ewc_reference_code})",
            designed_function=profile.designed_function.value,
            inherent_dispositions=[d.value for d in profile.inherent_dispositions],
            assigned_role=role,
            target_bin_name=bin_info["name"],
            target_bin_color_hex=bin_info["color_hex"],
            target_bin_badge=bin_info["badge"],
            is_hazardous_alert=is_hazardous,
            requires_retake=False,
            fol_inference_trace=fol_trace,
            bfo_conditions_check={
                "Điều kiện (a) [Hết công năng thiết kế]": "Đúng (True)" if cond_a else "Sai (False)",
                "Điều kiện (b) [Có ý định thải bỏ]": "Đúng (True)" if cond_b else "Sai (False)",
                "Điều kiện (c) [Khả thi hiện thực hóa Disposition]": cond_c_status,
            },
            preprocessing_steps=steps
        )

    @staticmethod
    def _get_preprocessing_instructions(
        category: MaterialCategory,
        role: BFORole,
        context: ObservationContext
    ) -> List[str]:
        """Sinh hướng dẫn sơ chế tại nguồn cụ thể theo từng tổ hợp (Thực thể + Vai trò)."""
        if role == BFORole.HAZARDOUS_WASTE:
            if category == MaterialCategory.METAL_BATTERY:
                return [
                    "CẢNH BÁO NGUY HẠI: Tuyệt đối KHÔNG vứt chung vào thùng rác sinh hoạt hoặc thùng tái chế thông thường (nguy cơ cháy nổ & rò rỉ chì/thủy ngân/lithium).",
                    "Dùng băng dính cách điện dán kín 2 đầu cực của viên pin/ắc quy để chống đoản mạch.",
                    "Đặt nhẹ nhàng vào Thùng Đỏ (Rác Nguy Hại) hoặc điểm thu hồi pin cũ chuyên dụng."
                ]
            if category == MaterialCategory.MEDICAL:
                return [
                    "CẢNH BÁO SINH HỌC: Vật phẩm y tế có nguy cơ lây nhiễm mầm bệnh hoặc gây thương tích kim đâm.",
                    "Đối với bơm kim tiêm: Đậy nắp an toàn (nếu có) và bỏ vào hộp kháng thủng chuyên dụng.",
                    "Đối với khẩu trang, găng tay, vỉ thuốc: Buộc kín trong túi riêng trước khi bỏ vào Thùng Đỏ (Rác Nguy Hại)."
                ]
            return [
                "Vật thể đã bị nhiễm hóa chất độc hại/dung môi công nghiệp, không thể tái chế.",
                "Cô lập trong túi kín và bỏ vào Thùng Đỏ (Rác Nguy Hại) để xử lý đốt tiêu hủy an toàn."
            ]

        if role == BFORole.RECYCLABLE_RESOURCE:
            if category == MaterialCategory.PLASTIC:
                return [
                    "Đổ hết chất lỏng còn sót lại và tráng sơ bằng nước sạch.",
                    "Tháo rời nắp chai và nhãn màng co (nếu dễ bóc).",
                    "Bóp dẹp vỏ chai/hộp nhựa dọc theo thân để tiết kiệm tối đa thể tích thùng chứa rồi bỏ vào Thùng Vàng."
                ]
            if category == MaterialCategory.PAPER:
                return [
                    "Gỡ bỏ ghim bấm kim loại, băng keo nhựa hoặc dây buộc trên bề mặt giấy/bìa carton.",
                    "Gấp phẳng các thùng carton, giữ khô ráo và đặt vào Thùng Vàng (Tái chế)."
                ]
            if category == MaterialCategory.GLASS:
                return [
                    "Súc rửa sạch cặn đồ uống bên trong chai/lọ thủy tinh.",
                    "Tháo nắp kim loại hoặc nắp nhựa ra riêng.",
                    "Đặt nhẹ nhàng vào Thùng Vàng (Tái chế), tránh ném mạnh gây vỡ."
                ]

        if role == BFORole.ORGANIC_COMPOST:
            return [
                "Loại bỏ toàn bộ tăm tre, túi nilon, dây buộc hoặc bao bì nhựa lẫn trong rác hữu cơ.",
                "Chắt bỏ phần nước súp/dầu thừa trước khi đổ vào Thùng Xanh Lá (Rác Hữu Cơ) để ủ phân компост."
            ]

        # Trường hợp BFORole.RESIDUAL_WASTE (Suy thoái vai trò do bẩn dầu mỡ hoặc thiếu hạ tầng)
        if context.contamination == ContaminationState.FOOD_GREASE_SOILED:
            return [
                f"Mặc dù '{category.value}' có đặc tính cố hữu là tái chế được, nhưng do đã bám bẩn dầu mỡ/thức ăn nặng nên sẽ làm hỏng cả lô tái chế (Theo Milivinti 2025 - Điều kiện c).",
                "Gạt bỏ phần thức ăn thừa lớn vào thùng Hữu cơ (nếu có).",
                "Gói gọn bao bì/giấy bẩn này và bỏ vào Thùng Xám/Trắng (Rác Còn Lại)."
            ]

        return [
            "Khu vực hiện tại chưa có hạ tầng thu hồi tái chế chuyên biệt cho nhóm vật liệu này.",
            "Ép gọn vật thể để giảm thể tích và bỏ vào Thùng Xám/Trắng (Rác Còn Lại)."
        ]


# =====================================================================
# 6. KHỐI KIỂM THỬ ĐỘC LẬP (UNIT TEST TRÊN MÁY CÁ NHÂN)
# Chạy lệnh: python knowledge/waste_rules.py
# =====================================================================
if __name__ == "__main__":
    engine = BFOReasoningEngine()

    test_scenarios = [
        # Kịch bản 1: Chai nhựa sạch tại trạm có hạ tầng tái chế -> Tài nguyên tái chế
        ("Plastic", 0.92, ObservationContext(
            contamination=ContaminationState.CLEAN_INTACT,
            infrastructure=InfrastructureContext.FULL_RECYCLING_FACILITY
        )),
        # Kịch bản 2: Cùng chai/túi nhựa đó nhưng dính đầy dầu mỡ thức ăn -> Suy thoái thành Rác còn lại
        ("Plastic", 0.88, ObservationContext(
            contamination=ContaminationState.FOOD_GREASE_SOILED,
            infrastructure=InfrastructureContext.FULL_RECYCLING_FACILITY
        )),
        # Kịch bản 3: Cùng chai nhựa sạch nhưng bị đựng hóa chất độc hại -> Chuyển thành Rác nguy hại
        ("Plastic", 0.90, ObservationContext(
            contamination=ContaminationState.CHEMICAL_CONTAMINATED,
            infrastructure=InfrastructureContext.FULL_RECYCLING_FACILITY
        )),
        # Kịch bản 4: Pin cũ (dù bề ngoài sạch sẽ) -> Luôn khóa vào Rác nguy hại (Zero-tolerance)
        ("Metal_Battery", 0.95, ObservationContext(
            contamination=ContaminationState.CLEAN_INTACT,
            infrastructure=InfrastructureContext.FULL_RECYCLING_FACILITY
        )),
        # Kịch bản 5: Ảnh chụp mờ, độ tin cậy thấp (< 0.60) -> Kích hoạt cổng an toàn yêu cầu chụp lại
        ("Glass", 0.45, ObservationContext()),
    ]

    print("=" * 80)
    print("KIỂM THỬ ĐỘNG CƠ SUY DIỄN BẢN THỂ HỌC BFO (MILIVINTI, 2025)")
    print("=" * 80)
    for idx, (label, conf, ctx) in enumerate(test_scenarios, 1):
        res = engine.evaluate(label, conf, ctx)
        print(f"\n[Test {idx}] Input: Label='{label}' | Conf={conf:.2f} | Context={ctx.contamination.value}")
        print(f"  -> Thực thể BFO : {res.bfo_entity_iri}")
        print(f"  -> Vai trò gán  : {res.assigned_role.value}")
        print(f"  -> Thùng đích   : {res.target_bin_badge} ({res.target_bin_name})")
        print(f"  -> Logic Trace  : {res.fol_inference_trace}")