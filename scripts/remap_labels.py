"""
File: scripts/remap_labels.py
Mục đích: 
  1. Quy đổi 16 nhãn phân mảnh từ Roboflow về 6 lớp Thực thể Vật chất (BFO Material Entity)[cite: 16, 32].
  2. Chuẩn hóa đồng nhất mọi nhãn đa giác (Polygon > 4 tọa độ) về đúng chuẩn Bounding Box 4 tọa độ (xc, yc, w, h).
"""

import shutil
import yaml
from pathlib import Path
from collections import Counter

# 1. Định nghĩa 6 lớp Thực thể Vật chất đích (Target BFO Material Entities)
TARGET_CLASSES = {
    0: "Plastic",
    1: "Metal_Battery",
    2: "Paper",
    3: "Glass",
    4: "Organic",
    5: "Medical"
}

# 2. Bảng ánh xạ O(1) từ Old Class ID (0..15) -> New Class ID (0..5)[cite: 16, 32]
OLD_TO_NEW_ID_MAP = {
    0: 1,   # 'Battery'              -> 1: Metal_Battery
    1: 4,   # 'BiologicalWaste'      -> 4: Organic
    2: 2,   # 'Cardboard Paper'      -> 2: Paper
    3: 3,   # 'Glass'                -> 3: Glass
    4: 3,   # 'Glass-Bottle'         -> 3: Glass
    5: 5,   # 'Glove'                -> 5: Medical
    6: 0,   # 'HDPE Plastic'         -> 0: Plastic
    7: 5,   # 'Mask'                 -> 5: Medical
    8: 5,   # 'Medicine'             -> 5: Medical
    9: 2,   # 'Newspaper'            -> 2: Paper
    10: 0,  # 'PET Bottle'           -> 0: Plastic
    11: 2,  # 'Paper'                -> 2: Paper
    12: 0,  # 'Single-Use-Plastic'   -> 0: Plastic
    13: 0,  # 'Single-layer Plastic' -> 0: Plastic
    14: 0,  # 'Squeeze Tube'         -> 0: Plastic
    15: 5   # 'syringe'              -> 5: Medical
}


def convert_to_yolo_bbox(coords: list[float]) -> tuple[float, float, float, float] | None:
    """
    Chuyển đổi mảng tọa độ bất kỳ (BBox 4 số hoặc Polygon 2N số) về chuẩn (x_center, y_center, width, height).
    Trả về None nếu tọa độ bị suy biến (chiều rộng hoặc chiều cao <= 0).
    """
    # Kẹp giá trị (Clamp) vào đoạn [0.0, 1.0] để tránh lỗi tràn biên do làm tròn
    coords = [max(0.0, min(1.0, c)) for c in coords]

    if len(coords) == 4:
        xc, yc, w, h = coords
    elif len(coords) >= 6 and len(coords) % 2 == 0:
        xs = coords[0::2]
        ys = coords[1::2]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        xc = (x_min + x_max) / 2.0
        yc = (y_min + y_max) / 2.0
        w = x_max - x_min
        h = y_max - y_min
    else:
        return None

    # Loại bỏ các hộp rỗng (diện tích = 0)
    if w <= 1e-5 or h <= 1e-5:
        return None

    return xc, yc, w, h


def remap_dataset(raw_root: Path, processed_root: Path) -> None:
    if processed_root.exists():
        shutil.rmtree(processed_root)
    processed_root.mkdir(parents=True, exist_ok=True)

    new_class_counts = Counter()
    polygon_converted_count = 0
    corrupted_lines = 0
    total_files_processed = 0

    for split in ["train", "valid", "test"]:
        raw_split_dir = raw_root / split
        if not raw_split_dir.exists():
            continue

        raw_img_dir = raw_split_dir / "images"
        raw_lbl_dir = raw_split_dir / "labels"

        proc_img_dir = processed_root / split / "images"
        proc_lbl_dir = processed_root / split / "labels"
        proc_img_dir.mkdir(parents=True, exist_ok=True)
        proc_lbl_dir.mkdir(parents=True, exist_ok=True)

        # 1. Sao chép ảnh
        for img_path in raw_img_dir.glob("*.*"):
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                shutil.copy2(img_path, proc_img_dir / img_path.name)

        # 2. Xử lý file nhãn .txt
        for lbl_path in raw_lbl_dir.glob("*.txt"):
            if lbl_path.name == "classes.txt":
                continue

            new_lines = []
            with open(lbl_path, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        corrupted_lines += 1
                        continue

                    try:
                        old_id = int(parts[0])
                        raw_coords = [float(x) for x in parts[1:]]
                    except ValueError:
                        corrupted_lines += 1
                        continue

                    if old_id not in OLD_TO_NEW_ID_MAP:
                        corrupted_lines += 1
                        continue

                    if len(raw_coords) > 4:
                        polygon_converted_count += 1

                    bbox = convert_to_yolo_bbox(raw_coords)
                    if bbox is None:
                        corrupted_lines += 1
                        continue

                    new_id = OLD_TO_NEW_ID_MAP[old_id]
                    new_class_counts[new_id] += 1
                    xc, yc, w, h = bbox
                    new_lines.append(f"{new_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

            out_lbl_path = proc_lbl_dir / lbl_path.name
            with open(out_lbl_path, "w", encoding="utf-8") as f_out:
                if new_lines:
                    f_out.write("\n".join(new_lines) + "\n")

            total_files_processed += 1

    # 3. Sinh file data.yaml chuẩn
    yaml_data = {
        "path": str(processed_root.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(TARGET_CLASSES),
        "names": [TARGET_CLASSES[i] for i in sorted(TARGET_CLASSES.keys())]
    }

    yaml_out_path = processed_root / "data.yaml"
    with open(yaml_out_path, "w", encoding="utf-8") as yf:
        yaml.dump(yaml_data, yf, sort_keys=False, allow_unicode=True)

    print("=================================================================")
    print("BÁO CÁO CHUẨN HÓA DỮ LIỆU & HÌNH HỌC BBOX (6 LỚP BFO)")
    print("=================================================================")
    print(f"- Tổng số file nhãn (.txt) đã xử lý       : {total_files_processed}")
    print(f"- Số nhãn Polygon đã ép về BBox 4 tọa độ  : {polygon_converted_count}")
    print(f"- Số dòng nhãn lỗi/suy biến đã lọc bỏ     : {corrupted_lines}")
    print("-----------------------------------------------------------------")
    total_valid = sum(new_class_counts.values())
    for new_id in sorted(TARGET_CLASSES.keys()):
        cname = TARGET_CLASSES[new_id]
        cnt = new_class_counts[new_id]
        pct = (cnt / total_valid * 100) if total_valid > 0 else 0.0
        print(f"  Class [{new_id}] {cname:<15} : {cnt:6d} mẫu ({pct:5.2f}%)")
    print(f"Tổng cộng: {total_valid} bounding boxes chuẩn 5 cột.")


if __name__ == "__main__":
    remap_dataset(Path("/content/data/raw"), Path("/content/data/processed"))
