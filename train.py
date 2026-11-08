"""
File: train.py
Mục đích: Giao diện dòng lệnh (CLI) chuẩn hóa quy trình huấn luyện (Training) và kiểm định (Validation/Testing)
          mô hình YOLOv12n trên 6 Thực thể Vật chất BFO theo bộ siêu tham số của Dipo et al. (2025) - Table 2.
Môi trường chạy:
  - Huấn luyện (Yêu cầu GPU CUDA / Colab):
      python train.py --mode train --data data/processed/data.yaml --epochs 100 --batch 16
  - Kiểm định trọng số best.pt sẵn có:
      python train.py --mode val --weights models/best.pt --data data/processed/data.yaml --split test
"""

from __future__ import annotations
import argparse
import shutil
from pathlib import Path
import torch
from ultralytics import YOLO


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CLI Pipeline huấn luyện và kiểm định YOLOv12 cho hệ thống phân loại rác BFO (HUST)."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "val"],
        default="train",
        help="Chế độ thực thi: 'train' (huấn luyện mới) hoặc 'val' (kiểm định trọng số best.pt).",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/data.yaml",
        help="Đường dẫn tới file cấu hình data.yaml (6 lớp BFO Material Entities).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolo12n.pt",
        help="Đường dẫn trọng số khởi tạo (yolo12n.pt khi train, hoặc models/best.pt khi val).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Số vòng lặp huấn luyện (Table 2 - Dipo et al., 2025: 100 epochs).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Kích thước lô dữ liệu (Table 2 - Dipo et al., 2025: Batch size = 16).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Kích thước ma trận ảnh chuẩn hóa đầu vào (Section 3.3: 640x640).",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.01,
        help="Tốc độ học ban đầu của bộ tối ưu hóa SGD (Table 2: 0.01).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0005,
        help="Hệ số phạt chuẩn hóa L2 chống Overfitting (Table 2: 0.0005).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Ngưỡng dừng sớm Early Stopping (Section 4.4: 50 epochs).",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="test",
        help="Tập dữ liệu dùng để chấm điểm khi chạy chế độ --mode val.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Thư mục gốc lưu trữ kết quả thực thi.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="yolov12n_bfo_6classes",
        help="Tên phiên thực nghiệm.",
    )
    return parser.parse_args()


def run_training(args: argparse.Namespace, device: int | str) -> None:
    """Thực thi huấn luyện YOLOv12 theo đúng cấu hình bài báo Dipo et al. (2025)."""
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"[LỖI] Không tìm thấy '{data_path}'. "
            f"Hãy chạy 'python scripts/remap_labels.py' để chuẩn hóa dữ liệu 6 lớp trước!"
        )

    print("=" * 72)
    print(f"[TRAIN MODE] Bắt đầu huấn luyện YOLOv12 trên thiết bị: {device}")
    print(f"  - Dataset Config : {data_path.resolve()}")
    print(f"  - Base Weights   : {args.weights}")
    print(f"  - Hyperparams    : epochs={args.epochs}, batch={args.batch}, imgsz={args.imgsz}, SGD(lr0={args.lr0})")
    print("=" * 72)

    model = YOLO(args.weights)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        optimizer="SGD",
        lr0=args.lr0,
        weight_decay=args.weight_decay,
        patience=args.patience,
        degrees=15.0,       # Augmentation: Xoay ngẫu nhiên [-15, +15] độ (Section 3.2)
        hsv_v=0.25,         # Augmentation: Biến thiên độ sáng +-25% (Section 3.2)
        fliplr=0.5,         # Augmentation: Lật ngang đối xứng 50% (Section 3.2)
        scale=0.5,          # Augmentation: Co giãn tỷ lệ vật thể (Section 3.2)
        device=device,
        workers=2,
        project=args.project,
        name=args.name,
        exist_ok=True,
        pretrained=True,
        plots=True,
        save=True,
    )

    # Tự động đồng bộ trọng số best.pt và biểu đồ đánh giá vào thư mục models/
    run_dir = Path(args.project) / args.name
    best_ckpt = run_dir / "weights" / "best.pt"
    models_dir = Path("models")
    metrics_dir = models_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if best_ckpt.exists():
        shutil.copy2(best_ckpt, models_dir / "best.pt")
        print(f"\n[OK] Đã xuất trọng số tốt nhất vào: {models_dir / 'best.pt'}")

    for artifact in [
        "results.png",
        "results.csv",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "labels.jpg",
    ]:
        src = run_dir / artifact
        if src.exists():
            shutil.copy2(src, metrics_dir / artifact)
    print(f"[OK] Đã đồng bộ bộ biểu đồ thực nghiệm vào: {metrics_dir}/")


def run_validation(args: argparse.Namespace, device: int | str) -> None:
    """Chấm điểm độc lập mô hình đã huấn luyện trên tập valid hoặc test."""
    weights_path = Path(args.weights)
    data_path = Path(args.data)

    if not weights_path.exists():
        raise FileNotFoundError(f"[LỖI] Không tìm thấy file trọng số tại: {weights_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"[LỖI] Không tìm thấy file cấu hình dataset tại: {data_path}")

    print("=" * 72)
    print(f"[VAL MODE] Đang kiểm định mô hình '{weights_path}' trên tập '{args.split}'...")
    print("=" * 72)

    model = YOLO(str(weights_path))
    metrics = model.val(
        data=str(data_path),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        plots=True,
    )

    mp = float(metrics.box.mp)
    mr = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    f1_score = (2.0 * mp * mr) / (mp + mr + 1e-8)

    print("\n" + "=" * 72)
    print("BÁO CÁO KẾT QUẢ KIỂM ĐỊNH MÔ HÌNH YOLOv12 (6 LỚP THỰC THỂ BFO)")
    print("=" * 72)
    print(f"  - Mean Precision (P) : {mp:.4f}  (Benchmark Dipo et al., 2025 : 0.7300)")
    print(f"  - Mean Recall (R)    : {mr:.4f}  (Benchmark Dipo et al., 2025 : 0.7700)")
    print(f"  - F1-Score           : {f1_score:.4f}  (Mục tiêu PEAS               >= 0.7500)")
    print(f"  - mAP@0.5            : {map50:.4f}  (Mục tiêu PEAS               >= 0.7800)")
    print(f"  - mAP@0.5:0.95       : {map50_95:.4f}")
    print("=" * 72)


def main() -> None:
    args = parse_arguments()
    device: int | str = 0 if torch.cuda.is_available() else "cpu"

    if args.mode == "train":
        run_training(args, device)
    else:
        run_validation(args, device)


if __name__ == "__main__":
    main()