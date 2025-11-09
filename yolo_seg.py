"""
Dental X-Ray Segmentation with YOLOv8
Panoramik dental röntgen görüntülerinde diş segmentasyonu
"""

from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import yaml
from collections import Counter

# ==================== VERİ SETİ ANALİZİ ====================


def analyze_dataset(dataset_path):
    """Veri setini analiz et - class dağılımı, görüntü boyutları vb."""

    dataset_path = Path(dataset_path)
    stats = {
        "train": {"count": 0, "classes": Counter()},
        "valid": {"count": 0, "classes": Counter()},
        "test": {"count": 0, "classes": Counter()},
    }

    for split in ["train", "valid", "test"]:
        labels_dir = dataset_path / split / "labels"
        if not labels_dir.exists():
            continue

        label_files = list(labels_dir.glob("*.txt"))
        stats[split]["count"] = len(label_files)

        for label_file in label_files:
            with open(label_file, "r") as f:
                for line in f:
                    if line.strip():
                        class_id = int(line.split()[0])
                        stats[split]["classes"][class_id] += 1

    print("=" * 50)
    print("VERİ SETİ ANALİZİ")
    print("=" * 50)
    for split, data in stats.items():
        print(f"\n{split.upper()}:")
        print(f"  Görüntü sayısı: {data['count']}")
        print(f"  Class dağılımı: {dict(data['classes'])}")
        total_instances = sum(data["classes"].values())
        if total_instances > 0:
            print(f"  Toplam instance: {total_instances}")
            for cls, count in sorted(data["classes"].items()):
                pct = (count / total_instances) * 100
                print(f"    Class {cls}: {count} ({pct:.1f}%)")

    return stats


def check_image_sizes(dataset_path, n_samples=10):
    """Görüntü boyutlarını kontrol et"""
    dataset_path = Path(dataset_path)
    sizes = []

    for split in ["train", "valid", "test"]:
        img_dir = dataset_path / split / "images"
        if not img_dir.exists():
            continue

        for img_file in list(img_dir.glob("*"))[:n_samples]:
            img = cv2.imread(str(img_file))
            if img is not None:
                sizes.append(img.shape[:2])

    if sizes:
        print(f"\n{'=' * 50}")
        print("GÖRÜNTÜ BOYUTLARI")
        print(f"{'=' * 50}")
        heights, widths = zip(*sizes)
        print(f"Örnek boyutlar: {sizes[:5]}")
        print(f"Ortalama: {int(np.mean(heights))} x {int(np.mean(widths))}")
        print(f"Min: {min(heights)} x {min(widths)}")
        print(f"Max: {max(heights)} x {max(widths)}")

        # Önerilen imgsz
        max_dim = max(max(heights), max(widths))
        recommended_imgsz = 2 ** int(np.ceil(np.log2(max_dim * 0.5)))
        print(f"Önerilen imgsz: {recommended_imgsz}")


# ==================== MASK KONTROLÜ ====================


def check_masks(dataset_path, n_samples=5):
    """Mask dosyalarını kontrol et"""
    dataset_path = Path(dataset_path)

    print(f"\n{'=' * 50}")
    print("MASK DOSYALARI KONTROLÜ")
    print(f"{'=' * 50}")

    for split in ["train"]:  # sadece train'i kontrol et
        mask_dir = dataset_path / split / "masks"
        if not mask_dir.exists():
            print(f"{split}/masks dizini bulunamadı")
            continue

        mask_files = list(mask_dir.glob("*"))[:n_samples]

        for mask_file in mask_files:
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                unique_vals = np.unique(mask)
                print(f"  {mask_file.name}:")
                print(f"    Shape: {mask.shape}")
                print(f"    Unique values: {unique_vals}")
                print(f"    Non-zero pixels: {np.count_nonzero(mask)}")

    print("\nNot: Maskler tamamen siyahsa (0), labels dosyası yeterlidir.")


# ==================== MODEL EĞİTİMİ ====================


def train_model(data_yaml_path, model_size="n", epochs=100, imgsz=1280, batch=8):
    """
    YOLO segmentation modelini eğit

    Args:
        data_yaml_path: data.yaml dosyasının yolu
        model_size: 'n', 's', 'm', 'l', 'x' (nano, small, medium, large, xlarge)
        epochs: eğitim epoch sayısı
        imgsz: görüntü boyutu (dental için 1280-1600 önerilir)
        batch: batch size
    """

    # Model yükle
    model = YOLO(f"yolov8{model_size}-seg.pt")

    print(f"\n{'=' * 50}")
    print(f"MODEL EĞİTİMİ BAŞLIYOR - YOLOv8{model_size.upper()}-SEG")
    print(f"{'=' * 50}\n")

    # Eğitim
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        # Optimization
        patience=20,
        save=True,
        save_period=10,
        # Augmentation (dental X-ray için uygun)
        degrees=10.0,  # hafif rotasyon
        translate=0.1,  # hafif kaydırma
        scale=0.2,  # hafif ölçekleme
        flipud=0.0,  # dental'de üst-alt çevirme YAPMA
        fliplr=0.5,  # sol-sağ çevirme OK
        mosaic=0.5,  # mosaic augmentation
        # Project settings
        project="dental_segmentation",
        name=f"tooth_seg_yolov8{model_size}",
        exist_ok=True,
        # Advanced
        amp=True,  # mixed precision training
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.0005,
        warmup_epochs=3,
        # Monitoring
        plots=True,
        verbose=True,
        save_dir=f"dental_segmentation/tooth_seg_yolov8{model_size}",
    )

    print(f"\n{'=' * 50}")
    print("EĞİTİM TAMAMLANDI!")
    print(f"{'=' * 50}")
    print(f"En iyi model: {results.save_dir}/weights/best.pt")

    return model, results


# ==================== DEĞERLENDİRME ====================


def validate_model(model_path, data_yaml_path):
    """Modeli değerlendir"""

    model = YOLO(model_path)

    print(f"\n{'=' * 50}")
    print("MODEL DEĞERLENDİRME")
    print(f"{'=' * 50}\n")

    metrics = model.val(
        data=data_yaml_path, split="test", save_json=True, plots=True  # veya 'val'
    )

    print(f"\nmAP50: {metrics.seg.map50:.3f}")
    print(f"mAP50-95: {metrics.seg.map:.3f}")

    return metrics


# ==================== INFERENCE ====================


def predict_and_visualize(model_path, image_path, save_dir="predictions"):
    """Tek görüntü üzerinde tahmin yap ve görselleştir"""

    model = YOLO(model_path)

    # Tahmin
    results = model.predict(
        source=image_path,
        save=True,
        save_txt=True,
        save_conf=True,
        conf=0.25,
        iou=0.7,
        project=save_dir,
        name="results",
        exist_ok=True,
        line_width=2,
        show_labels=True,
        show_conf=True,
    )

    # Sonuçları yazdır
    for r in results:
        print(f"\nGörüntü: {r.path}")
        print(f"Tespit edilen instance sayısı: {len(r.masks) if r.masks else 0}")
        if r.masks:
            for i, (cls, conf) in enumerate(zip(r.boxes.cls, r.boxes.conf)):
                print(f"  Instance {i+1}: Class {int(cls)}, Confidence: {conf:.2f}")

    return results


# ==================== BATCH INFERENCE ====================


def batch_predict(model_path, images_dir, save_dir="predictions"):
    """Tüm test görüntüleri üzerinde tahmin"""

    model = YOLO(model_path)

    results = model.predict(
        source=images_dir,
        save=True,
        save_txt=True,
        conf=0.25,
        project=save_dir,
        name="batch_results",
        exist_ok=True,
        stream=True,  # memory efficient
    )

    # Stream results
    for r in results:
        pass  # process as needed

    print(f"\nTahminler kaydedildi: {save_dir}/batch_results")


# ==================== KULLANIM ÖRNEĞİ ====================

if __name__ == "__main__":

    # Veri seti yolu
    DATASET_PATH = "/kaggle/input/data-v2/data_v2"

    DATA_YAML = "data.yaml"

    # 1. VERİ SETİ ANALİZİ
    print("1. Veri seti analiz ediliyor...")
    stats = analyze_dataset(DATASET_PATH)
    check_image_sizes(DATASET_PATH)

    # 2. MODEL EĞİTİMİ
    print("\n2. Model eğitiliyor...")
    model, results = train_model(
        data_yaml_path=DATA_YAML,
        model_size="l",  # başlangıç için nano
        epochs=100,
        imgsz=1024,  # dental için uygun boyut
        batch=4,  # GPU'nuza göre ayarlayın
    )

    # 3. DEĞERLENDİRME
    print("\n3. Model değerlendiriliyor...")
    best_model_path = "dental_segmentation/tooth_seg_yolov8l/weights/best.pt"
    metrics = validate_model(best_model_path, DATA_YAML)

    # 4. ÖRNEK TAHMİN
    print("\n4. Örnek tahmin yapılıyor...")
    test_image = f"{DATASET_PATH}/test/images/A84.png"
    results = predict_and_visualize(best_model_path, test_image)

    print("\n" + "=" * 50)
    print("TÜM İŞLEMLER TAMAMLANDI!")
    print("=" * 50)
