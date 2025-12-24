import os
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from PIL import Image


class KFoldEvaluator:
    def __init__(self, base_path, n_folds=5):
        self.base_path = Path(base_path)
        self.n_folds = n_folds
        self.folds = [f"fold_{i}" for i in range(1, n_folds + 1)]

    # ---------------------------------------------------------
    # VALIDATION METRICS
    # ---------------------------------------------------------
    def load_validation_metrics(self):
        val_metrics = []

        for fold in self.folds:
            results_path = self.base_path / fold / "runs" / "results.csv"
            if results_path.exists():
                df = pd.read_csv(results_path)
                last_metrics = df.iloc[-1].to_dict()
                last_metrics["fold"] = fold
                val_metrics.append(last_metrics)
                print(f"{fold} validation metrikleri yüklendi")
            else:
                print(f"Uyarı: {results_path} bulunamadı")

        return pd.DataFrame(val_metrics)

    # ---------------------------------------------------------
    # TEST METRICS (predictions.json)
    # ---------------------------------------------------------
    def load_test_metrics(self):
        test_results = []

        for fold in self.folds:
            pred_path = self.base_path / fold / "test" / "predictions.json"
            if pred_path.exists():
                with open(pred_path, "r") as f:
                    data = json.load(f)
                    test_results.append({"fold": fold, "test_data": data})
                print(f"{fold} test metrikleri yüklendi")
            else:
                print(f"Uyarı: {pred_path} bulunamadı")

        return test_results

    # ---------------------------------------------------------
    # PARSE TEST METRICS
    # ---------------------------------------------------------
    def parse_test_metrics(self, test_results):
        metrics_list = []

        for result in test_results:
            fold = result["fold"]
            data = result["test_data"]
            metrics = {"fold": fold}

            if isinstance(data, dict):
                if "metrics" in data:
                    metrics.update(data["metrics"])
                elif "results" in data:
                    metrics.update(data["results"])
                else:
                    # numeric key-value pairs
                    for k, v in data.items():
                        if isinstance(v, (int, float)):
                            metrics[k] = v

            metrics_list.append(metrics)

        return pd.DataFrame(metrics_list)

    # ---------------------------------------------------------
    # COMBINED ANALYSIS REPORT
    # ---------------------------------------------------------
    def create_combined_analysis_report(self, val_metrics, test_results, output_path):
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        folds = val_metrics["fold"].values
        x = np.arange(len(folds))

        # 1. VAL vs TEST mAP
        ax1 = axes[0, 0]
        if "metrics/mAP50-95(B)" in val_metrics.columns:
            val_map = val_metrics["metrics/mAP50-95(B)"].values
            ax1.bar(x - 0.25, val_map, width=0.5, label="Validation")

            test_df = self.parse_test_metrics(test_results)
            test_map_col = None
            for col in test_df.columns:
                if "map" in col.lower() and "50" in col.lower():
                    test_map_col = col
                    break

            if test_map_col:
                ax1.bar(x + 0.25, test_df[test_map_col], width=0.5, label="Test")

            ax1.set_xticks(x)
            ax1.set_xticklabels(folds)
            ax1.set_title("Validation vs Test mAP50-95")
            ax1.legend()
            ax1.grid(axis="y", alpha=0.3)

        # 2. Precision-Recall Scatter
        ax2 = axes[0, 1]
        if (
            "metrics/precision(B)" in val_metrics.columns
            and "metrics/recall(B)" in val_metrics.columns
        ):
            prec = val_metrics["metrics/precision(B)"].values
            rec = val_metrics["metrics/recall(B)"].values
            ax2.scatter(rec, prec, s=200, c=x, cmap="viridis")

            for i, fold in enumerate(folds):
                ax2.annotate(fold, (rec[i], prec[i]), fontsize=8)

            ax2.set_xlabel("Recall")
            ax2.set_ylabel("Precision")
            ax2.set_title("Validation PR Scatter")
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.grid(alpha=0.3)

        # 3. Loss comparison
        ax3 = axes[1, 0]
        loss_cols = [
            c for c in val_metrics.columns if "val/box_loss" in c or "val/seg_loss" in c
        ]
        if len(loss_cols) == 2:
            box_loss = val_metrics[loss_cols[0]].values
            seg_loss = val_metrics[loss_cols[1]].values
            ax3.bar(x - 0.25, box_loss, width=0.5, label="Box Loss")
            ax3.bar(x + 0.25, seg_loss, width=0.5, label="Seg Loss")
            ax3.set_xticks(x)
            ax3.set_xticklabels(folds)
            ax3.set_title("Validation Loss Comparison")
            ax3.legend()
            ax3.grid(axis="y", alpha=0.3)

        # 4. Stability (CV)
        ax4 = axes[1, 1]
        key_metrics = [
            "metrics/mAP50-95(B)",
            "metrics/precision(B)",
            "metrics/recall(B)",
        ]
        metric_names = ["mAP50-95", "Precision", "Recall"]

        cvs = []
        for m in key_metrics:
            if m in val_metrics.columns:
                values = val_metrics[m].values
                cv = (values.std() / values.mean()) * 100
                cvs.append(cv)
            else:
                cvs.append(0)

        ax4.barh(metric_names, cvs)
        for i, cv in enumerate(cvs):
            ax4.text(cv + 0.3, i, f"{cv:.2f}%", va="center")

        ax4.set_title("Model Stability (Coefficient of Variation)")
        ax4.set_xlabel("CV (%)")
        ax4.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / "combined_analysis_report.png", dpi=300)
        plt.close()

        cv_df = pd.DataFrame({"Metric": metric_names, "CV (%)": cvs})
        cv_df.to_csv(output_path / "model_stability.csv", index=False)

        return cv_df

    # ---------------------------------------------------------
    # TEST SUMMARY
    # ---------------------------------------------------------
    def create_test_summary(self, test_results, output_path):
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        test_df = self.parse_test_metrics(test_results)
        numeric_cols = [
            col
            for col in test_df.columns
            if col != "fold" and np.issubdtype(test_df[col].dtype, np.number)
        ]

        if not numeric_cols:
            print("Uyarı: Test metrikleri numeric değil.")
            return None

        summary = pd.DataFrame(
            {
                "Metric": numeric_cols,
                "Mean": [test_df[c].mean() for c in numeric_cols],
                "Std": [test_df[c].std() for c in numeric_cols],
                "Min": [test_df[c].min() for c in numeric_cols],
                "Max": [test_df[c].max() for c in numeric_cols],
            }
        )

        summary.to_csv(output_path / "test_summary.csv", index=False)
        print("Test özet tablosu oluşturuldu.")

        return summary

    # ---------------------------------------------------------
    # GROUND TRUTH LABEL YÜKLEME
    # ---------------------------------------------------------
    def load_ground_truth_labels(self, fold):
        labels_dir = self.base_path / fold / "test" / "labels"
        if not labels_dir.exists():
            print(f"GT Label klasörü yok: {labels_dir}")
            return {}

        gt_data = {}
        for file in labels_dir.glob("*.txt"):
            boxes = []
            with open(file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        boxes.append(cls)
            gt_data[file.stem] = boxes

        return gt_data

    # ---------------------------------------------------------
    # IoU HESABI
    # ---------------------------------------------------------
    def iou(self, box1, box2):
        # box: [x1, y1, x2, y2]
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])

        inter = max(0, xB - xA) * max(0, yB - yA)
        if inter == 0:
            return 0.0

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union

    # ---------------------------------------------------------
    # CONFUSION MATRIX
    # ---------------------------------------------------------
    def compute_confusion_matrix_from_predictions(
        self, predictions_path, fold, iou_thr=0.5
    ):
        with open(predictions_path, "r") as f:
            preds = json.load(f)

        gt_data = self.load_ground_truth_labels(fold)
        if not gt_data:
            print("GT bulunamadı, confusion matrix hesaplanamadı.")
            return None

        # Sınıf sayısını belirle
        all_classes = set()
        for boxes in gt_data.values():
            all_classes.update(boxes)

        for p in preds:
            if "category_id" in p:
                all_classes.add(p["category_id"])

        n_classes = max(all_classes) + 1

        cm = np.zeros((n_classes, n_classes), dtype=int)

        # Predictionları imaja göre grupla
        pred_by_image = defaultdict(list)
        for p in preds:
            img_id = p.get("image_id", p.get("file_name", "").split(".")[0])
            pred_by_image[img_id].append(p)

        for img, gt_classes in gt_data.items():
            pred_items = pred_by_image.get(img, [])

            pred_classes = [p["category_id"] for p in pred_items]

            # Simple class matching
            used_preds = set()
            for gt in gt_classes:
                matched = False
                for i, pc in enumerate(pred_classes):
                    if i not in used_preds and pc == gt:
                        used_preds.add(i)
                        cm[gt, pc] += 1
                        matched = True
                        break
                if not matched:
                    # FN
                    pass

            # False positives
            for i, pc in enumerate(pred_classes):
                if i not in used_preds:
                    pass

        return cm

    # ---------------------------------------------------------
    # CONFUSION MATRIX GÖRSELLEŞTİRME
    # ---------------------------------------------------------
    def plot_confusion_matrix(
        self, matrix, title, save_path, class_names=None, normalize=True
    ):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if matrix is None:
            print(f"Uyarı: {title} için matrix None")
            return

        mat = np.array(matrix, dtype=float)

        # Normalize (satır bazında = true labels)
        if normalize and mat.sum() > 0:
            row_sums = mat.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            mat_norm = mat / row_sums
        else:
            mat_norm = mat

        # class_names default
        n = mat.shape[0]
        if class_names is None:
            class_names = [f"class_{i}" for i in range(n)]

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        im0 = axes[0].imshow(mat, aspect="auto")
        axes[0].set_title(f"{title} (Raw counts)")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("True")
        axes[0].set_xticks(range(n))
        axes[0].set_yticks(range(n))
        axes[0].set_xticklabels(class_names, rotation=45, ha="right")
        axes[0].set_yticklabels(class_names)

        # text values
        for i in range(n):
            for j in range(n):
                val = int(mat[i, j])
                if val != 0:
                    axes[0].text(
                        j,
                        i,
                        f"{val}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="white" if val > mat.max() / 2 else "black",
                    )

        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        # normalized
        im1 = axes[1].imshow(mat_norm, vmin=0, vmax=1, aspect="auto")
        axes[1].set_title(f"{title} (Normalized by True Label)")
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("True")
        axes[1].set_xticks(range(n))
        axes[1].set_yticks(range(n))
        axes[1].set_xticklabels(class_names, rotation=45, ha="right")
        axes[1].set_yticklabels(class_names)

        for i in range(n):
            for j in range(n):
                val = mat_norm[i, j]
                if val > 0:
                    axes[1].text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="white" if val > 0.5 else "black",
                    )

        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # overall accuracy
        total = mat.sum()
        if total > 0:
            acc = np.trace(mat) / total
            fig.suptitle(
                f"{title} — Overall Accuracy: {acc:.2%}",
                fontsize=14,
                fontweight="bold",
                y=0.98,
            )

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Confusion matrix kaydedildi: {save_path}")

    # ---------------------------------------------------------
    # TÜM FOLD'LAR İÇİN CONFUSION MATRIX HESAPLA
    # ---------------------------------------------------------
    def load_confusion_matrices(self, cm_type="test"):
        matrices = []
        for fold in self.folds:
            if cm_type == "test":
                pred_path = self.base_path / fold / "test" / "predictions.json"
                if pred_path.exists():
                    print(f"  {fold} için confusion matrix hesaplanıyor...")
                    cm = self.compute_confusion_matrix_from_predictions(pred_path, fold)
                    if cm is not None:
                        matrices.append(cm)
                        print(f"    ✓ {fold}: shape {cm.shape}, total {cm.sum()}")
                else:
                    print(f"  ✗ {fold} - {pred_path} bulunamadı")
            else:
                # validation için predictions genellikle yok
                print(
                    f"  Not: {fold} validation confusion matrix hesaplanamıyor (predictions yok)"
                )
        return matrices

    # ---------------------------------------------------------
    # AVG & STD CONFUSION MATRIX
    # ---------------------------------------------------------
    def calculate_average_confusion_matrix(self, matrices):
        if not matrices:
            return None, None
        arrs = [np.array(m, dtype=float) for m in matrices]
        # make shapes equal by padding if needed
        max_r = max(a.shape[0] for a in arrs)
        max_c = max(a.shape[1] for a in arrs)
        padded = []
        for a in arrs:
            p = np.zeros((max_r, max_c), dtype=float)
            p[: a.shape[0], : a.shape[1]] = a
            padded.append(p)
        stacked = np.stack(padded, axis=0)
        avg = np.mean(stacked, axis=0)
        std = np.std(stacked, axis=0)
        return avg, std

    def create_confusion_matrix_average(
        self, cm_type="test", save_path=None, class_names=None
    ):
        print(f"\n{cm_type.capitalize()} confusion matrix'leri hesaplanıyor...")
        matrices = self.load_confusion_matrices(cm_type)
        if not matrices:
            print("  Uyarı: Hiç matrix bulunamadı.")
            return None, None

        avg, std = self.calculate_average_confusion_matrix(matrices)
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            self.plot_confusion_matrix(
                avg,
                f"Average {cm_type.capitalize()} Confusion Matrix",
                save_path,
                class_names=class_names,
                normalize=True,
            )

            std_path = save_path.parent / f"{save_path.stem}_std{save_path.suffix}"
            self.plot_confusion_matrix(
                std,
                f"{cm_type.capitalize()} Confusion Matrix Std Dev",
                std_path,
                class_names=class_names,
                normalize=False,
            )

            # per fold
            per_dir = save_path.parent / f"{cm_type}_confusion_per_fold"
            per_dir.mkdir(exist_ok=True)
            for i, m in enumerate(matrices):
                fold = self.folds[i] if i < len(self.folds) else f"fold_{i+1}"
                p = per_dir / f"{fold}_cm.png"
                self.plot_confusion_matrix(
                    m, f"{fold} {cm_type} CM", p, class_names=class_names
                )

        return avg, std

    # ---------------------------------------------------------
    # BASİT GRAFİK/FONKSİYONLAR (METRİK PLOTLARI)
    # ---------------------------------------------------------
    def plot_metrics_comparison(self, metrics_df, save_path):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # seçilecek önemli metrikler
        key_metrics = [
            c
            for c in metrics_df.columns
            if any(x in c.lower() for x in ["precision", "recall", "map", "f1"])
        ]
        if not key_metrics:
            print("Uyarı: Karşılaştırılacak metrik bulunamadı.")
            return

        n = len(key_metrics)
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
        axes = np.array(axes).reshape(-1)

        for i, metric in enumerate(key_metrics):
            ax = axes[i]
            # fold bazında değerler
            vals = []
            folds_present = []
            for fold in self.folds:
                row = metrics_df[metrics_df["fold"] == fold]
                if not row.empty and metric in row.columns:
                    vals.append(float(row.iloc[0][metric]))
                    folds_present.append(fold)
            if not vals:
                ax.text(0.5, 0.5, "No data", ha="center")
            else:
                ax.boxplot([vals], labels=["folds"])
                ax.scatter([1] * len(vals), vals, alpha=0.7)
                for j, v in enumerate(vals):
                    ax.text(1.05, v, folds_present[j], fontsize=8)
                ax.set_title(metric)
                ax.grid(alpha=0.3)

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Metrik karşılaştırma grafiği kaydedildi: {save_path}")

    def plot_fold_comparison_bar(self, metrics_df, save_path):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        key_metrics = [
            c
            for c in metrics_df.columns
            if any(x in c.lower() for x in ["map50-95", "map50", "precision", "recall"])
        ]
        if not key_metrics:
            return
        key_metrics = key_metrics[:4]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for i, metric in enumerate(key_metrics):
            ax = axes[i]
            vals = []
            folds_present = []
            for fold in self.folds:
                row = metrics_df[metrics_df["fold"] == fold]
                if not row.empty and metric in row.columns:
                    vals.append(float(row.iloc[0][metric]))
                    folds_present.append(fold)

            x = np.arange(len(vals))
            bars = ax.bar(x, vals, alpha=0.7)
            meanv = np.mean(vals) if vals else 0
            ax.axhline(meanv, color="r", linestyle="--", label=f"Mean {meanv:.4f}")
            for xi, v in zip(x, vals):
                ax.text(xi, v + 0.005, f"{v:.4f}", ha="center", fontsize=9)
            ax.set_xticks(x)
            ax.set_xticklabels(folds_present, rotation=45)
            ax.set_title(metric)
            ax.legend()
            ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Fold karşılaştırma grafiği kaydedildi: {save_path}")

    def create_summary_table(self, metrics_df, save_path):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        numeric_cols = [
            c
            for c in metrics_df.columns
            if c not in ["fold", "epoch"]
            and np.issubdtype(metrics_df[c].dtype, np.number)
        ]
        summary = pd.DataFrame(
            {
                "Metric": numeric_cols,
                "Mean": [metrics_df[c].mean() for c in numeric_cols],
                "Std": [metrics_df[c].std() for c in numeric_cols],
                "Min": [metrics_df[c].min() for c in numeric_cols],
                "Max": [metrics_df[c].max() for c in numeric_cols],
            }
        )
        summary.to_csv(save_path, index=False)
        print(f"✓ Özet tablo kaydedildi: {save_path}")
        return summary

    def plot_learning_curves(self, save_path):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # örnek fold'tan kolonları al
        sample = None
        for fold in self.folds:
            p = self.base_path / fold / "runs" / "results.csv"
            if p.exists():
                sample = pd.read_csv(p)
                break
        if sample is None:
            print("Uyarı: Öğrenme eğrileri için results.csv bulunamadı.")
            return

        # hangi metrikler çizilecek
        metric_configs = [
            ("Val Box Loss", ["val/box_loss"]),
            ("Val Seg Loss", ["val/seg_loss"]),
            ("Val Precision (Box)", ["metrics/precision(B)"]),
            ("Val Recall (Box)", ["metrics/recall(B)"]),
            ("Val mAP50 (Box)", ["metrics/mAP50(B)"]),
            ("Val mAP50-95 (Box)", ["metrics/mAP50-95(B)"]),
        ]

        n = len(metric_configs)
        cols = 3
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
        axes = axes.flatten()

        for idx, (title, possible_cols) in enumerate(metric_configs):
            ax = axes[idx]
            plotted = False
            for fold in self.folds:
                p = self.base_path / fold / "runs" / "results.csv"
                if p.exists():
                    df = pd.read_csv(p)
                    for col in possible_cols:
                        if col in df.columns:
                            ax.plot(df.index, df[col], label=fold, alpha=0.8)
                            plotted = True
                            break
            if plotted:
                ax.set_title(title)
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Value")
                ax.legend(fontsize=8)
                ax.grid(alpha=0.3)
            else:
                ax.text(0.5, 0.5, f"No metric: {possible_cols[0]}", ha="center")
                ax.set_title(title)

        for j in range(idx + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Learning curves kaydedildi: {save_path}")

    # ---------------------------------------------------------
    # ORTALAMA METRİK HESAPLA (SEÇİCİ)
    # ---------------------------------------------------------
    def calculate_average_metrics(self, metrics_df):
        numeric = metrics_df.select_dtypes(include=[np.number]).columns
        avg = {}
        for c in numeric:
            if c != "epoch":
                avg[c] = {
                    "mean": float(metrics_df[c].mean()),
                    "std": float(metrics_df[c].std()),
                    "min": float(metrics_df[c].min()),
                    "max": float(metrics_df[c].max()),
                }
        return avg

    # ---------------------------------------------------------
    # TAM PIPELINE
    # ---------------------------------------------------------
    def run_full_evaluation(self, output_dir="kfold_analysis", class_names=None):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print("K-FOLD EVALUATION START")
        print("=" * 60)

        # 1. Validation metrics
        print("\n1) Validation metrikleri yükleniyor...")
        val_metrics = self.load_validation_metrics()
        if not val_metrics.empty:
            print("  - Özet tablo oluşturuluyor...")
            summary = self.create_summary_table(
                val_metrics, output_path / "validation_summary.csv"
            )
            print(summary.to_string(index=False))

            print("  - Metrik görselleştirmeleri...")
            self.plot_metrics_comparison(
                val_metrics, output_path / "validation_metrics_boxplot.png"
            )
            self.plot_fold_comparison_bar(
                val_metrics, output_path / "validation_fold_comparison.png"
            )
            print("  - Öğrenme eğrileri...")
            self.plot_learning_curves(output_path / "learning_curves_all_folds.png")
        else:
            print("  Uyarı: Validation metrikleri boş.")

        # 2. Test metrics
        print("\n2) Test metrikleri yükleniyor...")
        test_results = self.load_test_metrics()
        if test_results:
            print("  - Test özet tablosu oluşturuluyor...")
            test_summary = self.create_test_summary(test_results, output_path)
            if test_summary is not None:
                print(test_summary.to_string(index=False))
        else:
            print("  Uyarı: Test results bulunamadı.")

        # 3) Confusion matrices
        print("\n3) Confusion matrix analizi...")
        avg_val = None
        if True:  # validation için genelde predictions yok
            pass

        avg_test_cm, std_test_cm = self.create_confusion_matrix_average(
            "test",
            save_path=output_path / "average_test_confusion_matrix.png",
            class_names=class_names,
        )

        # 4) Combined analysis (val vs test)
        print("\n4) Birleşik analiz raporu...")
        if (not val_metrics.empty) and test_results:
            cv_df = self.create_combined_analysis_report(
                val_metrics, test_results, output_path
            )
            print(cv_df.to_string(index=False))
        else:
            print("  Birleşik rapor oluşturmak için yeterli veri yok.")

        print("\n" + "=" * 60)
        print("EVALUATION FINISHED. Results saved to", output_path.resolve())
        print("=" * 60)

        return val_metrics, test_results


# ---------------------------------------------------------
# ANA
# ---------------------------------------------------------
if __name__ == "__main__":
    # örnek base_path - bunu kendi yoluna göre değiştir
    base_path = r"C:\Users\sahin\Desktop\k_fold\folds"
    evaluator = KFoldEvaluator(base_path=base_path, n_folds=5)

    # opsiyonel: class names eklersen confusion görseller daha anlamlı olur
    class_names = None  # örn: ["background","person","car",...]

    val_metrics, test_results = evaluator.run_full_evaluation(
        output_dir="kfold_analysis", class_names=class_names
    )
