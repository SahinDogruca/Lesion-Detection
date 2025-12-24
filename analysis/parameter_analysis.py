import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ultralytics import YOLO
import cv2
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    f1_score,
)
import warnings

warnings.filterwarnings("ignore")

# Türkçe karakter desteği için
plt.rcParams["font.family"] = "DejaVu Sans"
sns.set_style("whitegrid")


class YOLOModelComparator:
    def __init__(self, base_path, output_dir="comparison_results"):
        self.base_path = Path(base_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Model klasörleri
        self.model_folders = ["adam1", "adam5", "sgd1", "sgd2", "sgd_5"]

        # Sınıf isimleri
        self.class_names = [
            "dentigeroz kist",
            "keratokist",
            "radikuler kist",
            "ameloblastoma",
            "odontoma",
        ]

        # Sonuçları saklamak için
        self.results = {}
        self.test_results = {}
        self.training_stability = {}

    def load_data_yaml(self):
        """data.yaml dosyasını yükle"""
        yaml_path = self.base_path / "data" / "data.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data

    def load_model_params(self, model_name):
        """Model parametrelerini args.yaml'dan yükle"""
        args_path = self.base_path / "params" / model_name / "args.yaml"
        with open(args_path, "r", encoding="utf-8") as f:
            args = yaml.safe_load(f)
        return args

    def load_training_results(self, model_name):
        """Eğitim sonuçlarını results.csv'den yükle"""
        csv_path = self.base_path / "params" / model_name / "results.csv"
        df = pd.read_csv(csv_path)
        # Sütun isimlerindeki boşlukları temizle
        df.columns = df.columns.str.strip()
        return df

    def analyze_training_stability(self, model_name):
        """Eğitim stabilitesini analiz et"""
        df = self.load_training_results(model_name)

        # Son 10 epoch'u al
        last_epochs = df.tail(10)

        stability_metrics = {}

        # Loss metrikleri (sütun isimlerini kontrol et)
        loss_cols = [col for col in df.columns if "loss" in col.lower()]

        for loss_col in loss_cols:
            if loss_col in df.columns:
                # Variance
                stability_metrics[f"{loss_col}_variance"] = last_epochs[loss_col].var()

                # Trend (son 10 epoch'ta düzenli azalma)
                trend = np.polyfit(range(len(last_epochs)), last_epochs[loss_col], 1)[0]
                stability_metrics[f"{loss_col}_trend"] = trend

                # Ani sıçrama sayısı
                diff = last_epochs[loss_col].diff().abs()
                threshold = diff.mean() + 2 * diff.std()
                jumps = (diff > threshold).sum()
                stability_metrics[f"{loss_col}_jumps"] = jumps

        # Train-Val fark (overfitting kontrolü)
        if "train/box_loss" in df.columns and "val/box_loss" in df.columns:
            train_val_diff = abs(
                df["train/box_loss"].iloc[-1] - df["val/box_loss"].iloc[-1]
            )
            stability_metrics["train_val_diff"] = train_val_diff

        return stability_metrics

    def evaluate_model_on_test(self, model_name):
        """Modeli test setinde değerlendir"""
        print(f"\n{'='*60}")
        print(f"📊 {model_name} modeli test ediliyor...")
        print(f"{'='*60}")

        # Model yolu
        model_path = self.base_path / "params" / model_name / "best.pt"

        # Data yaml yolu
        data_yaml = self.base_path / "data" / "data.yaml"

        # Modeli yükle
        model = YOLO(str(model_path))

        # Test seti üzerinde değerlendirme
        results = model.val(
            data=str(data_yaml), split="test", save_json=True, plots=True
        )

        # Metrikleri kaydet
        metrics = {
            "mAP50": results.box.map50,
            "mAP50-95": results.box.map,
            "precision": results.box.mp,
            "recall": results.box.mr,
            "f1": 2
            * (results.box.mp * results.box.mr)
            / (results.box.mp + results.box.mr + 1e-10),
        }

        # Sınıf bazlı metrikler
        class_metrics = {}
        if hasattr(results.box, "maps"):
            for i, class_name in enumerate(self.class_names):
                class_metrics[class_name] = {
                    "ap50": results.box.maps[i] if i < len(results.box.maps) else 0,
                    "precision": (
                        results.box.p[i]
                        if hasattr(results.box, "p") and i < len(results.box.p)
                        else 0
                    ),
                    "recall": (
                        results.box.r[i]
                        if hasattr(results.box, "r") and i < len(results.box.r)
                        else 0
                    ),
                }
                # F1 hesapla
                p = class_metrics[class_name]["precision"]
                r = class_metrics[class_name]["recall"]
                class_metrics[class_name]["f1"] = 2 * (p * r) / (p + r + 1e-10)

        metrics["class_metrics"] = class_metrics

        print(f"✓ mAP@50: {metrics['mAP50']:.4f}")
        print(f"✓ mAP@50-95: {metrics['mAP50-95']:.4f}")
        print(f"✓ Precision: {metrics['precision']:.4f}")
        print(f"✓ Recall: {metrics['recall']:.4f}")
        print(f"✓ F1-Score: {metrics['f1']:.4f}")

        return metrics

    def calculate_class_balance_score(self, class_metrics):
        """Sınıflar arası denge skorunu hesapla"""
        f1_scores = [m["f1"] for m in class_metrics.values()]

        # Standart sapma (düşük = dengeli)
        f1_std = np.std(f1_scores)

        # En düşük F1 skoru
        min_f1 = np.min(f1_scores)

        # Skor: yüksek = iyi
        balance_score = (1 - f1_std) * 0.5 + min_f1 * 0.5

        return balance_score, f1_std, min_f1

    def calculate_stability_score(self, stability_metrics):
        """Eğitim stabilite skorunu hesapla"""
        scores = []

        # Loss variance'ları topla
        variances = [v for k, v in stability_metrics.items() if "variance" in k]
        if variances:
            # Normalize et (düşük variance = yüksek skor)
            avg_variance = np.mean(variances)
            variance_score = 1 / (1 + avg_variance * 10)  # 0-1 aralığına normalize
            scores.append(variance_score)

        # Trend skorları (negatif trend = azalan loss = iyi)
        trends = [v for k, v in stability_metrics.items() if "trend" in k]
        if trends:
            avg_trend = np.mean(trends)
            trend_score = max(0, -avg_trend)  # Negatif trend'i pozitif skora çevir
            scores.append(min(1.0, trend_score))

        # Jump skorları (az sıçrama = iyi)
        jumps = [v for k, v in stability_metrics.items() if "jumps" in k]
        if jumps:
            avg_jumps = np.mean(jumps)
            jump_score = 1 / (1 + avg_jumps)
            scores.append(jump_score)

        return np.mean(scores) if scores else 0.5

    def calculate_final_score(self, model_name):
        """Final skoru hesapla"""
        test_metrics = self.test_results[model_name]
        stability = self.training_stability[model_name]

        # 1. Test Performansı (50%)
        test_perf = (
            test_metrics["mAP50"] * 0.40
            + test_metrics["mAP50-95"] * 0.30
            + test_metrics["f1"] * 0.20
            + test_metrics["recall"] * 0.10
        )

        # 2. Sınıf Dengesi (25%)
        balance_score, f1_std, min_f1 = self.calculate_class_balance_score(
            test_metrics["class_metrics"]
        )

        # 3. Eğitim Stabilitesi (15%)
        stability_score = self.calculate_stability_score(stability)

        # 4. Generalization (10%)
        # Train-val farkı
        gen_score = 1 - min(1.0, stability.get("train_val_diff", 0.5))

        # Final Skor
        final_score = (
            test_perf * 0.50
            + balance_score * 0.25
            + stability_score * 0.15
            + gen_score * 0.10
        )

        return {
            "final_score": final_score,
            "test_performance": test_perf,
            "class_balance": balance_score,
            "training_stability": stability_score,
            "generalization": gen_score,
            "f1_std": f1_std,
            "min_class_f1": min_f1,
        }

    def run_full_analysis(self):
        """Tüm modeller için tam analiz"""
        print("\n" + "=" * 80)
        print("🚀 YOLO Model Karşılaştırma ve Analiz Sistemi Başlatılıyor...")
        print("=" * 80)

        # Her model için değerlendirme
        for model_name in self.model_folders:
            print(f"\n📦 Model: {model_name}")

            # Test seti değerlendirmesi
            self.test_results[model_name] = self.evaluate_model_on_test(model_name)

            # Eğitim stabilite analizi
            self.training_stability[model_name] = self.analyze_training_stability(
                model_name
            )

            # Final skor hesaplama
            self.results[model_name] = self.calculate_final_score(model_name)

        print("\n" + "=" * 80)
        print("✅ Tüm modeller analiz edildi!")
        print("=" * 80)

        # Görselleştirmeleri oluştur
        self.create_visualizations()

        # Rapor oluştur
        self.create_report()

        return self.results

    def create_visualizations(self):
        """Tüm görselleştirmeleri oluştur"""
        print("\n📊 Görselleştirmeler oluşturuluyor...")

        # 1. Final Skor Karşılaştırması
        self.plot_final_scores()

        # 2. Metrik Karşılaştırma
        self.plot_metric_comparison()

        # 3. Sınıf Bazlı Performans
        self.plot_class_performance()

        # 4. Training Loss Karşılaştırması
        self.plot_training_losses()

        # 5. Radar Chart
        self.plot_radar_chart()

        print("✓ Tüm görselleştirmeler kaydedildi!")

    def plot_final_scores(self):
        """Final skorları görselleştir"""
        fig, ax = plt.subplots(figsize=(14, 8))

        models = list(self.results.keys())

        # Alt skorlar
        test_perf = [self.results[m]["test_performance"] for m in models]
        class_bal = [self.results[m]["class_balance"] for m in models]
        stability = [self.results[m]["training_stability"] for m in models]
        general = [self.results[m]["generalization"] for m in models]

        x = np.arange(len(models))
        width = 0.2

        ax.bar(
            x - 1.5 * width, test_perf, width, label="Test Performance (50%)", alpha=0.8
        )
        ax.bar(
            x - 0.5 * width, class_bal, width, label="Class Balance (25%)", alpha=0.8
        )
        ax.bar(
            x + 0.5 * width,
            stability,
            width,
            label="Training Stability (15%)",
            alpha=0.8,
        )
        ax.bar(x + 1.5 * width, general, width, label="Generalization (10%)", alpha=0.8)

        ax.set_xlabel("Models", fontsize=12, fontweight="bold")
        ax.set_ylabel("Score", fontsize=12, fontweight="bold")
        ax.set_title("Model Performance Breakdown", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "final_scores_breakdown.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # Final skor tablosu
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis("tight")
        ax.axis("off")

        final_scores = [self.results[m]["final_score"] for m in models]

        # Sıralama
        sorted_indices = np.argsort(final_scores)[::-1]
        sorted_models = [models[i] for i in sorted_indices]
        sorted_scores = [final_scores[i] for i in sorted_indices]

        table_data = []
        for i, (model, score) in enumerate(zip(sorted_models, sorted_scores)):
            rank = i + 1
            emoji = (
                "🥇"
                if rank == 1
                else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
            )
            table_data.append(
                [
                    f"{emoji} {model}",
                    f"{score:.4f}",
                    f"{self.results[model]['test_performance']:.4f}",
                    f"{self.results[model]['class_balance']:.4f}",
                    f"{self.results[model]['training_stability']:.4f}",
                    f"{self.results[model]['generalization']:.4f}",
                ]
            )

        table = ax.table(
            cellText=table_data,
            colLabels=[
                "Model",
                "Final Score",
                "Test Perf",
                "Class Bal",
                "Stability",
                "General",
            ],
            cellLoc="center",
            loc="center",
            colWidths=[0.2, 0.15, 0.15, 0.15, 0.15, 0.15],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Renklendirme
        for i in range(len(table_data)):
            for j in range(len(table_data[0])):
                if i == 0:  # En iyi model
                    table[(i + 1, j)].set_facecolor("#90EE90")

        plt.title("Final Model Rankings", fontsize=14, fontweight="bold", pad=20)
        plt.savefig(
            self.output_dir / "final_rankings_table.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def plot_metric_comparison(self):
        """Ana metrikleri karşılaştır"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        models = list(self.test_results.keys())
        metrics = ["mAP50", "mAP50-95", "precision", "recall", "f1"]

        for idx, metric in enumerate(metrics):
            values = [self.test_results[m][metric] for m in models]

            bars = axes[idx].bar(models, values, alpha=0.7, edgecolor="black")
            axes[idx].set_title(metric.upper(), fontsize=12, fontweight="bold")
            axes[idx].set_ylabel("Score", fontsize=10)
            axes[idx].grid(axis="y", alpha=0.3)
            axes[idx].set_ylim([0, 1])

            # En iyi değeri vurgula
            max_idx = np.argmax(values)
            bars[max_idx].set_color("gold")
            bars[max_idx].set_edgecolor("red")
            bars[max_idx].set_linewidth(2)

            # Değerleri göster
            for i, v in enumerate(values):
                axes[idx].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)

        axes[-1].axis("off")

        plt.suptitle(
            "Test Set Performance Metrics Comparison", fontsize=16, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig(
            self.output_dir / "metric_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def plot_class_performance(self):
        """Sınıf bazlı performans"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        models = list(self.test_results.keys())
        metrics_to_plot = ["precision", "recall", "f1", "ap50"]

        for idx, metric in enumerate(metrics_to_plot):
            data = []
            for class_name in self.class_names:
                class_values = [
                    self.test_results[m]["class_metrics"]
                    .get(class_name, {})
                    .get(metric, 0)
                    for m in models
                ]
                data.append(class_values)

            x = np.arange(len(self.class_names))
            width = 0.15

            for i, model in enumerate(models):
                offset = (i - len(models) / 2) * width
                values = [data[j][i] for j in range(len(self.class_names))]
                axes[idx].bar(x + offset, values, width, label=model, alpha=0.8)

            axes[idx].set_xlabel("Classes", fontsize=10)
            axes[idx].set_ylabel(metric.upper(), fontsize=10)
            axes[idx].set_title(
                f"Class-wise {metric.upper()}", fontsize=12, fontweight="bold"
            )
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(
                self.class_names, rotation=45, ha="right", fontsize=8
            )
            axes[idx].legend(fontsize=8)
            axes[idx].grid(axis="y", alpha=0.3)

        plt.suptitle(
            "Class-wise Performance Comparison", fontsize=16, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig(
            self.output_dir / "class_performance.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def plot_training_losses(self):
        """Training loss karşılaştırması"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()

        loss_types = [
            "train/box_loss",
            "val/box_loss",
            "train/cls_loss",
            "val/cls_loss",
        ]

        for idx, loss_type in enumerate(loss_types):
            for model in self.model_folders:
                df = self.load_training_results(model)
                if loss_type in df.columns:
                    axes[idx].plot(
                        df["epoch"], df[loss_type], label=model, linewidth=2, alpha=0.7
                    )

            axes[idx].set_xlabel("Epoch", fontsize=10)
            axes[idx].set_ylabel("Loss", fontsize=10)
            axes[idx].set_title(
                loss_type.replace("/", " ").upper(), fontsize=12, fontweight="bold"
            )
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)

        plt.suptitle("Training Loss Curves Comparison", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(
            self.output_dir / "training_losses.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def plot_radar_chart(self):
        """Radar chart ile genel karşılaştırma"""
        categories = [
            "Test Perf",
            "Class Balance",
            "Stability",
            "Generalization",
            "mAP@50",
        ]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        for model in self.model_folders:
            values = [
                self.results[model]["test_performance"],
                self.results[model]["class_balance"],
                self.results[model]["training_stability"],
                self.results[model]["generalization"],
                self.test_results[model]["mAP50"],
            ]
            values += values[:1]

            ax.plot(angles, values, "o-", linewidth=2, label=model)
            ax.fill(angles, values, alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        ax.set_title(
            "Model Comparison Radar Chart", fontsize=14, fontweight="bold", pad=20
        )
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "radar_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def create_report(self):
        """Markdown rapor oluştur"""
        report_path = self.output_dir / "analysis_report.md"

        # Modelleri sırala
        sorted_models = sorted(
            self.results.keys(),
            key=lambda x: self.results[x]["final_score"],
            reverse=True,
        )

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(
                "# 🏥 Panoramik Radyografi Lezyon Tespiti - Model Karşılaştırma Raporu\n\n"
            )
            f.write(
                f"**Analiz Tarihi:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            )
            f.write("---\n\n")

            # Executive Summary
            f.write("## 📋 Executive Summary\n\n")
            best_model = sorted_models[0]
            f.write(f"### 🥇 En İyi Model: **{best_model}**\n\n")
            f.write(
                f"- **Final Skor:** {self.results[best_model]['final_score']:.4f}\n"
            )
            f.write(f"- **mAP@50:** {self.test_results[best_model]['mAP50']:.4f}\n")
            f.write(
                f"- **mAP@50-95:** {self.test_results[best_model]['mAP50-95']:.4f}\n"
            )
            f.write(f"- **F1-Score:** {self.test_results[best_model]['f1']:.4f}\n")
            f.write(f"- **Recall:** {self.test_results[best_model]['recall']:.4f}\n\n")

            # Model Rankings
            f.write("## 🏆 Model Sıralaması\n\n")
            f.write(
                "| Sıra | Model | Final Skor | Test Perf | Class Balance | Stability | Generalization |\n"
            )
            f.write(
                "|------|-------|------------|-----------|---------------|-----------|----------------|\n"
            )

            for i, model in enumerate(sorted_models):
                rank_emoji = (
                    "🥇"
                    if i == 0
                    else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
                )
                f.write(
                    f"| {rank_emoji} | {model} | {self.results[model]['final_score']:.4f} | "
                    f"{self.results[model]['test_performance']:.4f} | "
                    f"{self.results[model]['class_balance']:.4f} | "
                    f"{self.results[model]['training_stability']:.4f} | "
                    f"{self.results[model]['generalization']:.4f} |\n"
                )

            f.write("\n---\n\n")

            # Detaylı Model Analizi
            f.write("## 📊 Detaylı Model Analizi\n\n")

            for model in sorted_models:
                f.write(f"### {model}\n\n")

                # Test metrikleri
                f.write("#### Test Set Performance\n\n")
                f.write("| Metrik | Değer |\n")
                f.write("|--------|-------|\n")
                for metric in ["mAP50", "mAP50-95", "precision", "recall", "f1"]:
                    f.write(f"| {metric} | {self.test_results[model][metric]:.4f} |\n")

                # Sınıf bazlı performans
                f.write("\n#### Sınıf Bazlı Performans\n\n")
                f.write("| Sınıf | Precision | Recall | F1-Score | AP@50 |\n")
                f.write("|-------|-----------|--------|----------|-------|\n")

                for class_name in self.class_names:
                    cm = self.test_results[model]["class_metrics"].get(class_name, {})
                    f.write(
                        f"| {class_name} | {cm.get('precision', 0):.4f} | "
                        f"{cm.get('recall', 0):.4f} | {cm.get('f1', 0):.4f} | "
                        f"{cm.get('ap50', 0):.4f} |\n"
                    )

                # Güçlü/Zayıf yönler
                f.write("\n#### 💪 Güçlü Yönler\n\n")
                class_f1s = {
                    k: v["f1"]
                    for k, v in self.test_results[model]["class_metrics"].items()
                }
                best_classes = sorted(
                    class_f1s.items(), key=lambda x: x[1], reverse=True
                )[:2]
                for cls, score in best_classes:
                    f.write(f"- **{cls}**: F1={score:.4f}\n")

                f.write("\n#### ⚠️ Geliştirme Alanları\n\n")
                worst_classes = sorted(class_f1s.items(), key=lambda x: x[1])[:2]
                for cls, score in worst_classes:
                    f.write(f"- **{cls}**: F1={score:.4f} (düşük performans)\n")

                f.write("\n---\n\n")

            # Klinik Öneriler
            f.write("## 🏥 Klinik Kullanım Önerileri\n\n")
            f.write(
                "1. **False Negative Kontrolü:** Recall değeri en yüksek model tercih edilmeli\n"
            )
            f.write(
                "2. **Nadir Lezyon Tespiti:** Ameloblastoma ve odontoma için özel dikkat\n"
            )
            f.write(
                "3. **Güvenilirlik:** Training stability skoru yüksek modeller daha tutarlı\n"
            )
            f.write(
                "4. **Balanced Performance:** Class balance skoru dengeli tespitler sağlar\n\n"
            )

            f.write("---\n\n")
            f.write(
                "*Rapor otomatik olarak YOLO Model Comparison System tarafından oluşturulmuştur.*\n"
            )

        print(f"✓ Rapor kaydedildi: {report_path}")


# Ana çalıştırma
if __name__ == "__main__":
    # Base path
    BASE_PATH = r"C:\Users\sahin\Desktop\yolo_params"

    # Comparator oluştur
    comparator = YOLOModelComparator(BASE_PATH)

    # Tam analiz
    results = comparator.run_full_analysis
