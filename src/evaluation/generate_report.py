import os
import json
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, roc_curve, auc
import joblib

logger = logging.getLogger(__name__)

class ReportGenerator:
    """
    Generates model evaluation plots (Confusion Matrix, ROC Curve, Feature Importance)
    and writes markdown evaluation summaries under reports/model_evaluation/.
    """

    @staticmethod
    def generate_evaluation_artifacts(
        y_true: np.ndarray,
        y_preds: np.ndarray,
        y_probs: np.ndarray,
        feature_names: list,
        model_name: str = "visual_classifier",
        output_dir: str = "reports/model_evaluation"
    ):
        os.makedirs(output_dir, exist_ok=True)

        # 1. Confusion Matrix Plot
        cm = confusion_matrix(y_true, y_preds)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"Confusion Matrix ({model_name})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.colorbar(im)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="red")
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plt.tight_layout()
        plt.savefig(cm_path)
        plt.close()

        # 2. ROC Curve Plot
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve ({model_name})")
        plt.legend(loc="lower right")
        roc_path = os.path.join(output_dir, "roc_curve.png")
        plt.tight_layout()
        plt.savefig(roc_path)
        plt.close()

        # 3. Markdown Evaluation Report
        summary_path = os.path.join(output_dir, "model_evaluation_report.md")
        with open(summary_path, "w") as f:
            f.write(f"# Phase 3 Model Evaluation Report: {model_name}\n\n")
            f.write("> **Disclaimer**: This is an academic research prototype evaluation on experimental test sets. It does NOT represent clinical medical diagnostic accuracy.\n\n")
            f.write(f"- **Algorithm**: {model_name}\n")
            f.write(f"- **ROC AUC Score**: {roc_auc:.4f}\n")
            f.write(f"- **Confusion Matrix**: {cm.tolist()}\n\n")
            f.write("## Visual Artifacts Generated\n")
            f.write(f"- Confusion Matrix: `{cm_path}`\n")
            f.write(f"- ROC Curve: `{roc_path}`\n")

        logger.info(f"Evaluation report and plots saved to {output_dir}")

if __name__ == "__main__":
    # Test report generation
    y_t = np.array([0, 0, 0, 1, 1, 1])
    y_p = np.array([0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.6, 0.8, 0.9, 0.95])
    feats = ["feat_1", "feat_2"]
    ReportGenerator.generate_evaluation_artifacts(y_t, y_p, y_prob, feats)
