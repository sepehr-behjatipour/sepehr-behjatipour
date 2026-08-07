"""
Evaluation utilities: classification report, confusion matrix, per-class
ROC curves (one-vs-rest, since this is a 3-class problem), and feature
importance plots.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


def evaluate_model(model, X_test, y_test, target_names, results_dir="results"):
    """Compute and save the full evaluation suite for a fitted model."""
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_test)

    print("\n=== Classification Report ===")
    report = classification_report(y_test, y_pred, target_names=target_names, digits=4)
    print(report)
    with open(Path(results_dir) / "classification_report.txt", "w") as f:
        f.write(report)

    plot_confusion_matrix(y_test, y_pred, target_names, Path(results_dir) / "confusion_matrix.png")

    if hasattr(model, "predict_proba"):
        auc = plot_multiclass_roc(
            model, X_test, y_test, target_names, Path(results_dir) / "roc_curves.png"
        )
        print(f"Macro-average ROC-AUC (one-vs-rest): {auc:.4f}")

    return y_pred


def plot_confusion_matrix(y_true, y_pred, target_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    disp.plot(ax=ax, cmap="Purples", colorbar=False)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix to {save_path}")


def plot_multiclass_roc(model, X_test, y_test, target_names, save_path):
    """One-vs-rest ROC curves for a multi-class classifier."""
    classes = list(range(len(target_names)))
    y_test_bin = label_binarize(y_test, classes=classes)
    y_probs = model.predict_proba(X_test)

    plt.figure(figsize=(6, 6))
    aucs = []
    for i, name in enumerate(target_names):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_probs[:, i])
        auc = roc_auc_score(y_test_bin[:, i], y_probs[:, i])
        aucs.append(auc)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (One-vs-Rest)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved ROC curves to {save_path}")

    return float(np.mean(aucs))


def plot_feature_importance(model, feature_names, save_path, top_n=10, X_test=None, y_test=None):
    """Plot feature importance for tree-based models, |coefficients| for
    linear models, or permutation importance as a model-agnostic fallback
    (e.g. for SVM, which exposes neither of the above)."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        title = "Feature Importance (Tree-based Model)"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).mean(axis=0)
        title = "Mean |Coefficient| Across Classes (Linear Model)"
    elif X_test is not None and y_test is not None:
        from sklearn.inspection import permutation_importance

        result = permutation_importance(
            model, X_test, y_test, n_repeats=20, random_state=42, scoring="accuracy"
        )
        importances = result.importances_mean
        title = "Permutation Importance (accuracy drop when feature is shuffled)"
    else:
        print("Model does not expose feature importances or coefficients — skipping plot.")
        return

    imp_series = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(7, 5))
    imp_series.sort_values().plot(kind="barh", color="#7B2CBF")
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved feature importance plot to {save_path}")
