"""
Small utility helpers: reproducibility seeding and EDA plots.
"""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def plot_class_balance(y, target_names, save_path):
    counts = y.value_counts().sort_index()
    plt.figure(figsize=(5, 4))
    plt.bar([target_names[i] for i in counts.index], counts.values, color="#9D4EDD")
    plt.ylabel("Number of samples")
    plt.title("Class Balance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved class balance plot to {save_path}")


def plot_correlation_heatmap(X, save_path):
    plt.figure(figsize=(10, 8))
    corr = X.corr()
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, square=True)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved correlation heatmap to {save_path}")


def plot_pca_projection(X, y, target_names, save_path):
    """2D PCA projection to visualize class separability."""
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(6, 5))
    colors = ["#5A189A", "#E85D75", "#48CAE4"]
    for i, name in enumerate(target_names):
        mask = y == i
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=name, alpha=0.7, color=colors[i % 3])

    var_explained = pca.explained_variance_ratio_.sum()
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"PCA Projection ({var_explained:.1%} variance explained)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved PCA projection to {save_path}")
