# 🍷 Wine Cultivar Classification

A lightweight, end-to-end classical machine learning project: predicting which of three
grape cultivars a wine came from, based purely on its chemical analysis. Built to
demonstrate a clean, correct ML workflow — EDA, preprocessing, model comparison,
hyperparameter tuning, and rigorous evaluation — without needing a GPU or any dataset
download.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.3+-f89939.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

<p align="center">
  <img src="assets/pca_projection.png" alt="PCA projection of the three cultivars" width="420"/>
</p>

## 🎯 Motivation

Not every ML problem needs a deep neural network. This project is a deliberately compact
example of doing classical ML *right*: comparing multiple algorithms fairly with
cross-validation instead of guessing, tuning hyperparameters systematically instead of by
hand, avoiding data leakage when scaling features, and reporting per-class metrics instead
of a single accuracy number that can hide problems.

## 📊 Dataset

The [UCI Wine dataset](https://archive.ics.uci.edu/dataset/109/wine), bundled directly with
scikit-learn (`sklearn.datasets.load_wine`) — no download needed. It contains 178 samples,
13 numeric features from a chemical analysis (alcohol content, malic acid, ash, flavanoids,
color intensity, proline, etc.), and 3 target classes corresponding to three different grape
cultivars grown in the same region of Italy.

## 🧠 Approach

1. **EDA** — class balance, feature correlation heatmap, and a PCA projection to sanity-check
   that the classes are actually separable before modeling.
2. **Preprocessing** — stratified train/test split, then `StandardScaler` fit only on the
   training set (avoids data leakage).
3. **Model comparison** — 4 candidate algorithms (Logistic Regression, KNN, Random Forest,
   SVM with RBF kernel) evaluated with 5-fold stratified cross-validation, so the choice of
   "best" model is backed by evidence, not intuition.
4. **Hyperparameter tuning** — `GridSearchCV` on the best candidate from step 3.
5. **Evaluation** — classification report, confusion matrix, one-vs-rest ROC curves, and
   permutation/feature importance on the held-out test set.

## 📁 Project Structure

```
wine-classifier-ml/
├── notebooks/
│   └── wine_classification.ipynb   # full narrative walkthrough
├── src/
│   ├── data_prep.py                # loading + train/test split + scaling
│   ├── train_models.py             # model comparison + hyperparameter grids
│   ├── evaluate.py                 # metrics, confusion matrix, ROC, importance
│   ├── utils.py                    # EDA plots, seeding
│   └── run_pipeline.py             # runs the entire pipeline end-to-end
├── results/                        # generated metrics + plots (populated by running the pipeline)
├── assets/                         # plots embedded in this README
├── requirements.txt
└── README.md
```

## 🚀 Quickstart

```bash
git clone https://github.com/<your-username>/wine-classifier-ml.git
cd wine-classifier-ml
pip install -r requirements.txt
python -m src.run_pipeline
```

That's it — no dataset to download. The whole pipeline (EDA + training 4 models with CV +
hyperparameter tuning + evaluation) runs in well under a minute on a laptop CPU, and saves
every plot and metric to `results/`.

Or explore interactively: `jupyter notebook notebooks/wine_classification.ipynb`

## 📈 Results

Cross-validated accuracy for each candidate model (5-fold stratified CV on the training set):

| Model | CV Accuracy |
|---|---|
| Logistic Regression | 97.9% ± 2.8% |
| K-Nearest Neighbors | 95.1% ± 3.5% |
| Random Forest | 97.9% ± 2.8% |
| **SVM (RBF kernel)** | **98.6% ± 1.7%** |

**Best model**: SVM with RBF kernel, tuned via GridSearchCV to `C=10, gamma='scale'`
(best CV accuracy: **99.3%**).

**Held-out test set performance** (36 samples never seen during training or tuning):

| Metric | Value |
|---|---|
| Accuracy | **94.4%** |
| Macro avg F1-score | 0.941 |
| Macro-average ROC-AUC (one-vs-rest) | 1.000 |

<p align="center">
  <img src="assets/confusion_matrix.png" alt="Confusion matrix" width="380"/>
</p>

The one class with lower recall (class_2, 80%) is the smallest class in the test split
(10 samples) — with so few test examples, a couple of misclassifications swing the
per-class recall a lot. See `results/classification_report.txt` for the full breakdown after
running the pipeline yourself (re-running with a different train/test split will shift these
numbers slightly).

<p align="center">
  <img src="assets/feature_importance.png" alt="Feature importance" width="500"/>
</p>

Since the best model (SVM) has no built-in feature importances, the plot above uses
**permutation importance**: each feature is shuffled and we measure how much test accuracy
drops — a model-agnostic way to rank feature usefulness.

## ⚠️ Limitations

- Only 178 samples from a single wine-growing region — results may not generalize to other
  regions or vintages.
- The dataset is small and clean by design (it's a classic ML benchmark), so near-99%
  accuracy here doesn't necessarily reflect what's achievable on messier real-world data.

## 🔮 Future Work

- Test the tuned model on an external wine dataset to check generalization.
- Try ensembling the top candidates (voting classifier).
- Add SHAP values for a more principled, per-prediction explanation of feature contributions.

## 📄 License

MIT — see [LICENSE](LICENSE).

---

*Built as an independent learning project to practice a rigorous, leakage-free classical ML
workflow: cross-validated model selection, proper hyperparameter tuning, and honest
evaluation reporting.*
