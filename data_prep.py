"""
Data loading and preprocessing for the UCI Wine dataset.

The dataset is bundled with scikit-learn (originally from the UCI ML
Repository), so no download or internet connection is required. It contains
the results of a chemical analysis of 178 wines grown in the same region of
Italy but derived from three different cultivars (grape varieties).
"""

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data():
    """Load the dataset as a pandas DataFrame plus target array.

    Returns:
        X (DataFrame): 13 numeric features (alcohol content, malic acid,
            ash, flavanoids, color intensity, proline, etc.)
        y (Series): target, 3 classes (0, 1, 2) corresponding to grape cultivar
        feature_names (list[str])
        target_names (list[str]): ['class_0', 'class_1', 'class_2']
    """
    data = load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y, list(data.feature_names), list(data.target_names)


def prepare_splits(X, y, test_size=0.2, random_state=42):
    """Stratified train/test split + feature scaling.

    Scaling is fit ONLY on the training set to avoid data leakage, then
    applied to both train and test sets. Scaling matters a lot here because
    features live on very different scales (e.g. 'proline' ranges in the
    hundreds while 'hue' ranges around 0-1).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
