"""
Trains and compares several classical ML models on the wine dataset using
stratified cross-validation, then tunes the best-performing model with
GridSearchCV.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def get_candidate_models(random_state=42):
    """Return a dict of model name -> untrained estimator."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=5000, random_state=random_state),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=random_state),
        "SVM (RBF kernel)": SVC(kernel="rbf", probability=True, random_state=random_state),
    }


def compare_models(X_train, y_train, cv_folds=5, random_state=42):
    """
    Run stratified k-fold cross-validation for each candidate model.

    Returns:
        results: dict of model_name -> {'mean_accuracy': float, 'std_accuracy': float}
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    models = get_candidate_models(random_state)
    results = {}

    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        results[name] = {"mean_accuracy": scores.mean(), "std_accuracy": scores.std()}
        print(f"{name:22s} | CV accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")

    return results


def tune_model(estimator, param_grid, X_train, y_train, cv_folds=5, random_state=42):
    """Generic hyperparameter tuning helper via GridSearchCV."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    grid = GridSearchCV(estimator, param_grid=param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)

    print(f"Best params: {grid.best_params_}")
    print(f"Best CV accuracy: {grid.best_score_:.4f}")

    return grid.best_estimator_, grid.best_params_, grid.best_score_


# Default hyperparameter grids for each candidate model, used by the
# training script / notebook when tuning the best model from comparison.
PARAM_GRIDS = {
    "Logistic Regression": {"C": [0.01, 0.1, 1, 10, 100]},
    "K-Nearest Neighbors": {"n_neighbors": [3, 5, 7, 9, 11], "weights": ["uniform", "distance"]},
    "Random Forest": {
        "n_estimators": [100, 200, 400],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
    },
    "SVM (RBF kernel)": {"C": [0.1, 1, 10, 100], "gamma": ["scale", "auto", 0.01, 0.1]},
}
