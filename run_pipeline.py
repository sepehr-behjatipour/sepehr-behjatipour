"""
End-to-end pipeline: load data -> EDA -> compare models -> tune best model
-> evaluate on test set -> save all plots/metrics to results/.

Usage:
    python -m src.run_pipeline
"""

import json
from pathlib import Path

import joblib

from src.data_prep import load_data, prepare_splits
from src.evaluate import evaluate_model, plot_feature_importance
from src.train_models import PARAM_GRIDS, compare_models, get_candidate_models, tune_model
from src.utils import plot_class_balance, plot_correlation_heatmap, plot_pca_projection, set_seed


def main(results_dir="results"):
    set_seed(42)
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    X, y, feature_names, target_names = load_data()
    print(f"Dataset shape: {X.shape}, classes: {target_names}")

    print("\nGenerating EDA plots...")
    plot_class_balance(y, target_names, Path(results_dir) / "class_balance.png")
    plot_correlation_heatmap(X, Path(results_dir) / "correlation_heatmap.png")
    plot_pca_projection(X.values, y.values, target_names, Path(results_dir) / "pca_projection.png")

    print("\nSplitting and scaling data...")
    X_train, X_test, y_train, y_test, scaler = prepare_splits(X, y)

    print("\nComparing candidate models with 5-fold cross-validation...")
    cv_results = compare_models(X_train, y_train)

    best_model_name = max(cv_results, key=lambda k: cv_results[k]["mean_accuracy"])
    print(f"\nBest model from CV: {best_model_name}")

    print(f"\nTuning {best_model_name} with GridSearchCV...")
    base_model = get_candidate_models()[best_model_name]
    best_model, best_params, best_cv_score = tune_model(
        base_model, PARAM_GRIDS[best_model_name], X_train, y_train
    )

    print("\nEvaluating tuned model on held-out test set...")
    evaluate_model(best_model, X_test, y_test, target_names, results_dir=results_dir)
    plot_feature_importance(
        best_model,
        feature_names,
        Path(results_dir) / "feature_importance.png",
        X_test=X_test,
        y_test=y_test,
    )

    # Persist model + scaler + summary metrics
    joblib.dump(best_model, Path(results_dir) / "best_model.joblib")
    joblib.dump(scaler, Path(results_dir) / "scaler.joblib")

    summary = {
        "best_model": best_model_name,
        "best_params": best_params,
        "cv_results": cv_results,
        "best_cv_accuracy": best_cv_score,
        "test_accuracy": float(best_model.score(X_test, y_test)),
    }
    with open(Path(results_dir) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Test accuracy: {summary['test_accuracy']:.4f}")
    print(f"All results saved to {results_dir}/")

    return summary


if __name__ == "__main__":
    main()
