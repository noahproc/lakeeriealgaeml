"""
Model training, cross-validation, hyperparameter optimisation (Optuna),
and SHAP-based interpretability for Chl-a prediction.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import optuna
import shap
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_validate, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = _ROOT / "models"

N_CV_SPLITS = 3   # small dataset – 3 folds keeps test sets ~15 samples each


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------
def _make_pipeline(estimator) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])


# ---------------------------------------------------------------------------
# Model comparison (all candidates, TimeSeriesSplit CV)
# ---------------------------------------------------------------------------
CANDIDATE_MODELS: dict[str, Pipeline] = {
    "Ridge": _make_pipeline(Ridge(alpha=10.0)),
    "Random Forest": _make_pipeline(
        RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    ),
    "Gradient Boosting": _make_pipeline(
        GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                  max_depth=4, random_state=42)
    ),
    "MLP": _make_pipeline(
        MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu",
                     solver="adam", max_iter=2000, random_state=42,
                     early_stopping=True, validation_fraction=0.15)
    ),
    # XGBoost placeholder – will be replaced by Optuna-tuned version
    "XGBoost": _make_pipeline(
        XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4,
                     subsample=0.8, colsample_bytree=0.8,
                     random_state=42, verbosity=0)
    ),
}


def compare_models(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Run TimeSeriesSplit cross-validation for all candidates.
    Returns a dict: model_name → {RMSE, R2, MAE, RMSE_std, R2_std}.
    """
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    results: dict[str, dict] = {}

    for name, pipe in CANDIDATE_MODELS.items():
        cv = cross_validate(
            pipe, X, y, cv=tscv,
            scoring=[
                "neg_root_mean_squared_error",
                "r2",
                "neg_mean_absolute_error",
            ],
            return_train_score=False,
            error_score="raise",
        )
        results[name] = {
            "RMSE": float(-cv["test_neg_root_mean_squared_error"].mean()),
            "RMSE_std": float(cv["test_neg_root_mean_squared_error"].std()),
            "R2": float(cv["test_r2"].mean()),
            "R2_std": float(cv["test_r2"].std()),
            "MAE": float(-cv["test_neg_mean_absolute_error"].mean()),
        }
        print(f"  {name:<22} RMSE={results[name]['RMSE']:.3f}  R²={results[name]['R2']:.3f}")

    return results


# ---------------------------------------------------------------------------
# Optuna hyperparameter tuning for XGBoost
# ---------------------------------------------------------------------------
def tune_xgboost(X: np.ndarray, y: np.ndarray, n_trials: int = 60) -> dict:
    """Return best hyperparameter dict found by Optuna."""
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_clean = scaler.fit_transform(imputer.fit_transform(X))

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 50, 400),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 5.0, log=True),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
        )
        model = XGBRegressor(**params, random_state=42, verbosity=0)
        rmse_scores = []
        for train_idx, val_idx in tscv.split(X_clean):
            model.fit(X_clean[train_idx], y[train_idx])
            preds = model.predict(X_clean[val_idx])
            rmse_scores.append(mean_squared_error(y[val_idx], preds) ** 0.5)
        return float(np.mean(rmse_scores))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"\n  Best Optuna RMSE: {study.best_value:.3f}")
    print(f"  Best params: {study.best_params}")
    return study.best_params


# ---------------------------------------------------------------------------
# Final model training + SHAP
# ---------------------------------------------------------------------------
def train_final_model(X: np.ndarray, y: np.ndarray,
                      feature_names: list[str],
                      xgb_params: dict | None = None) -> dict:
    """
    Train the final XGBoost model on a chronological train/test split.
    Computes SHAP values on the held-out test set.

    Returns a dict of all trained artefacts and diagnostics.
    """
    # Chronological split (no shuffle – respects temporal order)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Preprocessing fit on train only
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train_p = scaler.fit_transform(imputer.fit_transform(X_train))
    X_test_p = scaler.transform(imputer.transform(X_test))

    # Model
    params = xgb_params or {}
    params.setdefault("random_state", 42)
    params.setdefault("verbosity", 0)
    model = XGBRegressor(**params)
    model.fit(X_train_p, y_train)

    y_pred = model.predict(X_test_p)
    metrics = {
        "RMSE": float(mean_squared_error(y_test, y_pred) ** 0.5),
        "R2": float(r2_score(y_test, y_pred)),
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }
    print(f"\n  Final model — RMSE={metrics['RMSE']:.3f}  R²={metrics['R2']:.3f}  MAE={metrics['MAE']:.3f}")

    # SHAP (TreeExplainer – exact, no approximations)
    explainer = shap.TreeExplainer(model)
    # Use training set for background; compute on full dataset for richer plots
    X_all_p = scaler.transform(imputer.transform(X))
    shap_values = explainer.shap_values(X_all_p)
    shap_expected = float(explainer.expected_value)

    return {
        "model": model,
        "imputer": imputer,
        "scaler": scaler,
        "metrics": metrics,
        "shap_values": shap_values,          # shape (n_samples, n_features)
        "shap_expected": shap_expected,
        "X_test_processed": X_test_p,
        "y_test": y_test,
        "y_pred": y_pred,
        "feature_names": feature_names,
        "X_all_processed": X_all_p,
    }


# ---------------------------------------------------------------------------
# Prediction helpers (used by dashboard)
# ---------------------------------------------------------------------------
def predict(X_raw: np.ndarray, model, imputer, scaler) -> np.ndarray:
    """Apply saved preprocessing → model.predict."""
    X_p = scaler.transform(imputer.transform(X_raw))
    return model.predict(X_p)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_artefacts(artefacts: dict, cv_results: dict,
                   feature_names: list[str], xgb_params: dict) -> None:
    MODELS_DIR.mkdir(exist_ok=True)

    joblib.dump(artefacts["model"],   MODELS_DIR / "model.joblib")
    joblib.dump(artefacts["imputer"], MODELS_DIR / "imputer.joblib")
    joblib.dump(artefacts["scaler"],  MODELS_DIR / "scaler.joblib")

    np.save(str(MODELS_DIR / "shap_values.npy"),      artefacts["shap_values"])
    np.save(str(MODELS_DIR / "X_all_processed.npy"),  artefacts["X_all_processed"])
    np.save(str(MODELS_DIR / "y_test.npy"),            artefacts["y_test"])
    np.save(str(MODELS_DIR / "y_pred.npy"),            artefacts["y_pred"])
    np.save(str(MODELS_DIR / "X_test_processed.npy"), artefacts["X_test_processed"])

    metadata = {
        "trained_at": datetime.now().isoformat(),
        "feature_names": feature_names,
        "xgb_params": xgb_params,
        "test_metrics": artefacts["metrics"],
        "cv_results": cv_results,
        "shap_expected_value": artefacts["shap_expected"],
    }
    with open(MODELS_DIR / "metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"\n  Artefacts saved → {MODELS_DIR}")


def load_artefacts() -> dict:
    """Load all saved model artefacts (used by the dashboard)."""
    meta_path = MODELS_DIR / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            "No trained model found. Run `python train_models.py` first."
        )
    with open(meta_path) as fh:
        metadata = json.load(fh)

    return {
        "model":             joblib.load(MODELS_DIR / "model.joblib"),
        "imputer":           joblib.load(MODELS_DIR / "imputer.joblib"),
        "scaler":            joblib.load(MODELS_DIR / "scaler.joblib"),
        "shap_values":       np.load(str(MODELS_DIR / "shap_values.npy")),
        "X_all_processed":   np.load(str(MODELS_DIR / "X_all_processed.npy")),
        "y_test":            np.load(str(MODELS_DIR / "y_test.npy")),
        "y_pred":            np.load(str(MODELS_DIR / "y_pred.npy")),
        "X_test_processed":  np.load(str(MODELS_DIR / "X_test_processed.npy")),
        "metadata":          metadata,
    }
