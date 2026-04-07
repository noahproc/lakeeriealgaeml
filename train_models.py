"""
Training script for the Western Lake Erie Algal Bloom ML system.

Steps
-----
1. Load and clean the WLE datashare CSV
2. Build multivariate feature matrix (9 features vs. single-feature originals)
3. Run TimeSeriesSplit cross-validation for 5 candidate models
4. Optuna hyperparameter search for XGBoost (60 trials)
5. Train final XGBoost on 80/20 chronological split
6. Compute SHAP values (TreeExplainer – exact)
7. Save all artefacts to models/

Usage
-----
    python train_models.py              # full run (Optuna tuning enabled)
    python train_models.py --fast       # skip Optuna, use sensible defaults
"""

import argparse
import sys
from pathlib import Path

# Make src/ importable from project root
sys.path.insert(0, str(Path(__file__).parent / "src"))

from algal_bloom.data import load_and_clean, get_XY
from algal_bloom.models import (
    compare_models,
    tune_xgboost,
    train_final_model,
    save_artefacts,
)


def main(fast: bool = False) -> None:
    print("=" * 60)
    print("  Lake Erie Algal Bloom ML – Model Training")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    print("\n[1/4]  Loading & cleaning dataset…")
    df = load_and_clean()
    X, y, feature_names = get_XY(df)
    print(f"       {X.shape[0]} surface samples · {X.shape[1]} features")
    print(f"       Features: {feature_names}")
    print(f"       Target range: {y.min():.1f} – {y.max():.1f} µg/L Chl-a")

    # ------------------------------------------------------------------
    # 2. Model comparison (TimeSeriesSplit CV)
    # ------------------------------------------------------------------
    print("\n[2/4]  Cross-validating candidate models (TimeSeriesSplit, 3 folds)…")
    cv_results = compare_models(X, y)

    # ------------------------------------------------------------------
    # 3. Optuna tuning
    # ------------------------------------------------------------------
    if fast:
        print("\n[3/4]  --fast flag: skipping Optuna, using default XGBoost params.")
        best_params = {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }
    else:
        print("\n[3/4]  Optuna hyperparameter search (XGBoost, 60 trials)…")
        best_params = tune_xgboost(X, y, n_trials=60)

    # ------------------------------------------------------------------
    # 4. Final model + SHAP
    # ------------------------------------------------------------------
    print("\n[4/4]  Training final XGBoost model + computing SHAP values…")
    artefacts = train_final_model(X, y, feature_names, xgb_params=best_params)

    # ------------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------------
    save_artefacts(artefacts, cv_results, feature_names, best_params)

    print("\n" + "=" * 60)
    print("  Training complete. Launch the dashboard with:")
    print("  streamlit run dashboard/app.py")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Skip Optuna tuning (faster, slightly less optimal)")
    args = parser.parse_args()
    main(fast=args.fast)
