#!/usr/bin/env python3
"""
Export trained model results to data.js for the static HTML dashboard.

Run from the project root:
    python dashboard/export_data.py

Output: dashboard/data.js   (imported by dashboard/index.html)
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

MODELS_DIR = ROOT / "models"
DATA_PATH  = ROOT / "data" / "raw" / "2025_WLE_Weekly_Datashare_CSV.csv"
OUT_PATH   = ROOT / "dashboard" / "data.js"

# ── Load ────────────────────────────────────────────────────────────────────
print("Loading model artifacts…")
metadata    = json.loads((MODELS_DIR / "metadata.json").read_text())
y_test      = np.load(MODELS_DIR / "y_test.npy").tolist()
y_pred      = np.load(MODELS_DIR / "y_pred.npy").tolist()
shap_values = np.load(MODELS_DIR / "shap_values.npy")

# ── SHAP importance ──────────────────────────────────────────────────────────
mean_abs_shap = np.abs(shap_values).mean(axis=0).tolist()
feature_names = metadata["feature_names"]

DISPLAY = {
    "Temp_C":               "Water Temp (°C)",
    "Turbidity_NTU":        "Turbidity (NTU)",
    "DO_mgL-1":             "Dissolved O₂ (mg/L)",
    "PAR_uEcm-2s-1":        "PAR (Light)",
    "Secchi_Depth_m":       "Secchi Depth (m)",
    "Extracted_PC_ugL-1":   "Phycocyanin (µg/L)",
    "Wind_speed_ms-1":      "Wind Speed (m/s)",
    "date_sin":             "Seasonality (sin)",
    "date_cos":             "Seasonality (cos)",
}

shap_pairs = sorted(zip(feature_names, mean_abs_shap),
                    key=lambda x: x[1], reverse=True)

# ── Station coordinates from raw CSV ────────────────────────────────────────
print("Computing station coordinates from raw data…")
try:
    raw = pd.read_csv(DATA_PATH)
    # Normalise column names
    raw.columns = [c.strip() for c in raw.columns]

    # Identify lat/lon columns (flexible naming)
    lat_col = next((c for c in raw.columns if "lat" in c.lower()), None)
    lon_col = next((c for c in raw.columns if "lon" in c.lower()), None)
    site_col = next((c for c in raw.columns if "site" in c.lower()), "Site")

    if lat_col and lon_col and site_col in raw.columns:
        coords_df = (
            raw[[site_col, lat_col, lon_col]]
            .dropna()
            .groupby(site_col)[[lat_col, lon_col]]
            .mean()
            .reset_index()
        )
        stations = [
            {"id": row[site_col],
             "lat": round(float(row[lat_col]), 5),
             "lon": round(float(row[lon_col]), 5)}
            for _, row in coords_df.iterrows()
        ]
    else:
        stations = []
except Exception as e:
    print(f"  Warning: could not load station coords ({e})")
    stations = []

# Fallback approximate coords (Western Lake Erie monitoring network)
if not stations:
    stations = [
        {"id": "WE2",  "lat": 41.762, "lon": -83.382},
        {"id": "WE4",  "lat": 41.710, "lon": -83.275},
        {"id": "WE6",  "lat": 41.840, "lon": -82.980},
        {"id": "WE8",  "lat": 41.870, "lon": -82.737},
        {"id": "WE9",  "lat": 41.760, "lon": -82.770},
        {"id": "WE12", "lat": 41.880, "lon": -82.880},
        {"id": "WE13", "lat": 41.758, "lon": -82.960},
        {"id": "WE16", "lat": 41.660, "lon": -82.880},
    ]

# ── Assemble data payload ────────────────────────────────────────────────────
data = {
    "trained_at":       metadata["trained_at"],
    "test_metrics":     metadata["test_metrics"],
    "cv_results":       metadata["cv_results"],
    "xgb_params":       metadata["xgb_params"],
    "shap_expected":    metadata["shap_expected_value"],
    "n_samples":        metadata["test_metrics"]["n_train"] + metadata["test_metrics"]["n_test"],
    "observations": [
        {"x": round(float(o), 2), "y": round(float(p), 2)}
        for o, p in zip(y_test, y_pred)
    ],
    "shap_importance": [
        {"feature": DISPLAY.get(f, f), "value": round(float(v), 4)}
        for f, v in shap_pairs
    ],
    "stations": stations,
}

# ── Write ────────────────────────────────────────────────────────────────────
OUT_PATH.write_text(f"const DASHBOARD_DATA = {json.dumps(data, indent=2)};\n")
print(f"✓  Written to {OUT_PATH}")
print("   Open dashboard/index.html in your browser.")
