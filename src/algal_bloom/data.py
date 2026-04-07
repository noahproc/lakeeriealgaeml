"""
Data loading, cleaning, and feature engineering for the WLE algal bloom dataset.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = _ROOT / "data" / "raw" / "2025_WLE_Weekly_Datashare_CSV.csv"

# ---------------------------------------------------------------------------
# Column sets
# ---------------------------------------------------------------------------
TARGET = "Extracted_CHLa_ugL-1"

# All columns we want to try to cast to float
_RAW_NUMERIC = [
    "Lat_deg", "Long_deg", "Temp_C", "Turbidity_NTU",
    "DO_mgL-1", "PAR_uEcm-2s-1", "Secchi_Depth_m",
    "Extracted_PC_ugL-1", "Wind_speed_ms-1",
    "Extracted_CHLa_ugL-1", "Date_Dec",
]

# Feature columns used for the predictive model (excluding lat/lon, which are
# handled separately by the spatial GP interpolation module)
FEATURE_COLS = [
    "Temp_C",
    "Turbidity_NTU",
    "DO_mgL-1",
    "PAR_uEcm-2s-1",
    "Secchi_Depth_m",
    "Extracted_PC_ugL-1",
    "Wind_speed_ms-1",
    "date_sin",
    "date_cos",
]

FEATURE_DISPLAY_NAMES = {
    "Temp_C": "Temperature (°C)",
    "Turbidity_NTU": "Turbidity (NTU)",
    "DO_mgL-1": "Dissolved Oxygen (mg/L)",
    "PAR_uEcm-2s-1": "PAR (µE·cm⁻²·s⁻¹)",
    "Secchi_Depth_m": "Secchi Depth (m)",
    "Extracted_PC_ugL-1": "Phycocyanin (µg/L)",
    "Wind_speed_ms-1": "Wind Speed (m/s)",
    "date_sin": "Seasonality (sin)",
    "date_cos": "Seasonality (cos)",
}

# Severity thresholds (µg/L Chl-a) – WHO / US EPA guidelines
SEVERITY_THRESHOLDS = {"low": 10, "moderate": 30, "high": 50}


# ---------------------------------------------------------------------------
# Main loading function
# ---------------------------------------------------------------------------
def load_and_clean(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load the WLE datashare CSV and return a clean, surface-only DataFrame
    with engineered features ready for modelling.

    Key steps:
      - Keep only surface samples (Sample_Depth_category == 'S')
      - Replace 'BDL' (below detection limit) with 0
      - Replace 'NS', 'N/A', '' with NaN
      - Cast numeric columns
      - Encode date cyclically (sin/cos day-of-year)
      - Sort by date (required for TimeSeriesSplit)
    """
    df = pd.read_csv(path, low_memory=False)

    # Surface samples only – removes bottom-depth duplicate rows
    df = df[df["Sample_Depth_category"] == "S"].copy()

    # Standardise missing-value sentinels
    df = df.replace({"BDL": 0, "NS": np.nan, "N/A": np.nan, "": np.nan})

    # Numeric coercion
    for col in _RAW_NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse date
    df["date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Cyclical day-of-year encoding
    df["date_doy"] = df["date"].dt.dayofyear.astype(float)
    df["date_sin"] = np.sin(2 * np.pi * df["date_doy"] / 365.25)
    df["date_cos"] = np.cos(2 * np.pi * df["date_doy"] / 365.25)

    # Sort chronologically (crucial for TimeSeriesSplit)
    df = df.sort_values("date").reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Feature / target extraction
# ---------------------------------------------------------------------------
def get_modelling_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows that have at least Temp_C, Turbidity_NTU, and target."""
    required = ["Temp_C", "Turbidity_NTU", TARGET]
    return df.dropna(subset=required).copy()


def get_XY(df: pd.DataFrame):
    """
    Return (X, y, feature_names) after dropping rows that are entirely
    missing across all features.  Remaining NaNs are handled downstream
    by the median imputer in the model pipeline.
    """
    sub = df[FEATURE_COLS + [TARGET]].copy()
    # Drop rows missing the target or both primary predictors
    sub = sub.dropna(subset=[TARGET, "Temp_C", "Turbidity_NTU"])
    X = sub[FEATURE_COLS].values.astype(float)
    y = sub[TARGET].values.astype(float)
    return X, y, FEATURE_COLS


# ---------------------------------------------------------------------------
# Station coordinate lookup
# ---------------------------------------------------------------------------
def station_coords(df: pd.DataFrame) -> dict:
    """Return {site_name: (lat, lon)} averaged over all readings."""
    coords = (
        df.groupby("Site")[["Lat_deg", "Long_deg"]]
        .mean()
        .dropna()
    )
    return {site: (row["Lat_deg"], row["Long_deg"]) for site, row in coords.iterrows()}


# ---------------------------------------------------------------------------
# Seasonal profile helpers (used by forecast tab)
# ---------------------------------------------------------------------------
def seasonal_doy_features(doy_array: np.ndarray) -> np.ndarray:
    """Convert array of day-of-year values → [sin, cos] matrix."""
    sin = np.sin(2 * np.pi * doy_array / 365.25)
    cos = np.cos(2 * np.pi * doy_array / 365.25)
    return np.column_stack([sin, cos])


def per_station_seasonal_means(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each station, compute the mean of each feature grouped by
    week-of-year (used to synthesise feature vectors for future dates).
    """
    df = df.copy()
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    return df.groupby(["Site", "week"])[FEATURE_COLS].mean().reset_index()
