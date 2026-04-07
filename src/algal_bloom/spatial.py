"""
Gaussian Process spatial interpolation across Western Lake Erie.

Given observed (or model-predicted) Chl-a values at the 8 monitoring
stations, this module fits a GP over (lon, lat) space and returns a
continuous Chl-a mean surface + uncertainty (std) map at pixel resolution.

The GP kernel uses a Matérn 3/2 covariance (smooth but not infinitely
differentiable, appropriate for environmental fields) plus white noise.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import shapely as sv
from shapely.geometry import box
from shapely import points
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel, Matern, WhiteKernel,
)

_ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Geographic constants  (Western Basin of Lake Erie)
# ---------------------------------------------------------------------------
WEST_BOUNDS = (-83.8, 41.3, -82.4, 42.3)   # (lon_min, lat_min, lon_max, lat_max)
MAP_SIZE = (1300, 600)                       # (nx, ny) pixels

# Severity colour map:  low (green) → moderate (yellow) → high (red)
SEVERITY_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "bloom_severity",
    [(0.0, "#2ecc71"), (0.3, "#f1c40f"), (0.6, "#e67e22"), (1.0, "#c0392b")],
)
SEVERITY_NORM = mcolors.Normalize(vmin=0, vmax=80)   # µg/L Chl-a


# ---------------------------------------------------------------------------
# Shoreline & grid (cached after first call)
# ---------------------------------------------------------------------------
_cache: dict = {}


def _load_geo() -> tuple:
    """
    Load shoreline shapefile, build pixel grid, and compute land mask.
    Results are cached in-process so Streamlit only pays this cost once.
    """
    if "land_mask" in _cache:
        return (_cache["western_shoreline"], _cache["land_mask"],
                _cache["lon_grid"], _cache["lat_grid"])

    shp_path = _ROOT / "data" / "geo" / "us_medium_shoreline.shp"
    shoreline = gpd.read_file(shp_path).set_crs(epsg=4326)

    west_box = box(*WEST_BOUNDS)
    western_poly = gpd.GeoDataFrame(geometry=[west_box], crs="EPSG:4326")
    western_shoreline = gpd.clip(shoreline, western_poly)

    nx, ny = MAP_SIZE
    lon = np.linspace(WEST_BOUNDS[0], WEST_BOUNDS[2], nx)
    lat = np.linspace(WEST_BOUNDS[1], WEST_BOUNDS[3], ny)
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    land_geom = western_shoreline.unary_union.buffer(0.001)
    pts = points(lon_grid, lat_grid)
    land_mask = sv.covers(land_geom, pts)           # True where land

    _cache.update(
        western_shoreline=western_shoreline,
        land_mask=land_mask,
        lon_grid=lon_grid,
        lat_grid=lat_grid,
    )
    return western_shoreline, land_mask, lon_grid, lat_grid


# ---------------------------------------------------------------------------
# GP kernel & fitting
# ---------------------------------------------------------------------------
def _build_gp() -> GaussianProcessRegressor:
    """
    Matérn(ν=3/2) × ConstantKernel + WhiteKernel.

    length_scale ~0.3° ≈ 25 km – reasonable for inter-station distances
    in the Western Basin (≈ 60 km across).
    """
    kernel = (
        ConstantKernel(constant_value=500.0, constant_value_bounds=(1e-2, 1e5))
        * Matern(length_scale=0.3, length_scale_bounds=(0.05, 2.0), nu=1.5)
        + WhiteKernel(noise_level=5.0, noise_level_bounds=(1e-3, 1e3))
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=8,
        normalize_y=True,
        alpha=1e-6,
    )


def interpolate(
    station_lons: np.ndarray,
    station_lats: np.ndarray,
    station_chla: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, GaussianProcessRegressor]:
    """
    Fit GP on station observations, predict on full water-pixel grid.

    Parameters
    ----------
    station_lons, station_lats : 1-D arrays of station coordinates
    station_chla               : 1-D array of observed/predicted Chl-a (µg/L)

    Returns
    -------
    mean_grid : (ny, nx) float array  – Chl-a mean prediction (NaN on land)
    std_grid  : (ny, nx) float array  – prediction std / uncertainty (NaN on land)
    gp        : fitted GaussianProcessRegressor
    """
    _, land_mask, lon_grid, lat_grid = _load_geo()
    water_mask = ~land_mask
    ny, nx = lon_grid.shape

    X_train = np.column_stack([station_lons, station_lats])
    y_train = np.clip(np.array(station_chla, dtype=float), 0, None)

    gp = _build_gp()
    gp.fit(X_train, y_train)

    # Predict only on water pixels (far fewer points → manageable)
    X_pred = np.column_stack([lon_grid[water_mask], lat_grid[water_mask]])
    y_mean, y_std = gp.predict(X_pred, return_std=True)

    mean_grid = np.full((ny, nx), np.nan)
    std_grid = np.full((ny, nx), np.nan)
    mean_grid[water_mask] = np.clip(y_mean, 0, None)
    std_grid[water_mask] = y_std

    return mean_grid, std_grid, gp


# ---------------------------------------------------------------------------
# Map rendering
# ---------------------------------------------------------------------------
def render_map(
    mean_grid: np.ndarray,
    std_grid: np.ndarray,
    station_coords: dict[str, tuple[float, float]],
    station_chla: dict[str, float] | None = None,
    show_uncertainty: bool = False,
    title: str = "",
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """
    Render the GP-interpolated Chl-a map with optional uncertainty overlay.

    Parameters
    ----------
    mean_grid        : output of interpolate()
    std_grid         : output of interpolate()
    station_coords   : {site: (lat, lon)}
    station_chla     : {site: chl_value} for annotation labels
    show_uncertainty : if True, replace colour with normalised std
    title            : map title string
    figsize          : matplotlib figure size
    """
    western_shoreline, land_mask, lon_grid, lat_grid = _load_geo()

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#0d1117")

    # ---- background --------------------------------------------------------
    bg = np.zeros((*lon_grid.shape, 4))         # RGBA
    bg[land_mask] = [0.12, 0.12, 0.15, 1.0]    # dark land
    bg[~land_mask] = [0.05, 0.15, 0.3, 1.0]    # dark water

    ax.imshow(bg, extent=WEST_BOUNDS, origin="lower", aspect="auto", zorder=1)

    # ---- Chl-a surface -----------------------------------------------------
    if show_uncertainty:
        # Normalised uncertainty: std / (max_std + eps)
        display = std_grid.copy()
        valid = np.isfinite(display)
        display[~valid] = np.nan
        unc_max = np.nanpercentile(display, 95)
        norm = mcolors.Normalize(vmin=0, vmax=max(unc_max, 1))
        cmap = cm.get_cmap("YlOrRd")
        cbar_label = "Prediction Uncertainty  (1σ Chl-a  µg/L)"
    else:
        display = mean_grid
        norm = SEVERITY_NORM
        cmap = SEVERITY_CMAP
        cbar_label = "Chlorophyll-a  (µg/L)"

    # Mask land before imshow
    masked = np.ma.masked_where(land_mask | ~np.isfinite(display), display)
    im = ax.imshow(
        masked, extent=WEST_BOUNDS, origin="lower", aspect="auto",
        cmap=cmap, norm=norm, alpha=0.85, zorder=2,
        interpolation="bilinear",
    )

    # ---- shoreline overlay -------------------------------------------------
    western_shoreline.plot(
        ax=ax, facecolor="none", edgecolor="#e8e8e8",
        linewidth=0.8, zorder=3,
    )

    # ---- station markers ---------------------------------------------------
    for site, (lat, lon) in station_coords.items():
        if not (np.isfinite(lat) and np.isfinite(lon)):
            continue
        chla_val = (station_chla or {}).get(site)
        dot_color = (
            mcolors.to_hex(SEVERITY_CMAP(SEVERITY_NORM(chla_val)))
            if chla_val is not None else "#ffffff"
        )
        ax.scatter(lon, lat, s=120, color=dot_color,
                   edgecolors="white", linewidths=1.2, zorder=5)
        label_text = f"{site}"
        if chla_val is not None:
            label_text += f"\n{chla_val:.1f} µg/L"
        ax.annotate(
            label_text, xy=(lon, lat),
            xytext=(4, 4), textcoords="offset points",
            fontsize=7, color="white",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5),
            zorder=6,
        )

    # ---- colour bar --------------------------------------------------------
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02,
                        orientation="vertical")
    cbar.set_label(cbar_label, color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=8)

    # Severity legend (only on mean map)
    if not show_uncertainty:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#2ecc71", label="Low  (< 10 µg/L)"),
            Patch(facecolor="#f1c40f", label="Moderate  (10–30 µg/L)"),
            Patch(facecolor="#e67e22", label="High  (30–50 µg/L)"),
            Patch(facecolor="#c0392b", label="Severe  (> 50 µg/L)"),
        ]
        ax.legend(handles=legend_elements, loc="lower left",
                  fontsize=7, framealpha=0.6,
                  facecolor="#1a1a2e", labelcolor="white")

    ax.set_title(title, color="white", fontsize=11, pad=8)
    ax.set_xlabel("Longitude", color="#aaaaaa", fontsize=8)
    ax.set_ylabel("Latitude",  color="#aaaaaa", fontsize=8)
    ax.tick_params(colors="#aaaaaa", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

    plt.tight_layout()
    return fig
