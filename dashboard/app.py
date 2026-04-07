"""
Western Lake Erie Algal Bloom Monitor
======================================
Streamlit dashboard integrating:
  • GP-interpolated severity maps with uncertainty overlays
  • Multivariate XGBoost model with TimeSeriesSplit cross-validation
  • SHAP feature importance (beeswarm + waterfall)
  • 365-day seasonal forecast with confidence band
  • Station-level data explorer with correlation heatmap

Run:  streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make src importable when running from project root or dashboard/
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go
import plotly.express as px
import shap
import streamlit as st

from algal_bloom.data import (
    DATA_PATH, FEATURE_COLS, FEATURE_DISPLAY_NAMES, TARGET,
    load_and_clean, station_coords, get_XY, per_station_seasonal_means,
    seasonal_doy_features,
)
from algal_bloom.models import load_artefacts, predict
from algal_bloom.spatial import interpolate, render_map, SEVERITY_CMAP, SEVERITY_NORM

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Lake Erie Algal Bloom Monitor",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark-mode CSS injection
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .block-container { padding-top: 1.5rem; }
    h1, h2, h3 { color: #e0e0e0; }
    .metric-card {
        background: #1e2530; border-radius: 8px;
        padding: 12px 16px; margin-bottom: 8px;
    }
    .stTabs [data-baseweb="tab"] { font-size: 15px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached data / model loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading dataset…")
def _load_data():
    return load_and_clean(DATA_PATH)


@st.cache_resource(show_spinner="Loading trained model…")
def _load_model():
    return load_artefacts()


@st.cache_data(show_spinner="Computing station coordinates…")
def _station_coords(df):
    return station_coords(df)


@st.cache_data(show_spinner="Computing seasonal means…")
def _seasonal_means(df):
    return per_station_seasonal_means(df)


# ---------------------------------------------------------------------------
# Helper: predict Chl-a for all stations on an arbitrary date
# ---------------------------------------------------------------------------
def predict_stations_for_date(
    target_date: pd.Timestamp,
    df: pd.DataFrame,
    seasonal_df: pd.DataFrame,
    artefacts: dict,
) -> dict[str, float]:
    """
    Predict Chl-a at each station for target_date.

    Strategy:
      - If the date is in the dataset, use actual measured features.
      - Otherwise, use per-station seasonal-mean features for that week.
    """
    week = target_date.isocalendar().week
    date_sin = float(np.sin(2 * np.pi * target_date.dayofyear / 365.25))
    date_cos = float(np.cos(2 * np.pi * target_date.dayofyear / 365.25))

    # Check if observed data exists for this date (within 1-day tolerance)
    obs_mask = (df["date"] - target_date).abs() <= pd.Timedelta("1D")
    obs_rows = df[obs_mask]

    predictions = {}
    for site in df["Site"].unique():
        site_obs = obs_rows[obs_rows["Site"] == site]
        if not site_obs.empty:
            row = site_obs.iloc[0]
            feat_vec = [
                row.get("Temp_C", np.nan),
                row.get("Turbidity_NTU", np.nan),
                row.get("DO_mgL-1", np.nan),
                row.get("PAR_uEcm-2s-1", np.nan),
                row.get("Secchi_Depth_m", np.nan),
                row.get("Extracted_PC_ugL-1", np.nan),
                row.get("Wind_speed_ms-1", np.nan),
                date_sin, date_cos,
            ]
        else:
            # Fall back to seasonal average for this station/week
            mask = (seasonal_df["Site"] == site) & (seasonal_df["week"] == week)
            sea = seasonal_df[mask]
            if sea.empty:
                # Use overall station mean
                mask2 = seasonal_df["Site"] == site
                sea = seasonal_df[mask2]
            if sea.empty:
                predictions[site] = None
                continue
            row_s = sea.iloc[0]
            feat_vec = [
                row_s.get("Temp_C", np.nan),
                row_s.get("Turbidity_NTU", np.nan),
                row_s.get("DO_mgL-1", np.nan),
                row_s.get("PAR_uEcm-2s-1", np.nan),
                row_s.get("Secchi_Depth_m", np.nan),
                row_s.get("Extracted_PC_ugL-1", np.nan),
                row_s.get("Wind_speed_ms-1", np.nan),
                date_sin, date_cos,
            ]

        X = np.array(feat_vec, dtype=float).reshape(1, -1)
        pred = float(predict(X, artefacts["model"],
                             artefacts["imputer"], artefacts["scaler"])[0])
        predictions[site] = max(0.0, pred)

    return predictions


# ---------------------------------------------------------------------------
# Tabs: define once, all content inline
# ---------------------------------------------------------------------------
def main():
    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown("""
    # 🌊 Western Lake Erie Algal Bloom Monitor
    **Gaussian-Process interpolated severity maps · XGBoost + Optuna · SHAP interpretability**
    """)
    st.markdown("---")

    # ── Load data & model ──────────────────────────────────────────────────
    try:
        artefacts = _load_model()
        model_ready = True
    except FileNotFoundError as exc:
        st.error(f"⚠️  {exc}")
        model_ready = False

    df = _load_data()
    coords = _station_coords(df)
    seasonal_df = _seasonal_means(df)
    metadata = artefacts["metadata"] if model_ready else {}

    tab_map, tab_perf, tab_forecast, tab_explore = st.tabs([
        "🗺️  Lake Map",
        "📊  Model Performance",
        "📈  Seasonal Forecast",
        "🔬  Data Explorer",
    ])

    # ══════════════════════════════════════════════════════════════════════
    # TAB 1 – LAKE MAP
    # ══════════════════════════════════════════════════════════════════════
    with tab_map:
        left, right = st.columns([3, 1])

        with right:
            st.subheader("Controls")

            obs_dates = sorted(df["date"].dropna().unique())
            obs_date_labels = [pd.Timestamp(d).strftime("%b %d, %Y") for d in obs_dates]

            date_mode = st.radio(
                "Date mode",
                ["Observed sampling dates", "Custom date"],
                horizontal=True,
            )

            if date_mode == "Observed sampling dates":
                idx = st.selectbox(
                    "Sampling date",
                    range(len(obs_dates)),
                    format_func=lambda i: obs_date_labels[i],
                    index=len(obs_dates) - 1,
                )
                selected_date = pd.Timestamp(obs_dates[idx])
                is_observed = True
            else:
                picked = st.date_input(
                    "Custom date",
                    value=pd.Timestamp("2025-08-01"),
                    min_value=pd.Timestamp("2025-01-01"),
                    max_value=pd.Timestamp("2025-12-31"),
                )
                selected_date = pd.Timestamp(picked)
                is_observed = False

            show_unc = st.toggle("Show uncertainty layer", value=False)
            st.caption("Uncertainty = GP prediction standard deviation (σ)")

        with left:
            if not model_ready:
                st.warning("Train the model first: `python train_models.py`")
                st.stop()

            with st.spinner("Running GP interpolation…"):
                station_preds = predict_stations_for_date(
                    selected_date, df, seasonal_df, artefacts
                )

            valid_sites = {s: v for s, v in station_preds.items() if v is not None}
            lons = np.array([coords[s][1] for s in valid_sites if s in coords])
            lats = np.array([coords[s][0] for s in valid_sites if s in coords])
            chla = np.array([valid_sites[s] for s in valid_sites if s in coords])

            if len(lons) < 3:
                st.warning("Not enough station data to interpolate. Select a different date.")
            else:
                mean_grid, std_grid, _ = interpolate(lons, lats, chla)

                data_tag = "Observed" if is_observed else "Model-predicted"
                fig = render_map(
                    mean_grid, std_grid,
                    station_coords={s: coords[s] for s in valid_sites if s in coords},
                    station_chla=valid_sites,
                    show_uncertainty=show_unc,
                    title=f"Chlorophyll-a  •  {selected_date.strftime('%B %d, %Y')}  [{data_tag}]",
                    figsize=(11, 5.5),
                )
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        # Station summary metrics
        if model_ready and valid_sites:
            st.markdown("#### Station Predictions")
            cols = st.columns(len(valid_sites))
            for i, (site, chl) in enumerate(sorted(valid_sites.items())):
                if chl is None:
                    continue
                if chl < 10:
                    sev, col = "Low", "normal"
                elif chl < 30:
                    sev, col = "Moderate", "off"
                elif chl < 50:
                    sev, col = "High", "inverse"
                else:
                    sev, col = "Severe", "inverse"
                cols[i].metric(
                    label=site,
                    value=f"{chl:.1f} µg/L",
                    delta=sev,
                    delta_color=col,
                )

    # ══════════════════════════════════════════════════════════════════════
    # TAB 2 – MODEL PERFORMANCE
    # ══════════════════════════════════════════════════════════════════════
    with tab_perf:
        if not model_ready:
            st.warning("No trained model found.")
            st.stop()

        st.subheader("Cross-validated Model Comparison")
        st.caption(
            "TimeSeriesSplit (3-fold, chronological) · All models use median imputation + StandardScaler"
        )

        cv_results = metadata.get("cv_results", {})
        if cv_results:
            rows = []
            for name, res in cv_results.items():
                rows.append({
                    "Model": name,
                    "RMSE (µg/L)": f"{res['RMSE']:.2f} ± {res['RMSE_std']:.2f}",
                    "R²": f"{res['R2']:.3f} ± {res['R2_std']:.3f}",
                    "MAE (µg/L)": f"{res['MAE']:.2f}",
                })
            cmp_df = pd.DataFrame(rows)
            # Highlight best model
            best_r2 = max(cv_results.items(), key=lambda x: x[1]["R2"])[0]
            st.dataframe(
                cmp_df.style.apply(
                    lambda row: ["background-color: #1e3a2f" if row["Model"] == best_r2 else ""
                                 for _ in row],
                    axis=1,
                ),
                use_container_width=True, hide_index=True,
            )
            st.caption(f"✅  Best model: **{best_r2}** (highest mean CV R²)")

        # Test-set metrics
        test_metrics = metadata.get("test_metrics", {})
        if test_metrics:
            st.markdown("#### Final XGBoost (held-out test set)")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("RMSE", f"{test_metrics['RMSE']:.2f} µg/L")
            m2.metric("R²", f"{test_metrics['R2']:.3f}")
            m3.metric("MAE", f"{test_metrics['MAE']:.2f} µg/L")
            m4.metric("Train / Test",
                      f"{test_metrics['n_train']} / {test_metrics['n_test']} samples")

        st.markdown("---")

        # ── Observed vs Predicted ──────────────────────────────────────────
        col_scatter, col_shap = st.columns(2)

        with col_scatter:
            st.subheader("Observed vs Predicted (test set)")
            y_test = artefacts["y_test"]
            y_pred = artefacts["y_pred"]
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=y_test, y=y_pred, mode="markers",
                marker=dict(size=8, color=y_test,
                            colorscale="RdYlGn_r", showscale=True,
                            colorbar=dict(title="Obs Chl-a (µg/L)", x=1.12)),
                name="Predictions",
                hovertemplate="Observed: %{x:.1f}<br>Predicted: %{y:.1f}<extra></extra>",
            ))
            lim = [min(y_test.min(), y_pred.min()) - 2,
                   max(y_test.max(), y_pred.max()) + 2]
            fig_scatter.add_trace(go.Scatter(
                x=lim, y=lim, mode="lines",
                line=dict(dash="dash", color="white", width=1),
                name="Perfect fit", showlegend=False,
            ))
            fig_scatter.update_layout(
                xaxis_title="Observed Chl-a (µg/L)",
                yaxis_title="Predicted Chl-a (µg/L)",
                template="plotly_dark", height=380,
                margin=dict(t=10, b=40, l=50, r=10),
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col_shap:
            st.subheader("SHAP Feature Importance")
            shap_values = artefacts["shap_values"]
            feature_names_display = [
                FEATURE_DISPLAY_NAMES.get(f, f) for f in metadata.get("feature_names", FEATURE_COLS)
            ]
            fig_shap, ax_shap = plt.subplots(figsize=(6, 4))
            fig_shap.patch.set_facecolor("#0e1117")
            ax_shap.set_facecolor("#0e1117")
            shap.summary_plot(
                shap_values,
                artefacts["X_all_processed"],
                feature_names=feature_names_display,
                plot_type="bar",
                color="#3498db",
                show=False,
                max_display=9,
            )
            ax_shap = plt.gca()
            ax_shap.set_facecolor("#0e1117")
            ax_shap.tick_params(colors="white", labelsize=8)
            ax_shap.set_xlabel("Mean |SHAP value|", color="white", fontsize=9)
            ax_shap.title.set_color("white")
            for spine in ax_shap.spines.values():
                spine.set_edgecolor("#444")
            plt.tight_layout()
            st.pyplot(fig_shap, use_container_width=True)
            plt.close(fig_shap)

        # ── SHAP beeswarm ──────────────────────────────────────────────────
        st.subheader("SHAP Beeswarm  (feature impact direction & magnitude)")
        fig_bee, ax_bee = plt.subplots(figsize=(10, 4))
        fig_bee.patch.set_facecolor("#0e1117")
        shap.summary_plot(
            shap_values,
            artefacts["X_all_processed"],
            feature_names=feature_names_display,
            show=False,
            max_display=9,
            plot_size=None,
        )
        ax_bee = plt.gca()
        ax_bee.set_facecolor("#0e1117")
        ax_bee.tick_params(colors="white", labelsize=8)
        ax_bee.set_xlabel("SHAP value  (impact on Chl-a prediction)", color="white", fontsize=9)
        for spine in ax_bee.spines.values():
            spine.set_edgecolor("#444")
        plt.tight_layout()
        st.pyplot(fig_bee, use_container_width=True)
        plt.close(fig_bee)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 3 – SEASONAL FORECAST
    # ══════════════════════════════════════════════════════════════════════
    with tab_forecast:
        st.subheader("365-Day Seasonal Chlorophyll-a Forecast")
        st.caption(
            "Seasonal curve derived from model predictions using per-station "
            "historical feature averages by week-of-year.  "
            "Shaded band = ± 1 std across all stations."
        )

        if not model_ready:
            st.warning("No trained model found.")
            st.stop()

        year_sel = st.slider("Year", 2025, 2030, 2025)
        site_sel = st.multiselect(
            "Stations", list(df["Site"].unique()),
            default=list(df["Site"].unique()),
        )

        doys = np.arange(1, 366)
        dates_arr = pd.to_datetime(
            [f"{year_sel}-01-01"]) + pd.to_timedelta(doys - 1, unit="D")

        # Compute forecast per station
        forecast_records = []
        for site in site_sel:
            for doy, date_val in zip(doys, dates_arr):
                week = date_val.isocalendar().week
                date_sin = float(np.sin(2 * np.pi * doy / 365.25))
                date_cos = float(np.cos(2 * np.pi * doy / 365.25))

                mask = (seasonal_df["Site"] == site) & (seasonal_df["week"] == week)
                sea = seasonal_df[mask]
                if sea.empty:
                    mask2 = seasonal_df["Site"] == site
                    sea = seasonal_df[mask2]
                if sea.empty:
                    continue

                row_s = sea.iloc[0]
                feat_vec = np.array([
                    row_s.get("Temp_C", np.nan),
                    row_s.get("Turbidity_NTU", np.nan),
                    row_s.get("DO_mgL-1", np.nan),
                    row_s.get("PAR_uEcm-2s-1", np.nan),
                    row_s.get("Secchi_Depth_m", np.nan),
                    row_s.get("Extracted_PC_ugL-1", np.nan),
                    row_s.get("Wind_speed_ms-1", np.nan),
                    date_sin, date_cos,
                ], dtype=float).reshape(1, -1)

                pred = float(predict(feat_vec, artefacts["model"],
                                     artefacts["imputer"], artefacts["scaler"])[0])
                forecast_records.append({
                    "date": date_val, "doy": doy, "site": site,
                    "chla": max(0.0, pred),
                })

        if not forecast_records:
            st.warning("No forecast data. Check station selection.")
        else:
            fcast_df = pd.DataFrame(forecast_records)
            agg = fcast_df.groupby("date")["chla"].agg(["mean", "std"]).reset_index()
            agg["std"] = agg["std"].fillna(0)

            # ---- Plotly forecast chart -----------------------------------
            fig_fc = go.Figure()

            # Band
            fig_fc.add_trace(go.Scatter(
                x=pd.concat([agg["date"], agg["date"][::-1]]),
                y=pd.concat([agg["mean"] + agg["std"],
                             (agg["mean"] - agg["std"]).clip(0)[::-1]]),
                fill="toself",
                fillcolor="rgba(52, 152, 219, 0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                name="±1 std (stations)",
                showlegend=True,
            ))

            # Mean line
            fig_fc.add_trace(go.Scatter(
                x=agg["date"], y=agg["mean"],
                mode="lines",
                line=dict(color="#3498db", width=2.5),
                name="Station mean",
            ))

            # Individual station lines
            palette = px.colors.qualitative.Set2
            for i, site in enumerate(site_sel):
                sdf = fcast_df[fcast_df["site"] == site]
                fig_fc.add_trace(go.Scatter(
                    x=sdf["date"], y=sdf["chla"],
                    mode="lines",
                    line=dict(width=1, color=palette[i % len(palette)]),
                    name=site, opacity=0.7,
                ))

            # Severity threshold lines
            for label, thresh, clr in [
                ("Low→Moderate (10)", 10, "#f1c40f"),
                ("Moderate→High (30)", 30, "#e67e22"),
                ("High→Severe (50)", 50, "#c0392b"),
            ]:
                fig_fc.add_hline(y=thresh, line_dash="dot",
                                 line_color=clr, opacity=0.6,
                                 annotation_text=label,
                                 annotation_font_color=clr)

            # Observed data points
            obs_filtered = df[df["Site"].isin(site_sel)].dropna(subset=[TARGET, "date"])
            if not obs_filtered.empty:
                fig_fc.add_trace(go.Scatter(
                    x=obs_filtered["date"],
                    y=obs_filtered[TARGET],
                    mode="markers",
                    marker=dict(size=7, color="white",
                                line=dict(color="#3498db", width=1.5)),
                    name="Observed",
                ))

            fig_fc.update_layout(
                xaxis_title="Date",
                yaxis_title="Chlorophyll-a (µg/L)",
                template="plotly_dark",
                height=480,
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1),
                hovermode="x unified",
                margin=dict(t=40, b=40, l=60, r=20),
            )
            st.plotly_chart(fig_fc, use_container_width=True)

            # Peak bloom prediction
            peak_row = agg.loc[agg["mean"].idxmax()]
            st.info(
                f"🔴  **Projected peak bloom:** {peak_row['date'].strftime('%B %d, %Y')}  "
                f"·  {peak_row['mean']:.1f} µg/L (mean across selected stations)"
            )

    # ══════════════════════════════════════════════════════════════════════
    # TAB 4 – DATA EXPLORER
    # ══════════════════════════════════════════════════════════════════════
    with tab_explore:
        st.subheader("Station Data Explorer")

        site_filter = st.multiselect(
            "Filter by station", sorted(df["Site"].unique()),
            default=sorted(df["Site"].unique()),
        )
        filtered = df[df["Site"].isin(site_filter)].copy()

        display_cols = ["date", "Site", "Temp_C", "Turbidity_NTU",
                        "DO_mgL-1", "PAR_uEcm-2s-1", "Secchi_Depth_m",
                        "Extracted_PC_ugL-1", "Wind_speed_ms-1", TARGET]
        display_cols = [c for c in display_cols if c in filtered.columns]

        st.dataframe(
            filtered[display_cols].sort_values("date", ascending=False)
            .rename(columns=FEATURE_DISPLAY_NAMES)
            .style.background_gradient(subset=["Extracted_CHLa_ugL-1"],
                                       cmap="YlOrRd"),
            use_container_width=True, height=320,
        )

        # ---- Correlation heatmap -----------------------------------------
        st.subheader("Feature Correlation Matrix")
        num_cols = [c for c in FEATURE_COLS if c not in ("date_sin", "date_cos")]
        num_cols_full = num_cols + [TARGET]
        corr = filtered[num_cols_full].dropna(how="all").corr()
        corr.columns = [FEATURE_DISPLAY_NAMES.get(c, c) for c in corr.columns]
        corr.index = corr.columns

        fig_corr = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            template="plotly_dark",
        )
        fig_corr.update_layout(height=420, margin=dict(t=30, b=10, l=10, r=10))
        st.plotly_chart(fig_corr, use_container_width=True)

        # ---- Time-series scatter per feature --------------------------------
        st.subheader("Feature Time Series")
        feat_sel = st.selectbox(
            "Feature",
            [c for c in display_cols if c not in ("date", "Site")],
            format_func=lambda c: FEATURE_DISPLAY_NAMES.get(c, c),
        )
        fig_ts = px.scatter(
            filtered, x="date", y=feat_sel, color="Site",
            template="plotly_dark",
            labels={"date": "Date", feat_sel: FEATURE_DISPLAY_NAMES.get(feat_sel, feat_sel)},
        )
        fig_ts.update_traces(marker_size=8)
        fig_ts.update_layout(height=340, margin=dict(t=10))
        st.plotly_chart(fig_ts, use_container_width=True)

    # ---- Footer ----------------------------------------------------------------
    st.markdown("---")
    trained_at = metadata.get("trained_at", "–")
    n_features = len(metadata.get("feature_names", FEATURE_COLS))
    st.caption(
        f"Model trained: {trained_at[:19]}  ·  "
        f"Features: {n_features}  ·  "
        f"Dataset: {len(df)} surface observations across {df['Site'].nunique()} stations  ·  "
        f"Spatial interpolation: Gaussian Process (Matérn 3/2 kernel)"
    )


if __name__ == "__main__":
    main()
