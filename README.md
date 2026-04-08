# Lake Erie Algal Bloom Prediction — Supervised ML Pipeline

The goal of this project is to predict chlorophyll-A concentrations in the western basin of Lake Erie, which serve as a key proxy for harmful algal bloom (HAB) severity. Predictions are based on environmental inputs including turbidity, water temperature, and sampling date. The western basin faces the most significant HAB threat in the Great Lakes region — most notably in 2014, when algal contamination rendered Toledo, OH's water supply unsafe for three days, affecting over 400,000 residents.

The end goal of the pipeline is to predict algal bloom severity by day of year and map predicted severity onto a plot of the Lake Erie shoreline, providing an interpretable, visual output for environmental monitoring.

Full methodology, step-by-step implementation, and model evaluation are documented in the Jupyter Notebook (`notebooks/alg_graphical.ipynb`).

---

## Datasets

**Weekly Sampling Data**
[NOAA GLERL — WLE Weekly HABs Data](https://www.glerl.noaa.gov/res/HABs_and_Hypoxia/wle-weekly-current)
Weekly environmental measurements from buoy stations in the western basin, used to train, validate, and test the model. Raw CSV is stored in `data/raw/`.

**Shoreline Data**
[NOAA Medium-Resolution Shoreline](https://shoreline.noaa.gov/med-res.html)
Shapefile mapping the Lake Erie shoreline, processed with GeoPandas and visualized with Matplotlib. Geodata files are stored in `data/geo/`.

---

## Project Structure
```
lakeeriealgaeml/
│
├── assets/
│   └── lake_erie_stations.png          # Plot of buoy station locations on the shoreline
│
├── dashboard/                          # Interactive web dashboard
│   ├── app.py                          # Dashboard application logic
│   ├── data.js                         # Frontend data handling
│   ├── export_data.py                  # Exports model outputs for dashboard use
│   └── index.html                      # Dashboard entry point
│
├── data/
│   ├── geo/                            # Shoreline geodata (GeoPandas/Matplotlib)
│   │   ├── western_basin_shoreline.cpg
│   │   ├── western_basin_shoreline.dbf
│   │   ├── western_basin_shoreline.shp
│   │   └── western_basin_shoreline.shx
│   └── raw/
│       └── 2025_WLE_Weekly_Datashare_CSV.csv   # NOAA weekly sampling data
│
├── models/                             # Trained model artifacts and processed data
│   ├── legacy/                         # Previous model versions
│   ├── X_all_processed.npy
│   ├── X_test_processed.npy
│   ├── imputer.joblib
│   ├── metadata.json
│   ├── model.joblib
│   ├── scaler.joblib
│   ├── shap_values.npy
│   ├── y_pred.npy
│   └── y_test.npy
│
├── notebooks/
│   ├── alg_graphical.ipynb             # Full modeling pipeline (EDA, training, evaluation)
│   └── user_alg.py                     # User-facing prediction interface
│
├── scalers/                            # Exported scalers and per-feature models
│   ├── alg_date_model.sav
│   ├── alg_temp_model.sav
│   ├── alg_turbidity_model.sav
│   ├── date_X_scaler.sav
│   ├── date_y_scaler.sav
│   ├── temp_X_scaler.sav
│   ├── temp_y_scaler.sav
│   ├── turb_X_scaler.sav
│   └── turb_y_scaler.sav
│
├── src/                                # Source modules
├── .gitignore
├── requirements.txt                    # Python dependencies
├── train_models.py                     # Model training script
├── user_alg.py                         # User-facing CLI prediction tool
└── README.md
```

---

## File Reference

| File / Folder | Description |
| --- | --- |
| `notebooks/alg_graphical.ipynb` | Core notebook: EDA, feature engineering, model training, SHAP analysis, and shoreline visualization |
| `train_models.py` | Standalone script to retrain all models from raw data |
| `user_alg.py` | CLI tool: input a temperature or turbidity value and receive a projected chlorophyll-A level |
| `dashboard/` | Interactive web dashboard for exploring predictions and station data |
| `data/raw/` | Raw NOAA weekly sampling CSV |
| `data/geo/` | Shoreline shapefiles for GeoPandas mapping |
| `models/` | Serialized model artifacts, scalers, SHAP values, and test outputs |
| `scalers/` | Per-feature trained models and scalers (date, temperature, turbidity) |
| `assets/lake_erie_stations.png` | Map of sampling station locations across the western basin |

---

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

To retrain models from raw data:
```bash
python train_models.py
```

To run a prediction from the command line:
```bash
python user_alg.py
```

To launch the dashboard locally, open `dashboard/index.html` in a browser or run `dashboard/app.py` depending on your setup.

---

## Stack

- **Python** — NumPy, Pandas, Scikit-learn, Matplotlib, GeoPandas
- **Modeling** — Neural networks via Scikit-learn, SHAP for interpretability
- **Geospatial** — GeoPandas + Matplotlib for shoreline mapping
- **Dashboard** — HTML/JS frontend with Python data export
