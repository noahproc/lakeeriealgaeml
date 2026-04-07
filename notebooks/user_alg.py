# ----- Libraries -----
import numpy as np
import joblib

# ----- Load Models -----
temp_model = joblib.load("alg_temp_model.sav")
turb_model = joblib.load("alg_turbidity_model.sav")
date_model = joblib.load("alg_date_model.sav")

# ----- Load Scalers -----
temp_X_scaler = joblib.load("temp_X_scaler.sav")
temp_y_scaler = joblib.load("temp_y_scaler.sav")

turb_X_scaler = joblib.load("turb_X_scaler.sav")
turb_y_scaler = joblib.load("turb_y_scaler.sav")

date_X_scaler = joblib.load("date_X_scaler.sav")
date_y_scaler = joblib.load("date_y_scaler.sav")

# ----- User Inputs for Predictions -----
user_interact = True

while user_interact:
    search_input = input(
        "Search by temperature (t), turbidity (tu), or quit (q)? "
    ).strip().lower()

    if search_input == 'q':
        break

    elif search_input == 't':
        temp_user = float(input("Enter water temperature (°C): ").strip())

        X_user = np.array([[temp_user]])
        X_user_scaled = temp_X_scaler.transform(X_user)

        pred_scaled = temp_model.predict(X_user_scaled)
        pred = temp_y_scaler.inverse_transform(
            pred_scaled.reshape(-1, 1)
        ).ravel()[0]

        print(f"\nPredicted Chlorophyll-A (µg/L): {pred:.2f}")
    
    elif search_input == 'tu':
        turbidity_user = float(input("Enter turbidity value: ").strip())

        X_user = np.array([[turbidity_user]])
        X_user_scaled = turb_X_scaler.transform(X_user)

        pred_scaled = turb_model.predict(X_user_scaled)
        pred = turb_y_scaler.inverse_transform(
            pred_scaled.reshape(-1, 1)
        ).ravel()[0]

        print(f"\nPredicted Chlorophyll-A (µg/L): {pred:.2f}")
    
    else:
        print("Unknown option.")
