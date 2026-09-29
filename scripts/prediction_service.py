from scripts.dss_logic import generate_dss_recommendation


# ==================================================
# MODEL PREDICTOR CONFIGURATION
# ==================================================

PREDICTORS = [
    "consecutive_dry_days",
    "rainfall_1d",
    "rainfall_3d",
    "rainfall_5d",
    "rainfall_7d",
    "rainfall_14d",
    "rainfall_30d",
    "rainy_days_7d",
    "dry_days_7d",
    "rainy_days_14d",
    "dry_days_14d",
    "rainy_days_30d",
    "dry_days_30d",
    "doy_sin",
    "doy_cos"
]


# ==================================================
# GENERATE PREDICTION
# ==================================================

def generate_prediction(
    df,
    model,
    irrigation_available=False,
    threshold=0.25
):
    """
    Generate a dry-spell risk prediction and
    OziAgro DSS recommendation.

    The dataframe must already contain:
    - valid_prediction_period
    - model features created by feature_engineering.py
    """

    # Select only valid prediction origins
    valid_rows = df[
        df["valid_prediction_period"] == 1
    ]

    if valid_rows.empty:
        raise ValueError(
            "No valid prediction origin is available."
        )

    # Select the latest valid prediction origin
    latest_valid_row = valid_rows.iloc[[-1]]

    prediction_date = latest_valid_row["date"].iloc[0]

    # Select the exact predictors used by the model
    X_latest = latest_valid_row[PREDICTORS].copy()

    # Ensure numerical input
    X_latest = X_latest.astype(float)

    # Generate probability of a future dry spell
    predicted_probability = model.predict_proba(
        X_latest
    )[0, 1]

    # Convert prediction into DSS recommendation
    dss_result = generate_dss_recommendation(
        predicted_probability=predicted_probability,
        irrigation_available=irrigation_available,
        threshold=threshold
    )

    # Add prediction date to the DSS result
    dss_result["prediction_date"] = prediction_date

    return dss_result


# ==================================================
# COMPLETE OZIAGRO PREDICTION SERVICE
# ==================================================

def run_prediction_service(
    df,
    model,
    irrigation_available=False,
    threshold=0.25
):
    """
    Run the complete OziAgro prediction service.

    This function provides the application-facing
    entry point for the OziAgro DSS.
    """

    result = generate_prediction(
        df=df,
        model=model,
        irrigation_available=irrigation_available,
        threshold=threshold
    )

    return result