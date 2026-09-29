import streamlit as st

from scripts.data_loader import load_and_validate_data
from scripts.preprocessing import define_prediction_period
from scripts.model_training import load_model
from scripts.feature_engineering import create_features
from scripts.prediction_service import run_prediction_service


# ==================================================
# OZIAGRO CONFIGURATION
# ==================================================

FILE_PATH = "data/Bende_Daily_Rainfall.xlsx"
MODEL_PATH = "outputs/results/xgboost_dry_spell_model.pkl"

SEASON_START_DOY = 60
SEASON_END_DOY = 323
PREDICTION_HORIZON = 14

DRY_DAY_THRESHOLD = 1.0
RISK_THRESHOLD = 0.45


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="OziAgro DSS",
    page_icon="🌾",
    layout="centered"
)


# ==================================================
# PAGE TITLE
# ==================================================

st.title("🌾 OziAgro Climate Decision Support System")

st.write(
    "Rainfed rice dry-spell risk assessment for Bende LGA, "
    "Abia State."
)

st.caption(
    "The system estimates the probability that a dry spell of "
    "7 or more consecutive dry days will occur within the "
    "next 14 days."
)

st.divider()


# ==================================================
# FARMER INPUT
# ==================================================

st.subheader("Farm Information")

irrigation_available = st.radio(
    "Is irrigation available for your farm?",
    options=["No", "Yes"]
)

irrigation_available = (
    irrigation_available == "Yes"
)


# ==================================================
# RUN PREDICTION
# ==================================================

if st.button("Assess Planting Risk"):

    with st.spinner("Assessing dry-spell risk..."):

        # Load rainfall data
        df = load_and_validate_data(FILE_PATH)

        # Define prediction period
        df = define_prediction_period(
            df,
            season_start_doy=SEASON_START_DOY,
            season_end_doy=SEASON_END_DOY,
            prediction_horizon=PREDICTION_HORIZON
        )

        # Load trained model
        model = load_model(MODEL_PATH)

        # Create dry-day variable
        df["dry_day"] = (
            df["rain_mm"] < DRY_DAY_THRESHOLD
        ).astype(int)

        # Create model features
        df = create_features(df)

        # Generate prediction and DSS recommendation
        result = run_prediction_service(
            df=df,
            model=model,
            irrigation_available=irrigation_available,
            threshold=RISK_THRESHOLD
        )


    # ==================================================
    # DISPLAY RESULTS
    # ==================================================

    st.divider()

    st.subheader("OziAgro Risk Assessment")

    prediction_date = result["prediction_date"]

    st.write(
    "**Prediction date:**",
    prediction_date.strftime("%d %B %Y")
    )

    st.write(
    "**Prediction horizon:**",
    "Next 14 days"
    )

    probability = result["predicted_probability"]

    st.metric(
    "Predicted Dry-Spell Probability",
    f"{probability:.1%}"
    )

    st.write(
        "**Risk level:**",
        result["risk_level"]
    )

    st.write(
        "**Irrigation available:**",
        "Yes" if result["irrigation_available"] else "No"
    )

    st.subheader("Recommendation")

    st.info(
        result["recommendation"]
    )