from scripts.data_loader import load_and_validate_data
from scripts.preprocessing import define_prediction_period
from scripts.model_training import load_model
from scripts.feature_engineering import create_features
from scripts.prediction_service import run_prediction_service


# ==================================================
# OZIAGRO PREDICTION CONFIGURATION
# ==================================================

FILE_PATH = "data/Bende_Daily_Rainfall.xlsx"
MODEL_PATH = "outputs/results/xgboost_dry_spell_model.pkl"

SEASON_START_DOY = 60
SEASON_END_DOY = 323
PREDICTION_HORIZON = 14

DRY_DAY_THRESHOLD = 1.0
RISK_THRESHOLD = 0.45

IRRIGATION_AVAILABLE = False


# ==================================================
# LOAD DATA
# ==================================================

df = load_and_validate_data(FILE_PATH)


# ==================================================
# DEFINE PREDICTION PERIOD
# ==================================================

df = define_prediction_period(
    df,
    season_start_doy=SEASON_START_DOY,
    season_end_doy=SEASON_END_DOY,
    prediction_horizon=PREDICTION_HORIZON
)


# ==================================================
# LOAD TRAINED MODEL
# ==================================================

model = load_model(MODEL_PATH)


# ==================================================
# CREATE DRY-DAY VARIABLE
# ==================================================

df["dry_day"] = (
    df["rain_mm"] < DRY_DAY_THRESHOLD
).astype(int)


# ==================================================
# CREATE MODEL FEATURES
# ==================================================

df = create_features(df)


# ==================================================
# GENERATE PREDICTION
# ==================================================

result = run_prediction_service(
    df=df,
    model=model,
    irrigation_available=IRRIGATION_AVAILABLE,
    threshold=RISK_THRESHOLD
)


# ==================================================
# DISPLAY DSS RESULT
# ==================================================

print("\n" + "=" * 50)
print("OZIAGRO DSS RESULT")
print("=" * 50)

print(
    "Predicted dry-spell probability:",
    result["predicted_probability"]
)

print(
    "Risk level:",
    result["risk_level"]
)

print(
    "Irrigation available:",
    result["irrigation_available"]
)

print(
    "Recommendation:",
    result["recommendation"]
)


# ==================================================
# PIPELINE SUMMARY
# ==================================================

print("\n" + "=" * 50)
print("PREDICTION PIPELINE")
print("=" * 50)

print("Rainfall data loaded successfully.")
print("Rows:", len(df))
print("Latest date:", df["date"].max())