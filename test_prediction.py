from scripts.data_loader import load_and_validate_data
from scripts.preprocessing import define_prediction_period
from scripts.target_creation import create_dry_spell_target
from scripts.feature_engineering import create_features
from scripts.model_training import train_xgboost, save_model, load_model
from scripts.prediction_service import generate_prediction


# ==========================================
# LOAD DATA
# ==========================================

FILE_PATH = "data/Bende_Daily_Rainfall.xlsx"

df = load_and_validate_data(FILE_PATH)


# ==========================================
# PREPARE DATA
# ==========================================

df = define_prediction_period(df)

df = create_dry_spell_target(df)

df = create_features(df)


# ==========================================
# PREPARE MODEL DATASET
# ==========================================

(
    model_df,
    train_df,
    validation_df,
    test_df,
    predictors,
    target
) = __import__(
    "scripts.model_training",
    fromlist=["prepare_model_dataset"]
).prepare_model_dataset(df)


# ==========================================
# TRAIN XGBOOST
# ==========================================

(
    xgb_model,
    xgb_train_probability,
    xgb_validation_probability,
    xgb_test_probability,
    xgb_y_train,
    xgb_y_validation,
    xgb_y_test
) = train_xgboost(
    train_df,
    validation_df,
    test_df,
    predictors,
    target
)

MODEL_PATH = "outputs/results/xgboost_dry_spell_model.pkl"

save_model(
    xgb_model,
    MODEL_PATH
)
xgb_model = load_model(MODEL_PATH)

# ==========================================
# GENERATE DSS PREDICTION
# ==========================================

result = generate_prediction(
    df=df,
    model=xgb_model,
    irrigation_available=False,
    threshold=0.25
)


# ==========================================
# DISPLAY RESULT
# ==========================================

print("\n" + "=" * 50)
print("PREDICTION SERVICE TEST")
print("=" * 50)

print(
    "Predicted dry-spell probability:",
    round(result["predicted_probability"], 4)
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