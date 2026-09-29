from scripts.data_loader import load_and_validate_data
from scripts.preprocessing import define_prediction_period
from scripts.target_creation import (
    create_dry_spell_target,
    verify_dry_spell_target
)
from scripts.feature_engineering import create_features

from scripts.model_training import (
    prepare_model_dataset,
    audit_target_leakage,
    train_random_forest,
    train_logistic_regression,
    train_xgboost,
    evaluate_model,
    analyze_validation_thresholds
)

from scripts.model_evaluation import (
    calculate_feature_importance,
    analyze_probability_distribution
)

from scripts.dss_logic import (
    generate_dss_recommendation,
    validate_dss_logic
)
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import confusion_matrix
from scripts.data_loader import load_and_validate_data
import pandas as pd

# ==========================================
# CONFIGURATION
# ==========================================

FILE_PATH = "data/Bende_Daily_Rainfall.xlsx"

DSS_THRESHOLD = 0.25

IRRIGATION_AVAILABLE = False


# ==========================================
# LOAD AND VALIDATE DATA
# ==========================================

df = load_and_validate_data(FILE_PATH)


# ==========================================
# DEFINE MODEL PERIOD
# ==========================================

df = define_prediction_period(df)


# ==========================================
# CREATE DRY-SPELL TARGET
# ==========================================

df = create_dry_spell_target(df)


# ==========================================
# VERIFY DRY-SPELL TARGET
# ==========================================

df = verify_dry_spell_target(df)


# ==========================================
# CREATE PREDICTOR FEATURES
# ==========================================

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
) = prepare_model_dataset(df)


# ==========================================
# AUDIT TARGET LEAKAGE
# ==========================================

audit_target_leakage(
    model_df,
    predictors,
    target
)


# ==========================================
# RANDOM FOREST
# ==========================================

(
    rf_model,
    rf_train_probability,
    rf_validation_probability,
    rf_test_probability,
    rf_y_train,
    rf_y_validation,
    rf_y_test
) = train_random_forest(
    train_df,
    validation_df,
    test_df,
    predictors,
    target
)


# ==========================================
# RANDOM FOREST FEATURE IMPORTANCE
# ==========================================

feature_importance = calculate_feature_importance(
    rf_model,
    predictors
)


# ==========================================
# RANDOM FOREST VALIDATION PERFORMANCE
# ==========================================

rf_validation_results = evaluate_model(
    rf_y_validation,
    rf_validation_probability,
    threshold=DSS_THRESHOLD,
    dataset_name="Random Forest Validation"
)


# ==========================================
# RANDOM FOREST THRESHOLD ANALYSIS
# ==========================================

analyze_validation_thresholds(
    rf_y_validation,
    rf_validation_probability
)


# ==========================================
# RANDOM FOREST INDEPENDENT TEST PERFORMANCE
# ==========================================

rf_test_results = evaluate_model(
    rf_y_test,
    rf_test_probability,
    threshold=DSS_THRESHOLD,
    dataset_name="Random Forest Independent Test"
)


# ==========================================
# RANDOM FOREST TEST THRESHOLD COMPARISON
# ==========================================

for threshold in [0.15, 0.20, 0.25]:

    evaluate_model(
        rf_y_test,
        rf_test_probability,
        threshold=threshold,
        dataset_name=f"Random Forest Test - Threshold {threshold}"
    )


# ==========================================
# RANDOM FOREST PROBABILITY DISTRIBUTION
# ==========================================

analyze_probability_distribution(
    rf_y_validation,
    rf_validation_probability,
    dataset_name="Random Forest Validation"
)

analyze_probability_distribution(
    rf_y_test,
    rf_test_probability,
    dataset_name="Random Forest Independent Test"
)


# ==========================================
# LOGISTIC REGRESSION
# ==========================================

(
    logistic_model,
    logistic_train_probability,
    logistic_validation_probability,
    logistic_test_probability,
    logistic_y_train,
    logistic_y_validation,
    logistic_y_test
) = train_logistic_regression(
    train_df,
    validation_df,
    test_df,
    predictors,
    target
)


# ==========================================
# LOGISTIC REGRESSION VALIDATION PERFORMANCE
# ==========================================

logistic_validation_results = evaluate_model(
    logistic_y_validation,
    logistic_validation_probability,
    threshold=0.25,
    dataset_name="Logistic Regression Validation"
)


# ==========================================
# LOGISTIC REGRESSION INDEPENDENT TEST PERFORMANCE
# ==========================================

logistic_test_results = evaluate_model(
    logistic_y_test,
    logistic_test_probability,
    threshold=0.25,
    dataset_name="Logistic Regression Independent Test"
)


# ==========================================
# XGBOOST
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


# ==========================================
# XGBOOST VALIDATION PERFORMANCE
# ==========================================

xgb_validation_results = evaluate_model(
    xgb_y_validation,
    xgb_validation_probability,
    threshold=0.25,
    dataset_name="XGBoost Validation"
)


# ==========================================
# XGBOOST INDEPENDENT TEST PERFORMANCE
# ==========================================

xgb_test_results = evaluate_model(
    xgb_y_test,
    xgb_test_probability,
    threshold=0.25,
    dataset_name="XGBoost Independent Test"
)

# ==================================================
# XGBOOST VALIDATION THRESHOLD ANALYSIS
# ==================================================

print("\n" + "=" * 60)
print("XGBOOST VALIDATION THRESHOLD ANALYSIS")
print("=" * 60)

validation_thresholds = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80
]

threshold_results = []

for threshold in validation_thresholds:

    validation_pred = (
        xgb_validation_probability >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
    xgb_y_validation,
    validation_pred
    ).ravel()

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    false_plant_rate = (
        fp / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    false_wait_rate = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else 0
    )

    threshold_results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "false_plant_rate": false_plant_rate,
        "false_wait_rate": false_wait_rate
    })


threshold_results_df = pd.DataFrame(
    threshold_results
)

print("\nVALIDATION THRESHOLD RESULTS")

print(
    threshold_results_df.round(4).to_string(
        index=False
    )
)

print("\nHIGHEST VALIDATION F1")

best_f1_row = threshold_results_df.loc[
    threshold_results_df["f1"].idxmax()
]

print(
    best_f1_row.to_string()
)

print("\nLOWEST FALSE-PLANT RATE")

lowest_false_plant_row = threshold_results_df.loc[
    threshold_results_df["false_plant_rate"].idxmin()
]

print(
    lowest_false_plant_row.to_string()
)

print("\nLOWEST FALSE-WAIT RATE")

lowest_false_wait_row = threshold_results_df.loc[
    threshold_results_df["false_wait_rate"].idxmin()
]

print(
    lowest_false_wait_row.to_string()
)

# ============================================================
# DSS THRESHOLD SELECTION — FALSE-PLANT RATE CONSTRAINT
# ============================================================

MAX_ACCEPTABLE_FALSE_PLANT_RATE = 0.10

eligible_thresholds = threshold_results_df[
    threshold_results_df["false_plant_rate"]
    <= MAX_ACCEPTABLE_FALSE_PLANT_RATE
].copy()

eligible_thresholds = eligible_thresholds.sort_values(
    by=["recall", "f1"],
    ascending=[False, False]
)

print("\n" + "=" * 60)
print("DSS THRESHOLD SELECTION — VALIDATION")
print("=" * 60)

print(
    "Maximum acceptable false-plant rate:",
    MAX_ACCEPTABLE_FALSE_PLANT_RATE
)

print("\nELIGIBLE THRESHOLDS")

print(
    eligible_thresholds.round(4).to_string(
        index=False
    )
)

if not eligible_thresholds.empty:

    selected_threshold_row = eligible_thresholds.iloc[0]

    selected_dss_threshold = (
        selected_threshold_row["threshold"]
    )

    print("\nSELECTED VALIDATION THRESHOLD")

    print(
        "Threshold:",
        selected_dss_threshold
    )

    print(
        "Recall:",
        round(
            selected_threshold_row["recall"],
            4
        )
    )

    print(
        "False-plant rate:",
        round(
            selected_threshold_row["false_plant_rate"],
            4
        )
    )

    print(
        "False-wait rate:",
        round(
            selected_threshold_row["false_wait_rate"],
            4
        )
    )

    print(
        "F1:",
        round(
            selected_threshold_row["f1"],
            4
        )
    )

else:

    print(
        "\nNo threshold satisfies the "
        "false-plant-rate constraint."
    )

    selected_dss_threshold = None


# ==========================================
# XGBOOST INDEPENDENT TEST THRESHOLD COMPARISON
# ==========================================

for threshold in [0.05, 0.25, 0.65]:

    evaluate_model(
        xgb_y_test,
        xgb_test_probability,
        threshold=threshold,
        dataset_name=f"XGBoost Test - Threshold {threshold}"
    )

# ============================================================
# SELECTED DSS THRESHOLD — INDEPENDENT TEST EVALUATION
# ============================================================

selected_test_probability = xgb_test_probability

selected_test_predictions = (
    selected_test_probability >= selected_dss_threshold
).astype(int)

tn, fp, fn, tp = confusion_matrix(
    xgb_y_test,
    selected_test_predictions
).ravel()

test_precision = (
    tp / (tp + fp)
    if (tp + fp) > 0
    else 0
)

test_recall = (
    tp / (tp + fn)
    if (tp + fn) > 0
    else 0
)

test_specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0
)

test_f1 = (
    2 * test_precision * test_recall
    / (test_precision + test_recall)
    if (test_precision + test_recall) > 0
    else 0
)

test_false_plant_rate = (
    fp / (tn + fp)
    if (tn + fp) > 0
    else 0
)

test_false_wait_rate = (
    fn / (fn + tp)
    if (fn + tp) > 0
    else 0
)

print("\n" + "=" * 60)
print("SELECTED DSS THRESHOLD — INDEPENDENT TEST SET")
print("=" * 60)

print(
    "Locked validation threshold:",
    selected_dss_threshold
)

print("\nTEST PERFORMANCE")

print(
    "Precision:",
    round(test_precision, 4)
)

print(
    "Recall:",
    round(test_recall, 4)
)

print(
    "Specificity:",
    round(test_specificity, 4)
)

print(
    "F1:",
    round(test_f1, 4)
)

print(
    "False-plant rate:",
    round(test_false_plant_rate, 4)
)

print(
    "False-wait rate:",
    round(test_false_wait_rate, 4)
)

print("\nCONFUSION MATRIX")

print("True negatives :", tn)
print("False positives:", fp)
print("False negatives:", fn)
print("True positives  :", tp)

# ==========================================
# MODEL COMPARISON
# ==========================================

model_comparison = {

    "Random Forest": rf_test_results,

    "Logistic Regression": logistic_test_results,

    "XGBoost": xgb_test_results
}


print("\n" + "=" * 70)
print("MODEL COMPARISON — INDEPENDENT TEST SET")
print("=" * 70)

for model_name, results in model_comparison.items():

    print(f"\n{model_name}")

    print(
        "ROC-AUC:",
        round(results["roc_auc"], 4)
    )

    print(
        "PR-AUC:",
        round(results["pr_auc"], 4)
    )

    print(
        "Brier Score:",
        round(results["brier_score"], 4)
    )

    print(
        "Precision:",
        round(results["precision"], 4)
    )

    print(
        "Recall:",
        round(results["recall"], 4)
    )

    print(
        "F1 Score:",
        round(results["f1"], 4)
    )

    print(
        "False-Plant Rate:",
        round(results["false_plant_rate"], 4)
    )

    print(
        "False-Wait Rate:",
        round(results["false_wait_rate"], 4)
    )

# ==========================================
# XGBOOST DSS THRESHOLD COMPARISON
# ==========================================

dss_thresholds = [0.05, 0.25, 0.65]

print("\n" + "=" * 70)
print("XGBOOST DSS THRESHOLD COMPARISON")
print("=" * 70)

for threshold in dss_thresholds:

    high_risk_cases = (
        xgb_test_probability >= threshold
    ).sum()

    low_risk_cases = (
        xgb_test_probability < threshold
    ).sum()

    print(f"\nTHRESHOLD: {threshold}")

    print(
        "High-risk predictions:",
        high_risk_cases
    )

    print(
        "Low-risk predictions:",
        low_risk_cases
    )

    print(
        "Percentage classified high risk:",
        round(
            high_risk_cases
            / len(xgb_test_probability)
            * 100,
            2
        ),
        "%"
    )


# ==========================================
# VALIDATE OZIAGRO DSS LOGIC
# ==========================================

validate_dss_logic(
    threshold=DSS_THRESHOLD
)

# ==========================================
# GENERATE OZIAGRO DSS RECOMMENDATION
# ==========================================

# Use the selected XGBoost model for the prototype DSS
test_probability = xgb_test_probability[0]

dss_result = generate_dss_recommendation(
    predicted_probability=test_probability,
    irrigation_available=IRRIGATION_AVAILABLE,
    threshold=DSS_THRESHOLD
)


# ==========================================
# DISPLAY DSS RESULT
# ==========================================

print("\n" + "=" * 50)
print("OZIAGRO DSS RECOMMENDATION")
print("=" * 50)

print(
    "\nPredicted dry-spell probability:",
    round(dss_result["predicted_probability"], 4)
)

print(
    "Risk level:",
    dss_result["risk_level"]
)

print(
    "Irrigation available:",
    dss_result["irrigation_available"]
)

print(
    "Recommendation:",
    dss_result["recommendation"]
)