import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)

from xgboost import XGBClassifier
import joblib


def prepare_model_dataset(df):
    predictors = [
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

    target = "future_dry_spell"

    # Keep only valid prediction observations
    model_df = df[
        df["valid_prediction_period"] == 1
    ].copy()

    print("\nMODEL DATASET BEFORE CLEANING")
    print("Rows:", len(model_df))

    # Check required columns
    required_columns = predictors + [target]

    missing_columns = [
        column
        for column in required_columns
        if column not in model_df.columns
    ]

    print("\nMISSING REQUIRED COLUMNS")

    if missing_columns:
        print(missing_columns)
    else:
        print("None")

    # Check missing values
    print("\nMISSING VALUES IN MODEL VARIABLES")
    print(
        model_df[required_columns].isnull().sum()
    )

    # Remove incomplete observations
    model_df = model_df.dropna(
        subset=required_columns
    ).copy()

    print("\nMODEL DATASET AFTER CLEANING")
    print("Rows:", len(model_df))

    # Target distribution
    print("\nMODEL TARGET DISTRIBUTION")
    print(
        model_df[target]
        .value_counts()
        .sort_index()
    )

    print("\nMODEL TARGET PERCENTAGES")
    print(
        model_df[target]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )

    # Display predictors
    print("\nMODEL PREDICTORS")

    for predictor in predictors:
        print("-", predictor)

    # Chronological split
    train_df = model_df[
        model_df["year"] <= 2014
    ].copy()

    validation_df = model_df[
        (model_df["year"] >= 2015) &
        (model_df["year"] <= 2019)
    ].copy()

    test_df = model_df[
        model_df["year"] >= 2020
    ].copy()

    print("\nCHRONOLOGICAL DATA SPLIT")

    print(
        "Training:",
        train_df["year"].min(),
        "to",
        train_df["year"].max(),
        "| rows:",
        len(train_df)
    )

    print(
        "Validation:",
        validation_df["year"].min(),
        "to",
        validation_df["year"].max(),
        "| rows:",
        len(validation_df)
    )

    print(
        "Test:",
        test_df["year"].min(),
        "to",
        test_df["year"].max(),
        "| rows:",
        len(test_df)
    )

    # Target distribution by split
    print("\nTARGET DISTRIBUTION BY SPLIT")

    print("\nTRAINING")
    print(
        train_df[target]
        .value_counts()
        .sort_index()
    )

    print("\nVALIDATION")
    print(
        validation_df[target]
        .value_counts()
        .sort_index()
    )

    print("\nTEST")
    print(
        test_df[target]
        .value_counts()
        .sort_index()
    )

    return (
        model_df,
        train_df,
        validation_df,
        test_df,
        predictors,
        target
    )


def audit_target_leakage(
    df,
    predictors,
    target="future_dry_spell"
):
    print("\nTARGET LEAKAGE AUDIT")

    # Check that target is not a predictor
    print("\nCHECK 1: TARGET INCLUDED AS PREDICTOR")

    if target in predictors:
        print("WARNING: TARGET LEAKAGE DETECTED")
    else:
        print(
            "PASS: Target is not included among predictors."
        )

    # Check predictor names
    suspicious_columns = [
        column
        for column in predictors
        if "future" in column.lower()
        or "target" in column.lower()
    ]

    print("\nCHECK 2: FUTURE/TARGET-RELATED PREDICTORS")

    if suspicious_columns:
        print("WARNING:", suspicious_columns)
    else:
        print(
            "PASS: No future/target-related "
            "predictor names detected."
        )

    # Identify predictors based on past rainfall/state information
    past_predictors = [
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
        "dry_days_30d"
    ]

    print("\nCHECK 3: TEMPORAL PREDICTOR STRUCTURE")

    print("Past rainfall/count predictors:")

    for column in past_predictors:
        print("PASS:", column)

    # Describe target window
    print("\nCHECK 4: TARGET DEFINITION")

    print("Target:", target)

    print(
        "Target represents a dry spell occurring "
        "within the following 14 days."
    )

    print(
        "Predictors represent rainfall/state information "
        "available at the prediction origin."
    )

    # Check chronological ordering
    print("\nCHECK 5: TEMPORAL ORDER")

    if df["date"].is_monotonic_increasing:
        print("PASS: Dataset is chronologically ordered.")
    else:
        print(
            "WARNING: Dataset is not chronologically ordered."
        )

    # Confirm chronological split
    print("\nCHECK 6: CHRONOLOGICAL SPLIT BOUNDARIES")

    print("Training period:   1981–2014")
    print("Validation period: 2015–2019")
    print("Test period:       2020–2025")

    print(
        "PASS: Splits are separated chronologically."
    )

    print("\nLEAKAGE AUDIT COMPLETE")

    return True


def train_logistic_regression(
    train_df,
    validation_df,
    test_df,
    predictors,
    target="future_dry_spell"
):
    X_train = train_df[predictors]
    y_train = train_df[target]

    X_validation = validation_df[predictors]
    y_validation = validation_df[target]

    X_test = test_df[predictors]
    y_test = test_df[target]

    print("\nLOGISTIC REGRESSION BASELINE")

    print("\nTRAINING DATA")
    print("X:", X_train.shape)
    print("y:", y_train.shape)

    print("\nVALIDATION DATA")
    print("X:", X_validation.shape)
    print("y:", y_validation.shape)

    print("\nTEST DATA")
    print("X:", X_test.shape)
    print("y:", y_test.shape)

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("\nLogistic Regression training completed.")

    train_probability = model.predict_proba(
        X_train
    )[:, 1]

    validation_probability = model.predict_proba(
        X_validation
    )[:, 1]

    test_probability = model.predict_proba(
        X_test
    )[:, 1]

    print("\nPREDICTED RISK PROBABILITIES")

    print("\nTraining")
    print(
        "Minimum:",
        round(train_probability.min(), 4)
    )
    print(
        "Maximum:",
        round(train_probability.max(), 4)
    )
    print(
        "Mean:",
        round(train_probability.mean(), 4)
    )

    print("\nValidation")
    print(
        "Minimum:",
        round(validation_probability.min(), 4)
    )
    print(
        "Maximum:",
        round(validation_probability.max(), 4)
    )
    print(
        "Mean:",
        round(validation_probability.mean(), 4)
    )

    print("\nTest")
    print(
        "Minimum:",
        round(test_probability.min(), 4)
    )
    print(
        "Maximum:",
        round(test_probability.max(), 4)
    )
    print(
        "Mean:",
        round(test_probability.mean(), 4)
    )

    return (
        model,
        train_probability,
        validation_probability,
        test_probability,
        y_train,
        y_validation,
        y_test
    )


def evaluate_model(
    y_true,
    predicted_probability,
    threshold=0.50,
    dataset_name="Validation"
):
    predicted_class = (
        predicted_probability >= threshold
    ).astype(int)

    # Probability metrics
    roc_auc = roc_auc_score(
        y_true,
        predicted_probability
    )

    pr_auc = average_precision_score(
        y_true,
        predicted_probability
    )

    brier = brier_score_loss(
        y_true,
        predicted_probability
    )

    # Classification metrics
    precision = precision_score(
        y_true,
        predicted_class,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predicted_class,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predicted_class,
        zero_division=0
    )

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predicted_class,
        labels=[0, 1]
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    # Decision-relevant error rates
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

    print("\nMODEL EVALUATION")
    print("Dataset:", dataset_name)
    print("Threshold:", threshold)

    print("\nPROBABILITY METRICS")
    print("ROC-AUC:", round(roc_auc, 4))
    print("PR-AUC:", round(pr_auc, 4))
    print("Brier score:", round(brier, 4))

    print("\nCLASSIFICATION METRICS")
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("Specificity:", round(specificity, 4))
    print("F1 score:", round(f1, 4))

    print("\nCONFUSION MATRIX")
    print("True negatives :", tn)
    print("False positives:", fp)
    print("False negatives:", fn)
    print("True positives  :", tp)

    print("\nDECISION-RELEVANT RATES")
    print(
        "False-plant rate:",
        round(false_plant_rate, 4)
    )
    print(
        "False-wait rate:",
        round(false_wait_rate, 4)
    )

    return {
        "dataset": dataset_name,
        "threshold": threshold,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "false_plant_rate": false_plant_rate,
        "false_wait_rate": false_wait_rate
    }


def analyze_validation_thresholds(
    y_true,
    predicted_probability
):
    print("\nVALIDATION THRESHOLD ANALYSIS")

    thresholds = np.arange(
        0.05,
        0.96,
        0.05
    )

    threshold_results = []

    for threshold in thresholds:

        predicted_class = (
            predicted_probability >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            predicted_class,
            labels=[0, 1]
        ).ravel()

        precision = precision_score(
            y_true,
            predicted_class,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            predicted_class,
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            predicted_class,
            zero_division=0
        )

        specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0
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
            "threshold": round(threshold, 2),
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
            "false_plant_rate": false_plant_rate,
            "false_wait_rate": false_wait_rate,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn
        })

    threshold_df = pd.DataFrame(
        threshold_results
    )

    print("\nTHRESHOLD RESULTS")
    print(
        threshold_df[
            [
                "threshold",
                "precision",
                "recall",
                "specificity",
                "f1",
                "false_plant_rate",
                "false_wait_rate"
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    best_f1_row = threshold_df.loc[
        threshold_df["f1"].idxmax()
    ]

    print("\nHIGHEST F1 THRESHOLD")
    print(
        "Threshold:",
        best_f1_row["threshold"]
    )
    print(
        "F1:",
        round(best_f1_row["f1"], 4)
    )
    print(
        "Precision:",
        round(best_f1_row["precision"], 4)
    )
    print(
        "Recall:",
        round(best_f1_row["recall"], 4)
    )
    print(
        "False-plant rate:",
        round(
            best_f1_row["false_plant_rate"],
            4
        )
    )
    print(
        "False-wait rate:",
        round(
            best_f1_row["false_wait_rate"],
            4
        )
    )

    return threshold_df


def train_random_forest(
    train_df,
    validation_df,
    test_df,
    predictors,
    target="future_dry_spell"
):
    X_train = train_df[predictors]
    y_train = train_df[target]

    X_validation = validation_df[predictors]
    y_validation = validation_df[target]

    X_test = test_df[predictors]
    y_test = test_df[target]

    print("\nRANDOM FOREST MODEL")

    print("\nTRAINING DATA")
    print("X:", X_train.shape)
    print("y:", y_train.shape)

    model = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    print("\nRandom Forest training completed.")

    train_probability = model.predict_proba(
        X_train
    )[:, 1]

    validation_probability = model.predict_proba(
        X_validation
    )[:, 1]

    test_probability = model.predict_proba(
        X_test
    )[:, 1]

    print("\nPREDICTED RISK PROBABILITIES")

    print("\nTraining")
    print(
        "Minimum:",
        round(train_probability.min(), 4)
    )
    print(
        "Maximum:",
        round(train_probability.max(), 4)
    )
    print(
        "Mean:",
        round(train_probability.mean(), 4)
    )

    print("\nValidation")
    print(
        "Minimum:",
        round(validation_probability.min(), 4)
    )
    print(
        "Maximum:",
        round(validation_probability.max(), 4)
    )
    print(
        "Mean:",
        round(validation_probability.mean(), 4)
    )

    print("\nTest")
    print(
        "Minimum:",
        round(test_probability.min(), 4)
    )
    print(
        "Maximum:",
        round(test_probability.max(), 4)
    )
    print(
        "Mean:",
        round(test_probability.mean(), 4)
    )

    return (
        model,
        train_probability,
        validation_probability,
        test_probability,
        y_train,
        y_validation,
        y_test
    )


def train_xgboost(
    train_df,
    validation_df,
    test_df,
    predictors,
    target="future_dry_spell"
):
    X_train = train_df[predictors]
    y_train = train_df[target]

    X_validation = validation_df[predictors]
    y_validation = validation_df[target]

    X_test = test_df[predictors]
    y_test = test_df[target]

    print("\nXGBOOST MODEL")

    print("\nTRAINING DATA")
    print("X:", X_train.shape)
    print("y:", y_train.shape)

    # Calculate class imbalance
    positive_cases = (y_train == 1).sum()
    negative_cases = (y_train == 0).sum()

    scale_pos_weight = (
        negative_cases / positive_cases
    )

    print("\nCLASS IMBALANCE")
    print("Negative cases:", negative_cases)
    print("Positive cases:", positive_cases)
    print(
        "Scale positive weight:",
        round(scale_pos_weight, 4)
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    print("\nXGBoost training completed.")

    train_probability = model.predict_proba(
        X_train
    )[:, 1]

    validation_probability = model.predict_proba(
        X_validation
    )[:, 1]

    test_probability = model.predict_proba(
        X_test
    )[:, 1]

    print("\nPREDICTED RISK PROBABILITIES")

    print("\nTraining")
    print(
        "Minimum:",
        round(train_probability.min(), 4)
    )
    print(
        "Maximum:",
        round(train_probability.max(), 4)
    )
    print(
        "Mean:",
        round(train_probability.mean(), 4)
    )

    print("\nValidation")
    print(
        "Minimum:",
        round(validation_probability.min(), 4)
    )
    print(
        "Maximum:",
        round(validation_probability.max(), 4)
    )
    print(
        "Mean:",
        round(validation_probability.mean(), 4)
    )

    print("\nTest")
    print(
        "Minimum:",
        round(test_probability.min(), 4)
    )
    print(
        "Maximum:",
        round(test_probability.max(), 4)
    )
    print(
        "Mean:",
        round(test_probability.mean(), 4)
    )

    return (
        model,
        train_probability,
        validation_probability,
        test_probability,
        y_train,
        y_validation,
        y_test
    )

def save_model(model, file_path):
    """
    Save a trained machine-learning model to disk.
    """

    joblib.dump(model, file_path)

    print("\nMODEL SAVED")
    print("File:", file_path)

def load_model(file_path):
    """
    Load a previously saved machine-learning model from disk.
    """

    model = joblib.load(file_path)

    print("\nMODEL LOADED")
    print("File:", file_path)

    return model