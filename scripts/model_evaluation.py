import pandas as pd


def calculate_feature_importance(
    model,
    predictors
):
    """
    Calculate and display Random Forest feature importance.
    """

    importance = pd.DataFrame({
        "feature": predictors,
        "importance": model.feature_importances_
    })

    importance = importance.sort_values(
        by="importance",
        ascending=False
    ).reset_index(drop=True)

    print("\n" + "=" * 50)
    print("RANDOM FOREST FEATURE IMPORTANCE")
    print("=" * 50)

    print(
        importance.to_string(index=False)
    )

    return importance


def analyze_probability_distribution(
    y_true,
    probabilities,
    dataset_name="Dataset"
):
    """
    Summarize predicted probability distributions
    for observed dry-spell events and non-events.
    """

    analysis = pd.DataFrame({
        "actual": y_true,
        "probability": probabilities
    })

    event_probabilities = analysis.loc[
        analysis["actual"] == 1,
        "probability"
    ]

    non_event_probabilities = analysis.loc[
        analysis["actual"] == 0,
        "probability"
    ]

    print("\n" + "=" * 50)
    print("PROBABILITY DISTRIBUTION ANALYSIS")
    print("=" * 50)

    print("Dataset:", dataset_name)

    print("\nObserved dry-spell events")
    print(
        event_probabilities.describe().round(4)
    )

    print("\nObserved non-events")
    print(
        non_event_probabilities.describe().round(4)
    )

    print("\nSelected probability percentiles")

    print(
        pd.DataFrame({
            "event_probability": (
                event_probabilities
                .quantile([0.25, 0.50, 0.75, 0.90])
                .round(4)
            ),
            "non_event_probability": (
                non_event_probabilities
                .quantile([0.25, 0.50, 0.75, 0.90])
                .round(4)
            )
        })
    )

    return analysis