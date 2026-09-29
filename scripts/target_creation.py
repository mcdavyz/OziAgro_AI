import pandas as pd


def create_dry_spell_target(
    df,
    dry_day_threshold=1.0,
    dry_spell_length=7,
    prediction_horizon=14
):
    # Define dry days
    df["dry_day"] = (
        df["rain_mm"] < dry_day_threshold
    ).astype(int)

    # Initialize target
    df["future_dry_spell"] = None

    # Identify valid prediction origins
    valid_indices = df.index[
        df["valid_prediction_period"] == 1
    ]

    # Determine whether a 7-day dry spell occurs
    # within the following 14 days
    for i in valid_indices:

        future_days = df["dry_day"].iloc[
            i + 1 : i + 1 + prediction_horizon
        ].to_numpy()

        consecutive_dry = 0
        dry_spell_found = False

        for day in future_days:

            if day == 1:
                consecutive_dry += 1
            else:
                consecutive_dry = 0

            if consecutive_dry >= dry_spell_length:
                dry_spell_found = True
                break

        df.loc[i, "future_dry_spell"] = (
            1 if dry_spell_found else 0
        )

    # Convert target to numeric
    df["future_dry_spell"] = pd.to_numeric(
        df["future_dry_spell"],
        errors="coerce"
    )

    # Report target definition
    print("\nDRY-SPELL TARGET DEFINITION")
    print("Dry-day threshold:", dry_day_threshold, "mm")
    print(
        "Dry-spell length:",
        dry_spell_length,
        "consecutive days"
    )
    print(
        "Prediction horizon:",
        prediction_horizon,
        "days"
    )

    print("\nTARGET COUNTS")
    print(
        df["future_dry_spell"]
        .value_counts(dropna=False)
    )

    print("\nTARGET PERCENTAGES")
    print(
        df["future_dry_spell"]
        .value_counts(
            normalize=True,
            dropna=True
        )
        .mul(100)
        .round(2)
    )

    print("\nFIRST TARGET OBSERVATIONS")
    print(
        df[
            [
                "date",
                "doy",
                "rain_mm",
                "dry_day",
                "future_dry_spell"
            ]
        ].head(80)
    )

    return df


def verify_dry_spell_target(
    df,
    prediction_horizon=14
):
    # Identify positive dry-spell cases
    positive_cases = df.index[
        df["future_dry_spell"] == 1
    ]

    print("\nNUMBER OF POSITIVE CASES")
    print(len(positive_cases))

    print("\nFIRST 10 POSITIVE CASES")

    for i in positive_cases[:10]:

        prediction_date = df.loc[i, "date"]

        future_window = df["dry_day"].iloc[
            i + 1 : i + 1 + prediction_horizon
        ]

        longest_dry_spell = 0
        current_dry_spell = 0

        for day in future_window:

            if day == 1:
                current_dry_spell += 1
                longest_dry_spell = max(
                    longest_dry_spell,
                    current_dry_spell
                )
            else:
                current_dry_spell = 0

        print(
            prediction_date.date(),
            "→ longest future dry spell:",
            longest_dry_spell,
            "days"
        )

    # Target distribution by year
    print("\nTARGET DISTRIBUTION BY YEAR")

    yearly_target = (
        df[df["valid_prediction_period"] == 1]
        .groupby("year")["future_dry_spell"]
        .agg(
            total_predictions="count",
            dry_spell_events="sum"
        )
    )

    yearly_target["event_rate_%"] = (
        yearly_target["dry_spell_events"]
        / yearly_target["total_predictions"]
        * 100
    ).round(2)

    print(yearly_target)

    return df