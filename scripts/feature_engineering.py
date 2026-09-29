import numpy as np


def create_features(
    df,
    dry_day_threshold=1.0
):
    # Calculate current consecutive dry days
    consecutive_dry = []
    current_count = 0

    for dry in df["dry_day"]:

        if dry == 1:
            current_count += 1
        else:
            current_count = 0

        consecutive_dry.append(current_count)

    df["consecutive_dry_days"] = consecutive_dry

    print("\nCURRENT DRY-SPELL STATE")
    print(
         df[
        [
            "date",
            "doy",
            "rain_mm",
            "dry_day",
            "consecutive_dry_days"
        ]
    ].head(40)
    )

    print("\nMAXIMUM HISTORICAL CONSECUTIVE DRY DAYS")
    print(df["consecutive_dry_days"].max())

    # Create lagged rainfall predictors
    rainfall_windows = [1, 3, 5, 7, 14, 30]

    for window in rainfall_windows:
        df[f"rainfall_{window}d"] = (
            df["rain_mm"]
            .rolling(window=window)
            .sum()
            .shift(1)
        )

    print("\nPAST RAINFALL PREDICTORS")
    print(
        df[
            [
                "date",
                "rain_mm",
                "rainfall_1d",
                "rainfall_3d",
                "rainfall_5d",
                "rainfall_7d",
                "rainfall_14d",
                "rainfall_30d"
            ]
        ].head(40)
    )

    # Create recent rainy-day and dry-day counts
    day_count_windows = [7, 14, 30]

    for window in day_count_windows:

        df[f"rainy_days_{window}d"] = (
            (df["rain_mm"] >= dry_day_threshold)
            .rolling(window=window)
            .sum()
            .shift(1)
        )

        df[f"dry_days_{window}d"] = (
            df["dry_day"]
            .rolling(window=window)
            .sum()
            .shift(1)
        )

    print("\nRECENT RAINY-DAY AND DRY-DAY COUNTS")
    print(
        df[
            [
                "date",
                "rain_mm",
                "rainy_days_7d",
                "dry_days_7d",
                "rainy_days_14d",
                "dry_days_14d",
                "rainy_days_30d",
                "dry_days_30d"
            ]
        ].head(40)
    )

    # Create cyclical seasonal timing features
    df["doy_sin"] = np.sin(
        2 * np.pi * df["doy"] / 365.25
    )

    df["doy_cos"] = np.cos(
        2 * np.pi * df["doy"] / 365.25
    )

    print("\nSEASONAL TIMING FEATURES")
    print(
        df[
            [
                "date",
                "doy",
                "doy_sin",
                "doy_cos"
            ]
        ].iloc[55:70]
    )

    return df