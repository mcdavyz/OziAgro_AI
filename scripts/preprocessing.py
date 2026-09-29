def define_prediction_period(
    df,
    season_start_doy=60,
    season_end_doy=323,
    prediction_horizon=14
):
    # Define the rainfed rice modelling period
    df["rice_model_period"] = (
        df["doy"] >= season_start_doy
    ).astype(int)

    print("\nRAINFED RICE MODEL WINDOW")
    print("Season start DOY:", season_start_doy)

    print("\nMODEL PERIOD COUNTS")
    print(df["rice_model_period"].value_counts())

    print("\nFIRST DATES INCLUDED IN MODEL PERIOD")
    print(
        df.loc[
            df["rice_model_period"] == 1,
            ["date", "doy", "rain_mm"]
        ].head()
    )

    # Define valid prediction period
    # Prediction origins must leave enough days for the
    # 14-day future prediction horizon.
    last_prediction_doy = (
        season_end_doy - prediction_horizon
    )

    df["valid_prediction_period"] = (
        (df["doy"] >= season_start_doy) &
        (df["doy"] <= last_prediction_doy)
    ).astype(int)

    print("\nVALID PREDICTION WINDOW")
    print("Season start DOY:", season_start_doy)
    print("Season end DOY:", season_end_doy)
    print("Prediction horizon:", prediction_horizon, "days")
    print("Last prediction DOY:", last_prediction_doy)

    print("\nVALID PREDICTION PERIOD COUNTS")
    print(df["valid_prediction_period"].value_counts())

    print("\nFIRST VALID PREDICTION DATES")
    print(
        df.loc[
            df["valid_prediction_period"] == 1,
            ["date", "doy", "rain_mm"]
        ].head()
    )

    print("\nLAST VALID PREDICTION DATES")
    print(
        df.loc[
            df["valid_prediction_period"] == 1,
            ["date", "doy", "rain_mm"]
        ].tail()
    )

    return df