import pandas as pd


def load_and_validate_data(file_path):
    # Load data
    df = pd.read_excel(file_path)

    print("\nDATASET INFORMATION")
    print("First five rows:")
    print(df.head())

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nDataset shape:")
    print(df.shape)

    print("\nData types:")
    print(df.dtypes)

    # Check missing values
    print("\nMISSING VALUES")
    print(df.isnull().sum())

    # Check duplicate dates
    duplicate_dates = df["date"].duplicated().sum()

    print("\nDUPLICATE DATES")
    print(duplicate_dates)

    # Check date range
    print("\nDATE RANGE")
    print("Start:", df["date"].min())
    print("End:", df["date"].max())

    # Check missing dates
    expected_dates = pd.date_range(
        start=df["date"].min(),
        end=df["date"].max(),
        freq="D"
    )

    missing_dates = expected_dates.difference(df["date"])

    print("\nMISSING DATES")
    print(len(missing_dates))

    if len(missing_dates) > 0:
        print(missing_dates)
    else:
        print("No missing dates.")

    # Check negative rainfall values
    negative_rainfall = (df["rain_mm"] < 0).sum()

    print("\nNEGATIVE RAINFALL VALUES")
    print(negative_rainfall)

    return df