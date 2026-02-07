import pandas as pd

IN_PATH = "data/processed/riyadh_demand_weather.csv"
OUT_PATH = "data/processed/riyadh_features.csv"

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date")

    # time features
    df["dayofweek"] = df["date"].dt.weekday
    df["month"] = df["date"].dt.month

    # lag features
    df["lag_1"] = df["demand"].shift(1)
    df["lag_7"] = df["demand"].shift(7)
    df["rolling_7"] = df["demand"].shift(1).rolling(7).mean()

    return df

def main():
    df = pd.read_csv(IN_PATH, parse_dates=["date"])
    df = build_features(df)

    df = df.dropna()
    df.to_csv(OUT_PATH, index=False)

    print(f"Saved {OUT_PATH}")

if __name__ == "__main__":
    main()
