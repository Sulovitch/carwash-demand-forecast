import pandas as pd
import numpy as np

RAW_PATH = "data/raw/riyadh_weather.csv"
OUT_PATH = "data/processed/riyadh_demand_weather.csv"

np.random.seed(42)

def simulate_demand(df: pd.DataFrame) -> pd.DataFrame:
    base = 80

    demand = []

    for _, row in df.iterrows():
        d = base

        # weekend effect
        if row["date"].weekday() in [4, 5]:
            d += 25

        # temperature effect
        if row["temp_max"] > 42:
            d -= 20
        elif row["temp_max"] < 30:
            d += 10

        # rain effect
        if row["rain"] > 0:
            d += 15

        # wind / dust proxy
        if row["wind"] > 30:
            d -= 10

        # random noise
        d += np.random.normal(0, 5)

        demand.append(max(5, int(d)))

    df["demand"] = demand
    return df

def main():
    df = pd.read_csv(RAW_PATH, parse_dates=["date"])
    df = simulate_demand(df)

    df["is_weekend"] = df["date"].dt.weekday.isin([4, 5]).astype(int)

    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {OUT_PATH}")

if __name__ == "__main__":
    main()
