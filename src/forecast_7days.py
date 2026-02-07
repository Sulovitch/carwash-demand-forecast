import pandas as pd
import joblib
from datetime import timedelta

MODEL_PATH = "models/rf_demand.pkl"
DATA_PATH = "data/processed/riyadh_features.csv"

FEATURES = [
    "temp_max",
    "rain",
    "wind",
    "humidity",
    "pressure",
    "is_weekend",
    "dayofweek",
    "month",
    "lag_1",
    "lag_7",
    "rolling_7",
]

def forecast_next_7_days():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    model = joblib.load(MODEL_PATH)

    history = df.iloc[-1:].copy()
    recent = df.tail(7).copy()

    preds = []

    last_date = history["date"].values[0]

    for i in range(7):
        row = history.copy()

        row["lag_1"] = recent.iloc[-1]["demand"]
        row["lag_7"] = recent.iloc[0]["demand"]
        row["rolling_7"] = recent["demand"].mean()

        y_pred = model.predict(row[FEATURES])[0]

        preds.append(int(round(y_pred)))

        new_row = row.copy()
        new_row["demand"] = y_pred
        new_row["date"] = pd.to_datetime(last_date) + timedelta(days=i + 1)

        recent = pd.concat([recent.iloc[1:], new_row], ignore_index=True)
        history = new_row

    return preds

if __name__ == "__main__":
    forecasts = forecast_next_7_days()
    print("Next 7 days forecast:", forecasts)
