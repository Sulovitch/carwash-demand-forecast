import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import joblib

DATA_PATH = "data/processed/riyadh_features.csv"
MODEL_PATH = "models/rf_demand.pkl"

def main():
    df = pd.read_csv(DATA_PATH)

    features = [
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

    X = df[features]
    y = df["demand"]

    split = int(len(df) * 0.8)

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    print("Test MAE:", mae)

    joblib.dump(model, MODEL_PATH)
    print("Model saved to", MODEL_PATH)

if __name__ == "__main__":
    main()
