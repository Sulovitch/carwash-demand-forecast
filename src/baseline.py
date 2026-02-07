import pandas as pd
from sklearn.metrics import mean_absolute_error

DATA_PATH = "data/processed/riyadh_features.csv"

df = pd.read_csv(DATA_PATH)

y_true = df["demand"]

mae_lag1 = mean_absolute_error(y_true, df["lag_1"])
mae_lag7 = mean_absolute_error(y_true, df["lag_7"])

print("Baseline lag-1 MAE:", mae_lag1)
print("Baseline lag-7 MAE:", mae_lag7)
