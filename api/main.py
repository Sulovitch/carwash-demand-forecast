from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import timedelta
import requests
import logging
from fastapi.responses import HTMLResponse


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("forecast-api")

MODEL_PATH = "models/rf_demand.pkl"
DATA_PATH = "data/processed/riyadh_features.csv"

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
LAT = 24.7136
LON = 46.6753
def fetch_weather_forecast():
    params = {
        "latitude": LAT,
        "longitude": LON,
        "daily": [
            "temperature_2m_max",
            "precipitation_sum",
            "windspeed_10m_max",
            "relative_humidity_2m_max",
            "surface_pressure_mean",
        ],
        "timezone": "Asia/Riyadh",
    }

    r = requests.get(WEATHER_URL, params=params, timeout=10)
    r.raise_for_status()

    daily = r.json()["daily"]

    df = pd.DataFrame(daily)
    df.rename(
        columns={
            "time": "date",
            "temperature_2m_max": "temp_max",
            "precipitation_sum": "rain",
            "windspeed_10m_max": "wind",
            "relative_humidity_2m_max": "humidity",
            "surface_pressure_mean": "pressure",
        },
        inplace=True,
    )

    df["date"] = pd.to_datetime(df["date"])
    return df.head(7)


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

app = FastAPI(title="Car Wash Demand Forecast API")

@app.get("/", response_class=HTMLResponse)
def dashboard():
    weather_future = fetch_weather_forecast()
    history = df.iloc[-1:].copy()
    recent = df.tail(7).copy()
    last_date = pd.to_datetime(history["date"].values[0])

    rows = ""

    for i in range(7):
        row = history.copy()

        row["lag_1"] = recent.iloc[-1]["demand"]
        row["lag_7"] = recent.iloc[0]["demand"]
        row["rolling_7"] = recent["demand"].mean()

        row["temp_max"] = weather_future.iloc[i]["temp_max"]
        row["rain"] = weather_future.iloc[i]["rain"]
        row["wind"] = weather_future.iloc[i]["wind"]
        row["humidity"] = weather_future.iloc[i]["humidity"]
        row["pressure"] = weather_future.iloc[i]["pressure"]

        day = int(weather_future.iloc[i]["date"].weekday())
        row["dayofweek"] = day
        row["month"] = int(weather_future.iloc[i]["date"].month)
        row["is_weekend"] = int(day in [4, 5])

        y_pred = model.predict(row[FEATURES])[0]

        temp = float(row["temp_max"].values[0])
        rain = float(row["rain"].values[0])
        wind = float(row["wind"].values[0])
        humidity = float(row["humidity"].values[0])
        pred = int(round(y_pred))


        rows += f"""
        <tr>
            <td>{weather_future.iloc[i]["date"].date()}</td>
            <td>{temp:.1f}°C</td>
            <td>{rain} mm</td>
            <td>{wind} km/h</td>
            <td>{humidity}%</td>
            <td><b>{pred}</b></td>
        </tr>
        """

        new_row = row.copy()
        new_row["demand"] = y_pred
        new_row["date"] = last_date + timedelta(days=i + 1)

        recent = pd.concat([recent.iloc[1:], new_row], ignore_index=True)
        history = new_row

    html = f"""
    <html>
    <head>
        <title>Riyadh Car Wash Forecast</title>
        <style>
            body {{ font-family: Arial; padding: 30px; background:#f4f6f8; }}
            table {{ border-collapse: collapse; width: 100%; background:white; }}
            th, td {{ border: 1px solid #ddd; padding: 10px; text-align:center; }}
            th {{ background:#222; color:white; }}
            h1 {{ text-align:center; }}
        </style>
    </head>
    <body>
        <h1>🚗 Riyadh Car Wash Demand Forecast (Next 7 Days)</h1>
        <table>
            <tr>
                <th>Date</th>
                <th>Temp</th>
                <th>Rain</th>
                <th>Wind</th>
                <th>Humidity</th>
                <th>Predicted Demand</th>
            </tr>
            {rows}
        </table>
    </body>
    </html>
    """

    return HTMLResponse(content=html)


model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH, parse_dates=["date"])

class ForecastRequest(BaseModel):
    city: str = "Riyadh"

@app.post("/forecast")
def forecast(req: ForecastRequest):
    
    weather_future = fetch_weather_forecast()

    if len(weather_future) < 7:
     raise RuntimeError("Weather API did not return 7 days")

    history = df.iloc[-1:].copy()
    recent = df.tail(7).copy()

    preds = []
    last_date = pd.to_datetime(history["date"].values[0])


    for i in range(7):
        row = history.copy()

        row["lag_1"] = recent.iloc[-1]["demand"]
        row["lag_7"] = recent.iloc[0]["demand"]
        row["rolling_7"] = recent["demand"].mean()

        row["temp_max"] = weather_future.iloc[i]["temp_max"]
        row["rain"] = weather_future.iloc[i]["rain"]
        row["wind"] = weather_future.iloc[i]["wind"]
        row["humidity"] = weather_future.iloc[i]["humidity"]
        row["pressure"] = weather_future.iloc[i]["pressure"]


        day = int(weather_future.iloc[i]["date"].weekday())
        row["dayofweek"] = day
        row["month"] = int(weather_future.iloc[i]["date"].month)
        row["is_weekend"] = int(day in [4, 5])



        y_pred = model.predict(row[FEATURES])[0]

        preds.append({
        "date": str(weather_future.iloc[i]["date"].date()),
        "temp_max": float(row["temp_max"]),
        "rain": float(row["rain"]),
        "wind": float(row["wind"]),
        "humidity": float(row["humidity"]),
        "predicted_demand": int(round(y_pred))
    })


        new_row = row.copy()
        new_row["demand"] = y_pred
        new_row["date"] = last_date + timedelta(days=i + 1)


        recent = pd.concat([recent.iloc[1:], new_row], ignore_index=True)
        history = new_row

    logger.info(f"Forecast generated: {preds}")


    return {"city": req.city, "forecast": preds}

