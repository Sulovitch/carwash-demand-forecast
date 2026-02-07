import requests
import pandas as pd

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

LAT = 24.7136
LON = 46.6753

PARAMS = {
    "latitude": LAT,
    "longitude": LON,
    "start_date": "2023-01-01",
    "end_date": "2025-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "windspeed_10m_max",
        "relative_humidity_2m_max",
        "surface_pressure_mean",
    ],
    "timezone": "Asia/Riyadh",
}

def fetch_weather():
    r = requests.get(BASE_URL, params=PARAMS)
    r.raise_for_status()
    data = r.json()["daily"]

    df = pd.DataFrame(data)
    df.rename(
        columns={
            "time": "date",
            "temperature_2m_max": "temp_max",
            "temperature_2m_min": "temp_min",
            "precipitation_sum": "rain",
            "windspeed_10m_max": "wind",
            "relative_humidity_2m_max": "humidity",
            "surface_pressure_mean": "pressure",
        },
        inplace=True,
    )

    df.to_csv("data/raw/riyadh_weather.csv", index=False)
    print("Saved data/raw/riyadh_weather.csv")

if __name__ == "__main__":
    fetch_weather()
