import os
import requests
import pandas as pd
from datetime import datetime

def fetch_owm_forecast(lat: float, lon: float, station_name: str, api_key: str) -> pd.DataFrame:
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()

    rows = []
    for entry in data.get("list", []):
        timestamp = datetime.fromtimestamp(entry["dt"])
        wind_speed = entry["wind"]["speed"]
        wind_direction = entry["wind"]["deg"]
        air_temperature = entry["main"]["temp"]

        rows.append({
            "timestamp": timestamp,
            "wind_speed": wind_speed,
            "wind_direction": wind_direction,
            "air_temperature": air_temperature,
            "station": station_name
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


if __name__ == "__main__":
    api_key = "68f5a9aa11de1243f2a4cb1ea3ad48a0"

    # Fetch forecasts
    rochdale_df = fetch_owm_forecast(53.609, -2.179, "rochdale", api_key)
    crosby_df   = fetch_owm_forecast(53.4778, -3.0333, "crosby", api_key)

    # Combine both
    combined_df = pd.concat([rochdale_df, crosby_df], ignore_index=True)

    # ✅ Ensure folder exists before saving
    os.makedirs("openweathermap", exist_ok=True)

    # ✅ Save to CSV
    combined_df.to_csv("openweathermap/forecast_combined.csv", index=False)

    print("✅ OWM forecast data saved to openweathermap/forecast_combined.csv")