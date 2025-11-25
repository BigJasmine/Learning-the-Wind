import pandas as pd
import numpy as np

# Load deduplicated dataset
df = pd.read_csv("era5_cleaned.csv", parse_dates=["time"])

# --- Wind direction ---
df["wind_direction_deg"] = np.degrees(np.arctan2(df["windspeed"], df["windspeed"]))  # placeholder
# Better: if you still have u10 and v10, use atan2(v10, u10)

# --- Rolling averages ---
df["windspeed_roll24h"] = df["windspeed"].rolling(window=8).mean()   # 8 steps = 24h (3h each)
df["temperature_roll7d"] = df["temperature_C"].rolling(window=56).mean()  # 56 steps = 7 days

# --- Seasonal flags ---
df["month"] = df["time"].dt.month
df["season"] = df["month"] % 12 // 3 + 1   # 1=Winter, 2=Spring, 3=Summer, 4=Autumn
df["day_of_week"] = df["time"].dt.dayofweek

# --- Interaction term ---
df["windspeed_cubed"] = df["windspeed"] ** 3

# --- Lag feature ---
df["windspeed_lag1"] = df["windspeed"].shift(1)

# Save enriched dataset
df.to_csv("era5_features.csv", index=False)

print("✅ Feature-engineered dataset saved as era5_features.csv")