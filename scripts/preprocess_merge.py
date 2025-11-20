"""
preprocess_merge.py
Resample MIDAS hourly observations to 3-hourly and merge with OWM forecasts.
"""

import os
import pandas as pd

def resample_midas_to_3hourly(midas_df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample MIDAS hourly data to 3-hourly resolution.
    Aggregates wind_speed, wind_direction, air_temperature (and msl_pressure if present).
    """
    midas_df["timestamp"] = pd.to_datetime(midas_df["timestamp"])
    midas_df = midas_df.set_index("timestamp")

    # Only resample numeric columns
    numeric_cols = ["wind_speed", "wind_direction", "air_temperature"]
    if "msl_pressure" in midas_df.columns:
        numeric_cols.append("msl_pressure")

    resampled = midas_df[numeric_cols].resample("3h").mean().reset_index()

    # Reattach station info (assumes single station per call)
    resampled["station"] = midas_df["station"].iloc[0]

    return resampled


def merge_midas_owm(midas_df: pd.DataFrame, owm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge resampled MIDAS observations with OWM forecasts.
    Aligns on timestamp + station.
    """
    # Resample MIDAS
    midas_resampled = resample_midas_to_3hourly(midas_df)

    # Merge on timestamp + station
    merged = pd.merge(
        midas_resampled,
        owm_df,
        on=["timestamp", "station"],
        suffixes=("_obs", "_fc")
    )

    return merged


if __name__ == "__main__":
    # Paths to input files
    midas_path = "data/midas_combined.csv"                # output from midas_loader.py
    owm_path   = "openweathermap/forecast_combined.csv"   # output from owm_loader.py

    # Load data
    midas_df = pd.read_csv(midas_path, parse_dates=["timestamp"])
    owm_df   = pd.read_csv(owm_path, parse_dates=["timestamp"])

    # Merge
    merged_df = merge_midas_owm(midas_df, owm_df)

    print("✅ MIDAS + OWM merged dataset created")
    print(merged_df.head())
    print(f"Total rows: {len(merged_df)}")
    print(f"Stations included: {merged_df['station'].unique()}")

    # Ensure output folder exists
    os.makedirs("data/merged", exist_ok=True)

    # Save merged dataset
    merged_df.to_csv("data/merged/midas_owm_3hourly.csv", index=False)
    print("💾 Saved merged dataset to data/merged/midas_owm_3hourly.csv")