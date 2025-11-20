"""
midas_loader.py
Generalised loader for MIDAS Open hourly weather observation data (qc-version-1).
"""

import pandas as pd
import glob
import os

def load_midas_data(station_folder: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(station_folder, "*.csv"))
    df_list = []

    for f in files:
        # Skip metadata + 'data' marker line
        temp_df = pd.read_csv(
            f,
            skiprows=283,   # skip metadata block + 'data' line
            sep=",",        # comma-separated
            header=0,       # first row after skiprows is header
            low_memory=False
        )

        # Drop empty rows/columns
        temp_df = temp_df.dropna(axis=1, how="all")
        temp_df = temp_df.dropna(how="all")

        # Rename timestamp
        if "ob_time" in temp_df.columns:
            temp_df.rename(columns={"ob_time": "timestamp"}, inplace=True)
            temp_df["timestamp"] = pd.to_datetime(temp_df["timestamp"], errors="coerce")

        # Keep only useful variables
        cols = [c for c in ["timestamp", "wind_speed", "wind_direction", "air_temperature", "msl_pressure"] if c in temp_df.columns]
        temp_df = temp_df[cols]

        # ➡️ Add station name correctly
        station_folder_name = os.path.basename(os.path.dirname(os.path.dirname(f)))   # e.g. '01125_rochdale'
        if "_" in station_folder_name:
            station_name = station_folder_name.split("_")[-1]        # 'rochdale' or 'crosby'
        else:
            station_name = station_folder_name                       # fallback
        temp_df["station"] = station_name

        # Summary printout
        print(f"📂 File: {os.path.basename(f)}")
        print(f"   ➡ Columns kept: {list(temp_df.columns)}")
        print(f"   ➡ Rows kept: {len(temp_df)}")

        # Append to list
        df_list.append(temp_df)

    # Concatenate all years
    if df_list:
        df = pd.concat(df_list, ignore_index=True)
    else:
        df = pd.DataFrame()

    # Drop rows with missing timestamps
    if "timestamp" in df.columns:
        df = df.dropna(subset=["timestamp"])
        df = df.sort_values("timestamp")
        df.reset_index(drop=True, inplace=True)

    return df


def load_multiple_stations(station_folders: list) -> pd.DataFrame:
    """Load and combine multiple station datasets into one DataFrame."""
    all_data = []
    for folder in station_folders:
        df = load_midas_data(folder)
        all_data.append(df)
    combined = pd.concat(all_data, ignore_index=True)
    return combined


if __name__ == "__main__":
    # Define station paths
    rochdale_path = r"C:\Users\SURFACE\OneDrive - University of Bolton\Assessments\7006\DISSERTATION\Wind Codes\data\midas\01125_rochdale\qc-version-1"
    crosby_path   = r"C:\Users\SURFACE\OneDrive - University of Bolton\Assessments\7006\DISSERTATION\Wind Codes\data\midas\17309_crosby\qc-version-1"

    # Load both stations
    combined_df = load_multiple_stations([rochdale_path, crosby_path])

    print("✅ Combined MIDAS data loaded")
    print(combined_df.head())
    print(f"Total rows: {len(combined_df)}")
    print(f"Stations included: {combined_df['station'].unique()}")

    combined_df.to_csv("data/midas_combined.csv", index=False)