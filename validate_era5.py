import pandas as pd

df = pd.read_csv("era5_cleaned.csv", parse_dates=["time"])
print(df["time"].min(), df["time"].max())
print(df["time"].diff().value_counts().head())