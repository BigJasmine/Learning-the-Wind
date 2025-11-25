import pandas as pd

# Load the cleaned dataset
df = pd.read_csv("era5_cleaned.csv", parse_dates=["time"])

# --- Step 1: Remove duplicates ---
df = df.drop_duplicates(subset="time", keep="first").sort_values("time")

print("✅ Deduplicated dataset")
print("Start:", df["time"].min())
print("End:", df["time"].max())
print("Total rows:", len(df))

# --- Step 2: Check for missing 3-hourly steps ---
expected_range = pd.date_range(
    start=df["time"].min(),
    end=df["time"].max(),
    freq="3H"
)

missing = expected_range.difference(df["time"])

if len(missing) == 0:
    print("🎉 No missing 3-hourly steps — dataset is continuous!")
else:
    print(f"⚠️ Found {len(missing)} missing timestamps:")
    print(missing[:20])  # show first 20 gaps
    # Save full list to CSV for inspection
    pd.DataFrame(missing, columns=["missing_time"]).to_csv("missing_timestamps.csv", index=False)
    print("Full list saved to missing_timestamps.csv")