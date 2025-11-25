import xarray as xr
import pandas as pd
import glob
import numpy as np

files = sorted(glob.glob("era5_*.nc"))
print(f"Found {len(files)} files")

datasets = []

for f in files:
    print(f"Loading {f}...")
    ds = xr.open_dataset(f)

    # Normalize time coordinate
    if "valid_time" in ds.coords and "time" not in ds.coords:
        ds = ds.rename({"valid_time": "time"})

    # ERA5 variables
    u10 = ds['u10']
    v10 = ds['v10']
    t2m = ds['t2m'] - 273.15  # Kelvin → Celsius
    msl = ds['msl'] / 100.0   # Pa → hPa

    windspeed = np.sqrt(u10**2 + v10**2)

    ds_clean = xr.Dataset({
        'windspeed': windspeed,
        'temperature_C': t2m,
        'pressure_hPa': msl
    }, coords={'time': ds['time']})

    datasets.append(ds_clean)

# Concatenate with join override
merged = xr.concat(datasets, dim='time', join="override")

# Resample to 3-hourly (ERA5 is already 3-hourly, but ensures consistency)
merged = merged.resample(time="3H").mean()

# Export to CSV
df = merged.to_dataframe().reset_index()
df.to_csv("era5_cleaned.csv", index=False)

print("✅ Exported merged dataset to era5_cleaned.csv")