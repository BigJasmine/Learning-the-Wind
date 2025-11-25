import cdsapi
import calendar
import time

c = cdsapi.Client()

# Resume from Nov 2019 to 2024
years_monthly = list(range(2019, 2025))
months = [f"{m:02d}" for m in range(1, 13)]

def download_month(year, month, target_file, days):
    try:
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': [
                    '10m_u_component_of_wind',
                    '10m_v_component_of_wind',
                    '2m_temperature',
                    'mean_sea_level_pressure',
                ],
                'year': str(year),
                'month': month,
                'day': days,
                'time': ['00:00', '03:00', '06:00', '09:00',
                         '12:00', '15:00', '18:00', '21:00'],
                'area': [56, -6, 53, 0],
                'format': 'netcdf',
            },
            target_file
        )
        print(f"✅ Finished {target_file}")
        return True
    except Exception as e:
        print(f"⚠️ Error for {year}-{month}: {e}")
        return False

for year in years_monthly:
    for month in months:
        # Skip months before Nov 2019
        if year == 2019 and int(month) < 11:
            continue

        num_days = calendar.monthrange(year, int(month))[1]
        days = [f"{d:02d}" for d in range(1, num_days + 1)]
        target_file = f"era5_{year}_{month}.nc"

        print(f"▶ Starting {year}-{month} with {num_days} days...")

        success = download_month(year, month, target_file, days)

        if not success:
            print(f"🔄 Retrying {year}-{month} after short pause...")
            time.sleep(10)
            success = download_month(year, month, target_file, days)

        if not success:
            print(f"✂️ Splitting {year}-{month} into halves...")
            days_first_half = [f"{d:02d}" for d in range(1, 16)]
            days_second_half = [f"{d:02d}" for d in range(16, num_days + 1)]

            target_file_a = f"era5_{year}_{month}_a.nc"
            target_file_b = f"era5_{year}_{month}_b.nc"

            download_month(year, month, target_file_a, days_first_half)
            download_month(year, month, target_file_b, days_second_half)