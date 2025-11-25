import cdsapi

# Create a CDS API client
c = cdsapi.Client()

# Request a tiny sample of ERA5 data to test your setup
c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'variable': '2m_temperature',
        'year': '2020',
        'month': '01',
        'day': '01',
        'time': '12:00',
        'format': 'netcdf',
    },
    'test_era5.nc'
)

print("Download complete: test_era5.nc")