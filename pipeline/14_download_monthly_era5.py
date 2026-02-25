"""Download monthly ERA5 reanalysis data via the CDS API.

Downloads four datasets from the Copernicus Climate Data Store:

1. Single-level monthly means (instantaneous climate variables):
   - Mean sea level pressure (MSLP)
   - 2 m temperature

2. Single-level monthly means (instantaneous wind variables):
   - 100 m U/V wind components

3. Single-level monthly means (accumulated flux variables):
   - Total precipitation
   - Snowfall
   - Surface solar radiation downwards

4. Pressure-level monthly means:
   - Geopotential at 500 hPa
   - U/V wind components at 250 and 500 hPa

Instantaneous and accumulated variables are split into separate requests
because the CDS API returns them as separate files inside a zip archive.
Keeping the groups homogeneous avoids silent data loss during extraction.

Output files are stored as NetCDF in data/downloads/era5/.

Prerequisites:
  - CDS API credentials in ~/.cdsapirc
    (register at https://cds.climate.copernicus.eu/)
  - `cdsapi` package installed
"""

# %%
import tempfile
import zipfile
from pathlib import Path

import cdsapi
import xarray as xr

from woe.paths import ProjPaths

paths = ProjPaths()
paths.ensure_directories()

client = cdsapi.Client()

# Spatial extent: [North, West, South, East]
BOUNDING_BOX = [80, -90, 20, 40]

YEARS = [str(y) for y in range(1940, 2025)]
MONTHS = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

# %%
def extract_if_zipped(path: Path) -> None:
    """If `path` is a zip archive, replace it with a merged NetCDF of all .nc files inside.

    The CDS API may return instant and accumulated variables in separate .nc
    files inside a zip even when download_format='unarchived' is requested.
    This function merges all .nc files so no variables are silently dropped.
    """
    if not zipfile.is_zipfile(path):
        return
    print(f"  Extracting zip archive at {path.name}...")
    with zipfile.ZipFile(path) as zf:
        nc_files = [n for n in zf.namelist() if n.endswith(".nc")]
        if not nc_files:
            raise RuntimeError(f"No .nc file found inside zip: {path}")
        if len(nc_files) == 1:
            path.write_bytes(zf.read(nc_files[0]))
            print(f"  Extracted {nc_files[0]}")
            return
        # Multiple nc files: extract all to a temp dir, merge, write back
        print(f"  Merging {len(nc_files)} nc files: {nc_files}")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            datasets = []
            for nc_name in nc_files:
                p = tmp_path / Path(nc_name).name
                p.write_bytes(zf.read(nc_name))
                datasets.append(xr.open_dataset(p))
            merged = xr.merge(datasets)
        merged.to_netcdf(path)
        print(f"  Merged and saved to {path.name}")


# %%
# Single-level group A: instantaneous climate variables (MSLP, T2m)
output_climate = paths.era5_sl_climate_file
print(f"Downloading single-level climate variables -> {output_climate}")

client.retrieve(
    "reanalysis-era5-single-levels-monthly-means",
    {
        "product_type": "monthly_averaged_reanalysis",
        "variable": [
            "mean_sea_level_pressure", # NAO Index
            "2m_temperature",          # Heating demand
        ],
        "year": YEARS,
        "month": MONTHS,
        "time": "00:00",
        "area": BOUNDING_BOX,
        "format": "netcdf",
        "download_format": "unarchived",
    },
    str(output_climate),
)

# %%
extract_if_zipped(output_climate)
print(f"Saved to {output_climate}")

# %%
# Single-level group B: instantaneous wind variables (100 m U/V)
output_wind_solar = paths.era5_sl_wind_solar_file
print(f"Downloading single-level wind variables -> {output_wind_solar}")

client.retrieve(
    "reanalysis-era5-single-levels-monthly-means",
    {
        "product_type": "monthly_averaged_reanalysis",
        "variable": [
            "100m_u_component_of_wind", # Wind power (Zonal)
            "100m_v_component_of_wind", # Wind power (Meridional)
        ],
        "year": YEARS,
        "month": MONTHS,
        "time": "00:00",
        "area": BOUNDING_BOX,
        "format": "netcdf",
        "download_format": "unarchived",
    },
    str(output_wind_solar),
)

# %%
extract_if_zipped(output_wind_solar)
print(f"Saved to {output_wind_solar}")

# %%
# Single-level group C: accumulated flux variables (tp, sf, ssrd)
# Kept separate from instantaneous variables because the CDS API returns
# instant and accumulated fields in different files within the zip archive.
output_accumulated = paths.era5_sl_accumulated_file
print(f"Downloading single-level accumulated variables -> {output_accumulated}")

client.retrieve(
    "reanalysis-era5-single-levels-monthly-means",
    {
        "product_type": "monthly_averaged_reanalysis",
        "variable": [
            "total_precipitation",               # Hydrology
            "snowfall",                          # Hydrology / Albedo
            "surface_solar_radiation_downwards", # Solar power
        ],
        "year": YEARS,
        "month": MONTHS,
        "time": "00:00",
        "area": BOUNDING_BOX,
        "format": "netcdf",
        "download_format": "unarchived",
    },
    str(output_accumulated),
)

# %%
extract_if_zipped(output_accumulated)
print(f"Saved to {output_accumulated}")

# %%
# Pressure-level monthly means: geopotential, U/V wind at 250 and 500 hPa
output_pressure = paths.era5_monthly_pressure_levels_file
print(f"Downloading pressure-level variables -> {output_pressure}")

client.retrieve(
    "reanalysis-era5-pressure-levels-monthly-means",
    {
        "product_type": "monthly_averaged_reanalysis",
        "variable": [
            "geopotential",        # 500 hPa ridges/troughs
            "u_component_of_wind", # 250 hPa jet stream speed
            "v_component_of_wind", # 250 hPa jet stream direction
        ],
        "pressure_level": ["250", "500"],
        "year": YEARS,
        "month": MONTHS,
        "time": "00:00",
        "area": BOUNDING_BOX,
        "format": "netcdf",
        "download_format": "unarchived",
    },
    str(output_pressure),
)

# %%
extract_if_zipped(output_pressure)
print(f"Saved to {output_pressure}")

# %%
print("Done.")
