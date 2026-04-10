"""Download PECD NUTS 1 capacity factors for Germany (Bundesländer).

Downloads hourly capacity factor ratios at NUTS 1 sub-country resolution for
solar PV and onshore wind from the Copernicus Climate Data Store (CDS).
Offshore wind is intentionally excluded: PECD does not provide sub-national
offshore aggregation, so offshore analysis keeps using the NUTS 0 data from
the pan-European download in script 35.

The CDS dataset returns data for all European regions at the requested NUTS
level; filtering to Germany (column prefix "DE") happens in the processing
step (script 51).

Dataset: sis-energy-derived-reanalysis

Outputs (in data/downloads/pecd/nuts1_de/):
  PECD_NUTS1_2024_solar_photovoltaic_power_generation_capacity_factor_ratio.zip
  PECD_NUTS1_2024_wind_power_generation_onshore_capacity_factor_ratio.zip

Requires a ~/.cdsapirc file with a valid CDS API key.
"""

import cdsapi
from tqdm import tqdm

from woe.paths import ProjPaths

paths = ProjPaths()

YEARS = [2024]  # back-catalogue (1979+) is included regardless

VARIABLES = [
    "solar_photovoltaic_power_generation",
    "wind_power_generation_onshore",
]

PRODUCT_TYPE = "capacity_factor_ratio"
NUTS_LEVEL = "1"

paths.pecd_nuts1_de_downloads_path.mkdir(parents=True, exist_ok=True)

client = cdsapi.Client()

jobs = [(year, variable) for year in YEARS for variable in VARIABLES]

with tqdm(jobs, desc="Downloading PECD NUTS 1", unit="request") as pbar:
    for year, variable in pbar:
        output_file = (
            paths.pecd_nuts1_de_downloads_path
            / f"PECD_NUTS1_{year}_{variable}_{PRODUCT_TYPE}.zip"
        )
        pbar.set_postfix({"file": output_file.name})

        if output_file.exists():
            tqdm.write(f"  Skipping {output_file.name} (already exists)")
            continue

        request = {
            "variable": variable,
            "spatial_aggregation": "sub_country_level",
            "nuts_level": NUTS_LEVEL,
            "energy_product_type": PRODUCT_TYPE,
            "temporal_aggregation": "hourly",
            "year": year,
            "format": "zip",
        }

        client.retrieve("sis-energy-derived-reanalysis", request, str(output_file))
        tqdm.write(f"  Downloaded {output_file.name}")
