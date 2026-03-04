"""Download PECD (Pan-European Climate Database) capacity factors and power generation.

Downloads hourly capacity factor ratios and power generation for solar PV,
onshore/offshore wind, and hydro from the Copernicus Climate Data Store (CDS)
for the years 2023 and 2024. Requests are split by variable and product type
— one CDS request per combination per year — to reduce cost. Country-level
spatial aggregation is used for most variables; offshore wind uses
"maritime_country_level". Hydro capacity factor ratios are not available in
this dataset and are excluded. Data for all European countries/regions is
returned.

Dataset: sis-energy-derived-reanalysis

Outputs (in data/downloads/pecd/):
  One ZIP per (variable, product_type, year):
  e.g. PECD_2023_solar_photovoltaic_power_generation_capacity_factor_ratio.zip
  16 files total (3 variables × 2 product types + 2 hydro × 1 product type, × 2 years).

Requires a ~/.cdsapirc file with a valid CDS API key.
"""

import cdsapi
import requests
from tqdm import tqdm

from woe.paths import ProjPaths

paths = ProjPaths()

YEARS = [2024] # this downloads the full data starting 1980 anyways

VARIABLES = [
    "solar_photovoltaic_power_generation",
    "wind_power_generation_onshore",
    "wind_power_generation_offshore",
]

PRODUCT_TYPES = ["capacity_factor_ratio"]

paths.pecd_downloads_path.mkdir(parents=True, exist_ok=True)

client = cdsapi.Client()

jobs = [
    (year, variable, product_type)
    for year in YEARS
    for variable in VARIABLES
    for product_type in PRODUCT_TYPES
]

with tqdm(jobs, desc="Downloading PECD", unit="request") as pbar:
    for year, variable, product_type in pbar:
        output_file = paths.pecd_downloads_path / f"PECD_{year}_{variable}_{product_type}.zip"

        pbar.set_postfix({"file": output_file.name})

        if output_file.exists():
            tqdm.write(f"  Skipping {output_file.name} (already exists)")
            continue

        if variable == "wind_power_generation_offshore":
            spatial_aggregation = "maritime_country_level"
        else:
            spatial_aggregation = "country_level"

        request = {
            "variable": variable,
            "spatial_aggregation": spatial_aggregation,
            "energy_product_type": product_type,
            "temporal_aggregation": "hourly",
            "year": year,
            "format": "zip",
        }

        try:
            client.retrieve("sis-energy-derived-reanalysis", request, str(output_file))
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 400:
                tqdm.write(f"  Skipping {output_file.name} (invalid combination: {variable} + {product_type})")
                continue
            raise
        tqdm.write(f"  Downloaded {output_file.name}")
