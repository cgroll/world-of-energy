"""Process PECD NUTS 1 capacity factors for Germany.

Reads the ZIP files downloaded by 50_download_pecd_nuts1_de.py, filters to
German Bundesländer (NUTS 1 codes starting with "DE"), and combines them into
a single wide-format parquet file.

Output index: time (hourly DatetimeIndex)
Output columns: MultiIndex of (variable, region) where region is a NUTS 1
code (DE1..DEG).

Output:
  data/processed/pecd/pecd_nuts1_de.parquet
"""

# %%
import io
import re
import zipfile

import pandas as pd

from woe.paths import ProjPaths

paths = ProjPaths()

# %%
zip_files = sorted(paths.pecd_nuts1_de_downloads_path.glob("PECD_NUTS1_*.zip"))
print(f"Found {len(zip_files)} ZIP files in {paths.pecd_nuts1_de_downloads_path}")


# %%
def load_nuts1_zip(zip_path, variable: str) -> pd.DataFrame:
    """Stream one PECD NUTS 1 ZIP and return a wide DataFrame for German regions only.

    The full CSV contains hourly data for every European NUTS 1 region across
    the PECD back-catalogue and is too large to materialize in memory. We
    instead locate the header line, then re-open the CSV with `usecols` so
    pandas only parses Date + DE* columns.
    """
    with zipfile.ZipFile(zip_path) as z:
        csv_name = z.namelist()[0]

        # Pass 1: find header row index by streaming lines until one starts with "Date,"
        with z.open(csv_name) as f:
            text = io.TextIOWrapper(f, encoding="utf-8")
            header_idx = None
            for i, line in enumerate(text):
                if line.startswith("Date,"):
                    header_idx = i
                    break
            if header_idx is None:
                raise RuntimeError(f"No 'Date,' header line found in {zip_path.name}")

        # Pass 2: stream the full CSV but only parse Date + DE* columns
        with z.open(csv_name) as f:
            df = pd.read_csv(
                f,
                skiprows=header_idx,
                parse_dates=["Date"],
                index_col="Date",
                usecols=lambda c: c == "Date" or c.startswith("DE"),
            )

    df.index.name = "time"
    if df.index.duplicated().any():
        n_dups = int(df.index.duplicated().sum())
        print(f"  Dropping {n_dups} duplicate timestamps in {zip_path.name}")
        df = df[~df.index.duplicated(keep="first")]

    if df.shape[1] == 0:
        raise RuntimeError(f"No DE* columns found in {zip_path.name}")
    df.columns = pd.MultiIndex.from_tuples(
        [(variable, region) for region in df.columns],
        names=["variable", "region"],
    )
    return df


# %%
parts = []
for zip_path in zip_files:
    m = re.match(
        r"PECD_NUTS1_\d+_(.+)_capacity_factor_ratio\.zip$", zip_path.name
    )
    if not m:
        print(f"  Skipping unrecognised filename: {zip_path.name}")
        continue
    variable = m.group(1)
    print(f"Loading {variable} ...")
    df = load_nuts1_zip(zip_path, variable)
    print(
        f"  {len(df):,} rows · {len(df.columns)} DE regions · "
        f"{df.index[0]} – {df.index[-1]}"
    )
    parts.append(df)

# %%
combined = pd.concat(parts, axis=1).sort_index()
print(f"\nCombined shape: {combined.shape}")
print(f"Regions: {sorted(combined.columns.get_level_values('region').unique())}")

# %%
output = paths.pecd_nuts1_de_processed_file
output.parent.mkdir(parents=True, exist_ok=True)
combined.to_parquet(output)
print(f"\nSaved to {output}")
