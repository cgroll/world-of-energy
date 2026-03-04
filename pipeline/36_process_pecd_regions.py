"""Process PECD regional capacity factors and power generation.

Reads the ZIP files downloaded by 35_download_pecd_regions.py, one per
(variable, product_type) combination (duplicate year variants are skipped),
and combines them into a single wide-format parquet file.

Each ZIP contains a CSV with a metadata header followed by hourly data with
a Date column and one column per European country (NUTS 0 ISO-2 codes).
The full ERA5 back-catalogue is included (1979 onwards).

Output index: time (hourly DatetimeIndex)
Output columns: MultiIndex of (variable, product_type, country)

Output:
  data/processed/pecd/pecd_regions.parquet
"""

# %%
import io
import re
import zipfile

import pandas as pd

from woe.paths import ProjPaths

paths = ProjPaths()

# %%
# Discover ZIP files; keep one per (variable, product_type) — year variants are identical
zip_files = sorted(paths.pecd_downloads_path.glob("PECD_*.zip"))
print(f"Found {len(zip_files)} ZIP files in {paths.pecd_downloads_path}")

seen: set[tuple[str, str]] = set()
combos: list[tuple[str, str, object]] = []
for zip_path in zip_files:
    m = re.match(r"PECD_\d+_(.+)_(capacity_factor_ratio|power)\.zip$", zip_path.name)
    if not m:
        print(f"  Skipping unrecognised filename: {zip_path.name}")
        continue
    variable, product_type = m.group(1), m.group(2)
    key = (variable, product_type)
    if key not in seen:
        seen.add(key)
        combos.append((variable, product_type, zip_path))

print(f"Unique (variable, product_type) combinations: {len(combos)}")
for variable, product_type, zip_path in combos:
    print(f"  {variable} / {product_type}")


# %%
def load_pecd_zip(zip_path, variable: str, product_type: str) -> pd.DataFrame:
    """Load one PECD ZIP; return wide DataFrame with MultiIndex columns (variable, product_type, country)."""
    with zipfile.ZipFile(zip_path) as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            raw = f.read().decode("utf-8")

    lines = raw.splitlines()
    header_idx = next(i for i, line in enumerate(lines) if line.startswith("Date,"))
    df = pd.read_csv(
        io.StringIO("\n".join(lines[header_idx:])),
        parse_dates=["Date"],
        index_col="Date",
    )
    df.index.name = "time"
    df.columns = pd.MultiIndex.from_tuples(
        [(variable, product_type, country) for country in df.columns],
        names=["variable", "product_type", "country"],
    )
    return df


# %%
parts = []
for variable, product_type, zip_path in combos:
    print(f"Loading {variable} / {product_type} ...")
    df = load_pecd_zip(zip_path, variable, product_type)
    print(f"  {len(df):,} rows · {len(df.columns)} country columns · {df.index[0]} – {df.index[-1]}")
    parts.append(df)

# %%
combined = pd.concat(parts, axis=1).sort_index()
print(f"\nCombined shape: {combined.shape}")
print(combined.iloc[:3, :4])

# %%
output = paths.pecd_processed_file
output.parent.mkdir(parents=True, exist_ok=True)
combined.to_parquet(output)
print(f"\nSaved to {output}")
