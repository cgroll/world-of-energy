"""Download baseload generation and installed capacities from SMARD API.

This script downloads:
1. Baseload generation data (hourly): nuclear, biomass, hydro
2. Installed capacities (monthly): all generation technologies

Features:
- Incremental downloads: only fetches data newer than existing data
- Can be run from any working directory (uses resolved paths)
- DVC-compatible: won't retrigger when data exists, force flag fetches latest
"""

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from woe.paths import ProjPaths
from woe.smard import (
    DEFAULT_START_DATE,
    Resolution,
    Region,
    Variable,
    download_smard_data,
)


# Baseload generation variables (hourly)
BASELOAD_VARIABLES = [
    (Variable.NUCLEAR, "nuclear"),
    (Variable.BIOMASS, "biomass"),
    (Variable.HYDRO, "hydro"),
]

# Capacity variables (monthly)
CAPACITY_VARIABLES = [
    (Variable.CAPACITY_BIOMASS, "biomass"),
    (Variable.CAPACITY_HYDRO, "hydro"),
    (Variable.CAPACITY_WIND_OFFSHORE, "wind_offshore"),
    (Variable.CAPACITY_WIND_ONSHORE, "wind_onshore"),
    (Variable.CAPACITY_SOLAR, "solar"),
    (Variable.CAPACITY_OTHER_RENEWABLE, "other_renewable"),
    (Variable.CAPACITY_BROWN_COAL, "brown_coal"),
    (Variable.CAPACITY_HARD_COAL, "hard_coal"),
    (Variable.CAPACITY_NATURAL_GAS, "natural_gas"),
    (Variable.CAPACITY_PUMPED_STORAGE, "pumped_storage"),
]


def get_existing_data(file_path: Path) -> pd.DataFrame | None:
    """Load existing data if available.

    Args:
        file_path: Path to the parquet file.

    Returns:
        DataFrame with existing data or None if file doesn't exist.
    """
    if file_path.exists():
        try:
            df = pd.read_parquet(file_path)
            print(f"  Found existing data with {len(df)} records")
            return df
        except Exception as e:
            print(f"  Warning: Could not read existing file: {e}")
            return None
    return None


def get_last_timestamp(df: pd.DataFrame | None) -> datetime | None:
    """Get the last timestamp from existing data.

    Args:
        df: DataFrame with timestamp index.

    Returns:
        Last timestamp or None if no data.
    """
    if df is None or df.empty:
        return None

    if isinstance(df.index, pd.DatetimeIndex):
        return df.index.max().to_pydatetime()
    elif "timestamp" in df.columns:
        return pd.to_datetime(df["timestamp"]).max().to_pydatetime()
    return None


def download_single_variable(
    variable: Variable,
    variable_name: str,
    output_file: Path,
    region: str,
    resolution: str,
    force_full: bool = False,
) -> None:
    """Download a single variable from SMARD.

    Args:
        variable: The SMARD variable to download.
        variable_name: Name for the column.
        output_file: Path to save the parquet file.
        region: Region code.
        resolution: Time resolution.
        force_full: If True, download all data from DEFAULT_START_DATE.
    """
    print(f"\n{variable_name}:")
    print(f"  Output: {output_file}")

    # Determine start date
    existing_df = None if force_full else get_existing_data(output_file)
    last_timestamp = get_last_timestamp(existing_df)

    if last_timestamp:
        if resolution == Resolution.MONTH.value:
            # For monthly data, start from next month
            start_time = last_timestamp + timedelta(days=32)
            start_time = start_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            # For hourly data, start from next hour
            start_time = last_timestamp + timedelta(hours=1)
        print(f"  Incremental download from: {start_time}")
    else:
        start_time = DEFAULT_START_DATE
        print(f"  Full download from: {start_time}")

    # Check if we need to download
    if start_time >= datetime.now():
        print("  Data is already up to date. Nothing to download.")
        return

    # Download new data
    try:
        new_df = download_smard_data(
            region=region,
            resolution=resolution,
            variable=variable.value,
            variable_name=variable_name,
            start_time=start_time,
        )
    except RuntimeError as e:
        if "No data available after" in str(e):
            print("  No new data available from API.")
            return
        raise

    if new_df.empty:
        print("  No new data received from API.")
        return

    print(f"  Downloaded {len(new_df)} new records")

    # Combine with existing data if applicable
    if existing_df is not None and not existing_df.empty:
        if not isinstance(existing_df.index, pd.DatetimeIndex):
            if "timestamp" in existing_df.columns:
                existing_df = existing_df.set_index("timestamp")

        combined_df = pd.concat([existing_df, new_df])
        combined_df = combined_df.sort_index()
        combined_df = combined_df[~combined_df.index.duplicated(keep="last")]
        print(f"  Combined total: {len(combined_df)} records")
    else:
        combined_df = new_df

    # Save to parquet
    combined_df.to_parquet(output_file)
    print(f"  Saved to {output_file}")


def download_baseload_generation(paths: ProjPaths, force_full: bool = False) -> None:
    """Download baseload generation data (nuclear, biomass, hydro).

    Args:
        paths: Project paths configuration.
        force_full: If True, download all data from DEFAULT_START_DATE.
    """
    print("=" * 60)
    print("DOWNLOADING BASELOAD GENERATION (hourly)")
    print("=" * 60)

    # Map variables to output files
    variable_files = {
        Variable.NUCLEAR: paths.smard_nuclear_file,
        Variable.BIOMASS: paths.smard_biomass_file,
        Variable.HYDRO: paths.smard_hydro_file,
    }

    for variable, name in BASELOAD_VARIABLES:
        download_single_variable(
            variable=variable,
            variable_name=name,
            output_file=variable_files[variable],
            region=Region.DE_LU.value,
            resolution=Resolution.HOUR.value,
            force_full=force_full,
        )


def download_capacities(paths: ProjPaths, force_full: bool = False) -> None:
    """Download installed capacities from SMARD.

    Downloads monthly installed capacities for all generation technologies.

    Args:
        paths: Project paths configuration.
        force_full: If True, download all data from DEFAULT_START_DATE.
    """
    print("\n" + "=" * 60)
    print("DOWNLOADING INSTALLED CAPACITIES (monthly)")
    print("=" * 60)

    output_file = paths.smard_capacities_file
    print(f"\nOutput file: {output_file}")
    print(f"Downloading {len(CAPACITY_VARIABLES)} capacity variables")

    # Determine start date from existing data
    existing_df = None if force_full else get_existing_data(output_file)
    last_timestamp = get_last_timestamp(existing_df)

    if last_timestamp:
        start_time = last_timestamp + timedelta(days=32)
        start_time = start_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        print(f"Incremental download from: {start_time}")
    else:
        start_time = DEFAULT_START_DATE
        print(f"Full download from: {start_time}")

    if start_time >= datetime.now():
        print("Data is already up to date. Nothing to download.")
        return

    # Download each capacity variable
    all_data = {}
    for variable, col_name in CAPACITY_VARIABLES:
        print(f"\n  Downloading {col_name} capacity (ID: {variable.value})...")
        try:
            df = download_smard_data(
                region=Region.DE.value,
                resolution=Resolution.MONTH.value,
                variable=variable.value,
                variable_name=col_name,
                start_time=start_time,
            )
            if not df.empty:
                all_data[col_name] = df[col_name]
                print(f"    Downloaded {len(df)} records")
            else:
                print(f"    No data received")
        except RuntimeError as e:
            if "No data available after" in str(e):
                print(f"    No new data available")
            else:
                print(f"    Error: {e}")

    if not all_data:
        print("\nNo new capacity data received from API.")
        return

    # Combine all variables into single DataFrame
    new_df = pd.DataFrame(all_data)
    print(f"\nDownloaded {len(new_df)} new capacity records")

    # Combine with existing data if applicable
    if existing_df is not None and not existing_df.empty:
        if not isinstance(existing_df.index, pd.DatetimeIndex):
            if "timestamp" in existing_df.columns:
                existing_df = existing_df.set_index("timestamp")

        combined_df = pd.concat([existing_df, new_df])
        combined_df = combined_df.sort_index()
        combined_df = combined_df[~combined_df.index.duplicated(keep="last")]
        print(f"Combined total: {len(combined_df)} records")
    else:
        combined_df = new_df

    # Save to parquet
    combined_df.to_parquet(output_file)
    print(f"Saved capacities to {output_file}")


def main():
    """Main entry point for the download script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download baseload generation and installed capacities from SMARD API"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force full download from start date, ignoring existing data",
    )
    args = parser.parse_args()

    paths = ProjPaths()
    paths.ensure_directories()

    download_baseload_generation(paths, force_full=args.force)
    download_capacities(paths, force_full=args.force)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
