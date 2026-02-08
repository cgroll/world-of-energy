"""Download SMARD data for electricity price analysis.

This script downloads all data needed for electricity price analysis from the
SMARD API (German Federal Network Agency):

1. Day-ahead electricity prices (DE/LU market area)
2. Residual load components: solar, wind (onshore/offshore), total load
3. Baseload generation: nuclear, biomass, hydro
4. Installed capacities (monthly): all generation technologies

Features:
- Incremental downloads: only fetches data newer than existing data
- Can be run from any working directory (uses resolved paths)
- DVC-compatible: won't retrigger when data exists, force flag fetches latest
"""

from dataclasses import dataclass
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


@dataclass
class DownloadConfig:
    """Configuration for a single variable download."""

    variable: Variable
    output_file: Path
    region: str
    resolution: str
    column_name: str | None = None  # If None, uses variable.name

    @property
    def name(self) -> str:
        """Get the column name for this variable."""
        return self.column_name if self.column_name else self.variable.name


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


def get_start_time(
    last_timestamp: datetime | None, resolution: str
) -> datetime:
    """Calculate the start time for incremental download.

    Args:
        last_timestamp: Last timestamp in existing data, or None.
        resolution: Time resolution (hour, month, etc.).

    Returns:
        Start time for the download.
    """
    if last_timestamp is None:
        return DEFAULT_START_DATE

    if resolution == Resolution.MONTH.value:
        # For monthly data, start from next month
        start_time = last_timestamp + timedelta(days=32)
        return start_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    else:
        # For hourly data, start from next hour
        return last_timestamp + timedelta(hours=1)


def download_single_variable(config: DownloadConfig, force_full: bool = False) -> None:
    """Download a single variable from SMARD.

    Args:
        config: Download configuration for the variable.
        force_full: If True, download all data from DEFAULT_START_DATE.
    """
    print(f"\n{config.name}:")
    print(f"  Output: {config.output_file}")

    # Determine start date
    existing_df = None if force_full else get_existing_data(config.output_file)
    last_timestamp = get_last_timestamp(existing_df)
    start_time = get_start_time(last_timestamp, config.resolution)

    if last_timestamp:
        print(f"  Incremental download from: {start_time}")
    else:
        print(f"  Full download from: {start_time}")

    # Check if we need to download
    if start_time >= datetime.now():
        print("  Data is already up to date. Nothing to download.")
        return

    # Download new data
    try:
        new_df = download_smard_data(
            region=config.region,
            resolution=config.resolution,
            variable=config.variable.value,
            variable_name=config.name,
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
    combined_df.to_parquet(config.output_file)
    print(f"  Saved to {config.output_file}")


def download_capacities(paths: ProjPaths, force_full: bool = False) -> None:
    """Download installed capacities from SMARD.

    Downloads monthly installed capacities for all generation technologies
    into a single combined file.

    Args:
        paths: Project paths configuration.
        force_full: If True, download all data from DEFAULT_START_DATE.
    """
    print("\n" + "=" * 60)
    print("DOWNLOADING INSTALLED CAPACITIES (monthly)")
    print("=" * 60)

    capacity_variables = [
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

    output_file = paths.smard_capacities_file
    print(f"\nOutput file: {output_file}")
    print(f"Downloading {len(capacity_variables)} capacity variables")

    # Determine start date from existing data
    existing_df = None if force_full else get_existing_data(output_file)
    last_timestamp = get_last_timestamp(existing_df)
    start_time = get_start_time(last_timestamp, Resolution.MONTH.value)

    if last_timestamp:
        print(f"Incremental download from: {start_time}")
    else:
        print(f"Full download from: {start_time}")

    if start_time >= datetime.now():
        print("Data is already up to date. Nothing to download.")
        return

    # Download each capacity variable
    all_data = {}
    for variable, col_name in capacity_variables:
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
                print("    No data received")
        except RuntimeError as e:
            if "No data available after" in str(e):
                print("    No new data available")
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


def download_all_smard_data(force_full: bool = False) -> None:
    """Download all SMARD data for price analysis.

    Args:
        force_full: If True, download all data from DEFAULT_START_DATE
                   regardless of existing data.
    """
    paths = ProjPaths()
    paths.ensure_directories()

    # Define all hourly variables to download
    hourly_configs = [
        # Prices
        DownloadConfig(
            variable=Variable.PRICE_DE_LU,
            output_file=paths.smard_prices_file,
            region=Region.DE_LU.value,
            resolution=Resolution.HOUR.value,
            column_name="PRICE_DE_LU",
        ),
        # Residual load components
        DownloadConfig(
            variable=Variable.SOLAR,
            output_file=paths.smard_solar_file,
            region=Region.DE_LU.value,
            resolution=Resolution.HOUR.value,
        ),
        DownloadConfig(
            variable=Variable.WIND_ONSHORE,
            output_file=paths.smard_wind_onshore_file,
            region=Region.DE_LU.value,
            resolution=Resolution.HOUR.value,
        ),
        DownloadConfig(
            variable=Variable.WIND_OFFSHORE,
            output_file=paths.smard_wind_offshore_file,
            region=Region.DE_LU.value,
            resolution=Resolution.HOUR.value,
        ),
        DownloadConfig(
            variable=Variable.TOTAL_LOAD,
            output_file=paths.smard_total_load_file,
            region=Region.DE_LU.value,
            resolution=Resolution.HOUR.value,
        ),
        # Baseload generation
        DownloadConfig(
            variable=Variable.NUCLEAR,
            output_file=paths.smard_nuclear_file,
            region=Region.DE_LU.value,
            resolution=Resolution.HOUR.value,
            column_name="nuclear",
        ),
        DownloadConfig(
            variable=Variable.BIOMASS,
            output_file=paths.smard_biomass_file,
            region=Region.DE_LU.value,
            resolution=Resolution.HOUR.value,
            column_name="biomass",
        ),
        DownloadConfig(
            variable=Variable.HYDRO,
            output_file=paths.smard_hydro_file,
            region=Region.DE_LU.value,
            resolution=Resolution.HOUR.value,
            column_name="hydro",
        ),
    ]

    # Download hourly data
    print("=" * 60)
    print("DOWNLOADING HOURLY DATA (prices, generation, load)")
    print("=" * 60)
    print(f"Variables: {[c.name for c in hourly_configs]}")

    for config in hourly_configs:
        download_single_variable(config, force_full)

    # Download monthly capacity data
    download_capacities(paths, force_full)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


def main():
    """Main entry point for the download script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download SMARD data for electricity price analysis"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force full download from start date, ignoring existing data",
    )
    args = parser.parse_args()

    download_all_smard_data(force_full=args.force)


if __name__ == "__main__":
    main()
