"""Download DE/LU electricity prices from SMARD API.

This script downloads day-ahead electricity prices for the DE/LU market area
from the SMARD API (German Federal Network Agency).

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
            print(f"Found existing data with {len(df)} records")
            return df
        except Exception as e:
            print(f"Warning: Could not read existing file: {e}")
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

    # Handle both index-based and column-based timestamps
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index.max().to_pydatetime()
    elif "timestamp" in df.columns:
        return pd.to_datetime(df["timestamp"]).max().to_pydatetime()
    return None


def download_de_lu_prices(force_full: bool = False) -> None:
    """Download DE/LU electricity prices from SMARD.

    Downloads day-ahead prices for the DE/LU market area. If data already
    exists, only fetches new data after the last timestamp.

    Args:
        force_full: If True, download all data from DEFAULT_START_DATE
                   regardless of existing data.
    """
    paths = ProjPaths()
    paths.ensure_directories()

    output_file = paths.smard_prices_file
    variable = Variable.PRICE_DE_LU
    variable_name = variable.name

    print(f"Output file: {output_file}")
    print(f"Variable: {variable_name} ({variable.value})")

    # Determine start date
    existing_df = None if force_full else get_existing_data(output_file)
    last_timestamp = get_last_timestamp(existing_df)

    if last_timestamp:
        # Start from the day after the last timestamp to avoid duplicates
        start_time = last_timestamp + timedelta(hours=1)
        print(f"Incremental download from: {start_time}")
    else:
        start_time = DEFAULT_START_DATE
        print(f"Full download from: {start_time}")

    # Check if we need to download (only if start_time is before now)
    if start_time >= datetime.now():
        print("Data is already up to date. Nothing to download.")
        return

    # Download new data
    print(f"\nDownloading {variable_name} data...")
    try:
        new_df = download_smard_data(
            region=Region.DE_LU.value,
            resolution=Resolution.HOUR.value,
            variable=variable.value,
            variable_name=variable_name,
            start_time=start_time,
        )
    except RuntimeError as e:
        if "No data available after" in str(e):
            print("No new data available from API.")
            return
        raise

    if new_df.empty:
        print("No new data received from API.")
        return

    print(f"Downloaded {len(new_df)} new records")

    # Combine with existing data if applicable
    if existing_df is not None and not existing_df.empty:
        # Ensure both have the same structure
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
    print(f"Saved data to {output_file}")


def main():
    """Main entry point for the download script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download DE/LU electricity prices from SMARD API"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force full download from start date, ignoring existing data",
    )
    args = parser.parse_args()

    download_de_lu_prices(force_full=args.force)


if __name__ == "__main__":
    main()
