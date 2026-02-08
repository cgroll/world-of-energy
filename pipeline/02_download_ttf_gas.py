"""Download TTF natural gas prices from Yahoo Finance.

This script downloads daily TTF (Title Transfer Facility) natural gas futures
prices from Yahoo Finance using the yfinance library.

Features:
- Downloads maximum available history
- Can be run from any working directory (uses resolved paths)
- DVC-compatible: won't retrigger when data exists, force flag fetches latest
"""

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

from woe.paths import ProjPaths


# TTF Natural Gas Futures ticker on Yahoo Finance
TTF_TICKER = "TTF=F"


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


def download_ttf_prices(force_full: bool = False) -> None:
    """Download TTF natural gas prices from Yahoo Finance.

    Downloads daily TTF futures prices. If data already exists, only fetches
    new data after the last date.

    Args:
        force_full: If True, download all data regardless of existing data.
    """
    paths = ProjPaths()
    paths.ensure_directories()

    output_file = paths.ttf_gas_prices_file

    print(f"Output file: {output_file}")
    print(f"Ticker: {TTF_TICKER}")

    # Load existing data
    existing_df = None if force_full else get_existing_data(output_file)

    # Determine start date for download
    if existing_df is not None and not existing_df.empty:
        last_date = existing_df.index.max()
        start_date = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"Incremental download from: {start_date}")
    else:
        start_date = None  # Download maximum available history
        print("Full download (maximum period)")

    # Download data from Yahoo Finance
    print(f"\nDownloading {TTF_TICKER} data...")
    ticker = yf.Ticker(TTF_TICKER)

    if start_date:
        new_df = ticker.history(start=start_date)
    else:
        new_df = ticker.history(period="max")

    if new_df.empty:
        print("No new data available.")
        return

    # Clean up the dataframe
    # Keep only relevant columns and ensure clean index
    new_df = new_df[["Open", "High", "Low", "Close", "Volume"]]
    new_df.index = pd.to_datetime(new_df.index).tz_localize(None)
    new_df.index.name = "date"

    print(f"Downloaded {len(new_df)} new records")
    print(f"Date range: {new_df.index.min()} to {new_df.index.max()}")

    # Combine with existing data if applicable
    if existing_df is not None and not existing_df.empty:
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
    parser = argparse.ArgumentParser(
        description="Download TTF natural gas prices from Yahoo Finance"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force full download, ignoring existing data",
    )
    args = parser.parse_args()

    download_ttf_prices(force_full=args.force)


if __name__ == "__main__":
    main()
