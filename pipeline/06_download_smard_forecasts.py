"""Download SMARD forecast data for solar, wind, and load.

This script downloads day-ahead and intraday generation/load forecasts from the
SMARD API (German Federal Network Agency):

Day-ahead forecasts:
- Solar (Prognostizierte Erzeugung: Photovoltaik)
- Wind onshore (Prognostizierte Erzeugung: Onshore)
- Wind offshore (Prognostizierte Erzeugung: Offshore)
- Load (Prognostizierter Verbrauch: Gesamt)

Intraday forecasts:
- Solar (Prognostizierte Erzeugung Intraday: Photovoltaik)
- Wind onshore (Prognostizierte Erzeugung Intraday: Onshore)
- Wind offshore (Prognostizierte Erzeugung Intraday: Offshore)

All forecasts are downloaded at quarter-hourly resolution for the DE region,
each into its own parquet file.

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
    column_name: str | None = None

    @property
    def name(self) -> str:
        """Get the column name for this variable."""
        return self.column_name if self.column_name else self.variable.name


def get_existing_data(file_path: Path) -> pd.DataFrame | None:
    """Load existing data if available."""
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
    """Get the last timestamp from existing data."""
    if df is None or df.empty:
        return None
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index.max().to_pydatetime()
    elif "timestamp" in df.columns:
        return pd.to_datetime(df["timestamp"]).max().to_pydatetime()
    return None


def get_start_time(last_timestamp: datetime | None) -> datetime:
    """Calculate the start time for incremental download."""
    if last_timestamp is None:
        return DEFAULT_START_DATE
    return last_timestamp + timedelta(hours=1)


def download_single_variable(config: DownloadConfig, force_full: bool = False) -> None:
    """Download a single variable from SMARD.

    Args:
        config: Download configuration for the variable.
        force_full: If True, download all data from DEFAULT_START_DATE.
    """
    print(f"\n{config.name}:")
    print(f"  Output: {config.output_file}")

    existing_df = None if force_full else get_existing_data(config.output_file)
    last_timestamp = get_last_timestamp(existing_df)
    start_time = get_start_time(last_timestamp)

    if last_timestamp:
        print(f"  Incremental download from: {start_time}")
    else:
        print(f"  Full download from: {start_time}")

    if start_time >= datetime.now():
        print("  Data is already up to date. Nothing to download.")
        return

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

    combined_df.to_parquet(config.output_file)
    print(f"  Saved to {config.output_file}")


def download_all_forecasts(force_full: bool = False) -> None:
    """Download all SMARD forecast data.

    Args:
        force_full: If True, download all data from DEFAULT_START_DATE
                   regardless of existing data.
    """
    paths = ProjPaths()
    paths.ensure_directories()

    configs = [
        # Day-ahead
        DownloadConfig(
            variable=Variable.FORECAST_SOLAR,
            output_file=paths.smard_forecast_da_solar_file,
            region=Region.DE.value,
            resolution=Resolution.QUARTER_HOUR.value,
            column_name="da_solar",
        ),
        DownloadConfig(
            variable=Variable.FORECAST_ONSHORE,
            output_file=paths.smard_forecast_da_wind_onshore_file,
            region=Region.DE.value,
            resolution=Resolution.QUARTER_HOUR.value,
            column_name="da_wind_onshore",
        ),
        DownloadConfig(
            variable=Variable.FORECAST_OFFSHORE,
            output_file=paths.smard_forecast_da_wind_offshore_file,
            region=Region.DE.value,
            resolution=Resolution.QUARTER_HOUR.value,
            column_name="da_wind_offshore",
        ),
        DownloadConfig(
            variable=Variable.FORECAST_LOAD,
            output_file=paths.smard_forecast_da_load_file,
            region=Region.DE.value,
            resolution=Resolution.QUARTER_HOUR.value,
            column_name="da_load",
        ),
        # Intraday
        DownloadConfig(
            variable=Variable.FORECAST_INTRADAY_SOLAR,
            output_file=paths.smard_forecast_id_solar_file,
            region=Region.DE.value,
            resolution=Resolution.QUARTER_HOUR.value,
            column_name="id_solar",
        ),
        DownloadConfig(
            variable=Variable.FORECAST_INTRADAY_ONSHORE,
            output_file=paths.smard_forecast_id_wind_onshore_file,
            region=Region.DE.value,
            resolution=Resolution.QUARTER_HOUR.value,
            column_name="id_wind_onshore",
        ),
        DownloadConfig(
            variable=Variable.FORECAST_INTRADAY_OFFSHORE,
            output_file=paths.smard_forecast_id_wind_offshore_file,
            region=Region.DE.value,
            resolution=Resolution.QUARTER_HOUR.value,
            column_name="id_wind_offshore",
        ),
    ]

    print("=" * 60)
    print("DOWNLOADING SMARD FORECASTS (quarter-hourly)")
    print("=" * 60)
    print(f"Variables: {[c.name for c in configs]}")

    for config in configs:
        download_single_variable(config, force_full)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


def main():
    """Main entry point for the download script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download SMARD forecast data for solar, wind, and load"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force full download from start date, ignoring existing data",
    )
    args = parser.parse_args()

    download_all_forecasts(force_full=args.force)


if __name__ == "__main__":
    main()
