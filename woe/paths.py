"""Project paths configuration.

This module provides a centralized path configuration object that can be used
across all modules in the project. All paths are resolved relative to the
project root, making scripts runnable from any working directory.
"""

from pathlib import Path
from datetime import datetime


class ProjPaths:
    """Centralized project paths configuration.

    All paths are resolved relative to the project root directory, which is
    determined by the location of this file. This ensures that scripts can
    be run from any working directory.
    """

    def __init__(self):
        # Resolve paths relative to this file's location
        self._current_file_path = Path(__file__).resolve()
        self._pkg_src_path = self._current_file_path.parent  # woe/
        self._project_path = self._pkg_src_path.parent  # world-of-energy/

    @property
    def project_path(self) -> Path:
        """Root project directory."""
        return self._project_path

    @property
    def pkg_src_path(self) -> Path:
        """Source package directory (woe/)."""
        return self._pkg_src_path

    @property
    def data_path(self) -> Path:
        """Main data directory."""
        return self._project_path / "data"

    @property
    def downloads_path(self) -> Path:
        """Downloads directory for raw data."""
        return self.data_path / "downloads"

    @property
    def smard_downloads_path(self) -> Path:
        """SMARD data downloads directory."""
        return self.downloads_path / "smard"

    @property
    def processed_data_path(self) -> Path:
        """Processed data directory."""
        return self.data_path / "processed"

    @property
    def output_path(self) -> Path:
        """Output directory for results."""
        return self._project_path / "output"

    @property
    def reports_path(self) -> Path:
        """Reports output directory."""
        return self.output_path / "reports"

    @property
    def images_path(self) -> Path:
        """Images output directory."""
        return self.output_path / "images"

    @property
    def pipeline_path(self) -> Path:
        """Pipeline scripts directory."""
        return self._project_path / "pipeline"

    # SMARD-specific paths
    @property
    def smard_prices_file(self) -> Path:
        """Path to SMARD DE/LU prices parquet file."""
        return self.smard_downloads_path / "prices_de_lu.parquet"

    @property
    def smard_solar_file(self) -> Path:
        """Path to SMARD solar generation parquet file."""
        return self.smard_downloads_path / "solar.parquet"

    @property
    def smard_wind_onshore_file(self) -> Path:
        """Path to SMARD wind onshore generation parquet file."""
        return self.smard_downloads_path / "wind_onshore.parquet"

    @property
    def smard_wind_offshore_file(self) -> Path:
        """Path to SMARD wind offshore generation parquet file."""
        return self.smard_downloads_path / "wind_offshore.parquet"

    @property
    def smard_total_load_file(self) -> Path:
        """Path to SMARD total load parquet file."""
        return self.smard_downloads_path / "total_load.parquet"

    @property
    def smard_nuclear_file(self) -> Path:
        """Path to SMARD nuclear generation parquet file."""
        return self.smard_downloads_path / "nuclear.parquet"

    @property
    def smard_biomass_file(self) -> Path:
        """Path to SMARD biomass generation parquet file."""
        return self.smard_downloads_path / "biomass.parquet"

    @property
    def smard_hydro_file(self) -> Path:
        """Path to SMARD hydro generation parquet file."""
        return self.smard_downloads_path / "hydro.parquet"

    # SMARD forecast paths (day-ahead)
    @property
    def smard_forecast_da_solar_file(self) -> Path:
        """Path to SMARD day-ahead solar forecast parquet file."""
        return self.smard_downloads_path / "forecast_da_solar.parquet"

    @property
    def smard_forecast_da_wind_onshore_file(self) -> Path:
        """Path to SMARD day-ahead wind onshore forecast parquet file."""
        return self.smard_downloads_path / "forecast_da_wind_onshore.parquet"

    @property
    def smard_forecast_da_wind_offshore_file(self) -> Path:
        """Path to SMARD day-ahead wind offshore forecast parquet file."""
        return self.smard_downloads_path / "forecast_da_wind_offshore.parquet"

    @property
    def smard_forecast_da_load_file(self) -> Path:
        """Path to SMARD day-ahead load forecast parquet file."""
        return self.smard_downloads_path / "forecast_da_load.parquet"

    # SMARD forecast paths (intraday)
    @property
    def smard_forecast_id_solar_file(self) -> Path:
        """Path to SMARD intraday solar forecast parquet file."""
        return self.smard_downloads_path / "forecast_id_solar.parquet"

    @property
    def smard_forecast_id_wind_onshore_file(self) -> Path:
        """Path to SMARD intraday wind onshore forecast parquet file."""
        return self.smard_downloads_path / "forecast_id_wind_onshore.parquet"

    @property
    def smard_forecast_id_wind_offshore_file(self) -> Path:
        """Path to SMARD intraday wind offshore forecast parquet file."""
        return self.smard_downloads_path / "forecast_id_wind_offshore.parquet"

    @property
    def smard_capacities_file(self) -> Path:
        """Path to SMARD installed capacities parquet file."""
        return self.smard_downloads_path / "capacities.parquet"

    # Commodity price paths
    @property
    def ttf_gas_prices_file(self) -> Path:
        """Path to TTF natural gas prices parquet file."""
        return self.downloads_path / "ttf_gas_prices.parquet"

    @property
    def investing_com_path(self) -> Path:
        """Investing.com downloads directory."""
        return self.downloads_path / "investing_com"

    # ERA5 reanalysis paths
    @property
    def era5_downloads_path(self) -> Path:
        """ERA5 reanalysis downloads directory."""
        return self.downloads_path / "era5"

    @property
    def era5_monthly_aggregates_path(self) -> Path:
        """ERA5 monthly aggregate data directory."""
        return self.era5_downloads_path / "monthly_aggregates"

    @property
    def era5_monthly_nc_path(self) -> Path:
        """ERA5 monthly aggregate NetCDF files directory."""
        return self.era5_monthly_aggregates_path / "nc"

    @property
    def era5_sl_climate_file(self) -> Path:
        """Path to ERA5 monthly-mean single-level climate variables (MSLP, T2m, precip, snowfall)."""
        return self.era5_monthly_nc_path / "era5_sl_climate.nc"

    @property
    def era5_sl_wind_solar_file(self) -> Path:
        """Path to ERA5 monthly-mean single-level wind variables (100m wind U/V)."""
        return self.era5_monthly_nc_path / "era5_sl_wind_solar.nc"

    @property
    def era5_sl_accumulated_file(self) -> Path:
        """Path to ERA5 monthly-mean single-level accumulated flux variables (tp, sf, ssrd)."""
        return self.era5_monthly_nc_path / "era5_sl_accumulated.nc"

    @property
    def era5_monthly_pressure_levels_file(self) -> Path:
        """Path to ERA5 monthly-mean pressure-level variables (NetCDF)."""
        return self.era5_monthly_nc_path / "era5_monthly_pressure_levels.nc"

    @property
    def era5_monthly_zarr_path(self) -> Path:
        """ERA5 monthly aggregate Zarr store directory."""
        return self.era5_monthly_aggregates_path / "zarr" / "era5_monthly.zarr"

    @property
    def era5_nao_jetstream_path(self) -> Path:
        """ERA5 daily NAO/jet-stream data directory (nao_jetstream/)."""
        return self.era5_downloads_path / "nao_jetstream"

    @property
    def era5_snapshot_20250603_1200_path(self) -> Path:
        """ERA5 single-timestamp snapshot directory for 2025-06-03 12:00 UTC."""
        return self.era5_downloads_path / "20250603_1200"

    @property
    def era5_snapshot_20250603_1200_surface_file(self) -> Path:
        """ERA5 merged surface variables for 2025-06-03 12:00 UTC."""
        return self.era5_snapshot_20250603_1200_path / "era5_20250603_1200_surface.nc"

    @property
    def era5_snapshot_20250603_1200_pressure_file(self) -> Path:
        """ERA5 merged pressure-level variables for 2025-06-03 12:00 UTC."""
        return self.era5_snapshot_20250603_1200_path / "era5_20250603_1200_pressure_levels.nc"

    @property
    def era5_germany_monthly_file(self) -> Path:
        """ERA5 monthly spatial aggregates over Germany (parquet)."""
        return self.processed_data_path / "era5_germany_monthly.parquet"

    @property
    def era5_germany_monthly_ts_file(self) -> Path:
        """ERA5 monthly Germany spatial-mean time series (parquet)."""
        return self.processed_data_path / "era5" / "time_series" / "germany_monthly.parquet"

    @property
    def rotterdam_coal_prices_file(self) -> Path:
        """Path to API 2 Rotterdam coal futures prices CSV file."""
        return self.investing_com_path / "rotterdam_coal_futures.csv"

    @property
    def eu_carbon_prices_file(self) -> Path:
        """Path to EU ETS carbon allowance prices CSV file."""
        return self.investing_com_path / "carbon_emissions_futures.csv"

    def ensure_directories(self) -> None:
        """Create all necessary directories if they don't exist."""
        directories = [
            self.data_path,
            self.downloads_path,
            self.smard_downloads_path,
            self.era5_monthly_nc_path,
            self.processed_data_path,
            self.output_path,
            self.reports_path,
            self.images_path,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
