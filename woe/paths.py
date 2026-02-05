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

    def ensure_directories(self) -> None:
        """Create all necessary directories if they don't exist."""
        directories = [
            self.data_path,
            self.downloads_path,
            self.smard_downloads_path,
            self.processed_data_path,
            self.output_path,
            self.reports_path,
            self.images_path,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
