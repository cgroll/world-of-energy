"""SMARD data download and configuration module.

This module provides tools for downloading electricity market data from the
SMARD API (Strommarktdaten) maintained by the German Federal Network Agency.
"""

from woe.smard.api import download_smard_data
from woe.smard.config import DEFAULT_START_DATE, Resolution, Region, Variable

__all__ = [
    "download_smard_data",
    "DEFAULT_START_DATE",
    "Resolution",
    "Region",
    "Variable",
]
