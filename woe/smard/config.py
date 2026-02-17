"""Configuration module for SMARD data types and constants.

This module contains enumerations for various SMARD data parameters including
time resolutions, regions, and variable IDs for different types of power data.
"""

from datetime import datetime
from enum import Enum
from typing import List

# Default start date for data downloads (SMARD data starts around 2015)
DEFAULT_START_DATE = datetime(2015, 1, 1)


class Resolution(str, Enum):
    """Time resolution options for SMARD data queries."""
    HOUR = "hour"
    QUARTER_HOUR = "quarterhour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"


class Region(str, Enum):
    """Region identifiers for SMARD data queries."""
    DE = "DE"  # Germany
    AT = "AT"  # Austria
    LU = "LU"  # Luxembourg
    DE_LU = "DE-LU"  # Market area: DE/LU (from 01.10.2018)
    DE_AT_LU = "DE-AT-LU"  # Market area: DE/AT/LU (until 30.09.2018)
    FIFTY_HERTZ = "50Hertz"  # Control area (DE): 50Hertz
    AMPRION = "Amprion"  # Control area (DE): Amprion
    TENNET = "TenneT"  # Control area (DE): TenneT
    TRANSNET_BW = "TransnetBW"  # Control area (DE): TransnetBW
    APG = "APG"  # Control area (AT): APG
    CREOS = "Creos"  # Control area (LU): Creos


class Variable(int, Enum):
    """Variable IDs for different types of power data."""
    # Power Generation
    BROWN_COAL = 1223
    NUCLEAR = 1224
    WIND_OFFSHORE = 1225
    HYDRO = 1226
    OTHER_CONVENTIONAL = 1227
    OTHER_RENEWABLE = 1228
    BIOMASS = 4066
    WIND_ONSHORE = 4067
    SOLAR = 4068
    HARD_COAL = 4069
    PUMPED_STORAGE = 4070
    NATURAL_GAS = 4071

    # Power Consumption
    TOTAL_LOAD = 410
    RESIDUAL_LOAD = 4359
    PUMPED_STORAGE_LOAD = 4387

    # Market Prices
    PRICE_DE_LU = 4169
    PRICE_DE_LU_NEIGHBORS = 5078
    PRICE_BE = 4996
    PRICE_NO2 = 4997
    PRICE_AT = 4170
    PRICE_DK1 = 252
    PRICE_DK2 = 253
    PRICE_FR = 254
    PRICE_IT_NORTH = 255
    PRICE_NL = 256
    PRICE_PL = 257
    PRICE_PL2 = 258
    PRICE_CH = 259
    PRICE_SI = 260
    PRICE_CZ = 261
    PRICE_HU = 262

    # Forecasts - day-ahead (generation)
    FORECAST_OFFSHORE = 3791
    FORECAST_ONSHORE = 123
    FORECAST_SOLAR = 125
    FORECAST_OTHER = 715
    FORECAST_WIND_SOLAR = 5097
    FORECAST_TOTAL = 122

    # Forecasts - day-ahead (consumption)
    FORECAST_LOAD = 411

    # Forecasts - intraday (generation)
    FORECAST_INTRADAY_SOLAR = 5126
    FORECAST_INTRADAY_ONSHORE = 5127
    FORECAST_INTRADAY_OFFSHORE = 5128
    FORECAST_INTRADAY_WIND_SOLAR = 5129
    
    # Capacity
    CAPACITY_BIOMASS = 189
    CAPACITY_HYDRO = 3792
    CAPACITY_WIND_OFFSHORE = 4076
    CAPACITY_WIND_ONSHORE = 186
    CAPACITY_SOLAR = 188
    CAPACITY_OTHER_RENEWABLE = 194
    CAPACITY_BROWN_COAL = 4072
    CAPACITY_HARD_COAL = 4075
    CAPACITY_NATURAL_GAS = 198
    CAPACITY_PUMPED_STORAGE = 4074
    # CAPACITY_UNKNOWN = 207

    @classmethod
    def get_name(cls, value: int) -> str:
        """Get the enum name for a given integer value.
        
        Args:
            value: The integer value to look up.
            
        Returns:
            str: The name of the enum member with the given value.
            
        Raises:
            ValueError: If no enum member has the given value.
        """
        try:
            return cls(value).name
        except ValueError:
            raise ValueError(f"No Variable enum member has value {value}")

    @classmethod
    def get_value_to_name_map(cls) -> dict[int, str]:
        """Get a dictionary mapping integer values to enum names.
        
        Returns:
            dict[int, str]: A dictionary where keys are the integer values
                           and values are the corresponding enum names.
        """
        return {member.value: member.name for member in cls}

    @classmethod
    def get_generation_variables(cls) -> List[int]:
        """Get all power generation variable IDs.
        
        Returns:
            List[int]: List of all power generation variable IDs.
        """
        generation_vars = [
            cls.BROWN_COAL, cls.NUCLEAR, cls.WIND_OFFSHORE, cls.HYDRO,
            cls.OTHER_CONVENTIONAL, cls.OTHER_RENEWABLE, cls.BIOMASS,
            cls.WIND_ONSHORE, cls.SOLAR, cls.HARD_COAL, cls.PUMPED_STORAGE,
            cls.NATURAL_GAS
        ]
        return [var.value for var in generation_vars]

    @classmethod
    def get_consumption_variables(cls) -> List[int]:
        """Get all power consumption variable IDs.
        
        Returns:
            List[int]: List of all power consumption variable IDs.
        """
        consumption_vars = [
            cls.TOTAL_LOAD, cls.RESIDUAL_LOAD, cls.PUMPED_STORAGE_LOAD
        ]
        return [var.value for var in consumption_vars]

    @classmethod
    def get_price_variables(cls) -> List[int]:
        """Get all market price variable IDs.
        
        Returns:
            List[int]: List of all market price variable IDs.
        """
        price_vars = [
            cls.PRICE_DE_LU, cls.PRICE_DE_LU_NEIGHBORS, cls.PRICE_BE,
            cls.PRICE_NO2, cls.PRICE_AT, cls.PRICE_DK1, cls.PRICE_DK2,
            cls.PRICE_FR, cls.PRICE_IT_NORTH, cls.PRICE_NL, cls.PRICE_PL,
            cls.PRICE_PL2, cls.PRICE_CH, cls.PRICE_SI, cls.PRICE_CZ,
            cls.PRICE_HU
        ]
        return [var.value for var in price_vars]

    @classmethod
    def get_forecast_variables(cls) -> List[int]:
        """Get all forecast variable IDs.

        Returns:
            List[int]: List of all forecast variable IDs.
        """
        forecast_vars = [
            cls.FORECAST_OFFSHORE, cls.FORECAST_ONSHORE, cls.FORECAST_SOLAR,
            cls.FORECAST_OTHER, cls.FORECAST_WIND_SOLAR, cls.FORECAST_TOTAL,
            cls.FORECAST_LOAD,
            cls.FORECAST_INTRADAY_SOLAR, cls.FORECAST_INTRADAY_ONSHORE,
            cls.FORECAST_INTRADAY_OFFSHORE, cls.FORECAST_INTRADAY_WIND_SOLAR,
        ]
        return [var.value for var in forecast_vars]

    @classmethod
    def get_capacity_variables(cls) -> List[int]:
        """Get all capacity variable IDs.

        Returns:
            List[int]: List of all capacity variable IDs.
        """
        capacity_vars = [
            cls.CAPACITY_BIOMASS, 
            cls.CAPACITY_HYDRO, 
            cls.CAPACITY_WIND_OFFSHORE, 
            cls.CAPACITY_WIND_ONSHORE, 
            cls.CAPACITY_SOLAR, 
            cls.CAPACITY_OTHER_RENEWABLE, 
            cls.CAPACITY_BROWN_COAL, 
            cls.CAPACITY_HARD_COAL, 
            cls.CAPACITY_NATURAL_GAS, 
            cls.CAPACITY_PUMPED_STORAGE
        ]
        return [var.value for var in capacity_vars]


def get_all_resolutions() -> List[str]:
    """Get all possible time resolutions.
    
    Returns:
        List[str]: List of all available time resolution values.
    """
    return [r.value for r in Resolution]


def get_all_regions() -> List[str]:
    """Get all possible regions.
    
    Returns:
        List[str]: List of all available region identifiers.
    """
    return [r.value for r in Region]


def get_all_variables() -> List[int]:
    """Get all possible variable IDs.
    
    Returns:
        List[int]: List of all available variable IDs.
    """
    return [v.value for v in Variable]