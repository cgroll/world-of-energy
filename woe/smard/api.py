"""API functions for interacting with SMARD data."""

from datetime import datetime
import requests
import pandas as pd
from typing import List, Tuple, Optional

def download_smard_data(
    region: str, 
    resolution: str, 
    variable: int, 
    variable_name: str,
    start_time: Optional[datetime] = None
) -> pd.DataFrame:
    """Download data from SMARD API for given parameters.
    
    Note: The SMARD API uses a block-based data retrieval system where we first get
    timestamps marking the start of data blocks (e.g. weekly chunks), then fetch
    the actual observations for each block in separate requests.
    
    Args:
        region: Region code (e.g. 'DE' for Germany)
        resolution: Time resolution (e.g. 'day', 'hour')
        variable: Variable ID for the type of power data
        variable_name: Name of the variable to use as column label
        start_time: Optional datetime to specify start of data collection.
                   If None, returns all available data.
    
    Returns:
        pd.DataFrame: DataFrame with timestamp index and value column
        
    Raises:
        RuntimeError: If there's an error fetching data from the API
    """
    base_url = "https://www.smard.de/app"
    
    # Step 1: Get available timestamps
    index_url = f"{base_url}/chart_data/{variable}/{region}/index_{resolution}.json"
    response = requests.get(index_url)
    
    if response.status_code != 200:
        raise RuntimeError(f"Error fetching timestamps: {response.status_code}")
        
    timestamps = response.json()["timestamps"]
    
    if not timestamps:
        raise RuntimeError("No timestamps available for the specified parameters")

    # Filter timestamps if start_time is provided
    if start_time:
        start_timestamp = int(start_time.timestamp() * 1000)  # Convert to milliseconds
        timestamps = [ts for ts in timestamps if ts >= start_timestamp]
        
        if not timestamps:
            raise RuntimeError(f"No data available after {start_time}")

    # Initialize empty lists to store all data
    all_timestamps = []
    all_values = []

    # Step 2: Get data for each timestamp
    for timestamp in timestamps:
        data_url = f"{base_url}/chart_data/{variable}/{region}/{variable}_{region}_{resolution}_{timestamp}.json"
        data_response = requests.get(data_url)
        
        if data_response.status_code != 200:
            print(f"Warning: Error fetching data for timestamp {timestamp}: {data_response.status_code}")
            continue
            
        data = data_response.json()
        series_data: List[Tuple[int, float]] = data["series"]
        
        # Extract timestamps and values
        ts = [datetime.fromtimestamp(ts/1000) for ts, _ in series_data]
        values = [val for _, val in series_data]
        
        all_timestamps.extend(ts)
        all_values.extend(values)
    
    # Create and return dataframe
    df = pd.DataFrame({
        'timestamp': all_timestamps,
        variable_name: all_values
    })
    
    # Sort by timestamp and remove duplicates
    df = df.sort_values('timestamp').drop_duplicates(subset='timestamp')
    df.set_index('timestamp', inplace=True)
    df = df.dropna()
    
    return df 
