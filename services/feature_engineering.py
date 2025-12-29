"""
Feature Engineering Utilities for ML Forecast Worker
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

def add_automated_temporal_features(df: pd.DataFrame, date_column: str, config: dict = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Generate automated temporal features (Day of week, Month, Fourier terms)
    for a time series dataframe based on config.
    
    Args:
        df: Input DataFrame
        date_column: Name of the date column
        config: Configuration dictionary with toggles
        
    Returns:
        Tuple of (modified DataFrame, list of new feature names)
    """
    if config is None:
        config = {
            "auto_fourier": True,
            "auto_dofw": True,
            "auto_month": True
        }
        
    new_features = []
    
    try:
        # Ensure date column is datetime
        temp_dates = pd.to_datetime(df[date_column])
        
        # Detection of daily data for Fourier terms primarily, 
        # but Day of Week and Month are useful for any frequency >= Day.
        
        # Day of week - Cyclic Encoding (0-6)
        if config.get("auto_dofw", True):
            if 'feat_dofw_sin' not in df.columns:
                df['feat_dofw_sin'] = np.sin(2 * np.pi * temp_dates.dt.dayofweek / 7)
                new_features.append('feat_dofw_sin')
            if 'feat_dofw_cos' not in df.columns:
                df['feat_dofw_cos'] = np.cos(2 * np.pi * temp_dates.dt.dayofweek / 7)
                new_features.append('feat_dofw_cos')
        
        # Month - Cyclic Encoding (1-12)
        if config.get("auto_month", True):
            if 'feat_month_sin' not in df.columns:
                df['feat_month_sin'] = np.sin(2 * np.pi * (temp_dates.dt.month - 1) / 12)
                new_features.append('feat_month_sin')
            if 'feat_month_cos' not in df.columns:
                df['feat_month_cos'] = np.cos(2 * np.pi * (temp_dates.dt.month - 1) / 12)
                new_features.append('feat_month_cos')
            
        # Fourier terms for yearly seasonality (K=2)
        # Only really makes sense for data with at least daily resolution or higher
        mode_freq = temp_dates.diff().mode()
        is_daily_or_higher = not mode_freq.empty and mode_freq[0] <= pd.Timedelta(days=1)
        
        if config.get("auto_fourier", True) and is_daily_or_higher:
            day_of_year = temp_dates.dt.dayofyear
            for k in range(1, 4): # Increased to 3 harmonics for better capture
                sin_col = f'feat_sin_year_{k}'
                cos_col = f'feat_cos_year_{k}'
                
                if sin_col not in df.columns:
                    df[sin_col] = np.sin(2 * np.pi * k * day_of_year / 365.25)
                    new_features.append(sin_col)
                    
                if cos_col not in df.columns:
                    df[cos_col] = np.cos(2 * np.pi * k * day_of_year / 365.25)
                    new_features.append(cos_col)
        elif config.get("auto_fourier", True):
            logger.info("Skipping Fourier features as frequency detected is lower than daily.")
                    
    except Exception as e:
        logger.warning(f"Failed to generate automated features: {e}")
        
    return df, new_features
