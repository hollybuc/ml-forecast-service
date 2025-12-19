"""
Simple data loader for ML worker
"""
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Load data from CSV files"""
    
    def load_csv(self, file_path: str) -> pd.DataFrame:
        """Load CSV file into DataFrame"""
        logger.info(f"Loading CSV from: {file_path}")
        df = pd.read_csv(file_path)
        
        # Convert numeric columns from string to numeric
        for col in df.columns:
            if col != 'date':  # Skip date column
                try:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except:
                    pass
        
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        logger.info(f"Column types: {df.dtypes.to_dict()}")
        return df

