"""
Data Loader Module
Handles loading and validation of telemetry CSV files
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TelemetryDataLoader:
    """Loads and validates telemetry data from CSV files"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize data loader with configuration
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.required_columns = self._get_required_columns()
        
    def _get_required_columns(self) -> List[str]:
        """Get list of required column names from config"""
        cols = [self.config['columns']['timestamp']]
        cols.extend(self.config['columns']['suspension'])
        cols.extend(self.config['columns']['acceleration'])
        cols.extend(self.config['columns']['rotation'])
        return cols
    
    def load_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load telemetry CSV file
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame with telemetry data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If required columns are missing
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        logger.info(f"Loading telemetry data from {filepath}")
        
        # Load CSV
        df = pd.read_csv(filepath)
        
        # Validate columns
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Validate data types
        df = self._validate_datatypes(df)
        
        # Check for NaN values
        if df.isnull().any().any():
            logger.warning("Data contains NaN values. Interpolating...")
            df = df.interpolate(method='linear', limit_direction='both')
        
        logger.info(f"Loaded {len(df)} samples successfully")
        
        return df
    
    def _validate_datatypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all columns are numeric"""
        for col in self.required_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
        """
        Split dataframe into separate arrays for suspension, acceleration, rotation
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple of (timestamps, suspension_data, acceleration_data, rotation_data)
        """
        timestamps = df[self.config['columns']['timestamp']].values
        
        susp_data = df[self.config['columns']['suspension']].values
        acc_data = df[self.config['columns']['acceleration']].values
        rot_data = df[self.config['columns']['rotation']].values
        
        return timestamps, susp_data, acc_data, rot_data
    
    def get_sampling_rate(self, df: pd.DataFrame) -> float:
        """
        Calculate actual sampling rate from timestamps
        
        Args:
            df: Input dataframe
            
        Returns:
            Sampling rate in Hz
        """
        timestamps = df[self.config['columns']['timestamp']].values
        dt = np.diff(timestamps).mean() / 1000.0  # Convert ms to seconds
        sampling_rate = 1.0 / dt
        
        logger.info(f"Calculated sampling rate: {sampling_rate:.2f} Hz")
        
        return sampling_rate
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Perform quality checks on loaded data
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary with quality metrics
        """
        quality_report = {
            'total_samples': len(df),
            'duration_seconds': (df[self.config['columns']['timestamp']].max() - 
                                df[self.config['columns']['timestamp']].min()) / 1000.0,
            'missing_values': df.isnull().sum().to_dict(),
            'sampling_rate_hz': self.get_sampling_rate(df),
            'data_ranges': {}
        }
        
        # Check data ranges for each sensor
        for col in self.required_columns:
            if col != self.config['columns']['timestamp']:
                quality_report['data_ranges'][col] = {
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'mean': df[col].mean(),
                    'std': df[col].std()
                }
        
        return quality_report
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str):
        """
        Save processed data to CSV
        
        Args:
            df: Processed dataframe
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    loader = TelemetryDataLoader()
    
    # This would load actual data - for now just demonstrates the interface
    print("Data loader initialized successfully")
    print(f"Required columns: {loader.required_columns}")