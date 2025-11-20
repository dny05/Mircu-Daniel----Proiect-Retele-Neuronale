"""
Preprocessing Module
Signal filtering and normalization for telemetry data
"""

import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Optional
import yaml
import logging
from typing import Optional


logger = logging.getLogger(__name__)


class TelemetryPreprocessor:
    """Preprocesses raw telemetry signals"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize preprocessor with configuration
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.preproc_config = self.config['preprocessing']
        self.scaler = None
        
    def apply_butterworth_filter(
        self, 
        data: np.ndarray, 
        sampling_rate: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply Butterworth low-pass filter to remove noise
        
        Args:
            data: Input signal array (samples, channels)
            sampling_rate: Sampling rate in Hz (if None, uses config default)
            
        Returns:
            Filtered signal array
        """
        if sampling_rate is None:
            sampling_rate = self.preproc_config['butterworth']['sampling_rate']
        
        order = self.preproc_config['butterworth']['order']
        cutoff = self.preproc_config['butterworth']['cutoff_frequency']
        
        # Calculate normalized cutoff frequency (Nyquist frequency = sampling_rate / 2)
        nyquist = sampling_rate / 2
        normalized_cutoff = cutoff / nyquist
        
        # Design Butterworth filter
        b, a = signal.butter(order, normalized_cutoff, btype='low', analog=False)
        
        # Apply filter to each channel
        filtered_data = np.zeros_like(data)
        
        if data.ndim == 1:
            # Single channel
            filtered_data = signal.filtfilt(b, a, data)
        else:
            # Multiple channels
            for i in range(data.shape[1]):
                filtered_data[:, i] = signal.filtfilt(b, a, data[:, i])
        
        logger.debug(f"Applied Butterworth filter (order={order}, cutoff={cutoff}Hz)")
        
        return filtered_data
    
    def normalize_data(
        self, 
        data: np.ndarray, 
        method: Optional[str] = None,
        fit: bool = True
    ) -> np.ndarray:
        """
        Normalize data using standardization or min-max scaling
        
        Args:
            data: Input data array (samples, features)
            method: 'standardize' or 'minmax' (if None, uses config default)
            fit: If True, fit the scaler. If False, use existing scaler
            
        Returns:
            Normalized data array
        """
        if method is None:
            method = self.preproc_config['normalization']
        
        # Initialize scaler if needed
        if self.scaler is None or fit:
            if method == 'standardize':
                self.scaler = StandardScaler()
            elif method == 'minmax':
                self.scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown normalization method: {method}")
        
        # Reshape if 1D
        original_shape = data.shape
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Fit and transform or just transform
        if fit:
            normalized_data = self.scaler.fit_transform(data)
            logger.debug(f"Fitted and transformed data using {method}")
        else:
            normalized_data = self.scaler.transform(data)
            logger.debug(f"Transformed data using existing {method} scaler")
        
        # Reshape back if needed
        if len(original_shape) == 1:
            normalized_data = normalized_data.flatten()
        
        return normalized_data
    
    def create_windows(
        self, 
        data: np.ndarray, 
        window_size: Optional[int] = None,
        overlap: Optional[float] = None
    ) -> np.ndarray:
        """
        Create sliding windows from continuous data
        
        Args:
            data: Input data (samples, channels)
            window_size: Number of samples per window (if None, uses config)
            overlap: Overlap fraction 0-1 (if None, uses config)
            
        Returns:
            Array of windows (n_windows, window_size, channels)
        """
        if window_size is None:
            window_size = self.preproc_config['window_size']
        if overlap is None:
            overlap = self.preproc_config['window_overlap']
        
        step_size = int(window_size * (1 - overlap))
        
        # Calculate number of windows
        n_samples = len(data)
        n_windows = (n_samples - window_size) // step_size + 1
        
        # Handle multi-dimensional data
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        n_channels = data.shape[1]
        
        # Create windows
        windows = np.zeros((n_windows, window_size, n_channels))
        
        for i in range(n_windows):
            start_idx = i * step_size
            end_idx = start_idx + window_size
            
            if end_idx <= n_samples:
                windows[i] = data[start_idx:end_idx, :]
        
        logger.info(f"Created {n_windows} windows (size={window_size}, overlap={overlap*100}%)")
        
        return windows
    
    def preprocess_pipeline(
        self, 
        data: np.ndarray,
        sampling_rate: Optional[float] = None,
        apply_filter: bool = True,
        apply_normalization: bool = True,
        create_windows_flag: bool = True,
        fit_scaler: bool = True
    ) -> Tuple[np.ndarray, dict]:
        """
        Complete preprocessing pipeline
        
        Args:
            data: Raw input data
            sampling_rate: Sampling rate in Hz
            apply_filter: Whether to apply Butterworth filter
            apply_normalization: Whether to normalize data
            create_windows_flag: Whether to create windows
            fit_scaler: Whether to fit the scaler
            
        Returns:
            Tuple of (processed_data, metadata)
        """
        metadata = {
            'original_shape': data.shape,
            'steps_applied': []
        }
        
        processed_data = data.copy()
        
        # Step 1: Filtering
        if apply_filter:
            processed_data = self.apply_butterworth_filter(processed_data, sampling_rate)
            metadata['steps_applied'].append('butterworth_filter')
        
        # Step 2: Normalization
        if apply_normalization:
            processed_data = self.normalize_data(processed_data, fit=fit_scaler)
            metadata['steps_applied'].append('normalization')
        
        # Step 3: Windowing
        if create_windows_flag:
            processed_data = self.create_windows(processed_data)
            metadata['steps_applied'].append('windowing')
            metadata['n_windows'] = processed_data.shape[0]
            metadata['window_size'] = processed_data.shape[1]
        
        metadata['final_shape'] = processed_data.shape
        
        logger.info(f"Preprocessing complete: {metadata['steps_applied']}")
        
        return processed_data, metadata
    
    def denormalize_data(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform normalized data back to original scale
        
        Args:
            data: Normalized data
            
        Returns:
            Denormalized data
        """
        if self.scaler is None:
            raise ValueError("No scaler has been fitted yet")
        
        original_shape = data.shape
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        denormalized = self.scaler.inverse_transform(data)
        
        if len(original_shape) == 1:
            denormalized = denormalized.flatten()
        
        return denormalized


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Generate synthetic signal
    t = np.linspace(0, 10, 1000)
    signal_clean = np.sin(2 * np.pi * 2 * t)
    noise = np.random.normal(0, 0.1, len(t))
    signal_noisy = signal_clean + noise
    
    # Initialize preprocessor
    preprocessor = TelemetryPreprocessor()
    
    # Test filtering
    filtered = preprocessor.apply_butterworth_filter(signal_noisy, sampling_rate=100)
    print(f"Filtered signal shape: {filtered.shape}")
    
    # Test full pipeline
    processed, meta = preprocessor.preprocess_pipeline(signal_noisy.reshape(-1, 1))
    print(f"Processed shape: {processed.shape}")
    print(f"Metadata: {meta}")