"""
Feature Extraction Module
Extract statistical and frequency-domain features from windowed signals
"""

import numpy as np
from scipy import stats
from scipy.fft import fft, fftfreq
from typing import List, Dict
import yaml
import logging
from typing import Optional


logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extracts features from windowed telemetry data"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize feature extractor
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.feature_config = self.config['features']
        self.feature_functions = self._get_feature_functions()
        
    def _get_feature_functions(self) -> Dict:
        """Map feature names to extraction functions"""
        return {
            'mean': np.mean,
            'std': np.std,
            'min': np.min,
            'max': np.max,
            'rms': lambda x: np.sqrt(np.mean(x**2)),
            'peak_to_peak': lambda x: np.ptp(x),
            'median': np.median,
            'skewness': lambda x: stats.skew(x),
            'kurtosis': lambda x: stats.kurtosis(x)
        }
    
    def extract_statistical_features(self, window: np.ndarray) -> np.ndarray:
        """
        Extract statistical features from a single window
        
        Args:
            window: Signal window (window_size, n_channels)
            
        Returns:
            Feature vector (n_features,)
        """
        features = []
        
        # For each channel
        for channel_idx in range(window.shape[1]):
            channel_data = window[:, channel_idx]
            
            # Extract each statistical feature
            for feature_name in self.feature_config['statistical']:
                if feature_name in self.feature_functions:
                    feature_value = self.feature_functions[feature_name](channel_data)
                    features.append(feature_value)
        
        return np.array(features)
    
    def extract_frequency_features(
        self, 
        window: np.ndarray, 
        sampling_rate: float = 100.0
    ) -> np.ndarray:
        """
        Extract frequency-domain features using FFT
        
        Args:
            window: Signal window (window_size, n_channels)
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Frequency feature vector
        """
        features = []
        
        for channel_idx in range(window.shape[1]):
            channel_data = window[:, channel_idx]
            
            # Compute FFT
            n = len(channel_data)
            yf = fft(channel_data)
            xf = fftfreq(n, 1/sampling_rate)
            
            # Get magnitude spectrum (positive frequencies only)
            magnitude = np.abs(yf[:n//2])
            freqs = xf[:n//2]
            
            # Extract features
            # Dominant frequency
            dominant_freq = freqs[np.argmax(magnitude)]
            features.append(dominant_freq)
            
            # Spectral centroid
            spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
            features.append(spectral_centroid)
            
            # Spectral energy
            spectral_energy = np.sum(magnitude**2)
            features.append(spectral_energy)
        
        return np.array(features)
    
    def extract_features_from_windows(
        self, 
        windows: np.ndarray,
        sampling_rate: float = 100.0,
        include_frequency: bool = None
    ) -> np.ndarray:
        """
        Extract features from multiple windows
        
        Args:
            windows: Array of windows (n_windows, window_size, n_channels)
            sampling_rate: Sampling rate in Hz
            include_frequency: Whether to include frequency features
            
        Returns:
            Feature matrix (n_windows, n_features)
        """
        if include_frequency is None:
            include_frequency = self.feature_config.get('frequency_domain', False)
        
        n_windows = windows.shape[0]
        feature_list = []
        
        for i in range(n_windows):
            window = windows[i]
            
            # Statistical features
            stat_features = self.extract_statistical_features(window)
            
            # Frequency features (optional)
            if include_frequency:
                freq_features = self.extract_frequency_features(window, sampling_rate)
                combined_features = np.concatenate([stat_features, freq_features])
            else:
                combined_features = stat_features
            
            feature_list.append(combined_features)
        
        feature_matrix = np.array(feature_list)
        
        logger.info(f"Extracted features from {n_windows} windows: "
                   f"shape={feature_matrix.shape}")
        
        return feature_matrix
    
    def extract_suspension_independent_features(
        self, 
        windows: np.ndarray,
        sampling_rate: float = 100.0
    ) -> np.ndarray:
        """
        Extract features only from acceleration and rotation (for virtual sensor)
        
        Args:
            windows: Array of windows (n_windows, window_size, n_channels)
                    Assumes channels 0-2: acceleration, 3-5: rotation
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Feature matrix without suspension data
        """
        # Extract only acc and rot channels (assuming last 6 channels)
        acc_rot_windows = windows[:, :, -6:]  # Last 6 channels
        
        features = self.extract_features_from_windows(
            acc_rot_windows, 
            sampling_rate, 
            include_frequency=False
        )
        
        logger.info(f"Extracted suspension-independent features: shape={features.shape}")
        
        return features
    
    def get_feature_names(
        self, 
        channel_names: List[str],
        include_frequency: bool = None
    ) -> List[str]:
        """
        Generate feature names for interpretability
        
        Args:
            channel_names: List of channel names
            include_frequency: Whether frequency features are included
            
        Returns:
            List of feature names
        """
        if include_frequency is None:
            include_frequency = self.feature_config.get('frequency_domain', False)
        
        feature_names = []
        
        # Statistical features
        for channel in channel_names:
            for stat in self.feature_config['statistical']:
                feature_names.append(f"{channel}_{stat}")
        
        # Frequency features
        if include_frequency:
            for channel in channel_names:
                feature_names.append(f"{channel}_dominant_freq")
                feature_names.append(f"{channel}_spectral_centroid")
                feature_names.append(f"{channel}_spectral_energy")
        
        return feature_names
    
    def compute_correlation_features(self, window: np.ndarray) -> np.ndarray:
        """
        Compute cross-correlation features between channels
        
        Args:
            window: Signal window (window_size, n_channels)
            
        Returns:
            Correlation features
        """
        n_channels = window.shape[1]
        corr_features = []
        
        # Compute correlation matrix
        corr_matrix = np.corrcoef(window.T)
        
        # Extract upper triangular (excluding diagonal)
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                corr_features.append(corr_matrix[i, j])
        
        return np.array(corr_features)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Generate synthetic windowed data
    # 10 windows, 200 samples per window, 4 channels
    windows = np.random.randn(10, 200, 4)
    
    # Initialize extractor
    extractor = FeatureExtractor()
    
    # Extract features
    features = extractor.extract_features_from_windows(windows)
    print(f"Feature matrix shape: {features.shape}")
    
    # Get feature names
    channel_names = ['susp_fl', 'susp_fr', 'susp_rl', 'susp_rr']
    feature_names = extractor.get_feature_names(channel_names)
    print(f"Number of features: {len(feature_names)}")
    print(f"First 5 features: {feature_names[:5]}")