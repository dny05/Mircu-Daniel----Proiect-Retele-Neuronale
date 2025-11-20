"""
Generate Sample Telemetry Data
Creates synthetic CSV files for testing the application
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_telemetry_data(
    duration_seconds: float = 60.0,
    sampling_rate: float = 100.0,
    behavior_type: str = "neutral",
    output_path: str = None
) -> pd.DataFrame:
    """
    Generate synthetic telemetry data
    
    Args:
        duration_seconds: Duration of telemetry in seconds
        sampling_rate: Sampling rate in Hz
        behavior_type: 'understeer', 'oversteer', or 'neutral'
        output_path: Path to save CSV (if None, returns DataFrame)
        
    Returns:
        DataFrame with telemetry data
    """
    n_samples = int(duration_seconds * sampling_rate)
    dt = 1.0 / sampling_rate
    
    # Generate time array
    timestamps = np.arange(n_samples) * dt * 1000  # Convert to milliseconds
    
    # Generate base signals with noise
    t = np.linspace(0, duration_seconds, n_samples)
    
    # Suspension travel (simulate road bumps + body roll)
    # Base road profile
    road_profile = 0.02 * np.sin(2 * np.pi * 0.5 * t)  # 0.5 Hz bumps
    road_profile += 0.01 * np.sin(2 * np.pi * 2.0 * t)  # 2 Hz vibrations
    road_profile += 0.005 * np.random.randn(n_samples)  # Random noise
    
    # Simulate cornering effects
    cornering = 0.03 * np.sin(2 * np.pi * 0.1 * t)  # Slow corners
    
    # Suspension travel for each corner
    susp_fl = road_profile + cornering + 0.01 * np.random.randn(n_samples)
    susp_fr = road_profile - cornering + 0.01 * np.random.randn(n_samples)
    susp_rl = road_profile + cornering * 0.8 + 0.01 * np.random.randn(n_samples)
    susp_rr = road_profile - cornering * 0.8 + 0.01 * np.random.randn(n_samples)
    
    # Adjust based on behavior type
    if behavior_type == "understeer":
        # Front suspension compresses more in corners
        susp_fl += cornering * 0.5
        susp_fr -= cornering * 0.5
    elif behavior_type == "oversteer":
        # Rear suspension compresses more
        susp_rl += cornering * 0.5
        susp_rr -= cornering * 0.5
    
    # Acceleration (simulate braking, acceleration, lateral G)
    acc_x = 0.3 * np.sin(2 * np.pi * 0.15 * t)  # Longitudinal (braking/accel)
    acc_x += 0.1 * np.random.randn(n_samples)
    
    acc_y = 0.5 * np.sin(2 * np.pi * 0.1 * t)  # Lateral (cornering)
    acc_y += 0.1 * np.random.randn(n_samples)
    
    # Modify lateral acceleration based on behavior
    if behavior_type == "understeer":
        acc_y *= 0.8  # Less lateral grip
    elif behavior_type == "oversteer":
        acc_y *= 1.2  # More lateral movement
    
    acc_z = 9.81 + 0.5 * np.sin(2 * np.pi * 0.2 * t)  # Vertical
    acc_z += 0.2 * np.random.randn(n_samples)
    
    # Rotation (pitch, yaw, roll)
    rot_x = 0.1 * np.sin(2 * np.pi * 0.2 * t)  # Roll (deg/s)
    rot_x += 0.02 * np.random.randn(n_samples)
    
    rot_y = 0.05 * np.sin(2 * np.pi * 0.15 * t)  # Pitch (deg/s)
    rot_y += 0.01 * np.random.randn(n_samples)
    
    rot_z = 0.15 * np.sin(2 * np.pi * 0.1 * t)  # Yaw (deg/s)
    rot_z += 0.03 * np.random.randn(n_samples)
    
    # Modify rotation based on behavior
    if behavior_type == "understeer":
        rot_z *= 0.7  # Less rotation (car doesn't turn as much)
    elif behavior_type == "oversteer":
        rot_z *= 1.3  # More rotation (car turns more)
    
    # Create DataFrame
    data = {
        'index': np.arange(n_samples),
        'elapse_time': timestamps.astype(int),
        'susp_fl': susp_fl,
        'susp_fr': susp_fr,
        'susp_rl': susp_rl,
        'susp_rr': susp_rr,
        'acc_x': acc_x,
        'acc_y': acc_y,
        'acc_z': acc_z,
        'rot_x': rot_x,
        'rot_y': rot_y,
        'rot_z': rot_z
    }
    
    df = pd.DataFrame(data)
    
    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved telemetry data to {output_path}")
    
    return df


def main():
    """Generate sample datasets"""
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate different behavior types
    behaviors = [
        ("neutral", "neutral_behavior.csv"),
        ("understeer", "understeer_behavior.csv"),
        ("oversteer", "oversteer_behavior.csv")
    ]
    
    for behavior_type, filename in behaviors:
        logger.info(f"Generating {behavior_type} telemetry...")
        generate_telemetry_data(
            duration_seconds=60.0,
            sampling_rate=100.0,
            behavior_type=behavior_type,
            output_path=output_dir / filename
        )
    
    logger.info("✓ Sample data generation complete!")
    logger.info(f"Files saved in: {output_dir}")


if __name__ == "__main__":
    main()