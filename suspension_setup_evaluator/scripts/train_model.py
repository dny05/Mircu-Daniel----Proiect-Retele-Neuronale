"""
Training Script
Train suspension setup classifier on telemetry data
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import TelemetryDataLoader
from preprocessing import TelemetryPreprocessor
from feature_extraction import FeatureExtractor
from models import create_model
from trainer import ModelTrainer
from utils import plot_training_history, ensure_directory_structure

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_and_prepare_data(
    data_dir: str = "data/raw",
    val_split: float = 0.2
):
    """
    Load and prepare training data from CSV files
    
    Args:
        data_dir: Directory containing CSV files
        val_split: Validation split ratio
        
    Returns:
        Tuple of (features, labels)
    """
    logger.info("Loading training data...")
    
    data_dir = Path(data_dir)
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        logger.warning("No CSV files found. Generating synthetic data...")
        return generate_synthetic_data()
    
    loader = TelemetryDataLoader()
    preprocessor = TelemetryPreprocessor()
    extractor = FeatureExtractor()
    
    all_features = []
    all_labels = []
    
    # Load each file
    for csv_file in csv_files:
        logger.info(f"Processing {csv_file.name}...")
        
        # Load data
        df = loader.load_csv(csv_file)
        timestamps, susp_data, acc_data, rot_data = loader.split_data(df)
        
        # Combine all sensor data
        all_data = np.hstack([susp_data, acc_data, rot_data])
        
        # Preprocess
        sampling_rate = loader.get_sampling_rate(df)
        processed_windows, _ = preprocessor.preprocess_pipeline(
            all_data,
            sampling_rate=sampling_rate
        )
        
        # Extract features
        features = extractor.extract_features_from_windows(
            processed_windows,
            sampling_rate=sampling_rate
        )
        
        # Determine labels based on filename
        if "understeer" in csv_file.name.lower():
            labels = np.zeros(len(features), dtype=int)  # 0 = understeer
        elif "oversteer" in csv_file.name.lower():
            labels = np.ones(len(features), dtype=int)   # 1 = oversteer
        else:
            # Neutral or unknown - split 50/50
            labels = np.random.randint(0, 2, len(features))
        
        all_features.append(features)
        all_labels.append(labels)
    
    # Concatenate all data
    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    
    logger.info(f"Loaded {len(X)} samples with {X.shape[1]} features")
    logger.info(f"Class distribution: {np.bincount(y)}")
    
    return X, y


def generate_synthetic_data(n_samples: int = 2000):
    """
    Generate synthetic training data
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        Tuple of (features, labels)
    """
    logger.info(f"Generating {n_samples} synthetic samples...")
    
    n_features = 30
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Create labels with some logic
    # Understeer: certain features have positive correlation
    # Oversteer: certain features have negative correlation
    
    # Simple rule: sum of first 5 features
    decision_value = X[:, 0] + X[:, 1] + X[:, 2] - X[:, 3] - X[:, 4]
    y = (decision_value > 0).astype(int)
    
    # Add some noise to make it more realistic
    noise_idx = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
    y[noise_idx] = 1 - y[noise_idx]
    
    logger.info(f"Generated data - Class distribution: {np.bincount(y)}")
    
    return X, y


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 50,
    save_dir: str = "data/models"
):
    """
    Train the classifier model
    
    Args:
        X_train: Training features
        y_train: Training labels
        epochs: Number of training epochs
        save_dir: Directory to save model
        
    Returns:
        Training history
    """
    logger.info("Initializing model...")
    
    # Create model
    model = create_model("classifier")
    
    # Initialize trainer
    trainer = ModelTrainer(model)
    trainer.setup_optimizer_and_loss("classifier")
    
    # Prepare data
    train_loader, val_loader = trainer.prepare_data(X_train, y_train)
    
    # Train
    logger.info(f"Starting training for {epochs} epochs...")
    history = trainer.train(
        train_loader,
        val_loader,
        epochs=epochs,
        save_dir=save_dir
    )
    
    return history


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train suspension setup classifier')
    parser.add_argument('--data-dir', type=str, default='data/raw',
                       help='Directory containing training CSV files')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--save-dir', type=str, default='data/models',
                       help='Directory to save trained model')
    parser.add_argument('--synthetic', action='store_true',
                       help='Use synthetic data for training')
    
    args = parser.parse_args()
    
    # Ensure directories exist
    ensure_directory_structure()
    
    logger.info("=" * 60)
    logger.info("SUSPENSION SETUP CLASSIFIER TRAINING")
    logger.info("=" * 60)
    
    try:
        # Load or generate data
        if args.synthetic:
            X_train, y_train = generate_synthetic_data()
        else:
            X_train, y_train = load_and_prepare_data(args.data_dir)
        
        # Train model
        history = train_classifier(
            X_train,
            y_train,
            epochs=args.epochs,
            save_dir=args.save_dir
        )
        
        # Plot training history
        plot_path = Path(args.save_dir) / "training_history.png"
        plot_training_history(history, save_path=str(plot_path))
        
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"Model saved to: {args.save_dir}")
        logger.info(f"Final train loss: {history['train_loss'][-1]:.4f}")
        logger.info(f"Final val loss: {history['val_loss'][-1]:.4f}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()