"""
Utility Functions
Helper functions for the Suspension Setup Evaluator
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import json
import yaml
import logging

logger = logging.getLogger(__name__)


def plot_telemetry_signals(
    timestamps: np.ndarray,
    signals: Dict[str, np.ndarray],
    title: str = "Telemetry Signals",
    save_path: str = None
):
    """
    Plot multiple telemetry signals
    
    Args:
        timestamps: Time array
        signals: Dictionary of signal_name: signal_data
        title: Plot title
        save_path: Path to save figure (if None, shows plot)
    """
    n_signals = len(signals)
    fig, axes = plt.subplots(n_signals, 1, figsize=(12, 3*n_signals), sharex=True)
    
    if n_signals == 1:
        axes = [axes]
    
    for ax, (name, data) in zip(axes, signals.items()):
        ax.plot(timestamps, data, linewidth=0.8)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time (s)')
    axes[0].set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_feature_importance(
    feature_names: List[str],
    importance_scores: np.ndarray,
    top_n: int = 15,
    save_path: str = None
):
    """
    Plot feature importance scores
    
    Args:
        feature_names: List of feature names
        importance_scores: Importance scores for each feature
        top_n: Number of top features to show
        save_path: Path to save figure
    """
    # Sort by importance
    indices = np.argsort(importance_scores)[-top_n:]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(top_n), importance_scores[indices])
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Importance Score')
    plt.title(f'Top {top_n} Most Important Features')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info(f"Saved feature importance plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None,
    save_path: str = None
):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        save_path: Path to save figure
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    if class_names is None:
        class_names = ['Understeer', 'Oversteer']
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info(f"Saved confusion matrix to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: str = None
):
    """
    Plot training and validation loss curves
    
    Args:
        history: Dictionary with 'train_loss' and 'val_loss' keys
        save_path: Path to save figure
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info(f"Saved training history to {save_path}")
    else:
        plt.show()
    
    plt.close()


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None
) -> Dict[str, float]:
    """
    Calculate classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities (optional)
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary'),
        'recall': recall_score(y_true, y_pred, average='binary'),
        'f1_score': f1_score(y_true, y_pred, average='binary')
    }
    
    if y_proba is not None:
        from sklearn.metrics import roc_auc_score
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
    
    return metrics


def save_evaluation_report(
    report: Dict,
    output_path: str,
    format: str = 'json'
):
    """
    Save evaluation report in various formats
    
    Args:
        report: Evaluation report dictionary
        output_path: Output file path
        format: 'json' or 'txt'
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
    
    elif format == 'txt':
        with open(output_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("SUSPENSION SETUP EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            rec = report['recommendations']
            f.write(f"Detected Behavior: {rec['detected_behavior'].upper()}\n")
            f.write(f"Confidence: {rec['confidence']*100:.1f}%\n\n")
            
            if rec['reliable']:
                f.write("RECOMMENDATIONS:\n")
                f.write(f"{rec['recommendations']['message']}\n\n")
                
                f.write("Specific Actions:\n")
                for i, action in enumerate(rec['specific_actions'], 1):
                    f.write(f"  {i}. {action}\n")
    
    logger.info(f"Saved report to {output_path}")


def load_config(config_path: str = "config/config.yaml") -> Dict:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.debug(f"Loaded configuration from {config_path}")
    
    return config


def ensure_directory_structure():
    """Create required directory structure if it doesn't exist"""
    directories = [
        "data/raw",
        "data/processed",
        "data/models",
        "data/logs",
        "config"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logger.info("Directory structure verified")


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def create_summary_statistics(data: np.ndarray) -> Dict:
    """
    Calculate summary statistics for data
    
    Args:
        data: Input data array
        
    Returns:
        Dictionary of statistics
    """
    return {
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'median': float(np.median(data)),
        'q25': float(np.percentile(data, 25)),
        'q75': float(np.percentile(data, 75))
    }


def validate_csv_structure(filepath: str, required_columns: List[str]) -> Tuple[bool, str]:
    """
    Validate CSV file structure
    
    Args:
        filepath: Path to CSV file
        required_columns: List of required column names
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    import pandas as pd
    
    try:
        df = pd.read_csv(filepath, nrows=5)  # Read only first few rows
        
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            return False, f"Missing columns: {missing_cols}"
        
        return True, "Valid"
    
    except Exception as e:
        return False, f"Error reading file: {str(e)}"


if __name__ == "__main__":
    # Test utilities
    logging.basicConfig(level=logging.INFO)
    
    # Test directory creation
    ensure_directory_structure()
    
    # Test time formatting
    print(format_time(45.5))      # 45.5s
    print(format_time(125.3))     # 2m 5.3s
    print(format_time(7250))      # 2h 0m
    
    # Test summary statistics
    data = np.random.randn(1000)
    stats = create_summary_statistics(data)
    print(f"\nSummary Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")