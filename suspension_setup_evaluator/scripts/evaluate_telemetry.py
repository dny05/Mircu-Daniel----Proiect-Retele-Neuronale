"""
Evaluation Script
Evaluate telemetry data and generate setup recommendations
"""

import sys
import argparse
from pathlib import Path
import torch
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import TelemetryDataLoader
from preprocessing import TelemetryPreprocessor
from feature_extraction import FeatureExtractor
from models import create_model
from evaluator import SetupEvaluator
from utils import save_evaluation_report

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(model_path: str):
    """
    Load trained model
    
    Args:
        model_path: Path to model checkpoint
        
    Returns:
        Loaded model
    """
    logger.info(f"Loading model from {model_path}...")
    
    model = create_model("classifier")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info("✓ Model loaded successfully")
    
    return model


def process_telemetry(csv_path: str):
    """
    Process telemetry CSV file
    
    Args:
        csv_path: Path to telemetry CSV
        
    Returns:
        Extracted features
    """
    logger.info(f"Processing telemetry: {csv_path}")
    
    # Initialize components
    loader = TelemetryDataLoader()
    preprocessor = TelemetryPreprocessor()
    extractor = FeatureExtractor()
    
    # Load data
    df = loader.load_csv(csv_path)
    quality = loader.validate_data_quality(df)
    
    logger.info(f"Loaded {quality['total_samples']} samples "
               f"({quality['duration_seconds']:.1f}s @ {quality['sampling_rate_hz']:.1f}Hz)")
    
    # Split data
    timestamps, susp_data, acc_data, rot_data = loader.split_data(df)
    
    # Combine sensor data
    import numpy as np
    all_data = np.hstack([susp_data, acc_data, rot_data])
    
    # Preprocess
    processed_windows, metadata = preprocessor.preprocess_pipeline(
        all_data,
        sampling_rate=quality['sampling_rate_hz']
    )
    
    logger.info(f"Created {metadata['n_windows']} windows")
    
    # Extract features
    features = extractor.extract_features_from_windows(
        processed_windows,
        sampling_rate=quality['sampling_rate_hz']
    )
    
    logger.info(f"Extracted {features.shape[1]} features per window")
    
    return features


def evaluate_and_report(
    features,
    model,
    output_path: str = None
):
    """
    Evaluate features and generate report
    
    Args:
        features: Extracted features
        model: Trained model
        output_path: Path to save report
        
    Returns:
        Evaluation report
    """
    logger.info("Running evaluation...")
    
    # Create evaluator
    evaluator = SetupEvaluator(model)
    
    # Evaluate
    report = evaluator.evaluate_telemetry_file(features, output_path)
    
    # Print summary
    print("\n" + "=" * 70)
    print("  SUSPENSION SETUP EVALUATION RESULTS")
    print("=" * 70)
    
    rec = report['recommendations']
    eval_data = report['evaluation']
    
    print(f"\n🔍 Detected Behavior: {rec['detected_behavior'].upper()}")
    print(f"📊 Confidence: {rec['confidence']*100:.1f}%")
    print(f"✓  Reliability: {'HIGH' if rec['reliable'] else 'LOW'}")
    
    print(f"\n📈 Analysis:")
    print(f"   • Windows analyzed: {eval_data['n_windows']}")
    print(f"   • Understeer: {eval_data['understeer_ratio']*100:.1f}%")
    print(f"   • Oversteer: {eval_data['oversteer_ratio']*100:.1f}%")
    
    if rec['reliable']:
        print(f"\n💡 Recommendations:")
        print(f"   {rec['recommendations']['message']}")
        print(f"\n🔧 Specific Actions:")
        for i, action in enumerate(rec['specific_actions'], 1):
            print(f"   {i}. {action}")
    else:
        print(f"\n⚠️  Warning: Confidence too low for reliable recommendations")
        print(f"   Suggestion: Collect more data or check sensor calibration")
    
    print("\n" + "=" * 70 + "\n")
    
    return report


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate telemetry and recommend setup')
    parser.add_argument('csv_file', type=str,
                       help='Path to telemetry CSV file')
    parser.add_argument('--model', type=str, default='data/models/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save evaluation report (JSON)')
    parser.add_argument('--format', type=str, default='json', choices=['json', 'txt'],
                       help='Report format')
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("SUSPENSION SETUP EVALUATOR")
    logger.info("=" * 70)
    
    try:
        # Check if files exist
        csv_path = Path(args.csv_file)
        model_path = Path(args.model)
        
        if not csv_path.exists():
            logger.error(f"CSV file not found: {csv_path}")
            sys.exit(1)
        
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            logger.error("Train a model first using: python scripts/train_model.py")
            sys.exit(1)
        
        # Load model
        model = load_model(args.model)
        
        # Process telemetry
        features = process_telemetry(args.csv_file)
        
        # Evaluate and report
        output_path = args.output
        if output_path is None:
            # Auto-generate output path
            output_dir = Path("data/logs")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            filename = csv_path.stem + f"_evaluation.{args.format}"
            output_path = output_dir / filename
        
        report = evaluate_and_report(features, model, str(output_path))
        
        logger.info(f"✓ Report saved to: {output_path}")
        
        # Save in specified format
        if args.format == 'txt':
            from utils import save_evaluation_report
            txt_path = str(output_path).replace('.json', '.txt')
            save_evaluation_report(report, txt_path, format='txt')
            logger.info(f"✓ Text report saved to: {txt_path}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()