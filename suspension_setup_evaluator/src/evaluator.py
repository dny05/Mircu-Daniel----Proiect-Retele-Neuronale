"""
Evaluator Module
Evaluates telemetry data and provides setup recommendations
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import yaml
import logging
import json
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class SetupEvaluator:
    """Evaluates telemetry and provides suspension setup recommendations"""
    
    def __init__(
        self,
        classifier_model: torch.nn.Module,
        config_path: str = "config/config.yaml"
    ):
        """
        Initialize evaluator
        
        Args:
            classifier_model: Trained classifier model
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.classifier = classifier_model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classifier.to(self.device)
        self.classifier.eval()
        
        self.eval_config = self.config['evaluation']
        
        logger.info("Setup evaluator initialized")
    
    def evaluate_windows(self, features: np.ndarray) -> Dict:
        """
        Evaluate features from all windows
        
        Args:
            features: Feature matrix (n_windows, n_features)
            
        Returns:
            Dictionary with evaluation results
        """
        # Convert to tensor
        X = torch.FloatTensor(features).to(self.device)
        
        with torch.no_grad():
            # Get predictions
            logits = self.classifier(X)
            probs = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probs, dim=1)
            
            # Convert to numpy
            probs_np = probs.cpu().numpy()
            predictions_np = predictions.cpu().numpy()
        
        # Analyze results
        n_windows = len(predictions_np)
        n_understeer = np.sum(predictions_np == 0)
        n_oversteer = np.sum(predictions_np == 1)
        
        # Calculate confidence scores
        confidence_scores = np.max(probs_np, axis=1)
        avg_confidence = np.mean(confidence_scores)
        
        # Determine dominant behavior
        understeer_ratio = n_understeer / n_windows
        oversteer_ratio = n_oversteer / n_windows
        
        if understeer_ratio > oversteer_ratio:
            dominant_behavior = "understeer"
            behavior_confidence = understeer_ratio
        else:
            dominant_behavior = "oversteer"
            behavior_confidence = oversteer_ratio
        
        results = {
            'n_windows': n_windows,
            'n_understeer': int(n_understeer),
            'n_oversteer': int(n_oversteer),
            'understeer_ratio': float(understeer_ratio),
            'oversteer_ratio': float(oversteer_ratio),
            'dominant_behavior': dominant_behavior,
            'behavior_confidence': float(behavior_confidence),
            'avg_confidence': float(avg_confidence),
            'predictions_per_window': predictions_np.tolist(),
            'probabilities_per_window': probs_np.tolist()
        }
        
        logger.info(f"Evaluation complete: {dominant_behavior} detected "
                   f"({behavior_confidence*100:.1f}% of windows)")
        
        return results
    
    def generate_recommendations(self, evaluation_results: Dict) -> Dict:
        """
        Generate setup recommendations based on evaluation
        
        Args:
            evaluation_results: Results from evaluate_windows()
            
        Returns:
            Dictionary with recommendations
        """
        dominant_behavior = evaluation_results['dominant_behavior']
        confidence = evaluation_results['behavior_confidence']
        
        min_confidence = self.eval_config['min_confidence']
        
        recommendations = {
            'timestamp': datetime.now().isoformat(),
            'detected_behavior': dominant_behavior,
            'confidence': confidence,
            'reliable': confidence >= min_confidence,
            'recommendations': {}
        }
        
        # Get recommendation rules from config
        if confidence >= min_confidence:
            behavior_config = self.eval_config['recommendations'][dominant_behavior]
            
            recommendations['recommendations'] = {
                'front_camber': behavior_config['front_camber'],
                'front_toe': behavior_config['front_toe'],
                'rear_camber': behavior_config['rear_camber'],
                'message': behavior_config['message']
            }
            
            # Add specific adjustment suggestions
            if dominant_behavior == "understeer":
                recommendations['specific_actions'] = [
                    "Creștere camber negativ față (ex: -1.5° → -2.0°)",
                    "Creștere toe-out față (ex: 0° → 0.1° per roată)",
                    "Reducere camber spate dacă posibil",
                    "Verificare presiune pneuri față (posibil sub-gonflare)"
                ]
            else:  # oversteer
                recommendations['specific_actions'] = [
                    "Creștere camber negativ spate (ex: -1.0° → -1.5°)",
                    "Reducere toe-out față",
                    "Reducere camber față dacă posibil",
                    "Verificare presiune pneuri spate (posibil sub-gonflare)"
                ]
        else:
            recommendations['recommendations'] = {
                'message': f"Confidence prea scăzută ({confidence*100:.1f}%) pentru recomandări sigure. "
                          f"Sunt necesare mai multe date sau verificarea calității senzorilor."
            }
            recommendations['specific_actions'] = [
                "Rulați mai multe sesiuni de testare",
                "Verificați calibrarea senzorilor",
                "Asigurați-vă că pilotul conduce consistent"
            ]
        
        return recommendations
    
    def evaluate_telemetry_file(
        self,
        features: np.ndarray,
        output_path: Optional[str] = None
    ) -> Dict:
        """
        Complete evaluation pipeline for a telemetry file
        
        Args:
            features: Extracted features from telemetry
            output_path: Path to save evaluation report
            
        Returns:
            Complete evaluation report
        """
        # Evaluate windows
        evaluation_results = self.evaluate_windows(features)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(evaluation_results)
        
        # Combine into report
        report = {
            'evaluation': evaluation_results,
            'recommendations': recommendations
        }
        
        # Save report if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Saved evaluation report to {output_path}")
        
        return report
    
    def compare_setups(
        self,
        features_setup1: np.ndarray,
        features_setup2: np.ndarray,
        setup1_name: str = "Setup A",
        setup2_name: str = "Setup B"
    ) -> Dict:
        """
        Compare two different suspension setups
        
        Args:
            features_setup1: Features from first setup
            features_setup2: Features from second setup
            setup1_name: Name of first setup
            setup2_name: Name of second setup
            
        Returns:
            Comparison report
        """
        # Evaluate both setups
        results1 = self.evaluate_windows(features_setup1)
        results2 = self.evaluate_windows(features_setup2)
        
        comparison = {
            'setups': {
                setup1_name: results1,
                setup2_name: results2
            },
            'analysis': {}
        }
        
        # Compare behavior ratios
        diff_understeer = results2['understeer_ratio'] - results1['understeer_ratio']
        diff_oversteer = results2['oversteer_ratio'] - results1['oversteer_ratio']
        
        comparison['analysis']['understeer_change'] = float(diff_understeer)
        comparison['analysis']['oversteer_change'] = float(diff_oversteer)
        
        # Determine which setup is better
        if abs(results1['understeer_ratio'] - 0.5) < abs(results2['understeer_ratio'] - 0.5):
            comparison['analysis']['better_setup'] = setup1_name
            comparison['analysis']['reason'] = "Mai echilibrat (mai aproape de neutru)"
        else:
            comparison['analysis']['better_setup'] = setup2_name
            comparison['analysis']['reason'] = "Mai echilibrat (mai aproape de neutru)"
        
        logger.info(f"Setup comparison complete: {comparison['analysis']['better_setup']} "
                   f"este mai bun")
        
        return comparison
    
    def get_behavior_timeline(self, evaluation_results: Dict) -> List[Dict]:
        """
        Create timeline of behavior changes throughout the lap
        
        Args:
            evaluation_results: Results from evaluate_windows()
            
        Returns:
            List of behavior segments
        """
        predictions = evaluation_results['predictions_per_window']
        probabilities = evaluation_results['probabilities_per_window']
        
        timeline = []
        current_behavior = predictions[0]
        segment_start = 0
        
        for i, pred in enumerate(predictions):
            if pred != current_behavior or i == len(predictions) - 1:
                # End current segment
                segment = {
                    'window_start': segment_start,
                    'window_end': i,
                    'behavior': 'understeer' if current_behavior == 0 else 'oversteer',
                    'duration_windows': i - segment_start,
                    'avg_confidence': np.mean([p[current_behavior] for p in probabilities[segment_start:i]])
                }
                timeline.append(segment)
                
                # Start new segment
                current_behavior = pred
                segment_start = i
        
        return timeline


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    from models import create_model
    
    # Create and load model (would normally load trained weights)
    model = create_model("classifier")
    
    # Create evaluator
    evaluator = SetupEvaluator(model)
    
    # Create dummy features
    features = np.random.randn(50, 30)  # 50 windows, 30 features
    
    # Evaluate
    report = evaluator.evaluate_telemetry_file(features)
    
    print("\n=== Evaluation Report ===")
    print(f"Dominant behavior: {report['recommendations']['detected_behavior']}")
    print(f"Confidence: {report['recommendations']['confidence']*100:.1f}%")
    
    if report['recommendations']['reliable']:
        print("\nRecommendations:")
        for action in report['recommendations']['specific_actions']:
            print(f"  • {action}")