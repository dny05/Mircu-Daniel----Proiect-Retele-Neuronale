"""
Neural Network Models
Classifier for oversteer/understeer and regressor for virtual suspension sensor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
import yaml
import logging
from typing import Optional


logger = logging.getLogger(__name__)


class OversteerUndersteerClassifier(nn.Module):
    """
    MLP Classifier for detecting oversteer/understeer behavior
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize classifier
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__()
        
        if config is None:
            with open("config/model_config.yaml", 'r') as f:
                full_config = yaml.safe_load(f)
                config = full_config['classifier']['architecture']
        
        self.config = config
        
        # Build layers
        layers = []
        input_size = config['input_size']
        
        for hidden_size in config['hidden_layers']:
            layers.append(nn.Linear(input_size, hidden_size))
            
            if config.get('use_batch_norm', False):
                layers.append(nn.BatchNorm1d(hidden_size))
            
            layers.append(nn.ReLU())
            
            if config.get('dropout', 0) > 0:
                layers.append(nn.Dropout(config['dropout']))
            
            input_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(input_size, config['output_size']))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Initialized classifier with {self.count_parameters()} parameters")
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input features (batch_size, input_size)
            
        Returns:
            Class logits (batch_size, 2)
        """
        return self.network(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities
        
        Args:
            x: Input features
            
        Returns:
            Class probabilities (batch_size, 2)
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Predict class labels
        
        Args:
            x: Input features
            threshold: Decision threshold
            
        Returns:
            Predicted classes (0: understeer, 1: oversteer)
        """
        probs = self.predict_proba(x)
        return (probs[:, 1] > threshold).long()
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SuspensionVirtualSensor(nn.Module):
    """
    MLP Regressor for predicting suspension travel from IMU data
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize virtual sensor
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__()
        
        if config is None:
            with open("config/model_config.yaml", 'r') as f:
                full_config = yaml.safe_load(f)
                config = full_config['regressor']['architecture']
        
        self.config = config
        
        # Build layers
        layers = []
        input_size = config['input_size']
        
        for hidden_size in config['hidden_layers']:
            layers.append(nn.Linear(input_size, hidden_size))
            
            if config.get('use_batch_norm', False):
                layers.append(nn.BatchNorm1d(hidden_size))
            
            layers.append(nn.ReLU())
            
            if config.get('dropout', 0) > 0:
                layers.append(nn.Dropout(config['dropout']))
            
            input_size = hidden_size
        
        # Output layer (linear activation for regression)
        layers.append(nn.Linear(input_size, config['output_size']))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Initialized virtual sensor with {self.count_parameters()} parameters")
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input features from IMU (batch_size, input_size)
            
        Returns:
            Predicted suspension travel (batch_size, 4) [FL, FR, RL, RR]
        """
        return self.network(x)
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EnsembleModel(nn.Module):
    """
    Ensemble of multiple models for improved predictions
    """
    
    def __init__(self, model_class, n_models: int = 3, **model_kwargs):
        """
        Initialize ensemble
        
        Args:
            model_class: Class of base model
            n_models: Number of models in ensemble
            **model_kwargs: Arguments for base model
        """
        super().__init__()
        
        self.models = nn.ModuleList([
            model_class(**model_kwargs) for _ in range(n_models)
        ])
        
        self.n_models = n_models
        
        logger.info(f"Initialized ensemble with {n_models} models")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - average predictions
        
        Args:
            x: Input features
            
        Returns:
            Averaged predictions
        """
        predictions = torch.stack([model(x) for model in self.models])
        return predictions.mean(dim=0)
    
    def predict_with_uncertainty(self, x: torch.Tensor) -> tuple:
        """
        Predict with uncertainty estimation
        
        Args:
            x: Input features
            
        Returns:
            Tuple of (mean_prediction, std_prediction)
        """
        predictions = torch.stack([model(x) for model in self.models])
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        return mean_pred, std_pred


def create_model(model_type: str = "classifier", device: str = "cpu"):
    """
    Factory function to create models
    
    Args:
        model_type: Type of model ('classifier', 'regressor', or 'ensemble')
        device: Device to place model on
        
    Returns:
        Initialized model
    """
    with open("config/model_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    if model_type == "classifier":
        model = OversteerUndersteerClassifier(config['classifier']['architecture'])
    elif model_type == "regressor":
        model = SuspensionVirtualSensor(config['regressor']['architecture'])
    elif model_type == "ensemble_classifier":
        n_models = config.get('ensemble', {}).get('n_models', 3)
        model = EnsembleModel(
            OversteerUndersteerClassifier, 
            n_models=n_models,
            config=config['classifier']['architecture']
        )
    elif model_type == "ensemble_regressor":
        n_models = config.get('ensemble', {}).get('n_models', 3)
        model = EnsembleModel(
            SuspensionVirtualSensor,
            n_models=n_models,
            config=config['regressor']['architecture']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    logger.info(f"Created {model_type} on {device}")
    
    return model


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create classifier
    classifier = create_model("classifier")
    print(f"Classifier parameters: {classifier.count_parameters()}")
    
    # Test forward pass
    batch_size = 16
    input_size = 30
    x = torch.randn(batch_size, input_size)
    
    output = classifier(x)
    print(f"Classifier output shape: {output.shape}")
    
    probs = classifier.predict_proba(x)
    print(f"Probabilities shape: {probs.shape}")
    print(f"Sample probabilities: {probs[0]}")
    
    # Create regressor
    regressor = create_model("regressor")
    print(f"\nRegressor parameters: {regressor.count_parameters()}")
    
    # Test regressor
    x_imu = torch.randn(batch_size, 18)
    susp_pred = regressor(x_imu)
    print(f"Predicted suspension shape: {susp_pred.shape}")