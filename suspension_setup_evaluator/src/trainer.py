"""
Model Training Module
Handles training loop, validation, and model checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import yaml
import logging
from tqdm import tqdm
import joblib
from typing import Optional


logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains neural network models with validation and checkpointing"""
    
    def __init__(
        self, 
        model: nn.Module,
        config_path: str = "config/config.yaml",
        model_config_path: str = "config/model_config.yaml"
    ):
        """
        Initialize trainer
        
        Args:
            model: Neural network model to train
            config_path: Path to general config
            model_config_path: Path to model config
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        with open(model_config_path, 'r') as f:
            self.model_config = yaml.safe_load(f)
        
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.train_config = self.config['training']
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        logger.info(f"Trainer initialized on device: {self.device}")
    
    def prepare_data(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        validation_split: Optional[float] = None
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare data loaders for training and validation
        
        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,) or (n_samples, n_outputs)
            validation_split: Fraction for validation (if None, uses config)
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        if validation_split is None:
            validation_split = self.train_config['validation_split']
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y) if y.dtype == np.float32 or y.dtype == np.float64 else torch.LongTensor(y)
        
        # Create dataset
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # Split into train and validation
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = random_split(
            dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders
        batch_size = self.train_config['batch_size']
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0
        )
        
        logger.info(f"Data prepared: Train={len(train_dataset)}, Val={len(val_dataset)}")
        
        return train_loader, val_loader
    
    def setup_optimizer_and_loss(self, model_type: str = "classifier"):
        """
        Setup optimizer and loss function
        
        Args:
            model_type: 'classifier' or 'regressor'
        """
        # Optimizer
        lr = self.train_config['learning_rate']
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Learning rate scheduler
        if self.model_config.get('advanced', {}).get('lr_scheduler', {}).get('enabled', False):
            scheduler_config = self.model_config['advanced']['lr_scheduler']
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 10),
                min_lr=scheduler_config.get('min_lr', 1e-6)
            )
        else:
            self.scheduler = None
        
        # Loss function
        if model_type == "classifier":
            self.criterion = nn.CrossEntropyLoss()
        else:  # regressor
            self.criterion = nn.MSELoss()
        
        logger.info(f"Setup optimizer (lr={lr}) and loss ({self.criterion.__class__.__name__})")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        pbar = tqdm(train_loader, desc="Training")
        
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.model_config.get('advanced', {}).get('gradient_clip', {}).get('enabled', False):
                max_norm = self.model_config['advanced']['gradient_clip']['max_norm']
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / n_batches
        
        return {'train_loss': avg_loss}
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate model
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item()
                n_batches += 1
        
        avg_loss = total_loss / n_batches
        
        return {'val_loss': avg_loss}
    
    def train(
        self, 
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: Optional[int] = None,
        save_dir: str = "data/models"
    ) -> Dict[str, list]:
        """
        Complete training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs (if None, uses config)
            save_dir: Directory to save models
            
        Returns:
            Dictionary with training history
        """
        if epochs is None:
            epochs = self.train_config['epochs']
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch+1}/{epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Record history
            history['train_loss'].append(train_metrics['train_loss'])
            history['val_loss'].append(val_metrics['val_loss'])
            
            # Log
            logger.info(f"Train Loss: {train_metrics['train_loss']:.4f}, "
                       f"Val Loss: {val_metrics['val_loss']:.4f}")
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step(val_metrics['val_loss'])
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"Learning rate: {current_lr:.6f}")
            
            # Save best model
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.save_model(save_dir / "best_model.pth")
                logger.info("✓ Saved best model")
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Checkpoint saving
            if (epoch + 1) % self.config['model']['checkpoint_frequency'] == 0:
                self.save_model(save_dir / f"checkpoint_epoch_{epoch+1}.pth")
            
            # Early stopping
            patience = self.train_config['early_stopping_patience']
            if self.patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Save final model
        self.save_model(save_dir / "final_model.pth")
        
        logger.info("Training completed!")
        
        return history
    
    def save_model(self, filepath: str):
        """Save model state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load model state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Loaded model from {filepath}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    from models import create_model
    
    # Create dummy data
    X_train = np.random.randn(1000, 30)
    y_train = np.random.randint(0, 2, 1000)
    
    # Create model
    model = create_model("classifier")
    
    # Initialize trainer
    trainer = ModelTrainer(model)
    trainer.setup_optimizer_and_loss("classifier")
    
    # Prepare data
    train_loader, val_loader = trainer.prepare_data(X_train, y_train)
    
    # Train (just 2 epochs for demo)
    history = trainer.train(train_loader, val_loader, epochs=2)
    
    print(f"Training history: {history}")