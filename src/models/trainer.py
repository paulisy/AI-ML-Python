"""
Model training utilities for AgroWeather AI
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
import os
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import matplotlib.pyplot as plt

from .lstm_model import RainfallLSTM


class ModelTrainer:
    """
    Handles training and evaluation of LSTM models
    """
    
    def __init__(self, model: RainfallLSTM, device: str = 'cpu'):
        """
        Initialize trainer
        
        Args:
            model: LSTM model to train
            device: Device to use for training
        """
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': [],
            'epochs': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_model_state = None
    
    def create_data_loaders(self, 
                           X_train: np.ndarray, 
                           y_train: np.ndarray,
                           X_val: np.ndarray, 
                           y_val: np.ndarray,
                           batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
        """
        Create PyTorch data loaders
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            batch_size: Batch size for training
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def calculate_metrics(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
        """
        Calculate evaluation metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        with torch.no_grad():
            mse = nn.MSELoss()(y_pred, y_true).item()
            mae = nn.L1Loss()(y_pred, y_true).item()
            rmse = np.sqrt(mse)
            
            # R² score
            y_true_np = y_true.cpu().numpy()
            y_pred_np = y_pred.cpu().numpy()
            
            ss_res = np.sum((y_true_np - y_pred_np) ** 2)
            ss_tot = np.sum((y_true_np - np.mean(y_true_np)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
    
    def train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module) -> Tuple[float, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            criterion: Loss function
            
        Returns:
            Tuple of (average_loss, average_mae)
        """
        self.model.train()
        total_loss = 0
        total_mae = 0
        num_batches = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                mae = nn.L1Loss()(outputs, batch_y)
                total_loss += loss.item()
                total_mae += mae.item()
                num_batches += 1
        
        return total_loss / num_batches, total_mae / num_batches
    
    def validate_epoch(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """
        Validate for one epoch
        
        Args:
            val_loader: Validation data loader
            criterion: Loss function
            
        Returns:
            Tuple of (average_loss, average_mae)
        """
        self.model.eval()
        total_loss = 0
        total_mae = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                mae = nn.L1Loss()(outputs, batch_y)
                
                total_loss += loss.item()
                total_mae += mae.item()
                num_batches += 1
        
        return total_loss / num_batches, total_mae / num_batches
    
    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: np.ndarray,
              y_val: np.ndarray,
              epochs: int = 100,
              batch_size: int = 32,
              learning_rate: float = 0.001,
              patience: int = 10,
              save_dir: str = 'models/saved') -> Dict[str, Any]:
        """
        Train the model
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            patience: Early stopping patience
            save_dir: Directory to save model
            
        Returns:
            Training history and metrics
        """
        print("🚀 Starting LSTM Training")
        print("=" * 50)
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(
            X_train, y_train, X_val, y_val, batch_size
        )
        
        # Setup training
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Early stopping
        patience_counter = 0
        
        print(f"📊 Training Setup:")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Validation samples: {len(X_val)}")
        print()
        
        # Training loop
        for epoch in range(epochs):
            # Train
            train_loss, train_mae = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validate
            val_loss, val_mae = self.validate_epoch(val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_mae'].append(train_mae)
            self.history['val_mae'].append(val_mae)
            self.history['epochs'].append(epoch + 1)
            
            # Check for best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                
                # Save best model
                os.makedirs(save_dir, exist_ok=True)
                torch.save(self.model.state_dict(), f'{save_dir}/rainfall_lstm_best.pth')
            else:
                patience_counter += 1
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Train MAE: {train_mae:.4f} | "
                      f"Val MAE: {val_mae:.4f}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\n⏹️  Early stopping at epoch {epoch+1} (patience: {patience})")
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        # Save final model and history
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.model.state_dict(), f'{save_dir}/rainfall_lstm_full.pth')
        
        with open(f'{save_dir}/training_history.pkl', 'wb') as f:
            pickle.dump(self.history, f)
        
        # Training summary
        print("\n" + "=" * 50)
        print("✅ TRAINING COMPLETE!")
        print(f"   Best validation loss: {self.best_val_loss:.4f}")
        print(f"   Total epochs: {len(self.history['epochs'])}")
        print(f"   Models saved to: {save_dir}")
        
        return {
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'final_epoch': len(self.history['epochs'])
        }
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model on test set
        
        Args:
            X_test, y_test: Test data
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("📊 Evaluating model on test set...")
        
        self.model.eval()
        
        # Convert to tensors
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test).to(self.device)
        
        # Predict
        with torch.no_grad():
            predictions = self.model(X_test_tensor)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test_tensor, predictions)
        
        print(f"   Test MSE: {metrics['mse']:.4f}")
        print(f"   Test MAE: {metrics['mae']:.4f}")
        print(f"   Test RMSE: {metrics['rmse']:.4f}")
        print(f"   Test R²: {metrics['r2']:.4f}")
        
        return metrics
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """
        Plot training history
        
        Args:
            save_path: Path to save plot (optional)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.history['epochs'], self.history['train_loss'], label='Training Loss', color='blue')
        ax1.plot(self.history['epochs'], self.history['val_loss'], label='Validation Loss', color='red')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.legend()
        ax1.grid(True)
        
        # MAE plot
        ax2.plot(self.history['epochs'], self.history['train_mae'], label='Training MAE', color='blue')
        ax2.plot(self.history['epochs'], self.history['val_mae'], label='Validation MAE', color='red')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Training history plot saved to: {save_path}")
        
        plt.show()