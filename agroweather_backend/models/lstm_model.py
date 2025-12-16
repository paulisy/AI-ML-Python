"""
AgroWeather AI - LSTM Model Architecture
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class RainfallLSTM(nn.Module):
    """
    LSTM model for predicting rainfall
    
    Architecture:
    - 2 LSTM layers (64 units each)
    - Dropout layers (prevent overfitting)
    - Dense layer (32 units)
    - Output layer (1 unit - rainfall prediction)
    """
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        """
        Initialize the LSTM model
        
        Args:
            input_size: Number of features
            hidden_size: Number of LSTM units per layer
            num_layers: Number of LSTM layers
            dropout: Dropout rate (0.0-1.0)
        """
        super(RainfallLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Dropout layer after LSTM
        self.dropout_layer = nn.Dropout(dropout)

        # Dense (fully connected) layer
        self.fc1 = nn.Linear(hidden_size, 32)

        # Activation function
        self.relu = nn.ReLU()
        
        # Output layer
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
        
        Returns:
            Output tensor of shape (batch_size, 1) - predicted rainfall
        """
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        out, (h_n, c_n) = self.lstm(x, (h0, c0))

        # Use output from the last time step
        out = out[:, -1, :]  # Shape: (batch, hidden_size)

        # Apply dropout
        out = self.dropout_layer(out)
        
        # Dense layer
        out = self.fc1(out)  # Shape: (batch, 32)
        out = self.relu(out)
        
        # Output layer
        out = self.fc2(out)  # Shape: (batch, 1)
        
        return out

    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(input_size: int, device: str = 'cpu', **kwargs) -> RainfallLSTM:
    """
    Create and initialize the LSTM model

    Args:
        input_size: Number of features
        device: Device to use ('cpu' or 'cuda')
        **kwargs: Additional model parameters

    Returns:
        Initialized LSTM model
    """
    model = RainfallLSTM(input_size=input_size, **kwargs)
    model = model.to(device)
    return model


def model_summary(model: RainfallLSTM, input_shape: Tuple[int, int, int] = (1, 7, 40)) -> None:
    """
    Print model summary

    Args:
        model: LSTM model
        input_shape: Input shape (batch_size, sequence_length, input_size)
    """
    print("\n" + "="*70)
    print("MODEL ARCHITECTURE SUMMARY")
    print("="*70)

    print(f"\nInput shape: {input_shape}")
    print(f"   → Batch size: {input_shape[0]}")
    print(f"   → Sequence length: {input_shape[1]} days")
    print(f"   → Features: {input_shape[2]}")

    print(f"\nModel Architecture:")
    print(model)

    print(f"\nTotal trainable parameters: {model.count_parameters():,}")
    
    # Calculate approximate model size
    param_size = model.count_parameters() * 4 / (1024**2)  # 4 bytes per float32
    print(f"Approximate model size: {param_size:.2f} MB")
    
    print("\n" + "="*70)