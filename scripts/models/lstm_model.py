"""
AgroWeather AI - LSTM Model Architecture
"""
import torch
import torch.nn as nn
import numpy as np


class RainfallLSTM(nn.Module):
    """
    LSTM model for predicting rainfall
    
    Architecture:
    - 2 LSTM layers (64 units each)
    - Dropout layers (prevent overfitting)
    - Dense layer (32 units)
    - Output layer (1 unit - rainfall prediction)
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        """
        Initialize the LSTM model
        
        Args:
            input_size (int): Number of features (40 in our case)
            hidden_size (int): Number of LSTM units per layer
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout rate (0.0-1.0)
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

    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
               In our case: (batch, 7, 40)
        
        Returns:
            Output tensor of shape (batch_size, 1) - predicted rainfall
        """

        # Initialize hidden state and cell state
        # Shape: (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        # out shape: (batch, seq_length, hidden_size)
        # h_n shape: (num_layers, batch, hidden_size)
        # c_n shape: (num_layers, batch, hidden_size)
        out, (h_n, c_n) = self.lstm(x, (h0, c0))

        # We only want the output from the last time step
        # out[:, -1, :] means: all batches, last timestep, all features
        out = out[:, -1, :]  # Shape: (batch, hidden_size)

        # Apply dropout
        out = self.dropout_layer(out)
        
        # Dense layer
        out = self.fc1(out)  # Shape: (batch, 32)
        out = self.relu(out)
        
        # Output layer
        out = self.fc2(out)  # Shape: (batch, 1)
        
        return out

    def count_parameters(self):
        """
        Count total trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(input_size, device='cpu'):
    """
    Create and initialize the LSTM model

    Args:
        input_size (int): Number of features (40 in our case)
        device (str): Device to use ('cpu' or 'cuda')

    Returns:
        model: Initialized LSTM model
    """
    model = RainfallLSTM(input_size=input_size)
    model = model.to(device)
    return model


def model_summary(model, input_shape=(1, 7, 40)):
    """
    Print model summary

    Args:
        model: LSTM model
        input_shape (tuple): Input shape (batch_size, sequence_length, input_size)
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