#!/usr/bin/env python3
"""
Test script to verify the LSTM model import works correctly
"""

try:
    from lstm_model import create_model
    print("✅ Successfully imported create_model from lstm_model")
    
    # Test creating a model
    model = create_model(input_size=29)  # Using 29 features as per your data
    print(f"✅ Successfully created model with {model.count_parameters():,} parameters")
    
    # Test forward pass
    import torch
    dummy_input = torch.randn(1, 7, 29)  # batch=1, seq=7, features=29
    output = model(dummy_input)
    print(f"✅ Forward pass successful! Output shape: {output.shape}")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
except Exception as e:
    print(f"❌ Error: {e}")