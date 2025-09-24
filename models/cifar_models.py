#!/usr/bin/env python3
"""
CIFAR Models for Delayed Generalization Research

This module provides CNN architectures optimized for CIFAR-10 and CIFAR-100
robustness and corruption studies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class CIFAR10CModel(nn.Module):
    """
    CNN model for CIFAR-10-C classification with corruption robustness.
    
    This model is designed for studying robustness vs accuracy tradeoffs
    and delayed generalization patterns in corrupted CIFAR-10 data.
    """
    
    def __init__(self, num_classes: int = 10, input_size: int = 32, model_size: str = 'medium'):
        super().__init__()
        
        # Scale model based on size parameter
        if model_size == 'small':
            base_channels = 32
            dropout_conv = 0.1
            dropout_fc = 0.3
        elif model_size == 'medium':
            base_channels = 64
            dropout_conv = 0.25
            dropout_fc = 0.5
        elif model_size == 'large':
            base_channels = 96
            dropout_conv = 0.3
            dropout_fc = 0.5
        else:
            base_channels = 64
            dropout_conv = 0.25
            dropout_fc = 0.5
        
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_conv),
            
            # Second conv block
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_conv),
            
            # Third conv block
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_conv),
        )
        
        # Calculate the size after conv layers
        conv_output_size = input_size // (2 ** 3)  # 3 max pool operations
        classifier_input_size = base_channels * 4 * conv_output_size * conv_output_size
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, base_channels * 8),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_fc),
            nn.Linear(base_channels * 8, base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_fc),
            nn.Linear(base_channels * 4, num_classes)
        )
        
        # Store attributes for analysis
        self.num_classes = num_classes
        self.input_size = input_size
        self.model_size = model_size
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def get_feature_representations(self, x):
        """Get intermediate feature representations for analysis."""
        features = self.features(x)
        flattened = features.view(features.size(0), -1)
        return {
            'conv_features': features,
            'flattened_features': flattened,
            'logits': self.classifier(flattened)
        }


class CIFAR100CModel(nn.Module):
    """
    Enhanced CNN model for CIFAR-100-C classification (100 classes).
    
    This model is designed for studying robustness patterns in the more
    challenging 100-class CIFAR-100 setting with corruptions.
    """
    
    def __init__(self, num_classes: int = 100, input_size: int = 32, model_size: str = 'medium'):
        super().__init__()
        
        # Scale model based on complexity needed for 100 classes
        if model_size == 'small':
            base_channels = 64
            hidden_dim = 512
        elif model_size == 'medium':
            base_channels = 96
            hidden_dim = 1024
        elif model_size == 'large':
            base_channels = 128
            hidden_dim = 2048
        else:
            base_channels = 96
            hidden_dim = 1024
        
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Second conv block  
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Third conv block
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Fourth conv block (additional for 100 classes)
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 8, base_channels * 8, 3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Dropout2d(0.5),
        )
        
        # Calculate classifier input size
        classifier_input_size = base_channels * 8 * 2 * 2
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Store attributes for analysis
        self.num_classes = num_classes
        self.input_size = input_size
        self.model_size = model_size
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def get_feature_representations(self, x):
        """Get intermediate feature representations for analysis."""
        features = self.features(x)
        flattened = features.view(features.size(0), -1)
        return {
            'conv_features': features,
            'flattened_features': flattened,
            'logits': self.classifier(flattened)
        }


def create_cifar10c_model(
    model_size: str = 'medium',
    num_classes: int = 10,
    input_size: int = 32,
    **kwargs
) -> nn.Module:
    """
    Factory function to create CIFAR-10-C models for robustness experiments.
    
    Args:
        model_size: Size of model ('small', 'medium', 'large')
        num_classes: Number of output classes (default: 10)
        input_size: Input image size (default: 32 for CIFAR)
        **kwargs: Additional arguments passed to model constructor
        
    Returns:
        Initialized CIFAR-10-C model
    """
    return CIFAR10CModel(
        num_classes=num_classes,
        input_size=input_size,
        model_size=model_size,
        **kwargs
    )


def create_cifar100c_model(
    model_size: str = 'medium',
    num_classes: int = 100,
    input_size: int = 32,
    **kwargs
) -> nn.Module:
    """
    Factory function to create CIFAR-100-C models for robustness experiments.
    
    Args:
        model_size: Size of model ('small', 'medium', 'large')
        num_classes: Number of output classes (default: 100)
        input_size: Input image size (default: 32 for CIFAR)
        **kwargs: Additional arguments passed to model constructor
        
    Returns:
        Initialized CIFAR-100-C model
    """
    return CIFAR100CModel(
        num_classes=num_classes,
        input_size=input_size,
        model_size=model_size,
        **kwargs
    )


# Model registry for easy access
CIFAR_MODELS = {
    'cifar10c_small': lambda **kwargs: create_cifar10c_model(model_size='small', **kwargs),
    'cifar10c_medium': lambda **kwargs: create_cifar10c_model(model_size='medium', **kwargs),
    'cifar10c_large': lambda **kwargs: create_cifar10c_model(model_size='large', **kwargs),
    'cifar100c_small': lambda **kwargs: create_cifar100c_model(model_size='small', **kwargs),
    'cifar100c_medium': lambda **kwargs: create_cifar100c_model(model_size='medium', **kwargs),
    'cifar100c_large': lambda **kwargs: create_cifar100c_model(model_size='large', **kwargs),
}


def get_model_info():
    """Get information about available CIFAR models."""
    return {
        'available_models': list(CIFAR_MODELS.keys()),
        'cifar10c_models': ['cifar10c_small', 'cifar10c_medium', 'cifar10c_large'],
        'cifar100c_models': ['cifar100c_small', 'cifar100c_medium', 'cifar100c_large'],
        'description': 'CNN models for CIFAR-10-C and CIFAR-100-C robustness experiments'
    }


if __name__ == "__main__":
    # Test model creation
    print("Testing CIFAR model creation...")
    
    # Test CIFAR-10-C models
    for size in ['small', 'medium', 'large']:
        model = create_cifar10c_model(model_size=size)
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"CIFAR-10-C {size} model:")
        print(f"  Total parameters: {total_params:,}")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 32, 32)
        output = model(dummy_input)
        print(f"  Output shape: {output.shape}")
        print()
    
    # Test CIFAR-100-C models
    for size in ['small', 'medium', 'large']:
        model = create_cifar100c_model(model_size=size)
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"CIFAR-100-C {size} model:")
        print(f"  Total parameters: {total_params:,}")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 32, 32)
        output = model(dummy_input)
        print(f"  Output shape: {output.shape}")
        print()
    
    print("All CIFAR models created successfully!")