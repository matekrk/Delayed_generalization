#!/usr/bin/env python3
"""
CNN Models for Delayed Generalization Research

This module provides CNN architectures optimized for studying delayed generalization
phenomena, particularly simplicity bias in vision tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class ColoredMNISTModel(nn.Module):
    """
    CNN model for Colored MNIST experiments studying simplicity bias.
    
    This model is designed to study how networks initially learn color features
    before transitioning to learning shape features.
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        num_channels: int = 3,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        model_size: str = 'medium'
    ):
        super().__init__()
        
        # Determine base channels based on model size
        if model_size == 'tiny':
            base_channels = 16
        elif model_size == 'small':
            base_channels = 32
        elif model_size == 'medium':
            base_channels = 64
        elif model_size == 'large':
            base_channels = 128
        else:
            base_channels = 64
        
        # First convolutional block
        layers = [
            nn.Conv2d(num_channels, base_channels, kernel_size=3, padding=1),
        ]
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(base_channels))
        layers.extend([
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
        ])
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(base_channels))
        layers.extend([
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout / 2)
        ])
        
        # Second convolutional block
        layers.extend([
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
        ])
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(base_channels * 2))
        layers.extend([
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
        ])
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(base_channels * 2))
        layers.extend([
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout)
        ])
        
        # Third convolutional block
        layers.extend([
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
        ])
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(base_channels * 4))
        layers.extend([
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout)
        ])
        
        self.features = nn.Sequential(*layers)
        
        # Calculate the size of flattened features
        # For 28x28 input with 3 maxpool layers (each /2): 28 -> 14 -> 7 -> 3
        # So final feature map is 3x3
        feature_size = base_channels * 4 * 3 * 3
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_size, base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 2, num_classes)
        )
        
        # Store attributes for analysis
        self.num_classes = num_classes
        self.num_channels = num_channels
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


class SimpleColoredMNISTModel(nn.Module):
    """
    Simple CNN model for Colored MNIST with minimal architecture.
    Useful for studying basic simplicity bias phenomena.
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        num_channels: int = 3,
        dropout: float = 0.1,
        use_batch_norm: bool = False
    ):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout)
        )
        
        # For 28x28 input: 28 -> 14 -> 7
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def create_colored_mnist_model(
    model_type: str = 'medium',
    num_classes: int = 10,
    num_channels: int = 3,
    dropout: float = 0.1,
    use_batch_norm: bool = True,
    **kwargs
) -> nn.Module:
    """
    Factory function to create colored MNIST models for simplicity bias experiments.
    
    Args:
        model_type: Type/size of model ('simple', 'tiny', 'small', 'medium', 'large')
        num_classes: Number of output classes (default: 10 for MNIST digits)
        num_channels: Number of input channels (default: 3 for RGB)
        dropout: Dropout probability
        use_batch_norm: Whether to use batch normalization
        **kwargs: Additional arguments passed to model constructor
        
    Returns:
        Initialized CNN model
    """
    
    if model_type == 'simple':
        return SimpleColoredMNISTModel(
            num_classes=num_classes,
            num_channels=num_channels,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
            **kwargs
        )
    else:
        # For 'tiny', 'small', 'medium', 'large'
        return ColoredMNISTModel(
            num_classes=num_classes,
            num_channels=num_channels,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
            model_size=model_type,
            **kwargs
        )


# Model registry for easy access
COLORED_MNIST_MODELS = {
    'simple': SimpleColoredMNISTModel,
    'tiny': lambda **kwargs: ColoredMNISTModel(model_size='tiny', **kwargs),
    'small': lambda **kwargs: ColoredMNISTModel(model_size='small', **kwargs),
    'medium': lambda **kwargs: ColoredMNISTModel(model_size='medium', **kwargs),
    'large': lambda **kwargs: ColoredMNISTModel(model_size='large', **kwargs),
}


def get_model_info():
    """Get information about available colored MNIST models."""
    return {
        'available_models': list(COLORED_MNIST_MODELS.keys()),
        'default_model': 'medium',
        'description': 'CNN models for colored MNIST simplicity bias experiments'
    }


if __name__ == "__main__":
    # Test model creation
    print("Testing colored MNIST model creation...")
    
    for model_type in ['simple', 'tiny', 'small', 'medium', 'large']:
        model = create_colored_mnist_model(model_type)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Model: {model_type}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 28, 28)
        output = model(dummy_input)
        print(f"  Output shape: {output.shape}")
        print()
    
    print("All models created successfully!")