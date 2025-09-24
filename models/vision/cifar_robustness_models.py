#!/usr/bin/env python3
"""
CIFAR Robustness Models for Delayed Generalization Research

This module contains CNN architectures designed for studying robustness
and delayed generalization patterns in CIFAR-10-C and CIFAR-100-C datasets.
"""

import torch
import torch.nn as nn
from typing import Optional


class CIFAR10CModel(nn.Module):
    """CNN model for CIFAR-10-C robustness classification"""
    
    def __init__(self, num_classes: int = 10, input_size: int = 32):
        super().__init__()
        
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Second conv block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Third conv block
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        
        # Calculate the size after conv layers
        conv_output_size = input_size // (2 ** 3)  # 3 max pool operations
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * conv_output_size * conv_output_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class CIFAR100CModel(nn.Module):
    """Enhanced CNN model for CIFAR-100-C classification (100 classes)"""
    
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
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def create_cifar_robustness_model(
    model_type: str = 'cifar10c',
    num_classes: Optional[int] = None,
    model_size: str = 'medium',
    **kwargs
) -> nn.Module:
    """
    Factory function to create CIFAR robustness models
    
    Args:
        model_type: Type of model ('cifar10c' or 'cifar100c')
        num_classes: Number of classes (auto-set based on model_type if None)
        model_size: Model size ('small', 'medium', 'large') - only for CIFAR100C
        **kwargs: Additional model parameters
        
    Returns:
        Configured model
    """
    
    if model_type == 'cifar10c':
        if num_classes is None:
            num_classes = 10
        return CIFAR10CModel(num_classes=num_classes, **kwargs)
    elif model_type == 'cifar100c':
        if num_classes is None:
            num_classes = 100
        return CIFAR100CModel(num_classes=num_classes, model_size=model_size, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test the models
    batch_size = 8
    test_input = torch.randn(batch_size, 3, 32, 32)
    
    print("Testing CIFAR Robustness Models")
    print("=" * 40)
    
    # Test CIFAR10CModel
    cifar10c_model = create_cifar_robustness_model('cifar10c')
    cifar10c_output = cifar10c_model(test_input)
    print(f"CIFAR10CModel: {cifar10c_output.shape}")
    print(f"Parameters: {sum(p.numel() for p in cifar10c_model.parameters()):,}")
    
    # Test CIFAR100CModel
    cifar100c_model = create_cifar_robustness_model('cifar100c', model_size='medium')
    cifar100c_output = cifar100c_model(test_input)
    print(f"CIFAR100CModel: {cifar100c_output.shape}")
    print(f"Parameters: {sum(p.numel() for p in cifar100c_model.parameters()):,}")