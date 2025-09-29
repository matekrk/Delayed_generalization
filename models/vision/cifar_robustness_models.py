#!/usr/bin/env python3
"""
CIFAR Robustness Models for Delayed Generalization Research

This module contains CNN architectures designed for studying robustness
and delayed generalization patterns in CIFAR-10-C and CIFAR-100-C datasets.
"""

import torch
import torch.nn as nn
from typing import Optional


class CIFARModel(nn.Module):
    """Unified CNN model for CIFAR-10/100 with configurable size variants"""
    
    def __init__(self, num_classes: int = 10, input_size: int = 32, model_size: str = 'small'):
        """
        Args:
            num_classes: Number of output classes (10 for CIFAR-10, 100 for CIFAR-100)
            input_size: Input image size (32 for CIFAR)
            model_size: Model size variant
                - 'very_small': ~500K params
                - 'small': ~1M params  
                - 'medium': ~2.5M params
                - 'large': ~5M+ params
        """
        super().__init__()
        
        # Configure architecture based on model size
        if model_size == 'very_small':
            # ~500K params - minimal architecture
            channels = [32, 64, 128]
            hidden_dim = 256
            conv_blocks = 3
        elif model_size == 'small':
            # ~1M params - balanced architecture
            channels = [48, 96, 192]
            hidden_dim = 384
            conv_blocks = 3
        elif model_size == 'medium':
            # ~2.5M params - good capacity
            channels = [64, 128, 256]
            hidden_dim = 512
            conv_blocks = 3
        elif model_size == 'large':
            # ~5M+ params - high capacity
            channels = [96, 192, 384, 768]
            hidden_dim = 2048
            conv_blocks = 4
        else:
            raise ValueError(f"Unknown model_size: {model_size}. Choose from ['very_small', 'small', 'medium', 'large']")
        
        self.model_size = model_size
        self.num_classes = num_classes
        
        # Build convolutional feature extractor
        layers = []
        in_channels = 3
        
        for i, out_channels in enumerate(channels):
            # Conv block with batch norm
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.25)
            ])
            in_channels = out_channels
        
        self.features = nn.Sequential(*layers)
        
        # Calculate feature map size after convolutions
        conv_output_size = input_size // (2 ** conv_blocks)
        classifier_input_size = channels[-1] * conv_output_size * conv_output_size
        
        # Adaptive pooling for consistent feature size if needed
        if conv_output_size < 1:
            self.features.add_module('adaptive_pool', nn.AdaptiveAvgPool2d((2, 2)))
            classifier_input_size = channels[-1] * 2 * 2
        
        # Build classifier
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def get_num_parameters(self):
        """Return the number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Legacy compatibility classes
class CIFAR10CModel(CIFARModel):
    """Legacy wrapper for CIFAR-10-C compatibility"""
    
    def __init__(self, num_classes: int = 10, input_size: int = 32):
        super().__init__(num_classes=num_classes, input_size=input_size, model_size='small')


class CIFAR100CModel(CIFARModel):
    """Legacy wrapper for CIFAR-100-C compatibility"""
    
    def __init__(self, num_classes: int = 100, input_size: int = 32, model_size: str = 'medium'):
        super().__init__(num_classes=num_classes, input_size=input_size, model_size=model_size)


def create_cifar_robustness_model(
    model_type: str = 'cifar10c',
    num_classes: Optional[int] = None,
    model_size: str = 'small',
    **kwargs
) -> nn.Module:
    """
    Factory function to create CIFAR robustness models
    
    Args:
        model_type: Type of model ('cifar10c', 'cifar100c', or 'cifar')
        num_classes: Number of classes (auto-set based on model_type if None)
        model_size: Model size ('very_small', 'small', 'medium', 'large')
        **kwargs: Additional model parameters
        
    Returns:
        Configured CIFARModel
    """
    
    if model_type == 'cifar10c' or model_type == 'cifar10':
        if num_classes is None:
            num_classes = 10
        return CIFARModel(num_classes=num_classes, model_size=model_size, **kwargs)
    elif model_type == 'cifar100c' or model_type == 'cifar100':
        if num_classes is None:
            num_classes = 100
        return CIFARModel(num_classes=num_classes, model_size=model_size, **kwargs)
    elif model_type == 'cifar':
        # Generic CIFAR model - requires explicit num_classes
        if num_classes is None:
            raise ValueError("num_classes must be specified for generic 'cifar' model_type")
        return CIFARModel(num_classes=num_classes, model_size=model_size, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test the models
    batch_size = 8
    test_input = torch.randn(batch_size, 3, 32, 32)
    
    print("Testing Unified CIFAR Models")
    print("=" * 50)
    
    # Test all model sizes
    model_sizes = ['very_small', 'small', 'medium', 'large']
    
    for size in model_sizes:
        print(f"\n{size.upper()} MODEL:")
        print("-" * 20)
        
        # CIFAR-10 variant
        cifar10_model = CIFARModel(num_classes=10, model_size=size)
        cifar10_output = cifar10_model(test_input)
        cifar10_params = cifar10_model.get_num_parameters()
        print(f"CIFAR-10 ({size}): Output shape {cifar10_output.shape}, Parameters: {cifar10_params:,}")
        
        # CIFAR-100 variant
        cifar100_model = CIFARModel(num_classes=100, model_size=size)
        cifar100_output = cifar100_model(test_input)
        cifar100_params = cifar100_model.get_num_parameters()
        print(f"CIFAR-100 ({size}): Output shape {cifar100_output.shape}, Parameters: {cifar100_params:,}")
    
    print("\n" + "=" * 50)
    print("Factory function test:")
    
    # Test factory function
    model_legacy = create_cifar_robustness_model('cifar10c', model_size='small')
    model_new = create_cifar_robustness_model('cifar', num_classes=10, model_size='small')
    
    print(f"Legacy CIFAR10C: {model_legacy.get_num_parameters():,} parameters")
    print(f"New CIFAR: {model_new.get_num_parameters():,} parameters")
    print(f"Both models equivalent: {model_legacy.get_num_parameters() == model_new.get_num_parameters()}")