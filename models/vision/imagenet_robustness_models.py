#!/usr/bin/env python3
"""
ImageNet Robustness Models for Delayed Generalization Research

This module contains model architectures designed for studying robustness
and delayed generalization patterns in ImageNet-C dataset.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class ImageNetModel(nn.Module):
    """Wrapper for ImageNet models with unified interface"""
    
    def __init__(self, backbone: str = 'resnet50', num_classes: int = 1000, pretrained: bool = False):
        """
        Args:
            backbone: Backbone architecture ('resnet18', 'resnet50', 'resnet101', 'mobilenet_v2', etc.)
            num_classes: Number of output classes (1000 for ImageNet)
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        
        self.backbone_name = backbone
        self.num_classes = num_classes
        
        # Create backbone
        if backbone == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained)
            if num_classes != 1000:
                self.model.fc = nn.Linear(512, num_classes)
        elif backbone == 'resnet34':
            self.model = models.resnet34(pretrained=pretrained)
            if num_classes != 1000:
                self.model.fc = nn.Linear(512, num_classes)
        elif backbone == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
            if num_classes != 1000:
                self.model.fc = nn.Linear(2048, num_classes)
        elif backbone == 'resnet101':
            self.model = models.resnet101(pretrained=pretrained)
            if num_classes != 1000:
                self.model.fc = nn.Linear(2048, num_classes)
        elif backbone == 'mobilenet_v2':
            self.model = models.mobilenet_v2(pretrained=pretrained)
            if num_classes != 1000:
                self.model.classifier[1] = nn.Linear(1280, num_classes)
        elif backbone == 'efficientnet_b0':
            self.model = models.efficientnet_b0(pretrained=pretrained)
            if num_classes != 1000:
                self.model.classifier[1] = nn.Linear(1280, num_classes)
        elif backbone == 'vit_b_16':
            self.model = models.vit_b_16(pretrained=pretrained)
            if num_classes != 1000:
                self.model.heads.head = nn.Linear(768, num_classes)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Initialize weights if not pretrained
        if not pretrained:
            self._initialize_weights()
    
    def forward(self, x):
        return self.model(x)
    
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
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def get_num_parameters(self):
        """Count total number of parameters"""
        return sum(p.numel() for p in self.parameters())


class RobustImageNetModel(nn.Module):
    """ImageNet model with robustness-specific modifications"""
    
    def __init__(
        self, 
        backbone: str = 'resnet50',
        num_classes: int = 1000,
        pretrained: bool = False,
        dropout: float = 0.5,
        augmentation_aware: bool = False
    ):
        """
        Args:
            backbone: Backbone architecture
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            dropout: Dropout rate for robustness
            augmentation_aware: Whether to use augmentation-aware batch norm
        """
        super().__init__()
        
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.augmentation_aware = augmentation_aware
        
        # Create base model
        self.base_model = ImageNetModel(backbone, num_classes, pretrained)
        
        # Add dropout for robustness
        if dropout > 0 and backbone.startswith('resnet'):
            # Insert dropout before final fc layer
            self.dropout = nn.Dropout(dropout)
            self.use_dropout = True
        else:
            self.use_dropout = False
    
    def forward(self, x):
        if self.use_dropout and self.backbone_name.startswith('resnet'):
            # Extract features before final layer
            x = self.base_model.model.conv1(x)
            x = self.base_model.model.bn1(x)
            x = self.base_model.model.relu(x)
            x = self.base_model.model.maxpool(x)
            
            x = self.base_model.model.layer1(x)
            x = self.base_model.model.layer2(x)
            x = self.base_model.model.layer3(x)
            x = self.base_model.model.layer4(x)
            
            x = self.base_model.model.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
            x = self.base_model.model.fc(x)
            return x
        else:
            return self.base_model(x)
    
    def get_num_parameters(self):
        """Count total number of parameters"""
        return sum(p.numel() for p in self.parameters())


def create_imagenet_robustness_model(
    model_type: str = 'resnet50',
    num_classes: int = 1000,
    pretrained: bool = False,
    dropout: float = 0.5,
    robust: bool = True
) -> nn.Module:
    """
    Factory function to create ImageNet robustness models
    
    Args:
        model_type: Type of model ('resnet18', 'resnet50', 'resnet101', 'mobilenet_v2', etc.)
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        dropout: Dropout rate for robustness
        robust: Whether to use robust variant with additional features
        
    Returns:
        Model instance
    """
    
    if robust:
        model = RobustImageNetModel(
            backbone=model_type,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout
        )
    else:
        model = ImageNetModel(
            backbone=model_type,
            num_classes=num_classes,
            pretrained=pretrained
        )
    
    return model


# Convenience functions for common architectures
def resnet18_imagenet(num_classes: int = 1000, pretrained: bool = False, robust: bool = False) -> nn.Module:
    """ResNet-18 for ImageNet"""
    return create_imagenet_robustness_model('resnet18', num_classes, pretrained, robust=robust)


def resnet50_imagenet(num_classes: int = 1000, pretrained: bool = False, robust: bool = False) -> nn.Module:
    """ResNet-50 for ImageNet"""
    return create_imagenet_robustness_model('resnet50', num_classes, pretrained, robust=robust)


def resnet101_imagenet(num_classes: int = 1000, pretrained: bool = False, robust: bool = False) -> nn.Module:
    """ResNet-101 for ImageNet"""
    return create_imagenet_robustness_model('resnet101', num_classes, pretrained, robust=robust)


def mobilenet_v2_imagenet(num_classes: int = 1000, pretrained: bool = False, robust: bool = False) -> nn.Module:
    """MobileNet-V2 for ImageNet"""
    return create_imagenet_robustness_model('mobilenet_v2', num_classes, pretrained, robust=robust)


if __name__ == "__main__":
    # Test model creation
    print("Testing ImageNet Robustness Models")
    print("=" * 50)
    
    models_to_test = ['resnet18', 'resnet50', 'mobilenet_v2']
    
    for model_type in models_to_test:
        model = create_imagenet_robustness_model(model_type, num_classes=1000, robust=True)
        print(f"\n{model_type}:")
        print(f"  Parameters: {model.get_num_parameters():,}")
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        y = model(x)
        print(f"  Output shape: {y.shape}")
        assert y.shape == (2, 1000), f"Expected shape (2, 1000), got {y.shape}"
    
    print("\n✓ All models tested successfully!")
