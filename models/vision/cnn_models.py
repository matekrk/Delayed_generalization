#!/usr/bin/env python3
"""
CNN Models for Colored MNIST Simplicity Bias Experiments

This module provides CNN architectures for studying simplicity bias in colored MNIST,
where models initially learn color features before shape features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class SimpleCNN(nn.Module):
    """Simple CNN for colored MNIST classification"""
    
    def __init__(
        self,
        num_classes: int = 10,
        dropout: float = 0.2,
        use_batch_norm: bool = True
    ):
        super().__init__()
        
        self.use_batch_norm = use_batch_norm
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Batch normalization layers
        if use_batch_norm:
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(64)
            self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layers
        # After 3 pooling operations: 28 -> 14 -> 7 -> 3 (with padding)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First conv block
        x = self.conv1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Second conv block
        x = self.conv2(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Third conv block
        x = self.conv3(x)
        if self.use_batch_norm:
            x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

    def get_features(self, x: torch.Tensor, layer: str = 'conv3') -> torch.Tensor:
        """Extract features from a specific layer for analysis"""
        
        # First conv block
        x = self.conv1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = F.relu(x)
        if layer == 'conv1':
            return x
        x = self.pool(x)
        
        # Second conv block  
        x = self.conv2(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = F.relu(x)
        if layer == 'conv2':
            return x
        x = self.pool(x)
        
        # Third conv block
        x = self.conv3(x)
        if self.use_batch_norm:
            x = self.bn3(x)
        x = F.relu(x)
        if layer == 'conv3':
            return x
        x = self.pool(x)
        
        # Flatten and FC layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        if layer == 'fc1':
            return x
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        if layer == 'fc2':
            return x
        
        return x


class ColorInvariantCNN(nn.Module):
    """CNN with techniques to encourage shape learning over color"""
    
    def __init__(
        self,
        num_classes: int = 10,
        dropout: float = 0.3,
        color_jitter: bool = True,
        grayscale_prob: float = 0.2
    ):
        super().__init__()
        
        self.color_jitter = color_jitter
        self.grayscale_prob = grayscale_prob
        
        # Convolutional layers with more aggressive regularization
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)
        self.dropout_2d = nn.Dropout2d(0.1)
        
        # Fully connected layers with larger capacity
        self.fc1 = nn.Linear(256 * 1 * 1, 512)  # After 4 pooling: 28->14->7->3->1
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
    def _augment_input(self, x: torch.Tensor) -> torch.Tensor:
        """Apply augmentations to encourage shape learning"""
        if not self.training:
            return x
            
        # Randomly convert some samples to grayscale
        if self.grayscale_prob > 0:
            batch_size = x.size(0)
            grayscale_mask = torch.rand(batch_size) < self.grayscale_prob
            
            if grayscale_mask.any():
                # Convert to grayscale by averaging channels
                grayscale = x[grayscale_mask].mean(dim=1, keepdim=True)
                grayscale = grayscale.repeat(1, 3, 1, 1)
                x[grayscale_mask] = grayscale
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply augmentations
        x = self._augment_input(x)
        
        # First conv block
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout_2d(x)
        
        # Second conv block
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout_2d(x)
        
        # Third conv block
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout_2d(x)
        
        # Fourth conv block
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        
        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class FeatureExtractorCNN(nn.Module):
    """CNN designed for feature analysis and visualization"""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        # Color processing branch
        self.color_conv1 = nn.Conv2d(3, 16, kernel_size=1)  # 1x1 conv for color
        self.color_conv2 = nn.Conv2d(16, 32, kernel_size=1)
        
        # Shape processing branch  
        self.shape_conv1 = nn.Conv2d(3, 32, kernel_size=7, padding=3)
        self.shape_conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.shape_conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Combined processing
        self.combined_conv = nn.Conv2d(32 + 128, 256, kernel_size=3, padding=1)
        
        # Pooling and normalization
        self.pool = nn.MaxPool2d(2, 2)
        self.bn_color1 = nn.BatchNorm2d(16)
        self.bn_color2 = nn.BatchNorm2d(32)
        self.bn_shape1 = nn.BatchNorm2d(32)
        self.bn_shape2 = nn.BatchNorm2d(64)
        self.bn_shape3 = nn.BatchNorm2d(128)
        self.bn_combined = nn.BatchNorm2d(256)
        
        # Classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, num_classes)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Color branch (learns global color statistics)
        color_features = F.relu(self.bn_color1(self.color_conv1(x)))
        color_features = F.relu(self.bn_color2(self.color_conv2(color_features)))
        color_features = self.pool(color_features)  # [B, 32, 14, 14]
        
        # Shape branch (learns spatial patterns)
        shape_features = F.relu(self.bn_shape1(self.shape_conv1(x)))
        shape_features = self.pool(shape_features)  # [B, 32, 14, 14]
        shape_features = F.relu(self.bn_shape2(self.shape_conv2(shape_features)))
        shape_features = self.pool(shape_features)  # [B, 64, 7, 7]
        shape_features = F.relu(self.bn_shape3(self.shape_conv3(shape_features)))
        
        # Resize color features to match shape features
        color_features = F.interpolate(color_features, size=shape_features.shape[2:], mode='bilinear')
        
        # Combine features
        combined = torch.cat([color_features, shape_features], dim=1)
        combined = F.relu(self.bn_combined(self.combined_conv(combined)))
        
        # Global pooling and classification
        pooled = self.global_pool(combined)
        features = pooled.view(pooled.size(0), -1)
        logits = self.classifier(features)
        
        # Return both logits and intermediate features for analysis
        feature_dict = {
            'color_features': color_features,
            'shape_features': shape_features,
            'combined_features': combined,
            'final_features': features
        }
        
        return logits, feature_dict


def create_colored_mnist_model(model_type: str = 'simple', **kwargs) -> nn.Module:
    """
    Factory function to create colored MNIST models
    
    Args:
        model_type: Type of model ('simple', 'invariant', 'feature_extractor')
        **kwargs: Additional model parameters
        
    Returns:
        Configured model
    """
    
    if model_type == 'simple':
        return SimpleCNN(**kwargs)
    elif model_type == 'invariant':
        return ColorInvariantCNN(**kwargs)
    elif model_type == 'feature_extractor':
        return FeatureExtractorCNN(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test the models
    batch_size = 8
    test_input = torch.randn(batch_size, 3, 28, 28)
    
    print("Testing Colored MNIST Models")
    print("=" * 40)
    
    # Test SimpleCNN
    simple_model = create_colored_mnist_model('simple')
    simple_output = simple_model(test_input)
    print(f"SimpleCNN: {simple_output.shape}")
    print(f"Parameters: {sum(p.numel() for p in simple_model.parameters()):,}")
    
    # Test ColorInvariantCNN
    invariant_model = create_colored_mnist_model('invariant')
    invariant_output = invariant_model(test_input)
    print(f"ColorInvariantCNN: {invariant_output.shape}")
    print(f"Parameters: {sum(p.numel() for p in invariant_model.parameters()):,}")
    
    # Test FeatureExtractorCNN
    feature_model = create_colored_mnist_model('feature_extractor')
    feature_output, features = feature_model(test_input)
    print(f"FeatureExtractorCNN: {feature_output.shape}")
    print(f"Parameters: {sum(p.numel() for p in feature_model.parameters()):,}")
    print(f"Feature shapes: {[(k, v.shape) for k, v in features.items()]}")