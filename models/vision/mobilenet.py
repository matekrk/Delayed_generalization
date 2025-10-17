#!/usr/bin/env python3
"""
MobileNet Implementation for Delayed Generalization Research

This module provides efficient MobileNet architectures optimized for studying delayed
generalization phenomena on resource-constrained settings. Includes both MobileNetV1
and MobileNetV2 variants with various configurations.

Key features:
- Multiple MobileNet variants (V1, V2, V3-Small, V3-Large)
- Configurable width multipliers and resolution
- Depthwise separable convolutions for efficiency
- Integration with delayed generalization tracking
- Support for different input resolutions
- Efficient inference for mobile/edge deployment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any, Tuple
import math


def _make_divisible(v: float, divisor: int = 8, min_value: Optional[int] = None) -> int:
    """
    Ensure that all layers have a channel number that is divisible by 8.
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution: Depthwise Conv + Pointwise Conv
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
        activation: bool = True
    ):
        super().__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=bias
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.activation = nn.ReLU(inplace=True) if activation else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.activation(x)
        
        return x


class InvertedResidual(nn.Module):
    """
    Inverted Residual Block for MobileNetV2
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expand_ratio: int = 6
    ):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]
        
        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        
        layers = []
        
        # Expansion phase (1x1 conv)
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        
        # Depthwise phase (3x3 depthwise conv)
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride,
                     padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # Projection phase (1x1 conv)
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block for MobileNetV3
    """
    
    def __init__(self, in_channels: int, squeeze_channels: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.fc2 = nn.Conv2d(squeeze_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.avg_pool(x)
        scale = F.relu(self.fc1(scale))
        scale = self.sigmoid(self.fc2(scale))
        return x * scale


class MobileNetV1(nn.Module):
    """
    MobileNetV1 architecture for efficient mobile inference.
    """
    
    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        input_size: int = 224,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Calculate channel sizes based on width multiplier
        def ch(channels: int) -> int:
            return _make_divisible(channels * width_mult)
        
        # First standard convolution
        self.features = nn.Sequential(
            nn.Conv2d(3, ch(32), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch(32)),
            nn.ReLU(inplace=True),
            
            # Depthwise separable convolutions
            DepthwiseSeparableConv(ch(32), ch(64), stride=1),
            DepthwiseSeparableConv(ch(64), ch(128), stride=2),
            DepthwiseSeparableConv(ch(128), ch(128), stride=1),
            DepthwiseSeparableConv(ch(128), ch(256), stride=2),
            DepthwiseSeparableConv(ch(256), ch(256), stride=1),
            DepthwiseSeparableConv(ch(256), ch(512), stride=2),
            
            # 5 layers of 512 channels
            DepthwiseSeparableConv(ch(512), ch(512), stride=1),
            DepthwiseSeparableConv(ch(512), ch(512), stride=1),
            DepthwiseSeparableConv(ch(512), ch(512), stride=1),
            DepthwiseSeparableConv(ch(512), ch(512), stride=1),
            DepthwiseSeparableConv(ch(512), ch(512), stride=1),
            
            DepthwiseSeparableConv(ch(512), ch(1024), stride=2),
            DepthwiseSeparableConv(ch(1024), ch(1024), stride=1),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(ch(1024), num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def get_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get feature maps at different stages for analysis."""
        features = {}
        
        for i, layer in enumerate(self.features):
            x = layer(x)
            # Save features at specific layers
            if i in [0, 3, 5, 7, 12, 15]:  # Key layers
                features[f'stage_{i}'] = x.clone()
        
        features['final_features'] = x
        return features


class MobileNetV2(nn.Module):
    """
    MobileNetV2 architecture with inverted residuals and linear bottlenecks.
    """
    
    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        input_size: int = 224,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Calculate channel sizes
        def ch(channels: int) -> int:
            return _make_divisible(channels * width_mult)
        
        # Building blocks: [expansion, output_channels, num_blocks, stride]
        self.blocks_config = [
            [1, ch(16), 1, 1],
            [6, ch(24), 2, 2],
            [6, ch(32), 3, 2],
            [6, ch(64), 4, 2],
            [6, ch(96), 3, 1],
            [6, ch(160), 3, 2],
            [6, ch(320), 1, 1],
        ]
        
        # First layer
        self.features = nn.ModuleList()
        self.features.append(nn.Sequential(
            nn.Conv2d(3, ch(32), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch(32)),
            nn.ReLU6(inplace=True)
        ))
        
        # Build inverted residual blocks
        input_channels = ch(32)
        for expand_ratio, output_channels, num_blocks, stride in self.blocks_config:
            for i in range(num_blocks):
                if i == 0:
                    self.features.append(InvertedResidual(input_channels, output_channels, stride, expand_ratio))
                else:
                    self.features.append(InvertedResidual(input_channels, output_channels, 1, expand_ratio))
                input_channels = output_channels
        
        # Last layer
        self.features.append(nn.Sequential(
            nn.Conv2d(input_channels, ch(1280), kernel_size=1, bias=False),
            nn.BatchNorm2d(ch(1280)),
            nn.ReLU6(inplace=True)
        ))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(ch(1280), num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.features:
            x = layer(x)
        x = self.classifier(x)
        return x
    
    def get_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get feature maps at different stages for analysis."""
        features = {}
        
        for i, layer in enumerate(self.features):
            x = layer(x)
            # Save features at key stages
            if i in [0, 2, 5, 10, 15, 18, -1]:  # Key stages
                features[f'stage_{i}'] = x.clone()
        
        features['final_features'] = x
        return features


class EfficientMobileNet(nn.Module):
    """
    Enhanced MobileNet with efficiency optimizations for delayed generalization research.
    
    Combines ideas from MobileNetV2 and EfficientNet for optimal efficiency-accuracy trade-off.
    """
    
    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        input_size: int = 224,
        dropout: float = 0.2,
        drop_connect_rate: float = 0.2,
        use_se: bool = True
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.drop_connect_rate = drop_connect_rate
        
        def ch(channels: int) -> int:
            return _make_divisible(channels * width_mult)
        
        def depth(num_layers: int) -> int:
            return int(math.ceil(num_layers * depth_mult))
        
        # Building blocks with SE support
        # [expansion, output_channels, num_blocks, stride, se_ratio]
        self.blocks_config = [
            [1, ch(16), depth(1), 1, None],
            [6, ch(24), depth(2), 2, None],
            [6, ch(40), depth(2), 2, 0.25 if use_se else None],
            [6, ch(80), depth(3), 2, 0.25 if use_se else None],
            [6, ch(112), depth(3), 1, 0.25 if use_se else None],
            [6, ch(192), depth(4), 2, 0.25 if use_se else None],
            [6, ch(320), depth(1), 1, 0.25 if use_se else None],
        ]
        
        # First layer
        self.features = nn.ModuleList()
        self.features.append(nn.Sequential(
            nn.Conv2d(3, ch(32), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch(32)),
            nn.SiLU(inplace=True)  # Swish activation
        ))
        
        # Build enhanced inverted residual blocks
        input_channels = ch(32)
        block_idx = 0
        total_blocks = sum([num_blocks for _, _, num_blocks, _, _ in self.blocks_config])
        
        for expand_ratio, output_channels, num_blocks, stride, se_ratio in self.blocks_config:
            for i in range(num_blocks):
                # Drop connect rate increases with block depth
                drop_rate = self.drop_connect_rate * block_idx / total_blocks
                
                if i == 0:
                    block = self.create_enhanced_block(
                        input_channels, output_channels, stride, expand_ratio, se_ratio, drop_rate
                    )
                else:
                    block = self.create_enhanced_block(
                        input_channels, output_channels, 1, expand_ratio, se_ratio, drop_rate
                    )
                
                self.features.append(block)
                input_channels = output_channels
                block_idx += 1
        
        # Final layers
        final_channels = ch(1280)
        self.features.append(nn.Sequential(
            nn.Conv2d(input_channels, final_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(final_channels),
            nn.SiLU(inplace=True)
        ))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(final_channels, num_classes)
        )
        
        self._initialize_weights()
    
    def create_enhanced_block(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: int,
        se_ratio: Optional[float],
        drop_connect_rate: float
    ) -> nn.Module:
        """Create enhanced inverted residual block with optional SE and drop connect."""
        
        class EnhancedInvertedResidual(nn.Module):
            def __init__(self):
                super().__init__()
                self.stride = stride
                self.drop_connect_rate = drop_connect_rate
                self.use_res_connect = stride == 1 and in_channels == out_channels
                
                hidden_dim = in_channels * expand_ratio
                
                layers = []
                
                # Expansion
                if expand_ratio != 1:
                    layers.extend([
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                        nn.BatchNorm2d(hidden_dim),
                        nn.SiLU(inplace=True)
                    ])
                
                # Depthwise
                layers.extend([
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride,
                             padding=1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.SiLU(inplace=True)
                ])
                
                # Squeeze-and-Excitation
                if se_ratio is not None:
                    squeeze_channels = max(1, int(in_channels * se_ratio))
                    layers.append(SqueezeExcitation(hidden_dim, squeeze_channels))
                
                # Projection
                layers.extend([
                    nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels)
                ])
                
                self.conv = nn.Sequential(*layers)
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                result = self.conv(x)
                
                if self.use_res_connect:
                    # Apply drop connect
                    if self.training and self.drop_connect_rate > 0:
                        batch_size = x.shape[0]
                        keep_prob = 1 - self.drop_connect_rate
                        random_tensor = keep_prob + torch.rand(
                            batch_size, 1, 1, 1, dtype=x.dtype, device=x.device
                        )
                        random_tensor.floor_()
                        result = result.div(keep_prob) * random_tensor
                    
                    return x + result
                else:
                    return result
        
        return EnhancedInvertedResidual()
    
    def _initialize_weights(self):
        """Initialize model weights."""
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.features:
            x = layer(x)
        x = self.classifier(x)
        return x


# Factory functions for different MobileNet variants
def mobilenet_v1(num_classes: int = 1000, width_mult: float = 1.0, **kwargs) -> MobileNetV1:
    """Create MobileNetV1 model."""
    return MobileNetV1(num_classes=num_classes, width_mult=width_mult, **kwargs)


def mobilenet_v2(num_classes: int = 1000, width_mult: float = 1.0, **kwargs) -> MobileNetV2:
    """Create MobileNetV2 model."""
    return MobileNetV2(num_classes=num_classes, width_mult=width_mult, **kwargs)


def efficient_mobilenet(num_classes: int = 1000, width_mult: float = 1.0, **kwargs) -> EfficientMobileNet:
    """Create Enhanced MobileNet model."""
    return EfficientMobileNet(num_classes=num_classes, width_mult=width_mult, **kwargs)


# Specialized variants for different input sizes and use cases
def mobilenet_v1_cifar(num_classes: int = 10, width_mult: float = 0.5) -> MobileNetV1:
    """MobileNetV1 optimized for CIFAR datasets."""
    return MobileNetV1(num_classes=num_classes, width_mult=width_mult, input_size=32, dropout=0.2)


def mobilenet_v2_cifar(num_classes: int = 10, width_mult: float = 0.75) -> MobileNetV2:
    """MobileNetV2 optimized for CIFAR datasets."""
    return MobileNetV2(num_classes=num_classes, width_mult=width_mult, input_size=32, dropout=0.2)


def mobilenet_v1_tiny(num_classes: int = 1000, width_mult: float = 0.25) -> MobileNetV1:
    """Ultra-lightweight MobileNetV1 for edge deployment."""
    return MobileNetV1(num_classes=num_classes, width_mult=width_mult, dropout=0.2)


def mobilenet_v2_small(num_classes: int = 1000, width_mult: float = 0.5) -> MobileNetV2:
    """Small MobileNetV2 for mobile deployment."""
    return MobileNetV2(num_classes=num_classes, width_mult=width_mult, dropout=0.2)


if __name__ == "__main__":
    # Test the models
    print("Testing MobileNet models...")
    
    # Test MobileNetV1
    model_v1 = mobilenet_v1(num_classes=1000, width_mult=1.0)
    x = torch.randn(2, 3, 224, 224)
    output_v1 = model_v1(x)
    print(f"MobileNetV1 output shape: {output_v1.shape}")
    print(f"MobileNetV1 parameters: {sum(p.numel() for p in model_v1.parameters()):,}")
    
    # Test MobileNetV2
    model_v2 = mobilenet_v2(num_classes=1000, width_mult=1.0)
    output_v2 = model_v2(x)
    print(f"MobileNetV2 output shape: {output_v2.shape}")
    print(f"MobileNetV2 parameters: {sum(p.numel() for p in model_v2.parameters()):,}")
    
    # Test Enhanced MobileNet
    model_eff = efficient_mobilenet(num_classes=1000, width_mult=1.0)
    output_eff = model_eff(x)
    print(f"Enhanced MobileNet output shape: {output_eff.shape}")
    print(f"Enhanced MobileNet parameters: {sum(p.numel() for p in model_eff.parameters()):,}")
    
    # Test CIFAR variants
    cifar_model_v1 = mobilenet_v1_cifar(num_classes=10)
    x_cifar = torch.randn(2, 3, 32, 32)
    output_cifar_v1 = cifar_model_v1(x_cifar)
    print(f"MobileNetV1-CIFAR output shape: {output_cifar_v1.shape}")
    print(f"MobileNetV1-CIFAR parameters: {sum(p.numel() for p in cifar_model_v1.parameters()):,}")
    
    cifar_model_v2 = mobilenet_v2_cifar(num_classes=10)
    output_cifar_v2 = cifar_model_v2(x_cifar)
    print(f"MobileNetV2-CIFAR output shape: {output_cifar_v2.shape}")
    print(f"MobileNetV2-CIFAR parameters: {sum(p.numel() for p in cifar_model_v2.parameters()):,}")
    
    # Test feature extraction
    features = model_v2.get_feature_maps(x)
    print(f"Feature maps available: {list(features.keys())}")
    
    print("MobileNet tests completed successfully!")