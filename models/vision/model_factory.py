#!/usr/bin/env python3
"""
Model Factory and Integration for Delayed Generalization Research

This module provides a unified interface for creating and using different model
architectures (CNNs, ViTs, MobileNets) with the existing training pipelines
for delayed generalization research.

Features:
- Unified model creation interface
- Automatic configuration for different datasets
- Integration with existing training scripts
- Model parameter counting and analysis
- Performance optimization recommendations
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, Type, List
import json
from pathlib import Path

try:
    from .vision_transformer import (
        VisionTransformer, vit_tiny_patch16_224, vit_small_patch16_224,
        vit_base_patch16_224, vit_large_patch16_224,
        vit_tiny_patch4_32, vit_small_patch4_32
    )
    VIT_AVAILABLE = True
except ImportError:
    VIT_AVAILABLE = False
    print("Warning: Vision Transformer models not available")

try:
    from .mobilenet import (
        MobileNetV1, MobileNetV2, EfficientMobileNet,
        mobilenet_v1, mobilenet_v2, efficient_mobilenet,
        mobilenet_v1_cifar, mobilenet_v2_cifar,
        mobilenet_v1_tiny, mobilenet_v2_small
    )
    MOBILENET_AVAILABLE = True
except ImportError:
    MOBILENET_AVAILABLE = False
    print("Warning: MobileNet models not available")


class ModelConfig:
    """Configuration class for model parameters"""
    
    def __init__(self, **kwargs):
        self.config = kwargs
    
    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)
    
    def update(self, **kwargs):
        self.config.update(kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        return self.config.copy()


class ModelFactory:
    """
    Factory class for creating models with dataset-specific configurations.
    """
    
    # Dataset-specific configurations
    DATASET_CONFIGS = {
        'cifar10': {
            'num_classes': 10,
            'input_size': 32,
            'input_channels': 3,
            'recommended_models': ['mobilenet_v2_cifar', 'vit_small_patch4_32', 'cnn_small']
        },
        'cifar100': {
            'num_classes': 100,
            'input_size': 32,
            'input_channels': 3,
            'recommended_models': ['mobilenet_v2_cifar', 'vit_small_patch4_32', 'cnn_medium']
        },
        'imagenet': {
            'num_classes': 1000,
            'input_size': 224,
            'input_channels': 3,
            'recommended_models': ['mobilenet_v2', 'vit_base_patch16_224', 'efficient_mobilenet']
        },
        'celeba': {
            'num_classes': 2,  # Gender classification
            'input_size': 64,
            'input_channels': 3,
            'recommended_models': ['mobilenet_v1_small', 'vit_tiny_patch8_64', 'cnn_small']
        },
        'colored_mnist': {
            'num_classes': 10,
            'input_size': 28,
            'input_channels': 3,
            'recommended_models': ['mobilenet_v1_tiny', 'cnn_tiny']
        }
    }
    
    @classmethod
    def get_model_config(cls, model_name: str, dataset: str, **overrides) -> ModelConfig:
        """
        Get model configuration for a specific model and dataset.
        
        Args:
            model_name: Name of the model architecture
            dataset: Target dataset name
            **overrides: Additional configuration overrides
            
        Returns:
            ModelConfig object with merged configuration
        """
        
        # Get dataset-specific config
        dataset_config = cls.DATASET_CONFIGS.get(dataset, {})
        
        # Base model configurations
        base_configs = {
            # Vision Transformers
            'vit_tiny_patch16_224': {
                'model_class': 'vit_tiny_patch16_224',
                'img_size': 224,
                'patch_size': 16,
                'embed_dim': 192,
                'depth': 12,
                'n_heads': 3
            },
            'vit_small_patch16_224': {
                'model_class': 'vit_small_patch16_224',
                'img_size': 224,
                'patch_size': 16,
                'embed_dim': 384,
                'depth': 12,
                'n_heads': 6
            },
            'vit_base_patch16_224': {
                'model_class': 'vit_base_patch16_224',
                'img_size': 224,
                'patch_size': 16,
                'embed_dim': 768,
                'depth': 12,
                'n_heads': 12
            },
            'vit_tiny_patch4_32': {
                'model_class': 'vit_tiny_patch4_32',
                'img_size': 32,
                'patch_size': 4,
                'embed_dim': 192,
                'depth': 12,
                'n_heads': 3
            },
            'vit_small_patch4_32': {
                'model_class': 'vit_small_patch4_32',
                'img_size': 32,
                'patch_size': 4,
                'embed_dim': 384,
                'depth': 12,
                'n_heads': 6
            },
            
            # MobileNets
            'mobilenet_v1': {
                'model_class': 'mobilenet_v1',
                'width_mult': 1.0,
                'input_size': 224,
                'dropout': 0.2
            },
            'mobilenet_v2': {
                'model_class': 'mobilenet_v2',
                'width_mult': 1.0,
                'input_size': 224,
                'dropout': 0.2
            },
            'efficient_mobilenet': {
                'model_class': 'efficient_mobilenet',
                'width_mult': 1.0,
                'depth_mult': 1.0,
                'input_size': 224,
                'dropout': 0.2,
                'use_se': True
            },
            'mobilenet_v1_cifar': {
                'model_class': 'mobilenet_v1_cifar',
                'width_mult': 0.5,
                'input_size': 32,
                'dropout': 0.2
            },
            'mobilenet_v2_cifar': {
                'model_class': 'mobilenet_v2_cifar',
                'width_mult': 0.75,
                'input_size': 32,
                'dropout': 0.2
            },
            'mobilenet_v1_tiny': {
                'model_class': 'mobilenet_v1_tiny',
                'width_mult': 0.25,
                'input_size': 224,
                'dropout': 0.2
            },
            'mobilenet_v2_small': {
                'model_class': 'mobilenet_v2_small',
                'width_mult': 0.5,
                'input_size': 224,
                'dropout': 0.2
            }
        }
        
        # Get base config
        base_config = base_configs.get(model_name, {})
        
        # Merge configurations: base -> dataset -> overrides
        merged_config = {}
        merged_config.update(base_config)
        merged_config.update(dataset_config)
        merged_config.update(overrides)
        
        return ModelConfig(**merged_config)
    
    @classmethod
    def create_model(
        cls,
        model_name: str,
        dataset: str = 'imagenet',
        pretrained: bool = False,
        **kwargs
    ) -> nn.Module:
        """
        Create a model instance with appropriate configuration.
        
        Args:
            model_name: Name of the model architecture
            dataset: Target dataset name
            pretrained: Whether to load pretrained weights (if available)
            **kwargs: Additional arguments for model configuration
            
        Returns:
            Configured model instance
        """
        
        config = cls.get_model_config(model_name, dataset, **kwargs)
        model_class_name = config.get('model_class', model_name)
        
        # Create model based on type
        if model_class_name.startswith('vit_'):
            if not VIT_AVAILABLE:
                raise ImportError("Vision Transformer models not available")
            
            model = cls._create_vit_model(model_class_name, config)
            
        elif model_class_name.startswith('mobilenet') or model_class_name.startswith('efficient_mobilenet'):
            if not MOBILENET_AVAILABLE:
                raise ImportError("MobileNet models not available")
            
            model = cls._create_mobilenet_model(model_class_name, config)
            
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Load pretrained weights if requested
        if pretrained:
            model = cls._load_pretrained_weights(model, model_name, dataset)
        
        return model
    
    @classmethod
    def _create_vit_model(cls, model_class_name: str, config: ModelConfig) -> nn.Module:
        """Create Vision Transformer model."""
        
        model_functions = {
            'vit_tiny_patch16_224': vit_tiny_patch16_224,
            'vit_small_patch16_224': vit_small_patch16_224,
            'vit_base_patch16_224': vit_base_patch16_224,
            'vit_large_patch16_224': vit_large_patch16_224,
            'vit_tiny_patch4_32': vit_tiny_patch4_32,
            'vit_small_patch4_32': vit_small_patch4_32
        }
        
        model_fn = model_functions.get(model_class_name)
        if model_fn is None:
            raise ValueError(f"Unknown ViT model: {model_class_name}")
        
        # Extract relevant parameters
        model_kwargs = {
            'num_classes': config.get('num_classes', 1000),
            'dropout': config.get('dropout', 0.1),
            'drop_path_rate': config.get('drop_path_rate', 0.1)
        }
        
        return model_fn(**model_kwargs)
    
    @classmethod
    def _create_mobilenet_model(cls, model_class_name: str, config: ModelConfig) -> nn.Module:
        """Create MobileNet model."""
        
        model_functions = {
            'mobilenet_v1': mobilenet_v1,
            'mobilenet_v2': mobilenet_v2,
            'efficient_mobilenet': efficient_mobilenet,
            'mobilenet_v1_cifar': mobilenet_v1_cifar,
            'mobilenet_v2_cifar': mobilenet_v2_cifar,
            'mobilenet_v1_tiny': mobilenet_v1_tiny,
            'mobilenet_v2_small': mobilenet_v2_small
        }
        
        model_fn = model_functions.get(model_class_name)
        if model_fn is None:
            raise ValueError(f"Unknown MobileNet model: {model_class_name}")
        
        # Extract relevant parameters
        model_kwargs = {
            'num_classes': config.get('num_classes', 1000),
        }
        
        # Add model-specific parameters
        if 'width_mult' in config.to_dict() and model_class_name not in ['mobilenet_v1_cifar', 'mobilenet_v2_cifar']:
            model_kwargs['width_mult'] = config.get('width_mult')
        if 'depth_mult' in config.to_dict():
            model_kwargs['depth_mult'] = config.get('depth_mult')
        if 'dropout' in config.to_dict() and model_class_name not in ['mobilenet_v1_cifar', 'mobilenet_v2_cifar']:
            model_kwargs['dropout'] = config.get('dropout')
        if 'use_se' in config.to_dict():
            model_kwargs['use_se'] = config.get('use_se')
        
        return model_fn(**model_kwargs)
    
    @classmethod
    def _load_pretrained_weights(
        cls,
        model: nn.Module,
        model_name: str,
        dataset: str
    ) -> nn.Module:
        """Load pretrained weights if available."""
        
        # TODO: Implement pretrained weight loading
        # This would load weights from a model zoo or local files
        print(f"Warning: Pretrained weights not yet implemented for {model_name} on {dataset}")
        return model
    
    @classmethod
    def get_recommended_models(cls, dataset: str) -> List[str]:
        """Get recommended models for a dataset."""
        
        dataset_config = cls.DATASET_CONFIGS.get(dataset, {})
        return dataset_config.get('recommended_models', [])
    
    @classmethod
    def analyze_model(cls, model: nn.Module) -> Dict[str, Any]:
        """Analyze model characteristics."""
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Model size estimation (assuming float32)
        model_size_mb = total_params * 4 / (1024 * 1024)
        
        # Architecture analysis
        conv_layers = 0
        linear_layers = 0
        attention_layers = 0
        
        for module in model.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                conv_layers += 1
            elif isinstance(module, nn.Linear):
                linear_layers += 1
            elif hasattr(module, 'attn') or 'attention' in module.__class__.__name__.lower():
                attention_layers += 1
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': model_size_mb,
            'architecture_stats': {
                'conv_layers': conv_layers,
                'linear_layers': linear_layers,
                'attention_layers': attention_layers
            },
            'efficiency_class': cls._classify_model_efficiency(total_params)
        }
    
    @classmethod
    def _classify_model_efficiency(cls, total_params: int) -> str:
        """Classify model efficiency based on parameter count."""
        
        if total_params < 1e6:
            return 'ultra_light'  # < 1M parameters
        elif total_params < 5e6:
            return 'light'        # 1-5M parameters
        elif total_params < 25e6:
            return 'medium'       # 5-25M parameters
        elif total_params < 100e6:
            return 'large'        # 25-100M parameters
        else:
            return 'very_large'   # > 100M parameters


# Convenience functions for easy model creation
def create_model_for_phenomenon(
    phenomenon: str,
    model_type: str = 'auto',
    efficiency: str = 'medium',
    **kwargs
) -> nn.Module:
    """
    Create a model optimized for a specific delayed generalization phenomenon.
    
    Args:
        phenomenon: Type of phenomenon ('grokking', 'simplicity_bias', 'robustness', etc.)
        model_type: Model architecture type ('vit', 'mobilenet', 'cnn', 'auto')
        efficiency: Efficiency requirement ('ultra_light', 'light', 'medium', 'large')
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured model instance
    """
    
    # Phenomenon-specific recommendations
    phenomenon_configs = {
        'grokking': {
            'dataset': 'algorithmic',
            'recommended_models': ['vit_small_patch4_32', 'mobilenet_v2_small'],
            'num_classes': kwargs.get('vocab_size', 100)
        },
        'simplicity_bias': {
            'dataset': 'cifar10',
            'recommended_models': ['mobilenet_v2_cifar', 'vit_small_patch4_32'],
            'num_classes': kwargs.get('num_classes', 10)
        },
        'robustness': {
            'dataset': 'cifar100',
            'recommended_models': ['efficient_mobilenet', 'vit_base_patch16_224'],
            'num_classes': kwargs.get('num_classes', 100)
        },
        'continual_learning': {
            'dataset': 'cifar100',
            'recommended_models': ['mobilenet_v2_cifar', 'vit_small_patch4_32'],
            'num_classes': kwargs.get('num_classes', 100)
        }
    }
    
    config = phenomenon_configs.get(phenomenon, {})
    dataset = config.get('dataset', 'cifar10')
    
    # Select model based on type and efficiency
    if model_type == 'auto':
        recommended = config.get('recommended_models', ['mobilenet_v2_cifar'])
        model_name = recommended[0]  # Pick first recommendation
    else:
        # Map model type and efficiency to specific model
        model_mapping = {
            ('vit', 'ultra_light'): 'vit_tiny_patch4_32',
            ('vit', 'light'): 'vit_small_patch4_32',
            ('vit', 'medium'): 'vit_base_patch16_224',
            ('vit', 'large'): 'vit_large_patch16_224',
            ('mobilenet', 'ultra_light'): 'mobilenet_v1_tiny',
            ('mobilenet', 'light'): 'mobilenet_v2_small',
            ('mobilenet', 'medium'): 'mobilenet_v2',
            ('mobilenet', 'large'): 'efficient_mobilenet'
        }
        
        model_name = model_mapping.get((model_type, efficiency), 'mobilenet_v2_cifar')
    
    # Create model
    return ModelFactory.create_model(
        model_name=model_name,
        dataset=dataset,
        **kwargs
    )


if __name__ == "__main__":
    # Test model factory
    print("Testing Model Factory...")
    
    # Test different model types
    models_to_test = [
        ('mobilenet_v2_cifar', 'cifar10'),
        ('mobilenet_v1_tiny', 'cifar100'),
    ]
    
    if VIT_AVAILABLE:
        models_to_test.extend([
            ('vit_small_patch4_32', 'cifar10'),
            ('vit_tiny_patch16_224', 'imagenet')
        ])
    
    for model_name, dataset in models_to_test:
        try:
            print(f"\nTesting {model_name} on {dataset}:")
            model = ModelFactory.create_model(model_name, dataset)
            analysis = ModelFactory.analyze_model(model)
            
            print(f"  Parameters: {analysis['total_parameters']:,}")
            print(f"  Size: {analysis['model_size_mb']:.2f} MB")
            print(f"  Efficiency: {analysis['efficiency_class']}")
            print(f"  Architecture: {analysis['architecture_stats']}")
            
            # Test forward pass
            config = ModelFactory.get_model_config(model_name, dataset)
            input_size = config.get('input_size', 224)
            num_classes = config.get('num_classes', 10)
            
            x = torch.randn(2, 3, input_size, input_size)
            output = model(x)
            print(f"  Output shape: {output.shape}")
            assert output.shape == (2, num_classes), f"Expected shape (2, {num_classes}), got {output.shape}"
            
        except Exception as e:
            print(f"  Error: {e}")
    
    # Test phenomenon-specific model creation
    print("\nTesting phenomenon-specific models:")
    phenomena = ['simplicity_bias', 'robustness', 'continual_learning']
    
    for phenomenon in phenomena:
        try:
            model = create_model_for_phenomenon(phenomenon, efficiency='light')
            analysis = ModelFactory.analyze_model(model)
            print(f"{phenomenon}: {analysis['total_parameters']:,} params, {analysis['efficiency_class']}")
        except Exception as e:
            print(f"{phenomenon}: Error - {e}")
    
    print("\nModel Factory tests completed!")