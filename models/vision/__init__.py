#!/usr/bin/env python3
"""
Vision Models Package for Delayed Generalization Research

This package provides state-of-the-art vision model architectures optimized for
studying delayed generalization phenomena:

- Vision Transformers (ViT): Attention-based models for analyzing attention patterns
- MobileNets: Efficient CNNs for resource-constrained settings  
- Model Factory: Unified interface for model creation and configuration

Usage:
    from models.vision import ModelFactory, create_model_for_phenomenon
    
    # Create model for specific phenomenon
    model = create_model_for_phenomenon('grokking', efficiency='light')
    
    # Create model with factory
    model = ModelFactory.create_model('vit_small_patch4_32', 'cifar10')
"""

try:
    from .vision_transformer import (
        VisionTransformer,
        vit_tiny_patch16_224,
        vit_small_patch16_224, 
        vit_base_patch16_224,
        vit_large_patch16_224,
        vit_tiny_patch4_32,
        vit_small_patch4_32
    )
    VIT_MODELS = [
        'VisionTransformer',
        'vit_tiny_patch16_224',
        'vit_small_patch16_224',
        'vit_base_patch16_224', 
        'vit_large_patch16_224',
        'vit_tiny_patch4_32',
        'vit_small_patch4_32'
    ]
except ImportError as e:
    print(f"Warning: Vision Transformer models not available: {e}")
    VIT_MODELS = []

try:
    from .mobilenet import (
        MobileNetV1,
        MobileNetV2,
        EfficientMobileNet,
        mobilenet_v1,
        mobilenet_v2,
        efficient_mobilenet,
        mobilenet_v1_cifar,
        mobilenet_v2_cifar,
        mobilenet_v1_tiny,
        mobilenet_v2_small
    )
    MOBILENET_MODELS = [
        'MobileNetV1',
        'MobileNetV2', 
        'EfficientMobileNet',
        'mobilenet_v1',
        'mobilenet_v2',
        'efficient_mobilenet',
        'mobilenet_v1_cifar',
        'mobilenet_v2_cifar',
        'mobilenet_v1_tiny',
        'mobilenet_v2_small'
    ]
except ImportError as e:
    print(f"Warning: MobileNet models not available: {e}")
    MOBILENET_MODELS = []

try:
    from .model_factory import (
        ModelFactory,
        ModelConfig,
        create_model_for_phenomenon
    )
    FACTORY_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Model Factory not available: {e}")
    FACTORY_AVAILABLE = False

# Export all available models and utilities
__all__ = []

if VIT_MODELS:
    __all__.extend(VIT_MODELS)

if MOBILENET_MODELS:
    __all__.extend(MOBILENET_MODELS)

if FACTORY_AVAILABLE:
    __all__.extend(['ModelFactory', 'ModelConfig', 'create_model_for_phenomenon'])

# Model registry for easy access
MODEL_REGISTRY = {}

# Register ViT models
if VIT_MODELS:
    try:
        MODEL_REGISTRY.update({
            'vit_tiny_patch16_224': vit_tiny_patch16_224,
            'vit_small_patch16_224': vit_small_patch16_224,
            'vit_base_patch16_224': vit_base_patch16_224,
            'vit_large_patch16_224': vit_large_patch16_224,
            'vit_tiny_patch4_32': vit_tiny_patch4_32,
            'vit_small_patch4_32': vit_small_patch4_32,
        })
    except NameError:
        pass

# Register MobileNet models
if MOBILENET_MODELS:
    try:
        MODEL_REGISTRY.update({
            'mobilenet_v1': mobilenet_v1,
            'mobilenet_v2': mobilenet_v2,
            'efficient_mobilenet': efficient_mobilenet,
            'mobilenet_v1_cifar': mobilenet_v1_cifar,
            'mobilenet_v2_cifar': mobilenet_v2_cifar,
            'mobilenet_v1_tiny': mobilenet_v1_tiny,
            'mobilenet_v2_small': mobilenet_v2_small,
        })
    except NameError:
        pass


def list_available_models():
    """List all available models in the registry."""
    return list(MODEL_REGISTRY.keys())


def get_model_info():
    """Get information about available model architectures."""
    info = {
        'vision_transformers': {
            'available': len(VIT_MODELS) > 0,
            'models': VIT_MODELS
        },
        'mobilenets': {
            'available': len(MOBILENET_MODELS) > 0,
            'models': MOBILENET_MODELS
        },
        'factory': {
            'available': FACTORY_AVAILABLE
        },
        'total_models': len(MODEL_REGISTRY)
    }
    return info


def create_model(model_name: str, **kwargs):
    """
    Create a model from the registry.
    
    Args:
        model_name: Name of the model to create
        **kwargs: Arguments to pass to the model constructor
        
    Returns:
        Model instance
    """
    if model_name not in MODEL_REGISTRY:
        available = list_available_models()
        raise ValueError(f"Model '{model_name}' not found. Available models: {available}")
    
    model_fn = MODEL_REGISTRY[model_name]
    return model_fn(**kwargs)


# Convenience function for quick model creation
def quick_model(architecture: str, size: str = 'small', dataset: str = 'cifar10', **kwargs):
    """
    Quickly create a model with common configurations.
    
    Args:
        architecture: Model architecture ('vit', 'mobilenet')
        size: Model size ('tiny', 'small', 'base', 'large')  
        dataset: Target dataset ('cifar10', 'cifar100', 'imagenet')
        **kwargs: Additional model arguments
        
    Returns:
        Model instance
    """
    
    # Map to specific model names
    model_map = {
        ('vit', 'tiny', 'cifar10'): 'vit_tiny_patch4_32',
        ('vit', 'tiny', 'cifar100'): 'vit_tiny_patch4_32',
        ('vit', 'small', 'cifar10'): 'vit_small_patch4_32',
        ('vit', 'small', 'cifar100'): 'vit_small_patch4_32',
        ('vit', 'tiny', 'imagenet'): 'vit_tiny_patch16_224',
        ('vit', 'small', 'imagenet'): 'vit_small_patch16_224',
        ('vit', 'base', 'imagenet'): 'vit_base_patch16_224',
        ('vit', 'large', 'imagenet'): 'vit_large_patch16_224',
        
        ('mobilenet', 'tiny', 'cifar10'): 'mobilenet_v1_cifar',
        ('mobilenet', 'tiny', 'cifar100'): 'mobilenet_v1_cifar',
        ('mobilenet', 'small', 'cifar10'): 'mobilenet_v2_cifar',
        ('mobilenet', 'small', 'cifar100'): 'mobilenet_v2_cifar',
        ('mobilenet', 'tiny', 'imagenet'): 'mobilenet_v1_tiny',
        ('mobilenet', 'small', 'imagenet'): 'mobilenet_v2_small',
        ('mobilenet', 'base', 'imagenet'): 'mobilenet_v2',
        ('mobilenet', 'large', 'imagenet'): 'efficient_mobilenet',
    }
    
    model_name = model_map.get((architecture, size, dataset))
    if model_name is None:
        raise ValueError(f"No model found for {architecture}-{size} on {dataset}")
    
    return create_model(model_name, **kwargs)


if __name__ == "__main__":
    # Test the package
    print("Testing models.vision package...")
    
    # Print available models
    info = get_model_info()
    print(f"Available models: {info}")
    
    # Test model creation if models are available
    if MODEL_REGISTRY:
        print(f"\nTesting model creation...")
        
        # Test first available model
        first_model_name = list(MODEL_REGISTRY.keys())[0]
        try:
            model = create_model(first_model_name, num_classes=10)
            print(f"Successfully created {first_model_name}")
            print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        except Exception as e:
            print(f"Error creating {first_model_name}: {e}")
        
        # Test quick model creation
        try:
            if 'mobilenet_v2_cifar' in MODEL_REGISTRY:
                quick = quick_model('mobilenet', 'small', 'cifar10', num_classes=10)
                print(f"Quick model created successfully")
            elif 'vit_tiny_patch4_32' in MODEL_REGISTRY:
                quick = quick_model('vit', 'tiny', 'cifar10', num_classes=10)
                print(f"Quick ViT model created successfully")
        except Exception as e:
            print(f"Error with quick model: {e}")
    
    else:
        print("No models available for testing")
    
    print("Package test completed!")