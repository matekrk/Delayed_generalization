#!/usr/bin/env python3
"""
Models Package for Delayed Generalization Research

This package provides comprehensive model architectures for studying delayed
generalization phenomena across different domains:

- Vision: CNNs, Vision Transformers, MobileNets
- Language: Transformers, RNNs (future)
- Hybrid: Multi-modal models (future)

Usage:
    from models.vision import ModelFactory, create_model_for_phenomenon
    from models import create_model_for_research
"""

try:
    from .vision import (
        ModelFactory,
        ModelConfig, 
        create_model_for_phenomenon,
        list_available_models as list_vision_models,
        get_model_info as get_vision_info
    )
    VISION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Vision models not available: {e}")
    VISION_AVAILABLE = False


def create_model_for_research(
    research_area: str,
    model_type: str = 'auto',
    efficiency: str = 'medium',
    **kwargs
):
    """
    Create a model optimized for specific research areas in delayed generalization.
    
    Args:
        research_area: Research area ('vision', 'nlp', 'multimodal')
        model_type: Specific model type or 'auto' for automatic selection
        efficiency: Efficiency requirement ('ultra_light', 'light', 'medium', 'large')
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured model instance
    """
    
    if research_area == 'vision':
        if not VISION_AVAILABLE:
            raise ImportError("Vision models not available")
        
        # Map to phenomenon if provided
        phenomenon = kwargs.get('phenomenon', 'general')
        return create_model_for_phenomenon(
            phenomenon=phenomenon,
            model_type=model_type,
            efficiency=efficiency,
            **kwargs
        )
    
    elif research_area == 'nlp':
        # TODO: Implement NLP model creation
        raise NotImplementedError("NLP models not yet implemented")
        
    elif research_area == 'multimodal':
        # TODO: Implement multimodal model creation
        raise NotImplementedError("Multimodal models not yet implemented")
        
    else:
        raise ValueError(f"Unknown research area: {research_area}")


def get_research_capabilities():
    """Get information about available research capabilities."""
    
    capabilities = {
        'vision': {
            'available': VISION_AVAILABLE,
            'phenomena': [
                'grokking',
                'simplicity_bias', 
                'robustness',
                'continual_learning',
                'phase_transitions'
            ] if VISION_AVAILABLE else [],
            'architectures': [
                'vision_transformers',
                'mobilenets',
                'cnns'
            ] if VISION_AVAILABLE else []
        },
        'nlp': {
            'available': False,
            'phenomena': [],
            'architectures': []
        },
        'multimodal': {
            'available': False,
            'phenomena': [],
            'architectures': []
        }
    }
    
    if VISION_AVAILABLE:
        vision_info = get_vision_info()
        capabilities['vision'].update(vision_info)
    
    return capabilities


__all__ = [
    'create_model_for_research',
    'get_research_capabilities'
]

if VISION_AVAILABLE:
    __all__.extend([
        'ModelFactory',
        'ModelConfig',
        'create_model_for_phenomenon'
    ])


if __name__ == "__main__":
    # Test the models package
    print("Testing models package...")
    
    capabilities = get_research_capabilities()
    print(f"Research capabilities: {capabilities}")
    
    if VISION_AVAILABLE:
        print("\nTesting vision model creation...")
        try:
            model = create_model_for_research(
                research_area='vision',
                phenomenon='simplicity_bias',
                efficiency='light',
                num_classes=10
            )
            params = sum(p.numel() for p in model.parameters())
            print(f"Created vision model with {params:,} parameters")
        except Exception as e:
            print(f"Error creating vision model: {e}")
    
    print("Models package test completed!")