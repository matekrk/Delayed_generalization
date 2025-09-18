#!/usr/bin/env python3
"""
GradCAM Implementation for Data Attribution in Delayed Generalization Research

This module provides GradCAM (Gradient-weighted Class Activation Mapping) 
implementation for visualizing which parts of an image contribute most to 
the model's predictions. This is useful for analyzing delayed generalization
phenomena and detecting simplicity bias.

Based on "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
by Selvaraju et al. (2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import warnings

# Optional import for OpenCV
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    warnings.warn("OpenCV not available. Image resizing will use simpler methods.")


class GradCAM:
    """
    GradCAM implementation for CNN-based models.
    
    This class provides gradient-based localization to understand which regions
    of the input image are most important for the model's predictions.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: Union[str, nn.Module],
        use_cuda: bool = True
    ):
        """
        Initialize GradCAM.
        
        Args:
            model: The neural network model
            target_layer: Target layer for GradCAM (layer name string or module)
            use_cuda: Whether to use CUDA if available
        """
        self.model = model
        self.target_layer = target_layer
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        
        self.model.eval()
        self.model.to(self.device)
        
        # Storage for gradients and feature maps
        self.gradients = None
        self.feature_maps = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""
        if isinstance(self.target_layer, str):
            # Find layer by name
            target_module = self._find_layer_by_name(self.target_layer)
        else:
            target_module = self.target_layer
        
        if target_module is None:
            raise ValueError(f"Could not find target layer: {self.target_layer}")
        
        # Forward hook to capture feature maps
        def forward_hook(module, input, output):
            self.feature_maps = output
        
        # Backward hook to capture gradients
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        target_module.register_forward_hook(forward_hook)
        target_module.register_backward_hook(backward_hook)
    
    def _find_layer_by_name(self, layer_name: str) -> Optional[nn.Module]:
        """Find a layer in the model by its name."""
        for name, module in self.model.named_modules():
            if name == layer_name:
                return module
        
        # If exact match not found, try partial match
        matches = []
        for name, module in self.model.named_modules():
            if layer_name in name:
                matches.append((name, module))
        
        if len(matches) == 1:
            return matches[0][1]
        elif len(matches) > 1:
            warnings.warn(f"Multiple layers match '{layer_name}': {[m[0] for m in matches]}. Using first match.")
            return matches[0][1]
        
        return None
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        guided: bool = False
    ) -> np.ndarray:
        """
        Generate Class Activation Map using GradCAM.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class index (if None, uses predicted class)
            guided: Whether to use guided backpropagation
            
        Returns:
            CAM as numpy array
        """
        # Ensure input is on correct device and requires grad
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        class_score = output[0, target_class]
        class_score.backward(retain_graph=True)
        
        # Generate CAM
        if self.gradients is None or self.feature_maps is None:
            raise RuntimeError("Gradients or feature maps not captured. Check target layer.")
        
        # Global average pooling of gradients to get weights
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=False)  # (batch, channels)
        
        # Weighted combination of feature maps
        cam = torch.zeros(self.feature_maps.shape[2:], device=self.device)
        for i in range(weights.shape[1]):
            cam += weights[0, i] * self.feature_maps[0, i]
        
        # Apply ReLU to focus on positive influences
        cam = F.relu(cam)
        
        # Normalize CAM
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # Convert to numpy and resize to input size
        cam = cam.cpu().detach().numpy()
        input_h, input_w = input_tensor.shape[2], input_tensor.shape[3]
        
        if CV2_AVAILABLE:
            cam = cv2.resize(cam, (input_w, input_h))
        else:
            # Simple interpolation fallback
            from scipy.ndimage import zoom
            zoom_h = input_h / cam.shape[0]
            zoom_w = input_w / cam.shape[1]
            cam = zoom(cam, (zoom_h, zoom_w), order=1)
        
        return cam
    
    def visualize_cam(
        self,
        input_tensor: torch.Tensor,
        cam: np.ndarray,
        title: str = "GradCAM",
        save_path: Optional[str] = None,
        alpha: float = 0.4
    ) -> None:
        """
        Visualize GradCAM overlay on original image.
        
        Args:
            input_tensor: Original input tensor (1, C, H, W)
            cam: Generated CAM
            title: Plot title
            save_path: Path to save the visualization
            alpha: Transparency of CAM overlay
        """
        # Convert input tensor to numpy
        if input_tensor.shape[1] == 3:  # RGB
            img = input_tensor[0].cpu().detach().numpy().transpose(1, 2, 0)
        else:  # Grayscale
            img = input_tensor[0, 0].cpu().detach().numpy()
        
        # Normalize image to [0, 1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # CAM
        im1 = axes[1].imshow(cam, cmap='jet')
        axes[1].set_title('GradCAM')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Overlay
        axes[2].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        axes[2].imshow(cam, cmap='jet', alpha=alpha)
        axes[2].set_title(f'{title} Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()