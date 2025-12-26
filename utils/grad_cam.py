# Grad-CAM++ (Gradient-weighted Class Activation Mapping Plus Plus)
# Enhanced explainability method with better gradient weighting

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional

class GradCAMPlusPlus:
    """
    Grad-CAM++ - Enhanced Grad-CAM with improved gradient weighting
    
    Reference: "Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks"
    https://arxiv.org/abs/1710.11063
    """
    
    def __init__(self, model, target_layer):
        """
        Initialize Grad-CAM++
        
        Args:
            model: PyTorch model
            target_layer: Target convolutional layer for visualization
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Register hooks
        forward_hook_handle = self.target_layer.register_forward_hook(forward_hook)
        backward_hook_handle = self.target_layer.register_backward_hook(backward_hook)
        
        self.hooks.append(forward_hook_handle)
        self.hooks.append(backward_hook_handle)
    
    def remove_hooks(self):
        """Remove registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def generate_cam(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate Grad-CAM++ heatmap
        
        Args:
            input_tensor: Input image tensor (B, C, H, W)
            target_class: Target class for visualization (None for predicted class)
        
        Returns:
            Grad-CAM++ heatmap (H, W)
        """
        batch_size, _, height, width = input_tensor.shape
        
        # Forward pass
        self.model.eval()
        with torch.enable_grad():
            input_tensor.requires_grad_(True)
            output = self.model(input_tensor)
            
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            
            # Backward pass
            self.model.zero_grad()
            target_output = output[0, target_class]
            target_output.backward()
        
        # Get gradients and activations
        gradients = self.gradients.cpu().numpy()
        activations = self.activations.cpu().numpy()
        
        # Calculate second derivative
        second_derivative = gradients ** 2
        third_derivative = second_derivative * gradients
        
        # Calculate weights using Grad-CAM++ formula
        numerator = second_derivative
        denominator = 2 * second_derivative + np.sum(third_derivative, axis=(2, 3), keepdims=True)
        denominator = np.where(denominator == 0, 1, denominator)  # Avoid division by zero
        
        weights = numerator / denominator
        
        # Apply ReLU to weights
        weights = np.maximum(weights, 0)
        
        # Calculate weighted activation map
        cam = np.sum(weights * activations, axis=1)[0]  # (H, W)
        
        # Apply ReLU
        cam = np.maximum(cam, 0)
        
        # Normalize
        cam_min = cam.min()
        cam_max = cam.max()
        
        if cam_max - cam_min != 0:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        
        return cam
    
    def generate_and_visualize(self, 
                              input_tensor: torch.Tensor,
                              original_image: np.ndarray,
                              target_class: Optional[int] = None,
                              alpha: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Grad-CAM++ and visualize on original image
        
        Args:
            input_tensor: Input tensor
            original_image: Original image as numpy array (H, W, 3)
            target_class: Target class for visualization
            alpha: Transparency of overlay
        
        Returns:
            Tuple of (overlay_image, heatmap)
        """
        # Generate CAM
        cam = self.generate_cam(input_tensor, target_class)
        
        # Resize CAM to match original image
        cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
        
        # Convert to 0-255 range
        cam_resized = (cam_resized * 255).astype(np.uint8)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Create overlay
        overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)
        
        return overlay.astype(np.uint8), heatmap.astype(np.uint8)
    
    def generate_multiple_cams(self, 
                              input_tensor: torch.Tensor,
                              original_image: np.ndarray,
                              num_classes: int) -> dict:
        """
        Generate Grad-CAM++ for multiple classes
        
        Args:
            input_tensor: Input tensor
            original_image: Original image
            num_classes: Number of classes
        
        Returns:
            Dictionary with CAMs for each class
        """
        results = {}
        
        for class_idx in range(num_classes):
            try:
                overlay, heatmap = self.generate_and_visualize(
                    input_tensor, original_image, target_class=class_idx
                )
                results[class_idx] = {
                    'overlay': overlay,
                    'heatmap': heatmap
                }
            except Exception as e:
                print(f"Error generating CAM for class {class_idx}: {e}")
                continue
        
        return results


class GradCAMPlus:
    """Alternative Grad-CAM implementation (simpler version)"""
    
    def __init__(self, model, target_layer):
        """Initialize Grad-CAM"""
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        forward_hook_handle = self.target_layer.register_forward_hook(forward_hook)
        backward_hook_handle = self.target_layer.register_backward_hook(backward_hook)
        
        self.hooks.append(forward_hook_handle)
        self.hooks.append(backward_hook_handle)
    
    def remove_hooks(self):
        """Remove hooks"""
        for hook in self.hooks:
            hook.remove()
    
    def generate_cam(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap (simpler version)
        
        Args:
            input_tensor: Input tensor
            target_class: Target class
        
        Returns:
            CAM heatmap
        """
        # Forward pass
        self.model.eval()
        with torch.enable_grad():
            input_tensor.requires_grad_(True)
            output = self.model(input_tensor)
            
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            
            # Backward pass
            self.model.zero_grad()
            output[0, target_class].backward()
        
        # Calculate weights
        gradients = self.gradients.cpu()
        activations = self.activations.cpu()
        
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1)
        cam = F.relu(cam[0])
        
        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam.numpy()
    
    def visualize(self,
                 input_tensor: torch.Tensor,
                 original_image: np.ndarray,
                 target_class: Optional[int] = None,
                 alpha: float = 0.5) -> np.ndarray:
        """
        Visualize CAM on original image
        
        Args:
            input_tensor: Input tensor
            original_image: Original image
            target_class: Target class
            alpha: Transparency
        
        Returns:
            Overlay image
        """
        cam = self.generate_cam(input_tensor, target_class)
        
        # Resize CAM
        cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
        cam_resized = (cam_resized * 255).astype(np.uint8)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_TURBO)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)
        
        return overlay.astype(np.uint8)

# Wrapper functions for easy use with Flask app

def _find_target_layer(model):
    """Helper function to find target layer for Grad-CAM"""
    target_layer = None
    if hasattr(model, 'backbone'):
        # For ConvNeXt, find last stage
        if hasattr(model.backbone, 'stages'):
            target_layer = model.backbone.stages[-1]
        elif hasattr(model.backbone, 'features'):
            # For DenseNet, find last Conv2d in features
            for module in reversed(list(model.backbone.features.modules())):
                if isinstance(module, nn.Conv2d):
                    target_layer = module
                    break
        else:
            # Fallback: find last Conv2d layer
            for module in reversed(list(model.backbone.modules())):
                if isinstance(module, nn.Conv2d):
                    target_layer = module
                    break
    else:
        # Fallback: find last Conv2d in entire model
        for module in reversed(list(model.modules())):
            if isinstance(module, nn.Conv2d):
                target_layer = module
                break
    return target_layer

class _ModelWrapper(nn.Module):
    """Wrapper to make model return only output for Grad-CAM compatibility"""
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        output, _ = self.model(x)
        return output

def generate_gradcam_plusplus(model, input_tensor, target_class, original_image, device='cuda'):
    """
    Generate and overlay Grad-CAM++ heatmap on original image
    Wrapper function for easy use with Flask app
    
    Args:
        model: PyTorch model (returns (output, features))
        input_tensor: Preprocessed input tensor (B, C, H, W)
        target_class: Target class index for visualization
        original_image: Original image as numpy array (H, W, 3) RGB
        device: Device to run on
    
    Returns:
        Overlay image with Grad-CAM++ heatmap
    """
    try:
        # Wrap model to return only output (Grad-CAM expects this)
        wrapped_model = _ModelWrapper(model)
        wrapped_model.eval()
        
        # Find target layer in original model's backbone
        target_layer = _find_target_layer(model)
        
        if target_layer is None:
            print("Warning: Could not find target layer, returning original image")
            return original_image
        
        # Create Grad-CAM++ instance with wrapped model
        gradcam = GradCAMPlusPlus(wrapped_model, target_layer)
        
        try:
            # Ensure input_tensor is on correct device
            if isinstance(input_tensor, torch.Tensor):
                input_tensor_grad = input_tensor.clone().detach().to(device).requires_grad_(True)
            else:
                input_tensor_grad = torch.tensor(input_tensor).to(device).requires_grad_(True)
            
            # Generate overlay
            overlay, _ = gradcam.generate_and_visualize(
                input_tensor_grad,
                original_image,
                target_class=target_class,
                alpha=0.5
            )
            
            # Clean up hooks
            gradcam.remove_hooks()
            
            return overlay
        except Exception as e:
            print(f"Error generating Grad-CAM: {e}")
            import traceback
            traceback.print_exc()
            try:
                gradcam.remove_hooks()
            except:
                pass
            return original_image
    except Exception as e:
        print(f"Error in generate_gradcam_plusplus: {e}")
        import traceback
        traceback.print_exc()
        return original_image

def generate_lime_explanation(model, original_image, processed_tensor, class_names, device='cuda', num_samples=50):
    """
    Generate LIME explanation for the prediction.
    
    Args:
        model: PyTorch model
        original_image: Original image as numpy array (H, W, 3) RGB
        processed_tensor: Preprocessed tensor (not used but kept for compatibility)
        class_names: Dictionary of class names
        device: Device to run on
        num_samples: Number of LIME samples (default 50 for performance)
    
    Returns:
        LIME explanation image
    """
    try:
        from lime import lime_image
        from skimage.segmentation import mark_boundaries
    except ImportError:
        print("LIME not available, returning original image")
        return original_image
    
    try:
        # Import preprocessor
        try:
            from .preprocessor import preprocess_image
        except ImportError:
            from utils.preprocessor import preprocess_image
        
        def batch_predict(images):
            """Batch prediction function for LIME"""
            model.eval()
            batch_tensors = []
            for img in images:
                processed = preprocess_image(img, 384)
                batch_tensors.append(processed)
            
            batch = torch.stack(batch_tensors).to(device)
            
            with torch.no_grad():
                outputs, _ = model(batch)
                probs = F.softmax(outputs, dim=1)
            
            return probs.cpu().numpy()
        
        # Create LIME explainer
        explainer = lime_image.LimeImageExplainer()
        
        # Generate explanation with configurable samples
        explanation = explainer.explain_instance(
            original_image,
            batch_predict,
            top_labels=1,
            hide_color=0,
            num_samples=num_samples
        )
        
        # Get mask for top prediction
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=10,
            hide_rest=False
        )
        
        # Create visualization
        lime_img = mark_boundaries(temp / 255.0, mask)
        lime_img = (lime_img * 255).astype(np.uint8)
        
        return lime_img
    except Exception as e:
        print(f"Error generating LIME explanation: {e}")
        import traceback
        traceback.print_exc()
        return original_image
