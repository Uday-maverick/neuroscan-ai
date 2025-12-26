"""
Model loading and inference utilities - Exactly matching notebook architecture
"""
import torch
import torch.nn as nn
import timm
from pathlib import Path
import json

class ConvNextModel(nn.Module):
    """
    ConvNeXt model - Exactly matching notebook lines 609-634
    """
    def __init__(self, num_classes=4):
        super().__init__()
        # Match notebook line 613: pretrained=True
        self.backbone = timm.create_model('convnext_base', pretrained=True, num_classes=0)
        # Match notebook lines 614-619: classifier with dropout 0.5
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),  # Notebook line 617
            nn.Linear(512, num_classes)
        )
        self.gradients = None
        self.activations = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward(self, x):
        features = self.backbone(x)
        # Match notebook: register gradient hook for Grad-CAM
        if features.requires_grad:
            features.register_hook(self.save_gradient)
        self.activations = features
        out = self.classifier(features)
        return out, features

class DenseNetModel(nn.Module):
    """
    DenseNet model - Matching notebook lines 636-661
    """
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = timm.create_model('densenet201', pretrained=True, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        self.gradients = None
        self.activations = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward(self, x):
        features = self.backbone(x)
        if features.requires_grad:
            features.register_hook(self.save_gradient)
        self.activations = features
        out = self.classifier(features)
        return out, features

def load_model(model_path, model_name='ConvNeXt', num_classes=4, device='cuda'):
    """
    Load pretrained model from checkpoint - matching notebook format
    Notebook saves: {'model_state_dict': ..., 'epoch': ..., 'accuracy': ..., 'metrics': ...}
    """
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    
    # Create model architecture matching notebook exactly
    if model_name == 'ConvNeXt':
        model = ConvNextModel(num_classes=num_classes)
    elif model_name == 'DenseNet':
        model = DenseNetModel(num_classes=num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported. Use 'ConvNeXt' or 'DenseNet'")
    
    # Load weights - handle notebook checkpoint format
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        # Older PyTorch versions don't have weights_only
        checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats from notebook
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        # Direct state dict
        state_dict = checkpoint
    
    # Handle compiled models (torch.compile adds '_orig_mod.' prefix)
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    
    # Load state dict
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"Warning: Strict loading failed: {e}")
        print("Attempting with strict=False...")
        model.load_state_dict(state_dict, strict=False)
    
    model.to(device)
    model.eval()
    
    return model

def predict_image(model, image_tensor, device='cuda'):
    """
    Make prediction on a single image - matching notebook inference
    """
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)
        outputs, _ = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()
    
    return pred_class, confidence, probs[0].cpu().numpy()