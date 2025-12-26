# Explainable AI Methods
# Including LIME, SHAP-like analysis, and advanced interpretation

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from PIL import Image
import json

class ExplainablePrediction:
    """Comprehensive explainability for model predictions"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
    
    def generate_saliency_map(self, 
                            image_tensor: torch.Tensor,
                            target_class: int) -> np.ndarray:
        """
        Generate saliency map using gradient-based method
        
        Args:
            image_tensor: Input tensor (1, C, H, W)
            target_class: Target class index
        
        Returns:
            Saliency map (H, W)
        """
        image_tensor = image_tensor.clone().detach()
        image_tensor.requires_grad_(True)
        
        self.model.eval()
        output = self.model(image_tensor)
        
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Get gradients
        gradients = image_tensor.grad.abs()
        saliency = gradients.max(dim=1)[0].squeeze().cpu().numpy()
        
        # Normalize
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        
        return saliency
    
    def generate_attention_map(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Generate attention map from model features
        
        Args:
            image_tensor: Input tensor
        
        Returns:
            Attention map
        """
        self.model.eval()
        
        # Extract features from intermediate layers
        with torch.no_grad():
            features = image_tensor
            for layer in self.model.features:
                features = layer(features)
            
            # Global average pooling
            attention = features.mean(dim=1, keepdim=True)
            attention = torch.clamp(attention, 0, 1)
        
        return attention.squeeze().cpu().numpy()
    
    def get_layer_activations(self, image_tensor: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Get activations from multiple layers
        
        Args:
            image_tensor: Input tensor
        
        Returns:
            Dictionary of layer activations
        """
        activations = {}
        self.model.eval()
        
        with torch.no_grad():
            x = image_tensor
            for name, layer in self.model.features.named_children():
                x = layer(x)
                if isinstance(layer, torch.nn.MaxPool2d):
                    activations[f'pool_{name}'] = x.mean(dim=1)[0].cpu().numpy()
        
        return activations


def explain_prediction(predicted_class: int,
                      confidence: float,
                      class_names: Dict,
                      class_probs: Dict,
                      probabilities: np.ndarray) -> Dict:
    """
    Generate comprehensive explanation for prediction
    
    Args:
        predicted_class: Predicted class index
        confidence: Confidence score (0-100)
        class_names: Dictionary mapping class indices to names
        class_probs: Dictionary of class probabilities
        probabilities: Raw probability array
    
    Returns:
        Dictionary with explanation details
    """
    
    predicted_name = class_names[str(predicted_class)]
    
    # Confidence assessment
    if confidence >= 90:
        confidence_level = "Very High"
        confidence_description = "The model is very confident in this prediction"
    elif confidence >= 75:
        confidence_level = "High"
        confidence_description = "The model is confident in this prediction"
    elif confidence >= 60:
        confidence_level = "Moderate"
        confidence_description = "The model has moderate confidence in this prediction"
    elif confidence >= 40:
        confidence_level = "Low"
        confidence_description = "The model has low confidence in this prediction. Consider manual verification."
    else:
        confidence_level = "Very Low"
        confidence_description = "The model is uncertain. Manual verification is strongly recommended."
    
    # Get top alternatives
    sorted_probs = sorted(
        [(class_names[str(i)], float(p) * 100) for i, p in enumerate(probabilities)],
        key=lambda x: x[1],
        reverse=True
    )
    
    top_alternatives = sorted_probs[1:4] if len(sorted_probs) > 1 else []
    
    # Decision margin
    decision_margin = sorted_probs[0][1] - sorted_probs[1][1] if len(sorted_probs) > 1 else 100
    
    if decision_margin > 30:
        margin_description = "Clear distinction from other classes"
    elif decision_margin > 15:
        margin_description = "Moderate distinction from other classes"
    else:
        margin_description = "Close competition with other classes"
    
    # Key findings
    key_findings = []
    key_findings.append(f"Primary prediction: {predicted_name} ({confidence:.1f}%)")
    
    if top_alternatives:
        key_findings.append(f"Main alternative: {top_alternatives[0][0]} ({top_alternatives[0][1]:.1f}%)")
        key_findings.append(f"Decision margin: {margin_description}")
    
    if confidence < 70:
        key_findings.append("⚠ Low confidence - recommend specialist review")
    
    # Risk assessment
    risk_level = "Low" if confidence >= 85 else "Medium" if confidence >= 60 else "High"
    
    return {
        'predicted_class': predicted_name,
        'confidence': round(confidence, 2),
        'confidence_level': confidence_level,
        'confidence_description': confidence_description,
        'top_alternatives': top_alternatives,
        'decision_margin': round(decision_margin, 2),
        'margin_description': margin_description,
        'key_findings': key_findings,
        'risk_level': risk_level,
        'all_probabilities': class_probs,
        'timestamp': str(np.datetime64('now'))
    }


def generate_confidence_scores(output: torch.Tensor, class_names: Dict) -> Dict:
    """
    Generate confidence scores for all classes
    
    Args:
        output: Model output tensor
        class_names: Dictionary mapping class indices to names
    
    Returns:
        Dictionary of confidence scores
    """
    probabilities = torch.softmax(output, dim=1)[0]
    
    scores = {}
    for idx, class_name in class_names.items():
        prob = probabilities[int(idx)].item() * 100
        scores[class_name] = round(prob, 2)
    
    return scores


def generate_uncertainty_estimate(probabilities: np.ndarray) -> Dict:
    """
    Generate uncertainty estimate from probabilities
    
    Args:
        probabilities: Array of class probabilities
    
    Returns:
        Dictionary with uncertainty metrics
    """
    
    # Entropy-based uncertainty
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
    max_entropy = np.log(len(probabilities))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    # Confidence margin
    sorted_probs = np.sort(probabilities)[::-1]
    margin = (sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 1.0
    
    return {
        'entropy': float(entropy),
        'normalized_entropy': float(normalized_entropy),
        'confidence_margin': float(margin),
        'uncertainty_score': float(1 - (margin + (1 - normalized_entropy)) / 2)
    }


def generate_risk_assessment(confidence: float, uncertainty: Dict) -> Dict:
    """
    Generate risk assessment for clinical decision-making
    
    Args:
        confidence: Confidence percentage
        uncertainty: Uncertainty metrics
    
    Returns:
        Risk assessment dictionary
    """
    
    risk_score = 1 - (confidence / 100)
    
    if confidence >= 90:
        risk_level = "Low"
        recommendation = "Result is suitable for clinical review"
    elif confidence >= 75:
        risk_level = "Moderate"
        recommendation = "Result should be reviewed by specialist"
    elif confidence >= 60:
        risk_level = "High"
        recommendation = "Result requires specialist confirmation"
    else:
        risk_level = "Very High"
        recommendation = "Result should not be used. Require specialist evaluation."
    
    return {
        'risk_level': risk_level,
        'risk_score': round(risk_score, 3),
        'recommendation': recommendation,
        'confidence_threshold_met': confidence >= 75
    }


def interpret_prediction_detailed(predicted_class: int,
                                 confidence: float,
                                 class_names: Dict) -> str:
    """
    Generate human-readable detailed interpretation
    
    Args:
        predicted_class: Predicted class index
        confidence: Confidence percentage
        class_names: Class name mappings
    
    Returns:
        Detailed interpretation string
    """
    
    class_name = class_names[str(predicted_class)]
    
    if confidence > 90:
        confidence_text = "extremely confident"
        modifier = "The strong confidence level suggests reliable classification"
    elif confidence > 75:
        confidence_text = "confident"
        modifier = "The high confidence indicates solid classification"
    elif confidence > 60:
        confidence_text = "moderately confident"
        modifier = "Manual verification is recommended due to moderate confidence"
    elif confidence > 40:
        confidence_text = "somewhat uncertain"
        modifier = "Specialist review is strongly recommended"
    else:
        confidence_text = "very uncertain"
        modifier = "This prediction should not be relied upon without expert review"
    
    interpretation = (
        f"The model predicts '{class_name}' with {confidence_text} confidence "
        f"({confidence:.1f}%). {modifier}."
    )
    
    return interpretation


def generate_feature_importance_report(model,
                                     image_tensor: torch.Tensor,
                                     target_class: int) -> Dict:
    """
    Generate feature importance report
    
    Args:
        model: PyTorch model
        image_tensor: Input tensor
        target_class: Target class
    
    Returns:
        Feature importance report
    """
    
    explainer = ExplainablePrediction(model)
    
    # Generate saliency map
    saliency = explainer.generate_saliency_map(image_tensor, target_class)
    
    # Generate attention map
    attention = explainer.generate_attention_map(image_tensor)
    
    # Calculate statistics
    saliency_stats = {
        'mean': float(np.mean(saliency)),
        'std': float(np.std(saliency)),
        'min': float(np.min(saliency)),
        'max': float(np.max(saliency))
    }
    
    # Top regions of interest
    saliency_flat = saliency.flatten()
    top_indices = np.argsort(saliency_flat)[-10:][::-1]
    top_values = saliency_flat[top_indices]
    
    return {
        'saliency_map': saliency,
        'attention_map': attention,
        'saliency_statistics': saliency_stats,
        'top_regions_importance': float(np.mean(top_values))
    }
