"""
Image preprocessing utilities - Exactly matching notebook transforms (lines 488-506)
"""
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

def get_transforms(img_size=384, is_train=False):
    """
    Get image transforms - Exactly matching notebook lines 488-506
    """
    if is_train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        # Validation/test transforms - matching notebook line 502-505
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

def preprocess_image(image, img_size=384):
    """
    Preprocess image for model inference - matching notebook
    Args:
        image: numpy array (H, W, C) in RGB format
        img_size: target image size (384 from notebook CONFIG)
    Returns:
        torch tensor ready for inference
    """
    transform = get_transforms(img_size=img_size, is_train=False)
    transformed = transform(image=image)
    return transformed['image']
