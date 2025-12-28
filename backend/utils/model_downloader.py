"""
Model Downloader Utility
Downloads model files from Hugging Face Hub
"""

import os
from pathlib import Path
from huggingface_hub import hf_hub_download
import logging

logger = logging.getLogger(__name__)

def download_model_from_hf(
    repo_id="udayislam/alzheimer-mri-convnext-classifier",
    filename="best_model.pth",
    local_dir="models",
    force_download=False
):
    """
    Download model from Hugging Face Hub
    
    Args:
        repo_id: Hugging Face repository ID
        filename: Name of the model file to download
        local_dir: Local directory to save the model
        force_download: Force re-download even if file exists
    
    Returns:
        Path to the downloaded model file
    """
    local_dir_path = Path(local_dir)
    local_dir_path.mkdir(parents=True, exist_ok=True)
    
    model_path = local_dir_path / filename
    
    # Check if model already exists
    if model_path.exists() and not force_download:
        logger.info(f"Model already exists at {model_path}")
        return str(model_path)
    
    try:
        logger.info(f"Downloading model from Hugging Face: {repo_id}/{filename}")
        logger.info("This may take 30-60 seconds depending on your network speed...")
        
        # Download from Hugging Face Hub
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(local_dir_path),
            local_dir_use_symlinks=False,
            force_download=force_download
        )
        
        logger.info(f"Model downloaded successfully to {downloaded_path}")
        return downloaded_path
        
    except Exception as e:
        logger.error(f"Failed to download model from Hugging Face: {e}")
        raise RuntimeError(
            f"Could not download model from Hugging Face Hub. "
            f"Please check your internet connection and try again. Error: {e}"
        )

def ensure_model_exists(model_path, repo_id="udayislam/alzheimer-mri-convnext-classifier"):
    """
    Ensure model exists, download if necessary
    
    Args:
        model_path: Path where model should exist
        repo_id: Hugging Face repository ID
    
    Returns:
        Path to the model file
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        logger.info(f"Model not found at {model_path}, downloading from Hugging Face...")
        filename = model_path.name
        local_dir = str(model_path.parent)
        return download_model_from_hf(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir
        )
    
    return str(model_path)
