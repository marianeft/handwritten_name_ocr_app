#utils_ocr.py

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import os

# Import config for IMG_HEIGHT and MAX_IMG_WIDTH
from config import IMG_HEIGHT, MAX_IMG_WIDTH

# --- Image Preprocessing Functions ---

def load_image_as_grayscale(image_path: str) -> Image.Image:
    """Loads an image from path and converts it to grayscale PIL Image."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")
    return Image.open(image_path).convert('L') # 'L' for grayscale

def binarize_image(img: Image.Image) -> Image.Image:
    """
    Binarizes a grayscale PIL Image using Otsu's method.
    Returns a PIL Image.
    """
    # Convert PIL Image to OpenCV format (numpy array)
    img_np = np.array(img)
    
    # Apply Otsu's binarization
    # cv2.THRESH_BINARY_INV means white text on black background
    # cv2.THRESH_BINARY means black text on white background
    # We want white text on black background for consistency with typical OCR datasets (value > threshold = 0, else 255)
    _, binary_img = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Convert back to PIL Image
    return Image.fromarray(binary_img)

def resize_image_for_ocr(img: Image.Image, img_height: int) -> Image.Image:
    """
    Resizes a PIL Image to a fixed height while maintaining aspect ratio.
    Also ensures the width does not exceed MAX_IMG_WIDTH.
    """
    width, height = img.size
    
    # Calculate new width based on target height, maintaining aspect ratio
    new_width = int(width * (img_height / height))
    
    # NEW: If the calculated new_width exceeds MAX_IMG_WIDTH, scale down proportionally
    if new_width > MAX_IMG_WIDTH:
        new_width = MAX_IMG_WIDTH
        # Recalculate height to maintain aspect ratio with the new max width
        # This might not be strictly necessary if IMG_HEIGHT is fixed, but good practice
        # to ensure consistency if MAX_IMG_WIDTH forces a downscale.
        # However, since IMG_HEIGHT is fixed, we should prioritize it.
        # The primary goal here is to cap the width.
        # If the image is too wide, we'll just cap its width and keep the height.
        # This means aspect ratio might be slightly distorted if the original width was very large.
        # For OCR, maintaining height is often more critical.
        
        # A better approach for capping width while preserving aspect ratio for a fixed height:
        # If new_width > MAX_IMG_WIDTH, we need to resize to MAX_IMG_WIDTH,
        # which would imply a new height. But we have a fixed IMG_HEIGHT.
        # So, if new_width > MAX_IMG_WIDTH, we must *crop* or accept distortion.
        # For simplicity and to avoid distortion for a fixed height, we'll just cap the width.
        # This means the image will be scaled down to fit the height, and then if its width
        # is still too large, it will be resized to MAX_IMG_WIDTH, potentially distorting aspect ratio.
        # A more advanced approach might involve cropping the width or splitting the image.
        
        # Let's refine: If the image, when scaled to IMG_HEIGHT, is wider than MAX_IMG_WIDTH,
        # we should scale it down to MAX_IMG_WIDTH while maintaining its aspect ratio.
        # This means its height will become *less* than IMG_HEIGHT.
        # Then, we'll pad it back up to IMG_HEIGHT if needed.
        
        # Simpler approach: Resize to IMG_HEIGHT. If resulting width > MAX_IMG_WIDTH, crop it.
        # This is common in OCR to handle very long lines of text.
        resized_img = img.resize((new_width, img_height), Image.Resampling.LANCZOS)
        if resized_img.width > MAX_IMG_WIDTH:
            # Crop the image from the left to MAX_IMG_WIDTH
            resized_img = resized_img.crop((0, 0, MAX_IMG_WIDTH, img_height))
        return resized_img
    
    return img.resize((new_width, img_height), Image.Resampling.LANCZOS) # Use LANCZOS for high-quality downsampling

def normalize_image_for_model(img_tensor: torch.Tensor) -> torch.Tensor:
    """
    Normalizes a torch.Tensor image (grayscale) for input into the model.
    Puts pixel values in range [-1, 1].
    Assumes image is already a torch.Tensor with values in [0, 1] (e.g., after ToTensor).
    """
    # Formula: (pixel_value - mean) / std_dev
    # For [0, 1] to [-1, 1], mean = 0.5, std_dev = 0.5
    img_tensor = (img_tensor - 0.5) / 0.5 
    return img_tensor

def preprocess_user_image_for_ocr(image_pil: Image.Image, target_height: int) -> torch.Tensor:
    """
    Applies all necessary preprocessing steps to a user-uploaded PIL Image
    to prepare it for the OCR model.
    """
    # Define a transformation pipeline similar to the dataset, but including ToTensor
    transform_pipeline = transforms.Compose([
        transforms.Lambda(lambda img: binarize_image(img)), # PIL Image -> PIL Image
        # Use the updated resize function that also handles MAX_IMG_WIDTH
        transforms.Lambda(lambda img: resize_image_for_ocr(img, target_height)), # PIL Image -> PIL Image
        transforms.ToTensor(), # PIL Image -> Tensor [0, 1]
        transforms.Lambda(normalize_image_for_model) # Tensor [0, 1] -> Tensor [-1, 1]
    ])
    
    processed_image = transform_pipeline(image_pil)
    
    # Add a batch dimension (C, H, W) -> (1, C, H, W) for single image inference
    return processed_image.unsqueeze(0)