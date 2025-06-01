<<<<<<< HEAD
#utils_ocr.py

import cv2
from matplotlib.pylab import f
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

# --- Image Preprocessing for OCR ---

def load_image_as_grayscale(image_path: str) -> Image.Image:
    """Loads an image from path and converts it to grayscale PIL Image."""
    # Use PIL for robust image loading and conversion to grayscale 'L' mode
    img = Image.open(image_path).convert('L')
    return img

def binarize_image(image_pil: Image.Image) -> Image.Image:
    """Binarizes a grayscale PIL Image (black and white)."""
    # Convert PIL to OpenCV format (numpy array)
    img_np = np.array(image_pil)
    # Apply Otsu's thresholding for adaptive binarization
    _, img_bin = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Invert colors: Handwritten text usually dark on light. OCR models often
    # prefer light text on dark background. Check your training data's style.
    # This example assumes dark text on light background and inverts to white text on black.
    img_bin = 255 - img_bin
    return Image.fromarray(img_bin)

def resize_image_for_ocr(image_pil: Image.Image, target_height: int) -> Image.Image:
    """
    Resizes a PIL Image to a target height while maintaining aspect ratio.
    Pads width if necessary to avoid distortion.
    """
    original_width, original_height = image_pil.size
    # Calculate new width based on target height and original aspect ratio
    new_width = int(original_width * (target_height / original_height))
    resized_img = image_pil.resize((new_width, target_height), Image.LANCZOS)
    return resized_img

def normalize_image_for_model(image_pil: Image.Image) -> torch.Tensor:
    """
    Converts a PIL Image to a PyTorch Tensor and normalizes pixel values.
    """
    # Convert to tensor (scales to 0-1 automatically)
    tensor_transform = transforms.ToTensor()
    img_tensor = tensor_transform(image_pil)
    # For grayscale images, mean and std are single values.
    # Adjust normalization values if your training data uses different ones.
    img_tensor = transforms.Normalize((0.5,), (0.5,))(img_tensor) # Normalize to [-1, 1]
    return img_tensor

def preprocess_user_image_for_ocr(uploaded_image_pil: Image.Image, target_height: int) -> torch.Tensor:
    """
    Combines all preprocessing steps for a single user-uploaded image
    to prepare it for the OCR model.
    """
    # Ensure it's grayscale
    img_gray = uploaded_image_pil.convert('L')

    # Binarize
    img_bin = binarize_image(img_gray)

    # Resize (maintain aspect ratio)
    img_resized = resize_image_for_ocr(img_bin, target_height)

    # Normalize and convert to tensor
    img_tensor = normalize_image_for_model(img_resized)

    # Add batch dimension: (C, H, W) -> (1, C, H, W)
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor

def pad_image_tensor(image_tensor: torch.Tensor, max_width: int) -> torch.Tensor:
    """
    Pads a single image tensor to a max_width with zeros.
    Input tensor shape: (C, H, W)
    Output tensor shape: (C, H, max_width)
    """
    C, H, W = image_tensor.shape
    if W > max_width:
        # If image is wider than max_width, you might want to crop or resize it.
        # For this example, we'll just return a warning or clip.
        # A more robust solution might split text lines or use a different resizing strategy.
        print(f"Warning: Image width {W} exceeds max_width {max_width}. Cropping.")
        return image_tensor[:, :, :max_width] # Simple cropping
    padding = max_width - W
    # Pad on the right (P_left, P_right, P_top, P_bottom)
    padded_tensor = f.pad(image_tensor, (0, padding), 'constant', 0)
=======
#utils_ocr.py

import cv2
from matplotlib.pylab import f
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

# --- Image Preprocessing for OCR ---

def load_image_as_grayscale(image_path: str) -> Image.Image:
    """Loads an image from path and converts it to grayscale PIL Image."""
    # Use PIL for robust image loading and conversion to grayscale 'L' mode
    img = Image.open(image_path).convert('L')
    return img

def binarize_image(image_pil: Image.Image) -> Image.Image:
    """Binarizes a grayscale PIL Image (black and white)."""
    # Convert PIL to OpenCV format (numpy array)
    img_np = np.array(image_pil)
    # Apply Otsu's thresholding for adaptive binarization
    _, img_bin = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Invert colors: Handwritten text usually dark on light. OCR models often
    # prefer light text on dark background. Check your training data's style.
    # This example assumes dark text on light background and inverts to white text on black.
    img_bin = 255 - img_bin
    return Image.fromarray(img_bin)

def resize_image_for_ocr(image_pil: Image.Image, target_height: int) -> Image.Image:
    """
    Resizes a PIL Image to a target height while maintaining aspect ratio.
    Pads width if necessary to avoid distortion.
    """
    original_width, original_height = image_pil.size
    # Calculate new width based on target height and original aspect ratio
    new_width = int(original_width * (target_height / original_height))
    resized_img = image_pil.resize((new_width, target_height), Image.LANCZOS)
    return resized_img

def normalize_image_for_model(image_pil: Image.Image) -> torch.Tensor:
    """
    Converts a PIL Image to a PyTorch Tensor and normalizes pixel values.
    """
    # Convert to tensor (scales to 0-1 automatically)
    tensor_transform = transforms.ToTensor()
    img_tensor = tensor_transform(image_pil)
    # For grayscale images, mean and std are single values.
    # Adjust normalization values if your training data uses different ones.
    img_tensor = transforms.Normalize((0.5,), (0.5,))(img_tensor) # Normalize to [-1, 1]
    return img_tensor

def preprocess_user_image_for_ocr(uploaded_image_pil: Image.Image, target_height: int) -> torch.Tensor:
    """
    Combines all preprocessing steps for a single user-uploaded image
    to prepare it for the OCR model.
    """
    # Ensure it's grayscale
    img_gray = uploaded_image_pil.convert('L')

    # Binarize
    img_bin = binarize_image(img_gray)

    # Resize (maintain aspect ratio)
    img_resized = resize_image_for_ocr(img_bin, target_height)

    # Normalize and convert to tensor
    img_tensor = normalize_image_for_model(img_resized)

    # Add batch dimension: (C, H, W) -> (1, C, H, W)
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor

def pad_image_tensor(image_tensor: torch.Tensor, max_width: int) -> torch.Tensor:
    """
    Pads a single image tensor to a max_width with zeros.
    Input tensor shape: (C, H, W)
    Output tensor shape: (C, H, max_width)
    """
    C, H, W = image_tensor.shape
    if W > max_width:
        # If image is wider than max_width, you might want to crop or resize it.
        # For this example, we'll just return a warning or clip.
        # A more robust solution might split text lines or use a different resizing strategy.
        print(f"Warning: Image width {W} exceeds max_width {max_width}. Cropping.")
        return image_tensor[:, :, :max_width] # Simple cropping
    padding = max_width - W
    # Pad on the right (P_left, P_right, P_top, P_bottom)
    padded_tensor = f.pad(image_tensor, (0, padding), 'constant', 0)
>>>>>>> ee59e5b21399d8b323cff452a961ea2fd6c65308
    return padded_tensor