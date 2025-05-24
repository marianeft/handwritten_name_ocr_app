#data_handler_ocr.py

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import torch.nn.functional as F

# Import utility functions and config
from config import CHARS, BLANK_TOKEN, IMG_HEIGHT, TRAIN_IMAGES_DIR, TEST_IMAGES_DIR
from utils_ocr import load_image_as_grayscale, binarize_image, resize_image_for_ocr, normalize_image_for_model

class CharIndexer:
    """Manages character-to-index and index-to-character mappings."""
    def __init__(self, chars: str, blank_token: str):
        self.char_to_idx = {char: i for i, char in enumerate(chars)}
        self.idx_to_char = {i: char for i, char in enumerate(chars)}
        self.blank_token_idx = len(chars) # Index for the blank token
        self.idx_to_char[self.blank_token_idx] = blank_token # Add blank token to idx_to_char
        self.num_classes = len(chars) + 1 # Total classes including blank

    def encode(self, text: str) -> list[int]:
        """Converts a text string to a list of integer indices."""
        return [self.char_to_idx[char] for char in text]

    def decode(self, indices: list[int]) -> str:
        """Converts a list of integer indices back to a text string."""
        # CTC decoding often produces repeated characters and blank tokens.
        # This simple decoder removes blanks and duplicates.
        decoded_text = []
        for i, idx in enumerate(indices):
            if idx == self.blank_token_idx:
                continue
            # Remove consecutive duplicates
            if i > 0 and indices[i-1] == idx:
                continue
            decoded_text.append(self.idx_to_char[idx])
        return "".join(decoded_text)

class OCRDataset(Dataset):
    """
    Custom PyTorch Dataset for the Handwritten Name Recognition task.
    Loads images and their corresponding text labels.
    """
    def __init__(self, dataframe: pd.DataFrame, char_indexer: CharIndexer, image_dir: str, transform=None):
        """
        Initializes the OCR Dataset.
        Args:
            dataframe (pd.DataFrame): A DataFrame containing 'image_path' and 'label' columns.
            char_indexer (CharIndexer): An instance of CharIndexer for character encoding.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.data = dataframe
        self.char_indexer = char_indexer
        self.image_dir = image_dir
        self.transform = transform


    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        raw_filename_entry = self.data.iloc[idx]['FILENAME']
        ground_truth_text = self.data.iloc[idx]['IDENTITY']

        filename = raw_filename_entry.split(',')[0].strip() # .strip() removes any whitespace
        # Construct the full image path
        img_path = os.path.join(self.image_dir, filename)
        # Ensure ground_truth_text is a string
        ground_truth_text = str(ground_truth_text)

        # Load and transform image
        try:
            image = Image.open(img_path).convert('L') # Convert to grayscale
        except FileNotFoundError:
            print(f"Error: Image file not found at {img_path}. Skipping this item.")
            raise # Re-raise to let the main traceback be seen.

        if self.transform:
            image = self.transform(image)
        
        image_width = image.size(2) # Assuming image is a tensor (C, H, W) after transform

        text_encoded = torch.tensor(self.char_indexer.encode(ground_truth_text), dtype=torch.long)
        text_length = len(text_encoded)

        return image, text_encoded, image_width, text_length

def ocr_collate_fn(batch: list) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function for the DataLoader to handle variable-width images
    and variable-length text sequences for CTC loss.
    """
    images, texts, image_widths, text_lengths = zip(*batch)

    # Pad images to the maximum width in the current batch
    max_batch_width = max(image_widths)
    padded_images = [F.pad(img, (0, max_batch_width - img.shape[2]), 'constant', 0) for img in images]
    images_batch = torch.stack(padded_images, 0) # Stack to (N, C, H, max_W)

    # Concatenate all text sequences and get their lengths
    texts_batch = torch.cat(texts, 0)
    text_lengths_tensor = torch.tensor(text_lengths, dtype=torch.long)
    image_widths_tensor = torch.tensor(image_widths, dtype=torch.long) # Actual widths

    return images_batch, texts_batch, image_widths_tensor, text_lengths_tensor


def load_ocr_dataframes(train_csv_path: str, test_csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads training and testing dataframes.
    Assumes CSVs have 'filename' and 'name' columns.
    """
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)
    return train_df, test_df

def create_ocr_dataloaders(train_df: pd.DataFrame, test_df: pd.DataFrame,
                           char_indexer: CharIndexer, batch_size: int) -> tuple[DataLoader, DataLoader]:
    """
    Creates PyTorch DataLoader objects for OCR training and testing datasets,
    using specific image directories for train/test.
    """
    train_dataset = OCRDataset(train_df, TRAIN_IMAGES_DIR, char_indexer)
    test_dataset = OCRDataset(test_df, TEST_IMAGES_DIR, char_indexer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, collate_fn=ocr_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=0, collate_fn=ocr_collate_fn)
    return train_loader, test_loader