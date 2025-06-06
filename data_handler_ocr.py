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
from config import (
    VOCABULARY, BLANK_TOKEN, BLANK_TOKEN_SYMBOL, IMG_HEIGHT,
    TRAIN_IMAGES_DIR, TEST_IMAGES_DIR,
    TRAIN_SAMPLES_LIMIT, TEST_SAMPLES_LIMIT 
)
from utils_ocr import load_image_as_grayscale, binarize_image, resize_image_for_ocr, normalize_image_for_model

class CharIndexer:
    """Manages character-to-index and index-to-character mappings."""
    def __init__(self, vocabulary_string: str, blank_token_symbol: str):
        self.chars = sorted(list(set(vocabulary_string)))
        self.char_to_idx = {char: i for i, char in enumerate(self.chars)}
        self.idx_to_char = {i: char for i, char in enumerate(self.chars)}
        
        if blank_token_symbol not in self.char_to_idx:
            raise ValueError(f"Blank token symbol '{blank_token_symbol}' not found in provided vocabulary string: '{vocabulary_string}'")
            
        self.blank_token_idx = self.char_to_idx[blank_token_symbol]
        self.num_classes = len(self.chars)

        if self.blank_token_idx >= self.num_classes:
             raise ValueError(f"Blank token index ({self.blank_token_idx}) is out of range for num_classes ({self.num_classes}). This indicates a configuration mismatch.")

        print(f"CharIndexer initialized: num_classes={self.num_classes}, blank_token_idx={self.blank_token_idx}")
        print(f"Mapped blank symbol: '{self.idx_to_char[self.blank_token_idx]}'")

    def encode(self, text: str) -> list[int]:
        """Converts a text string to a list of integer indices."""
        encoded_list = []
        for char in text:
            if char in self.char_to_idx:
                encoded_list.append(self.char_to_idx[char])
            else:
                print(f"Warning: Character '{char}' not found in CharIndexer vocabulary. Mapping to blank token.")
                encoded_list.append(self.blank_token_idx)
        return encoded_list

    def decode(self, indices: list[int]) -> str:
        """Converts a list of integer indices back to a text string."""
        decoded_text = []
        for i, idx in enumerate(indices):
            if idx == self.blank_token_idx:
                continue # Skip blank tokens
            
            if i > 0 and indices[i-1] == idx:
                continue
            
            if idx in self.idx_to_char:
                decoded_text.append(self.idx_to_char[idx])
            else:
                print(f"Warning: Index {idx} not found in CharIndexer's idx_to_char mapping during decoding.")
                
        return "".join(decoded_text)

class OCRDataset(Dataset):
    """
    Custom PyTorch Dataset for the Handwritten Name Recognition task.
    Loads images and their corresponding text labels.
    """
    def __init__(self, dataframe: pd.DataFrame, char_indexer: CharIndexer, image_dir: str, transform=None):
        self.data = dataframe
        self.char_indexer = char_indexer
        self.image_dir = image_dir
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Lambda(lambda img: binarize_image(img)), 
                transforms.Lambda(lambda img: resize_image_for_ocr(img, IMG_HEIGHT)), # Resize image to fixed height
                transforms.ToTensor(), # Convert PIL Image to PyTorch Tensor (H, W) -> (1, H, W), scales to [0,1]
                transforms.Lambda(normalize_image_for_model) # Normalize pixel values to [-1, 1]
            ])
        else:
            self.transform = transform


    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        raw_filename_entry = self.data.loc[idx, 'FILENAME'] 
        ground_truth_text = self.data.loc[idx, 'IDENTITY']

        filename = raw_filename_entry.split(',')[0].strip()
        img_path = os.path.join(self.image_dir, filename)
        ground_truth_text = str(ground_truth_text)

        try:
            image = load_image_as_grayscale(img_path) # Returns PIL Image 'L'
        except FileNotFoundError:
            print(f"Error: Image file not found at {img_path}. Skipping this item.")
            raise

        if self.transform:
            image = self.transform(image)
        
        image_width = image.shape[2] # Assuming image is (C, H, W) after transform

        text_encoded = torch.tensor(self.char_indexer.encode(ground_truth_text), dtype=torch.long)
        text_length = len(text_encoded)

        return image, text_encoded, image_width, text_length

def ocr_collate_fn(batch: list) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function for the DataLoader to handle variable-width images
    and variable-length text sequences for CTC loss.
    """
    images, texts, image_widths, text_lengths = zip(*batch)

    max_batch_width = max(image_widths)
    padded_images = [F.pad(img, (0, max_batch_width - img.shape[2]), 'constant', 0) for img in images]
    images_batch = torch.stack(padded_images, 0)

    texts_batch = torch.cat(texts, 0)
    text_lengths_tensor = torch.tensor(list(text_lengths), dtype=torch.long)
    image_widths_tensor = torch.tensor(image_widths, dtype=torch.long)

    return images_batch, texts_batch, image_widths_tensor, text_lengths_tensor


def load_ocr_dataframes(train_csv_path: str, test_csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads training and testing dataframes.
    Assumes CSVs have 'FILENAME' and 'IDENTITY' columns.
    Applies dataset limits from config.py.
    """
    train_df = pd.read_csv(train_csv_path, encoding='ISO-8859-1')
    test_df = pd.read_csv(test_csv_path, encoding='ISO-8859-1')

    # Apply limits if they are set (not 0)
    if TRAIN_SAMPLES_LIMIT > 0:
        train_df = train_df.head(TRAIN_SAMPLES_LIMIT)
        print(f"Limited training data to {TRAIN_SAMPLES_LIMIT} samples.")
    if TEST_SAMPLES_LIMIT > 0:
        test_df = test_df.head(TEST_SAMPLES_LIMIT)
        print(f"Limited test data to {TEST_SAMPLES_LIMIT} samples.")

    return train_df, test_df

def create_ocr_dataloaders(train_df: pd.DataFrame, test_df: pd.DataFrame,
                           char_indexer: CharIndexer, batch_size: int) -> tuple[DataLoader, DataLoader]:
    """
    Creates PyTorch DataLoader objects for OCR training and testing datasets,
    using specific image directories for train/test.
    """
    train_dataset = OCRDataset(train_df, char_indexer, TRAIN_IMAGES_DIR)
    test_dataset = OCRDataset(test_df, char_indexer, TEST_IMAGES_DIR)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, collate_fn=ocr_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=0, collate_fn=ocr_collate_fn)
    return train_loader, test_loader
