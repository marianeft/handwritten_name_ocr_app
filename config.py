# config.py

import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, 'images', 'train')
TEST_IMAGES_DIR = os.path.join(DATA_DIR, 'images', 'test')

TRAIN_CSV_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_CSV_PATH = os.path.join(DATA_DIR, 'test.csv')

MODEL_SAVE_PATH = os.path.join(MODELS_DIR, 'handwritten_name_ocr_model.pth')

# --- Character Set and OCR Configuration ---
# This character set MUST cover all characters present in your dataset.
# Add any special characters if needed.
# The order here is crucial as it defines the indices for your characters.
CHARS = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"

# Define the character for the blank token. It MUST NOT be in CHARS.
BLANK_TOKEN_SYMBOL = 'Þ'

# Construct the full vocabulary string. It's conventional to put the blank token last.
# This VOCABULARY string is what you pass to CharIndexer.
VOCABULARY = CHARS + BLANK_TOKEN_SYMBOL

# NUM_CLASSES is the total number of unique symbols in the vocabulary, including the blank.
NUM_CLASSES = len(VOCABULARY)

# BLANK_TOKEN is the actual index of the blank symbol within the VOCABULARY.
# Since we appended it last, its index will be len(CHARS).
BLANK_TOKEN = VOCABULARY.find(BLANK_TOKEN_SYMBOL)

# --- Sanity Checks (Highly Recommended) ---
if BLANK_TOKEN == -1:
    raise ValueError(f"Error: BLANK_TOKEN_SYMBOL '{BLANK_TOKEN_SYMBOL}' not found in VOCABULARY. Check config.py definitions.")
if BLANK_TOKEN >= NUM_CLASSES:
     raise ValueError(f"Error: BLANK_TOKEN index ({BLANK_TOKEN}) must be less than NUM_CLASSES ({NUM_CLASSES}).")

print(f"Config Loaded: NUM_CLASSES={NUM_CLASSES}, BLANK_TOKEN_INDEX={BLANK_TOKEN}")
print(f"Vocabulary Length: {len(VOCABULARY)}")
print(f"Blank Symbol: '{BLANK_TOKEN_SYMBOL}' at index {BLANK_TOKEN}")


# --- Image Preprocessing Parameters ---
IMG_HEIGHT = 32

# --- Training Parameters ---
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 3