# config.py

import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, 'images')
TEST_IMAGES_DIR = os.path.join(DATA_DIR, 'images')

TRAIN_CSV_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_CSV_PATH = os.path.join(DATA_DIR, 'test.csv')

MODEL_SAVE_PATH = os.path.join(MODELS_DIR, 'handwritten_name_ocr_model.pth')

# --- Character Set and OCR Configuration ---
CHARS = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
BLANK_TOKEN_SYMBOL = 'Þ' 
VOCABULARY = CHARS + BLANK_TOKEN_SYMBOL
NUM_CLASSES = len(VOCABULARY)
BLANK_TOKEN = VOCABULARY.find(BLANK_TOKEN_SYMBOL)

# --- Sanity Checks ---
if BLANK_TOKEN == -1:
    raise ValueError(f"Error: BLANK_TOKEN_SYMBOL '{BLANK_TOKEN_SYMBOL}' not found in VOCABULARY. Check config.py definitions.")
if BLANK_TOKEN >= NUM_CLASSES:
     raise ValueError(f"Error: BLANK_TOKEN index ({BLANK_TOKEN}) must be less than NUM_CLASSES ({NUM_CLASSES}).")

print(f"Config Loaded: NUM_CLASSES={NUM_CLASSES}, BLANK_TOKEN_INDEX={BLANK_TOKEN}")
print(f"Vocabulary Length: {len(VOCABULARY)}")
print(f"Blank Symbol: '{BLANK_TOKEN_SYMBOL}' at index {BLANK_TOKEN}")


# --- Image Preprocessing Parameters ---
IMG_HEIGHT = 32 # Target height for all input images to the model
MAX_IMG_WIDTH = 1024 # Adjust this value based on your typical image widths and available RAM

# --- Training Parameters ---
BATCH_SIZE = 10

# NEW: Dataset Limits
TRAIN_SAMPLES_LIMIT = 1000 
TEST_SAMPLES_LIMIT = 1000 

NUM_EPOCHS = 5
LEARNING_RATE = 0.001
