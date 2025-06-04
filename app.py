# -*- coding: utf-8 -*-
# app.py

import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F # Added F for log_softmax in inference
import torchvision.transforms as transforms
import traceback # For detailed error logging

# Import all necessary configuration values from config.py
from config import (
    IMG_HEIGHT, NUM_CLASSES, BLANK_TOKEN, VOCABULARY, BLANK_TOKEN_SYMBOL,
    TRAIN_CSV_PATH, TEST_CSV_PATH, TRAIN_IMAGES_DIR, TEST_IMAGES_DIR,
    MODEL_SAVE_PATH, BATCH_SIZE, NUM_EPOCHS
)

# Import classes and functions from data_handler_ocr.py and model_ocr.py
from data_handler_ocr import CharIndexer, OCRDataset, ocr_collate_fn, load_ocr_dataframes, create_ocr_dataloaders
from model_ocr import CRNN, train_ocr_model, save_ocr_model, load_ocr_model, ctc_greedy_decode
from utils_ocr import preprocess_user_image_for_ocr, binarize_image, resize_image_for_ocr, normalize_image_for_model # Ensure these are imported if needed


# --- Global Variables ---
# These will hold the model and char_indexer instance after training or loading
trained_ocr_model = None
char_indexer = None
training_history = None
# Determine the device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Streamlit App Setup ---
st.set_page_config(layout="wide", page_title="Handwritten Name OCR App") # Changed to wide layout for better display

st.title("📝 Handwritten Name Recognition (OCR) App") # Updated title for consistency
st.markdown("""
    This application uses a Convolutional Recurrent Neural Network (CRNN) to perform
    Optical Character Recognition (OCR) on handwritten names. You can upload an image
    of a handwritten name for prediction or train a new model using the provided dataset.

    **Note:** Training a robust OCR model can be time-consuming.
""")

# --- Initialize CharIndexer ---
# CRITICAL FIX: Initialize CharIndexer with VOCABULARY and BLANK_TOKEN_SYMBOL
# This resolves the ValueError: "Blank token symbol '95' not found..."
char_indexer = CharIndexer(vocabulary_string=VOCABULARY, blank_token_symbol=BLANK_TOKEN_SYMBOL)

# --- Model Loading / Initialization ---
@st.cache_resource # Cache the model to prevent reloading on every rerun
def get_and_load_ocr_model_cached(num_classes, model_path):
    """
    Initializes the OCR model and attempts to load a pre-trained model.
    If no pre-trained model exists, a new model instance is returned.
    """
    model_instance = CRNN(num_classes=num_classes, cnn_output_channels=512, rnn_hidden_size=256, rnn_num_layers=2)
    
    if os.path.exists(model_path):
        st.sidebar.info("Loading pre-trained OCR model...")
        try:
            # Load model to CPU first, then move to device
            model_instance.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            st.sidebar.success("OCR model loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading model: {e}. A new model will be initialized.")
            # If loading fails, re-initialize an untrained model
            model_instance = CRNN(num_classes=num_classes, cnn_output_channels=512, rnn_hidden_size=256, rnn_num_layers=2)
    else:
        st.sidebar.warning("No pre-trained OCR model found. Please train a model using the sidebar option.")
    
    return model_instance

# Get the model instance
ocr_model = get_and_load_ocr_model_cached(char_indexer.num_classes, MODEL_SAVE_PATH)
# Determine the device (GPU if available, else CPU)
ocr_model.to(device)
ocr_model.eval() # Set model to evaluation mode for inference by default

# --- Sidebar for Model Training ---
st.sidebar.header("Model Training (Optional)")
st.sidebar.markdown("If you want to train a new model or no model is found:")

# Add input fields for dataset limits in the sidebar
st.sidebar.subheader("Dataset Limits (Optional)")
train_limit = st.sidebar.number_input("Limit Training Samples (0 for all)", min_value=0, value=0, step=100)
test_limit = st.sidebar.number_input("Limit Test Samples (0 for all)", min_value=0, value=0, step=100)

# Convert 0 to None for the function
train_limit_val = None if train_limit == 0 else train_limit
test_limit_val = None if test_limit == 0 else test_limit

# Add input fields for Early Stopping parameters
st.sidebar.subheader("Early Stopping (Optional)")
early_stopping_patience = st.sidebar.number_input(
    "Early Stopping Patience (epochs with no improvement, 0 to disable)",
    min_value=0, value=5, step=1
)
early_stopping_min_delta = st.sidebar.number_input(
    "Early Stopping Min Delta (min change to be improvement)",
    min_value=0.0, value=0.001, step=0.0001, format="%.4f"
)

# Convert 0 patience to None to disable early stopping
es_patience_val = None if early_stopping_patience == 0 else early_stopping_patience


# Initialize Streamlit widgets outside the button block
training_progress_bar = st.sidebar.empty() # Placeholder for progress bar in sidebar
status_text = st.sidebar.empty()            # Placeholder for status messages in sidebar

if st.sidebar.button("📊 Train New OCR Model"): # Keep button in sidebar as per user's last provided code
    # Clear previous messages/widgets if button is clicked again
    training_progress_bar.progress(0) # Reset progress bar
    training_progress_bar.empty()
    status_text.empty() # Clear status text

    # Check for existence of CSVs and image directories
    if not os.path.exists(TRAIN_CSV_PATH) or not os.path.isdir(TRAIN_IMAGES_DIR):
        status_text.error(f"Training CSV '{TRAIN_CSV_PATH}' or Images directory '{TRAIN_IMAGES_DIR}' not found!")
    elif not os.path.exists(TEST_CSV_PATH) or not os.path.isdir(TEST_IMAGES_DIR):
        status_text.warning(f"Test CSV '{TEST_CSV_PATH}' or Images directory '{TEST_IMAGES_DIR}' not found. "
                   "Evaluation might be affected or skipped. Please ensure all data paths are correct.")
    else:
        status_text.info(f"Training a new CRNN model for {NUM_EPOCHS} epochs. This will take significant time...")
        
        # Define the progress bar instance here for the callback
        training_progress_bar_instance = training_progress_bar.progress(0.0, text="Training in progress. Please wait.")

        def update_progress_callback_sidebar(value, text):
            """Callback function to update Streamlit progress bar in sidebar."""
            training_progress_bar_instance.progress(int(value * 100))
            status_text.text(text) # Update status text in sidebar

        try:
            train_df, test_df = load_ocr_dataframes(TRAIN_CSV_PATH, TEST_CSV_PATH)
            status_text.success("Training and Test DataFrames loaded successfully.")

        
            char_indexer = CharIndexer(vocabulary_string=VOCABULARY, blank_token_symbol=BLANK_TOKEN_SYMBOL)
            status_text.success(f"CharIndexer initialized with {char_indexer.num_classes} classes.")

            # Pass the limits to create_ocr_dataloaders
            train_loader, test_loader = create_ocr_dataloaders(
                train_df, test_df, char_indexer, BATCH_SIZE,
                train_limit=train_limit_val, test_limit=test_limit_val # Pass the limits here
            )
            status_text.success("DataLoaders created successfully.")

            
            ocr_model_for_training = CRNN(num_classes=NUM_CLASSES) # Create a new instance for training
            ocr_model_for_training.to(device)
            status_text.info(f"CRNN model initialized and moved to {device}.")

            status_text.write("Training in progress... This may take a while.")
            trained_ocr_model, training_history = train_ocr_model(
                model=ocr_model_for_training, # Pass the new instance
                train_loader=train_loader,
                test_loader=test_loader,
                char_indexer=char_indexer, # Pass char_indexer for CER calculation
                epochs=NUM_EPOCHS,
                device=device,
                progress_callback=update_progress_callback_sidebar, # Pass the sidebar callback
                early_stopping_patience=es_patience_val, # Pass early stopping patience
                early_stopping_min_delta=early_stopping_min_delta # Pass early stopping min delta
            )
            status_text.success("OCR model training finished!")
            update_progress_callback_sidebar(1.0, "Training complete!")

            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            save_ocr_model(trained_ocr_model, MODEL_SAVE_PATH)
            status_text.success(f"Trained model saved to `{MODEL_SAVE_PATH}`")

            # Display training history chart in the main section, not sidebar
            if training_history:
                st.subheader("Training History Plots")
                history_df = pd.DataFrame({
                    'Epoch': range(1, len(training_history['train_loss']) + 1),
                    'Train Loss': training_history['train_loss'],
                    'Test Loss': training_history['test_loss'],
                    'Test CER (%)': [cer * 100 for cer in training_history['test_cer']],
                    'Test Exact Match Accuracy (%)': [acc * 100 for acc in training_history['test_exact_match_accuracy']]
                })

                st.markdown("**Loss over Epochs**")
                st.line_chart(history_df.set_index('Epoch')[['Train Loss', 'Test Loss']])
                st.caption("Lower loss indicates better model performance.")

                st.markdown("**Character Error Rate (CER) over Epochs**")
                st.line_chart(history_df.set_index('Epoch')[['Test CER (%)']])
                st.caption("Lower CER indicates fewer character errors (0% is perfect).")

                st.markdown("**Exact Match Accuracy over Epochs**")
                st.line_chart(history_df.set_index('Epoch')[['Test Exact Match Accuracy (%)']])
                st.caption("Higher exact match accuracy indicates more perfectly recognized names.")

                st.markdown("**Performance Metrics over Epochs (CER vs. Exact Match Accuracy)**")
                st.line_chart(history_df.set_index('Epoch')[['Test CER (%)', 'Test Exact Match Accuracy (%)']])
                st.caption("CER should decrease, Accuracy should increase.")

        except Exception as e:
            status_text.error(f"An error occurred during training: {e}")
            status_text.exception(e) # Display full traceback in Streamlit
            update_progress_callback_sidebar(0.0, "Training failed!")

# --- Main Content: Name Prediction ---
st.header("Predict Your Handwritten Name")
st.markdown("Upload a clear image of a single handwritten name or word.")

uploaded_file = st.file_uploader("🖼️ Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        # Open the uploaded image
        image_pil = Image.open(uploaded_file).convert('L') # Ensure grayscale
        # Use use_container_width for deprecation warning fix
        st.image(image_pil, caption="Uploaded Image", use_container_width=True) 
        st.write("---")
        st.write("Processing and Recognizing...")

        # Preprocess the image for the model using utils_ocr function
        processed_image_tensor = preprocess_user_image_for_ocr(image_pil, IMG_HEIGHT).to(device)
        
        ocr_model.eval() # Ensure model is in evaluation mode
        with torch.no_grad(): # Disable gradient calculation for inference
            output = ocr_model(processed_image_tensor) # (sequence_length, batch_size, num_classes)
        
        # Decode the prediction using the global char_indexer
        predicted_texts = ctc_greedy_decode(output, char_indexer)
        predicted_text = predicted_texts[0] # Get the first (and only) prediction

        st.success(f"Recognized Text: **{predicted_text}**")

    except Exception as e:
        st.error(f"Error processing image or recognizing text: {e}")
        st.info("💡 **Tips for best results:**\n"
                "- Ensure the handwritten text is clear and on a clean background.\n"
                "- Only include one name/word per image.\n"
                "- The model is trained on specific characters. Unusual symbols might not be recognized.")
        st.exception(e) # Display full traceback in Streamlit

st.markdown("""
    ---
    *Built using Streamlit, PyTorch, OpenCV, and EditDistance ©2025 by MFT*
    """)