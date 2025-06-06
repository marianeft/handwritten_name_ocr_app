# -*- coding: utf-8 -*-
# app.py

import os
# Disable Streamlit file watcher to prevent conflicts with PyTorch
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import traceback

# Import all necessary configuration values from config.py
from config import (
    IMG_HEIGHT, NUM_CLASSES, BLANK_TOKEN, VOCABULARY, BLANK_TOKEN_SYMBOL,
    TRAIN_CSV_PATH, TEST_CSV_PATH, TRAIN_IMAGES_DIR, TEST_IMAGES_DIR,
    MODEL_SAVE_PATH, BATCH_SIZE, NUM_EPOCHS
)

# Import classes and functions from data_handler_ocr.py and model_ocr.py
from data_handler_ocr import CharIndexer, OCRDataset, ocr_collate_fn, load_ocr_dataframes, create_ocr_dataloaders
from model_ocr import CRNN, train_ocr_model, save_ocr_model, load_ocr_model, ctc_greedy_decode
from utils_ocr import preprocess_user_image_for_ocr, binarize_image, resize_image_for_ocr, normalize_image_for_model


# --- Global Variables ---
ocr_model = None
char_indexer = None 
training_history = None 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Streamlit App Setup ---
st.set_page_config(layout="wide", page_title="Handwritten Name OCR App",)


st.title("📝 Handwritten Name Recognition (OCR) App")
st.markdown("""
    This application uses a Convolutional Recurrent Neural Network (CRNN) to perform
    Optical Character Recognition (OCR) on handwritten names. You can upload an image
    of a handwritten name for prediction or train a new model using the provided dataset.
     
    **Note:** Training a robust OCR model can be time-consuming.
""")

# --- Initialize CharIndexer ---
# This initializes char_indexer once when the script starts
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

# Get the model instance and assign it to the global 'ocr_model'
ocr_model = get_and_load_ocr_model_cached(char_indexer.num_classes, MODEL_SAVE_PATH)
# Ensure the model is on the correct device for inference
ocr_model.to(device)
ocr_model.eval() # Set model to evaluation mode for inference by default


# --- Sidebar for Model Training ---
st.sidebar.header("Train OCR Model")
st.sidebar.write("Click the button below to start training the OCR model.")

# Progress bar and label for training in the sidebar
progress_bar_sidebar = st.sidebar.progress(0)
progress_label_sidebar = st.sidebar.empty()

def update_progress_callback_sidebar(value, text):
    progress_bar_sidebar.progress(int(value * 100))
    progress_label_sidebar.text(text)

if st.sidebar.button("📊 Start Training"):
    progress_bar_sidebar.progress(0)
    progress_label_sidebar.empty()
    st.empty() # Clear any previous status messages in the main section

    if not os.path.exists(TRAIN_CSV_PATH) or not os.path.isdir(TRAIN_IMAGES_DIR):
        st.sidebar.error(f"Training CSV '{TRAIN_CSV_PATH}' or Images directory '{TRAIN_IMAGES_DIR}' not found!")
    elif not os.path.exists(TEST_CSV_PATH) or not os.path.isdir(TEST_IMAGES_DIR):
        st.sidebar.warning(f"Test CSV '{TEST_CSV_PATH}' or Images directory '{TEST_IMAGES_DIR}' not found. "
                   "Evaluation might be affected or skipped. Please ensure all data paths are correct.")
    else:
        st.sidebar.info(f"Training a new CRNN model for {NUM_EPOCHS} epochs. This will take significant time...")
        
        try:
            train_df, test_df = load_ocr_dataframes(TRAIN_CSV_PATH, TEST_CSV_PATH)
            st.sidebar.success("Training and Test DataFrames loaded successfully.")

            st.sidebar.success(f"CharIndexer initialized with {char_indexer.num_classes} classes.")

            train_loader, test_loader = create_ocr_dataloaders(train_df, test_df, char_indexer, BATCH_SIZE)
            st.sidebar.success("DataLoaders created successfully.")
            
            ocr_model.train() # Ensure it's in training mode before passing to train_ocr_model

            st.sidebar.write("Training in progress... This may take a while.")
            ocr_model, training_history = train_ocr_model(
                model=ocr_model,
                train_loader=train_loader,
                test_loader=test_loader,
                char_indexer=char_indexer,
                epochs=NUM_EPOCHS,
                device=device,
                progress_callback=update_progress_callback_sidebar # Use the sidebar callback
            )
            st.sidebar.success("OCR model training finished!")
            update_progress_callback_sidebar(1.0, "Training complete!")

            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            save_ocr_model(ocr_model, MODEL_SAVE_PATH) # Save the now trained ocr_model
            st.sidebar.success(f"Trained model saved to `{MODEL_SAVE_PATH}`")

            # Training history charts will now be displayed in the main content area (col2)
            # This block will be executed after training is complete.

        except Exception as e:
            st.sidebar.error(f"An error occurred during training: {e}")
            st.exception(e) # Display full traceback in main section
            update_progress_callback_sidebar(0.0, "Training failed!")

# --- Sidebar for Model Loading ---
st.sidebar.header("Load Pre-trained Model")
st.sidebar.write("If you have a saved model, you can load it here instead of training.")

if st.sidebar.button("💾 Load Model"):
    if os.path.exists(MODEL_SAVE_PATH):
        try:
            loaded_model = CRNN(num_classes=char_indexer.num_classes)
            load_ocr_model(loaded_model, MODEL_SAVE_PATH)
            loaded_model.to(device)
            
            st.sidebar.success(f"Model loaded successfully from `{MODEL_SAVE_PATH}`")
        except Exception as e:
            st.sidebar.error(f"Error loading model: {e}")
            st.exception(e) # Display full traceback in main section
    else:
        st.sidebar.warning(f"No model found at `{MODEL_SAVE_PATH}`. Please train a model first or check the path.")

# --- Main Content: Prediction Section and Training History  ---

# Display training history chart
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
    st.write("---") # Separator after charts


# Predict on a New Image

if ocr_model is None:
    st.warning("Please train or load a model before attempting prediction.")
else:
    uploaded_file = st.file_uploader("🖼️ Choose an image...", type=["png", "jpg", "jpeg", "jfif"])

    if uploaded_file is not None:
        try:
            image_pil = Image.open(uploaded_file).convert('L')
            st.image(image_pil, caption="Uploaded Image", use_container_width=True) 
            st.write("---")
            st.write("Processing and Recognizing...")

            processed_image_tensor = preprocess_user_image_for_ocr(image_pil, IMG_HEIGHT).to(device)
            
            ocr_model.eval()
            with torch.no_grad():
                output = ocr_model(processed_image_tensor)
            
            predicted_texts = ctc_greedy_decode(output, char_indexer)
            predicted_text = predicted_texts[0]

            st.success(f"Recognized Text: **{predicted_text}**")

        except Exception as e:
            st.error(f"Error processing image or recognizing text: {e}")
            st.info("💡 **Tips for best results:**\n"
                    "- Ensure the handwritten text is clear and on a clean background.\n"
                    "- Only include one name/word per image.\n"
                    "- The model is trained on specific characters. Unusual symbols might not be recognized.")
            st.exception(e)

st.markdown("""
    ---
    *Built using Streamlit, PyTorch, OpenCV, and EditDistance ©2025 by MFT*
    """)
