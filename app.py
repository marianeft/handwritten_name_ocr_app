# app.py

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F # Added F for log_softmax in inference
import torchvision.transforms as transforms
import os
import traceback # For detailed error logging

# Import custom modules
from config import CHARS, BLANK_TOKEN, IMG_HEIGHT, TRAIN_CSV_PATH, TEST_CSV_PATH, \
                    TRAIN_IMAGES_DIR, TEST_IMAGES_DIR, MODEL_SAVE_PATH, NUM_CLASSES, NUM_EPOCHS, BATCH_SIZE
from data_handler_ocr import CharIndexer, OCRDataset
from model_ocr import CRNN, train_ocr_model, save_ocr_model, load_ocr_model, ctc_greedy_decode
from utils_ocr import preprocess_user_image_for_ocr

# --- Streamlit App Setup ---
st.set_page_config(page_title="Handwritten Name Recognizer", layout="centered")

st.title("📝 Handwritten Name Recognition (OCR)")
st.markdown("""
    This application uses a Convolutional Recurrent Neural Network (CRNN) to perform
    Optical Character Recognition (OCR) on handwritten names. You can upload an image
    of a handwritten name for prediction or train a new model using the provided dataset.

    **Note:** Training a robust OCR model can be time-consuming.
""")

# --- Initialize CharIndexer ---
# The CHARS variable should contain all possible characters your model can recognize.
# Make sure it's comprehensive based on your dataset.
char_indexer = CharIndexer(CHARS, BLANK_TOKEN)
# For robustness, it's best to always use char_indexer.num_classes
# If NUM_CLASSES from config is used to initialize CRNN, ensure it matches char_indexer.num_classes

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ocr_model.to(device)
ocr_model.eval() # Set model to evaluation mode for inference by default

# --- Sidebar for Model Training ---
st.sidebar.header("Model Training (Optional)")
st.sidebar.markdown("If you want to train a new model or no model is found:")

# Initialize Streamlit widgets outside the button block
training_progress_bar = st.sidebar.empty() # Placeholder for progress bar
status_text = st.sidebar.empty()            # Placeholder for status messages

if st.sidebar.button("📊 Train New OCR Model"):
    # Clear previous messages/widgets if button is clicked again
    training_progress_bar.empty()
    status_text.empty()

    # Check for existence of CSVs and image directories
    if not os.path.exists(TRAIN_CSV_PATH) or not os.path.exists(TEST_CSV_PATH) or \
       not os.path.isdir(TRAIN_IMAGES_DIR) or not os.path.isdir(TEST_IMAGES_DIR):
        status_text.error(f"""Dataset files or image directories not found.
        Please ensure '{TRAIN_CSV_PATH}', '{TEST_CSV_PATH}', and directories '{TRAIN_IMAGES_DIR}'
        and '{TEST_IMAGES_DIR}' exist. Refer to your project structure.""")
    else:
        status_text.write(f"Training a new CRNN model for {NUM_EPOCHS} epochs. This will take significant time...")
        
        training_progress_bar_instance = training_progress_bar.progress(0.0, text="Training in progress. Please wait.")

        try:
            train_df = pd.read_csv(TRAIN_CSV_PATH, delimiter=';', names=['FILENAME', 'IDENTITY'], header=None)
            test_df = pd.read_csv(TEST_CSV_PATH, delimiter=';', names=['FILENAME', 'IDENTITY'], header=None)

            # Define standard image transforms for consistency
            train_transform = transforms.Compose([
                transforms.Resize((IMG_HEIGHT, 100)), # Resize to fixed height, width will be 100 (adjust as needed for variable width)
                transforms.ToTensor(), # Converts PIL Image to PyTorch Tensor (H, W) -> (C, H, W), normalizes to [0,1]
            ])
            test_transform = transforms.Compose([
                transforms.Resize((IMG_HEIGHT, 100)), # Same transformation as train
                transforms.ToTensor(),
            ])

            # Create dataset instances
            train_dataset = OCRDataset(dataframe=train_df, char_indexer=char_indexer, image_dir=TRAIN_IMAGES_DIR, transform=train_transform)
            test_dataset = OCRDataset(dataframe=test_df, char_indexer=char_indexer, image_dir=TEST_IMAGES_DIR, transform=test_transform)

            # Create DataLoader instances
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) # num_workers=0 for Windows
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

            # Train the model, passing the progress callback
            trained_ocr_model, training_history = train_ocr_model(
                ocr_model, # Pass the initialized model instance
                train_loader,
                test_loader,
                char_indexer, # Pass char_indexer for CER calculation
                epochs=NUM_EPOCHS,
                device=device,
                progress_callback=training_progress_bar_instance.progress # Pass the instance's progress method
            )

            # Ensure the directory for saving the model exists
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            save_ocr_model(trained_ocr_model, MODEL_SAVE_PATH)
            status_text.success(f"Model training complete and saved to `{MODEL_SAVE_PATH}`!")

            # Display training history chart
            st.sidebar.subheader("Training History Plots")

            history_df = pd.DataFrame({
                'Epoch': range(1, len(training_history['train_loss']) + 1),
                'Train Loss': training_history['train_loss'],
                'Test Loss': training_history['test_loss'],
                'Test CER (%)': [cer * 100 for cer in training_history['test_cer']], # Convert CER to percentage for display
                'Test Exact Match Accuracy (%)': [acc * 100 for acc in training_history['test_exact_match_accuracy']] # Convert to percentage
            })

            # Plot 1: Training and Test Loss
            st.sidebar.markdown("**Loss over Epochs**")
            st.sidebar.line_chart(
                history_df.set_index('Epoch')[['Train Loss', 'Test Loss']]
            )
            st.sidebar.caption("Lower loss indicates better model performance.")

            # Plot 2: Character Error Rate (CER)
            st.sidebar.markdown("**Character Error Rate (CER) over Epochs**")
            st.sidebar.line_chart(
                history_df.set_index('Epoch')[['Test CER (%)']]
            )
            st.sidebar.caption("Lower CER indicates fewer character errors (0% is perfect).")

            # Plot 3: Exact Match Accuracy
            st.sidebar.markdown("**Exact Match Accuracy over Epochs**")
            st.sidebar.line_chart(
                history_df.set_index('Epoch')[['Test Exact Match Accuracy (%)']]
            )
            st.sidebar.caption("Higher exact match accuracy indicates more perfectly recognized names.")

            # Update the global model instance to the newly trained one for immediate inference
            ocr_model = trained_ocr_model
            ocr_model.eval()

        except Exception as e:
            status_text.error(f"An error occurred during training: {e}")
            st.sidebar.text(traceback.format_exc()) # Show full traceback for debugging

# --- Main Content: Name Prediction ---
st.header("Predict Your Handwritten Name")
st.markdown("Upload a clear image of a single handwritten name or word.")

uploaded_file = st.file_uploader("🖼️ Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        # Open the uploaded image
        image_pil = Image.open(uploaded_file).convert('L') # Ensure grayscale
        st.image(image_pil, caption="Uploaded Image", use_column_width=True)
        st.write("---")
        st.write("Processing and Recognizing...")

        # Preprocess the image for the model using utils_ocr function
        processed_image_tensor = preprocess_user_image_for_ocr(image_pil, IMG_HEIGHT).to(device)

        # Make prediction
        ocr_model.eval() # Ensure model is in evaluation mode
        with torch.no_grad(): # Disable gradient calculation for inference
            output = ocr_model(processed_image_tensor) # (sequence_length, batch_size, num_classes)
            
            # ctc_greedy_decode expects (sequence_length, batch_size, num_classes)
            # It returns a list of strings, so get the first element for single image inference.
            predicted_texts = ctc_greedy_decode(output, char_indexer)
            predicted_text = predicted_texts[0] # Get the first (and only) prediction

        st.success(f"Recognized Text: **{predicted_text}**")

    except Exception as e:
        st.error(f"Error processing image or recognizing text: {e}")
        st.info("💡 **Tips for best results:**\n"
                "- Ensure the handwritten text is clear and on a clean background.\n"
                "- Only include one name/word per image.\n"
                "- The model is trained on specific characters. Unusual symbols might not be recognized.")
        st.text(traceback.format_exc())

st.markdown("""
    ---
    *Built using Streamlit, PyTorch, OpenCV, and EditDistance ©2025 by MFT*
    """)