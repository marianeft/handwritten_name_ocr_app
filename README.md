# Handwritten Name Recognition (OCR) App ✍🏻

*An end-to-end Streamlit application for training and predicting handwritten names using a CRNN model.*

---

## Table of Contents 

* [Overview](#overview)
* [Quickstart](#quickstart)
* [Features](#features)
* [Project Structure](#project-structure)
* [Project Index](#project-index)
* [Roadmap](#roadmap)
* [Contribution](#contribution)
* [License](#license)
* [Acknowledgements](#acknowledgements)

---

## 🕹️ Overview

This project implements a Handwritten Name Recognition (OCR) system using a Convolutional Recurrent Neural Network (CRNN) architecture built with PyTorch. The application is presented as an interactive web interface using Streamlit, allowing users to:

1.  **Train** a new OCR model from a local dataset.
2.  **Load** a pre-trained model.
3.  **Predict** text from uploaded handwritten image files.
4.  **Upload** the local dataset to the Hugging Face Hub for sharing and versioning.

The CRNN model combines a CNN backbone for feature extraction from images and a Bidirectional LSTM layer for sequence modeling, followed by a linear layer for character classification using CTC (Connectionist Temporal Classification) Loss.

---

## 🚩 Quickstart

Follow these steps to get the application up and running on your local machine.

### Prerequisites

* Python 3.8+
* `pip` (Python package installer)

#### 1. Clone the Repository (or set up your project folder)

Ensure your project structure matches the expected layout (e.g., `app.py`, `config.py`, `data/`, `models/` etc.).

#### 2. Create and Activate a Virtual Environment
It's highly recommended to use a virtual environment to manage dependencies.

``` bash
# Navigate to your project root directory
cd path/to/your/handwritten_name_ocr_app

# Create a virtual environment named 'venvy'
python -m venv venvy

# Activate the virtual environment
# On Windows (Command Prompt):
.\venvy\Scripts\activate.bat

# On Windows (PowerShell):
.\venvy\Scripts\Activate.ps1

# On macOS/Linux:
source venvy/bin/activate
```

#### 3. Install Dependencies
With your virtual environment activated, install all required Python packages:
`pip install streamlit` `pandas` `numpy` `Pillow` `torch` `torchvision` `scikit-learn` `tqdm` `editdistance` `huggingface_hub`


*Note on PyTorch (torch and torchvision): 
The command above installs the CPU-only version of PyTorch. If you have a CUDA-enabled GPU and want to leverage it for faster training, please refer to the official PyTorch website (pytorch.org/get-started/locally/) for specific installation commands tailored to your CUDA version.*

#### 4. Prepare Your Dataset
The application expects a dataset structured as follows:
``` bash
data/
├── images/
│   ├── train/
│   │   ├── image1.png
│   │   ├── image2.png
│   │   └── ...
│   └── test/
│       ├── image_test1.png
│       ├── image_test2.png
│       └── ...
├── train.csv
└── test.csv
```

#### 5. Clear Python Cache *(Important!)*
After making code changes or installing new packages, it's crucial to clear Python's compiled cache to ensure the latest code is used.

```bash
find . -name "__pycache__" -exec rm -rf {} +  # For macOS/Linux

Get-ChildItem -Path . -Include __pycache__ -Recurse | Remove-Item -Recurse -Force # For Windows PowerShell
```

### 6. Run the Streamlit Application
With your virtual environment activated and dependencies installed:
`streamlit run app.py`


*This will open the application in your web browser.*

## ✏️ Features 
- **CRNN Model Architecture**: Utilizes a Convolutional Recurrent Neural Network for robust OCR.
- **CTC Loss**: Employs Connectionist Temporal Classification for sequence prediction.
**Model Training**: Train a new OCR model from your local image and CSV datasets.
- **Pre-trained Model Loading**: Load previously saved models to avoid retraining.
- **Handwritten Text Prediction**: Upload an image and get instant text recognition.
- **Training Progress Visualization**: Real-time updates and plots for training loss, CER, and accuracy.
- **Hugging Face Hub Integration**: Seamlessly upload your dataset to the Hugging Face Hub for easy sharing and version control.
- **Responsive UI**: Built with Streamlit for an intuitive and user-friendly experience.


## 🏗️ Project Structure
```
handwritten_name_ocr_app/
├── app.py                  # Main Streamlit application file
├── config.py               # Configuration settings (paths, model params, chars)
├── data/                   # Directory for datasets
│   ├── images/
│   │   ├── train/          # Training images
│   │   └── test/           # Testing images
│   ├── train.csv           # Training labels
│   └── test.csv            # Testing labels
├── data_handler_ocr.py     # Custom PyTorch Dataset and DataLoader logic
├── models/                 # Directory to save/load trained models
│   └── handwritten_name_ocr_model.pth # Default model save path
├── model_ocr.py            # Defines the CRNN model architecture and training/evaluation functions
├── utils_ocr.py            # Utility functions for image preprocessing
├── requirements.txt        # List of Python dependencies
└── venvy/                  # Python virtual environment (created by `python -m venv venvy`)
    └── ...
````

## 🗃️ Project Index

`app.py`: The central Streamlit application. Handles UI, triggers training/prediction, and integrates with Hugging Face Hub.

`config.py`: Stores global configuration variables such as file paths, image dimensions, character sets, and training hyperparameters.

`data_handler_ocr.py`: Contains the CharIndexer class for character-to-index mapping and the OCRDataset and ocr_collate_fn for efficient data loading and batching for PyTorch.

`model_ocr.py`: Defines the CNN_Backbone, BidirectionalLSTM, and CRNN (the main OCR model) classes. It also includes functions for train_ocr_model, evaluate_model, save_ocr_model, load_ocr_model, and ctc_greedy_decode.

``utils_ocr.py``: Provides helper functions for image preprocessing steps like binarization, resizing, and normalization, used before feeding images to the model.



##  📌 Roadmap
- Advanced Data Augmentation: Implement more sophisticated augmentation techniques (e.g., elastic deformations, random noise) for training data.
- Beam Search Decoding: Replace greedy decoding with beam search for potentially more accurate predictions.
- Error Analysis Dashboard: Integrate a more detailed error analysis section to visualize common recognition mistakes.
- Support for Multiple Languages: Extend character sets and train on multilingual datasets.
- Deployment to Cloud Platforms: Provide instructions for deploying the Streamlit app to platforms like Hugging Face Spaces, Heroku, or AWS.
- Pre-trained Model Download: Allow users to download pre-trained models directly from Hugging Face Hub.
- Interactive Drawing Pad: Enable users to draw a name directly in the app for recognition.

## 🎁 Contribution
Contributions are welcome! If you have suggestions, bug reports, or want to contribute code, please feel free to *fork the repository.* 
- Create a new branch (git checkout -b feature/your-feature-name).
Make your changes.
- Commit your changes (git commit -m 'Add new feature').
- Push to the branch (git push origin feature/your-feature-name).
- Open a Pull Request.

## ⚖️ License
This project is licensed under the MIT License - see the LICENSE file for details.

## ✨ Acknowledgements
**Streamlit**: For building interactive web applications with ease.

**PyTorch**: The open-source machine learning framework.

**Hugging** Face Hub: For model and dataset sharing.

**OpenCV**: For image processing utilities (implicitly used via utils_ocr).

**EditDistance**: For efficient calculation of character error rate.

**tqdm**: For progress bars during training.

---

*Built using Streamlit, PyTorch, OpenCV, and EditDistance © 2025 by **MFT***