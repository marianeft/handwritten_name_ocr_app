# model_ocr.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader # Keep DataLoader for type hinting
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import editdistance

# Import config and char_indexer
# Ensure these imports align with your current config.py
from config import IMG_HEIGHT, NUM_CLASSES, BLANK_TOKEN 
from data_handler_ocr import CharIndexer 
# You might also need to import binarize_image, resize_image_for_ocr, normalize_image_for_model
# if they are used directly in model_ocr.py for internal preprocessing (e.g., in evaluate_model if not using DataLoader)
# For now, assuming they are handled by DataLoader transforms.
from utils_ocr import binarize_image, resize_image_for_ocr, normalize_image_for_model # Add this for completeness if needed elsewhere


class CNN_Backbone(nn.Module):
    """
    CNN feature extractor for OCR. Designed to produce features suitable for RNN.
    Output feature map should have height 1 after the final pooling/reduction.
    """
    def __init__(self, input_channels=1, output_channels=512):
        super(CNN_Backbone, self).__init__()
        self.cnn = nn.Sequential(
            # First block
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2), # H: 32 -> 16, W: W_in -> W_in/2

            # Second block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2), # H: 16 -> 8, W: W_in/2 -> W_in/4

            # Third block (with two conv layers)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            # This MaxPool2d effectively brings height from 8 to 4, with a small width adjustment due to padding
            # The original comment (W/4 + 1) is due to padding=1 and stride=1 on width, which is fine.
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)), # H: 8 -> 4, W: (W/4) -> (W/4 + 1) (approx)

            # Fourth block
            nn.Conv2d(256, output_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            # This AdaptiveAvgPool2d makes sure the height dimension becomes 1
            # while preserving the width. This is crucial for RNN input.
            nn.AdaptiveAvgPool2d((1, None)) # Output height 1, preserve width
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, H, W) e.g., (B, 1, 32, W_img)
        
        # Pass through the CNN layers
        conv_features = self.cnn(x) # Output: (N, cnn_out_channels, 1, W_prime)

        # Squeeze the height dimension (which is 1)
        # This transforms (N, C_out, 1, W_prime) to (N, C_out, W_prime)
        conv_features = conv_features.squeeze(2) 
        
        # Permute for RNN input: (sequence_length, batch_size, input_size)
        # This transforms (N, C_out, W_prime) to (W_prime, N, C_out)
        conv_features = conv_features.permute(2, 0, 1) 

        # Return the CNN features, ready for the RNN layer in CRNN
        return conv_features

class BidirectionalLSTM(nn.Module):
    """Bidirectional LSTM layer for sequence modeling."""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.5):
        super(BidirectionalLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            bidirectional=True, dropout=dropout, batch_first=False)
        # batch_first=False expects input as (sequence_length, batch_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x) # [0] returns the output, [1] returns (h_n, c_n)
        return output

class CRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network for OCR.
    Combines CNN for feature extraction, LSTMs for sequence modeling,
    and a final linear layer for character prediction.
    """
    def __init__(self, num_classes: int, cnn_output_channels: int = 512,
                 rnn_hidden_size: int = 256, rnn_num_layers: int = 2):
        super(CRNN, self).__init__()
        self.cnn = CNN_Backbone(output_channels=cnn_output_channels)
        # Input to LSTM is the number of channels from the CNN output
        self.rnn = BidirectionalLSTM(cnn_output_channels, rnn_hidden_size, rnn_num_layers)
        # Output of bidirectional LSTM is hidden_size * 2
        self.fc = nn.Linear(rnn_hidden_size * 2, num_classes) # Final linear layer for classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, H, W) e.g., (B, 1, 32, W_img)
        
        # 1. Pass through the CNN to extract features
        conv_features = self.cnn(x) # Output: (W_prime, N, C_out) after permute in CNN_Backbone

        # 2. Pass CNN features through the RNN (LSTM)
        rnn_features = self.rnn(conv_features) # Output: (W_prime, N, rnn_hidden_size * 2)

        # 3. Pass RNN features through the final fully connected layer
        # Apply the linear layer to each time step independently
        # output will be (W_prime, N, num_classes)
        output = self.fc(rnn_features) 
        
        return output


# --- Decoding Function ---
def ctc_greedy_decode(output: torch.Tensor, char_indexer: CharIndexer) -> list[str]:
    """
    Performs greedy decoding on the CTC output.
    output: (sequence_length, batch_size, num_classes) - raw logits
    """
    # Apply log_softmax to get probabilities for argmax
    log_probs = F.log_softmax(output, dim=2)
    
    # Permute to (batch_size, sequence_length, num_classes) for argmax along class dim
    # This gives us the index of the most probable character at each time step for each sample in the batch.
    predicted_indices = torch.argmax(log_probs.permute(1, 0, 2), dim=2).cpu().numpy()

    decoded_texts = []
    for seq in predicted_indices:
        # Use char_indexer's decode method, which handles blank removal and duplicate collapse
        decoded_texts.append(char_indexer.decode(seq.tolist())) # Convert numpy array to list
    return decoded_texts

# --- Evaluation Function ---
def evaluate_model(model: nn.Module, dataloader: DataLoader, char_indexer: CharIndexer, device: str):
    model.eval() # Set model to evaluation mode
    # CTCLoss needs the blank token index, which is available from char_indexer
    criterion = nn.CTCLoss(blank=char_indexer.blank_token_idx, zero_infinity=True) 
    total_loss = 0
    all_predictions = []
    all_ground_truths = []

    with torch.no_grad(): # Disable gradient calculation for evaluation
        for inputs, targets_padded, _, target_lengths in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            targets_padded = targets_padded.to(device)
            target_lengths = target_lengths.to(device)

            output = model(inputs) # (seq_len, batch_size, num_classes)

            # Calculate input_lengths for CTCLoss. This is the sequence length produced by the CNN/RNN.
            # It's the `output.shape[0]` (sequence_length) for each item in the batch.
            outputs_seq_len_for_ctc = torch.full(
                size=(output.shape[1],), # batch_size
                fill_value=output.shape[0], # actual sequence length (T) from model output
                dtype=torch.long,
                device=device 
            )
            
            # CTC Loss calculation requires log_softmax on the output logits
            log_probs_for_loss = F.log_softmax(output, dim=2) # (T, N, C)

            loss = criterion(log_probs_for_loss, targets_padded, outputs_seq_len_for_ctc, target_lengths)
            total_loss += loss.item() * inputs.size(0) # Multiply by batch size for correct average

            # Decode predictions for metrics
            decoded_preds = ctc_greedy_decode(output, char_indexer)
            
            # Reconstruct ground truths from encoded tensors
            ground_truths = []
            # Loop through each sample in the batch
            for i in range(targets_padded.size(0)):
                # Extract the actual target sequence for the i-th sample using its length
                # Convert to list before passing to char_indexer.decode
                ground_truths.append(char_indexer.decode(targets_padded[i, :target_lengths[i]].tolist()))

            all_predictions.extend(decoded_preds)
            all_ground_truths.extend(ground_truths)

    avg_loss = total_loss / len(dataloader.dataset)

    # Calculate Character Error Rate (CER)
    cer_sum = 0
    total_chars = 0
    for pred, gt in zip(all_predictions, all_ground_truths):
        cer_sum += editdistance.eval(pred, gt)
        total_chars += len(gt)
    char_error_rate = cer_sum / total_chars if total_chars > 0 else 0.0

    # Calculate Exact Match Accuracy (Word-level Accuracy)
    exact_match_accuracy = accuracy_score(all_ground_truths, all_predictions)

    return avg_loss, char_error_rate, exact_match_accuracy

# --- Training Function ---
def train_ocr_model(model: nn.Module, train_loader: DataLoader,
                    test_loader: DataLoader, char_indexer: CharIndexer,
                    epochs: int, device: str, progress_callback=None) -> tuple[nn.Module, dict]:
    """
    Trains the OCR model using CTC loss.
    """
    # CTCLoss needs the blank token index
    criterion = nn.CTCLoss(blank=char_indexer.blank_token_idx, zero_infinity=True) 
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Using a fixed LR for now
    # Using ReduceLROnPlateau to adjust LR based on test loss (monitor 'min' loss)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5)

    model.to(device) # Ensure model is on the correct device
    model.train() # Set model to training mode

    training_history = {
        'train_loss': [],
        'test_loss': [],
        'test_cer': [],
        'test_exact_match_accuracy': []
    }

    for epoch in range(epochs):
        running_loss = 0.0
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Train)")
        for images, texts_encoded, _, text_lengths in pbar_train:
            images = images.to(device)
            # Ensure target tensors are on the correct device for CTCLoss calculation
            texts_encoded = texts_encoded.to(device)
            text_lengths = text_lengths.to(device)

            optimizer.zero_grad() # Clear gradients from previous step
            outputs = model(images) # (sequence_length_from_cnn, batch_size, num_classes)

            # `outputs.shape[0]` is the actual sequence length (T) produced by the model.
            # CTC loss expects `input_lengths` to be a tensor of shape (batch_size,) with these values.
            outputs_seq_len_for_ctc = torch.full(
                size=(outputs.shape[1],), # batch_size
                fill_value=outputs.shape[0], # actual sequence length (T) from model output
                dtype=torch.long,
                device=device 
            )
            
            # CTC Loss calculation requires log_softmax on the output logits
            log_probs_for_loss = F.log_softmax(outputs, dim=2) # (T, N, C)

            # Use outputs_seq_len_for_ctc for the input_lengths argument
            loss = criterion(log_probs_for_loss, texts_encoded, outputs_seq_len_for_ctc, text_lengths)
            loss.backward() # Backpropagate
            optimizer.step() # Update model weights

            running_loss += loss.item() * images.size(0) # Multiply by batch size for correct average
            pbar_train.set_postfix(loss=loss.item())

        epoch_train_loss = running_loss / len(train_loader.dataset)
        training_history['train_loss'].append(epoch_train_loss)

        # Evaluate on test set using the dedicated function
        # Ensure model is in eval mode before calling evaluate_model
        model.eval() 
        test_loss, test_cer, test_exact_match_accuracy = evaluate_model(model, test_loader, char_indexer, device)
        training_history['test_loss'].append(test_loss)
        training_history['test_cer'].append(test_cer)
        training_history['test_exact_match_accuracy'].append(test_exact_match_accuracy)

        # Adjust learning rate based on test loss (this is where scheduler.step() is called)
        scheduler.step(test_loss)

        print(f"Epoch {epoch+1}/{epochs}: Train Loss={epoch_train_loss:.4f}, "
              f"Test Loss={test_loss:.4f}, Test CER={test_cer:.4f}, Test Exact Match Acc={test_exact_match_accuracy:.4f}")

        if progress_callback:
            # Update progress bar with current epoch and key metrics
            progress_val = (epoch + 1) / epochs
            progress_callback(progress_val, text=f"Epoch {epoch+1}/{epochs} done. Test CER: {test_cer:.4f}, Test Exact Match Acc: {test_exact_match_accuracy:.4f}")

        model.train() # Set model back to training mode after evaluation

    return model, training_history

def save_ocr_model(model: nn.Module, path: str):
    """Saves the state dictionary of the trained OCR model."""
    torch.save(model.state_dict(), path)
    print(f"OCR model saved to {path}")

def load_ocr_model(model: nn.Module, path: str):
    """
    Loads a trained OCR model's state dictionary.
    Includes map_location to handle loading models trained on GPU to CPU, and vice versa.
    """
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu'))) # Always load to CPU first
    model.eval() # Set to evaluation mode
    print(f"OCR model loaded from {path}")