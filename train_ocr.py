import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import easyocr
import Levenshtein

# Add parent directory for dataset utilities
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from crnn_dataset import CRNNDataset, collate_fn, VOCAB

def decode_batch_predictions(preds, vocab):
    """
    Decodes CTC prediction logits [T, B, C] into character strings.
    """
    preds = preds.permute(1, 0, 2)  # [B, T, C]
    _, max_indices = preds.max(2)   # [B, T]
    
    decoded_strings = []
    for i in range(max_indices.shape[0]):
        seq = max_indices[i]
        chars = []
        for j in range(seq.shape[0]):
            # Skip blank tokens (0) and consecutive duplicates
            if seq[j] != 0 and (not (j > 0 and seq[j - 1] == seq[j])):
                idx = seq[j].item() - 1
                if 0 <= idx < len(vocab):
                    chars.append(vocab[idx])
        decoded_strings.append("".join(chars))
    return decoded_strings

def calculate_cer(pred_strs, label_strs):
    cer_sum = 0
    total_chars = 0
    for pred, label in zip(pred_strs, label_strs):
        cer_sum += Levenshtein.distance(pred, label)
        total_chars += len(label)
    return cer_sum, total_chars

def evaluate(model, dataloader, device, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    total_cer_dist = 0
    total_cer_chars = 0
    
    with torch.no_grad():
        for imgs, labels, targets_flat, target_lengths in dataloader:
            imgs = imgs.to(device)
            targets_flat = targets_flat.to(device)
            target_lengths = target_lengths.to(device)
            
            preds = model(imgs, text=None)
            preds = preds.permute(1, 0, 2)  # [T, B, C]
            
            T, B, C = preds.size()
            input_lengths = torch.full(size=(B,), fill_value=T, dtype=torch.long).to(device)
            
            log_probs = nn.functional.log_softmax(preds, dim=2)
            loss = criterion(log_probs, targets_flat, input_lengths, target_lengths)
            
            total_loss += loss.item() * B
            decoded_preds = decode_batch_predictions(preds, VOCAB)
            
            cer_dist, cer_chars = calculate_cer(decoded_preds, labels)
            total_cer_dist += cer_dist
            total_cer_chars += cer_chars
            
            for pred, true_label in zip(decoded_preds, labels):
                if pred == true_label:
                    correct += 1
                total += 1
                
    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / total if total > 0 else 0.0
    cer = total_cer_dist / total_cer_chars if total_cer_chars > 0 else 0.0
    
    return avg_loss, accuracy, cer

def main():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/dataset0_ocr_finetuning")
    
    batch_size = 32
    num_epochs = 30
    learning_rate = 1e-4
    patience = 5
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")
    
    train_dataset = CRNNDataset(data_dir, split="train")
    valid_dataset = CRNNDataset(data_dir, split="valid")
    test_dataset = CRNNDataset(data_dir, split="test")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    print("Loading EasyOCR recognizer backbone...")
    reader = easyocr.Reader(['en'], gpu=False)
    model = reader.recognizer.to(device)
    
    criterion = nn.CTCLoss(blank=0, reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_valid_acc = -1.0
    epochs_no_improve = 0
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)
    best_model_path = os.path.join(models_dir, "best_easyocr_finetuned.pt")
    
    print("Starting OCR fine-tuning loop...")
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        
        for batch_idx, (imgs, labels, targets_flat, target_lengths) in enumerate(train_loader):
            imgs = imgs.to(device)
            targets_flat = targets_flat.to(device)
            target_lengths = target_lengths.to(device)
            
            optimizer.zero_grad()
            
            preds = model(imgs, text=None)
            preds = preds.permute(1, 0, 2)
            
            T, B, C = preds.size()
            input_lengths = torch.full(size=(B,), fill_value=T, dtype=torch.long).to(device)
            
            log_probs = nn.functional.log_softmax(preds, dim=2)
            loss = criterion(log_probs, targets_flat, input_lengths, target_lengths)
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item() * B
            
        avg_train_loss = total_train_loss / len(train_dataset)
        valid_loss, valid_acc, valid_cer = evaluate(model, valid_loader, device, criterion)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc*100:.2f}% | Valid CER: {valid_cer:.4f}")
        
        if valid_acc >= best_valid_acc:
            best_valid_acc = valid_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  Saved best checkpoint (Acc: {valid_acc*100:.2f}%)")
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
            
    print("\nEvaluating best checkpoint on test set...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        
    test_loss, test_acc, test_cer = evaluate(model, test_loader, device, criterion)
    print(f"Test Set Results | Loss: {test_loss:.4f} | Accuracy: {test_acc*100:.2f}% | CER: {test_cer:.4f}")

if __name__ == "__main__":
    main()
