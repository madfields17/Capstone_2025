import torch
import torchaudio
import pandas as pd
import yaml
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
import time
import numpy as np
import os
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

from Model import RawNetBaseline

# === Define Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Paths ===
data_dir = Path("Standardized_full_data")
spoof_metadata = pd.read_csv("Standardized_full_data/Metadata TTS data_full_new.csv")
real_metadata = pd.read_csv("Standardized_full_data/REAL_train_and_val.csv")

# === Merge Metadata ===
real_metadata = real_metadata.rename(columns={"file_name": "Filename"})
metadata = pd.concat([spoof_metadata, real_metadata], ignore_index=True)
metadata = metadata[['Filename', 'spoof_or_real', 'train_or_val']]
metadata['Filename'] = metadata['Filename'].apply(lambda x: Path(x).with_suffix('.wav').name)

# === Custom Dataset ===
class DeepfakeAudioDataset(Dataset):
    def __init__(self, df, base_path):
        self.df = df
        self.base_path = Path(base_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = 1 if row['spoof_or_real'] == "spoof" else 0
        file_path = self.base_path / row['Filename']
        if not file_path.exists():
            print(f"Missing file: {file_path}")
            return torch.zeros(64600), label, 64600  # dummy waveform if missing
        waveform, sr = torchaudio.load(file_path)
        waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono
        return waveform.squeeze(0), label, waveform.shape[1]  # (T,), label, length

# === Collate Function ===
def collate_fn(batch):
    batch = [b for b in batch if b is not None]  # Filter out None entries
    batch.sort(key=lambda x: x[2], reverse=True)
    waveforms, labels, lengths = zip(*batch)
    waveforms = [torch.tensor(w, dtype=torch.float32) for w in waveforms]
    waveforms = pad_sequence(waveforms, batch_first=True, padding_value=0.0)
    return waveforms.to(device), torch.tensor(labels).to(device)

# === Create Datasets ===
train_df = metadata[metadata['train_or_val'] == 'train']
val_df = metadata[metadata['train_or_val'] == 'val']

train_dataset = DeepfakeAudioDataset(train_df, data_dir / "Training")
val_dataset = DeepfakeAudioDataset(val_df, data_dir / "Val")

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

# === Load YAML Config ===
with open("model_config_RawNet.yaml", 'r') as f:
    config = yaml.safe_load(f)

# === Load Pretrained RawNet2 ===
model = RawNetBaseline(config['model'], device).to(device)
model.load_state_dict(torch.load("pre_trained_DF_RawNet2.pth", map_location=device))
print("Loaded pre-trained RawNet2")

# === Set Up Training ===
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
swa_model = AveragedModel(model)
swa_scheduler = SWALR(optimizer, swa_lr=5e-5)
swa_start = 30
num_epochs = 75

# === Function to Compute EER and Threshold ===
def compute_eer(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.absolute((fnr - fpr)))
    eer_threshold = thresholds[eer_index]
    eer = fpr[eer_index]
    return eer, eer_threshold

# === Initialize Best EER Tracking ===
best_val_eer = float('inf')
best_threshold = None
os.makedirs("saved_checkpoints", exist_ok=True)

# === Training Loop ===
for epoch in range(num_epochs):
    start_time = time.time()
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for batch_x, batch_y in tqdm(train_loader, desc=f"[Epoch {epoch+1}] Training"):
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_x.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)

    train_acc = correct / total
    train_loss = running_loss / total
    print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

    # === Validation ===
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    all_labels, all_probs = [], []

    with torch.no_grad():
        for batch_x, batch_y in tqdm(val_loader, desc=f"[Epoch {epoch+1}] Validation"):
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Spoof prob

            val_loss += loss.item() * batch_x.size(0)
            preds = outputs.argmax(dim=1)
            val_correct += (preds == batch_y).sum().item()
            val_total += batch_y.size(0)

            all_labels.extend(batch_y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    val_acc = val_correct / val_total
    val_loss = val_loss / val_total
    val_eer, threshold = compute_eer(all_labels, all_probs)

    if val_eer < best_val_eer:
        best_val_eer = val_eer
        best_threshold = threshold
        with open("best_threshold.txt", "w") as f:
            f.write(str(best_threshold))
        print(f"New best threshold saved: {best_threshold:.4f} (EER: {val_eer:.4f})")

        torch.save(model.state_dict(), f"saved_checkpoints/epoch_{epoch+1}_EER_{val_eer:.4f}.pth")

    # === SWA Update ===
    if epoch >= swa_start:
        swa_model.update_parameters(model)
        swa_scheduler.step()
    else:
        swa_scheduler.step()

    end_time = time.time()
    epoch_duration = end_time - start_time

    print(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val EER: {val_eer:.4f}")
    print(f"Epoch {epoch+1} Duration: {epoch_duration:.2f} seconds")

# === Final SWA Preparation ===
print("Finalizing SWA model with batch norm stats...")
update_bn(train_loader, swa_model)
torch.save(swa_model.module.state_dict(), "swa_rawnet2.pth")
print("Saved SWA model to swa_rawnet2.pth")
