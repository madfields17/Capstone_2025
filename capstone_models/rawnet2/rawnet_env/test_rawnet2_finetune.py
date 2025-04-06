import torch
import torchaudio
import pandas as pd
import yaml
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

from Model import RawNetBaseline

# === Define Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Paths ===
data_dir = Path("Standardized_full_data/new_evaluation_wav")
metadata = pd.read_csv("Standardized_full_data/new_evaluation_metadata.csv")
config_path = Path("model_config_RawNet.yaml")
# checkpoint_path = Path("finetuned_rawnet2_epoch10.pth")
checkpoint_path = Path("swa_rawnet2.pth")
threshold_path = Path("best_threshold.txt")

# metadata['file_name'] = metadata['file_name'].apply(lambda x: x if x.endswith('.wav') else x + '.wav')
# metadata['file_name'] = metadata['file_name'].apply(lambda x: Path(x).name)

# existing_files = set(f.name for f in Path(data_dir).glob("*.wav"))
# metadata = metadata[metadata['file_name'].isin(existing_files)]

# === Custom Dataset ===
class DeepfakeAudioDataset(Dataset):
    def __init__(self, df, base_path):
        self.df = df.reset_index(drop=True)
        self.base_path = Path(base_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = 0  # all real samples
        file_path = self.base_path / row['file_name']
        if not file_path.exists():
            print(f"Missing file: {file_path}")
            return torch.zeros(64600), label, row['file_name'], 64600
        waveform, sr = torchaudio.load(file_path)
        waveform = waveform.mean(dim=0, keepdim=True)
        return waveform.squeeze(0), label, row['file_name'], waveform.shape[1]

# === Collate Function ===
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    batch.sort(key=lambda x: x[3], reverse=True)
    waveforms, labels, filenames, lengths = zip(*batch)
    waveforms = [torch.tensor(w, dtype=torch.float32) for w in waveforms]
    waveforms = pad_sequence(waveforms, batch_first=True, padding_value=0.0)
    return waveforms.to(device), torch.tensor(labels).to(device), filenames

# === Load Data ===
dataset = DeepfakeAudioDataset(metadata, data_dir)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

# === Load Config and Model ===
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

model = RawNetBaseline(config['model'], device).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()
print(f"Loaded model from {checkpoint_path}")

# === Load Best Threshold ===
with open(threshold_path, 'r') as f:
    best_threshold = float(f.read().strip())
print(f"Loaded best threshold: {best_threshold:.4f}")

# === Run Inference ===
y_true, y_pred, y_scores, filenames = [], [], [], []
with torch.no_grad():
    for batch_x, batch_y, batch_filenames in tqdm(dataloader, desc="Evaluating on Real Test Set"):
        outputs = model(batch_x)
        probs = torch.softmax(outputs, dim=1)[:, 1]  # Spoof probability

        preds = (probs >= best_threshold).long()

        y_true.extend(batch_y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_scores.extend(probs.cpu().numpy())
        filenames.extend(batch_filenames)

# === Compute Metrics ===
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_true, y_scores)

# === Confusion Matrix ===
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0, 0], cm[0, 1], 0, 0)
tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

print("\nReal Test Set Metrics:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC AUC  : {roc_auc:.4f}")
print(f"TNR      : {tnr:.4f}")
print(f"FPR      : {fpr:.4f}")

# === Save Predictions ===
pred_df = pd.DataFrame({
    "file_name": filenames,
    "True Label": y_true,
    "Predicted Label": y_pred,
    "Spoof Probability": y_scores
})

# Merge with metadata
metadata['file_name'] = metadata['file_name'].apply(lambda x: Path(x).name)
pred_df['file_name'] = pred_df['file_name'].apply(lambda x: Path(x).name)
merged_df = metadata.merge(pred_df, on="file_name", how="inner")

# === Grouped Metrics (TNR/FPR) ===
def group_metrics(df, group_col):
    records = []
    for group, gdf in df.groupby(group_col):
        if len(gdf) < 2:
            continue
        cm = confusion_matrix(gdf['True Label'], gdf['Predicted Label'])
        if cm.shape == (1, 2):
            tn, fp = cm[0][0], cm[0][1]
        else:
            tn, fp = 0, 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        records.append({group_col: group, "TNR": tnr, "FPR": fpr})
    return pd.DataFrame(records)

gender_metrics = group_metrics(merged_df, "gender")
region_metrics = group_metrics(merged_df, "region")

# === Save Grouped Metrics Summary ===
summary_csv = pd.concat([
    gender_metrics.assign(Group="gender"),
    region_metrics.assign(Group="region")
])
summary_csv.to_csv("summary_group_metrics_finetuned.csv", index=False)
print("\nSaved summary metrics to summary_group_metrics_finetuned.csv")

# === Visualization ===
def plot_group_metrics(df, group_col):
    df.set_index(group_col)[["TNR", "FPR"]].plot(kind="bar", figsize=(10, 6))
    plt.title(f"TNR and FPR by {group_col.capitalize()}")
    plt.ylabel("Rate")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"tnr_fpr_by_{group_col}.png")
    plt.close()

plot_group_metrics(gender_metrics, "gender")
plot_group_metrics(region_metrics, "region")
print("Saved TNR/FPR plots by gender and region.")

# === Save detailed results ===
merged_df.to_csv("evaluation_results_real_test_set.csv", index=False)
print("Saved detailed results to evaluation_results_real_test_set.csv")
