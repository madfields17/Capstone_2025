import argparse
import torch
import yaml
import numpy as np
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
from torchaudio import load as torchaudio_load
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import roc_curve

from Model import RawNetBaseline


# === Dataset Definition ===
class DeepfakeAudioDataset(torch.utils.data.Dataset):
    def __init__(self, df, base_path):
        self.df = df.reset_index(drop=True)
        self.base_path = Path(base_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = 1 if row['spoof_or_real'] == 'real' else 0
        file_path = self.base_path / row['Filename']
        waveform, sr = torchaudio_load(file_path)
        waveform = waveform.mean(dim=0, keepdim=True)  # convert to mono
        return waveform.squeeze(0), label


# === Collate function ===
def collate_fn(batch):
    waveforms, labels = zip(*batch)
    waveforms = [torch.tensor(w, dtype=torch.float32) for w in waveforms]
    waveforms = pad_sequence(waveforms, batch_first=True, padding_value=0.0)
    return waveforms, torch.tensor(labels)


# === Compute EER ===
def compute_eer(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.abs(fnr - fpr))
    eer = fpr[eer_index]
    threshold = thresholds[eer_index]
    return eer, threshold


# === Main ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to model_config_RawNet.yaml")
    parser.add_argument("--weights", type=str, required=True, help="Path to .pth weights")
    parser.add_argument("--output_dir", type=str, default=".", help="Where to store EER/threshold results")
    args = parser.parse_args()

    # Load model config
    with open(args.config, 'r') as f:
        model_config = yaml.safe_load(f)['model']

    # === Paths ===
    spoof_meta = pd.read_csv("./Standardized_full_data/Metadata TTS data_full_new.csv")
    real_meta = pd.read_csv("./Standardized_full_data/REAL_train_and_val.csv")
    real_meta = real_meta.rename(columns={"file_name": "Filename"})
    metadata = pd.concat([spoof_meta, real_meta], ignore_index=True)
    metadata['Filename'] = metadata['Filename'].apply(lambda x: Path(x).with_suffix('.wav').name)
    val_df = metadata[metadata['train_or_val'] == 'val']

    # === Dataset and Loader
    dataset = DeepfakeAudioDataset(val_df, "./Standardized_full_data/Val")
    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # === Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RawNetBaseline(model_config, device).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    # === Inference
    y_true, y_scores = [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # bonafide probability
            y_scores.extend(probs.cpu().numpy())
            y_true.extend(batch_y.numpy())

    # === Save Predictions
    df_out = pd.DataFrame({"label": y_true, "prediction_score": y_scores})
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weight_name = Path(args.weights).stem
    pred_path = output_dir / f"{weight_name}_val_predictions.csv"
    df_out.to_csv(pred_path, index=False)

    # === Compute EER
    eer, threshold = compute_eer(y_true, y_scores)
    with open(output_dir / f"{weight_name}_EER_threshold.txt", "w") as f:
        f.write(f"EER: {eer:.4f}\n")
        f.write(f"Threshold: {threshold:.4f}\n")

    print(f"✅ EER: {eer:.4f}, Threshold: {threshold:.4f}")
    print(f"📄 Saved predictions: {pred_path}")
    print(f"📄 Saved threshold: {weight_name}_EER_threshold.txt")


if __name__ == "__main__":
    main()

