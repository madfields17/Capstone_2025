import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
import yaml

from data_utils import Dataset_Mozilla_TSSD,Dataset_Mozilla_RawNet2
from model import  DownStreamLinearClassifier, RawNetEncoderBaseline, RawNetBaseline, SSDNet1D, SAMOArgs  # SSDNet is the Res-TSSDNet Model
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# === Define Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Define Paths ===
metadata_file = Path("../../datasets/evaluation-data/evaluation-metadata-standardized.csv")
wav_dir = Path("../../datasets/evaluation-data/evaluation-set-standardized")
output_file = Path("../finetune-2/tssd-evaluation-results-standardized.csv")
merge_output_file = Path("../finetune-2/final-results-tssd-standardized.csv")
checkpoint_path = Path("../finetune-2/swa_tssdnet.pth")

# === Load Pretrained TSSDNet Model ===
model = SSDNet1D()
check_point = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(check_point)
model = model.to(device)
print(f"Loaded model from {checkpoint_path}")

# === Custom Collate Function for Variable-Length Audio ===
def collate_fn(batch):
    """
    Custom collate function for RawNet2:
    - Sorts waveforms by length (descending).
    - Converts list of tensors into a batch tensor.
    """
    batch.sort(key=lambda x: x[2], reverse=True)
    waveforms, file_names, lengths = zip(*batch)
    waveforms = [torch.tensor(waveform, dtype=torch.float32) for waveform in waveforms]
    waveforms = torch.nn.utils.rnn.pad_sequence(
        waveforms, batch_first=True, padding_value=0.0
    ).to(device)
    return waveforms, file_names, lengths

# === Create DataLoader ===
dataset = Dataset_Mozilla_TSSD(base_dir=wav_dir)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate_fn)

# === Perform Inference and Save Scores ===
results = []
with torch.no_grad():
    for batch_x, file_names, lengths in tqdm(dataloader, desc="Processing Audio Files"):
        batch_x = tuple(batch_x)
        batch_x = torch.stack(batch_x).unsqueeze(1).to(device)
        outputs = model(batch_x)
        probs = F.softmax(outputs, dim=1) 
        standardized_scores = probs[:, 1].cpu().numpy()  # Extract spoof probability.
        for file_name, score in zip(file_names, standardized_scores):
            results.append({"wav_path": file_name, "spoof_score": score})

# === Save Predictions to CSV ===
df = pd.DataFrame(results)
df.to_csv(output_file, index=False)
print(f"Predictions saved to {output_file}")

# === Merge Predictions with Metadata ===
metadata = pd.read_csv(metadata_file)
df["wav_filename"] = df["wav_path"].apply(lambda x: Path(x).name)
metadata["wav_filename"] = metadata["file_name"].str.replace(".mp3", ".wav", regex=False)
merged_df = metadata.merge(df, on="wav_filename", how="left")
merged_df.to_csv(merge_output_file, index=False)
print(f"✅ Merged results saved to {merge_output_file}")