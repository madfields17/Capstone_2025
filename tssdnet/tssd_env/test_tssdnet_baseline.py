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
metadata_file = Path("../../datasets/evaluation-data/evaluation-metadata.csv")
wav_dir = Path("../../datasets/evaluation-data/evaluation-set")
output_file = Path("../baseline-updated-results/tssd-evaluation-results.csv")
merge_output_file = Path("../baseline-updated-results/final-results-tssd.csv")
tssdnet_model_path = Path("Res_TSSDNet_time_frame_61_ASVspoof2019_LA_Loss_0.0017_dEER_0.74%_eEER_1.64%.pth")

# === Load Pretrained TSSDNet Model ===
def load_model(model_name: str):
    if model_name == "ResTSSDNetModel":
        res_tssdnet_model = SSDNet1D()  # Use the dictionary directly
        check_point = torch.load(tssdnet_model_path, map_location=torch.device('cpu'))
        res_tssdnet_model.load_state_dict(check_point['model_state_dict'])
        res_tssdnet_model = res_tssdnet_model.to('cpu')
        print(f'res_tssdnet_model model loaded: {res_tssdnet_model}')
        
        # Print number of parameters
        nb_params = sum(p.numel() for p in res_tssdnet_model.parameters())
        print(f"Number of res_tssdnet_model params: {nb_params}")

        return res_tssdnet_model

Net = load_model("ResTSSDNetModel")
Net.eval()

# === Custom Collate Function for Variable-Length Audio ===
def collate_fn(batch):
    """
    Custom collate function for RawNet2:
    - Sorts waveforms by length (descending).
    - Converts list of tensors into a batch tensor.
    """
    batch.sort(key=lambda x: x[2], reverse=True)  # Sort by length (descending)
    waveforms, file_names, lengths = zip(*batch)  # Unzip batch elements

    # Ensure waveforms is a tuple of tensors
    waveforms = [torch.tensor(waveform, dtype=torch.float32) for waveform in waveforms]

    # Convert waveforms list to a padded batch tensor
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
        batch_x = torch.stack(batch_x).unsqueeze(1).to(device)  # Correct way to batch tensors
        outputs = Net(batch_x)  # Forward pass

        # Swap the logits (reverse order)
        outputs = outputs[:, [1, 0]]

        probs = F.softmax(outputs, dim=1)  # Compute probabilities
        standardized_scores = probs[:, 1].cpu().numpy()  # Extract bonafide probability (now correctly reversed)



        for file_name, score in zip(file_names, standardized_scores):
            results.append({"wav_path": file_name, "prediction_score": score})

# === Save Predictions to CSV ===
df = pd.DataFrame(results)
df.to_csv(output_file, index=False)
print(f"Predictions saved to {output_file}")

# === Merge Predictions with Metadata ===
# Load metadata
metadata = pd.read_csv(metadata_file)

# Extract only the filename from the `wav_path` column
df["wav_filename"] = df["wav_path"].apply(lambda x: Path(x).name)

# Ensure filenames match by converting .mp3 names to .wav in metadata
metadata["wav_filename"] = metadata["file_name"].str.replace(".mp3", ".wav", regex=False)

# Merge metadata with results on the cleaned WAV file name
merged_df = metadata.merge(df, on="wav_filename", how="left")

# Save the final merged dataset
merged_df.to_csv(merge_output_file, index=False)
print(f"✅ Merged results saved to {merge_output_file}")