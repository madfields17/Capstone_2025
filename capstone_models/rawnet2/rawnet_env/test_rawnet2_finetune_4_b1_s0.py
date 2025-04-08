import torch
import torchaudio
import pandas as pd
import yaml
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from Model import  DownStreamLinearClassifier, RawNetEncoderBaseline, RawNetBaseline, SSDNet1D, SAMOArgs  # SSDNet is the Res-TSSDNet Model
from data_utils import Dataset_Mozilla,Dataset_Mozilla_RawNet2#, collate_fn
from tqdm import tqdm


# === Define Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Define Paths ===
metadata_file = Path("Standardized_full_data/new_evaluation_metadata.csv")
wav_dir = Path("Standardized_full_data/new_evaluation_wav")
output_file = Path("rawnet2_evaluation_results_finetune_7_b1_s0.csv")
merge_output_file = Path("final_results_rawnet2_finetune_7_b1_s0.csv")
rawnet2_config_path = Path("model_config_RawNet.yaml")
rawnet2_model_path = Path("swa_rawnet2_3.pth")

# === Load YAML Configuration ===
with open(rawnet2_config_path, 'r') as f_yaml:
    rawnet2_config = yaml.safe_load(f_yaml)

# === Load Pretrained RawNet2 Model ===
def load_model(model_name: str, config: dict):
    if model_name == "RawNet2":
        rawnet2_model = RawNetBaseline(config['model'], device)  # Use the dictionary directly
        rawnet2_model = rawnet2_model.to(device)
        rawnet2_model.load_state_dict(torch.load(rawnet2_model_path, map_location=device))
        print(f'RawNetBaseline model loaded: {rawnet2_model_path}')
        
        # Print number of parameters
        nb_params = sum(p.numel() for p in rawnet2_model.parameters())
        print(f"Number of RawNet2 params: {nb_params}")

        return rawnet2_model

Net = load_model("RawNet2", rawnet2_config)
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
dataset = Dataset_Mozilla_RawNet2(base_dir="Standardized_full_data/new_evaluation_wav")
dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate_fn)

# === Perform Inference and Save Scores ===
results = []
with torch.no_grad():
    for batch_x, file_names, lengths in tqdm(dataloader, desc="Processing Audio Files"):
        batch_x = tuple(batch_x)
        batch_x = torch.stack(batch_x).squeeze(1).to(device)  # Correct way to batch tensors
        outputs = Net(batch_x)  # Forward pass

        probs = torch.exp(outputs)  # Convert log-softmax outputs to probabilities
        scores = probs[:, 1].cpu().numpy()  # Extract bonafide probability


        for file_name, score in zip(file_names, scores):
            results.append({"wav_path": file_name, "bonafide_prediction_score": score})

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
