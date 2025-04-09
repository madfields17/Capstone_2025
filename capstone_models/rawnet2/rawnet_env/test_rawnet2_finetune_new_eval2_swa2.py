import torch
import torchaudio
import pandas as pd
import yaml
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from Model import DownStreamLinearClassifier, RawNetEncoderBaseline, RawNetBaseline, SSDNet1D, SAMOArgs
from data_utils import Dataset_Mozilla_RawNet2
from tqdm import tqdm

# === Define Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Define Paths ===
metadata_file = Path("Standardized_full_data/new_evaluation_metadata_2.csv")
wav_dir = Path("Standardized_full_data/new_evaluation_wav_2")
output_file = Path("finetune_evaluation_new_evaluation_set_2_swa2.csv")
merge_output_file = Path("finetune_evaluation_new_evaluation_set_2_swa2_merged.csv")
rawnet2_config_path = Path("model_config_RawNet.yaml")
rawnet2_model_path = Path("swa_rawnet2_2.pth")

# === Load YAML Configuration ===
with open(rawnet2_config_path, 'r') as f_yaml:
    rawnet2_config = yaml.safe_load(f_yaml)

# === Load Pretrained RawNet2 Model ===
def load_model(model_name: str, config: dict):
    if model_name == "RawNet2":
        rawnet2_model = RawNetBaseline(config['model'], device)
        rawnet2_model = rawnet2_model.to(device)
        rawnet2_model.load_state_dict(torch.load(rawnet2_model_path, map_location=device))
        print(f'RawNetBaseline model loaded: {rawnet2_model_path}')

        nb_params = sum(p.numel() for p in rawnet2_model.parameters())
        print(f"Number of RawNet2 params: {nb_params}")
        return rawnet2_model

Net = load_model("RawNet2", rawnet2_config)
Net.eval()

# === Custom Collate Function ===
def collate_fn(batch):
    batch.sort(key=lambda x: x[2], reverse=True)
    waveforms, file_names, lengths = zip(*batch)
    waveforms = [torch.tensor(waveform, dtype=torch.float32) for waveform in waveforms]
    waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True, padding_value=0.0).to(device)
    return waveforms, file_names, lengths

# === Create DataLoader ===
dataset = Dataset_Mozilla_RawNet2(base_dir=str(wav_dir))
dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate_fn)

# === Run Inference ===
results = []
with torch.no_grad():
    for batch_x, file_names, lengths in tqdm(dataloader, desc="Processing Audio Files"):
        #batch_x = torch.stack(batch_x).squeeze(1).to(device)
        batch_x = batch_x.to(device)
        outputs = Net(batch_x)
        probs = torch.exp(outputs)
        scores = probs[:, 1].cpu().numpy()  # bonafide probability if label=1 for bonafide

        for file_name, score in zip(file_names, scores):
            results.append({"wav_path": file_name, "prediction_score": score})

# === Save Predictions ===
df = pd.DataFrame(results)
df.to_csv(output_file, index=False)
print(f"Predictions saved to {output_file}")

# === Merge with Metadata ===
metadata = pd.read_csv(metadata_file)
df["wav_filename"] = df["wav_path"].apply(lambda x: Path(x).name)
metadata["wav_filename"] = metadata["file_name"]  # Already contains .wav
merged_df = metadata.merge(df, on="wav_filename", how="left")
merged_df.to_csv(merge_output_file, index=False)
print(f"✅ Merged results saved to {merge_output_file}")

