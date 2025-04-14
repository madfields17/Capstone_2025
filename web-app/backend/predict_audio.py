import torch
import torch.nn.functional as F
import yaml

from model import RawNetBaseline
from pathlib import Path
from preprocess_audio import standardize_audio_from_bytes

THRESHOLD = 0.98482287

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load YAML Configuration ===
def load_yaml_config():
    rawnet2_config_path = Path("model_config_rawnet.yaml")
    with open(rawnet2_config_path, 'r') as f_yaml:
        return yaml.safe_load(f_yaml)

# === Load Pretrained RawNet2 Model ===
def load_model(config: dict):
    rawnet2_model = RawNetBaseline(config['model'], device=device)
    rawnet2_model_path = Path("best_rawnet2.pth")
    rawnet2_model.load_state_dict(torch.load(rawnet2_model_path, map_location=device))
    rawnet2_model.eval()
    return rawnet2_model

# === Process Audio Clip ===
def process_audio_clip(audio_file):
    y = standardize_audio_from_bytes(audio_file)
    waveform = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        outputs = Net(waveform)
        probs = torch.exp(outputs)
        score = probs[0, 1].item()  # Bona Fide Score
    return score

# === The function to call from the React frontend ===
def predict_from_audio_clip(audio_file):
    rawnet2_config = load_yaml_config()
    global Net
    Net = load_model(rawnet2_config)
    prediction_score = process_audio_clip(audio_file)
    prediction = "Bona Fide" if prediction_score > THRESHOLD else "Spoofed"
    return {
        "Prediction": prediction,
        "Bona Fide Score": prediction_score
    }