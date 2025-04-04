import argparse
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from importlib import import_module
from utils import set_seed
from pathlib import Path
import os
import pandas as pd
from data_utils import NONBIASDataset


def compute_eer_from_predictions(score_file_path):
    df = pd.read_csv(score_file_path)
    scores = df["prediction_score"].values
    labels = df["label"].values

    target_scores = scores[labels == 1]  # real
    spoof_scores = scores[labels == 0]  # spoof

    if len(target_scores) == 0 or len(spoof_scores) == 0:
        print("⚠️ Not enough spoof or real samples to compute EER")
        return float("nan")

    # Basic EER computation
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.abs(fnr - fpr))
    eer = fpr[eer_index]
    threshold = thresholds[eer_index]
    return eer, threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--weights", type=str, required=True, help="Path to .pth model weights")
    parser.add_argument("--output_dir", type=str, default=".", help="Where to save the threshold output")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f_json:
        config = json.load(f_json)

    model_config = config["model_config"]
    set_seed(args.seed, config)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    module = import_module("models.{}".format(model_config["architecture"]))
    model_class = getattr(module, "Model")
    model = model_class(model_config).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    # Prepare validation dataset
    val_dataset = NONBIASDataset(
        real_metadata_path="./Standardized_full_data/REAL_train_and_val_cleaned.csv",
        spoof_metadata_path="./Standardized_full_data/Metadata TTS data_full.csv",
        split="val",
        base_dir="./Standardized_full_data/Val"
    )
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=config["batch_size"], drop_last=False, pin_memory=True)

    # Run inference and save predictions
    y_true, y_scores = [], []
    with torch.no_grad():
        for x, label in val_loader:
            x = x.to(device)
            _, output = model(x)  # 👈 fix here

            if output.shape[-1] == 1:
                score = torch.sigmoid(output).squeeze().cpu().numpy()
            else:
                score = torch.softmax(output, dim=1)[:, 1].cpu().numpy()

            y_true.extend(label.cpu().numpy())
            y_scores.extend(score)

    # Save prediction CSV
    df_out = pd.DataFrame({"label": y_true, "prediction_score": y_scores})
    weight_name = Path(args.weights).stem
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    pred_csv_path = output_path / f"{weight_name}_val_predictions.csv"
    df_out.to_csv(pred_csv_path, index=False)

    # Compute EER using your function
    eer, threshold = compute_eer_from_predictions(pred_csv_path)

    # Save result
    result_path = output_path / f"{weight_name}_EER_threshold.txt"
    with open(result_path, "w") as f:
        f.write(f"EER: {eer:.4f}\n")
        f.write(f"Threshold: {threshold:.4f}\n")

    print(f"✅ EER: {eer:.4f}, Threshold: {threshold:.4f}")
    print(f"📄 Saved to: {result_path}")


if __name__ == "__main__":
    main()
