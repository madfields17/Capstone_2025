"""
Main script that trains, validates, and evaluates
various models including AASIST.

AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import argparse
import json
import os
import sys
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA
from tqdm import tqdm

from data_utils import (NONBIASDataset,
                        Dataset_Mozilla
                        )

from evaluation import calculate_tDCF_EER, compute_eer
from utils import create_optimizer, seed_worker, set_seed, str_to_bool

warnings.filterwarnings("ignore", category=FutureWarning)

def main(args: argparse.Namespace) -> None:
    """
    Main function.
    Trains, validates, and evaluates the ASVspoof detection model.
    """
    # load experiment configurations
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())

    print("✅ Loaded config from:", args.config)
    print("🧩 Config keys:", list(config.keys()))
    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]
    track = config["track"]
    assert track in ["NB"], "Invalid track given"
    if "eval_all_best" not in config:
        config["eval_all_best"] = "True"
    if "freq_aug" not in config:
        config["freq_aug"] = "False"

    # make experiment reproducible
    set_seed(args.seed, config)

    # define database related paths
    output_dir = Path(args.output_dir)
    database_path = Path(config["database_path"])
    # dev_trial_path = 
    # eval_trial_path = 

    # define model related paths
    model_tag = "{}_{}_ep{}_bs{}".format(
        track,
        os.path.splitext(os.path.basename(args.config))[0],
        config["num_epochs"], config["batch_size"])
    if args.comment:
        model_tag = model_tag + "_{}".format(args.comment)
    model_tag = output_dir / model_tag
    model_save_path = model_tag / "weights"
    eval_score_path = model_tag / "mozilla_evaluation_results.csv"  # Ensure CSV format
    writer = SummaryWriter(model_tag)
    os.makedirs(model_save_path, exist_ok=True)
    copy(args.config, model_tag / "config.conf")

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    # if device == "cpu":
    #     raise ValueError("GPU not detected!")
    
    def get_model(model_config, device):
        module = import_module("models.{}".format(model_config["architecture"]))
        model_class = getattr(module, "Model")
        model = model_class(model_config).to(device)
        model.load_state_dict(torch.load(config["model_path"], map_location=device))  # <- We'll update this too
        return model

    # define model architecture
    model = get_model(model_config, device)


    train_dataset = NONBIASDataset(
        real_metadata_path="./Standardized_full_data/REAL_train_and_val_cleaned.csv",
        spoof_metadata_path="./Standardized_full_data/Metadata TTS data_full.csv",
        split="train",
        base_dir="./Standardized_full_data/Training_test"
    )

    val_dataset = NONBIASDataset(
        real_metadata_path="./Standardized_full_data/REAL_train_and_val_cleaned.csv",
        spoof_metadata_path="./Standardized_full_data/Metadata TTS data_full.csv",
        split="val",
        base_dir="./Standardized_full_data/Val_test"
    )

    gen = torch.Generator()
    gen.manual_seed(args.seed)
    trn_loader = DataLoader(train_dataset, shuffle=True, batch_size=config["batch_size"], drop_last=True, pin_memory=True, worker_init_fn=seed_worker, generator=gen)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=config["batch_size"], drop_last=False, pin_memory=True)
    
    # DEBUG: Check which files are expected vs. available
    expected_files = val_dataset.metadata["file_name"]
    available_files = pd.Series(os.listdir(val_dataset.base_dir))
    available_files = available_files[available_files.str.endswith(".wav")]

    # What's in metadata but not in the folder
    missing_files = expected_files[~expected_files.isin(available_files)]
    # What's in the folder but not in metadata
    extra_files = available_files[~available_files.isin(expected_files)]

    # For train set
    print(f"\n🧪 Number of training samples: {len(train_dataset)}")
    print(f"📦 Number of training batches: {len(trn_loader)}")

    train_labels = pd.Series([label for _, label in train_dataset])
    num_real_train = (train_labels == 1).sum()
    num_spoof_train = (train_labels == 0).sum()
    print(f"👤 Real samples in train: {num_real_train}")
    print(f"👻 Spoof samples in train: {num_spoof_train}")

    # For val set
    print(f"\n🧪 Number of validation samples: {len(val_dataset)}")
    print(f"📦 Number of validation batches: {len(val_loader)}")

    val_labels = pd.Series([label for _, label in val_dataset])
    num_real_val = (val_labels == 1).sum()
    num_spoof_val = (val_labels == 0).sum()
    print(f"👤 Real samples in val: {num_real_val}")
    print(f"👻 Spoof samples in val: {num_spoof_val}")

    # File checks (assumes this is run inside the Dataset class or you have the data available)
    print(f"\n📂 Files listed in metadata: {len(expected_files)}")
    print(f"📁 Files actually in folder: {len(available_files)}")
    print(f"🚫 Missing files from folder: {len(missing_files)}")
    print(f"➕ Extra files not in metadata: {len(extra_files)}")

    # Optional: print some sample filenames
    if not missing_files.empty:
        print("\n❗ Missing file examples:")
        print(missing_files.head())

    if not extra_files.empty:
        print("\n📎 Extra file examples:")
        print(extra_files.head())

    eval_set = Dataset_Mozilla(base_dir="mozilla_evaluation_wav", use_random_pad=True)  # Target the WAV files
    eval_loader = DataLoader(eval_set,
                             batch_size=config["batch_size"],
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True)

    # CHANGE FOR TRAINED MODEL
    # evaluates pretrained model and exit script
    if args.eval:
        model.load_state_dict(
            torch.load(config["model_path"], map_location=device))
        print("Model loaded : {}".format(config["model_path"]))
        print("Start evaluation...")
        produce_evaluation_file(eval_loader, model, device,
                                eval_score_path)
        calculate_tDCF_EER(cm_scores_file=eval_score_path,
                           asv_score_file=database_path /
                           config["asv_score_path"],
                           output_file=model_tag / "t-DCF_EER.txt")
        print("DONE.")
        # eval_eer, eval_tdcf = calculate_tDCF_EER(
        #     cm_scores_file=eval_score_path,
        #     asv_score_file=database_path / config["asv_score_path"],
        #     output_file=model_tag/"loaded_model_t-DCF_EER.txt")
        sys.exit(0)

    # get optimizer and scheduler
    optim_config["steps_per_epoch"] = len(trn_loader)
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
    optimizer_swa = SWA(optimizer)

    # Change as we see fit
    best_dev_eer = 1.
    # best_eval_eer = 100.
    # best_dev_tdcf = 0.05
    # best_eval_tdcf = 1.
    n_swa_update = 0  # number of snapshots of model to use in SWA
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")

    # make directory for metric logging
    metric_path = model_tag / "metrics"
    os.makedirs(metric_path, exist_ok=True)

    # Training
    for epoch in range(config["num_epochs"]):
        print("Start training epoch{:03d}".format(epoch))
        running_loss = train_epoch(trn_loader, model, optimizer, device,
                                   scheduler, config)
        produce_evaluation_file(val_loader, model, device,
                                metric_path/"val_score.txt")
        
        # might need to CHANGE!
        # val_eer, val_tdcf = calculate_tDCF_EER(
        #     cm_scores_file=metric_path/"val_score.txt",
        #     asv_score_file=database_path/config["asv_score_path"],
        #     output_file=metric_path/"dev_t-DCF_EER_{}epo.txt".format(epoch),
        #     printout=False)
        val_eer = compute_eer_from_predictions(metric_path/"val_score.txt")
        
        print("DONE.\nLoss:{:.5f}, dev_eer: {:.3f}".format(
            running_loss, val_eer))
        
        writer.add_scalar("loss", running_loss, epoch)
        writer.add_scalar("val_eer", val_eer, epoch)
        # writer.add_scalar("val_tdcf", val_tdcf, epoch)

        # best_dev_tdcf = min(val_tdcf, best_dev_tdcf)
        if best_dev_eer >= val_eer:
            print("best model find at epoch", epoch)
            best_dev_eer = val_eer
            torch.save(model.state_dict(),
                       model_save_path / "epoch_{}_{:03.3f}.pth".format(epoch, val_eer))
        
            # # do evaluation whenever best model is renewed
            # if str_to_bool(config["eval_all_best"]):
            #     produce_evaluation_file(eval_loader, model, device,
            #                             eval_score_path)
            #     eval_eer, eval_tdcf = calculate_tDCF_EER(
            #         cm_scores_file=eval_score_path,
            #         asv_score_file=database_path / config["asv_score_path"],
            #         output_file=metric_path /
            #         "t-DCF_EER_{:03d}epo.txt".format(epoch)
            #         )
            #     log_text = "epoch{:03d}, ".format(epoch)
            #     if eval_eer < best_eval_eer:
            #         log_text += "best eer, {:.4f}%".format(eval_eer)
            #         best_eval_eer = eval_eer
            #     if eval_tdcf < best_eval_tdcf:
            #         log_text += "best tdcf, {:.4f}".format(eval_tdcf)
            #         best_eval_tdcf = eval_tdcf
            #         torch.save(model.state_dict(),
            #                 model_save_path / "best.pth")
            #     if len(log_text) > 0:
            #         print(log_text)
            #         f_log.write(log_text + "\n")

            print("Saving epoch {} for swa".format(epoch))  
            optimizer_swa.update_swa()
            n_swa_update += 1
        writer.add_scalar("best_dev_eer", best_dev_eer, epoch)
        # writer.add_scalar("best_dev_tdcf", best_dev_tdcf, epoch)

    print("Start final evaluation")
    epoch += 1
    if n_swa_update > 0:
        optimizer_swa.swap_swa_sgd()
        optimizer_swa.bn_update(trn_loader, model, device=device)

        # Re-evaluate on val using SWA model
        produce_evaluation_file(val_loader, model, device, metric_path / "val_score_swa.txt")
        final_val_eer = compute_eer_from_predictions(metric_path / "val_score_swa.txt")
        print(f"📌 Final SWA EER: {final_val_eer:.3f}")
    # produce_evaluation_file(eval_loader, model, device, eval_score_path,
    #                         eval_trial_path)
    # eval_eer, eval_tdcf = calculate_tDCF_EER(cm_scores_file=eval_score_path,
    #                                             asv_score_file=database_path /
    #                                             config["asv_score_path"],
    #                                             output_file=model_tag / "t-DCF_EER.txt")
    # f_log = open(model_tag / "metric_log.txt", "a")
    # f_log.write("=" * 5 + "\n")
    # # f_log.write("EER: {:.3f}, min t-DCF: {:.5f}".format(eval_eer, eval_tdcf))
    # f_log.close()

    torch.save(model.state_dict(),
               model_save_path / "swa.pth")
    
    # if eval_eer <= best_eval_eer:
    #     best_eval_eer = eval_eer
    # if eval_tdcf <= best_eval_tdcf:
    #     best_eval_tdcf = eval_tdcf
    #     torch.save(model.state_dict(),
    #                model_save_path / "best.pth")
    # print("Exp FIN. EER: {:.3f}, min t-DCF: {:.5f}".format(
    #     best_eval_eer, best_eval_tdcf))
    
    # UPDATE trial path?
# def produce_evaluation_file(
#         data_loader: DataLoader,
#         model,
#         device: torch.device,
#         save_path: str,
#         trial_path: str) -> None:
#         """Perform evaluation and save the score to a file"""
#         model.eval()
#         with open(trial_path, "r") as f_trl:
#             trial_lines = f_trl.readlines()
#         fname_list = []
#         score_list = []
#         for batch_x, utt_id in data_loader:
#             batch_x = batch_x.to(device)
#             with torch.no_grad():
#                 _, batch_out = model(batch_x)
#                 batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
#             # add outputs
#             fname_list.extend(utt_id)
#             score_list.extend(batch_score.tolist())

#         assert len(trial_lines) == len(fname_list) == len(score_list)
#         with open(save_path, "w") as fh:
#             for fn, sco, trl in zip(fname_list, score_list, trial_lines):
#                 _, utt_id, _, src, key = trl.strip().split(' ')
#                 assert fn == utt_id
#                 fh.write("{} {} {} {}\n".format(utt_id, src, key, sco))
#         print("Scores saved to {}".format(save_path))

def produce_evaluation_file(data_loader, model, device, save_path):
    model.eval()
    score_data = []

    for batch_x, batch_y in data_loader:
        batch_x = batch_x.to(device)
        with torch.no_grad():
            _, batch_out = model(batch_x)
            batch_scores = batch_out[:, 1].cpu().numpy().ravel()  # logits

        for label, score in zip(batch_y.numpy(), batch_scores):
            score_data.append((int(label), score))

    df = pd.DataFrame(score_data, columns=["label", "prediction_score"])
    df.to_csv(save_path, index=False)
    print(f"✅ Logit scores saved to {save_path}")

def train_epoch(trn_loader: DataLoader, model, optimizer, device, scheduler, config: argparse.Namespace):
    """Train the model for one epoch"""
    running_loss = 0.0
    num_total = 0.0
    model.train()

    # Set loss function with class weights
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    progress_bar = tqdm(trn_loader, desc="Training", leave=True)
    for batch_x, batch_y in progress_bar:
        batch_size = batch_x.size(0)
        num_total += batch_size

        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        # Freq_aug flag from config if applicable
        _, batch_out = model(batch_x, Freq_aug=str_to_bool(config["freq_aug"]))
        batch_loss = criterion(batch_out, batch_y)
        running_loss += batch_loss.item() * batch_size

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
            scheduler.step()
        elif scheduler is None:
            pass
        else:
            raise ValueError("scheduler error, got:{}".format(scheduler))
        
        progress_bar.set_postfix(loss=batch_loss.item())

    running_loss /= num_total
    return running_loss

def compute_eer_from_predictions(score_file_path):
    df = pd.read_csv(score_file_path)
    scores = df["prediction_score"].values
    labels = df["label"].values

    target_scores = scores[labels == 1]  # real
    spoof_scores = scores[labels == 0]  # spoof

    if len(target_scores) == 0 or len(spoof_scores) == 0:
        print("⚠️ Not enough spoof or real samples to compute EER")
        return float("nan")

    eer, _ = compute_eer(target_scores, spoof_scores)
    return eer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FairVoice detection system")
    parser.add_argument("--config",
                        dest="config",
                        type=str,
                        help="configuration file",
                        required=True)
    
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        help="output directory for results",
        default="./exp_result",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="random seed (default: 1234)")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="when this flag is given, evaluates given model and exit")
    parser.add_argument("--comment",
                        type=str,
                        default=None,
                        help="comment to describe the saved model")
    parser.add_argument("--eval_model_weights",
                        type=str,
                        default=None,
                        help="directory to the model weight file (can be also given in the config file)")
    main(parser.parse_args())