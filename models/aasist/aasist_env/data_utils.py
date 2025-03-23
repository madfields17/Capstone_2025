import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import Dataset
from pathlib import Path
import torchaudio


___author__ = "Hemlata Tak, Jee-weon Jung"
__email__ = "tak@eurecom.fr, jeeweon.jung@navercorp.com"


def genSpoof_list(dir_meta, is_train=False, is_eval=False):

    d_meta = {}
    file_list = []
    with open(dir_meta, "r") as f:
        l_meta = f.readlines()

    if is_train:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list

    elif is_eval:
        for line in l_meta:
            _, key, _, _, _ = line.strip().split(" ")
            #key = line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len >= max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x


class Dataset_ASVspoof2019_train(Dataset):
    def __init__(self, list_IDs, labels, base_dir):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)"""
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = sf.read(str(self.base_dir / f"flac/{key}.flac"))
        X_pad = pad_random(X, self.cut)
        x_inp = Tensor(X_pad)
        y = self.labels[key]
        return x_inp, y


class Dataset_ASVspoof2019_devNeval(Dataset):
    def __init__(self, list_IDs, base_dir):
        """self.list_IDs	: list of strings (each string: utt key),
        """
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = sf.read(str(self.base_dir / f"flac/{key}.flac"))
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, key
    
def random_pad_new(x: np.ndarray, min_len: int = 1025, max_len: int = 64600):
    """
    Pads or clips input audio `x` to ensure it's at least `min_len` samples.
    - If `x` is shorter than `min_len`, it is repeated until it reaches `min_len`.
    - If `x` is longer than `max_len`, it gets clipped.
    """
    x_len = x.shape[0]

    if x_len >= min_len:
        return x[:max_len]  # Trim to max_len if necessary

    # Repeat and trim to match min_len
    num_repeats = int(np.ceil(min_len / x_len))
    padded_x = np.tile(x, num_repeats)[:min_len]

    return padded_x



class Dataset_Mozilla(Dataset):
    def __init__(self, base_dir, use_random_pad=False):
        """
        Dataset for Mozilla evaluation set, assuming 16kHz WAV files.
        Args:
            base_dir (str): Directory containing WAV files.
            use_random_pad (bool): If True, uses random padding instead of fixed padding.
        """
        self.base_dir = Path(base_dir)
        self.file_paths = list(self.base_dir.glob("*.wav"))  # Load WAV files
        self.cut = 64600  # Target length (~4 sec at 16kHz)
        self.use_random_pad = use_random_pad

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        # Load the WAV file (assumed to be 16kHz already)
        waveform, sample_rate = torchaudio.load(file_path)

        # Convert waveform to numpy for padding function
        waveform = waveform.squeeze(0).numpy()  # Convert from torch tensor to numpy

        # Apply padding
        if self.use_random_pad:
            waveform = pad_random(waveform, self.cut)
        else:
            waveform = pad(waveform, self.cut)

        # Convert back to PyTorch tensor
        waveform = torch.tensor(waveform, dtype=torch.float32)

        return waveform, str(file_path)
