import numpy as np
import torch
import torchaudio
from pathlib import Path
from torch.utils.data import Dataset

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
        stt = np.random.randint(0, x_len - max_len + 1)  # Ensure high > 0
        return x[stt:stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x

def random_pad(x: np.ndarray, min_len: int = 1025, max_len: int = 64600):
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

class Dataset_Mozilla_RawNet2(Dataset):
    def __init__(self, base_dir):
        """
        Mozilla dataset formatted for RawNet2.
        Ensures minimum length of 1025 samples.
        """
        self.base_dir = Path(base_dir)
        self.file_paths = list(self.base_dir.glob("*.wav"))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        waveform, sample_rate = torchaudio.load(file_path)

        # Convert stereo to mono if needed
        # if waveform.shape[0] > 1:
        #     waveform = waveform.mean(dim=0, keepdim=True)

        waveform = waveform.squeeze(0).numpy()  # Convert to NumPy for `random_pad`

        # Ensure the input is at least 1025 samples long
        # waveform = random_pad(waveform, min_len=1025, max_len=64600)
        waveform = pad_random(waveform, 64600)

        # Convert back to PyTorch tensor
        waveform = torch.tensor(waveform, dtype=torch.float32)

        return waveform, str(file_path), waveform.shape[0]
    
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
    

class Dataset_Mozilla_TSSD(Dataset):
    def __init__(self, base_dir):
        """
        Mozilla dataset formatted for RawNet2.
        Ensures minimum length of 1025 samples.
        """
        self.base_dir = Path(base_dir)
        self.file_paths = list(self.base_dir.glob("*.wav"))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        waveform, sample_rate = torchaudio.load(file_path)

        # Convert stereo to mono if needed
        # if waveform.shape[0] > 1:
        #     waveform = waveform.mean(dim=0, keepdim=True)

        waveform = waveform.squeeze(0).numpy()  # Convert to NumPy for `random_pad`

        # Ensure the input is at least 1025 samples long
        # waveform = random_pad(waveform, min_len=1025, max_len=64600)
        waveform = pad_random(waveform, 96000)

        # Convert back to PyTorch tensor
        waveform = torch.tensor(waveform, dtype=torch.float32)

        return waveform, str(file_path), waveform.shape[0]