import torchaudio
import torchaudio.transforms as T
import os
from pathlib import Path

# Define input and output directories
mp3_dir = Path("aasist_env/Mozilla_CV_Eval_Set")
wav_dir = Path("aasist_env/mozilla_evaluation_wav")

# Create the output directory if it doesn't exist
wav_dir.mkdir(parents=True, exist_ok=True)

# Target sample rate for conversion
TARGET_SAMPLE_RATE = 16000  # Standard for ASVspoof

def convert_mp3_to_wav(mp3_path, wav_path):
    """Convert an MP3 file to WAV format using torchaudio."""
    waveform, sample_rate = torchaudio.load(mp3_path)

    # Resample if needed
    if sample_rate != TARGET_SAMPLE_RATE:
        resampler = T.Resample(sample_rate, TARGET_SAMPLE_RATE)
        waveform = resampler(waveform)

    # Save as WAV
    torchaudio.save(wav_path, waveform, TARGET_SAMPLE_RATE)
    print(f"Converted: {mp3_path} → {wav_path}")

# Recursively find all MP3 files and convert them, storing in a flat directory
for mp3_file in mp3_dir.rglob("*.mp3"):
    wav_output_path = wav_dir / f"{mp3_file.stem}.wav"  # No subdirectories

    # Convert MP3 to WAV
    convert_mp3_to_wav(mp3_file, wav_output_path)

print("✅ All MP3 files converted to WAV format!")