import torchaudio
import torchaudio.transforms as T
import os
from pathlib import Path
import shutil

# Define input and output directories
input_dir = Path("aasist_env/Mozilla_CV_Eval_Set/eval")
output_dir = Path("aasist_env/mozilla_evaluation_wav")

# Create the output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# Target sample rate for conversion
TARGET_SAMPLE_RATE = 16000  # Standard for ASVspoof

def convert_mp3_to_wav(mp3_path, wav_path):
    """Convert an MP3 file to WAV format using torchaudio."""
    waveform, sample_rate = torchaudio.load(mp3_path)
    
    # Resample if needed
    if sample_rate != TARGET_SAMPLE_RATE:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE)
        waveform = resampler(waveform)
    
    # Save as WAV
    torchaudio.save(wav_path, waveform, TARGET_SAMPLE_RATE)
    print(f"Converted: {mp3_path} → {wav_path}")

# Recursively process files in the input directory
for file in input_dir.rglob("*"):
    if file.is_file():
        if file.suffix.lower() == ".mp3":
            # Convert MP3 to WAV (store in a flat structure)
            wav_output_path = output_dir / f"{file.stem}.wav"
            convert_mp3_to_wav(file, wav_output_path)
        elif file.suffix.lower() == ".wav":
            # Copy WAV files to the new directory
            wav_output_path = output_dir / file.name
            shutil.copy2(file, wav_output_path)
            print(f"Copied WAV: {file} → {wav_output_path}")

print("✅ All files processed: MP3 files converted to WAV and WAV files copied.")