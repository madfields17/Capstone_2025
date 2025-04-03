import os
import librosa
import soundfile as sf
import numpy as np
from scipy.signal import butter, filtfilt
from pathlib import Path
from tqdm import tqdm
from pydub import AudioSegment
from multiprocessing import Pool, cpu_count
from functools import partial

def convert_to_wav_array(file_path):
    if file_path.lower().endswith(".mp3"):
        audio = AudioSegment.from_mp3(file_path)
        wav_bytes = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2).raw_data
        samples = np.frombuffer(wav_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return samples, 16000
    else:
        return librosa.load(file_path, sr=None)

def resample_audio(y, sr, target_sr):
    return librosa.resample(y, orig_sr=sr, target_sr=target_sr), target_sr

def convert_to_mono(y):
    return librosa.to_mono(y) if y.ndim > 1 else y

def normalize_audio(y):
    return y / np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else y

def trim_silence(y, top_db=40):
    return librosa.effects.trim(y, top_db=top_db)[0]

def trim_duration(y, sr, max_duration=5.0):
    max_len = int(sr * max_duration)
    return y[:max_len] if len(y) > max_len else y

def butter_bandpass_filter(y, lowcut, highcut, sr, order=5):
    nyq = 0.5 * sr
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, y)

def process_file(filename, audio_path, output_dir, target_sr, lowcut, highcut, order):
    input_path = os.path.join(audio_path, filename)
    output_filename = Path(filename).with_suffix(".wav").name
    output_path = os.path.join(output_dir, output_filename)

    if not os.path.isfile(input_path) or os.path.exists(output_path):
        return f"⏭️ Skipping: {output_filename}"

    try:
        y, sr = convert_to_wav_array(input_path)
        if target_sr and sr != target_sr:
            y, sr = resample_audio(y, sr, target_sr)

        y = convert_to_mono(y)
        y = normalize_audio(y)
        y = trim_silence(y)
        y = trim_duration(y, sr)

        if lowcut and highcut:
            y = butter_bandpass_filter(y, lowcut, highcut, sr, order)

        sf.write(output_path, y, sr)
        return f"✅ Saved: {output_filename}"
    except Exception as e:
        return f"❌ Error with {filename}: {e}"

def standardize_audio_parallel(audio_path, target_sr=None, lowcut=None, highcut=None, order=5, output_dir="standardized_audio"):
    if not os.path.isdir(audio_path):
        print("❌ Invalid input path.")
        return

    os.makedirs(output_dir, exist_ok=True)
    file_list = [f for f in os.listdir(audio_path) if os.path.isfile(os.path.join(audio_path, f))]

    process_fn = partial(
        process_file,
        audio_path=audio_path,
        output_dir=output_dir,
        target_sr=target_sr,
        lowcut=lowcut,
        highcut=highcut,
        order=order,
    )

    print(f"🚀 Starting parallel audio standardization using {min(cpu_count(), 5)} workers...")
    with Pool(processes=min(cpu_count(), 5)) as pool:
        for result in tqdm(pool.imap_unordered(process_fn, file_list), total=len(file_list)):
            tqdm.write(result)

    print("🎉 All files processed!")

# === Entrypoint ===

if __name__ == "__main__":
    standardize_audio_parallel(
        audio_path="Unstandardized_full_data/Val",
        target_sr=16000,
        lowcut=300.0,
        highcut=3400.0,
        output_dir="Standardized_full_data/Val"
    )