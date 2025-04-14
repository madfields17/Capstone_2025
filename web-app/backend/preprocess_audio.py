import io
import numpy as np
import librosa

from pydub import AudioSegment
from scipy.signal import butter, filtfilt

def convert_to_wav_array(file):
    if file.filename.lower().endswith(".mp3"):
        audio = AudioSegment.from_mp3(file)
        wav_bytes = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2).raw_data
        samples = np.frombuffer(wav_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return samples, 16000
    else:
        audio_data = file.read()
        audio_stream = io.BytesIO(audio_data)
        samples, sr = librosa.load(audio_stream, sr=16000, mono=True) 
        return samples, sr

def resample_audio(y, sr, target_sr=16000):
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

def standardize_audio_from_bytes(audio_file):
    y, sr = convert_to_wav_array(audio_file) 
    if sr != 16000:
        y, sr = resample_audio(y, sr, 16000)
    y = convert_to_mono(y)
    y = normalize_audio(y)
    y = trim_silence(y)
    y = trim_duration(y, sr)
    y = butter_bandpass_filter(y, 300.0, 3400.0, sr)
    return y.copy()
