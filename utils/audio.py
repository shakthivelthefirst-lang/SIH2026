import os
import random
import numpy as np
import soundfile as sf
import librosa
import torch


def load_audio(file_path: str, target_sr: int = 16000) -> np.ndarray:
    """
    Load an audio file, convert to mono, resample to target_sr, and cast to float32.
    Handles corrupted files by returning zeros.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    try:
        audio, sr = sf.read(file_path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)

        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

        # Remove DC offset
        audio = audio - np.mean(audio)
        return audio.astype(np.float32)
    except Exception as e:
        print(f"Warning: Failed to load {file_path} ({e}). Returning 1-sec silence.")
        return np.zeros(target_sr, dtype=np.float32)


def save_audio(file_path: str, audio: np.ndarray, sr: int = 16000):
    """
    Save 1D numpy array as a WAV file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    # Clip safely to [-1.0, 1.0]
    audio = np.clip(audio, -1.0, 1.0)
    sf.write(file_path, audio.astype(np.float32), sr)


def calculate_rms(signal: np.ndarray, eps: float = 1e-8) -> float:
    """
    Calculate Root Mean Square (RMS) of a 1D audio signal.
    """
    return float(np.sqrt(np.mean(signal ** 2) + eps))


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """
    Dynamically mix clean speech with noise at a specified SNR (dB).
    Formula:
        SNR = 10 * log10(P_clean / P_noise) = 20 * log10(RMS_clean / RMS_noise)
        RMS_target_noise = RMS_clean / 10^(SNR / 20)
        scaled_noise = noise * (RMS_target_noise / (RMS_noise + eps))
        noisy = clean + scaled_noise
    Both clean and noisy are safely normalized by the maximum peak to prevent clipping.
    """
    clean_len = len(clean)
    noise_len = len(noise)

    if noise_len < clean_len:
        # Loop noise if it is shorter than clean speech
        repeats = int(np.ceil(clean_len / max(1, noise_len)))
        noise = np.tile(noise, repeats)[:clean_len]
    elif noise_len > clean_len:
        # Random crop of noise
        start = random.randint(0, noise_len - clean_len)
        noise = noise[start : start + clean_len]

    clean_rms = calculate_rms(clean, eps=eps)
    noise_rms = calculate_rms(noise, eps=eps)

    target_noise_rms = clean_rms / (10.0 ** (snr_db / 20.0))
    scaled_noise = noise * (target_noise_rms / (noise_rms + eps))
    noisy = clean + scaled_noise

    # Safe peak normalization to [-0.95, 0.95] if clipping would occur
    max_peak = max(np.max(np.abs(noisy)), np.max(np.abs(clean)))
    if max_peak > 0.95:
        norm_factor = 0.95 / max_peak
        clean = clean * norm_factor
        noisy = noisy * norm_factor

    return clean.astype(np.float32), noisy.astype(np.float32)


def extract_segment(audio: np.ndarray, target_length: int) -> np.ndarray:
    """
    Extract a random or centered segment of target_length samples.
    Pads with zeros if audio is shorter than target_length.
    """
    curr_len = len(audio)
    if curr_len < target_length:
        pad_len = target_length - curr_len
        return np.pad(audio, (0, pad_len), mode="constant")
    elif curr_len > target_length:
        start = random.randint(0, curr_len - target_length)
        return audio[start : start + target_length]
    return audio
