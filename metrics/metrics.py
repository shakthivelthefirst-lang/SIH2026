import numpy as np
import torch

# STOI import
try:
    from pystoi import stoi
    HAS_STOI = True
except ImportError:
    HAS_STOI = False

# PESQ import with graceful fallback
try:
    from pesq import pesq
    HAS_PESQ = True
except ImportError:
    HAS_PESQ = False


def calculate_snr(clean: np.ndarray, estimated: np.ndarray, eps: float = 1e-8) -> float:
    """
    Calculate conventional Signal-to-Noise Ratio (SNR) in dB.
    Formula: SNR = 10 * log10( ||s||^2 / (||s - s_hat||^2 + eps) )
    """
    clean = np.asarray(clean, dtype=np.float64)
    estimated = np.asarray(estimated, dtype=np.float64)

    min_len = min(len(clean), len(estimated))
    clean = clean[:min_len]
    estimated = estimated[:min_len]

    noise = clean - estimated
    clean_power = np.sum(clean ** 2)
    noise_power = np.sum(noise ** 2) + eps

    return float(10.0 * np.log10((clean_power + eps) / noise_power))


def calculate_sisnr_metric(clean: np.ndarray, estimated: np.ndarray, eps: float = 1e-8) -> float:
    """
    Calculate SI-SNR in dB for numpy 1D arrays.
    """
    clean = np.asarray(clean, dtype=np.float64)
    estimated = np.asarray(estimated, dtype=np.float64)

    min_len = min(len(clean), len(estimated))
    clean = clean[:min_len]
    estimated = estimated[:min_len]

    # Zero-mean
    clean = clean - np.mean(clean)
    estimated = estimated - np.mean(estimated)

    dot = np.sum(clean * estimated)
    clean_energy = np.sum(clean ** 2) + eps
    s_target = (dot / clean_energy) * clean
    e_noise = estimated - s_target

    target_norm = np.sum(s_target ** 2) + eps
    noise_norm = np.sum(e_noise ** 2) + eps

    return float(10.0 * np.log10(target_norm / noise_norm))


def calculate_stoi(clean: np.ndarray, estimated: np.ndarray, sr: int = 16000) -> float:
    """
    Calculate Short-Time Objective Intelligibility (STOI) score [0, 1].
    """
    if not HAS_STOI:
        return np.nan

    min_len = min(len(clean), len(estimated))
    if min_len < sr * 0.2:  # Minimum length check
        return np.nan

    try:
        score = stoi(clean[:min_len], estimated[:min_len], sr, extended=False)
        return float(score)
    except Exception:
        return np.nan


def calculate_pesq(clean: np.ndarray, estimated: np.ndarray, sr: int = 16000) -> float:
    """
    Calculate Perceptual Evaluation of Speech Quality (PESQ) score [-0.5, 4.5].
    Requires 16000 Hz or 8000 Hz.
    """
    if not HAS_PESQ:
        return np.nan

    min_len = min(len(clean), len(estimated))
    if min_len < sr * 0.5:  # PESQ requires at least ~0.5s
        return np.nan

    mode = "wb" if sr == 16000 else "nb"
    try:
        score = pesq(sr, clean[:min_len], estimated[:min_len], mode)
        return float(score)
    except Exception:
        return np.nan


def evaluate_all_metrics(clean: np.ndarray, estimated: np.ndarray, noisy: np.ndarray, sr: int = 16000) -> dict:
    """
    Evaluate all metrics (SNR, SI-SNR, STOI, PESQ) for clean, noisy, and enhanced signals.
    """
    snr_noisy = calculate_snr(clean, noisy)
    snr_enh = calculate_snr(clean, estimated)
    delta_snr = snr_enh - snr_noisy

    sisnr_noisy = calculate_sisnr_metric(clean, noisy)
    sisnr_enh = calculate_sisnr_metric(clean, estimated)
    delta_sisnr = sisnr_enh - sisnr_noisy

    stoi_noisy = calculate_stoi(clean, noisy, sr=sr)
    stoi_enh = calculate_stoi(clean, estimated, sr=sr)

    pesq_noisy = calculate_pesq(clean, noisy, sr=sr)
    pesq_enh = calculate_pesq(clean, estimated, sr=sr)

    return {
        "Noisy_SNR": snr_noisy,
        "Enhanced_SNR": snr_enh,
        "Delta_SNR": delta_snr,
        "Noisy_SI_SNR": sisnr_noisy,
        "Enhanced_SI_SNR": sisnr_enh,
        "Delta_SI_SNR": delta_sisnr,
        "Noisy_STOI": stoi_noisy,
        "Enhanced_STOI": stoi_enh,
        "Noisy_PESQ": pesq_noisy,
        "Enhanced_PESQ": pesq_enh,
    }
