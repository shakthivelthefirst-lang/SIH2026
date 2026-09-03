import numpy as np
import scipy.signal


def wiener_filter_enhance(noisy_audio: np.ndarray, sr: int = 16000, n_fft: int = 512, hop_length: int = 128, win_length: int = 512) -> np.ndarray:
    """
    Apply statistical Decision-Directed Wiener Filter speech enhancement.
    Args:
        noisy_audio: 1D numpy array of noisy speech
        sr: Sample rate (Hz)
        n_fft: FFT size
        hop_length: Hop length
        win_length: Window length
    Returns:
        Enhanced 1D audio array of same length
    """
    if len(noisy_audio) == 0:
        return noisy_audio

    orig_len = len(noisy_audio)
    window = scipy.signal.windows.hann(win_length)

    # Compute STFT
    _, _, Zxx = scipy.signal.stft(
        noisy_audio,
        fs=sr,
        window=window,
        nperseg=win_length,
        noverlap=win_length - hop_length,
        nfft=n_fft,
        boundary="zeros",
        padded=True
    )

    mag = np.abs(Zxx)  # [F, T]
    phase = np.angle(Zxx)
    power = mag ** 2
    n_freqs, n_frames = power.shape

    # Initial noise PSD estimation from the first few frames (assumed quiet or initial background)
    init_frames = min(max(5, int(n_frames * 0.1)), 20)
    noise_psd = np.mean(power[:, :init_frames], axis=1, keepdims=True) + 1e-10

    # Decision-directed parameters
    alpha = 0.98  # Smoothing factor
    min_gain = 0.05  # Gain floor to avoid musical noise

    prior_snr = np.zeros((n_freqs, 1))
    enhanced_mag = np.zeros_like(mag)

    for t in range(n_frames):
        curr_power = power[:, t : t + 1]
        post_snr = curr_power / noise_psd

        if t == 0:
            prior_snr = np.maximum(post_snr - 1.0, 0.0)
        else:
            prev_enh_power = (enhanced_mag[:, t - 1 : t]) ** 2
            prior_snr = alpha * (prev_enh_power / noise_psd) + (1.0 - alpha) * np.maximum(post_snr - 1.0, 0.0)

        # Wiener filter gain
        gain = prior_snr / (prior_snr + 1.0 + 1e-10)
        gain = np.maximum(gain, min_gain)

        enhanced_mag[:, t : t + 1] = gain * mag[:, t : t + 1]

        # Update noise PSD using minimum tracking / soft VAD update
        is_speech = np.mean(gain) > 0.35
        if not is_speech:
            noise_psd = 0.95 * noise_psd + 0.05 * curr_power

    # Reconstruct complex STFT
    enhanced_Zxx = enhanced_mag * np.exp(1j * phase)

    # Inverse STFT
    _, enhanced_audio = scipy.signal.istft(
        enhanced_Zxx,
        fs=sr,
        window=window,
        nperseg=win_length,
        noverlap=win_length - hop_length,
        nfft=n_fft,
        boundary="zeros"
    )

    # Match exact original length
    if len(enhanced_audio) > orig_len:
        enhanced_audio = enhanced_audio[:orig_len]
    elif len(enhanced_audio) < orig_len:
        enhanced_audio = np.pad(enhanced_audio, (0, orig_len - len(enhanced_audio)))

    return enhanced_audio.astype(np.float32)
