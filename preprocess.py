#!/usr/bin/env python3
"""
Hardware-Accelerated Speech Enhancement Pipeline - Preprocessing Module
Author: Audio DSP & Python Engineer
Reference: SPECIFICATION.md (SPEC-ML-RTL-001)

Features:
- Scans clean speech commands and isolated noise recordings (gun, helicopter, bomb).
- Resamples incoming .wav audio to 16,000 Hz mono and peak-normalizes to [-1.0, 1.0].
- Generates on-the-fly synthetic noisy mixtures at randomized SNR levels (-5 dB, 0 dB, 5 dB, 10 dB).
- Aligns duration by slicing or looping the noise vector to match speech length.
- Computes STFT with 512-point periodic Hann window and 256-sample hop (50% overlap).
- Extracts Input X (first 32 magnitude bins: 0 Hz to ~1000 Hz).
- Computes Ground Truth Target Y: 16-element Ideal Ratio Mask (IRM) clipped to [0.0, 1.0].
- Partitions data into Train (80%) and Validation (20%) sets.
- Exports compressed .npz archives containing X: [N, 32] and Y: [N, 16].
"""

import argparse
import math
import os
import sys
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# Attempt to import soundfile and scipy; fallbacks provided where possible
try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    import scipy.signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ==============================================================================
# Audio I/O & DSP Front-End
# ==============================================================================

def load_audio_wave_fallback(file_path: Path) -> Tuple[np.ndarray, int]:
    """Fallback standard-library WAV reader if soundfile is not present."""
    with wave.open(str(file_path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)

    dtype_map = {1: np.uint8, 2: np.int16, 3: None, 4: np.int32}
    if sampwidth == 3:  # 24-bit PCM
        a = np.frombuffer(raw_bytes, dtype=np.uint8)
        b = np.zeros((len(a) // 3, 4), dtype=np.uint8)
        b[:, 1:] = a.reshape(-1, 3)
        audio = b.view(np.int32).flatten() / 2147483648.0
    elif sampwidth in dtype_map and dtype_map[sampwidth] is not None:
        dtype = dtype_map[sampwidth]
        audio = np.frombuffer(raw_bytes, dtype=dtype).astype(np.float32)
        if sampwidth == 1:
            audio = (audio - 128.0) / 128.0
        elif sampwidth == 2:
            audio = audio / 32768.0
        elif sampwidth == 4:
            audio = audio / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels)
    return audio, framerate


def resample_signal(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample 1D audio signal to target_sr using polyphase filtering or linear interpolation."""
    if orig_sr == target_sr:
        return audio.astype(np.float32)

    if HAS_SCIPY:
        gcd = math.gcd(int(orig_sr), int(target_sr))
        up = int(target_sr // gcd)
        down = int(orig_sr // gcd)
        resampled = scipy.signal.resample_poly(audio, up, down)
        return resampled.astype(np.float32)
    else:
        # High-performance linear interpolation fallback
        orig_len = len(audio)
        target_len = int(round(orig_len * (target_sr / orig_sr)))
        orig_indices = np.linspace(0, orig_len - 1, orig_len, endpoint=True)
        target_indices = np.linspace(0, orig_len - 1, target_len, endpoint=True)
        resampled = np.interp(target_indices, orig_indices, audio)
        return resampled.astype(np.float32)


def load_and_normalize_audio(file_path: Path, target_sr: int = 16000) -> np.ndarray:
    """
    Load an audio file, convert to mono, resample to target_sr, and peak-normalize to [-1.0, 1.0].
    
    Parameters:
        file_path: Path to the .wav audio file.
        target_sr: Target sampling rate (default 16,000 Hz).
        
    Returns:
        1D float32 numpy array normalized to [-1.0, 1.0].
    """
    if HAS_SOUNDFILE:
        data, sr = sf.read(str(file_path), dtype="float32", always_2d=False)
    else:
        data, sr = load_audio_wave_fallback(file_path)

    # Convert multi-channel to mono
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Resample to 16 kHz
    if sr != target_sr:
        data = resample_signal(data, orig_sr=sr, target_sr=target_sr)

    # Peak normalization to [-1.0, 1.0]
    peak = float(np.max(np.abs(data)))
    if peak > 1e-8:
        data = data / peak
    else:
        data = np.zeros_like(data, dtype=np.float32)

    return data.astype(np.float32)


# ==============================================================================
# Synthetic Mixing & SNR Control
# ==============================================================================

def match_and_scale_noise(
    clean: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit noise duration to clean speech by looping or slicing, then scale to target SNR.
    
    Returns:
        (y_noisy, clean_scaled, noise_scaled): all 1D float32 arrays of identical length.
    """
    clean_len = len(clean)
    noise_len = len(noise)

    if noise_len < clean_len:
        # Loop noise to exceed clean length, then slice
        repeats = int(math.ceil(clean_len / max(noise_len, 1)))
        tiled_noise = np.tile(noise, repeats)
        noise_matched = tiled_noise[:clean_len]
    else:
        # Randomly slice a segment of noise matching speech duration
        max_start = noise_len - clean_len
        start_idx = rng.integers(0, max_start + 1) if max_start > 0 else 0
        noise_matched = noise[start_idx : start_idx + clean_len]

    # Calculate average signal powers
    clean_pwr = float(np.mean(clean ** 2))
    noise_pwr = float(np.mean(noise_matched ** 2))

    # Calculate scale factor for target SNR: SNR = 10 * log10(P_clean / P_noise_scaled)
    if noise_pwr > 1e-12 and clean_pwr > 1e-12:
        target_noise_pwr = clean_pwr / (10.0 ** (snr_db / 10.0))
        scale = math.sqrt(target_noise_pwr / noise_pwr)
    else:
        scale = 0.0

    noise_scaled = noise_matched * scale
    y_noisy = clean + noise_scaled

    # Peak normalization safeguard to prevent audio clipping beyond [-1.0, 1.0]
    max_peak = float(np.max(np.abs(y_noisy)))
    if max_peak > 1.0:
        y_noisy = y_noisy / max_peak
        clean = clean / max_peak
        noise_scaled = noise_scaled / max_peak

    return y_noisy.astype(np.float32), clean.astype(np.float32), noise_scaled.astype(np.float32)


# ==============================================================================
# Feature & Mask Extraction (DSP Pipeline)
# ==============================================================================

def extract_stft_features_and_mask(
    y_noisy: np.ndarray,
    clean: np.ndarray,
    noise: np.ndarray,
    n_fft: int = 512,
    hop_length: int = 256,
    frames_per_file: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract magnitude features and Ideal Ratio Mask (IRM) targets according to SPECIFICATION.md.
    
    Parameters:
        y_noisy: 1D mixture signal.
        clean: 1D clean speech signal.
        noise: 1D scaled noise signal.
        n_fft: FFT length (512 points).
        hop_length: Frame stride (256 samples, 50% overlap).
        frames_per_file: Optional cap on frames extracted per audio file.
        rng: Random generator instance.
        
    Returns:
        X: [N_frames, 32] float32 array (Bins 0 to 31 magnitude spectrum).
        Y: [N_frames, 16] float32 array (Bins 0 to 15 Ideal Ratio Mask).
    """
    sig_len = len(y_noisy)
    if sig_len < n_fft:
        # Zero-pad signal to at least one full FFT window
        pad_len = n_fft - sig_len
        y_noisy = np.pad(y_noisy, (0, pad_len))
        clean = np.pad(clean, (0, pad_len))
        noise = np.pad(noise, (0, pad_len))
        sig_len = len(y_noisy)

    n_frames = 1 + (sig_len - n_fft) // hop_length
    if n_frames <= 0:
        return np.empty((0, 32), dtype=np.float32), np.empty((0, 16), dtype=np.float32)

    # Periodic Hann window: w[n] = 0.5 * (1 - cos(2*pi*n / N))
    hann_window = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n_fft, dtype=np.float32) / n_fft))

    # Frame extraction via strided slicing
    shape = (n_frames, n_fft)
    strides = (y_noisy.strides[0] * hop_length, y_noisy.strides[0])
    
    y_frames = np.lib.stride_tricks.as_strided(y_noisy, shape=shape, strides=strides)
    s_frames = np.lib.stride_tricks.as_strided(clean, shape=shape, strides=strides)
    n_frames_arr = np.lib.stride_tricks.as_strided(noise, shape=shape, strides=strides)

    # Windowing
    y_win = y_frames * hann_window
    s_win = s_frames * hann_window
    n_win = n_frames_arr * hann_window

    # Compute Real-to-Complex FFT (257 positive frequency bins)
    Y_spec = np.abs(np.fft.rfft(y_win, n=n_fft, axis=-1))  # [n_frames, 257]
    S_spec = np.abs(np.fft.rfft(s_win, n=n_fft, axis=-1))  # [n_frames, 257]
    N_spec = np.abs(np.fft.rfft(n_win, n=n_fft, axis=-1))  # [n_frames, 257]

    # Input X: First 32 magnitude bins (0 Hz to ~1000 Hz)
    X = Y_spec[:, :32].astype(np.float32)

    # Target Y (Ideal Ratio Mask): |S[0:16]| / (|S[0:16]| + |N[0:16]| + 1e-8)
    S_16 = S_spec[:, :16]
    N_16 = N_spec[:, :16]
    IRM = S_16 / (S_16 + N_16 + 1e-8)
    Y = np.clip(IRM, 0.0, 1.0).astype(np.float32)

    # Subsample frames if frames_per_file constraint is specified
    if frames_per_file is not None and frames_per_file > 0 and n_frames > frames_per_file:
        if rng is not None:
            indices = np.sort(rng.choice(n_frames, size=frames_per_file, replace=False))
        else:
            indices = np.linspace(0, n_frames - 1, frames_per_file, dtype=int)
        X = X[indices]
        Y = Y[indices]

    return X, Y


# ==============================================================================
# Pipeline Execution & Dataset Management
# ==============================================================================

class SpeechDatasetPreprocessor:
    """Manages audio file scanning, synthetic dataset generation, and export."""

    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        snr_levels: List[float],
        frames_per_file: Optional[int] = None,
        val_split: float = 0.2,
        seed: int = 42,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 256,
    ):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.snr_levels = snr_levels
        self.frames_per_file = frames_per_file
        self.val_split = val_split
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.rng = np.random.default_rng(seed)

    def scan_files(self) -> Tuple[List[Path], Dict[str, List[Path]]]:
        """Scan directory structure for clean commands and isolated noises."""
        clean_dir = self.data_dir / "clean_command"
        if not clean_dir.exists():
            raise FileNotFoundError(f"Clean command folder not found: {clean_dir}")

        clean_files = sorted(list(clean_dir.glob("*.wav")) + list(clean_dir.glob("*.WAV")))
        if not clean_files:
            raise RuntimeError(f"No .wav files found in {clean_dir}")

        noise_categories = ["gun", "helicopter", "bomb"]
        noise_files: Dict[str, List[Path]] = {}

        total_noises = 0
        for cat in noise_categories:
            cat_dir = self.data_dir / cat
            if cat_dir.exists():
                files = sorted(list(cat_dir.glob("*.wav")) + list(cat_dir.glob("*.WAV")))
                noise_files[cat] = files
                total_noises += len(files)
            else:
                noise_files[cat] = []

        if total_noises == 0:
            raise RuntimeError(
                f"No noise files found in '{self.data_dir}'. Ensure gun/, helicopter/, or bomb/ folders contain .wav clips."
            )

        print(f"[Dataset Scan] Found {len(clean_files)} clean speech files.")
        for cat, files in noise_files.items():
            print(f"  - Noise '{cat}': {len(files)} files")

        return clean_files, noise_files

    def run(self) -> None:
        """Run complete preprocessing and export compressed .npz feature archives."""
        clean_files, noise_dict = self.scan_files()

        # Flatten available noise files
        all_noise_files = [f for files in noise_dict.values() for f in files]

        # Shuffle clean files for train/val split at utterance level
        shuffled_clean = list(clean_files)
        self.rng.shuffle(shuffled_clean)

        n_val = int(round(len(shuffled_clean) * self.val_split))
        val_clean_files = shuffled_clean[:n_val]
        train_clean_files = shuffled_clean[n_val:]

        print(f"\n[Dataset Partition] Utterance Split:")
        print(f"  - Training files:   {len(train_clean_files)} ({100*(1 - self.val_split):.0f}%)")
        print(f"  - Validation files: {len(val_clean_files)} ({100*self.val_split:.0f}%)")

        # Process sets
        print("\n[Processing Training Set]")
        X_train, Y_train = self._process_file_list(train_clean_files, all_noise_files, desc="Train Extraction")

        print("\n[Processing Validation Set]")
        X_val, Y_val = self._process_file_list(val_clean_files, all_noise_files, desc="Val Extraction")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        train_path = self.output_dir / "train_features.npz"
        val_path = self.output_dir / "val_features.npz"

        print(f"\n[Saving Datasets]")
        print(f"  - Train Features: X={X_train.shape}, Y={Y_train.shape} -> {train_path}")
        np.savez_compressed(train_path, X=X_train, Y=Y_train)

        print(f"  - Val Features:   X={X_val.shape}, Y={Y_val.shape} -> {val_path}")
        np.savez_compressed(val_path, X=X_val, Y=Y_val)

        print("\n[Preprocessing Complete] Ready for model training and Vivado co-simulation.")

    def _process_file_list(
        self,
        clean_file_list: List[Path],
        noise_file_list: List[Path],
        desc: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Iterate over clean files, mix with randomized noises at target SNRs, extract (X, Y)."""
        all_x: List[np.ndarray] = []
        all_y: List[np.ndarray] = []

        for clean_path in tqdm(clean_file_list, desc=desc, unit="file"):
            try:
                clean_audio = load_and_normalize_audio(clean_path, target_sr=self.sample_rate)
                if len(clean_audio) == 0:
                    continue

                # Sample random noise clip and SNR level
                noise_path = self.rng.choice(noise_file_list)
                noise_audio = load_and_normalize_audio(noise_path, target_sr=self.sample_rate)
                if len(noise_audio) == 0:
                    continue

                target_snr = float(self.rng.choice(self.snr_levels))

                # Synthetic mixing
                y_noisy, clean_scaled, noise_scaled = match_and_scale_noise(
                    clean=clean_audio,
                    noise=noise_audio,
                    snr_db=target_snr,
                    rng=self.rng,
                )

                # Feature extraction
                X_frames, Y_frames = extract_stft_features_and_mask(
                    y_noisy=y_noisy,
                    clean=clean_scaled,
                    noise=noise_scaled,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    frames_per_file=self.frames_per_file,
                    rng=self.rng,
                )

                if len(X_frames) > 0:
                    all_x.append(X_frames)
                    all_y.append(Y_frames)

            except Exception as ex:
                print(f"[Warning] Failed to process {clean_path.name}: {ex}", file=sys.stderr)

        if all_x:
            X_stacked = np.vstack(all_x).astype(np.float32)
            Y_stacked = np.vstack(all_y).astype(np.float32)
        else:
            X_stacked = np.empty((0, 32), dtype=np.float32)
            Y_stacked = np.empty((0, 16), dtype=np.float32)

        return X_stacked, Y_stacked


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Modular Audio DSP Preprocessor for FPGA Speech Enhancement Accelerator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data"),
        help="Root directory containing clean_command/ and noise folders (gun/, helicopter/, bomb/).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("processed_data"),
        help="Destination directory to save train_features.npz and val_features.npz.",
    )
    parser.add_argument(
        "--frames_per_file",
        type=int,
        default=None,
        help="Maximum frames retained per audio file (default: None, retains all frames).",
    )
    parser.add_argument(
        "--snr_levels",
        type=str,
        default="-5,0,5,10",
        help="Comma-separated list of SNR mixing levels in dB.",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Validation split ratio (0.0 to 1.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Audio sampling rate in Hz.",
    )
    parser.add_argument(
        "--n_fft",
        type=int,
        default=512,
        help="FFT size in samples (specifies 257 positive frequency bins).",
    )
    parser.add_argument(
        "--hop_length",
        type=int,
        default=256,
        help="Hop length / frame stride in samples (50%% overlap).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        snrs = [float(s.strip()) for s in args.snr_levels.split(",") if s.strip()]
    except ValueError:
        print(f"Error: Invalid --snr_levels '{args.snr_levels}'. Expected comma-separated floats.", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("FPGA Speech Enhancement: DSP Preprocessing Engine")
    print(f"Sampling Rate:      {args.sample_rate} Hz")
    print(f"FFT / Hop:          {args.n_fft} / {args.hop_length} samples (Hann window)")
    print(f"Input Feature Bins: 0..31 (0 Hz to ~1000 Hz)")
    print(f"Target Mask Bins:   0..15 (Ideal Ratio Mask)")
    print(f"SNR Levels:         {snrs} dB")
    print(f"Data Root:          {args.data_dir.resolve()}")
    print(f"Output Directory:   {args.output_dir.resolve()}")
    print("=" * 70)

    preprocessor = SpeechDatasetPreprocessor(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        snr_levels=snrs,
        frames_per_file=args.frames_per_file,
        val_split=args.val_split,
        seed=args.seed,
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
    )

    try:
        preprocessor.run()
    except Exception as e:
        print(f"\n[Execution Error] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
