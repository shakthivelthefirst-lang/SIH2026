"""
dataset.py - Dataset Loader & Dynamic Mixing Pipeline for Defence Speech Enhancement.

Key Capabilities:
1. Dynamic on-the-fly SNR mixing between -10 dB and +15 dB.
2. RMS-matched scaling: x_noisy = x_clean + 10^(-SNR/20) * (RMS_clean / RMS_noise) * x_noise
3. Peak normalization to 0.95 to eliminate digital clipping.
4. Automatic resampling to 16,000 Hz.
5. Built-in synthetic Defence Noise generator (gunshots, artillery, tank rumble, rotor blades)
   and clean speech generator for self-contained testing and bootstrapping.
"""

import math
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
import torchaudio


def calculate_rms(audio: torch.Tensor, eps: float = 1e-8) -> float:
    """Calculate Root Mean Square (RMS) energy of a 1D audio tensor."""
    return torch.sqrt(torch.mean(audio ** 2) + eps).item()


def peak_normalize(audio: torch.Tensor, target_peak: float = 0.95, eps: float = 1e-8) -> torch.Tensor:
    """Scale audio so maximum absolute peak equals target_peak."""
    max_val = torch.max(torch.abs(audio))
    if max_val > 0:
        return audio * (target_peak / (max_val + eps))
    return audio


def generate_synthetic_defence_noise(
    noise_type: str,
    num_samples: int = 48000,
    sr: int = 16000,
) -> torch.Tensor:
    """
    Generate physically realistic synthetic defence noise for testing and simulation:
    - 'gunshots': Short high-energy impulses with exponential reverberant decay.
    - 'artillery': Low-frequency sub-bass shockwave blast with long resonance.
    - 'tank_rumble': Heavy diesel engine harmonics + track mechanical rattling.
    - 'rotor_blades': Periodic amplitude/frequency modulated helicopter blade chop.
    """
    t = torch.linspace(0, num_samples / sr, num_samples)
    noise = torch.zeros(num_samples)

    if noise_type == "gunshots":
        # Multiple gunshot impulses
        num_shots = random.randint(2, 5)
        for _ in range(num_shots):
            shot_idx = random.randint(0, max(1, num_samples - 8000))
            decay_len = min(num_samples - shot_idx, int(sr * random.uniform(0.15, 0.45)))
            decay = torch.exp(-torch.linspace(0, 15, decay_len))
            impulse = (torch.randn(decay_len) * 2.0 - 1.0) * decay
            noise[shot_idx : shot_idx + decay_len] += impulse

    elif noise_type == "artillery":
        # Low frequency explosion thump (30-80 Hz) + shockwave
        blast_idx = random.randint(0, max(1, num_samples - 16000))
        blast_len = min(num_samples - blast_idx, int(sr * random.uniform(0.8, 1.8)))
        decay = torch.exp(-torch.linspace(0, 5, blast_len))
        sub_bass = torch.sin(2 * math.pi * random.uniform(35, 65) * t[:blast_len])
        noise_burst = torch.randn(blast_len) * decay * 1.5 + sub_bass * decay * 2.0
        noise[blast_idx : blast_idx + blast_len] += noise_burst

    elif noise_type == "tank_rumble":
        # Engine fundamental (40-90 Hz) + track rumble + vibration
        f0 = random.uniform(45, 75)
        engine = (
            1.0 * torch.sin(2 * math.pi * f0 * t)
            + 0.6 * torch.sin(2 * math.pi * (2 * f0) * t)
            + 0.4 * torch.sin(2 * math.pi * (3 * f0) * t)
        )
        treads = torch.randn(num_samples) * 0.35
        # Low-pass filter smoothing
        noise = engine + treads

    elif noise_type == "rotor_blades":
        # Helicopter blade passage frequency (10-25 Hz) chopping broadband noise
        bpf = random.uniform(12, 22)
        chop_envelope = (0.5 + 0.5 * torch.sin(2 * math.pi * bpf * t)) ** 4
        turbine_whine = 0.25 * torch.sin(2 * math.pi * random.uniform(1200, 2400) * t)
        noise = torch.randn(num_samples) * chop_envelope * 1.2 + turbine_whine

    else:
        # Generic broadband defence noise
        noise = torch.randn(num_samples)

    return noise


def generate_synthetic_clean_speech(
    num_samples: int = 48000,
    sr: int = 16000,
) -> torch.Tensor:
    """
    Synthesize mock speech formants (vowel-like harmonics + consonants) for testing.
    """
    t = torch.linspace(0, num_samples / sr, num_samples)
    speech = torch.zeros(num_samples)
    num_syllables = random.randint(4, 9)
    syllable_len = num_samples // num_syllables

    for i in range(num_syllables):
        start = i * syllable_len
        end = min(num_samples, start + int(syllable_len * 0.8))
        curr_len = end - start
        if curr_len <= 0:
            continue
        pitch = random.uniform(100, 240)  # Pitch in Hz
        f1, f2 = random.uniform(400, 800), random.uniform(1200, 2500)  # Formants
        env = torch.sin(torch.linspace(0, math.pi, curr_len)) ** 2
        t_s = t[:curr_len]
        voiced = (
            torch.sin(2 * math.pi * pitch * t_s)
            + 0.7 * torch.sin(2 * math.pi * f1 * t_s)
            + 0.4 * torch.sin(2 * math.pi * f2 * t_s)
        )
        speech[start:end] = voiced * env

    return speech


class SpeechEnhancementDataset(Dataset):
    """
    Streaming Speech Enhancement Dataset with Dynamic On-The-Fly Mixing.
    
    Args:
        clean_dir: Path to directory containing clean speech files (.wav, .flac)
        noise_dir: Path to directory containing defence noise files (.wav, .flac)
        sample_rate: Target audio sampling rate (16,000 Hz)
        chunk_duration: Target segment length in seconds (3.0 s -> 48,000 samples)
        snr_range: Range of uniform random SNR mixing in dB (-10 dB to +15 dB)
        synthetic_fallback: If True, generate synthetic audio if directories are empty/missing.
        dataset_size: Virtual epoch size when using synthetic fallback or repeating data.
    """
    def __init__(
        self,
        clean_dir: Optional[Union[str, Path]] = None,
        noise_dir: Optional[Union[str, Path]] = None,
        sample_rate: int = 16000,
        chunk_duration: float = 3.0,
        snr_range: Tuple[float, float] = (-10.0, 15.0),
        synthetic_fallback: bool = True,
        dataset_size: int = 1000,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.chunk_samples = int(chunk_duration * sample_rate)
        self.snr_min, self.snr_max = snr_range
        self.synthetic_fallback = synthetic_fallback
        self.dataset_size = dataset_size

        self.clean_files = self._gather_audio_files(clean_dir)
        self.noise_files = self._gather_audio_files(noise_dir)

        self.use_synthetic = (
            self.synthetic_fallback and (len(self.clean_files) == 0 or len(self.noise_files) == 0)
        )

        self.noise_types = ["gunshots", "artillery", "tank_rumble", "rotor_blades"]

    def _gather_audio_files(self, directory: Optional[Union[str, Path]]) -> List[Path]:
        if directory is None or not os.path.exists(directory):
            return []
        p = Path(directory)
        extensions = [".wav", ".flac", ".mp3", ".ogg"]
        files = [f for f in p.rglob("*") if f.suffix.lower() in extensions]
        return files

    def _load_and_resample(self, file_path: Path) -> torch.Tensor:
        """Load audio via SoundFile, convert to mono, and resample to 16 kHz."""
        wav_np, sr = sf.read(str(file_path))
        if len(wav_np.shape) > 1:
            wav_np = np.mean(wav_np, axis=-1)
        wav = torch.from_numpy(wav_np).float()
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            wav = resampler(wav.unsqueeze(0)).squeeze(0)
        return wav

    def _get_random_chunk(self, audio: torch.Tensor) -> torch.Tensor:
        """Slice random 3.0s chunk, padding with repeat if audio is shorter."""
        num_samples = audio.shape[-1]
        if num_samples < self.chunk_samples:
            # Repeat to fill chunk
            repeat_count = (self.chunk_samples // num_samples) + 1
            audio = audio.repeat(repeat_count)
            num_samples = audio.shape[-1]

        max_start = num_samples - self.chunk_samples
        start = random.randint(0, max_start)
        return audio[start : start + self.chunk_samples]

    def __len__(self) -> int:
        if self.use_synthetic:
            return self.dataset_size
        return max(len(self.clean_files), self.dataset_size)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Returns:
            noisy_audio: [48000] mixed time-domain waveform
            clean_audio: [48000] clean target waveform
            snr_db: Target SNR mixed for this sample
        """
        # 1. Fetch Clean Speech
        if self.use_synthetic or len(self.clean_files) == 0:
            clean = generate_synthetic_clean_speech(self.chunk_samples, self.sample_rate)
        else:
            clean_file = random.choice(self.clean_files)
            clean_raw = self._load_and_resample(clean_file)
            clean = self._get_random_chunk(clean_raw)

        # 2. Fetch Defence Noise
        if self.use_synthetic or len(self.noise_files) == 0:
            noise_type = random.choice(self.noise_types)
            noise = generate_synthetic_defence_noise(noise_type, self.chunk_samples, self.sample_rate)
        else:
            noise_file = random.choice(self.noise_files)
            noise_raw = self._load_and_resample(noise_file)
            noise = self._get_random_chunk(noise_raw)

        # 3. Sample Random SNR uniformly between -10 dB and +15 dB
        snr_db = random.uniform(self.snr_min, self.snr_max)

        # 4. Dynamic RMS Scaling:
        # x_noisy = x_clean + 10^(-SNR/20) * (RMS_clean / RMS_noise) * x_noise
        rms_clean = calculate_rms(clean)
        rms_noise = calculate_rms(noise)

        snr_factor = 10.0 ** (-snr_db / 20.0)
        noise_scaled = noise * (snr_factor * (rms_clean / (rms_noise + 1e-8)))
        noisy = clean + noise_scaled

        # 5. Peak Normalization: Scale to 0.95 peak amplitude
        noisy = peak_normalize(noisy, target_peak=0.95)
        # Scale clean speech with the identical gain factor to preserve SNR calibration
        clean = peak_normalize(clean, target_peak=0.95)

        return noisy.float(), clean.float(), snr_db
