"""
visualize.py - High-Resolution Spectrogram and Waveform Analysis Tool for Speech Enhancement.

Generates visual diagnostic plots comparing:
1. Time-Domain Waveforms (Clean, Noisy, Enhanced)
2. Log-Magnitude STFT Spectrograms
3. Estimated Complex Ratio Mask (cRM)
"""

import argparse
import logging
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

from dataset import generate_synthetic_clean_speech, generate_synthetic_defence_noise, peak_normalize
from modules import CausalDPCRN

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DPCRN_Visualizer")


def compute_spectrogram(audio: np.ndarray, n_fft: int = 512, hop_length: int = 128) -> np.ndarray:
    """Compute log-magnitude STFT spectrogram in dB."""
    audio_t = torch.from_numpy(audio).unsqueeze(0).float()
    window = torch.hann_window(n_fft)
    stft = torch.stft(
        audio_t,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=True,
    )
    mag = torch.abs(stft).squeeze(0).numpy()
    log_spec = 20.0 * np.log10(np.maximum(mag, 1e-6))
    return log_spec


def plot_speech_enhancement_analysis(
    clean_audio: np.ndarray,
    noise_audio: np.ndarray,
    noisy_audio: np.ndarray,
    enhanced_audio: np.ndarray,
    crm_mask: np.ndarray,
    sr: int = 16000,
    save_path: str = "spectrogram_analysis.png",
):
    """
    Generate a 5-row publication-quality comparative figure.
    """
    time_axis = np.linspace(0, len(clean_audio) / sr, len(clean_audio))
    duration = len(clean_audio) / sr

    fig, axs = plt.subplots(5, 2, figsize=(14, 12), gridspec_kw={"width_ratios": [1, 1.3]})
    plt.subplots_adjust(hspace=0.45, wspace=0.25)

    signals = [
        ("Clean Speech", clean_audio, "#1f77b4"),
        ("Defence Noise", noise_audio, "#d62728"),
        ("Noisy Input (Mixed)", noisy_audio, "#ff7f0e"),
        ("Enhanced Speech (DPCRN)", enhanced_audio, "#2ca02c"),
    ]

    for idx, (title, sig, color) in enumerate(signals):
        # 1. Waveform Plot
        axs[idx, 0].plot(time_axis, sig, color=color, linewidth=0.8)
        axs[idx, 0].set_title(f"{title} (Waveform)", fontsize=10, fontweight="bold")
        axs[idx, 0].set_xlim([0, duration])
        axs[idx, 0].set_ylim([-1.05, 1.05])
        axs[idx, 0].set_ylabel("Amplitude")
        axs[idx, 0].grid(True, linestyle="--", alpha=0.5)

        # 2. Spectrogram Plot
        spec = compute_spectrogram(sig)
        im = axs[idx, 1].imshow(
            spec,
            aspect="auto",
            origin="lower",
            extent=[0, duration, 0, sr / 2000],
            cmap="magma",
            vmin=-60,
            vmax=20,
        )
        axs[idx, 1].set_title(f"{title} (Spectrogram)", fontsize=10, fontweight="bold")
        axs[idx, 1].set_ylabel("Frequency (kHz)")
        fig.colorbar(im, ax=axs[idx, 1], format="%+2.0f dB", fraction=0.046, pad=0.04)

    # 5. Complex Ratio Mask Row
    axs[4, 0].plot(time_axis, noisy_audio, color="#ff7f0e", alpha=0.5, label="Noisy Input")
    axs[4, 0].plot(time_axis, enhanced_audio, color="#2ca02c", alpha=0.85, label="Enhanced Output")
    axs[4, 0].set_title("Waveform Overlay Comparison", fontsize=10, fontweight="bold")
    axs[4, 0].set_xlim([0, duration])
    axs[4, 0].set_ylim([-1.05, 1.05])
    axs[4, 0].set_xlabel("Time (seconds)")
    axs[4, 0].set_ylabel("Amplitude")
    axs[4, 0].legend(loc="upper right", fontsize=8)
    axs[4, 0].grid(True, linestyle="--", alpha=0.5)

    im_mask = axs[4, 1].imshow(
        crm_mask,
        aspect="auto",
        origin="lower",
        extent=[0, duration, 0, sr / 2000],
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    axs[4, 1].set_title("Learned Complex Ratio Mask |M|", fontsize=10, fontweight="bold")
    axs[4, 1].set_xlabel("Time (seconds)")
    axs[4, 1].set_ylabel("Frequency (kHz)")
    fig.colorbar(im_mask, ax=axs[4, 1], format="%.2f", fraction=0.046, pad=0.04)

    plt.suptitle("Causal DPCRN Defence Speech Enhancement Analysis", fontsize=14, fontweight="bold", y=0.995)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Spectrogram analysis figure saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Spectrogram Analysis")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_dpcrn_checkpoint.pth")
    parser.add_argument("--output_image", type=str, default="spectrogram_analysis.png")
    parser.add_argument("--noise_type", type=str, default="rotor_blades", choices=["gunshots", "artillery", "tank_rumble", "rotor_blades"])
    parser.add_argument("--debug", action="store_true", default=False, help="Enable remote debugger on port 5678")
    parser.add_argument("--debug_port", type=int, default=5678, help="Remote debugger port")
    parser.add_argument("--wait_for_debugger", action="store_true", default=False, help="Wait for IDE debugger to attach")
    args = parser.parse_args()

    if args.debug:
        import debugpy
        logger.info(f"Enabling debugpy remote debugger on 0.0.0.0:{args.debug_port}...")
        debugpy.listen(("0.0.0.0", args.debug_port))
        if args.wait_for_debugger:
            logger.info("[PAUSED] Waiting for debugger to attach from IDE...")
            debugpy.wait_for_client()
            logger.info("[CONNECTED] Debugger attached!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CausalDPCRN().to(device)
    model.eval()

    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        logger.info(f"Loaded weights from {args.checkpoint}")

    # Generate synthetic audio for testing
    logger.info(f"Synthesizing test clean speech and defence noise ({args.noise_type})...")
    clean = generate_synthetic_clean_speech(48000, 16000)
    noise = generate_synthetic_defence_noise(args.noise_type, 48000, 16000)

    # Mix at 0 dB SNR
    noisy = peak_normalize(clean + 0.8 * noise, target_peak=0.95)
    clean = peak_normalize(clean, target_peak=0.95)
    noise = peak_normalize(noise, target_peak=0.95)

    with torch.no_grad():
        noisy_t = noisy.unsqueeze(0).to(device)
        enh_wav, enh_spec, mask = model(noisy_t)
        enh_wav = enh_wav.squeeze(0).cpu().numpy()
        mask = mask.squeeze(0).cpu()  # [2, 257, T]
        mask_mag = torch.sqrt(mask[0]**2 + mask[1]**2).numpy()

    plot_speech_enhancement_analysis(
        clean_audio=clean.numpy(),
        noise_audio=noise.numpy(),
        noisy_audio=noisy.numpy(),
        enhanced_audio=enh_wav,
        crm_mask=mask_mag,
        sr=16000,
        save_path=args.output_image,
    )


if __name__ == "__main__":
    main()
