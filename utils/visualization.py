import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless / server environments
import matplotlib.pyplot as plt
import numpy as np
import librosa
import pandas as pd


def plot_spectrograms(clean: np.ndarray, noisy: np.ndarray, enhanced_dict: dict, sr: int = 16000, save_path: str = None, title: str = "Speech Enhancement Spectrogram Comparison"):
    """
    Plot and save spectrogram comparison for Clean, Noisy, and multiple enhanced signals (Wiener, CRN, DCCRN).
    """
    signals = {"Clean Speech": clean, "Noisy Speech": noisy}
    signals.update(enhanced_dict)

    n_plots = len(signals)
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3 * n_plots), sharex=True, sharey=True)
    if n_plots == 1:
        axes = [axes]

    for ax, (name, sig) in zip(axes, signals.items()):
        # Compute spectrogram in dB
        D = librosa.amplitude_to_db(np.abs(librosa.stft(sig, n_fft=512, hop_length=128)), ref=np.max)
        img = librosa.display.specshow(D, sr=sr, hop_length=128, x_axis="time", y_axis="hz", ax=ax, cmap="magma")
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_ylim(0, sr // 2)
        fig.colorbar(img, ax=ax, format="%+2.0f dB")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[*] Saved spectrogram plot to {save_path}")
    plt.close(fig)


def plot_loss_curves(crn_history: dict = None, dccrn_history: dict = None, save_path: str = None):
    """
    Plot training and validation loss curves for CRN and DCCRN.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if crn_history:
        epochs_train = range(1, len(crn_history.get("train_loss", [])) + 1)
        axes[0].plot(epochs_train, crn_history.get("train_loss", []), "b-", label="CRN Train Loss", linewidth=2)
        epochs_val = range(1, len(crn_history.get("val_loss", [])) + 1)
        axes[0].plot(epochs_val, crn_history.get("val_loss", []), "b--", label="CRN Val Loss", linewidth=2)

    if dccrn_history:
        epochs_train = range(1, len(dccrn_history.get("train_loss", [])) + 1)
        axes[0].plot(epochs_train, dccrn_history.get("train_loss", []), "r-", label="DCCRN Train Loss", linewidth=2)
        epochs_val = range(1, len(dccrn_history.get("val_loss", [])) + 1)
        axes[0].plot(epochs_val, dccrn_history.get("val_loss", []), "r--", label="DCCRN Val Loss", linewidth=2)

    axes[0].set_title("Training & Validation Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    # Right plot: Validation SI-SNR (if present)
    if crn_history and "val_si_snr" in crn_history:
        axes[1].plot(range(1, len(crn_history["val_si_snr"]) + 1), crn_history["val_si_snr"], "b-o", label="CRN Val SI-SNR")
    if dccrn_history and "val_si_snr" in dccrn_history:
        axes[1].plot(range(1, len(dccrn_history["val_si_snr"]) + 1), dccrn_history["val_si_snr"], "r-s", label="DCCRN Val SI-SNR")

    axes[1].set_title("Validation SI-SNR (dB)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("SI-SNR (dB)")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[*] Saved loss curves to {save_path}")
    plt.close(fig)


def plot_metric_vs_snr(comparison_df: pd.DataFrame, save_dir: str):
    """
    Generate evaluation comparison plots:
    1. SNR Improvement vs Input SNR
    2. STOI vs Input SNR
    3. PESQ vs Input SNR
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1. Delta SNR
    if "CRN_Delta_SNR" in comparison_df.columns and "DCCRN_Delta_SNR" in comparison_df.columns:
        plt.figure(figsize=(7, 5))
        snrs = comparison_df["SNR"]
        if "Wiener_Delta_SNR" in comparison_df.columns:
            plt.plot(snrs, comparison_df["Wiener_Delta_SNR"], "g--o", label="Wiener Filter", linewidth=2)
        plt.plot(snrs, comparison_df["CRN_Delta_SNR"], "b-o", label="CRN", linewidth=2)
        plt.plot(snrs, comparison_df["DCCRN_Delta_SNR"], "r-s", label="DCCRN", linewidth=2)
        plt.title("SNR Improvement (ΔSNR) vs Input SNR", fontsize=12, fontweight="bold")
        plt.xlabel("Input SNR (dB)")
        plt.ylabel("ΔSNR (dB)")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "delta_snr_vs_snr.png"), dpi=200)
        plt.close()

    # 2. STOI
    if "CRN_STOI" in comparison_df.columns and "DCCRN_STOI" in comparison_df.columns:
        plt.figure(figsize=(7, 5))
        snrs = comparison_df["SNR"]
        if "Noisy_STOI" in comparison_df.columns:
            plt.plot(snrs, comparison_df["Noisy_STOI"], "k:", label="Noisy Speech", linewidth=2)
        if "Wiener_STOI" in comparison_df.columns:
            plt.plot(snrs, comparison_df["Wiener_STOI"], "g--o", label="Wiener Filter", linewidth=2)
        plt.plot(snrs, comparison_df["CRN_STOI"], "b-o", label="CRN", linewidth=2)
        plt.plot(snrs, comparison_df["DCCRN_STOI"], "r-s", label="DCCRN", linewidth=2)
        plt.title("Speech Intelligibility (STOI) vs Input SNR", fontsize=12, fontweight="bold")
        plt.xlabel("Input SNR (dB)")
        plt.ylabel("STOI Score (0 to 1)")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "stoi_vs_snr.png"), dpi=200)
        plt.close()

    # 3. PESQ
    if "CRN_PESQ" in comparison_df.columns and "DCCRN_PESQ" in comparison_df.columns:
        plt.figure(figsize=(7, 5))
        snrs = comparison_df["SNR"]
        if "Noisy_PESQ" in comparison_df.columns and not comparison_df["Noisy_PESQ"].isna().all():
            plt.plot(snrs, comparison_df["Noisy_PESQ"], "k:", label="Noisy Speech", linewidth=2)
            if "Wiener_PESQ" in comparison_df.columns:
                plt.plot(snrs, comparison_df["Wiener_PESQ"], "g--o", label="Wiener Filter", linewidth=2)
            plt.plot(snrs, comparison_df["CRN_PESQ"], "b-o", label="CRN", linewidth=2)
            plt.plot(snrs, comparison_df["DCCRN_PESQ"], "r-s", label="DCCRN", linewidth=2)
            plt.title("Perceptual Quality (PESQ) vs Input SNR", fontsize=12, fontweight="bold")
            plt.xlabel("Input SNR (dB)")
            plt.ylabel("PESQ Score (-0.5 to 4.5)")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "pesq_vs_snr.png"), dpi=200)
            plt.close()
