"""
eval.py - Objective Metric Evaluation Suite (PESQ, STOI, SI-SNR Improvement).

Evaluates speech enhancement performance on:
- Wideband PESQ (Perceptual Evaluation of Speech Quality)
- STOI (Short-Time Objective Intelligibility)
- Delta SI-SNR (Scale-Invariant Signal-to-Noise Ratio Improvement)
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pystoi
import soundfile as sf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SpeechEnhancementDataset
from losses import calculate_sisnr
from modules import CausalDPCRN

try:
    import pesq
    HAS_PESQ = True
except ImportError:
    HAS_PESQ = False

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DPCRN_Evaluator")


def calculate_metrics(
    clean: np.ndarray,
    noisy: np.ndarray,
    enhanced: np.ndarray,
    sr: int = 16000,
) -> Dict[str, float]:
    """
    Compute objective speech quality metrics for a single sample.
    """
    # 1. SI-SNR
    clean_t = torch.from_numpy(clean).unsqueeze(0)
    noisy_t = torch.from_numpy(noisy).unsqueeze(0)
    enh_t = torch.from_numpy(enhanced).unsqueeze(0)

    sisnr_noisy = calculate_sisnr(noisy_t, clean_t).item()
    sisnr_enh = calculate_sisnr(enh_t, clean_t).item()
    delta_sisnr = sisnr_enh - sisnr_noisy

    # 2. STOI
    stoi_noisy = pystoi.stoi(clean, noisy, sr, extended=False)
    stoi_enh = pystoi.stoi(clean, enhanced, sr, extended=False)

    # 3. PESQ (Wideband)
    pesq_noisy = None
    pesq_enh = None
    if HAS_PESQ:
        try:
            pesq_noisy = pesq.pesq(sr, clean, noisy, "wb")
            pesq_enh = pesq.pesq(sr, clean, enhanced, "wb")
        except Exception:
            pass

    return {
        "sisnr_noisy": sisnr_noisy,
        "sisnr_enh": sisnr_enh,
        "delta_sisnr": delta_sisnr,
        "stoi_noisy": stoi_noisy,
        "stoi_enh": stoi_enh,
        "pesq_noisy": pesq_noisy,
        "pesq_enh": pesq_enh,
    }


@torch.no_grad()
def evaluate_dataset(
    model: CausalDPCRN,
    dataloader: DataLoader,
    device: torch.device,
    save_audio_dir: Optional[str] = None,
    max_eval_samples: int = 50,
) -> Dict[str, float]:
    model.eval()
    if save_audio_dir:
        os.makedirs(save_audio_dir, exist_ok=True)

    delta_sisnrs = []
    enh_sisnrs = []
    enh_stois = []
    enh_pesqs = []
    noisy_stois = []
    noisy_pesqs = []

    count = 0
    pbar = tqdm(dataloader, desc="Evaluating Speech Enhancement Metrics")
    for noisy, clean, snr in pbar:
        noisy_dev = noisy.to(device)
        enh_wav, _, _ = model(noisy_dev)
        enh_wav = enh_wav.cpu().numpy()
        noisy = noisy.numpy()
        clean = clean.numpy()

        for i in range(len(noisy)):
            m = calculate_metrics(clean[i], noisy[i], enh_wav[i], sr=16000)
            delta_sisnrs.append(m["delta_sisnr"])
            enh_sisnrs.append(m["sisnr_enh"])
            enh_stois.append(m["stoi_enh"])
            noisy_stois.append(m["stoi_noisy"])

            if m["pesq_enh"] is not None:
                enh_pesqs.append(m["pesq_enh"])
                noisy_pesqs.append(m["pesq_noisy"])

            if save_audio_dir and count < 10:
                sf.write(os.path.join(save_audio_dir, f"sample_{count:03d}_clean.wav"), clean[i], 16000)
                sf.write(os.path.join(save_audio_dir, f"sample_{count:03d}_noisy.wav"), noisy[i], 16000)
                sf.write(os.path.join(save_audio_dir, f"sample_{count:03d}_enhanced.wav"), enh_wav[i], 16000)

            count += 1
            if count >= max_eval_samples:
                break
        if count >= max_eval_samples:
            break

    summary = {
        "Mean Enhanced SI-SNR (dB)": float(np.mean(enh_sisnrs)),
        "Mean SI-SNR Improvement (dB)": float(np.mean(delta_sisnrs)),
        "Mean Enhanced STOI": float(np.mean(enh_stois)),
        "Mean Input Noisy STOI": float(np.mean(noisy_stois)),
    }
    if len(enh_pesqs) > 0:
        summary["Mean Enhanced PESQ-WB"] = float(np.mean(enh_pesqs))
        summary["Mean Input Noisy PESQ-WB"] = float(np.mean(noisy_pesqs))

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate Causal DPCRN on Speech Metrics")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_dpcrn_checkpoint.pth")
    parser.add_argument("--clean_dir", type=str, default="clean_speech")
    parser.add_argument("--noise_dir", type=str, default="defence_noise")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--save_audio", type=str, default="evaluation_audio")
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CausalDPCRN().to(device)
    if os.path.exists(args.checkpoint):
        logger.info(f"Loading weights from {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
    else:
        logger.warning(f"Checkpoint {args.checkpoint} not found; evaluating initialized model.")

    dataset = SpeechEnhancementDataset(
        clean_dir=args.clean_dir,
        noise_dir=args.noise_dir,
        chunk_duration=3.0,
        snr_range=(-10.0, 15.0),
        synthetic_fallback=True,
        dataset_size=args.num_samples,
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    results = evaluate_dataset(
        model=model,
        dataloader=dataloader,
        device=device,
        save_audio_dir=args.save_audio,
        max_eval_samples=args.num_samples,
    )

    logger.info("================ Final Evaluation Summary ================")
    for k, v in results.items():
        logger.info(f"  {k:30s}: {v:.3f}")


if __name__ == "__main__":
    main()
