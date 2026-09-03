import os
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from models.crn import CRN
from models.dccrn import DCCRN
from baselines.wiener_filter import wiener_filter_enhance
from datasets.mad_dataset import create_dataloaders, MAD_CATEGORIES
from metrics.metrics import evaluate_all_metrics, calculate_snr, calculate_sisnr_metric, calculate_stoi, calculate_pesq
from utils.checkpoint import load_checkpoint
from utils.audio import save_audio
from utils.visualization import plot_spectrograms, plot_metric_vs_snr


def evaluate_models(config: dict, test_loader=None, device: torch.device = None, crn_checkpoint: str = None, dccrn_checkpoint: str = None, num_eval_samples: int = None):
    """
    Complete evaluation comparing Noisy, Wiener Baseline, CRN, and DCCRN across multiple SNRs and defence noise categories.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
    results_dir = config.get("results_dir", "results")
    output_dir = config.get("output_dir", "outputs")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    sr = config.get("sample_rate", 16000)

    # Instantiate Models
    crn = CRN(
        n_fft=config.get("n_fft", 512),
        hop_length=config.get("hop_length", 128),
        win_length=config.get("win_length", 512),
        lstm_layers=config.get("lstm_layers", 2),
        hidden_size=config.get("hidden_size", 256)
    ).to(device)

    dccrn = DCCRN(
        n_fft=config.get("n_fft", 512),
        hop_length=config.get("hop_length", 128),
        win_length=config.get("win_length", 512),
        lstm_layers=config.get("lstm_layers", 2),
        hidden_size=config.get("hidden_size", 256)
    ).to(device)

    # Load Checkpoints if available
    crn_path = crn_checkpoint or os.path.join(checkpoint_dir, "crn_best.pth")
    if not os.path.exists(crn_path):
        crn_path = os.path.join(checkpoint_dir, "crn_latest.pth")

    dccrn_path = dccrn_checkpoint or os.path.join(checkpoint_dir, "dccrn_best.pth")
    if not os.path.exists(dccrn_path):
        dccrn_path = os.path.join(checkpoint_dir, "dccrn_latest.pth")

    has_crn = os.path.exists(crn_path)
    if has_crn:
        load_checkpoint(crn_path, crn, device=device)
        crn.eval()
        print(f"[*] Loaded CRN from {crn_path}")
    else:
        print(f"[!] Warning: CRN checkpoint not found at {crn_path}. Using initialized weights.")
        crn.eval()

    has_dccrn = os.path.exists(dccrn_path)
    if has_dccrn:
        load_checkpoint(dccrn_path, dccrn, device=device)
        dccrn.eval()
        print(f"[*] Loaded DCCRN from {dccrn_path}")
    else:
        print(f"[!] Warning: DCCRN checkpoint not found at {dccrn_path}. Using initialized weights.")
        dccrn.eval()

    if test_loader is None:
        _, _, test_loader = create_dataloaders(
            train_csv=config.get("train_csv", "../training.csv"),
            test_csv=config.get("test_csv", "../test.csv"),
            data_dir=config.get("data_dir", "../"),
            sample_rate=sr,
            segment_seconds=config.get("segment_seconds", 3.0),
            snr_levels=config.get("snr_levels", [-10, -5, 0, 5, 10, 15]),
            batch_size=1,
            num_workers=0
        )

    # Subdirectories for audio examples
    for sub in ["clean", "noisy", "wiener", "crn", "dccrn"]:
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

    records = []
    saved_examples_count = 0
    max_save_examples = 10

    print("\n[*] Running complete benchmark on test samples...")
    sample_idx = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            if num_eval_samples and sample_idx >= num_eval_samples:
                break

            clean_t = batch["clean"].to(device)  # [B, T]
            noisy_t = batch["noisy"].to(device)  # [B, T]
            target_snr = float(batch["snr"][0].item())
            noise_label = int(batch["noise_label"][0].item())
            noise_cat = batch["noise_category"][0]

            # 1. Model Enhancements
            crn_out_t, _ = crn(noisy_t)
            dccrn_out_t, _ = dccrn(noisy_t)

            clean_np = clean_t[0].cpu().numpy()
            noisy_np = noisy_t[0].cpu().numpy()
            crn_np = crn_out_t[0].cpu().numpy()
            dccrn_np = dccrn_out_t[0].cpu().numpy()

            # 2. Wiener Filter Baseline
            wiener_np = wiener_filter_enhance(noisy_np, sr=sr)

            # 3. Compute Metrics
            noisy_snr = calculate_snr(clean_np, noisy_np)
            noisy_sisnr = calculate_sisnr_metric(clean_np, noisy_np)
            noisy_stoi = calculate_stoi(clean_np, noisy_np, sr=sr)
            noisy_pesq = calculate_pesq(clean_np, noisy_np, sr=sr)

            wiener_snr = calculate_snr(clean_np, wiener_np)
            wiener_sisnr = calculate_sisnr_metric(clean_np, wiener_np)
            wiener_stoi = calculate_stoi(clean_np, wiener_np, sr=sr)
            wiener_pesq = calculate_pesq(clean_np, wiener_np, sr=sr)

            crn_snr = calculate_snr(clean_np, crn_np)
            crn_sisnr = calculate_sisnr_metric(clean_np, crn_np)
            crn_stoi = calculate_stoi(clean_np, crn_np, sr=sr)
            crn_pesq = calculate_pesq(clean_np, crn_np, sr=sr)

            dccrn_snr = calculate_snr(clean_np, dccrn_np)
            dccrn_sisnr = calculate_sisnr_metric(clean_np, dccrn_np)
            dccrn_stoi = calculate_stoi(clean_np, dccrn_np, sr=sr)
            dccrn_pesq = calculate_pesq(clean_np, dccrn_np, sr=sr)

            records.append({
                "sample_idx": sample_idx,
                "SNR": target_snr,
                "noise_label": noise_label,
                "noise_category": noise_cat,
                # Noisy
                "Noisy_SNR": noisy_snr,
                "Noisy_SI_SNR": noisy_sisnr,
                "Noisy_STOI": noisy_stoi,
                "Noisy_PESQ": noisy_pesq,
                # Wiener
                "Wiener_SNR": wiener_snr,
                "Wiener_Delta_SNR": wiener_snr - noisy_snr,
                "Wiener_SI_SNR": wiener_sisnr,
                "Wiener_STOI": wiener_stoi,
                "Wiener_PESQ": wiener_pesq,
                # CRN
                "CRN_SNR": crn_snr,
                "CRN_Delta_SNR": crn_snr - noisy_snr,
                "CRN_SI_SNR": crn_sisnr,
                "CRN_STOI": crn_stoi,
                "CRN_PESQ": crn_pesq,
                # DCCRN
                "DCCRN_SNR": dccrn_snr,
                "DCCRN_Delta_SNR": dccrn_snr - noisy_snr,
                "DCCRN_SI_SNR": dccrn_sisnr,
                "DCCRN_STOI": dccrn_stoi,
                "DCCRN_PESQ": dccrn_pesq,
            })

            # Save sample audio & spectrograms
            if saved_examples_count < max_save_examples:
                prefix = f"{noise_cat.lower()}_{int(target_snr)}dB_{sample_idx:03d}"
                save_audio(os.path.join(output_dir, "clean", f"{prefix}_clean.wav"), clean_np, sr=sr)
                save_audio(os.path.join(output_dir, "noisy", f"{prefix}_noisy.wav"), noisy_np, sr=sr)
                save_audio(os.path.join(output_dir, "wiener", f"{prefix}_wiener.wav"), wiener_np, sr=sr)
                save_audio(os.path.join(output_dir, "crn", f"{prefix}_crn.wav"), crn_np, sr=sr)
                save_audio(os.path.join(output_dir, "dccrn", f"{prefix}_dccrn.wav"), dccrn_np, sr=sr)

                # Spectrogram plot
                plot_spectrograms(
                    clean=clean_np,
                    noisy=noisy_np,
                    enhanced_dict={"Wiener Filter": wiener_np, "CRN": crn_np, "DCCRN": dccrn_np},
                    sr=sr,
                    save_path=os.path.join(output_dir, f"{prefix}_spectrogram.png"),
                    title=f"Spectrogram Comparison ({noise_cat}, Input SNR = {target_snr:0.0f} dB)"
                )
                saved_examples_count += 1

            sample_idx += 1

    df_results = pd.DataFrame(records)

    # 1. Model Comparison across SNR levels
    snr_summary = df_results.groupby("SNR").mean(numeric_only=True).reset_index()
    comparison_csv_path = os.path.join(results_dir, "model_comparison.csv")
    snr_summary.to_csv(comparison_csv_path, index=False)
    print(f"[*] Saved SNR comparison to {comparison_csv_path}")

    # 2. Individual Model CSVs
    crn_cols = ["SNR", "noise_category", "Noisy_SNR", "CRN_SNR", "CRN_Delta_SNR", "Noisy_SI_SNR", "CRN_SI_SNR", "Noisy_STOI", "CRN_STOI", "Noisy_PESQ", "CRN_PESQ"]
    crn_csv_path = os.path.join(results_dir, "crn_results.csv")
    df_results[crn_cols].to_csv(crn_csv_path, index=False)
    print(f"[*] Saved CRN results to {crn_csv_path}")

    dccrn_cols = ["SNR", "noise_category", "Noisy_SNR", "DCCRN_SNR", "DCCRN_Delta_SNR", "Noisy_SI_SNR", "DCCRN_SI_SNR", "Noisy_STOI", "DCCRN_STOI", "Noisy_PESQ", "DCCRN_PESQ"]
    dccrn_csv_path = os.path.join(results_dir, "dccrn_results.csv")
    df_results[dccrn_cols].to_csv(dccrn_csv_path, index=False)
    print(f"[*] Saved DCCRN results to {dccrn_csv_path}")

    # 3. Category Comparison Table
    cat_summary = df_results.groupby("noise_category").mean(numeric_only=True).reset_index()
    cat_display_cols = ["noise_category", "Noisy_SNR", "CRN_SNR", "DCCRN_SNR", "CRN_STOI", "DCCRN_STOI", "CRN_PESQ", "DCCRN_PESQ"]
    cat_summary_out = cat_summary[[c for c in cat_display_cols if c in cat_summary.columns]]
    cat_csv_path = os.path.join(results_dir, "category_comparison.csv")
    cat_summary_out.to_csv(cat_csv_path, index=False)
    print(f"[*] Saved Category comparison to {cat_csv_path}")

    # 4. Generate Metric vs SNR curves
    plot_metric_vs_snr(snr_summary, save_dir=results_dir)

    # 5. Print Summary Table
    print("\n" + "=" * 70)
    print("                 BENCHMARK SUMMARY (OVERALL AVERAGE)")
    print("=" * 70)
    summary_cols = ["SNR", "SI_SNR", "STOI", "PESQ"]
    for m in ["Noisy", "Wiener", "CRN", "DCCRN"]:
        m_snr = df_results[f"{m}_SNR"].mean()
        m_sisnr = df_results[f"{m}_SI_SNR"].mean()
        m_stoi = df_results[f"{m}_STOI"].mean()
        m_pesq = df_results[f"{m}_PESQ"].mean()
        print(f"{m:10s} | SNR: {m_snr:6.2f} dB | SI-SNR: {m_sisnr:6.2f} dB | STOI: {m_stoi:5.3f} | PESQ: {m_pesq:5.2f}")
    print("=" * 70)

    return snr_summary, cat_summary_out


def main():
    parser = argparse.ArgumentParser(description="Evaluate speech enhancement models (CRN, DCCRN, Wiener) on MAD test set.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--crn_checkpoint", type=str, default=None, help="Path to CRN checkpoint")
    parser.add_argument("--dccrn_checkpoint", type=str, default=None, help="Path to DCCRN checkpoint")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of test samples to evaluate")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    evaluate_models(
        config=config,
        crn_checkpoint=args.crn_checkpoint,
        dccrn_checkpoint=args.dccrn_checkpoint,
        num_eval_samples=args.num_samples
    )


if __name__ == "__main__":
    main()
