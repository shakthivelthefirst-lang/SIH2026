import torch
import numpy as np
from tqdm import tqdm
from losses.si_snr import calculate_si_snr, SISNRLoss
from losses.spectral_loss import SpectralLoss
from losses.complex_loss import ComplexSTFTLoss
from metrics.metrics import evaluate_all_metrics


def validate_epoch(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    model_type: str,
    device: torch.device,
    config: dict
) -> dict:
    """
    Run one validation epoch and compute average loss, SI-SNR, STOI, and PESQ.
    """
    model.eval()
    si_snr_fn = SISNRLoss()
    spectral_fn = SpectralLoss()
    complex_fn = ComplexSTFTLoss()

    total_loss = 0.0
    si_snr_list = []
    stoi_list = []
    delta_snr_list = []
    count = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validating {model_type.upper()}", leave=False):
            clean = batch["clean"].to(device)  # [B, T]
            noisy = batch["noisy"].to(device)  # [B, T]

            if model_type == "crn":
                enhanced, enh_mag = model(noisy)
                si_loss = si_snr_fn(enhanced, clean)
                
                # Target magnitude
                tgt_spec = model.stft_module.stft(clean)
                tgt_mag = model.stft_module.to_magnitude(tgt_spec)
                spec_loss = spectral_fn(enh_mag, tgt_mag)

                loss = config.get("si_snr_weight", 1.0) * si_loss + config.get("spectral_weight", 0.5) * spec_loss

            elif model_type == "dccrn":
                enhanced, enh_complex = model(noisy)
                si_loss = si_snr_fn(enhanced, clean)

                # Target complex spec
                tgt_complex = model.stft_module.stft(clean)
                comp_loss = complex_fn(enh_complex, tgt_complex)

                loss = config.get("si_snr_weight", 1.0) * si_loss + config.get("complex_weight", 0.5) * comp_loss

            total_loss += loss.item() * clean.size(0)

            # Compute metrics for batch
            si_snr_vals = calculate_si_snr(enhanced, clean)
            si_snr_list.extend(si_snr_vals.detach().cpu().numpy().tolist())

            # Evaluate a subset of samples with full metrics to keep validation fast
            if count < 50:
                for i in range(min(clean.size(0), 4)):
                    metrics = evaluate_all_metrics(
                        clean[i].cpu().numpy(),
                        enhanced[i].cpu().numpy(),
                        noisy[i].cpu().numpy(),
                        sr=config.get("sample_rate", 16000)
                    )
                    if not np.isnan(metrics["Enhanced_STOI"]):
                        stoi_list.append(metrics["Enhanced_STOI"])
                    delta_snr_list.append(metrics["Delta_SNR"])
                    count += 1

    avg_loss = total_loss / max(1, len(val_loader.dataset))
    avg_si_snr = float(np.mean(si_snr_list)) if si_snr_list else 0.0
    avg_stoi = float(np.mean(stoi_list)) if stoi_list else 0.0
    avg_delta_snr = float(np.mean(delta_snr_list)) if delta_snr_list else 0.0

    return {
        "val_loss": avg_loss,
        "val_si_snr": avg_si_snr,
        "val_stoi": avg_stoi,
        "val_delta_snr": avg_delta_snr
    }
