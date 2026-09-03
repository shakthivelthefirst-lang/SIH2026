import os
import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import argparse
import yaml
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from models.crn import CRN
from models.dccrn import DCCRN
from datasets.mad_dataset import create_dataloaders
from losses.si_snr import SISNRLoss
from losses.spectral_loss import SpectralLoss
from losses.complex_loss import ComplexSTFTLoss
from utils.checkpoint import save_checkpoint, save_history
from utils.visualization import plot_loss_curves
from validate import validate_epoch


def train_single_model(model_type: str, config: dict, train_loader, val_loader, device: torch.device) -> dict:
    """
    Train one model (CRN or DCCRN).
    """
    print(f"\n{'='*25} STARTING TRAINING: {model_type.upper()} {'='*25}")

    # Instantiate Model
    if model_type == "crn":
        model = CRN(
            n_fft=config.get("n_fft", 512),
            hop_length=config.get("hop_length", 128),
            win_length=config.get("win_length", 512),
            lstm_layers=config.get("lstm_layers", 2),
            hidden_size=config.get("hidden_size", 256)
        ).to(device)
    elif model_type == "dccrn":
        model = DCCRN(
            n_fft=config.get("n_fft", 512),
            hop_length=config.get("hop_length", 128),
            win_length=config.get("win_length", 512),
            lstm_layers=config.get("lstm_layers", 2),
            hidden_size=config.get("hidden_size", 256)
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] {model_type.upper()} Model Parameters: {num_params:,}")

    # Optimizer, Loss functions, Scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 0.001)),
        weight_decay=float(config.get("weight_decay", 1e-5))
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    si_snr_fn = SISNRLoss()
    spectral_fn = SpectralLoss()
    complex_fn = ComplexSTFTLoss()

    use_amp = config.get("mixed_precision", True) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    epochs = config.get("epochs", 50)
    patience = config.get("patience", 10)
    clip_val = config.get("gradient_clip", 5.0)
    checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
    results_dir = config.get("results_dir", "results")

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_si_snr": [],
        "val_stoi": [],
        "val_delta_snr": []
    }

    best_val_si_snr = -float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [{model_type.upper()}]")
        for batch in pbar:
            clean = batch["clean"].to(device)
            noisy = batch["noisy"].to(device)

            optimizer.zero_grad()

            with autocast(enabled=use_amp):
                if model_type == "crn":
                    enhanced, enh_mag = model(noisy)
                    si_loss = si_snr_fn(enhanced, clean)
                    
                    tgt_spec = model.stft_module.stft(clean)
                    tgt_mag = model.stft_module.to_magnitude(tgt_spec)
                    spec_loss = spectral_fn(enh_mag, tgt_mag)
                    
                    loss = config.get("si_snr_weight", 1.0) * si_loss + config.get("spectral_weight", 0.5) * spec_loss

                elif model_type == "dccrn":
                    enhanced, enh_complex = model(noisy)
                    si_loss = si_snr_fn(enhanced, clean)

                    tgt_complex = model.stft_module.stft(clean)
                    comp_loss = complex_fn(enh_complex, tgt_complex)

                    loss = config.get("si_snr_weight", 1.0) * si_loss + config.get("complex_weight", 0.5) * comp_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * clean.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_train_loss = running_loss / len(train_loader.dataset)
        history["train_loss"].append(epoch_train_loss)

        # Validation
        val_metrics = validate_epoch(model, val_loader, model_type, device, config)
        history["val_loss"].append(val_metrics["val_loss"])
        history["val_si_snr"].append(val_metrics["val_si_snr"])
        history["val_stoi"].append(val_metrics["val_stoi"])
        history["val_delta_snr"].append(val_metrics["val_delta_snr"])

        print(f"[*] Epoch {epoch}: Train Loss = {epoch_train_loss:.4f} | Val Loss = {val_metrics['val_loss']:.4f} | "
              f"Val SI-SNR = {val_metrics['val_si_snr']:.2f} dB | Val STOI = {val_metrics['val_stoi']:.3f} | "
              f"Val Delta_SNR = {val_metrics['val_delta_snr']:.2f} dB")

        # Step scheduler on validation SI-SNR
        scheduler.step(val_metrics["val_si_snr"])

        # Checkpointing
        is_best = val_metrics["val_si_snr"] > best_val_si_snr
        if is_best:
            best_val_si_snr = val_metrics["val_si_snr"]
            patience_counter = 0
        else:
            patience_counter += 1

        state = {
            "epoch": epoch,
            "model_type": model_type,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_si_snr": val_metrics["val_si_snr"],
            "config": config
        }
        save_checkpoint(state, is_best, checkpoint_dir, model_type)

        # Early Stopping
        if patience_counter >= patience:
            print(f"[!] Early stopping triggered after {epoch} epochs (patience={patience}).")
            break

    # Save training history
    history_path = os.path.join(results_dir, f"{model_type}_history.json")
    save_history(history, history_path)
    print(f"[*] Saved training history to {history_path}")

    return history


def main():
    parser = argparse.ArgumentParser(description="Train CRN and/or DCCRN speech enhancement models on MAD dataset.")
    parser.add_argument("--model", type=str, choices=["crn", "dccrn", "both"], default="both", help="Model to train: crn, dccrn, or both")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.lr is not None:
        config["learning_rate"] = args.lr

    # Set random seed
    torch.manual_seed(config.get("seed", 42))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_csv=config.get("train_csv", "../training.csv"),
        test_csv=config.get("test_csv", "../test.csv"),
        data_dir=config.get("data_dir", "../"),
        sample_rate=config.get("sample_rate", 16000),
        segment_seconds=config.get("segment_seconds", 3.0),
        snr_levels=config.get("snr_levels", [-10, -5, 0, 5, 10, 15]),
        batch_size=config.get("batch_size", 8),
        val_split=config.get("val_split", 0.2),
        num_workers=config.get("num_workers", 0),
        seed=config.get("seed", 42)
    )

    crn_hist = None
    dccrn_hist = None

    if args.model in ["crn", "both"]:
        crn_hist = train_single_model("crn", config, train_loader, val_loader, device)

    if args.model in ["dccrn", "both"]:
        dccrn_hist = train_single_model("dccrn", config, train_loader, val_loader, device)

    # Plot training curves
    plot_loss_curves(
        crn_history=crn_hist,
        dccrn_history=dccrn_hist,
        save_path=os.path.join(config.get("results_dir", "results"), "training_curves.png")
    )

    # If 'both' was selected, perform fair comparative evaluation on identical test set
    if args.model == "both":
        print(f"\n{'='*25} RUNNING COMPARATIVE EVALUATION ON TEST SAMPLES {'='*25}")
        from test import evaluate_models
        evaluate_models(config=config, test_loader=test_loader, device=device)


if __name__ == "__main__":
    main()
