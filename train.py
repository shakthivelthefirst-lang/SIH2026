"""
train.py - End-to-End Training & Validation Pipeline for Causal DPCRN.

Specifications:
- Optimizer: AdamW(lr=5e-4, weight_decay=1e-5)
- Scheduler: ReduceLROnPlateau(mode='min', factor=0.5, patience=3)
- Gradient Management: clip_grad_norm_(model.parameters(), max_norm=5.0)
- Mixed Precision: torch.cuda.amp.autocast() + GradScaler()
- Checkpointing: Saves 'best_dpcrn_checkpoint.pth' based on validation composite loss.
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SpeechEnhancementDataset
from losses import MultiDomainPerceptualLoss, calculate_sisnr
from modules import CausalDPCRN

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DPCRN_Trainer")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Causal DPCRN for Real-Time Speech Enhancement")
    parser.add_argument("--clean_dir", type=str, default="clean_speech", help="Path to clean speech directory")
    parser.add_argument("--noise_dir", type=str, default="defence_noise", help="Path to defence noise directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for AdamW")
    parser.add_argument("--grad_clip", type=float, default=5.0, help="Maximum gradient norm")
    parser.add_argument("--train_size", type=int, default=2000, help="Number of synthetic/training samples per epoch")
    parser.add_argument("--val_size", type=int, default=200, help="Number of validation samples per epoch")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker processes")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--checkpoint_name", type=str, default="best_dpcrn_checkpoint.pth", help="Checkpoint filename")
    parser.add_argument("--use_amp", action="store_true", default=True, help="Use Automatic Mixed Precision")
    parser.add_argument("--device", type=str, default="", help="Device: cuda, cpu, or auto-detect")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable remote debugging via debugpy on port 5678")
    parser.add_argument("--debug_port", type=int, default=5678, help="Remote debugger port")
    parser.add_argument("--wait_for_debugger", action="store_true", default=False, help="Wait for IDE to attach before starting training")
    return parser.parse_args()


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: MultiDomainPerceptualLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    grad_clip: float,
    use_amp: bool,
    epoch: int,
) -> dict:
    model.train()
    running_loss = 0.0
    running_sisnr = 0.0
    running_mag = 0.0
    running_cstft = 0.0
    total_batches = len(dataloader)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=False)
    for batch_idx, (noisy, clean, snr) in enumerate(pbar):
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            # Forward pass: waveform in -> enhanced waveform out
            enh_wav, enh_spec, _ = model(noisy)
            
            # STFT of clean target for multi-domain loss
            clean_spec = model.stft(clean)
            
            loss, loss_dict = criterion(
                est_wav=enh_wav,
                target_wav=clean,
                est_spec=enh_spec,
                target_spec=clean_spec,
            )

        if use_amp and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        running_loss += loss_dict["loss_total"]
        running_sisnr += loss_dict["sisnr_val"]
        running_mag += loss_dict["loss_mag"]
        running_cstft += loss_dict["loss_cstft"]

        pbar.set_postfix({
            "Loss": f"{loss_dict['loss_total']:.3f}",
            "SI-SNR": f"{loss_dict['sisnr_val']:.2f}dB",
            "Mag": f"{loss_dict['loss_mag']:.3f}",
        })

    metrics = {
        "loss_total": running_loss / total_batches,
        "sisnr": running_sisnr / total_batches,
        "loss_mag": running_mag / total_batches,
        "loss_cstft": running_cstft / total_batches,
    }
    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: MultiDomainPerceptualLoss,
    device: torch.device,
    use_amp: bool,
    epoch: int,
) -> dict:
    model.eval()
    running_loss = 0.0
    running_sisnr = 0.0
    running_sisnr_noisy = 0.0
    running_mag = 0.0
    running_cstft = 0.0
    total_batches = len(dataloader)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]", leave=False)
    for noisy, clean, snr in pbar:
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            enh_wav, enh_spec, _ = model(noisy)
            clean_spec = model.stft(clean)

            loss, loss_dict = criterion(
                est_wav=enh_wav,
                target_wav=clean,
                est_spec=enh_spec,
                target_spec=clean_spec,
            )

        # Baseline input SI-SNR
        noisy_sisnr = calculate_sisnr(noisy, clean)

        running_loss += loss_dict["loss_total"]
        running_sisnr += loss_dict["sisnr_val"]
        running_sisnr_noisy += torch.mean(noisy_sisnr).item()
        running_mag += loss_dict["loss_mag"]
        running_cstft += loss_dict["loss_cstft"]

    val_sisnr = running_sisnr / total_batches
    val_noisy_sisnr = running_sisnr_noisy / total_batches
    sisnr_improvement = val_sisnr - val_noisy_sisnr

    metrics = {
        "val_loss_total": running_loss / total_batches,
        "val_sisnr": val_sisnr,
        "val_noisy_sisnr": val_noisy_sisnr,
        "sisnr_improvement": sisnr_improvement,
        "val_loss_mag": running_mag / total_batches,
        "val_loss_cstft": running_cstft / total_batches,
    }
    return metrics


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    if args.debug:
        import debugpy
        logger.info(f"Enabling debugpy remote debugger on 0.0.0.0:{args.debug_port}...")
        debugpy.listen(("0.0.0.0", args.debug_port))
        if args.wait_for_debugger:
            logger.info("[PAUSED] Waiting for debugger to attach from IDE...")
            debugpy.wait_for_client()
            logger.info("[CONNECTED] Debugger attached!")

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using Compute Device: {device}")

    # 1. Instantiate Datasets & DataLoaders
    logger.info("Initializing Speech Enhancement Datasets...")
    train_dataset = SpeechEnhancementDataset(
        clean_dir=args.clean_dir,
        noise_dir=args.noise_dir,
        chunk_duration=3.0,
        snr_range=(-10.0, 15.0),
        synthetic_fallback=True,
        dataset_size=args.train_size,
    )
    val_dataset = SpeechEnhancementDataset(
        clean_dir=args.clean_dir,
        noise_dir=args.noise_dir,
        chunk_duration=3.0,
        snr_range=(-10.0, 15.0),
        synthetic_fallback=True,
        dataset_size=args.val_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # 2. Build Causal DPCRN Model
    logger.info("Instantiating Causal DPCRN Model...")
    model = CausalDPCRN(
        n_fft=512,
        hop_length=128,
        win_length=512,
        num_dual_path_blocks=2,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total Trainable Parameters: {total_params:,}")

    # 3. Loss, Optimizer, Scheduler, Scaler
    criterion = MultiDomainPerceptualLoss(
        alpha=10.0,
        beta=5.0,
        power_compression=0.3,
        n_fft=512,
        hop_length=128,
        win_length=512,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=args.use_amp and device.type == "cuda")

    best_val_loss = float("inf")
    checkpoint_path = Path(args.save_dir) / args.checkpoint_name

    logger.info(f"Starting Training for {args.epochs} Epochs...")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Training Phase
        train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            grad_clip=args.grad_clip,
            use_amp=args.use_amp,
            epoch=epoch,
        )

        # Validation Phase
        val_metrics = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=args.use_amp,
            epoch=epoch,
        )

        # Learning Rate Schedule Step
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_metrics["val_loss_total"])

        epoch_duration = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch:02d}/{args.epochs:02d} [{epoch_duration:.1f}s] "
            f"| LR: {current_lr:.2e} "
            f"| Train Loss: {train_metrics['loss_total']:.4f} (SI-SNR: {train_metrics['sisnr']:.2f} dB) "
            f"| Val Loss: {val_metrics['val_loss_total']:.4f} (SI-SNR: {val_metrics['val_sisnr']:.2f} dB, "
            f"Delta SI-SNR: +{val_metrics['sisnr_improvement']:.2f} dB)"
        )

        # Save Best Checkpoint
        if val_metrics["val_loss_total"] < best_val_loss:
            best_val_loss = val_metrics["val_loss_total"]
            logger.info(f"--> Validation loss improved! Saving checkpoint to {checkpoint_path}")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "val_metrics": val_metrics,
                    "config": {
                        "n_fft": 512,
                        "hop_length": 128,
                        "win_length": 512,
                        "num_dual_path_blocks": 2,
                    },
                },
                checkpoint_path,
            )

    total_time = time.time() - start_time
    logger.info(f"Training Complete in {total_time/60:.2f} minutes. Best Val Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
