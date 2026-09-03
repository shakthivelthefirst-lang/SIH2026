#!/usr/bin/env python3
"""
Hardware-Accelerated Speech Enhancement Pipeline - PyTorch Training Script
Author: Deep Learning Engineer (Edge / TinyML Specialist)
Reference: SPECIFICATION.md (SPEC-ML-RTL-001)

Model: TinySpeechMaskMLP
Hardware Target: AMD/Xilinx FPGA (Vivado Vivado HLS / RTL MAC Datapath)
- FC1: Linear(32, 32, bias=True) -> ReLU
- FC2: Linear(32, 16, bias=True) -> Sigmoid
- Strictly No BatchNorm, LayerNorm, or Dropout to guarantee deterministic 1:1 RTL mapping.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# ==============================================================================
# Model Definition (Hardware-Mapped TinyML Architecture)
# ==============================================================================

class TinySpeechMaskMLP(nn.Module):
    """
    Ultra-compact 2-layer MLP for real-time speech attenuation mask generation.
    
    Hardware Constraints:
    - Zero normalization layers (no BatchNorm, no LayerNorm)
    - Zero stochastic regularizers (no Dropout)
    - Direct mapping to DSP48 MAC slices + simple LUT / piecewise activations.
    """

    def __init__(self, in_features: int = 32, hidden_dim: int = 32, out_features: int = 16):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.out_features = out_features

        # Layer 1: Dense (32 -> 32) + ReLU
        self.fc1 = nn.Linear(in_features, hidden_dim, bias=True)
        self.relu = nn.ReLU()

        # Layer 2: Dense (32 -> 16) + Sigmoid Mask Scaling
        self.fc2 = nn.Linear(hidden_dim, out_features, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass:
        x: [Batch, 32] -> h: [Batch, 32] -> mask: [Batch, 16] in range [0.0, 1.0]
        """
        h = self.relu(self.fc1(x))
        mask = self.sigmoid(self.fc2(h))
        return mask


def print_model_summary(model: TinySpeechMaskMLP) -> None:
    """Print architectural breakdown and parameter count verified against RTL memory budget."""
    fc1_w = model.fc1.weight.numel()  # 32 * 32 = 1024
    fc1_b = model.fc1.bias.numel()    # 32
    fc1_total = fc1_w + fc1_b        # 1056

    fc2_w = model.fc2.weight.numel()  # 16 * 32 = 512
    fc2_b = model.fc2.bias.numel()    # 16
    fc2_total = fc2_w + fc2_b        # 528

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "=" * 75)
    print("TinySpeechMaskMLP Architecture Summary (FPGA Target Budget)")
    print("=" * 75)
    print(f"Layer 1 (FC1):    Dense({model.in_features} in, {model.hidden_dim} out) + ReLU")
    print(f"  - Weights (w1): {model.fc1.weight.shape[1]}x{model.fc1.weight.shape[0]} = {fc1_w:>4} params (INT8)")
    print(f"  - Biases  (b1): {fc1_b:>4} params (INT32)")
    print(f"  - Layer 1 Subtotal: {fc1_total:>4} params")
    print("-" * 75)
    print(f"Layer 2 (FC2):    Dense({model.hidden_dim} in, {model.out_features} out) + Sigmoid")
    print(f"  - Weights (w2): {model.fc2.weight.shape[1]}x{model.fc2.weight.shape[0]} = {fc2_w:>4} params (INT8)")
    print(f"  - Biases  (b2): {fc2_b:>4} params (INT32)")
    print(f"  - Layer 2 Subtotal: {fc2_total:>4} params")
    print("=" * 75)
    print(f"Total Trainable Parameters: {total_params} (Exact match to SPECIFICATION.md: 1584)")
    print("=" * 75 + "\n")


# ==============================================================================
# Data Loading & Validation
# ==============================================================================

def load_processed_dataset(
    file_path: Path,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load and validate NPZ archive containing X and Y tensors."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset archive not found: {file_path}\n"
            f"Please run 'python preprocess.py' first to generate processed features."
        )

    data = np.load(file_path)
    if "X" not in data or "Y" not in data:
        raise KeyError(f"Archive {file_path} must contain 'X' and 'Y' arrays.")

    x_np = data["X"].astype(np.float32)
    y_np = data["Y"].astype(np.float32)

    # Sanity checks on shapes and values
    if x_np.ndim != 2 or x_np.shape[1] != 32:
        raise ValueError(f"Expected X shape [N, 32], got {x_np.shape} from {file_path}")
    if y_np.ndim != 2 or y_np.shape[1] != 16:
        raise ValueError(f"Expected Y shape [N, 16], got {y_np.shape} from {file_path}")

    # Check for NaN / Inf
    if not np.all(np.isfinite(x_np)) or not np.all(np.isfinite(y_np)):
        raise ValueError(f"Non-finite (NaN or Inf) values detected in {file_path}")

    # Ensure mask is bounded [0.0, 1.0]
    y_np = np.clip(y_np, 0.0, 1.0)

    x_tensor = torch.from_numpy(x_np)
    y_tensor = torch.from_numpy(y_np)

    return x_tensor, y_tensor


def create_dataloaders(
    data_dir: Path,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders from processed NPZ files."""
    train_path = data_dir / "train_features.npz"
    val_path = data_dir / "val_features.npz"

    print(f"[DataLoader] Loading training data from: {train_path}")
    x_train, y_train = load_processed_dataset(train_path)
    print(f"  - Loaded {x_train.shape[0]} training samples.")

    print(f"[DataLoader] Loading validation data from: {val_path}")
    x_val, y_val = load_processed_dataset(val_path)
    print(f"  - Loaded {x_val.shape[0]} validation samples.")

    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=(len(train_ds) > batch_size),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, val_loader


# ==============================================================================
# Loss Curve Visualization
# ==============================================================================

def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    best_epoch: int,
    output_path: Path,
) -> None:
    """Generate and save publication-grade training and validation loss curves."""
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(9, 5.5), dpi=150)
    plt.plot(epochs, train_losses, label="Train Loss (MSE)", color="#2563EB", linewidth=2.0)
    plt.plot(epochs, val_losses, label="Val Loss (MSE)", color="#DC2626", linewidth=2.0)

    # Highlight best epoch
    best_loss = val_losses[best_epoch - 1]
    plt.axvline(
        x=best_epoch,
        color="#059669",
        linestyle="--",
        alpha=0.8,
        label=f"Best Model (Epoch {best_epoch}: {best_loss:.5f})",
    )
    plt.scatter([best_epoch], [best_loss], color="#059669", s=60, zorder=5)

    plt.title("TinySpeechMaskMLP Training & Validation Loss (MSE)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Mean Squared Error (MSE)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1", fontsize=10)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[Artifact Saved] Loss curve plot exported to: {output_path.resolve()}")


# ==============================================================================
# Training Engine
# ==============================================================================

def train_model(
    data_dir: Path,
    checkpoint_path: Path,
    plot_path: Path,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 10,
    device_str: Optional[str] = None,
    seed: int = 42,
) -> None:
    """Execute complete training loop with Adam optimizer and early stopping."""
    # Set reproducible seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Device selection
    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"[Device] Using compute device: {device}")

    # Dataloaders
    train_loader, val_loader = create_dataloaders(data_dir=data_dir, batch_size=batch_size)

    # Instantiate model
    model = TinySpeechMaskMLP(in_features=32, hidden_dim=32, out_features=16).to(device)
    print_model_summary(model)

    # Optimization objectives
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Early stopping tracking
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_state_dict: Optional[Dict[str, torch.Tensor]] = None

    train_loss_history: List[float] = []
    val_loss_history: List[float] = []

    print(f"Starting training for max {epochs} epochs (Early stopping patience: {patience})...\n")

    for epoch in range(1, epochs + 1):
        # ----------------- Training Step -----------------
        model.train()
        running_train_loss = 0.0
        num_train_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            pred_mask = model(batch_x)
            loss = criterion(pred_mask, batch_y)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            num_train_batches += 1

        avg_train_loss = running_train_loss / max(num_train_batches, 1)
        train_loss_history.append(avg_train_loss)

        # ---------------- Validation Step ----------------
        model.eval()
        running_val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                pred_mask = model(batch_x)
                loss = criterion(pred_mask, batch_y)

                running_val_loss += loss.item()
                num_val_batches += 1

        avg_val_loss = running_val_loss / max(num_val_batches, 1)
        val_loss_history.append(avg_val_loss)

        # Log epoch metrics
        improved = avg_val_loss < best_val_loss
        marker = " [BEST]" if improved else ""
        print(
            f"Epoch [{epoch:03d}/{epochs:03d}]  "
            f"Train Loss: {avg_train_loss:.6f}  |  "
            f"Val Loss: {avg_val_loss:.6f}{marker}"
        )

        # Check for improvement
        if improved:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Early Stopping] No improvement in validation loss for {patience} consecutive epochs.")
                print(f"Terminating training at epoch {epoch}.")
                break

    # ---------------- Save Best Deliverables ----------------
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if best_state_dict is not None:
        save_payload = {
            "model_state_dict": best_state_dict,
            "architecture": {
                "in_features": 32,
                "hidden_dim": 32,
                "out_features": 16,
            },
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "training_config": {
                "lr": lr,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
            },
        }
        torch.save(save_payload, checkpoint_path)
        print(f"\n[Artifact Saved] Best model checkpoint saved to: {checkpoint_path.resolve()}")
        print(f"  - Best Val Loss: {best_val_loss:.6f} achieved at Epoch {best_epoch}")

    # Plot Loss Curves
    plot_training_curves(
        train_losses=train_loss_history,
        val_losses=val_loss_history,
        best_epoch=best_epoch,
        output_path=plot_path,
    )

    print("\n[Training Pipeline Finished] Checkpoint is ready for post-training quantization and export to .mem.")


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train TinySpeechMaskMLP for FPGA Hardware Acceleration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("processed_data"),
        help="Directory containing train_features.npz and val_features.npz.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("model.pth"),
        help="File path to save the best model checkpoint (.pth).",
    )
    parser.add_argument(
        "--plot_path",
        type=Path,
        default=Path("training_loss.png"),
        help="File path to save the training/validation loss curve plot.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Mini-batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Initial learning rate for Adam optimizer.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="L2 weight decay regularization factor.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (number of epochs without val loss improvement).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device (e.g., 'cpu', 'cuda'). Defaults to auto-detection.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        train_model(
            data_dir=args.data_dir,
            checkpoint_path=args.checkpoint,
            plot_path=args.plot_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            device_str=args.device,
            seed=args.seed,
        )
    except Exception as e:
        print(f"\n[Training Error] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
