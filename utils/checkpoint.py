import os
import json
import torch


def save_checkpoint(state: dict, is_best: bool, checkpoint_dir: str, model_name: str):
    """
    Save model checkpoint and best model.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_path = os.path.join(checkpoint_dir, f"{model_name}_latest.pth")
    torch.save(state, latest_path)

    if is_best:
        best_path = os.path.join(checkpoint_dir, f"{model_name}_best.pth")
        torch.save(state, best_path)
        print(f"[*] Saved new best model to {best_path} (epoch {state.get('epoch', 0)})")


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, optimizer=None, scheduler=None, scaler=None, device="cpu") -> dict:
    """
    Load model checkpoint.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return checkpoint


def save_history(history: dict, filepath: str):
    """
    Save training history to JSON.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def load_history(filepath: str) -> dict:
    """
    Load training history from JSON.
    """
    if not os.path.exists(filepath):
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
