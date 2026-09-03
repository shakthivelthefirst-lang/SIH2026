import torch
import torch.nn as nn


def calculate_si_snr(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Calculate Scale-Invariant Signal-to-Noise Ratio (SI-SNR) in dB.
    Args:
        estimate: Estimated audio tensor [B, T] or [T]
        target: Target clean audio tensor [B, T] or [T]
        eps: Epsilon for numerical stability
    Returns:
        SI-SNR values in dB [B] or scalar
    """
    if estimate.dim() == 1:
        estimate = estimate.unsqueeze(0)
        target = target.unsqueeze(0)

    # Zero-mean normalization
    estimate = estimate - torch.mean(estimate, dim=-1, keepdim=True)
    target = target - torch.mean(target, dim=-1, keepdim=True)

    # Dot product along time dimension
    dot = torch.sum(estimate * target, dim=-1, keepdim=True)
    target_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + eps

    # Projection
    s_target = (dot / target_energy) * target
    e_noise = estimate - s_target

    target_norm = torch.sum(s_target ** 2, dim=-1) + eps
    noise_norm = torch.sum(e_noise ** 2, dim=-1) + eps

    si_snr = 10.0 * torch.log10(target_norm / noise_norm)
    return si_snr


class SISNRLoss(nn.Module):
    """
    Negative SI-SNR Loss for backpropagation.
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute negative mean SI-SNR loss.
        """
        si_snr = calculate_si_snr(estimate, target, eps=self.eps)
        return -torch.mean(si_snr)
