import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralLoss(nn.Module):
    """
    Magnitude Spectral Loss computing L1 difference on linear and log magnitude spectrograms.
    """
    def __init__(self, log_weight: float = 0.5, eps: float = 1e-7):
        super().__init__()
        self.log_weight = log_weight
        self.eps = eps

    def forward(self, estimate_mag: torch.Tensor, target_mag: torch.Tensor) -> torch.Tensor:
        """
        Args:
            estimate_mag: Magnitude spectrogram of enhanced audio [B, ..., T, F]
            target_mag: Magnitude spectrogram of clean audio [B, ..., T, F]
        Returns:
            Scalar spectral loss
        """
        # Linear magnitude L1 loss
        lin_loss = F.l1_loss(estimate_mag, target_mag)

        # Logarithmic magnitude L1 loss
        log_est = torch.log(estimate_mag + self.eps)
        log_tgt = torch.log(target_mag + self.eps)
        log_loss = F.l1_loss(log_est, log_tgt)

        return lin_loss + self.log_weight * log_loss
