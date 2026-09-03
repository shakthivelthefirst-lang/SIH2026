import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexSTFTLoss(nn.Module):
    """
    Complex STFT loss comparing Real and Imaginary components, as well as complex magnitude.
    """
    def __init__(self, mag_weight: float = 1.0):
        super().__init__()
        self.mag_weight = mag_weight

    def forward(self, est_complex: torch.Tensor, tgt_complex: torch.Tensor) -> torch.Tensor:
        """
        Args:
            est_complex: Estimated complex spectrogram [B, F, T] or [B, 2, T, F]
            tgt_complex: Target complex spectrogram [B, F, T] or [B, 2, T, F]
        """
        if est_complex.is_complex():
            real_loss = F.l1_loss(est_complex.real, tgt_complex.real)
            imag_loss = F.l1_loss(est_complex.imag, tgt_complex.imag)
            mag_loss = F.l1_loss(torch.abs(est_complex), torch.abs(tgt_complex))
        elif est_complex.dim() == 4 and est_complex.size(1) == 2:
            real_loss = F.l1_loss(est_complex[:, 0], tgt_complex[:, 0])
            imag_loss = F.l1_loss(est_complex[:, 1], tgt_complex[:, 1])
            est_mag = torch.sqrt(est_complex[:, 0] ** 2 + est_complex[:, 1] ** 2 + 1e-8)
            tgt_mag = torch.sqrt(tgt_complex[:, 0] ** 2 + tgt_complex[:, 1] ** 2 + 1e-8)
            mag_loss = F.l1_loss(est_mag, tgt_mag)
        else:
            raise ValueError(f"Unsupported complex tensor shape: {est_complex.shape}")

        return (real_loss + imag_loss) + self.mag_weight * mag_loss
