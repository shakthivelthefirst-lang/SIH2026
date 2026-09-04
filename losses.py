"""
losses.py - Multi-Domain Perceptual Loss Functions for Speech Enhancement.

Composite Objective:
L_total = L_SISNR + 10.0 * L_Mag + 5.0 * L_cSTFT

Components:
1. L_SISNR: Scale-Invariant Signal-to-Noise Ratio (negative SI-SNR in time domain)
2. L_Mag: L1 loss between power-law compressed spectral magnitudes (|S|^0.3)
3. L_cSTFT: L1 loss on real and imaginary components of compressed complex STFT
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_sisnr(
    estimate: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute Scale-Invariant Signal-to-Noise Ratio (SI-SNR) in dB.
    
    Args:
        estimate: Estimated time-domain signal [B, L]
        target: Ground-truth time-domain signal [B, L]
        eps: Small epsilon for numerical stability
        
    Returns:
        sisnr: SI-SNR values of shape [B]
    """
    # Ensure zero-mean
    estimate = estimate - torch.mean(estimate, dim=-1, keepdim=True)
    target = target - torch.mean(target, dim=-1, keepdim=True)

    # Dot products
    # s_target = (<estimate, target> / ||target||^2) * target
    dot = torch.sum(estimate * target, dim=-1, keepdim=True)
    target_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + eps
    s_target = (dot / target_energy) * target

    # e_noise = estimate - s_target
    e_noise = estimate - s_target

    s_target_energy = torch.sum(s_target ** 2, dim=-1) + eps
    e_noise_energy = torch.sum(e_noise ** 2, dim=-1) + eps

    sisnr = 10.0 * torch.log10(s_target_energy / e_noise_energy)
    return sisnr


class SISNRLoss(nn.Module):
    """
    Negative SI-SNR Loss for time-domain optimization.
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        sisnr = calculate_sisnr(estimate, target, eps=self.eps)
        return -torch.mean(sisnr)


class MultiDomainPerceptualLoss(nn.Module):
    """
    Composite Multi-Domain Perceptual Loss:
    L_total = L_SISNR + alpha * L_Mag + beta * L_cSTFT
    """
    def __init__(
        self,
        alpha: float = 10.0,
        beta: float = 5.0,
        power_compression: float = 0.3,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.power = power_compression
        self.eps = eps
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        self.sisnr_loss = SISNRLoss(eps=eps)
        self.register_buffer("window", torch.hann_window(win_length))

    def _stft(self, wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns real and imag parts [B, F, T]"""
        stft_res = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=True,
        )
        return stft_res.real, stft_res.imag

    def forward(
        self,
        est_wav: torch.Tensor,
        target_wav: torch.Tensor,
        est_spec: Optional[torch.Tensor] = None,
        target_spec: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total multi-domain loss.
        
        Args:
            est_wav: Estimated waveform [B, L]
            target_wav: Clean ground-truth waveform [B, L]
            est_spec: Optional estimated complex STFT [B, 2, F, T]
            target_spec: Optional target complex STFT [B, 2, F, T]
            
        Returns:
            total_loss: Scalar loss tensor for backprop
            loss_dict: Dictionary containing individual component losses and SI-SNR metric
        """
        # 1. Time-Domain SI-SNR Loss
        loss_sisnr = self.sisnr_loss(est_wav, target_wav)

        # 2. Extract STFT components if not provided
        if est_spec is None or target_spec is None:
            est_r, est_i = self._stft(est_wav)
            tgt_r, tgt_i = self._stft(target_wav)
        else:
            est_r, est_i = est_spec[:, 0], est_spec[:, 1]
            tgt_r, tgt_i = target_spec[:, 0], target_spec[:, 1]

        # 3. Power-law Compressed Magnitude Loss: L1(|S|^0.3 - |S_hat|^0.3)
        est_mag = torch.sqrt(est_r ** 2 + est_i ** 2 + self.eps)
        tgt_mag = torch.sqrt(tgt_r ** 2 + tgt_i ** 2 + self.eps)

        est_mag_comp = est_mag ** self.power
        tgt_mag_comp = tgt_mag ** self.power

        loss_mag = F.l1_loss(est_mag_comp, tgt_mag_comp)

        # 4. Power-law Compressed Complex STFT Loss:
        # S_comp = |S|^0.3 * exp(j * theta) = |S|^(0.3 - 1) * S = |S|^(-0.7) * S
        est_factor = est_mag ** (self.power - 1.0)
        tgt_factor = tgt_mag ** (self.power - 1.0)

        est_r_comp = est_r * est_factor
        est_i_comp = est_i * est_factor
        tgt_r_comp = tgt_r * tgt_factor
        tgt_i_comp = tgt_i * tgt_factor

        loss_cstft = F.l1_loss(est_r_comp, tgt_r_comp) + F.l1_loss(est_i_comp, tgt_i_comp)

        # 5. Composite Total Loss
        loss_total = loss_sisnr + self.alpha * loss_mag + self.beta * loss_cstft

        loss_dict = {
            "loss_total": loss_total.item(),
            "loss_sisnr": loss_sisnr.item(),
            "loss_mag": loss_mag.item(),
            "loss_cstft": loss_cstft.item(),
            "sisnr_val": (-loss_sisnr).item(),
        }

        return loss_total, loss_dict
