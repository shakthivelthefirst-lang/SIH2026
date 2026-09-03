import torch
import torch.nn as nn


class STFTModule(nn.Module):
    """
    Differentiable STFT / iSTFT wrapper for end-to-end training and inference.
    """
    def __init__(self, n_fft: int = 512, hop_length: int = 128, win_length: int = 512, window_type: str = "hann"):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))

    def stft(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute STFT of 1D/2D audio signal.
        Args:
            x: Tensor of shape [B, T] or [T]
        Returns:
            Complex tensor of shape [B, F, T_frames] where F = n_fft // 2 + 1 = 257.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Ensure window is on the same device and dtype as x
        window = self.window.to(dtype=x.dtype, device=x.device)
        
        # torch.stft returns complex tensor with shape [B, F, T_frames]
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True
        )
        return spec

    def istft(self, spec: torch.Tensor, length: int | None = None) -> torch.Tensor:
        """
        Compute inverse STFT.
        Args:
            spec: Complex tensor of shape [B, F, T_frames]
            length: Optional exact output length in samples to ensure exact length match.
        Returns:
            Audio waveform tensor of shape [B, T].
        """
        window = self.window.to(dtype=spec.real.dtype if spec.is_complex() else spec.dtype, device=spec.device)
        
        # If spec is not complex, assume it is [B, 2, T, F] or [B, 2, F, T]
        if not spec.is_complex():
            if spec.dim() == 4:
                # Assuming shape [B, 2, T, F] -> convert to complex [B, F, T]
                if spec.size(1) == 2:
                    real = spec[:, 0, :, :].transpose(1, 2)
                    imag = spec[:, 1, :, :].transpose(1, 2)
                    spec = torch.complex(real, imag)

        wav = torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            normalized=False,
            onesided=True,
            length=length
        )
        return wav

    @staticmethod
    def to_magnitude(spec: torch.Tensor) -> torch.Tensor:
        """
        Return magnitude spectrogram [B, 1, T_frames, F] from complex spec [B, F, T_frames].
        """
        # torch.abs(spec) gives [B, F, T] -> transpose to [B, 1, T, F]
        mag = torch.abs(spec).transpose(1, 2).unsqueeze(1)
        return mag

    @staticmethod
    def to_complex_2ch(spec: torch.Tensor) -> torch.Tensor:
        """
        Return 2-channel [B, 2, T_frames, F] representing (real, imag) from complex [B, F, T_frames].
        """
        real = spec.real.transpose(1, 2).unsqueeze(1)  # [B, 1, T, F]
        imag = spec.imag.transpose(1, 2).unsqueeze(1)  # [B, 1, T, F]
        return torch.cat([real, imag], dim=1)  # [B, 2, T, F]

    @staticmethod
    def from_complex_2ch(feat: torch.Tensor) -> torch.Tensor:
        """
        Convert 2-channel [B, 2, T_frames, F] to complex [B, F, T_frames].
        """
        real = feat[:, 0, :, :].transpose(1, 2)  # [B, F, T]
        imag = feat[:, 1, :, :].transpose(1, 2)  # [B, F, T]
        return torch.complex(real, imag)
