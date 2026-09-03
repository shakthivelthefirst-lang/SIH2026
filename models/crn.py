import torch
import torch.nn as nn
from utils.stft import STFTModule


class ConvBlock(nn.Module):
    """
    2D Convolution Block: Conv2d -> BatchNorm2d -> PReLU
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple = (3, 3), stride: tuple = (1, 2), padding: tuple = (1, 1)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ConvTransBlock(nn.Module):
    """
    2D Transposed Convolution Block: ConvTranspose2d -> BatchNorm2d -> PReLU
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple = (3, 3), stride: tuple = (1, 2), padding: tuple = (1, 1), output_padding: tuple = (0, 0), act: bool = True):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding
        )
        self.bn = nn.BatchNorm2d(out_channels) if act else nn.Identity()
        self.act = nn.PReLU() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class CRN(nn.Module):
    """
    Convolutional Recurrent Network (CRN) for Magnitude-Domain Speech Enhancement.
    Architecture:
        1. STFT -> Magnitude Spectrogram [B, 1, T, 257]
        2. 4-layer CNN Encoder
        3. 2-layer LSTM Bottleneck
        4. 4-layer CNN Decoder with Skip Connections
        5. Sigmoid Spectral Mask
        6. Masked STFT -> iSTFT -> Enhanced Waveform [B, T]
    """
    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        lstm_layers: int = 2,
        hidden_size: int = 256
    ):
        super().__init__()
        self.stft_module = STFTModule(n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        # 4 Encoder Blocks (Frequency downsampling: 257 -> 129 -> 65 -> 33 -> 17)
        self.enc1 = ConvBlock(1, 32, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1))
        self.enc2 = ConvBlock(32, 64, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1))
        self.enc3 = ConvBlock(64, 128, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1))
        self.enc4 = ConvBlock(128, 256, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1))

        # Bottleneck LSTM
        # Feature size at enc4: 256 channels * 17 freq bins = 4352
        self.lstm_in_features = 256 * 17
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=self.lstm_in_features,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True
        )
        self.lstm_proj = nn.Linear(hidden_size, self.lstm_in_features)

        # 4 Decoder Blocks with Skip Connections
        self.dec4 = ConvTransBlock(256 + 256, 128, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1), output_padding=(0, 0))
        self.dec3 = ConvTransBlock(128 + 128, 64, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1), output_padding=(0, 0))
        self.dec2 = ConvTransBlock(64 + 64, 32, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1), output_padding=(0, 0))
        self.dec1 = ConvTransBlock(32 + 32, 1, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1), output_padding=(0, 0), act=False)

        self.mask_act = nn.Sigmoid()

    def forward_spec(self, mag: torch.Tensor) -> torch.Tensor:
        """
        Forward pass on magnitude spectrogram [B, 1, T, 257].
        Returns predicted mask [B, 1, T, 257].
        """
        # Encoder
        e1 = self.enc1(mag)       # [B, 32, T, 129]
        e2 = self.enc2(e1)        # [B, 64, T, 65]
        e3 = self.enc3(e2)        # [B, 128, T, 33]
        e4 = self.enc4(e3)        # [B, 256, T, 17]

        # LSTM Bottleneck
        B, C, T, F = e4.shape
        # Permute to [B, T, C*F]
        lstm_in = e4.permute(0, 2, 1, 3).contiguous().view(B, T, C * F)
        lstm_out, _ = self.lstm(lstm_in)
        lstm_out = self.lstm_proj(lstm_out)
        lstm_out = lstm_out.view(B, T, C, F).permute(0, 2, 1, 3).contiguous()  # [B, 256, T, 17]

        # Decoder with Skip Connections
        d4 = self.dec4(torch.cat([lstm_out, e4], dim=1))  # [B, 128, T, 33]
        d3 = self.dec3(torch.cat([d4, e3], dim=1))        # [B, 64, T, 65]
        d2 = self.dec2(torch.cat([d3, e2], dim=1))        # [B, 32, T, 129]
        d1 = self.dec1(torch.cat([d2, e1], dim=1))        # [B, 1, T, 257]

        # Spectral Mask
        mask = self.mask_act(d1)
        return mask

    def forward(self, noisy_audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        End-to-end forward pass:
        Args:
            noisy_audio: [B, T]
        Returns:
            enhanced_audio: [B, T]
            enhanced_mag: [B, 1, T_frames, 257]
        """
        audio_len = noisy_audio.size(-1)

        # STFT
        complex_spec = self.stft_module.stft(noisy_audio)  # [B, 257, T_frames]
        noisy_mag = self.stft_module.to_magnitude(complex_spec)  # [B, 1, T_frames, 257]

        # Predict Mask
        mask = self.forward_spec(noisy_mag)  # [B, 1, T_frames, 257]
        enhanced_mag = mask * noisy_mag      # [B, 1, T_frames, 257]

        # Enhanced STFT combining estimated magnitude and noisy phase
        # noisy_mag: [B, 1, T, F] -> transpose back to [B, F, T]
        enh_mag_ft = enhanced_mag.squeeze(1).transpose(1, 2)  # [B, 257, T_frames]
        noisy_phase = torch.angle(complex_spec)               # [B, 257, T_frames]
        enh_complex = torch.polar(enh_mag_ft, noisy_phase)    # [B, 257, T_frames]

        # Inverse STFT
        enhanced_audio = self.stft_module.istft(enh_complex, length=audio_len)

        return enhanced_audio, enhanced_mag
