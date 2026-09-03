import torch
import torch.nn as nn
from utils.stft import STFTModule
from .complex_layers import ComplexBlock, ComplexTransBlock


class DCCRN(nn.Module):
    """
    Deep Complex Convolution Recurrent Network (DCCRN) for Complex STFT Speech Enhancement.
    Architecture:
        1. Complex STFT -> Real(Y), Imag(Y) [B, 1, T, 257]
        2. 4-layer Complex CNN Encoder (Channels: 1 -> 32 -> 64 -> 128 -> 256)
        3. Recurrent Complex Bottleneck (2-layer LSTM processing joint complex features)
        4. 4-layer Complex CNN Decoder with Complex Skip Connections
        5. Complex Ratio Mask (cRM) estimation:
           S_hat_real = M_real * Y_real - M_imag * Y_imag
           S_hat_imag = M_real * Y_imag + M_imag * Y_real
        6. Complex STFT -> iSTFT -> Enhanced Waveform [B, T]
    """
    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        lstm_layers: int = 2,
        hidden_size: int = 256,
        mask_scale: float = 1.0
    ):
        super().__init__()
        self.stft_module = STFTModule(n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.mask_scale = mask_scale

        # 4 Complex Encoder Blocks (Frequency downsampling: 257 -> 129 -> 65 -> 33 -> 17)
        self.enc1 = ComplexBlock(1, 32, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1))
        self.enc2 = ComplexBlock(32, 64, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1))
        self.enc3 = ComplexBlock(64, 128, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1))
        self.enc4 = ComplexBlock(128, 256, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1))

        # Complex Recurrent Bottleneck
        self.lstm_in_features = 256 * 17  # Features per component (real/imag)
        self.lstm = nn.LSTM(
            input_size=self.lstm_in_features * 2,  # Joint Real + Imag
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True
        )
        self.lstm_proj = nn.Linear(hidden_size, self.lstm_in_features * 2)

        # 4 Complex Decoder Blocks with Complex Skip Connections
        self.dec4 = ComplexTransBlock(256 + 256, 128, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1), output_padding=(0, 0))
        self.dec3 = ComplexTransBlock(128 + 128, 64, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1), output_padding=(0, 0))
        self.dec2 = ComplexTransBlock(64 + 64, 32, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1), output_padding=(0, 0))
        self.dec1 = ComplexTransBlock(32 + 32, 1, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1), output_padding=(0, 0), act=False)

        self.mask_act = nn.Tanh()

    def forward_spec(self, yr: torch.Tensor, yi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass on Real and Imag components: [B, 1, T, 257].
        Returns estimated complex mask (M_r, M_i): [B, 1, T, 257].
        """
        # Encoder
        e1_r, e1_i = self.enc1(yr, yi)  # [B, 32, T, 129]
        e2_r, e2_i = self.enc2(e1_r, e1_i)  # [B, 64, T, 65]
        e3_r, e3_i = self.enc3(e2_r, e2_i)  # [B, 128, T, 33]
        e4_r, e4_i = self.enc4(e3_r, e3_i)  # [B, 256, T, 17]

        # Bottleneck LSTM
        B, C, T, F = e4_r.shape
        e4_r_flat = e4_r.permute(0, 2, 1, 3).contiguous().view(B, T, C * F)
        e4_i_flat = e4_i.permute(0, 2, 1, 3).contiguous().view(B, T, C * F)
        lstm_in = torch.cat([e4_r_flat, e4_i_flat], dim=-1)  # [B, T, 2 * C * F]

        lstm_out, _ = self.lstm(lstm_in)
        lstm_out = self.lstm_proj(lstm_out)

        out_r_flat, out_i_flat = torch.chunk(lstm_out, 2, dim=-1)
        out_r = out_r_flat.view(B, T, C, F).permute(0, 2, 1, 3).contiguous()
        out_i = out_i_flat.view(B, T, C, F).permute(0, 2, 1, 3).contiguous()

        # Complex Decoder with Skip Connections
        d4_r, d4_i = self.dec4(torch.cat([out_r, e4_r], dim=1), torch.cat([out_i, e4_i], dim=1))
        d3_r, d3_i = self.dec3(torch.cat([d4_r, e3_r], dim=1), torch.cat([d4_i, e3_i], dim=1))
        d2_r, d2_i = self.dec2(torch.cat([d3_r, e2_r], dim=1), torch.cat([d3_i, e2_i], dim=1))
        d1_r, d1_i = self.dec1(torch.cat([d2_r, e1_r], dim=1), torch.cat([d2_i, e1_i], dim=1))

        # Complex Ratio Mask (bounded by mask_scale * tanh)
        mr = self.mask_act(d1_r) * self.mask_scale
        mi = self.mask_act(d1_i) * self.mask_scale

        return mr, mi

    def forward(self, noisy_audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        End-to-end forward pass for DCCRN:
        Args:
            noisy_audio: [B, T]
        Returns:
            enhanced_audio: [B, T]
            enhanced_complex: Complex spectrogram [B, 257, T_frames]
        """
        audio_len = noisy_audio.size(-1)

        # Complex STFT
        complex_spec = self.stft_module.stft(noisy_audio)  # [B, 257, T_frames]
        yr = complex_spec.real.transpose(1, 2).unsqueeze(1)  # [B, 1, T_frames, 257]
        yi = complex_spec.imag.transpose(1, 2).unsqueeze(1)  # [B, 1, T_frames, 257]

        # Predict Complex Ratio Mask
        mr, mi = self.forward_spec(yr, yi)  # [B, 1, T, 257]

        # Complex multiplication: S_hat = M * Y
        # (mr + j*mi) * (yr + j*yi) = (mr*yr - mi*yi) + j*(mr*yi + mi*yr)
        sr = mr * yr - mi * yi
        si = mr * yi + mi * yr

        # Transpose back to [B, 257, T_frames]
        sr_ft = sr.squeeze(1).transpose(1, 2)
        si_ft = si.squeeze(1).transpose(1, 2)
        enh_complex = torch.complex(sr_ft, si_ft)

        # Inverse STFT
        enhanced_audio = self.stft_module.istft(enh_complex, length=audio_len)

        return enhanced_audio, enh_complex
