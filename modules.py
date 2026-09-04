"""
modules.py - Modular Causal DPCRN (Dual-Path Complex Recurrent Network) Architecture
Designed for Real-Time Speech Enhancement & ANC on Edge Hardware (e.g. NVIDIA Jetson Orin).

Key Properties:
1. Strict Causality (zero future look-ahead)
2. Dual-Path Recurrent Processing (Intra-frequency BiLSTM + Inter-time Causal LSTM)
3. Complex Ratio Masking (cRM) in STFT domain
4. Streaming State Caching for Low Latency Frame-by-Frame Inference and ONNX Export
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv2d(nn.Module):
    """
    Causal 2D Convolution.
    Applies symmetric padding along the frequency axis and strictly causal (left) padding
    along the time axis to prevent future look-ahead.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (5, 2),
        stride: Tuple[int, int] = (2, 1),
        bias: bool = True,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.k_f, self.k_t = kernel_size
        self.s_f, self.s_t = stride
        self.pad_f = (self.k_f - 1) // 2

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=bias,
        )

    def forward(self, x: torch.Tensor, cache: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for both batch training and streaming inference.
        
        Args:
            x: Tensor of shape [B, C, F, T]
            cache: Optional past time frame cache of shape [B, C, F, k_t - 1]
            
        Returns:
            out: Convolved tensor of shape [B, C_out, F_out, T]
            new_cache: Updated time cache of shape [B, C, F, k_t - 1]
        """
        B, C, F_dim, T_dim = x.shape
        pad_t = self.k_t - 1

        if cache is not None:
            # Streaming mode: concatenate cached past frames
            x_time = torch.cat([cache, x], dim=-1)
            # Retain the last pad_t frames for subsequent steps
            new_cache = x_time[:, :, :, -pad_t:]
        else:
            # Batch mode: pad left with zeros in time dimension
            x_time = F.pad(x, (pad_t, 0, 0, 0))
            new_cache = x_time[:, :, :, -pad_t:]

        # Frequency padding: symmetric top and bottom
        x_padded = F.pad(x_time, (0, 0, self.pad_f, self.pad_f))
        out = self.conv(x_padded)
        return out, new_cache


class CausalConvTranspose2d(nn.Module):
    """
    Causal 2D Transposed Convolution.
    Mirrors CausalConv2d for upsampling along the frequency axis while maintaining causality in time.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (5, 2),
        stride: Tuple[int, int] = (2, 1),
        bias: bool = True,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.k_f, self.k_t = kernel_size
        self.s_f, self.s_t = stride
        self.pad_f = (self.k_f - 1) // 2

        self.deconv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(self.pad_f, 0),
            bias=bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor of shape [B, C, F, T]
            cache: Optional cache tensor of shape [B, C, F, k_t - 1]
        Returns:
            out: Upsampled tensor
            new_cache: Updated cache for streaming
        """
        T_dim = x.size(-1)
        pad_t = self.k_t - 1

        if cache is not None:
            # Streaming mode
            x_time = torch.cat([cache, x], dim=-1)
            new_cache = x_time[:, :, :, -pad_t:]
            out = self.deconv(x_time)
            # Second frame (index 1) represents causal deconv output at time t
            return out[:, :, :, 1:2], new_cache
        else:
            # Batch mode
            out = self.deconv(x)
            new_cache = x[:, :, :, -pad_t:]
            return out[:, :, :, :T_dim], new_cache


class ChannelwiseLayerNorm(nn.Module):
    """
    Channel-wise Layer Normalization for 2D tensors of shape [B, C, F, T].
    Normalizes exclusively across the channel dimension C at each (F, T) position.
    Strictly causal and frame-independent (zero look-ahead leakage).
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, F, T]
        mean = torch.mean(x, dim=1, keepdim=True)
        var = torch.var(x, dim=1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.gamma + self.beta


class DualPathBlock(nn.Module):
    """
    Dual-Path Recurrent Block:
    1. Intra-Chunk (Frequency Path): Bidirectional LSTM across frequency bins.
    2. Inter-Chunk (Time Path): Strictly Causal Unidirectional LSTM across time frames (with state caching).
    """
    def __init__(
        self,
        channels: int = 128,
        freq_hidden: int = 64,
        time_hidden: int = 128,
    ):
        super().__init__()
        self.channels = channels
        self.freq_hidden = freq_hidden
        self.time_hidden = time_hidden

        # 1. Intra-Frequency Path (BiLSTM)
        self.intra_lstm = nn.LSTM(
            input_size=channels,
            hidden_size=freq_hidden,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.intra_proj = nn.Linear(freq_hidden * 2, channels)
        self.intra_norm = nn.LayerNorm(channels)

        # 2. Inter-Time Path (Causal UniLSTM)
        self.inter_lstm = nn.LSTM(
            input_size=channels,
            hidden_size=time_hidden,
            num_layers=1,
            bidirectional=False,
            batch_first=True,
        )
        self.inter_proj = nn.Linear(time_hidden, channels)
        self.inter_norm = nn.LayerNorm(channels)

    def forward(
        self,
        x: torch.Tensor,
        inter_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Latent representation of shape [B, C, F, T]
            inter_state: Recurrent hidden/cell states for inter-time LSTM (h, c)
                         Each of shape [1, B * F, time_hidden]
        Returns:
            out: Processed tensor of shape [B, C, F, T]
            new_inter_state: Updated (h, c) for streaming
        """
        B, C, F_dim, T_dim = x.shape

        # ==========================================
        # 1. Intra-Frequency Path
        # ==========================================
        # [B, C, F, T] -> [B, T, F, C] -> [B * T, F, C]
        intra_in = x.permute(0, 3, 2, 1).contiguous().view(B * T_dim, F_dim, C)
        intra_out, _ = self.intra_lstm(intra_in)  # [B * T, F, 2 * freq_hidden]
        intra_out = self.intra_proj(intra_out)    # [B * T, F, C]
        intra_out = self.intra_norm(intra_in + intra_out)  # LayerNorm across C
        x_intra = intra_out.view(B, T_dim, F_dim, C).permute(0, 3, 2, 1)  # [B, C, F, T]

        # ==========================================
        # 2. Inter-Time Path (Causal)
        # ==========================================
        # [B, C, F, T] -> [B, F, T, C] -> [B * F, T, C]
        inter_in = x_intra.permute(0, 2, 3, 1).contiguous().view(B * F_dim, T_dim, C)
        inter_out, new_inter_state = self.inter_lstm(inter_in, inter_state)  # [B * F, T, time_hidden]
        inter_out = self.inter_proj(inter_out)  # [B * F, T, C]
        inter_out = self.inter_norm(inter_in + inter_out)  # LayerNorm across C
        out = inter_out.view(B, F_dim, T_dim, C).permute(0, 3, 1, 2)  # [B, C, F, T]

        return out, new_inter_state


class CausalDPCRN(nn.Module):
    """
    Complete Causal Dual-Path Complex Recurrent Network for Speech Enhancement.
    """
    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        num_dual_path_blocks: int = 2,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.num_dual_path_blocks = num_dual_path_blocks

        # STFT Window Buffer
        self.register_buffer("window", torch.hann_window(win_length))

        # -------------------------------------------------------------
        # 1. Complex Convolutional Encoder (Frequency: 257 -> 129 -> 65 -> 33)
        # -------------------------------------------------------------
        self.enc1 = CausalConv2d(2, 32, kernel_size=(5, 2), stride=(2, 1))
        self.bn1 = ChannelwiseLayerNorm(32)
        self.act1 = nn.PReLU()

        self.enc2 = CausalConv2d(32, 64, kernel_size=(5, 2), stride=(2, 1))
        self.bn2 = ChannelwiseLayerNorm(64)
        self.act2 = nn.PReLU()

        self.enc3 = CausalConv2d(64, 128, kernel_size=(5, 2), stride=(2, 1))
        self.bn3 = ChannelwiseLayerNorm(128)
        self.act3 = nn.PReLU()

        # -------------------------------------------------------------
        # 2. Dual-Path Recurrent Core
        # -------------------------------------------------------------
        self.dp_blocks = nn.ModuleList([
            DualPathBlock(channels=128, freq_hidden=64, time_hidden=128)
            for _ in range(num_dual_path_blocks)
        ])

        # -------------------------------------------------------------
        # 3. Complex Transposed-Convolutional Decoder (Frequency: 33 -> 65 -> 129 -> 257)
        # -------------------------------------------------------------
        self.dec3 = CausalConvTranspose2d(128, 64, kernel_size=(5, 2), stride=(2, 1))
        self.bn_dec3 = ChannelwiseLayerNorm(64)
        self.act_dec3 = nn.PReLU()

        self.dec2 = CausalConvTranspose2d(64, 32, kernel_size=(5, 2), stride=(2, 1))
        self.bn_dec2 = ChannelwiseLayerNorm(32)
        self.act_dec2 = nn.PReLU()

        self.dec1 = CausalConvTranspose2d(32, 2, kernel_size=(5, 2), stride=(2, 1))
        self.mask_act = nn.Tanh()

    def stft(self, wav: torch.Tensor) -> torch.Tensor:
        stft_res = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=True,
        )
        real = stft_res.real.unsqueeze(1)
        imag = stft_res.imag.unsqueeze(1)
        return torch.cat([real, imag], dim=1)

    def istft(self, spec: torch.Tensor, length: Optional[int] = None) -> torch.Tensor:
        real = spec[:, 0, :, :]
        imag = spec[:, 1, :, :]
        comp_spec = torch.complex(real, imag)
        wav = torch.istft(
            comp_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            length=length,
        )
        return wav

    def forward(
        self,
        noisy_wav: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L = noisy_wav.shape
        noisy_spec = self.stft(noisy_wav)
        mask = self.forward_spec(noisy_spec)

        y_real = noisy_spec[:, 0:1, :, :]
        y_imag = noisy_spec[:, 1:2, :, :]
        m_real = mask[:, 0:1, :, :]
        m_imag = mask[:, 1:2, :, :]

        s_real = y_real * m_real - y_imag * m_imag
        s_imag = y_real * m_imag + y_imag * m_real
        enhanced_spec = torch.cat([s_real, s_imag], dim=1)

        enhanced_wav = self.istft(enhanced_spec, length=L)
        return enhanced_wav, enhanced_spec, mask

    def forward_spec(self, noisy_spec: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1, _ = self.enc1(noisy_spec)
        e1 = self.act1(self.bn1(e1))

        e2, _ = self.enc2(e1)
        e2 = self.act2(self.bn2(e2))

        e3, _ = self.enc3(e2)
        e3 = self.act3(self.bn3(e3))

        # DPRNN Core
        x = e3
        for block in self.dp_blocks:
            x, _ = block(x, inter_state=None)

        # Decoder
        d3, _ = self.dec3(x)
        d3 = self.act_dec3(self.bn_dec3(d3))

        d2, _ = self.dec2(d3)
        d2 = self.act_dec2(self.bn_dec2(d2))

        d1, _ = self.dec1(d2)
        mask = self.mask_act(d1)
        return mask

    def init_streaming_states(
        self,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, torch.Tensor]:
        states = {
            # Encoder Conv caches: [B, C, F, 1]
            "conv1_cache": torch.zeros(batch_size, 2, 257, 1, device=device),
            "conv2_cache": torch.zeros(batch_size, 32, 129, 1, device=device),
            "conv3_cache": torch.zeros(batch_size, 64, 65, 1, device=device),
            # Decoder Deconv caches: [B, C, F, 1]
            "dec3_cache": torch.zeros(batch_size, 128, 33, 1, device=device),
            "dec2_cache": torch.zeros(batch_size, 64, 65, 1, device=device),
            "dec1_cache": torch.zeros(batch_size, 32, 129, 1, device=device),
        }
        # LSTM states: [1, B * 33, 128]
        for i in range(self.num_dual_path_blocks):
            states[f"h_dp_{i}"] = torch.zeros(1, batch_size * 33, 128, device=device)
            states[f"c_dp_{i}"] = torch.zeros(1, batch_size * 33, 128, device=device)
        return states

    def forward_streaming_step(
        self,
        spec_frame: torch.Tensor,
        states: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        next_states: Dict[str, torch.Tensor] = {}

        # Encoder
        e1, next_states["conv1_cache"] = self.enc1(spec_frame, states["conv1_cache"])
        e1 = self.act1(self.bn1(e1))

        e2, next_states["conv2_cache"] = self.enc2(e1, states["conv2_cache"])
        e2 = self.act2(self.bn2(e2))

        e3, next_states["conv3_cache"] = self.enc3(e2, states["conv3_cache"])
        e3 = self.act3(self.bn3(e3))

        # DPRNN
        x = e3
        for i, block in enumerate(self.dp_blocks):
            inter_state = (states[f"h_dp_{i}"], states[f"c_dp_{i}"])
            x, (h_next, c_next) = block(x, inter_state=inter_state)
            next_states[f"h_dp_{i}"] = h_next
            next_states[f"c_dp_{i}"] = c_next

        # Decoder
        d3, next_states["dec3_cache"] = self.dec3(x, states["dec3_cache"])
        d3 = self.act_dec3(self.bn_dec3(d3))

        d2, next_states["dec2_cache"] = self.dec2(d3, states["dec2_cache"])
        d2 = self.act_dec2(self.bn_dec2(d2))

        d1, next_states["dec1_cache"] = self.dec1(d2, states["dec1_cache"])
        mask = self.mask_act(d1)

        # Complex Masking
        y_real = spec_frame[:, 0:1, :, :]
        y_imag = spec_frame[:, 1:2, :, :]
        m_real = mask[:, 0:1, :, :]
        m_imag = mask[:, 1:2, :, :]

        s_real = y_real * m_real - y_imag * m_imag
        s_imag = y_real * m_imag + y_imag * m_real
        enhanced_spec_frame = torch.cat([s_real, s_imag], dim=1)

        return enhanced_spec_frame, next_states
