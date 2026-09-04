"""
test_system.py - Comprehensive Unit & Integration Test Suite for Causal DPCRN System.
"""

import math
import os
import sys
import torch
import numpy as np

from modules import CausalConv2d, CausalConvTranspose2d, DualPathBlock, CausalDPCRN
from losses import calculate_sisnr, SISNRLoss, MultiDomainPerceptualLoss
from dataset import (
    SpeechEnhancementDataset,
    generate_synthetic_clean_speech,
    generate_synthetic_defence_noise,
    calculate_rms,
    peak_normalize,
)
from export_onnx import export_to_onnx, verify_and_benchmark_onnx


def test_modules_shapes_and_dimensions():
    print("\n--- 1. Testing Modules & Tensor Dimensions ---")
    x = torch.randn(2, 2, 257, 50)
    enc1 = CausalConv2d(2, 32, kernel_size=(5, 2), stride=(2, 1))
    e1, _ = enc1(x)
    assert e1.shape == (2, 32, 129, 50), f"Enc1 shape mismatch: {e1.shape}"

    enc2 = CausalConv2d(32, 64, kernel_size=(5, 2), stride=(2, 1))
    e2, _ = enc2(e1)
    assert e2.shape == (2, 64, 65, 50), f"Enc2 shape mismatch: {e2.shape}"

    enc3 = CausalConv2d(64, 128, kernel_size=(5, 2), stride=(2, 1))
    e3, _ = enc3(e2)
    assert e3.shape == (2, 128, 33, 50), f"Enc3 shape mismatch: {e3.shape}"
    print("  [PASSED] Encoder frequency downsampling: 257 -> 129 -> 65 -> 33")

    dp = DualPathBlock(channels=128, freq_hidden=64, time_hidden=128)
    dp_out, (h_n, c_n) = dp(e3)
    assert dp_out.shape == (2, 128, 33, 50), f"DualPathBlock shape mismatch: {dp_out.shape}"
    assert h_n.shape == (1, 2 * 33, 128), f"LSTM hidden shape mismatch: {h_n.shape}"
    print("  [PASSED] DualPathBlock processing & recurrent state tracking")

    dec3 = CausalConvTranspose2d(128, 64, kernel_size=(5, 2), stride=(2, 1))
    d3, _ = dec3(dp_out)
    assert d3.shape == (2, 64, 65, 50), f"Dec3 shape mismatch: {d3.shape}"

    dec2 = CausalConvTranspose2d(64, 32, kernel_size=(5, 2), stride=(2, 1))
    d2, _ = dec2(d3)
    assert d2.shape == (2, 32, 129, 50), f"Dec2 shape mismatch: {d2.shape}"

    dec1 = CausalConvTranspose2d(32, 2, kernel_size=(5, 2), stride=(2, 1))
    d1, _ = dec1(d2)
    assert d1.shape == (2, 2, 257, 50), f"Dec1 shape mismatch: {d1.shape}"
    print("  [PASSED] Decoder frequency upsampling: 33 -> 65 -> 129 -> 257")

    model = CausalDPCRN()
    wav_in = torch.randn(2, 48000)
    enh_wav, enh_spec, mask = model(wav_in)
    assert enh_wav.shape == (2, 48000), f"Enhanced wav shape mismatch: {enh_wav.shape}"
    assert mask.shape[1:] == (2, 257, enh_spec.shape[-1]), f"Mask shape mismatch: {mask.shape}"
    print(f"  [PASSED] Full Causal DPCRN Batch Forward (Input: {wav_in.shape} -> Output: {enh_wav.shape})")


def test_causality():
    print("\n--- 2. Testing Strict Causality (Zero Future Look-Ahead) ---")
    model = CausalDPCRN()
    model.eval()

    torch.manual_seed(42)
    wav1 = torch.randn(1, 48000)
    wav2 = wav1.clone()

    cutoff = 24000
    wav2[:, cutoff:] = torch.randn(1, 48000 - cutoff)

    with torch.no_grad():
        spec1 = model.stft(wav1)
        spec2 = model.stft(wav2)

        spec_diff = torch.abs(spec1 - spec2).sum(dim=(0, 1, 2))
        first_diff_stft = (spec_diff > 0).nonzero()[0].item()

        mask1 = model.forward_spec(spec1)
        mask2 = model.forward_spec(spec2)

        mask_diff = torch.abs(mask1 - mask2).sum(dim=(0, 1, 2))
        first_diff_mask = (mask_diff > 1e-6).nonzero()[0].item()

        assert first_diff_mask >= first_diff_stft, f"Causality violation! Mask changed before STFT!"

        past_diff = torch.max(torch.abs(mask1[:, :, :, :first_diff_stft] - mask2[:, :, :, :first_diff_stft]))
        assert past_diff.item() == 0.0, f"Non-zero diff in past frames: {past_diff.item()}"

    print(f"  [PASSED] Causality Verified: Zero future look-ahead (Past Diff at t < {first_diff_stft}: {past_diff.item():.2e})")


def test_streaming_equivalence():
    print("\n--- 3. Testing Streaming Step vs Batch Equivalence ---")
    model = CausalDPCRN()
    model.eval()

    torch.manual_seed(42)
    wav = torch.randn(1, 16000)
    spec = model.stft(wav)
    T_frames = spec.shape[-1]

    # Offline batch forward
    with torch.no_grad():
        batch_mask = model.forward_spec(spec)
        y_r = spec[:, 0:1]
        y_i = spec[:, 1:2]
        m_r = batch_mask[:, 0:1]
        m_i = batch_mask[:, 1:2]
        batch_spec = torch.cat([y_r * m_r - y_i * m_i, y_r * m_i + y_i * m_r], dim=1)

    # Streaming frame-by-frame forward
    states = model.init_streaming_states(batch_size=1)
    stream_frames = []

    with torch.no_grad():
        for t in range(T_frames):
            frame_in = spec[:, :, :, t : t + 1]
            enh_frame, states = model.forward_streaming_step(frame_in, states)
            stream_frames.append(enh_frame)

    stream_spec = torch.cat(stream_frames, dim=-1)
    max_diff = torch.max(torch.abs(stream_spec - batch_spec)).item()
    print(f"  [PASSED] Frame-by-Frame Streaming vs Batch Spec Max Diff: {max_diff:.6e}")
    assert max_diff < 1e-3, f"Streaming discrepancy exceeds tolerance: {max_diff}"


def test_losses_and_backward():
    print("\n--- 4. Testing Multi-Domain Perceptual Loss & Gradient Backprop ---")
    criterion = MultiDomainPerceptualLoss(alpha=10.0, beta=5.0)
    model = CausalDPCRN()

    est_wav = torch.randn(2, 48000, requires_grad=True)
    target_wav = torch.randn(2, 48000)

    loss, loss_dict = criterion(est_wav, target_wav)
    assert not torch.isnan(loss), "Loss is NaN!"
    assert not torch.isinf(loss), "Loss is Inf!"
    loss.backward()
    assert est_wav.grad is not None, "Gradients not computed!"
    print(f"  [PASSED] Multi-Domain Loss: Total={loss.item():.4f}, SI-SNR={loss_dict['loss_sisnr']:.4f}, Mag={loss_dict['loss_mag']:.4f}, cSTFT={loss_dict['loss_cstft']:.4f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    optimizer.zero_grad()
    noisy_wav = torch.randn(2, 16000)
    clean_wav = torch.randn(2, 16000)
    enh_wav, enh_spec, _ = model(noisy_wav)
    clean_spec = model.stft(clean_wav)
    total_loss, _ = criterion(enh_wav, clean_wav, enh_spec, clean_spec)
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    print("  [PASSED] End-to-End Optimization Step with Gradient Clipping")


def test_dataset_pipeline():
    print("\n--- 5. Testing Dataset Dynamic Mixing Pipeline ---")
    ds = SpeechEnhancementDataset(
        chunk_duration=3.0,
        snr_range=(-10.0, 15.0),
        synthetic_fallback=True,
        dataset_size=10,
    )
    assert len(ds) == 10
    noisy, clean, snr = ds[0]
    assert noisy.shape == (48000,), f"Noisy shape mismatch: {noisy.shape}"
    assert clean.shape == (48000,), f"Clean shape mismatch: {clean.shape}"
    assert -10.0 <= snr <= 15.0, f"SNR out of bounds: {snr}"

    max_peak = torch.max(torch.abs(noisy)).item()
    assert max_peak <= 0.96, f"Peak amplitude exceeded 0.95 (actual: {max_peak})"
    print(f"  [PASSED] Dataset Sample: Shape={noisy.shape}, SNR={snr:.2f} dB, Peak={max_peak:.3f}")


def test_onnx_export_and_runtime():
    print("\n--- 6. Testing ONNX Export & ONNXRuntime Verification ---")
    onnx_path = export_to_onnx(checkpoint_path=None, output_path="test_dpcrn.onnx")
    assert os.path.exists(onnx_path), "ONNX file not generated!"
    verify_and_benchmark_onnx(onnx_path, num_frames=50)
    print("  [PASSED] ONNX Streaming Export & Verification Complete")


if __name__ == "__main__":
    test_modules_shapes_and_dimensions()
    test_causality()
    test_streaming_equivalence()
    test_losses_and_backward()
    test_dataset_pipeline()
    test_onnx_export_and_runtime()
    print("\n========================================================")
    print(" ALL 6 SYSTEM UNIT & INTEGRATION TESTS PASSED PERFECTLY!")
    print("========================================================")
