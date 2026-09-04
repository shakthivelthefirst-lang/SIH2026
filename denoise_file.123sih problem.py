"""
denoise_file.py - Command-line Tool to Enhance / Denoise Any Audio File Using Causal DPCRN.

Supports:
1. Batch Mode (PyTorch Offline Forward Pass)
2. Streaming Mode (Frame-by-Frame ONNX / PyTorch Low Latency Stream)
"""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio

from modules import CausalDPCRN
from realtime_demo import RealTimeStreamingProcessor

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DPCRN_Denoiser")


def denoise_audio_file(
    input_path: str,
    output_path: str = "enhanced_output.wav",
    mode: str = "streaming",
    checkpoint_path: Optional[str] = "checkpoints/best_dpcrn_checkpoint.pth",
    onnx_path: Optional[str] = "dpcrn_streaming.onnx",
    device_str: str = "",
):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # 1. Load Audio via SoundFile (robust cross-platform loader)
    logger.info(f"Loading input audio from: {input_path}")
    wav_np, sr = sf.read(input_path)
    if len(wav_np.shape) > 1:
        wav_np = np.mean(wav_np, axis=-1)
    if sr != 16000:
        logger.info(f"Resampling from {sr} Hz to 16000 Hz...")
        wav_t = torch.from_numpy(wav_np).unsqueeze(0).float()
        resampler = torchaudio.transforms.Resample(sr, 16000)
        wav_np = resampler(wav_t).squeeze(0).numpy()
    wav = torch.from_numpy(wav_np).unsqueeze(0).float()
    duration_s = len(wav_np) / 16000.0
    logger.info(f"Audio Duration: {duration_s:.2f}s ({len(wav_np):,} samples @ 16 kHz)")

    if mode == "streaming" and onnx_path and os.path.exists(onnx_path):
        # Streaming Mode via ONNX Runtime
        logger.info(f"Running Real-Time Streaming Enhancement via ONNX ({onnx_path})...")
        processor = RealTimeStreamingProcessor(onnx_model_path=onnx_path)
        hop = 128
        num_hops = len(wav_np) // hop
        enhanced_chunks = []

        t0 = time.perf_counter()
        for i in range(num_hops):
            chunk = wav_np[i * hop : (i + 1) * hop]
            enh_chunk, _ = processor.process_hop_frame(chunk)
            enhanced_chunks.append(enh_chunk)

        # Remaining samples
        rem = len(wav_np) % hop
        if rem > 0:
            padded_chunk = np.zeros(hop, dtype=np.float32)
            padded_chunk[:rem] = wav_np[num_hops * hop :]
            enh_chunk, _ = processor.process_hop_frame(padded_chunk)
            enhanced_chunks.append(enh_chunk[:rem])

        t1 = time.perf_counter()
        enhanced_wav = np.concatenate(enhanced_chunks)
        proc_time = t1 - t0
        rtf = proc_time / duration_s
        logger.info(f"Streaming Processing Complete in {proc_time:.3f}s (RTF: {rtf:.3f}x)")

    else:
        # Batch Mode via PyTorch
        if device_str:
            device = torch.device(device_str)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Running Batch Enhancement via PyTorch on device: {device}...")
        model = CausalDPCRN().to(device)
        model.eval()

        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=device)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict)
            logger.info(f"Loaded weights from {checkpoint_path}")
        else:
            logger.warning("No checkpoint found; running with initialized weights.")

        t0 = time.perf_counter()
        with torch.no_grad():
            wav_t = wav.to(device)  # [1, L]
            enh_t, _, _ = model(wav_t)
            enhanced_wav = enh_t.squeeze(0).cpu().numpy()
        t1 = time.perf_counter()
        proc_time = t1 - t0
        logger.info(f"Batch Processing Complete in {proc_time:.3f}s (RTF: {proc_time/duration_s:.3f}x)")

    # 3. Save Enhanced Audio
    sf.write(output_path, enhanced_wav, 16000)
    logger.info(f"Enhanced audio successfully written to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Denoise an Audio File with Causal DPCRN")
    parser.add_argument("--input", type=str, required=True, help="Path to input audio file")
    parser.add_argument("--output", type=str, default="enhanced_output.wav", help="Output audio file path")
    parser.add_argument("--mode", type=str, choices=["streaming", "batch"], default="streaming", help="Processing mode")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_dpcrn_checkpoint.pth", help="PyTorch checkpoint")
    parser.add_argument("--onnx", type=str, default="dpcrn_streaming.onnx", help="ONNX model path")
    parser.add_argument("--device", type=str, default="", help="Device: cuda, cpu")
    args = parser.parse_args()

    denoise_audio_file(
        input_path=args.input,
        output_path=args.output,
        mode=args.mode,
        checkpoint_path=args.checkpoint,
        onnx_path=args.onnx,
        device_str=args.device,
    )
