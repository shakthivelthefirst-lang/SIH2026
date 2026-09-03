import os
import time
import argparse
import yaml
import torch
import numpy as np

from models.crn import CRN
from models.dccrn import DCCRN
from utils.audio import load_audio, save_audio
from utils.checkpoint import load_checkpoint


def run_batch_inference(model: torch.nn.Module, audio: np.ndarray, device: torch.device) -> tuple[np.ndarray, float]:
    """
    Run full-file batch inference and measure execution time.
    """
    audio_t = torch.from_numpy(audio).unsqueeze(0).to(device)  # [1, T]

    # Warmup
    with torch.no_grad():
        _ = model(audio_t[:, :min(audio_t.size(1), 16000)])

    # Synchronize and measure
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    with torch.no_grad():
        enhanced_t, _ = model(audio_t)

    if device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    infer_time = end_time - start_time
    enhanced_np = enhanced_t.squeeze(0).cpu().numpy()
    return enhanced_np, infer_time


def run_streaming_inference(model: torch.nn.Module, audio: np.ndarray, chunk_size: int, hop_size: int, device: torch.device) -> tuple[np.ndarray, dict]:
    """
    Simulate streaming chunk-by-chunk processing.
    Args:
        chunk_size: Samples per chunk (e.g. 5120 for 320ms at 16kHz)
        hop_size: Hop advance in samples
    """
    audio_len = len(audio)
    enhanced_out = np.zeros(audio_len, dtype=np.float32)
    window = np.hanning(chunk_size).astype(np.float32)
    norm_acc = np.zeros(audio_len, dtype=np.float32)

    chunk_times = []

    pos = 0
    with torch.no_grad():
        while pos + chunk_size <= audio_len:
            chunk = audio[pos : pos + chunk_size]
            chunk_t = torch.from_numpy(chunk).unsqueeze(0).to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            enh_chunk_t, _ = model(chunk_t)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            chunk_times.append(t1 - t0)

            enh_chunk = enh_chunk_t.squeeze(0).cpu().numpy()
            enhanced_out[pos : pos + chunk_size] += enh_chunk * window
            norm_acc[pos : pos + chunk_size] += window

            pos += hop_size

    # Handle remaining samples
    if pos < audio_len:
        rem_len = audio_len - pos
        rem_chunk = np.pad(audio[pos:], (0, chunk_size - rem_len))
        chunk_t = torch.from_numpy(rem_chunk).unsqueeze(0).to(device)
        enh_chunk_t, _ = model(chunk_t)
        enh_chunk = enh_chunk_t.squeeze(0).cpu().numpy()[:rem_len]
        enhanced_out[pos:] += enh_chunk
        norm_acc[pos:] += 1.0

    # Normalize overlap-add
    norm_acc = np.maximum(norm_acc, 1e-6)
    enhanced_out = enhanced_out / norm_acc

    avg_chunk_proc_ms = float(np.mean(chunk_times) * 1000.0) if chunk_times else 0.0
    algorithmic_latency_ms = (chunk_size / 16000.0) * 1000.0
    total_latency_ms = algorithmic_latency_ms + avg_chunk_proc_ms

    latency_info = {
        "algorithmic_latency_ms": algorithmic_latency_ms,
        "processing_latency_ms": avg_chunk_proc_ms,
        "total_latency_ms": total_latency_ms,
        "num_chunks": len(chunk_times)
    }

    return enhanced_out, latency_info


def main():
    parser = argparse.ArgumentParser(description="Real-time speech enhancement inference for CRN and DCCRN.")
    parser.add_argument("--model", type=str, required=True, choices=["crn", "dccrn"], help="Model architecture: crn or dccrn")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Path to input noisy WAV audio file")
    parser.add_argument("--output", type=str, default="enhanced.wav", help="Path to save enhanced WAV audio file")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    parser.add_argument("--streaming", action="store_true", help="Enable streaming chunk-based inference mode")
    parser.add_argument("--chunk_ms", type=int, default=320, help="Streaming chunk size in milliseconds (e.g. 320, 640, 1000)")
    parser.add_argument("--device", type=str, default=None, help="Device to use: cuda or cpu")
    args = parser.parse_args()

    # Configuration
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {"sample_rate": 16000, "n_fft": 512, "hop_length": 128, "win_length": 512, "lstm_layers": 2, "hidden_size": 256}

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate Model
    if args.model == "crn":
        model = CRN(
            n_fft=config.get("n_fft", 512),
            hop_length=config.get("hop_length", 128),
            win_length=config.get("win_length", 512),
            lstm_layers=config.get("lstm_layers", 2),
            hidden_size=config.get("hidden_size", 256)
        ).to(device)
    else:
        model = DCCRN(
            n_fft=config.get("n_fft", 512),
            hop_length=config.get("hop_length", 128),
            win_length=config.get("win_length", 512),
            lstm_layers=config.get("lstm_layers", 2),
            hidden_size=config.get("hidden_size", 256)
        ).to(device)

    # Checkpoint resolution
    ckpt_path = args.checkpoint
    if not ckpt_path:
        default_best = os.path.join(config.get("checkpoint_dir", "checkpoints"), f"{args.model}_best.pth")
        default_latest = os.path.join(config.get("checkpoint_dir", "checkpoints"), f"{args.model}_latest.pth")
        ckpt_path = default_best if os.path.exists(default_best) else default_latest

    if os.path.exists(ckpt_path):
        load_checkpoint(ckpt_path, model, device=device)
        print(f"[*] Loaded checkpoint from {ckpt_path}")
    else:
        print(f"[!] Warning: No checkpoint found at {ckpt_path}. Running with initialized weights.")

    model.eval()

    # Load input audio
    sr = config.get("sample_rate", 16000)
    audio = load_audio(args.input, target_sr=sr)
    audio_duration = len(audio) / sr

    print(f"\n{'='*55}")
    print(f"  MODEL: {args.model.upper()} | DEVICE: {device}")
    print(f"  INPUT FILE: {args.input}")
    print(f"  AUDIO DURATION: {audio_duration:.2f} seconds ({len(audio)} samples)")
    print(f"{'='*55}")

    if args.streaming:
        chunk_samples = int(sr * (args.chunk_ms / 1000.0))
        hop_samples = chunk_samples // 2
        print(f"[*] Streaming Mode Active: Chunk = {args.chunk_ms} ms ({chunk_samples} samples), Hop = {hop_samples} samples")

        enhanced_audio, latency_info = run_streaming_inference(
            model=model,
            audio=audio,
            chunk_size=chunk_samples,
            hop_size=hop_samples,
            device=device
        )

        total_proc_time = latency_info["processing_latency_ms"] * latency_info["num_chunks"] / 1000.0
        rtf = total_proc_time / max(1e-6, audio_duration)

        print(f"\n--- LATENCY BREAKDOWN ---")
        print(f"  Algorithmic Latency : {latency_info['algorithmic_latency_ms']:6.2f} ms (Buffer window)")
        print(f"  Processing Latency  : {latency_info['processing_latency_ms']:6.2f} ms (Avg chunk compute)")
        print(f"  Total Chunk Latency : {latency_info['total_latency_ms']:6.2f} ms")
        print(f"  Real-Time Factor    : {rtf:.4f} ({'REAL-TIME ACHIEVED' if rtf < 1.0 else 'SLOWER THAN REAL-TIME'})")
        print(f"--------------------------")

    else:
        enhanced_audio, infer_time = run_batch_inference(model, audio, device)
        rtf = infer_time / max(1e-6, audio_duration)
        latency_ms = (infer_time / max(1, audio_duration)) * 1000.0

        print(f"\n--- PERFORMANCE METRICS ---")
        print(f"  Inference Time   : {infer_time*1000.0:6.2f} ms ({infer_time:.4f} s)")
        print(f"  Real-Time Factor : {rtf:.4f}")
        print(f"  Latency (RTF*1s) : {latency_ms:6.2f} ms/sec of audio")
        print(f"  Status           : {'REAL-TIME FEASIBLE' if rtf < 1.0 else 'NON-REAL-TIME'}")
        print(f"---------------------------")

    # Save output
    save_audio(args.output, enhanced_audio, sr=sr)
    print(f"[*] Saved enhanced audio to: {args.output}\n")


if __name__ == "__main__":
    main()
