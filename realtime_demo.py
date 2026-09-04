"""
realtime_demo.py - Streaming Real-Time Audio Pipeline Simulator for Edge Hardware.

Simulates real-time microphone stream input using:
1. Circular input audio ring buffer (512 samples window, 128 samples hop = 8 ms).
2. ONNX Runtime Streaming Inference Engine (evaluating single STFT frame + updating states).
3. Overlap-Add (OLA) inverse STFT buffer for real-time waveform reconstruction.
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch

from dataset import generate_synthetic_clean_speech, generate_synthetic_defence_noise, peak_normalize

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("RealTime_DPCRN_Streamer")


class RealTimeStreamingProcessor:
    """
    Real-time streaming audio enhancement engine with zero algorithmic look-ahead.
    
    Parameters:
        onnx_model_path: Path to exported ONNX streaming model
        n_fft: FFT length (512 samples = 32 ms)
        hop_length: Frame hop size (128 samples = 8 ms)
    """
    def __init__(
        self,
        onnx_model_path: str = "dpcrn_streaming.onnx",
        n_fft: int = 512,
        hop_length: int = 128,
    ):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = np.hanning(n_fft).astype(np.float32)

        # ONNX Runtime Session
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = 2

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_model_path, session_options, providers=providers)
        logger.info(f"Loaded ONNX Engine with Provider: {self.session.get_providers()[0]}")

        # Initialize State Caches
        self.reset_states()

        # Audio Buffers for Streaming STFT & Overlap-Add (OLA)
        self.in_buffer = np.zeros(self.n_fft, dtype=np.float32)
        self.out_buffer = np.zeros(self.n_fft, dtype=np.float32)

    def reset_states(self):
        """Reset internal recurrent and convolutional cache states."""
        self.states = {
            "conv1_cache_in": np.zeros((1, 2, 257, 1), dtype=np.float32),
            "conv2_cache_in": np.zeros((1, 32, 129, 1), dtype=np.float32),
            "conv3_cache_in": np.zeros((1, 64, 65, 1), dtype=np.float32),
            "h_dp_0_in": np.zeros((1, 33, 128), dtype=np.float32),
            "c_dp_0_in": np.zeros((1, 33, 128), dtype=np.float32),
            "h_dp_1_in": np.zeros((1, 33, 128), dtype=np.float32),
            "c_dp_1_in": np.zeros((1, 33, 128), dtype=np.float32),
            "dec3_cache_in": np.zeros((1, 128, 33, 1), dtype=np.float32),
            "dec2_cache_in": np.zeros((1, 64, 65, 1), dtype=np.float32),
            "dec1_cache_in": np.zeros((1, 32, 129, 1), dtype=np.float32),
        }

    def process_hop_frame(self, new_samples: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Process a single incoming chunk of hop_length samples (128 samples = 8 ms).
        
        Returns:
            enhanced_chunk: 128 enhanced samples
            latency_ms: Processing latency in milliseconds
        """
        assert len(new_samples) == self.hop_length

        t0 = time.perf_counter()

        # 1. Update Input Ring Buffer
        self.in_buffer[:-self.hop_length] = self.in_buffer[self.hop_length:]
        self.in_buffer[-self.hop_length:] = new_samples

        # 2. Windowing & Real-FFT (STFT analysis of current 512-sample frame)
        windowed = self.in_buffer * self.window
        stft_complex = np.fft.rfft(windowed, n=self.n_fft)  # Shape: (257,)

        spec_frame = np.zeros((1, 2, 257, 1), dtype=np.float32)
        spec_frame[0, 0, :, 0] = stft_complex.real
        spec_frame[0, 1, :, 0] = stft_complex.imag

        # 3. ONNX Streaming Forward Step
        ort_inputs = {"spec_frame": spec_frame, **self.states}
        ort_outputs = self.session.run(None, ort_inputs)

        # 4. Update State Caches
        enh_spec = ort_outputs[0]
        self.states["conv1_cache_in"] = ort_outputs[1]
        self.states["conv2_cache_in"] = ort_outputs[2]
        self.states["conv3_cache_in"] = ort_outputs[3]
        self.states["h_dp_0_in"] = ort_outputs[4]
        self.states["c_dp_0_in"] = ort_outputs[5]
        self.states["h_dp_1_in"] = ort_outputs[6]
        self.states["c_dp_1_in"] = ort_outputs[7]
        self.states["dec3_cache_in"] = ort_outputs[8]
        self.states["dec2_cache_in"] = ort_outputs[9]
        self.states["dec1_cache_in"] = ort_outputs[10]

        # 5. Inverse-FFT (iSTFT synthesis)
        enh_real = enh_spec[0, 0, :, 0]
        enh_imag = enh_spec[0, 1, :, 0]
        enh_complex = enh_real + 1j * enh_imag
        reconstructed_frame = np.fft.irfft(enh_complex, n=self.n_fft).astype(np.float32)

        # Apply synthesis window and Overlap-Add (OLA)
        reconstructed_frame = reconstructed_frame * self.window

        self.out_buffer += reconstructed_frame
        # Output the oldest hop_length samples
        enhanced_chunk = self.out_buffer[:self.hop_length].copy()

        # Shift output buffer left by hop_length
        self.out_buffer[:-self.hop_length] = self.out_buffer[self.hop_length:]
        self.out_buffer[-self.hop_length:] = 0.0

        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        return enhanced_chunk, latency_ms


def stream_audio_file(
    input_wav_path: Optional[str] = None,
    output_wav_path: str = "realtime_enhanced.wav",
    onnx_model_path: str = "dpcrn_streaming.onnx",
):
    """
    Stream an audio file frame-by-frame (8 ms hops) and measure real-time statistics.
    """
    processor = RealTimeStreamingProcessor(onnx_model_path=onnx_model_path)

    if input_wav_path and Path(input_wav_path).exists():
        logger.info(f"Loading input audio from {input_wav_path}...")
        audio, sr = sf.read(input_wav_path)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        if sr != 16000:
            logger.warning(f"Audio sample rate is {sr} Hz (expected 16000 Hz).")
    else:
        logger.info("Generating synthetic defence noisy speech stream (gunshots + tank rumble)...")
        clean = generate_synthetic_clean_speech(48000, 16000)
        noise = generate_synthetic_defence_noise("tank_rumble", 48000, 16000)
        audio = (clean + 0.8 * noise).numpy()
        audio = peak_normalize(torch.from_numpy(audio)).numpy()
        sf.write("realtime_noisy_input.wav", audio, 16000)
        logger.info("Saved noisy input to 'realtime_noisy_input.wav'.")

    hop = 128
    num_hops = len(audio) // hop
    enhanced_chunks = []
    latencies = []

    logger.info(f"Starting frame-by-frame streaming processing ({num_hops} frames @ 8 ms per frame)...")

    for i in range(num_hops):
        chunk = audio[i * hop : (i + 1) * hop]
        enh_chunk, lat_ms = processor.process_hop_frame(chunk)
        enhanced_chunks.append(enh_chunk)
        latencies.append(lat_ms)

    enhanced_audio = np.concatenate(enhanced_chunks)
    sf.write(output_wav_path, enhanced_audio, 16000)
    logger.info(f"Saved enhanced streaming output to '{output_wav_path}'.")

    avg_lat = np.mean(latencies[5:])
    p99_lat = np.percentile(latencies[5:], 99)
    max_lat = np.max(latencies[5:])

    logger.info("================ Real-Time Streaming Performance ================")
    logger.info(f"Frame Hop Duration     : 8.00 ms (128 samples @ 16 kHz)")
    logger.info(f"Algorithmic Latency     : 16.00 ms")
    logger.info(f"Mean Compute Latency    : {avg_lat:.2f} ms per frame")
    logger.info(f"99th Percentile Latency : {p99_lat:.2f} ms")
    logger.info(f"Peak Compute Latency    : {max_lat:.2f} ms")
    logger.info(f"Real-Time Factor (RTF)  : {avg_lat / 8.0:.3f}x")

    if avg_lat < 8.0:
        logger.info("[SUCCESS] System is FASTER than real-time (RTF < 1.0) on CPU!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Causal DPCRN Audio Streamer")
    parser.add_argument("--input", type=str, default=None, help="Input wav file path")
    parser.add_argument("--output", type=str, default="realtime_enhanced.wav", help="Output enhanced wav path")
    parser.add_argument("--onnx", type=str, default="dpcrn_streaming.onnx", help="Path to ONNX model")
    args = parser.parse_args()

    stream_audio_file(
        input_wav_path=args.input,
        output_wav_path=args.output,
        onnx_model_path=args.onnx,
    )
