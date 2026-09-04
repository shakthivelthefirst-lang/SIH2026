"""
app.py - Production-Grade Web Application Server for Causal DPCRN Speech Enhancement.

Features:
1. REST API for Audio File Enhancement (Streaming ONNX & PyTorch Batch).
2. Live Synthetic Defence Noise Simulation & Custom SNR Mixing.
3. WebSocket Streaming Endpoint for Real-Time Microphone Low-Latency Denoising.
4. Edge Telemetry (Algorithmic Latency, Model Parameter Stats, Real-Time Factor).
"""

import base64
import io
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from dataset import (
    calculate_rms,
    generate_synthetic_clean_speech,
    generate_synthetic_defence_noise,
    peak_normalize,
)
from losses import calculate_sisnr
from modules import CausalDPCRN
from realtime_demo import RealTimeStreamingProcessor

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DPCRN_WebApp")

app = FastAPI(title="Causal DPCRN Real-Time Speech Enhancement")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure temp / output directory
STATIC_DIR = Path("static")
TEMP_DIR = Path("static/temp")
STATIC_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# Mount Static Files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize global streaming processor
ONNX_MODEL_PATH = "dpcrn_streaming.onnx"
CHECKPOINT_PATH = "checkpoints/best_dpcrn_checkpoint.pth"

processor = None
if os.path.exists(ONNX_MODEL_PATH):
    try:
        processor = RealTimeStreamingProcessor(onnx_model_path=ONNX_MODEL_PATH)
        logger.info("Loaded ONNX Streaming Processor successfully.")
    except Exception as e:
        logger.warning(f"Could not load ONNX model: {e}")


@app.get("/")
def get_index():
    return FileResponse("static/index.html")


@app.get("/api/system_status")
def get_system_status():
    """Return model architecture details, latency targets, and edge benchmarks."""
    model_params = 736967
    onnx_exists = os.path.exists(ONNX_MODEL_PATH)
    ckpt_exists = os.path.exists(CHECKPOINT_PATH)
    onnx_size_mb = os.path.getsize(ONNX_MODEL_PATH) / (1024 * 1024) if onnx_exists else 0.0

    return {
        "model_name": "Causal DPCRN (Dual-Path Complex Recurrent Network)",
        "trainable_parameters": model_params,
        "onnx_exported": onnx_exists,
        "onnx_size_mb": round(onnx_size_mb, 2),
        "checkpoint_available": ckpt_exists,
        "sampling_rate": 16000,
        "n_fft": 512,
        "hop_length": 128,
        "frame_duration_ms": 8.0,
        "algorithmic_latency_ms": 16.0,
        "target_edge_hardware": "NVIDIA Jetson Orin Nano / AGX",
        "supported_noise_types": [
            {"id": "gunshots", "name": "Gunfire & Small Arms Impulses", "desc": "High-energy transient bursts with exponential decay"},
            {"id": "artillery", "name": "Heavy Artillery & Blast Shockwaves", "desc": "Sub-bass explosive resonance and shockwave rumble"},
            {"id": "tank_rumble", "name": "Tank Diesel Engine & Treads", "desc": "Low-frequency heavy engine harmonics + track rattle"},
            {"id": "rotor_blades", "name": "Helicopter Rotor Blade Chopping", "desc": "Periodic amplitude and frequency modulated chop"},
        ],
    }


@app.post("/api/generate_defence_sample")
async def generate_defence_sample(
    noise_type: str = Form("rotor_blades"),
    snr_db: float = Form(0.0),
    duration_s: float = Form(3.0),
):
    """
    Generate synthetic clean speech + defence noise at the requested SNR,
    run the Causal DPCRN streaming enhancement, and return audio URLs and metrics.
    """
    num_samples = int(duration_s * 16000)

    # 1. Synthesize Clean Speech & Defence Noise
    clean_t = generate_synthetic_clean_speech(num_samples, 16000)
    noise_t = generate_synthetic_defence_noise(noise_type, num_samples, 16000)

    # 2. Dynamic RMS Scaling & Peak Normalization
    rms_clean = calculate_rms(clean_t)
    rms_noise = calculate_rms(noise_t)
    snr_factor = 10.0 ** (-snr_db / 20.0)
    noise_scaled = noise_t * (snr_factor * (rms_clean / (rms_noise + 1e-8)))
    noisy_t = clean_t + noise_scaled

    noisy_t = peak_normalize(noisy_t, target_peak=0.95)
    clean_t = peak_normalize(clean_t, target_peak=0.95)

    noisy_np = noisy_t.numpy()
    clean_np = clean_t.numpy()

    # 3. Process via Streaming DPCRN Processor
    global processor
    if processor is None and os.path.exists(ONNX_MODEL_PATH):
        processor = RealTimeStreamingProcessor(onnx_model_path=ONNX_MODEL_PATH)

    processor.reset_states()
    hop = 128
    num_hops = len(noisy_np) // hop
    enhanced_chunks = []
    latencies = []

    t0 = time.perf_counter()
    for i in range(num_hops):
        chunk = noisy_np[i * hop : (i + 1) * hop]
        enh_chunk, lat = processor.process_hop_frame(chunk)
        enhanced_chunks.append(enh_chunk)
        latencies.append(lat)

    t1 = time.perf_counter()
    enhanced_np = np.concatenate(enhanced_chunks)
    proc_time = t1 - t0

    # 4. Compute Metrics
    clean_torch = torch.from_numpy(clean_np[:len(enhanced_np)]).unsqueeze(0)
    noisy_torch = torch.from_numpy(noisy_np[:len(enhanced_np)]).unsqueeze(0)
    enh_torch = torch.from_numpy(enhanced_np).unsqueeze(0)

    sisnr_noisy = calculate_sisnr(noisy_torch, clean_torch).item()
    sisnr_enh = calculate_sisnr(enh_torch, clean_torch).item()
    delta_sisnr = sisnr_enh - sisnr_noisy

    # 5. Save WAV Files to static/temp
    timestamp = int(time.time() * 1000)
    clean_file = f"clean_{timestamp}.wav"
    noisy_file = f"noisy_{timestamp}.wav"
    enh_file = f"enhanced_{timestamp}.wav"

    sf.write(str(TEMP_DIR / clean_file), clean_np, 16000)
    sf.write(str(TEMP_DIR / noisy_file), noisy_np, 16000)
    sf.write(str(TEMP_DIR / enh_file), enhanced_np, 16000)

    avg_lat = np.mean(latencies[5:]) if len(latencies) > 5 else np.mean(latencies)
    rtf = proc_time / duration_s

    return {
        "success": True,
        "clean_url": f"/static/temp/{clean_file}",
        "noisy_url": f"/static/temp/{noisy_file}",
        "enhanced_url": f"/static/temp/{enh_file}",
        "metrics": {
            "input_snr_db": round(snr_db, 2),
            "input_sisnr_db": round(sisnr_noisy, 2),
            "enhanced_sisnr_db": round(sisnr_enh, 2),
            "delta_sisnr_db": round(delta_sisnr, 2),
            "avg_latency_ms": round(float(avg_lat), 2),
            "rtf": round(float(rtf), 3),
            "frames_processed": num_hops,
        },
    }


@app.post("/api/denoise_uploaded_file")
async def denoise_uploaded_file(
    file: UploadFile = File(...),
    mode: str = Form("streaming"),
):
    """
    Denoise user uploaded audio file (WAV, FLAC, MP3, OGG).
    """
    contents = await file.read()
    bio = io.BytesIO(contents)
    wav_np, sr = sf.read(bio)

    if len(wav_np.shape) > 1:
        wav_np = np.mean(wav_np, axis=-1)

    if sr != 16000:
        wav_t = torch.from_numpy(wav_np).unsqueeze(0).float()
        resampler = torchaudio.transforms.Resample(sr, 16000)
        wav_np = resampler(wav_t).squeeze(0).numpy()

    duration_s = len(wav_np) / 16000.0

    global processor
    if processor is None and os.path.exists(ONNX_MODEL_PATH):
        processor = RealTimeStreamingProcessor(onnx_model_path=ONNX_MODEL_PATH)

    processor.reset_states()
    hop = 128
    num_hops = len(wav_np) // hop
    enhanced_chunks = []
    latencies = []

    t0 = time.perf_counter()
    for i in range(num_hops):
        chunk = wav_np[i * hop : (i + 1) * hop]
        enh_chunk, lat = processor.process_hop_frame(chunk)
        enhanced_chunks.append(enh_chunk)
        latencies.append(lat)

    # Remaining
    rem = len(wav_np) % hop
    if rem > 0:
        padded = np.zeros(hop, dtype=np.float32)
        padded[:rem] = wav_np[num_hops * hop :]
        enh_chunk, _ = processor.process_hop_frame(padded)
        enhanced_chunks.append(enh_chunk[:rem])

    t1 = time.perf_counter()
    enhanced_np = np.concatenate(enhanced_chunks)
    proc_time = t1 - t0

    timestamp = int(time.time() * 1000)
    noisy_file = f"uploaded_noisy_{timestamp}.wav"
    enh_file = f"uploaded_enhanced_{timestamp}.wav"

    sf.write(str(TEMP_DIR / noisy_file), wav_np, 16000)
    sf.write(str(TEMP_DIR / enh_file), enhanced_np, 16000)

    avg_lat = np.mean(latencies[5:]) if len(latencies) > 5 else np.mean(latencies)
    rtf = proc_time / max(0.01, duration_s)

    return {
        "success": True,
        "noisy_url": f"/static/temp/{noisy_file}",
        "enhanced_url": f"/static/temp/{enh_file}",
        "duration_s": round(duration_s, 2),
        "metrics": {
            "avg_latency_ms": round(float(avg_lat), 2),
            "proc_time_s": round(proc_time, 3),
            "rtf": round(float(rtf), 3),
            "frames_processed": num_hops,
        },
    }


@app.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    """
    Low-Latency WebSocket Streaming Endpoint for Real-Time Microphone Denoising.
    Receives raw Float32 array of 128 samples (8 ms hop), runs ONNX streaming step,
    and streams back enhanced 128 Float32 samples.
    """
    await websocket.accept()
    streamer = RealTimeStreamingProcessor(onnx_model_path=ONNX_MODEL_PATH)
    logger.info("WebSocket Streaming Client Connected.")

    try:
        while True:
            # Receive binary float32 samples (128 samples = 512 bytes)
            data = await websocket.receive_bytes()
            chunk = np.frombuffer(data, dtype=np.float32)

            if len(chunk) != 128:
                if len(chunk) < 128:
                    padded = np.zeros(128, dtype=np.float32)
                    padded[:len(chunk)] = chunk
                    chunk = padded
                else:
                    chunk = chunk[:128]

            enhanced_chunk, latency_ms = streamer.process_hop_frame(chunk)

            # Send back enhanced chunk as binary Float32 bytes
            await websocket.send_bytes(enhanced_chunk.astype(np.float32).tobytes())

    except WebSocketDisconnect:
        logger.info("WebSocket Streaming Client Disconnected.")
    except Exception as e:
        logger.error(f"WebSocket Error: {e}")


if __name__ == "__main__":
    import argparse
    import uvicorn
    
    parser = argparse.ArgumentParser(description="Run Causal DPCRN Web Application")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host IP to bind")
    parser.add_argument("--port", type=int, default=8000, help="Web server port")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable remote debugging via debugpy on port 5678")
    parser.add_argument("--debug_port", type=int, default=5678, help="Remote debugger port")
    parser.add_argument("--wait_for_debugger", action="store_true", default=False, help="Wait for IDE to attach before starting web app")
    args = parser.parse_args()

    if args.debug:
        import debugpy
        logger.info(f"Enabling debugpy remote debugger on 0.0.0.0:{args.debug_port}...")
        debugpy.listen(("0.0.0.0", args.debug_port))
        if args.wait_for_debugger:
            logger.info("[PAUSED] Waiting for debugger to attach from IDE...")
            debugpy.wait_for_client()
            logger.info("[CONNECTED] Debugger attached!")

    logger.info(f"Starting Web Application on http://{args.host}:{args.port}...")
    uvicorn.run("app:app", host=args.host, port=args.port, reload=False)
