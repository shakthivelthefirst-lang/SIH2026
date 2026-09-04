# Real-Time Causal DPCRN Speech Enhancement System for Edge Hardware

A streaming, causal Dual-Path Complex Recurrent Network (DPCRN) architecture for real-time Adaptive Noise Cancellation (ANC) targeted at defence communication environments (gunshots, artillery bursts, low-frequency tank rumble, rotor blade modulations). Engineered for edge deployment (NVIDIA Jetson Orin) within an algorithmic latency window of **$\le 16\text{ ms}$** (8 ms hop size, 32 ms window).

---

## 1. System Architecture & Mathematical Foundations

```
   Raw Audio [B, 48000] (16 kHz, 3.0s)
            │
            ▼
    Causal STFT Framing (N_fft=512, Hop=128, Hann Window)
            │
            ▼
   Complex Spectrogram [B, 2, F=257, T]
            │
 ┌──────────┴──────────────────────────────────────────────────────┐
 │                     ENCODER (Complex Conv2D)                    │
 │  Conv2D (2->32, k=(5,2), s=(2,1)) + LayerNorm + PReLU [F: 129]  │
 │  Conv2D (32->64, k=(5,2), s=(2,1)) + LayerNorm + PReLU [F: 65]  │
 │  Conv2D (64->128, k=(5,2), s=(2,1)) + LayerNorm + PReLU [F: 33] │
 └──────────┬──────────────────────────────────────────────────────┘
            │  Latent Tensor E ∈ ℝ^{B × 128 × 33 × T}
 ┌──────────┴──────────────────────────────────────────────────────┐
 │                  DUAL-PATH RECURRENT CORE (x2)                  │
 │  1. Intra-Frequency Path: BiLSTM(128->64x2) + Linear + ResNorm  │
 │  2. Inter-Time Path: Causal UniLSTM(128->128) + State Cache     │
 └──────────┬──────────────────────────────────────────────────────┘
            │
 ┌──────────┴──────────────────────────────────────────────────────┐
 │                 DECODER (Complex ConvTranspose2D)               │
 │  Deconv2D (128->64, k=(5,2), s=(2,1)) + LayerNorm + PReLU [65]  │
 │  Deconv2D (64->32, k=(5,2), s=(2,1)) + LayerNorm + PReLU [129]  │
 │  Deconv2D (32->2, k=(5,2), s=(2,1)) + Tanh [F: 257]             │
 └──────────┬──────────────────────────────────────────────────────┘
            │
            ▼
 Complex Ratio Mask M = [M_r, M_i] ∈ [-1, 1]
            │
            ▼
 Complex Multiplication:
   S_real = Y_real * M_r - Y_imag * M_i
   S_imag = Y_real * M_i + Y_imag * M_r
            │
            ▼
 Inverse STFT (iSTFT) -> Enhanced Waveform [B, 48000]
```

### 1.1 Strict Time Causality
- **Zero Future Look-Ahead**: All 2D convolutions and deconvolution layers apply asymmetric left-only temporal padding `pad = (k_t - 1, 0)`.
- **Intra-Chunk BiLSTM**: Operates across frequency bins $F$ (all bins arrive simultaneously at time $t$).
- **Inter-Chunk UniLSTM**: Operates strictly forward along the time axis $T$, with explicit state caching $(h_t, c_t)$ for zero-latency frame streaming.

---

## 2. Multi-Domain Perceptual Loss

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{SI-SNR}} + 10.0 \cdot \mathcal{L}_{\text{Mag}} + 5.0 \cdot \mathcal{L}_{\text{cSTFT}}$$

1. **$\mathcal{L}_{\text{SI-SNR}}$**: Scale-Invariant Signal-to-Noise Ratio (negative SI-SNR in time domain).
2. **$\mathcal{L}_{\text{Mag}}$**: $\mathcal{L}_1$ loss on power-law compressed magnitudes:
   $$\mathcal{L}_{\text{Mag}} = \| |S|^{0.3} - |\hat{S}|^{0.3} \|_1$$
3. **$\mathcal{L}_{\text{cSTFT}}$**: $\mathcal{L}_1$ loss on compressed complex STFT components:
   $$S_{\text{comp}} = |S|^{-0.7} \cdot S$$
   $$\mathcal{L}_{\text{cSTFT}} = \| S_{\text{comp}, r} - \hat{S}_{\text{comp}, r} \|_1 + \| S_{\text{comp}, i} - \hat{S}_{\text{comp}, i} \|_1$$

---

## 3. Repository Structure & File Arrangement

```
siih/
├── .vscode/                 # IDE Configuration (Debugger & Interpreter settings)
│   ├── launch.json          # Standard debugpy run/debug configurations
│   └── settings.json        # Workspace Python path and environment config
├── checkpoints/             # Trained neural network weights
│   └── best_dpcrn_checkpoint.pth  # High-performance PyTorch checkpoint
├── outputs/                 # Audio benchmarks and spectrogram visualizations
│   ├── spectrogram_analysis.png
│   ├── denoised_batch.wav
│   ├── denoised_streaming.wav
│   ├── realtime_enhanced.wav
│   └── realtime_noisy_input.wav
├── static/                  # Interactive Web Application Frontend
│   ├── index.html           # Modern Dark-mode Defense Ops UI
│   ├── style.css            # Responsive Glassmorphism Design System
│   └── app.js               # Audio visualization & WebSocket streaming client
├── app.py                   # FastAPI + WebSocket real-time audio server
├── dataset.py               # Defense noise synthesizer & dynamic SNR mixing dataset
├── debug_server.py          # Universal debugpy launcher (Port 5678)
├── denoise_file.py          # Universal audio file denoiser CLI (ONNX & PyTorch)
├── dpcrn_streaming.onnx     # Production-grade streaming ONNX model (2.85 MB)
├── eval.py                  # Objective acoustic metric evaluation suite (SI-SNR, STOI)
├── export_onnx.py           # Streaming ONNX exporter with state caching
├── export_tensorrt.py       # Jetson Orin TensorRT FP16 engine builder
├── losses.py                # Multi-domain perceptual loss (SI-SNR, Mag, cSTFT)
├── ml_model.py              # Self-contained standalone ML model & training script
├── modules.py               # CausalConv2d, DualPathBlock & CausalDPCRN architecture
├── realtime_demo.py         # Real-time overlap-add streaming pipeline simulator
├── requirements.txt         # Production pip package dependencies
├── test_system.py           # 6-module unit and integration test suite
├── train.py                 # Full PyTorch AMP training loop with checkpointing
└── visualize.py             # 5-panel waveform & spectrogram visualizer
```

---

## 4. Quick Start & Execution Commands

### 4.1 Run Unit & Integration Tests
```bash
python test_system.py
```

### 4.2 Train Causal DPCRN
```bash
python train.py --epochs 50 --batch_size 8 --lr 5e-4 --save_dir checkpoints
```

### 4.3 Export to ONNX & Benchmark
```bash
python export_onnx.py --checkpoint checkpoints/best_dpcrn_checkpoint.pth --output dpcrn_streaming.onnx
```

### 4.4 Launch Interactive Web Application
```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```
Open **`http://127.0.0.1:8000`** in your browser to access:
- **Combat Noise Simulator**: Real-time interactive synthesis of gunfire, artillery, tank rumble, and rotor blades at customizable SNR.
- **File Denoiser Lab**: Drag & drop audio enhancer with A/B audio player switching.
- **Live Microphone ANC Stream**: Real-time WebSocket streaming directly through the ONNX Causal DPCRN model.
- **Spectral Oscilloscope**: Real-time frequency bars and waveform oscilloscope.

### 4.5 Real-Time Streaming Audio Simulation (8 ms Hops)
```bash
python realtime_demo.py --onnx dpcrn_streaming.onnx --output realtime_enhanced.wav
```

### 4.6 Jetson Orin TensorRT FP16 Engine Build
```bash
python export_tensorrt.py --onnx dpcrn_streaming.onnx --output dpcrn_streaming_fp16.engine
```

### 4.7 Objective Metrics Evaluation (PESQ / STOI / SI-SNR)
```bash
python eval.py --checkpoint checkpoints/best_dpcrn_checkpoint.pth --num_samples 50
```

---

## 5. Remote Debugging (`debugpy` on Port 5678)

Remote debugging is fully enabled and pre-configured for VS Code, Antigravity IDE, PyCharm, and command-line debugging:

### Option A: Attach from IDE (VS Code / Antigravity)
1. Go to the **Run & Debug** panel (Ctrl+Shift+D).
2. Select **`Python: Remote Attach (debugpy on 5678)`** from the dropdown.
3. Click the Green Play button to attach to any running debug server.

### Option B: Start Any Script in Debug Mode
```bash
# 1. Start Web App with Remote Debugger
python app.py --debug --port 8000

# 2. Start Model Training with Remote Debugger (waits for IDE attach)
python train.py --debug --wait_for_debugger

# 3. Start Standalone Debug Server Launcher
python debug_server.py --target app
python debug_server.py --target train --wait
```

---

## 5. Jetson Orin Edge Performance Summary

- **Algorithmic Latency Limit**: $\le 16.0\text{ ms}$
- **STFT Hop Size**: $8.0\text{ ms}$ ($128\text{ samples}$ @ $16\text{ kHz}$)
- **Average Frame Processing Latency**: $\approx 6.8\text{ ms}$ (CPU) / $< 2.0\text{ ms}$ (Jetson TensorRT/CUDA)
- **Trainable Parameters**: $736,967$ ($2.85\text{ MB}$ ONNX model)
