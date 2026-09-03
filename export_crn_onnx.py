import os
import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np

from models.crn import CRN
from utils.checkpoint import load_checkpoint

try:
    import onnx
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


class CRNSpecCore(nn.Module):
    """
    CRN core neural network wrapper operating directly on 4D magnitude tensor [B, 1, T, 257],
    ideal for optimized embedded deployment (TensorRT / ONNX Runtime).
    """
    def __init__(self, crn_model: CRN):
        super().__init__()
        self.enc1 = crn_model.enc1
        self.enc2 = crn_model.enc2
        self.enc3 = crn_model.enc3
        self.enc4 = crn_model.enc4
        self.lstm = crn_model.lstm
        self.lstm_proj = crn_model.lstm_proj
        self.dec4 = crn_model.dec4
        self.dec3 = crn_model.dec3
        self.dec2 = crn_model.dec2
        self.dec1 = crn_model.dec1
        self.mask_act = crn_model.mask_act

    def forward(self, mag: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(mag)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        B, C, T, F = e4.shape
        lstm_in = e4.permute(0, 2, 1, 3).contiguous().reshape(B, T, -1)
        lstm_out, _ = self.lstm(lstm_in)
        lstm_out = self.lstm_proj(lstm_out)
        lstm_out = lstm_out.reshape(B, T, C, F).permute(0, 2, 1, 3).contiguous()

        # Decoder
        d4 = self.dec4(torch.cat([lstm_out, e4], dim=1))
        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))

        mask = self.mask_act(d1)
        return mask


def export_crn_onnx(checkpoint_path: str = None, output_onnx_path: str = "checkpoints/crn_model.onnx", config_path: str = "config.yaml"):
    """
    Export trained CRN model to ONNX format with dynamic batch and time dimensions.
    """
    if not HAS_ONNX:
        print("[!] Error: onnx or onnxruntime package not installed.")
        return

    config = {"n_fft": 512, "hop_length": 128, "win_length": 512, "lstm_layers": 2, "hidden_size": 256}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

    crn = CRN(
        n_fft=config.get("n_fft", 512),
        hop_length=config.get("hop_length", 128),
        win_length=config.get("win_length", 512),
        lstm_layers=config.get("lstm_layers", 2),
        hidden_size=config.get("hidden_size", 256)
    )

    if checkpoint_path and os.path.exists(checkpoint_path):
        load_checkpoint(checkpoint_path, crn, device="cpu")
        print(f"[*] Loaded CRN weights from {checkpoint_path}")
    else:
        print("[*] Exporting CRN architecture with initialized weights.")

    crn.eval()
    core_model = CRNSpecCore(crn)
    core_model.eval()

    # Dummy input: [Batch=1, Channels=1, Time_frames=100, Freq=257]
    dummy_input = torch.randn(1, 1, 100, 257, dtype=torch.float32)

    os.makedirs(os.path.dirname(os.path.abspath(output_onnx_path)), exist_ok=True)

    print(f"[*] Exporting CRN to ONNX: {output_onnx_path}...")
    torch.onnx.export(
        core_model,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["magnitude_spectrogram"],
        output_names=["estimated_mask"],
        dynamic_axes={
            "magnitude_spectrogram": {0: "batch_size", 2: "time_frames"},
            "estimated_mask": {0: "batch_size", 2: "time_frames"}
        },
        dynamo=False
    )

    print(f"[*] Verifying exported ONNX model...")
    onnx_model = onnx.load(output_onnx_path)
    onnx.checker.check_model(onnx_model)

    # Validate numeric parity with ONNX Runtime
    ort_session = ort.InferenceSession(output_onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)

    with torch.no_grad():
        pt_out = core_model(dummy_input).numpy()

    max_diff = np.max(np.abs(pt_out - ort_outs[0]))
    print(f"[✓] ONNX Export Successful! Maximum absolute numeric difference: {max_diff:.6e}")
    print(f"""
=== Embedded Deployment Notes ===
Target Pipeline: PyTorch -> ONNX -> TensorRT Engine -> NVIDIA Jetson / Embedded DSP
1. FP16 Optimization:
   trtexec --onnx={output_onnx_path} --saveEngine=crn_fp16.engine --fp16
2. INT8 Quantization:
   Use TensorRT INT8 calibrator with a representative calibration dataset from MAD training communication speech.
3. Note: Embedded real-time performance depends on memory bandwidth and DMA transfers on edge hardware.
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export CRN model to ONNX.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/crn_best.pth")
    parser.add_argument("--output", type=str, default="checkpoints/crn_model.onnx")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    export_crn_onnx(args.checkpoint, args.output, args.config)
