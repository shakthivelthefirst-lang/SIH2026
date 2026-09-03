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

from models.dccrn import DCCRN
from utils.checkpoint import load_checkpoint

try:
    import onnx
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


class DCCRNComplexCore(nn.Module):
    """
    DCCRN core neural network wrapper operating directly on Real and Imag spectrogram components:
    Inputs: yr [B, 1, T, 257], yi [B, 1, T, 257]
    Outputs: mr [B, 1, T, 257], mi [B, 1, T, 257]
    """
    def __init__(self, dccrn_model: DCCRN):
        super().__init__()
        self.enc1 = dccrn_model.enc1
        self.enc2 = dccrn_model.enc2
        self.enc3 = dccrn_model.enc3
        self.enc4 = dccrn_model.enc4
        self.lstm = dccrn_model.lstm
        self.lstm_proj = dccrn_model.lstm_proj
        self.dec4 = dccrn_model.dec4
        self.dec3 = dccrn_model.dec3
        self.dec2 = dccrn_model.dec2
        self.dec1 = dccrn_model.dec1
        self.mask_act = dccrn_model.mask_act
        self.mask_scale = dccrn_model.mask_scale

    def forward(self, yr: torch.Tensor, yi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e1_r, e1_i = self.enc1(yr, yi)
        e2_r, e2_i = self.enc2(e1_r, e1_i)
        e3_r, e3_i = self.enc3(e2_r, e2_i)
        e4_r, e4_i = self.enc4(e3_r, e3_i)

        B, C, T, F = e4_r.shape
        e4_r_flat = e4_r.permute(0, 2, 1, 3).contiguous().reshape(B, T, -1)
        e4_i_flat = e4_i.permute(0, 2, 1, 3).contiguous().reshape(B, T, -1)
        lstm_in = torch.cat([e4_r_flat, e4_i_flat], dim=-1)

        lstm_out, _ = self.lstm(lstm_in)
        lstm_out = self.lstm_proj(lstm_out)

        out_r_flat, out_i_flat = torch.chunk(lstm_out, 2, dim=-1)
        out_r = out_r_flat.reshape(B, T, C, F).permute(0, 2, 1, 3).contiguous()
        out_i = out_i_flat.reshape(B, T, C, F).permute(0, 2, 1, 3).contiguous()

        d4_r, d4_i = self.dec4(torch.cat([out_r, e4_r], dim=1), torch.cat([out_i, e4_i], dim=1))
        d3_r, d3_i = self.dec3(torch.cat([d4_r, e3_r], dim=1), torch.cat([d4_i, e3_i], dim=1))
        d2_r, d2_i = self.dec2(torch.cat([d3_r, e2_r], dim=1), torch.cat([d3_i, e2_i], dim=1))
        d1_r, d1_i = self.dec1(torch.cat([d2_r, e1_r], dim=1), torch.cat([d2_i, e1_i], dim=1))

        mr = self.mask_act(d1_r) * self.mask_scale
        mi = self.mask_act(d1_i) * self.mask_scale
        return mr, mi


def export_dccrn_onnx(checkpoint_path: str = None, output_onnx_path: str = "checkpoints/dccrn_model.onnx", config_path: str = "config.yaml"):
    """
    Export trained DCCRN model to ONNX format with dynamic batch and time dimensions.
    """
    if not HAS_ONNX:
        print("[!] Error: onnx or onnxruntime package not installed.")
        return

    config = {"n_fft": 512, "hop_length": 128, "win_length": 512, "lstm_layers": 2, "hidden_size": 256}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

    dccrn = DCCRN(
        n_fft=config.get("n_fft", 512),
        hop_length=config.get("hop_length", 128),
        win_length=config.get("win_length", 512),
        lstm_layers=config.get("lstm_layers", 2),
        hidden_size=config.get("hidden_size", 256)
    )

    if checkpoint_path and os.path.exists(checkpoint_path):
        load_checkpoint(checkpoint_path, dccrn, device="cpu")
        print(f"[*] Loaded DCCRN weights from {checkpoint_path}")
    else:
        print("[*] Exporting DCCRN architecture with initialized weights.")

    dccrn.eval()
    core_model = DCCRNComplexCore(dccrn)
    core_model.eval()

    dummy_yr = torch.randn(1, 1, 100, 257, dtype=torch.float32)
    dummy_yi = torch.randn(1, 1, 100, 257, dtype=torch.float32)

    os.makedirs(os.path.dirname(os.path.abspath(output_onnx_path)), exist_ok=True)

    print(f"[*] Exporting DCCRN to ONNX: {output_onnx_path}...")
    torch.onnx.export(
        core_model,
        (dummy_yr, dummy_yi),
        output_onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["real_spectrogram", "imag_spectrogram"],
        output_names=["mask_real", "mask_imag"],
        dynamic_axes={
            "real_spectrogram": {0: "batch_size", 2: "time_frames"},
            "imag_spectrogram": {0: "batch_size", 2: "time_frames"},
            "mask_real": {0: "batch_size", 2: "time_frames"},
            "mask_imag": {0: "batch_size", 2: "time_frames"}
        },
        dynamo=False
    )

    print(f"[*] Verifying exported ONNX model...")
    onnx_model = onnx.load(output_onnx_path)
    onnx.checker.check_model(onnx_model)

    # Validate numeric parity with ONNX Runtime
    ort_session = ort.InferenceSession(output_onnx_path)
    ort_inputs = {
        "real_spectrogram": dummy_yr.numpy(),
        "imag_spectrogram": dummy_yi.numpy()
    }
    ort_outs = ort_session.run(None, ort_inputs)

    with torch.no_grad():
        pt_mr, pt_mi = core_model(dummy_yr, dummy_yi)

    max_diff_r = np.max(np.abs(pt_mr.numpy() - ort_outs[0]))
    max_diff_i = np.max(np.abs(pt_mi.numpy() - ort_outs[1]))
    print(f"[✓] ONNX Export Successful! Max numeric diff: Real={max_diff_r:.6e}, Imag={max_diff_i:.6e}")
    print(f"""
=== Embedded Deployment Notes ===
Target Pipeline: PyTorch -> ONNX -> TensorRT Engine -> NVIDIA Jetson / Embedded DSP
1. FP16 Optimization:
   trtexec --onnx={output_onnx_path} --saveEngine=dccrn_fp16.engine --fp16
2. INT8 Quantization:
   Execute INT8 calibration on representative defence noise & communication segments.
3. Embedded Complex Processing:
   DCCRN processes real and imaginary parts in dual-stream convolutions; ensure memory alignment on edge SIMD/DSP units.
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export DCCRN model to ONNX.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/dccrn_best.pth")
    parser.add_argument("--output", type=str, default="checkpoints/dccrn_model.onnx")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    export_dccrn_onnx(args.checkpoint, args.output, args.config)
