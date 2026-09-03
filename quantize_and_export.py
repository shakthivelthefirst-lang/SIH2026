#!/usr/bin/env python3
"""
Hardware-Accelerated Speech Enhancement Pipeline - Quantization & Memory Export
Author: Digital ASIC / FPGA Verification Engineer
Reference: SPECIFICATION.md (SPEC-ML-RTL-001)

RTL Hardware Constraints:
- Activations / Inputs: INT8 signed ([-128, 127])
- Weights:              INT8 signed ([-128, 127])
- Biases:               INT32 signed ([-2147483648, 2147483647])
- Memory files:         Plaintext hexadecimal (.mem) for Verilog $readmemh

Deliverables:
- fpga_mem/weights_l1.mem         (1024 lines, 1 byte hex per line)
- fpga_mem/bias_l1.mem            (32 lines, 4 bytes hex per line)
- fpga_mem/weights_l2.mem         (512 lines, 1 byte hex per line)
- fpga_mem/bias_l2.mem            (16 lines, 4 bytes hex per line)
- fpga_mem/lut_sigmoid.mem        (256 lines, 1 byte hex per line)
- fpga_mem/tb_input_frame.mem     (32 lines, 1 byte hex per line)
- fpga_mem/tb_expected_output.mem (16 lines, 1 byte hex per line)
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ==============================================================================
# Model Architecture Definition
# ==============================================================================

class TinySpeechMaskMLP(nn.Module):
    """2-layer MLP matching FPGA hardware specification (1584 parameters)."""

    def __init__(self, in_features: int = 32, hidden_dim: int = 32, out_features: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_features, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.fc1(x))
        mask = self.sigmoid(self.fc2(h))
        return mask


# ==============================================================================
# Two's Complement Hex Formatting Helpers
# ==============================================================================

def to_hex_8(val: int) -> str:
    """Format signed 8-bit integer (-128..127) as 2-character 2's complement hex."""
    int_val = int(round(val))
    clamped = max(-128, min(127, int_val))
    return f"{(clamped & 0xFF):02X}"


def to_hex_32(val: int) -> str:
    """Format signed 32-bit integer as 8-character 2's complement hex."""
    int_val = int(round(val))
    clamped = max(-2147483648, min(2147483647, int_val))
    return f"{(clamped & 0xFFFFFFFF):08X}"


# ==============================================================================
# Sigmoid Lookup Table Generator
# ==============================================================================

def generate_sigmoid_lut() -> np.ndarray:
    """
    Generate 256-entry INT8 Sigmoid LUT.
    Maps 8-bit signed index [-128..127] (address 00..FF) to INT8 mask [0..127].
    Index represents input range approx [-4.0, +4.0] with step 1/32 (0.03125).
    """
    lut = np.zeros(256, dtype=np.int8)
    for addr in range(256):
        # Convert unsigned address (0..255) to signed index (-128..127)
        idx_signed = addr - 128
        # Fixed point float representation: scale 32.0 (Q3.5)
        x_float = idx_signed / 32.0
        sig_float = 1.0 / (1.0 + np.exp(-x_float))
        # Quantize to INT8 signed range [0, 127] (where 127 = 1.0 full gain)
        mask_int = int(round(sig_float * 127.0))
        lut[addr] = max(0, min(127, mask_int))
    return lut


# ==============================================================================
# Quantization & Memory Export Engine
# ==============================================================================

class ModelQuantizerExporter:
    """Performs post-training symmetric quantization and generates Verilog $readmemh files."""

    def __init__(
        self,
        model_path: Path,
        val_data_path: Path,
        output_dir: Path,
    ):
        self.model_path = model_path
        self.val_data_path = val_data_path
        self.output_dir = output_dir

    def load_weights(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, TinySpeechMaskMLP]:
        """Load floating-point parameters from PyTorch checkpoint."""
        model = TinySpeechMaskMLP(in_features=32, hidden_dim=32, out_features=16)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

        ckpt = torch.load(self.model_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        elif isinstance(ckpt, dict):
            model.load_state_dict(ckpt)
        else:
            raise ValueError(f"Unrecognized checkpoint format in {self.model_path}")

        model.eval()

        w1 = model.fc1.weight.detach().cpu().numpy()  # [32, 32]
        b1 = model.fc1.bias.detach().cpu().numpy()    # [32]
        w2 = model.fc2.weight.detach().cpu().numpy()  # [16, 32]
        b2 = model.fc2.bias.detach().cpu().numpy()    # [16]

        return w1, b1, w2, b2, model

    def load_calibration_data(self) -> np.ndarray:
        """Load validation features to determine dynamic range and calibration scales."""
        if self.val_data_path.exists():
            data = np.load(self.val_data_path)
            x_val = data["X"].astype(np.float32)
            print(f"[Calibration] Loaded {x_val.shape[0]} frames from {self.val_data_path}")
            return x_val
        else:
            print(f"[Warning] {self.val_data_path} not found. Generating synthetic calibration frame.")
            rng = np.random.default_rng(42)
            return rng.uniform(0.01, 0.8, size=(100, 32)).astype(np.float32)

    def run(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        w1, b1, w2, b2, float_model = self.load_weights()
        x_val = self.load_calibration_data()

        # ----------------------------------------------------------------------
        # 1. Weight Scale Factors (Symmetric Quantization: range [-128, 127])
        # ----------------------------------------------------------------------
        max_abs_w1 = float(np.max(np.abs(w1)))
        max_abs_w2 = float(np.max(np.abs(w2)))

        s_w1 = 127.0 / max(max_abs_w1, 1e-8)
        s_w2 = 127.0 / max(max_abs_w2, 1e-8)

        w1_int8 = np.clip(np.round(w1 * s_w1), -128, 127).astype(np.int8)
        w2_int8 = np.clip(np.round(w2 * s_w2), -128, 127).astype(np.int8)

        # ----------------------------------------------------------------------
        # 2. Input & Layer 1 Activation Scaling
        # ----------------------------------------------------------------------
        max_abs_x = float(np.max(np.abs(x_val)))
        s_x1 = 127.0 / max(max_abs_x, 1e-8)

        # Layer 1 Bias scale = S_w1 * S_x1
        s_b1 = s_w1 * s_x1
        b1_int32 = np.clip(np.round(b1 * s_b1), -2147483648, 2147483647).astype(np.int32)

        # Requantization Calibration for Layer 1 -> Layer 2
        # acc1 = sum(w1 * x) + b1
        x_val_int8 = np.clip(np.round(x_val * s_x1), -128, 127).astype(np.int32)
        acc1_val = np.dot(x_val_int8, w1_int8.T) + b1_int32
        relu1_val = np.maximum(0, acc1_val)

        max_relu1 = float(np.max(relu1_val))
        # Target: fit max ReLU activation into INT8 [0, 127]
        shift_1 = max(0, int(np.ceil(np.log2(max(max_relu1 / 127.0, 1.0)))))
        s_x2 = (s_w1 * s_x1) / (2.0 ** shift_1)

        # ----------------------------------------------------------------------
        # 3. Layer 2 Biases & Output Scaling
        # ----------------------------------------------------------------------
        # Layer 2 Bias scale = S_w2 * S_x2
        s_b2 = s_w2 * s_x2
        b2_int32 = np.clip(np.round(b2 * s_b2), -2147483648, 2147483647).astype(np.int32)

        # Calibrate Layer 2 Accumulator -> Sigmoid LUT input index
        act1_val_int8 = np.clip(relu1_val >> shift_1, 0, 127).astype(np.int32)
        acc2_val = np.dot(act1_val_int8, w2_int8.T) + b2_int32

        # Accumulator scale in float: S_acc2 = S_w2 * S_x2
        s_acc2 = s_w2 * s_x2
        # Map float logit scale to LUT step (where 32 units = 1.0 logit):
        # We need (acc2 >> shift_2) to be in range [-128, 127] corresponding to [-4.0, +4.0]
        shift_2 = max(0, int(round(np.log2(max(s_acc2 / 32.0, 1.0)))))

        # ----------------------------------------------------------------------
        # 4. Export Verilog $readmemh Memory Files
        # ----------------------------------------------------------------------
        # Weights L1: 32x32 = 1024 lines (row-major: neuron 0..31, input 0..31)
        w1_file = self.output_dir / "weights_l1.mem"
        with open(w1_file, "w") as f:
            for neuron_idx in range(32):
                for in_idx in range(32):
                    f.write(f"{to_hex_8(w1_int8[neuron_idx, in_idx])}\n")

        # Bias L1: 32 lines (INT32)
        b1_file = self.output_dir / "bias_l1.mem"
        with open(b1_file, "w") as f:
            for neuron_idx in range(32):
                f.write(f"{to_hex_32(b1_int32[neuron_idx])}\n")

        # Weights L2: 16x32 = 512 lines (row-major: neuron 0..15, input 0..31)
        w2_file = self.output_dir / "weights_l2.mem"
        with open(w2_file, "w") as f:
            for neuron_idx in range(16):
                for in_idx in range(32):
                    f.write(f"{to_hex_8(w2_int8[neuron_idx, in_idx])}\n")

        # Bias L2: 16 lines (INT32)
        b2_file = self.output_dir / "bias_l2.mem"
        with open(b2_file, "w") as f:
            for neuron_idx in range(16):
                f.write(f"{to_hex_32(b2_int32[neuron_idx])}\n")

        # Sigmoid LUT: 256 lines (INT8)
        sigmoid_lut = generate_sigmoid_lut()
        lut_file = self.output_dir / "lut_sigmoid.mem"
        with open(lut_file, "w") as f:
            for val in sigmoid_lut:
                f.write(f"{to_hex_8(val)}\n")

        # ----------------------------------------------------------------------
        # 5. Golden Vectors for Vivado Testbench
        # ----------------------------------------------------------------------
        # Select first validation frame
        first_frame = x_val[0]  # [32]
        tb_input_int8 = np.clip(np.round(first_frame * s_x1), -128, 127).astype(np.int8)

        # Pure bit-exact integer forward pass:
        # Layer 1 MAC
        tb_acc1 = np.dot(w1_int8.astype(np.int32), tb_input_int8.astype(np.int32)) + b1_int32
        # Layer 1 ReLU
        tb_relu1 = np.maximum(0, tb_acc1)
        # Layer 1 Arithmetic Right Shift (>>> shift_1)
        tb_act1 = np.clip(tb_relu1 >> shift_1, 0, 127).astype(np.int8)

        # Layer 2 MAC
        tb_acc2 = np.dot(w2_int8.astype(np.int32), tb_act1.astype(np.int32)) + b2_int32
        # Layer 2 Arithmetic Right Shift (>>> shift_2) & LUT address conversion
        tb_idx = np.clip(tb_acc2 >> shift_2, -128, 127)
        # Address index: signed -128..127 + 128 -> unsigned 0..255
        tb_lut_addr = (tb_idx + 128).astype(np.uint8)
        tb_expected_output = sigmoid_lut[tb_lut_addr]  # [16]

        # Export testbench files
        input_tb_file = self.output_dir / "tb_input_frame.mem"
        with open(input_tb_file, "w") as f:
            for v in tb_input_int8:
                f.write(f"{to_hex_8(v)}\n")

        output_tb_file = self.output_dir / "tb_expected_output.mem"
        with open(output_tb_file, "w") as f:
            for v in tb_expected_output:
                f.write(f"{to_hex_8(v)}\n")

        # ----------------------------------------------------------------------
        # 6. Verification & Fidelity Report
        # ----------------------------------------------------------------------
        # Compare against PyTorch float forward pass
        with torch.no_grad():
            float_in = torch.from_numpy(first_frame).unsqueeze(0)
            float_pred = float_model(float_in).squeeze(0).numpy()
            # Scale float prediction [0.0, 1.0] to [0, 127]
            float_pred_int8 = np.clip(np.round(float_pred * 127.0), 0, 127).astype(np.int8)

        error_lsb = np.abs(tb_expected_output.astype(np.int32) - float_pred_int8.astype(np.int32))
        max_lsb_error = int(np.max(error_lsb))
        mean_lsb_error = float(np.mean(error_lsb))

        # ----------------------------------------------------------------------
        # 7. Print Hardware Specifications & Parameter Table
        # ----------------------------------------------------------------------
        print("\n" + "=" * 80)
        print("ML-to-RTL Quantization & Verilog Memory Export Complete")
        print("=" * 80)
        print(f"Memory Output Directory:     {self.output_dir.resolve()}\n")

        print("Quantization Scale Factors:")
        print(f"  - S_w1 (Layer 1 Weights):   {s_w1:>12.4f}  (Max |W1|: {max_abs_w1:.4f})")
        print(f"  - S_x1 (Input Features):    {s_x1:>12.4f}  (Max |X|:  {max_abs_x:.4f})")
        print(f"  - S_b1 (Layer 1 Biases):    {s_b1:>12.4f}  (S_w1 * S_x1)")
        print(f"  - SHIFT_1 (L1 Acc Shift):   {shift_1:>12d}  (Right shift >>> {shift_1})")
        print(f"  - S_x2 (Layer 2 Inputs):    {s_x2:>12.4f}  ((S_w1 * S_x1) >> {shift_1})")
        print(f"  - S_w2 (Layer 2 Weights):   {s_w2:>12.4f}  (Max |W2|: {max_abs_w2:.4f})")
        print(f"  - S_b2 (Layer 2 Biases):    {s_b2:>12.4f}  (S_w2 * S_x2)")
        print(f"  - SHIFT_2 (L2 Acc Shift):   {shift_2:>12d}  (Right shift >>> {shift_2})\n")

        print("Memory Files Generated:")
        print(f"  [1] {w1_file.name:<25} (1024 bytes,  INT8,  8-bit hex)")
        print(f"  [2] {b1_file.name:<25} (  32 words, INT32, 32-bit hex)")
        print(f"  [3] {w2_file.name:<25} ( 512 bytes,  INT8,  8-bit hex)")
        print(f"  [4] {b2_file.name:<25} (  16 words, INT32, 32-bit hex)")
        print(f"  [5] {lut_file.name:<25} ( 256 bytes,  INT8, Sigmoid LUT)")
        print(f"  [6] {input_tb_file.name:<25} (  32 bytes, Testbench Stimulus)")
        print(f"  [7] {output_tb_file.name:<25} (  16 bytes, Testbench Golden Expected Output)\n")

        print("FPGA RTL Parameter Verilog Header:")
        print("--------------------------------------------------------------------------------")
        print(f"localparam L1_SHIFT = {shift_1};")
        print(f"localparam L2_SHIFT = {shift_2};")
        print("--------------------------------------------------------------------------------\n")

        print("Quantization Fidelity vs Floating-Point PyTorch:")
        print(f"  - Max LSB Discrepancy:     {max_lsb_error} LSB")
        print(f"  - Mean LSB Discrepancy:    {mean_lsb_error:.2f} LSB")
        print("=" * 80 + "\n")


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize PyTorch TinySpeechMaskMLP and Export Verilog $readmemh Memory Files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=Path("model.pth"),
        help="Path to trained floating-point PyTorch checkpoint.",
    )
    parser.add_argument(
        "--val_data",
        type=Path,
        default=Path("processed_data/val_features.npz"),
        help="Path to validation features for calibration.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("fpga_mem"),
        help="Destination directory for Verilog .mem files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exporter = ModelQuantizerExporter(
        model_path=args.model_path,
        val_data_path=args.val_data,
        output_dir=args.output_dir,
    )
    try:
        exporter.run()
    except Exception as err:
        print(f"\n[Quantization Error] {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
