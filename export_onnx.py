"""
export_onnx.py - Export Causal DPCRN to ONNX for Real-Time Streaming on Edge Hardware (Jetson Orin).

Key Features:
1. Streaming Step Wrapper with Explicit Recurrent Cache States:
   Inputs:
     - spec_frame: [B, 2, 257, 1]
     - conv1_cache: [B, 2, 257, 1]
     - conv2_cache: [B, 32, 129, 1]
     - conv3_cache: [B, 64, 65, 1]
     - h_dp_0, c_dp_0: [1, B * 33, 128]
     - h_dp_1, c_dp_1: [1, B * 33, 128]
     - dec3_cache: [B, 128, 33, 1]
     - dec2_cache: [B, 64, 65, 1]
     - dec1_cache: [B, 32, 129, 1]
   Outputs:
     - enh_spec_frame: [B, 2, 257, 1]
     - Updated caches for next frame
2. Numerical Verification between PyTorch and ONNX Runtime.
3. Edge Latency Benchmarking (ensuring frame processing latency <= 16 ms).
"""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from modules import CausalDPCRN

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DPCRN_ONNX_Exporter")


class StreamingDPCRNWrapper(nn.Module):
    """
    ONNX-friendly flat input/output wrapper for streaming Causal DPCRN.
    Accepts explicit state tensors and returns next state tensors.
    """
    def __init__(self, model: CausalDPCRN):
        super().__init__()
        self.model = model

    def forward(
        self,
        spec_frame: torch.Tensor,
        conv1_cache: torch.Tensor,
        conv2_cache: torch.Tensor,
        conv3_cache: torch.Tensor,
        h_dp_0: torch.Tensor,
        c_dp_0: torch.Tensor,
        h_dp_1: torch.Tensor,
        c_dp_1: torch.Tensor,
        dec3_cache: torch.Tensor,
        dec2_cache: torch.Tensor,
        dec1_cache: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        states = {
            "conv1_cache": conv1_cache,
            "conv2_cache": conv2_cache,
            "conv3_cache": conv3_cache,
            "h_dp_0": h_dp_0,
            "c_dp_0": c_dp_0,
            "h_dp_1": h_dp_1,
            "c_dp_1": c_dp_1,
            "dec3_cache": dec3_cache,
            "dec2_cache": dec2_cache,
            "dec1_cache": dec1_cache,
        }

        enh_spec, next_states = self.model.forward_streaming_step(spec_frame, states)

        return (
            enh_spec,
            next_states["conv1_cache"],
            next_states["conv2_cache"],
            next_states["conv3_cache"],
            next_states["h_dp_0"],
            next_states["c_dp_0"],
            next_states["h_dp_1"],
            next_states["c_dp_1"],
            next_states["dec3_cache"],
            next_states["dec2_cache"],
            next_states["dec1_cache"],
        )


def export_to_onnx(
    checkpoint_path: Optional[str] = None,
    output_path: str = "dpcrn_streaming.onnx",
    opset_version: int = 17,
) -> str:
    logger.info("Initializing Causal DPCRN model for ONNX export...")
    model = CausalDPCRN(n_fft=512, hop_length=128, win_length=512, num_dual_path_blocks=2)

    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading weights from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
    else:
        logger.warning("No checkpoint provided or found; exporting model with initialized weights.")

    model.eval()
    streaming_wrapper = StreamingDPCRNWrapper(model).eval()

    # Dummy inputs for tracing: batch_size=1, F=257, T=1
    dummy_spec = torch.randn(1, 2, 257, 1)
    dummy_conv1 = torch.zeros(1, 2, 257, 1)
    dummy_conv2 = torch.zeros(1, 32, 129, 1)
    dummy_conv3 = torch.zeros(1, 64, 65, 1)
    dummy_h0 = torch.zeros(1, 33, 128)
    dummy_c0 = torch.zeros(1, 33, 128)
    dummy_h1 = torch.zeros(1, 33, 128)
    dummy_c1 = torch.zeros(1, 33, 128)
    dummy_dec3 = torch.zeros(1, 128, 33, 1)
    dummy_dec2 = torch.zeros(1, 64, 65, 1)
    dummy_dec1 = torch.zeros(1, 32, 129, 1)

    input_names = [
        "spec_frame",
        "conv1_cache_in",
        "conv2_cache_in",
        "conv3_cache_in",
        "h_dp_0_in",
        "c_dp_0_in",
        "h_dp_1_in",
        "c_dp_1_in",
        "dec3_cache_in",
        "dec2_cache_in",
        "dec1_cache_in",
    ]

    output_names = [
        "enh_spec_frame",
        "conv1_cache_out",
        "conv2_cache_out",
        "conv3_cache_out",
        "h_dp_0_out",
        "c_dp_0_out",
        "h_dp_1_out",
        "c_dp_1_out",
        "dec3_cache_out",
        "dec2_cache_out",
        "dec1_cache_out",
    ]

    dynamic_axes = {
        "spec_frame": {0: "batch_size"},
        "conv1_cache_in": {0: "batch_size"},
        "conv2_cache_in": {0: "batch_size"},
        "conv3_cache_in": {0: "batch_size"},
        "h_dp_0_in": {1: "batch_freq"},
        "c_dp_0_in": {1: "batch_freq"},
        "h_dp_1_in": {1: "batch_freq"},
        "c_dp_1_in": {1: "batch_freq"},
        "dec3_cache_in": {0: "batch_size"},
        "dec2_cache_in": {0: "batch_size"},
        "dec1_cache_in": {0: "batch_size"},
        "enh_spec_frame": {0: "batch_size"},
        "conv1_cache_out": {0: "batch_size"},
        "conv2_cache_out": {0: "batch_size"},
        "conv3_cache_out": {0: "batch_size"},
        "h_dp_0_out": {1: "batch_freq"},
        "c_dp_0_out": {1: "batch_freq"},
        "h_dp_1_out": {1: "batch_freq"},
        "c_dp_1_out": {1: "batch_freq"},
        "dec3_cache_out": {0: "batch_size"},
        "dec2_cache_out": {0: "batch_size"},
        "dec1_cache_out": {0: "batch_size"},
    }

    logger.info(f"Exporting ONNX model to {output_path} (Opset {opset_version})...")
    torch.onnx.export(
        streaming_wrapper,
        (
            dummy_spec,
            dummy_conv1,
            dummy_conv2,
            dummy_conv3,
            dummy_h0,
            dummy_c0,
            dummy_h1,
            dummy_c1,
            dummy_dec3,
            dummy_dec2,
            dummy_dec1,
        ),
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False,
    )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"ONNX Export Successful! File Size: {file_size_mb:.2f} MB")
    return output_path


def verify_and_benchmark_onnx(onnx_path: str, num_frames: int = 100):
    """
    Verify numeric equivalence between PyTorch and ONNX Runtime, and benchmark streaming latency.
    """
    logger.info(f"Loading ONNX Runtime session for {onnx_path}...")
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.intra_op_num_threads = 4

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    ort_session = ort.InferenceSession(onnx_path, session_options, providers=providers)
    logger.info(f"Active ONNX Runtime Execution Provider: {ort_session.get_providers()[0]}")

    # Initialize PyTorch Model and states
    model = CausalDPCRN()
    model.eval()
    pt_wrapper = StreamingDPCRNWrapper(model).eval()

    # Initial states
    conv1 = np.zeros((1, 2, 257, 1), dtype=np.float32)
    conv2 = np.zeros((1, 32, 129, 1), dtype=np.float32)
    conv3 = np.zeros((1, 64, 65, 1), dtype=np.float32)
    h0 = np.zeros((1, 33, 128), dtype=np.float32)
    c0 = np.zeros((1, 33, 128), dtype=np.float32)
    h1 = np.zeros((1, 33, 128), dtype=np.float32)
    c1 = np.zeros((1, 33, 128), dtype=np.float32)
    dec3 = np.zeros((1, 128, 33, 1), dtype=np.float32)
    dec2 = np.zeros((1, 64, 65, 1), dtype=np.float32)
    dec1 = np.zeros((1, 32, 129, 1), dtype=np.float32)

    pt_conv1 = torch.from_numpy(conv1)
    pt_conv2 = torch.from_numpy(conv2)
    pt_conv3 = torch.from_numpy(conv3)
    pt_h0 = torch.from_numpy(h0)
    pt_c0 = torch.from_numpy(c0)
    pt_h1 = torch.from_numpy(h1)
    pt_c1 = torch.from_numpy(c1)
    pt_dec3 = torch.from_numpy(dec3)
    pt_dec2 = torch.from_numpy(dec2)
    pt_dec1 = torch.from_numpy(dec1)

    latencies_ms = []
    max_abs_diff = 0.0

    for i in range(num_frames):
        spec_frame_np = np.random.randn(1, 2, 257, 1).astype(np.float32)
        spec_frame_pt = torch.from_numpy(spec_frame_np)

        # PyTorch Streaming Step
        with torch.no_grad():
            (
                pt_out,
                pt_conv1,
                pt_conv2,
                pt_conv3,
                pt_h0,
                pt_c0,
                pt_h1,
                pt_c1,
                pt_dec3,
                pt_dec2,
                pt_dec1,
            ) = pt_wrapper(
                spec_frame_pt,
                pt_conv1,
                pt_conv2,
                pt_conv3,
                pt_h0,
                pt_c0,
                pt_h1,
                pt_c1,
                pt_dec3,
                pt_dec2,
                pt_dec1,
            )

        # ONNX Runtime Streaming Step
        ort_inputs = {
            "spec_frame": spec_frame_np,
            "conv1_cache_in": conv1,
            "conv2_cache_in": conv2,
            "conv3_cache_in": conv3,
            "h_dp_0_in": h0,
            "c_dp_0_in": c0,
            "h_dp_1_in": h1,
            "c_dp_1_in": c1,
            "dec3_cache_in": dec3,
            "dec2_cache_in": dec2,
            "dec1_cache_in": dec1,
        }

        t0 = time.perf_counter()
        ort_outputs = ort_session.run(None, ort_inputs)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

        (
            enh_spec_np,
            conv1,
            conv2,
            conv3,
            h0,
            c0,
            h1,
            c1,
            dec3,
            dec2,
            dec1,
        ) = ort_outputs

        diff = np.max(np.abs(pt_out.numpy() - enh_spec_np))
        if diff > max_abs_diff:
            max_abs_diff = diff

    avg_latency = np.mean(latencies_ms[10:]) if len(latencies_ms) > 10 else np.mean(latencies_ms)
    p95_latency = np.percentile(latencies_ms[10:], 95) if len(latencies_ms) > 10 else np.max(latencies_ms)
    max_latency = np.max(latencies_ms[10:]) if len(latencies_ms) > 10 else np.max(latencies_ms)

    logger.info("================ ONNX Benchmark Results ================")
    logger.info(f"Max Absolute Numeric Error (PyTorch vs ONNX): {max_abs_diff:.6e}")
    logger.info(f"Average Frame Processing Latency: {avg_latency:.2f} ms")
    logger.info(f"95th Percentile Latency: {p95_latency:.2f} ms")
    logger.info(f"Max Frame Latency: {max_latency:.2f} ms")
    logger.info("Algorithmic Latency Limit: 16.00 ms (Frame Hop: 8.00 ms)")

    if avg_latency < 16.0:
        logger.info("[PASSED] Real-time latency budget target <= 16 ms is MET!")
    else:
        logger.warning("[WARNING] Latency exceeds 16 ms budget on current device.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Causal DPCRN to ONNX")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_dpcrn_checkpoint.pth")
    parser.add_argument("--output", type=str, default="dpcrn_streaming.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--benchmark_frames", type=int, default=100)
    args = parser.parse_args()

    onnx_file = export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        opset_version=args.opset,
    )
    verify_and_benchmark_onnx(onnx_file, num_frames=args.benchmark_frames)
