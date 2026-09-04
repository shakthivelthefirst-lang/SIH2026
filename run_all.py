"""
run_all.py - Master Orchestrator to Run All Files in the Causal DPCRN Pipeline.

Runs all major components sequentially:
1. System Test Suite (test_system.py)
2. Standalone ML Model (ml_model.py)
3. Real-Time Streaming Simulator (realtime_demo.py)
4. Spectrogram & Waveform Visualizer (visualize.py)
5. Objective Evaluation Suite (eval.py)
"""

import os
import sys
import subprocess
import time

PYTHON_EXE = sys.executable
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))

PIPELINE_STEPS = [
    ("1. System Test Suite", "test_system.py", []),
    ("2. Standalone ML Model", "ml_model.py", []),
    ("3. Real-Time Streaming Simulator", "realtime_demo.py", ["--onnx", "dpcrn_streaming.onnx", "--output", "outputs/realtime_enhanced.wav"]),
    ("4. Waveform & Spectrogram Visualizer", "visualize.py", ["--output", "outputs/spectrogram_analysis.png"]),
    ("5. Objective Metric Evaluation Suite", "eval.py", ["--checkpoint", "checkpoints/best_dpcrn_checkpoint.pth", "--num_samples", "10"]),
]

def run_step(title: str, script_name: str, args: list) -> bool:
    print("\n" + "=" * 70)
    print(f"  [RUNNING] {title} ({script_name})")
    print("=" * 70)
    
    script_path = os.path.join(WORKSPACE_DIR, script_name)
    cmd = [PYTHON_EXE, script_path] + args
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, cwd=WORKSPACE_DIR, check=True)
        elapsed = time.time() - start_time
        print(f"\n  [SUCCESS] {title} completed successfully in {elapsed:.2f}s")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n  [FAILED] {title} failed with return code {e.returncode} in {elapsed:.2f}s")
        return False
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n  [ERROR] {title} encountered error: {e} in {elapsed:.2f}s")
        return False

def main():
    print("*" * 70)
    print("    CAUSAL DPCRN SPEECH ENHANCEMENT - RUN ALL PIPELINE FILES")
    print(f"    Python Interpreter: {PYTHON_EXE}")
    print(f"    Workspace: {WORKSPACE_DIR}")
    print("*" * 70)
    
    os.makedirs(os.path.join(WORKSPACE_DIR, "outputs"), exist_ok=True)
    
    results = {}
    for title, script, args in PIPELINE_STEPS:
        success = run_step(title, script, args)
        results[title] = success
        if not success:
            print(f"\n[!] Pipeline stopped due to failure in {title}")
            break
            
    print("\n" + "#" * 70)
    print("                      PIPELINE RUN SUMMARY")
    print("#" * 70)
    all_passed = True
    for title, success in results.items():
        status = "PASSED" if success else "FAILED"
        print(f"  - {title.ljust(45)}: [{status}]")
        if not success:
            all_passed = False
            
    if all_passed:
        print("\n[ALL COMPLETE] All files executed successfully!")
        print("Outputs saved in: d:\\siih\\outputs")
        print("Web Application available at: http://127.0.0.1:8000")
    print("#" * 70)

if __name__ == "__main__":
    main()
