"""
cli_menu.py - Interactive Command-Line Control Center for Causal DPCRN System.

Provides an easy interactive terminal menu to run any component with one keypress.
"""

import os
import sys
import subprocess

PYTHON_EXE = sys.executable
WORKSPACE = os.path.dirname(os.path.abspath(__file__))


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_banner():
    print("=" * 70)
    print("      CAUSAL DPCRN SPEECH ENHANCEMENT & ANC - CONTROL CENTER")
    print("=" * 70)
    print(" 1.  Run Full Pipeline Verification (run_all.py)")
    print(" 2.  Run Standalone ML Model & Training (ml_model.py)")
    print(" 3.  Run 6-Module Test Suite (test_system.py)")
    print(" 4.  Start Web Application Server (http://127.0.0.1:8000)")
    print(" 5.  Run Real-Time Audio Streaming Simulation (realtime_demo.py)")
    print(" 6.  Generate 5-Panel Spectrogram & Waveform Visuals (visualize.py)")
    print(" 7.  Run Objective Metrics Evaluation - SI-SNR/STOI (eval.py)")
    print(" 8.  Denoise a Custom Audio File (denoise_file.py)")
    print(" 9.  Export Streaming ONNX Model (export_onnx.py)")
    print(" 10. Start Remote Debugging Server on Port 5678 (debug_server.py)")
    print(" 0.  Exit")
    print("=" * 70)


def run_command(cmd_list):
    print(f"\n[EXECUTING] {' '.join(cmd_list)}\n")
    try:
        subprocess.run(cmd_list, cwd=WORKSPACE)
    except KeyboardInterrupt:
        print("\n[STOPPED] Execution interrupted by user.")
    input("\nPress Enter to return to main menu...")


def main():
    while True:
        clear_screen()
        print_banner()
        choice = input("Enter option [0-10]: ").strip()

        if choice == "0":
            print("\nExiting Control Center. Goodbye!")
            sys.exit(0)
        elif choice == "1":
            run_command([PYTHON_EXE, "run_all.py"])
        elif choice == "2":
            run_command([PYTHON_EXE, "ml_model.py"])
        elif choice == "3":
            run_command([PYTHON_EXE, "test_system.py"])
        elif choice == "4":
            run_command([PYTHON_EXE, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8000"])
        elif choice == "5":
            run_command([PYTHON_EXE, "realtime_demo.py"])
        elif choice == "6":
            run_command([PYTHON_EXE, "visualize.py"])
        elif choice == "7":
            run_command([PYTHON_EXE, "eval.py", "--checkpoint", "checkpoints/best_dpcrn_checkpoint.pth"])
        elif choice == "8":
            inp = input("Enter path to input .wav file (or press Enter for realtime_noisy_input.wav): ").strip()
            if not inp:
                inp = "realtime_noisy_input.wav"
            outp = input("Enter path to output .wav file (default: outputs/denoised_output.wav): ").strip()
            if not outp:
                outp = "outputs/denoised_output.wav"
            run_command([PYTHON_EXE, "denoise_file.py", "--input", inp, "--output", outp, "--mode", "streaming"])
        elif choice == "9":
            run_command([PYTHON_EXE, "export_onnx.py", "--checkpoint", "checkpoints/best_dpcrn_checkpoint.pth", "--output", "dpcrn_streaming.onnx"])
        elif choice == "10":
            run_command([PYTHON_EXE, "debug_server.py", "--target", "app"])
        else:
            input("\nInvalid option! Press Enter to try again...")


if __name__ == "__main__":
    main()
