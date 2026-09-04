"""
debug_server.py - Remote Debugging Server and Wrapper for Causal DPCRN System.

Enables remote debugging via `debugpy` protocol on port 5678 (compatible with VS Code, Antigravity, PyCharm).

Usage:
1. Start debug server for Web Application:
   python debug_server.py --target app

2. Start debug server for Training:
   python debug_server.py --target train

3. Start debug server for ML Model:
   python debug_server.py --target ml_model

4. Start debug server for Visualizer:
   python debug_server.py --target visualize

5. Standalone listener:
   python debug_server.py --listen-only --port 5678 --wait-for-client
"""

import argparse
import logging
import os
import sys

import debugpy

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("RemoteDebugger")


def enable_remote_debugging(
    host: str = "0.0.0.0",
    port: int = 5678,
    wait_for_client: bool = False,
):
    """
    Start debugpy listener on specified host and port.
    """
    logger.info(f"Starting debugpy remote debugger on {host}:{port}...")
    debugpy.listen((host, port))
    logger.info(f"[READY] Remote debugger is listening on port {port} (Attach from IDE via .vscode/launch.json)")

    if wait_for_client:
        logger.info("[PAUSED] Waiting for debugger client to attach before executing code...")
        debugpy.wait_for_client()
        logger.info("[CONNECTED] Debugger client attached! Continuing execution...")


def main():
    parser = argparse.ArgumentParser(description="Remote Debugger for Causal DPCRN")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Debugger host IP")
    parser.add_argument("--port", type=int, default=5678, help="Debugger listening port (default: 5678)")
    parser.add_argument("--wait", action="store_true", default=False, help="Wait for IDE to attach before running")
    parser.add_argument("--target", type=str, choices=["app", "train", "ml_model", "visualize", "test", "realtime"], default="app", help="Script to run under debugpy")
    parser.add_argument("--listen-only", action="store_true", default=False, help="Only start listener without launching script")
    args = parser.parse_args()

    enable_remote_debugging(host=args.host, port=args.port, wait_for_client=args.wait)

    if args.listen_only:
        logger.info("Debugger running in standalone mode. Press Ctrl+C to stop.")
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Debugger stopped.")
            return

    if args.target == "app":
        import uvicorn
        logger.info("Launching Web Application with remote debugging enabled on http://127.0.0.1:8000...")
        uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)

    elif args.target == "train":
        from train import main as train_main
        logger.info("Launching Model Training with remote debugging enabled...")
        train_main()

    elif args.target == "ml_model":
        import ml_model
        logger.info("Running ml_model.py with remote debugging enabled...")

    elif args.target == "visualize":
        from visualize import main as vis_main
        logger.info("Running visualize.py with remote debugging enabled...")
        vis_main()

    elif args.target == "test":
        import test_system
        logger.info("Running test_system.py with remote debugging enabled...")

    elif args.target == "realtime":
        from realtime_demo import stream_audio_file
        logger.info("Running realtime_demo.py with remote debugging enabled...")
        stream_audio_file()


if __name__ == "__main__":
    main()
