import subprocess
import sys
import os
from pathlib import Path

COLAB_BIN = "/home/jeeva/.local/bin/colab"
SESSION_NAME = "helmet-v1"
LOCAL_VIDEO = "/home/jeeva/Downloads/helmet_riders_clean.mp4"
REMOTE_VIDEO = "/content/helmet_riders_clean.mp4"
LOCAL_HELMET_MODEL = "models/helmet_detector.pt"
REMOTE_HELMET_MODEL = "/content/helmet_detector.pt"
SCRIPT_FILE = "run_helmet_pipeline.py"
OUTPUT_DIR = Path("results/violations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def run_colab_cmd(args, timeout=600):
    cmd = [COLAB_BIN] + args
    print(f"[ORCHESTRATOR] Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if res.stdout:
        print(res.stdout)
    if res.stderr:
        print(f"[STDERR] {res.stderr}")
    if res.returncode != 0:
        raise RuntimeError(f"Command failed with code {res.returncode}: {' '.join(cmd)}")
    return res

def main():
    print("=== SPRINT-8: Helmet Violation Pipeline Execution on Colab GPU ===")
    try:
        # 1. Provision T4 GPU session
        print("\n--- Step 1: Allocating Colab T4 GPU Session ---")
        run_colab_cmd(["new", "-s", SESSION_NAME, "--gpu", "T4"], timeout=60)

        # 2. Install dependencies via fast uv
        print("\n--- Step 2: Installing Dependencies (ultralytics, opencv) ---")
        run_colab_cmd(["install", "-s", SESSION_NAME, "ultralytics", "opencv-python-headless"], timeout=180)

        # 3. Upload dashcam video (6.7 MB) and model weights (6.0 MB)
        print("\n--- Step 3: Uploading Video and Helmet Detector Weights ---")
        run_colab_cmd(["upload", "-s", SESSION_NAME, LOCAL_VIDEO, REMOTE_VIDEO], timeout=180)
        run_colab_cmd(["upload", "-s", SESSION_NAME, LOCAL_HELMET_MODEL, REMOTE_HELMET_MODEL], timeout=180)

        # 4. Execute pipeline script
        print("\n--- Step 4: Executing Helmet Detection & Multi-Frame Tracking Pipeline ---")
        run_colab_cmd(["exec", "-s", SESSION_NAME, "-f", SCRIPT_FILE, "--timeout", "600"], timeout=660)

        # 5. Download generated artifacts
        print("\n--- Step 5: Downloading Annotated Video & Helmet Violation Log ---")
        run_colab_cmd(["download", "-s", SESSION_NAME, "/content/helmet_annotated.mp4", str(OUTPUT_DIR / "helmet_annotated.mp4")], timeout=180)
        run_colab_cmd(["download", "-s", SESSION_NAME, "/content/helmet_violations.json", str(OUTPUT_DIR / "helmet_violations.json")], timeout=60)
        run_colab_cmd(["download", "-s", SESSION_NAME, "/content/helmet_report.md", str(OUTPUT_DIR / "helmet_report.md")], timeout=60)
        print("\n=== All Helmet Violation Artifacts Successfully Saved to results/violations/ ===")

    finally:
        # 6. Guaranteed Colab VM Teardown
        print("\n--- Step 6: Guaranteed VM Teardown (Preserving Compute Units) ---")
        try:
            run_colab_cmd(["stop", "-s", SESSION_NAME], timeout=30)
            print("[ORCHESTRATOR] Verified: Colab VM terminated cleanly.")
        except Exception as e:
            print(f"[ORCHESTRATOR] Warning during stop: {e}")

if __name__ == "__main__":
    main()
