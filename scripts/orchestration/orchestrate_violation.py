import subprocess
import sys
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
COLAB_BIN = "/home/jeeva/.local/bin/colab"
SESSION_NAME = "traffic-v1"
LOCAL_VIDEO = "/home/jeeva/Downloads/wrongway_motion_raw.mp4"
REMOTE_VIDEO = "/content/wrongway_motion_raw.mp4"
SCRIPT_FILE = str(ROOT_DIR / "scripts" / "pipelines" / "run_violation_pipeline.py")
OUTPUT_DIR = ROOT_DIR / "results" / "violations"
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
    print("=== SPRINT-4: Traffic Violation Pipeline Execution ===")
    try:
        # 1. Provision T4 GPU session
        print("\n--- Step 1: Allocating Colab T4 GPU Session ---")
        run_colab_cmd(["new", "-s", SESSION_NAME, "--gpu", "T4"], timeout=60)

        # 2. Install dependencies via fast uv
        print("\n--- Step 2: Installing Dependencies on Remote VM ---")
        run_colab_cmd(["install", "-s", SESSION_NAME, "ultralytics", "opencv-python-headless"], timeout=120)

        # 3. Upload dashcam video
        print("\n--- Step 3: Uploading Dashcam Video (19.2 MB) ---")
        run_colab_cmd(["upload", "-s", SESSION_NAME, LOCAL_VIDEO, REMOTE_VIDEO], timeout=180)

        # 4. Execute pipeline script
        print("\n--- Step 4: Executing Violation Detection & Tracking Pipeline ---")
        run_colab_cmd(["exec", "-s", SESSION_NAME, "-f", SCRIPT_FILE, "--timeout", "600"], timeout=660)

        # 5. Download generated artifacts
        print("\n--- Step 5: Downloading Annotated Video & Violation Log ---")
        run_colab_cmd(["download", "-s", SESSION_NAME, "/content/wrongway_annotated.mp4", str(OUTPUT_DIR / "wrongway_annotated.mp4")], timeout=180)
        run_colab_cmd(["download", "-s", SESSION_NAME, "/content/violations.json", str(OUTPUT_DIR / "violations.json")], timeout=60)
        run_colab_cmd(["download", "-s", SESSION_NAME, "/content/violation_report.md", str(OUTPUT_DIR / "violation_report.md")], timeout=60)
        print("\n=== All Artifacts Downloaded Successfully to results/violations/ ===")

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
