"""Orchestrator for Integrated Helmet Violation & ANPR Pipeline on Colab GPU.

Handles remote T4 GPU provisioning, model & video deployment, execution,
download of video, audit logs, and evidence snapshots, and guarantees VM teardown.
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
COLAB_BIN = "/home/jeeva/.local/bin/colab"
SESSION_NAME = "traffic-integrated"
LOCAL_VIDEO = "/home/jeeva/Downloads/helmet_riders_clean.mp4"
REMOTE_VIDEO = "/content/helmet_riders_clean.mp4"
SCRIPT_FILE = str(ROOT_DIR / "scripts" / "pipelines" / "run_integrated_pipeline.py")

OUTPUT_DIR = ROOT_DIR / "results" / "integrated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOTS_DIR = OUTPUT_DIR / "snapshots"
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

# Also ensure static snapshots for web frontend
STATIC_SNAPSHOTS_DIR = ROOT_DIR / "static" / "snapshots"
STATIC_SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

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
    print("=== SPRINT-9: Integrated Helmet + License Plate OCR on Colab GPU ===")
    try:
        # 1. Allocate T4 GPU session
        print("\n--- Step 1: Allocating Colab T4 GPU Session ---")
        run_colab_cmd(["new", "-s", SESSION_NAME, "--gpu", "T4"], timeout=60)

        # 2. Install dependencies (ultralytics, easyocr, opencv)
        print("\n--- Step 2: Installing Dependencies on Remote VM ---")
        run_colab_cmd(["install", "-s", SESSION_NAME, "ultralytics", "easyocr", "opencv-python-headless"], timeout=200)

        # 3. Upload video and model weights
        print("\n--- Step 3: Uploading Video and Model Weights ---")
        run_colab_cmd(["upload", "-s", SESSION_NAME, LOCAL_VIDEO, REMOTE_VIDEO], timeout=180)
        run_colab_cmd(["upload", "-s", SESSION_NAME, "models/helmet_detector.pt", "/content/helmet_detector.pt"], timeout=180)
        run_colab_cmd(["upload", "-s", SESSION_NAME, "models/plate_detector.pt", "/content/plate_detector.pt"], timeout=180)

        # 4. Execute integrated pipeline
        print("\n--- Step 4: Executing Integrated Tracking, Helmet Detection & ANPR ---")
        run_colab_cmd(["exec", "-s", SESSION_NAME, "-f", SCRIPT_FILE, "--timeout", "600"], timeout=660)

        # 5. Download primary artifacts
        print("\n--- Step 5: Downloading Annotated Video & Extracted Results ---")
        run_colab_cmd(["download", "-s", SESSION_NAME, "/content/integrated_annotated.mp4", str(OUTPUT_DIR / "integrated_annotated.mp4")], timeout=180)
        run_colab_cmd(["download", "-s", SESSION_NAME, "/content/integrated_results.json", str(OUTPUT_DIR / "integrated_results.json")], timeout=60)
        run_colab_cmd(["download", "-s", SESSION_NAME, "/content/integrated_report.md", str(OUTPUT_DIR / "integrated_report.md")], timeout=60)

        # Download snapshot crops
        print("\n--- Step 6: Downloading Violation Snapshot Evidence Images ---")
        for inc_id in ["INC-001", "INC-002", "INC-003"]:
            remote_snap = f"/content/snapshots/{inc_id}.jpg"
            local_snap = SNAPSHOTS_DIR / f"{inc_id}.jpg"
            try:
                run_colab_cmd(["download", "-s", SESSION_NAME, remote_snap, str(local_snap)], timeout=60)
                # Copy to static/snapshots/ for web app
                shutil.copy(str(local_snap), str(STATIC_SNAPSHOTS_DIR / f"{inc_id}.jpg"))
            except Exception as e:
                print(f"[ORCHESTRATOR] Note on {inc_id} snapshot: {e}")

        # Also copy video to static for web app playback
        shutil.copy(str(OUTPUT_DIR / "integrated_annotated.mp4"), str(ROOT_DIR / "static" / "integrated_annotated.mp4"))

        print("\n=== All Integrated Artifacts Saved to results/integrated/ & static/ ===")

    finally:
        # 7. Guaranteed VM Teardown
        print("\n--- Step 7: Guaranteed VM Teardown (Preserving Compute Units) ---")
        try:
            run_colab_cmd(["stop", "-s", SESSION_NAME], timeout=30)
            print("[ORCHESTRATOR] Verified: Colab VM terminated cleanly.")
        except Exception as e:
            print(f"[ORCHESTRATOR] Warning during stop: {e}")

if __name__ == "__main__":
    main()
