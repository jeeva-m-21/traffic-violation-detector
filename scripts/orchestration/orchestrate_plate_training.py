import subprocess
import sys
import os
import glob
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
COLAB_BIN = "/home/jeeva/.local/bin/colab"
SESSION_NAME = "plate-train"
LOCAL_DATASET = "/home/jeeva/Downloads/Indian number plate.v2i.yolov8.zip"
SCRIPT_FILE = str(ROOT_DIR / "scripts" / "benchmarks" / "train_plate_detector.py")
OUTPUT_DIR = ROOT_DIR / "results" / "plates"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(ROOT_DIR / "models").mkdir(parents=True, exist_ok=True)

def run_colab_cmd(args, timeout=900):
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
    print("=== SPRINT-5: License Plate Detector Training on Colab GPU ===")
    
    # 0. Prepare 14MB dataset chunks (well within Jupyter proxy limits)
    chunks = sorted(glob.glob("/tmp/plate_part_*"))
    if not chunks:
        print("Splitting dataset into 14MB chunks...")
        subprocess.run(["split", "-b", "14M", LOCAL_DATASET, "/tmp/plate_part_"], check=True)
        chunks = sorted(glob.glob("/tmp/plate_part_*"))
    print(f"Prepared {len(chunks)} chunks (14MB each): {chunks}")

    try:
        # 1. Provision T4 GPU session
        print("\n--- Step 1: Allocating Colab T4 GPU Session ---")
        run_colab_cmd(["new", "-s", SESSION_NAME, "--gpu", "T4"], timeout=60)

        # 2. Install dependencies via fast uv
        print("\n--- Step 2: Installing Dependencies on Remote VM ---")
        run_colab_cmd(["install", "-s", SESSION_NAME, "ultralytics", "pyyaml"], timeout=120)

        # 3. Upload dataset chunks
        print("\n--- Step 3: Uploading Dataset Chunks (14MB each) ---")
        for idx, chunk in enumerate(chunks, 1):
            chunk_name = Path(chunk).name
            print(f"Uploading part {idx}/{len(chunks)}: {chunk} ({os.path.getsize(chunk)/1024/1024:.1f} MB)...")
            run_colab_cmd(["upload", "-s", SESSION_NAME, chunk, f"/content/{chunk_name}"], timeout=120)
        print("All chunks uploaded successfully!")

        # 4. Execute training & test benchmark script
        print("\n--- Step 4: Training YOLO11s on 1,215 Plate Images & Benchmarking ---")
        run_colab_cmd(["exec", "-s", SESSION_NAME, "-f", SCRIPT_FILE, "--timeout", "900"], timeout=960)

        # 5. Download trained weights and test metrics
        print("\n--- Step 5: Downloading Best Weights & Evaluation Metrics ---")
        run_colab_cmd(["download", "-s", SESSION_NAME, "/content/plate_detector.pt", "models/plate_detector.pt"], timeout=180)
        run_colab_cmd(["download", "-s", SESSION_NAME, "/content/plate_eval_metrics.json", str(OUTPUT_DIR / "plate_eval_metrics.json")], timeout=60)
        print("\n=== Plate Detector Model & Metrics Saved to models/ and results/plates/ ===")

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
