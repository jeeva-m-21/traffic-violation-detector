import os
import sys
import shutil
import zipfile
import json
import yaml
from pathlib import Path
import torch
from ultralytics import YOLO

print("=== Starting Indian License Plate Detector Training ===")
print("CUDA Available:", torch.cuda.is_available())
device = 0 if torch.cuda.is_available() else "cpu"
print(f"Using compute device: {device}")

ZIP_PATH = "/content/Indian_number_plate.zip"
DATASET_DIR = Path("/content/dataset")
PROJECT_DIR = Path("/content/plate_training")
OUTPUT_MODEL = Path("/content/plate_detector.pt")
METRICS_JSON = Path("/content/plate_eval_metrics.json")

# 1. Reassemble chunks if present and unzip dataset
if not os.path.exists(ZIP_PATH) and os.path.exists("/content/plate_part_aa"):
    print("Reassembling dataset zip from chunks...")
    os.system("cat /content/plate_part_* > /content/Indian_number_plate.zip")
    os.system("rm -f /content/plate_part_*")

assert os.path.exists(ZIP_PATH), f"Dataset zip missing at {ZIP_PATH}"
if not (DATASET_DIR / "data.yaml").exists():
    print(f"Extracting {ZIP_PATH} to {DATASET_DIR}...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        z.extractall(DATASET_DIR)
    print("Dataset extracted successfully.")

# 2. Configure data.yaml with absolute paths
yaml_path = DATASET_DIR / "data.yaml"
with open(yaml_path, "r") as f:
    cfg = yaml.safe_load(f)

cfg["train"] = str(DATASET_DIR / "train" / "images")
cfg["val"] = str(DATASET_DIR / "valid" / "images")
cfg["test"] = str(DATASET_DIR / "test" / "images")

with open(yaml_path, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)
print("Updated data.yaml with absolute paths:")
print(open(yaml_path).read())

# 3. Fine-tune YOLO11 on Number Plate Dataset
print("\n=== Launching Fine-Tuning on T4 GPU (15 Epochs) ===")
model = YOLO("yolo11s.pt")

train_results = model.train(
    data=str(yaml_path),
    epochs=15,
    imgsz=640,
    batch=16,
    device=device,
    project=str(PROJECT_DIR),
    name="yolo11s_plate",
    save=True,
    verbose=True,
    workers=4,
    patience=5
)

# 4. Evaluate on Test Split
print("\n=== Evaluating Best Checkpoint on Test Split (59 Images) ===")
best_weights = PROJECT_DIR / "yolo11s_plate" / "weights" / "best.pt"
assert best_weights.exists(), f"Best weights not found at {best_weights}"

shutil.copy(best_weights, OUTPUT_MODEL)
print(f"Copied best model to {OUTPUT_MODEL} ({OUTPUT_MODEL.stat().st_size / 1024 / 1024:.2f} MB)")

best_model = YOLO(str(OUTPUT_MODEL))
val_results = best_model.val(
    data=str(yaml_path),
    split="test",
    imgsz=640,
    device=device
)

test_map50 = float(val_results.box.map50)
test_map50_95 = float(val_results.box.map)
precision = float(val_results.box.mp)
recall = float(val_results.box.mr)

metrics = {
    "model": "yolo11s_plate",
    "epochs": 15,
    "test_split_size": 59,
    "mAP50": round(test_map50, 4),
    "mAP50_95": round(test_map50_95, 4),
    "precision": round(precision, 4),
    "recall": round(recall, 4)
}

with open(METRICS_JSON, "w") as f:
    json.dump(metrics, f, indent=2)

print("\n" + "="*50)
print(f"Test Set Evaluation Results:")
print(f"mAP@0.5: {test_map50:.4f} ({test_map50*100:.1f}%)")
print(f"mAP@[0.5:0.95]: {test_map50_95:.4f} ({test_map50_95*100:.1f}%)")
print(f"Precision: {precision:.4f} | Recall: {recall:.4f}")
print("="*50)
print("=== Plate Detector Training and Evaluation Complete! ===")
