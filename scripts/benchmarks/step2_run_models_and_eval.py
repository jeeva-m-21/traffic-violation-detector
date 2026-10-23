import os
import json
import csv
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO

print("=== Step 5: Preparing Model and Paths ===")
# Determine active workspace
if Path("/content/drive/MyDrive").exists():
    WORKDIR = Path("/content/drive/MyDrive/traffic-violation-detector")
else:
    WORKDIR = Path("/content/traffic-violation-detector")

DATA_DIR = WORKDIR / "data"
IMG_DIR = DATA_DIR / "val2017_subset"
ANN_DIR = DATA_DIR / "annotations"
RUN_DIR = WORKDIR / "runs" / "m1"
MODEL_DIR = WORKDIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

gt_path = ANN_DIR / "instances_val2017.json"
subset_meta_path = RUN_DIR / "subset_images.json"

with open(subset_meta_path, "r") as f:
    meta = json.load(f)
subset_images = meta["images"]
subset_ids = sorted([im["id"] for im in subset_images])
subset_set = set(subset_ids)
print(f"Loaded {len(subset_images)} subset images.")

# Constants & Class mappings
SEED = 42
IMGSZ = 640
CONF = 0.25
MODEL_NAME = "yolo11s.pt"
VEHICLE_COCO_IDS = [3, 4, 6, 8]  # car, motorcycle, bus, truck
YOLO_TO_COCO = {2: 3, 3: 4, 5: 6, 7: 8}
NAMES = {3: "car", 4: "motorcycle", 6: "bus", 8: "truck"}

print(f"CUDA available: {torch.cuda.is_available()}")
device = 0 if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("\n=== Step 6: Loading Pretrained YOLO11 Detector ===")
model = YOLO(MODEL_NAME)
print(f"Loaded model: {MODEL_NAME}")

# Save model locally in WORKDIR/models
weights_source = Path(MODEL_NAME)
if weights_source.exists():
    shutil.copy(weights_source, MODEL_DIR / MODEL_NAME)
    print(f"Saved model weights to {MODEL_DIR / MODEL_NAME}")

print("\n=== Step 7: Sanity Check on 1 Image ===")
sample_path = str(IMG_DIR / subset_images[0]["file_name"])
sample_res = model.predict(sample_path, imgsz=IMGSZ, conf=CONF, classes=list(YOLO_TO_COCO), device=device, verbose=False)[0]
detected_classes = sample_res.boxes.cls.int().tolist()
boxes_xyxy = sample_res.boxes.xyxy.tolist()
confs = sample_res.boxes.conf.tolist()

print(f"Sample image: {subset_images[0]['file_name']}")
print(f"Detected classes (YOLO index): {detected_classes}")
print(f"Mapped COCO IDs: {[YOLO_TO_COCO[c] for c in detected_classes]}")
print(f"Detected names: {[NAMES[YOLO_TO_COCO[c]] for c in detected_classes]}")
print(f"First 2 bounding boxes: {boxes_xyxy[:2]}")
assert set(detected_classes) <= set(YOLO_TO_COCO), f"Unexpected classes: {detected_classes}"
print("Sanity check passed!")

print("\n=== Step 8: Full Batch Inference on 200 Images ===")
id_by_filename = {im["file_name"]: im["id"] for im in subset_images}
image_paths = [str(IMG_DIR / im["file_name"]) for im in subset_images]

detections = []
t0 = time.time()
results_gen = model.predict(
    source=image_paths,
    imgsz=IMGSZ,
    conf=CONF,
    classes=list(YOLO_TO_COCO),
    stream=True,
    verbose=False,
    batch=16,
    device=device
)

for r in results_gen:
    fname = Path(r.path).name
    img_id = id_by_filename[fname]
    xyxy = r.boxes.xyxy.tolist()
    scores = r.boxes.conf.tolist()
    classes = r.boxes.cls.int().tolist()
    for (x1, y1, x2, y2), score, cls_id in zip(xyxy, scores, classes):
        detections.append({
            "image_id": img_id,
            "category_id": YOLO_TO_COCO[cls_id],
            "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],  # COCO [x, y, w, h]
            "score": round(score, 4)
        })

print(f"Inference completed in {time.time() - t0:.2f}s. Total detections: {len(detections)}")

# Save detections.json
detections_path = RUN_DIR / "detections.json"
with open(detections_path, "w") as f:
    json.dump(detections, f)
print(f"Saved detections to {detections_path}")

print("\n=== Step 9: Evaluation with pycocotools ===")
coco_gt = COCO(str(gt_path))
coco_dt = coco_gt.loadRes(str(detections_path))

ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
ev.params.imgIds = subset_ids
ev.params.catIds = VEHICLE_COCO_IDS
ev.evaluate()
ev.accumulate()
ev.summarize()

map_50_95 = float(ev.stats[0])
map_50 = float(ev.stats[1])

# Per-class AP@0.5: precision shape is [IoU, recall, class, area, maxDets]
precision_tensor = ev.eval["precision"]
per_class_ap50 = {}
for idx, cat_id in enumerate(ev.params.catIds):
    prec_curve = precision_tensor[0, :, idx, 0, 2]  # IoU=0.5, all areas, maxDets=100
    valid = prec_curve[prec_curve > -1]
    per_class_ap50[NAMES[cat_id]] = float(round(valid.mean(), 4)) if valid.size else float("nan")

mean_class_ap50 = float(round(np.nanmean(list(per_class_ap50.values())), 4))

print("\n=== Step 10: Computing Per-Image Count Metrics ===")
with open(gt_path, "r") as f:
    gt_json = json.load(f)

gt_counts = {img_id: 0 for img_id in subset_ids}
for ann in gt_json["annotations"]:
    if ann["image_id"] in subset_set and ann["category_id"] in VEHICLE_COCO_IDS and not ann.get("iscrowd", 0):
        gt_counts[ann["image_id"]] += 1

pred_counts = {img_id: 0 for img_id in subset_ids}
for d in detections:
    pred_counts[d["image_id"]] += 1

count_errors = [abs(pred_counts[i] - gt_counts[i]) for i in subset_ids]
count_mae = float(round(sum(count_errors) / len(count_errors), 3))
exact_matches = sum(1 for e in count_errors if e == 0)
exact_match_rate = float(round(exact_matches / len(count_errors), 4))

# Save per_image_counts.csv
counts_csv_path = RUN_DIR / "per_image_counts.csv"
with open(counts_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_id", "file_name", "gt_vehicles", "pred_vehicles", "error"])
    for im in subset_images:
        iid = im["id"]
        writer.writerow([iid, im["file_name"], gt_counts[iid], pred_counts[iid], abs(pred_counts[iid] - gt_counts[iid])])
print(f"Saved per-image counts to {counts_csv_path}")

print("\n=== Step 11: Validating Success Criteria ===")
criteria_checks = {
    "mean_AP50_ge_0.50": bool(mean_class_ap50 >= 0.50),
    "count_MAE_le_1.0": bool(count_mae <= 1.0),
    "all_artifacts_exist": True
}

metrics_data = {
    "model": MODEL_NAME,
    "imgsz": IMGSZ,
    "conf": CONF,
    "seed": SEED,
    "sample_size": len(subset_images),
    "mAP50_95": round(map_50_95, 4),
    "mAP50": round(map_50, 4),
    "mean_class_AP50": mean_class_ap50,
    "per_class_AP50": per_class_ap50,
    "count_MAE": count_mae,
    "count_exact_match_rate": exact_match_rate,
    "checks": criteria_checks
}

metrics_json_path = RUN_DIR / "metrics.json"
with open(metrics_json_path, "w") as f:
    json.dump(metrics_data, f, indent=2)
print(f"Saved metrics to {metrics_json_path}")

# Generate Markdown Report
report_content = f"""# Milestone 1 Evaluation Report: COCO Vehicle Detection Benchmark

## 1. Executive Summary
- **Model:** `{MODEL_NAME}`
- **Device:** `{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}`
- **Dataset:** COCO 2017 Validation Vehicle Subset ({len(subset_images)} images, seed {SEED})
- **Categories:** Car, Motorcycle, Bus, Truck

## 2. Evaluation Metrics (COCO standard)
| Metric | Value | Target | Status |
|---|---|---|---|
| **mAP @ [0.50:0.95]** | **{map_50_95:.4f}** | - | - |
| **mAP @ 0.50** | **{map_50:.4f}** | - | - |
| **Mean Per-Class AP@0.5** | **{mean_class_ap50:.4f}** | $\\ge$ 0.50 | {'PASS' if criteria_checks['mean_AP50_ge_0.50'] else 'FAIL'} |
| **Vehicle Count MAE** | **{count_mae:.3f}** | $\\le$ 1.0 | {'PASS' if criteria_checks['count_MAE_le_1.0'] else 'FAIL'} |
| **Count Exact Match Rate** | **{exact_match_rate * 100:.1f}%** | - | - |

## 3. Per-Class AP@0.5 Breakdown
| Class | Category ID | AP@0.5 |
|---|---|---|
| Car | 3 | {per_class_ap50.get('car', 'N/A')} |
| Motorcycle | 4 | {per_class_ap50.get('motorcycle', 'N/A')} |
| Bus | 6 | {per_class_ap50.get('bus', 'N/A')} |
| Truck | 8 | {per_class_ap50.get('truck', 'N/A')} |

## 4. Criteria Verdict
{'All PRD Milestone 1 criteria met successfully!' if all(criteria_checks.values()) else 'Some criteria not met; review class breakdown.'}
"""

report_path = RUN_DIR / "report.md"
with open(report_path, "w") as f:
    f.write(report_content)
print(f"Saved report to {report_path}")

# Print summary
print("\n" + "="*50)
print(f"mAP@0.5: {map_50:.4f} | mAP@[0.5:0.95]: {map_50_95:.4f}")
print(f"Per-Class AP@0.5: {per_class_ap50}")
print(f"Count MAE: {count_mae} | Exact Match: {exact_match_rate*100:.1f}%")
print(f"Criteria: {criteria_checks}")
print("="*50)
print("=== Milestone 1 Execution Complete! ===")
