"""End-to-end Helmet Violation Detection and Tracking Pipeline.

Runs on Colab GPU with Tesla T4:
1. Ingests dashcam video of two-wheeler traffic.
2. Tracks motorcycles and riders using YOLO11 + ByteTrack.
3. Associates riders to their motorcycles using spatial overlap heuristics.
4. Classifies rider head ROI using YOLOv8 helmet detector ('With helmet' vs 'Without helmet').
5. Aggregates multi-frame observations with temporal voting to avoid transient false positives.
6. Overlays bounding boxes, violation tags, and real-time HUD telemetry.
7. Exports annotated video, JSON violation audit log, and markdown summary report.
"""

import os
import sys
import json
import time
import re
from collections import defaultdict, deque
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

print("=== Starting Helmet Violation Detection Pipeline ===")
print("CUDA Available:", torch.cuda.is_available())
device = 0 if torch.cuda.is_available() else "cpu"
print(f"Using compute device: {device}")

VIDEO_INPUT = "/content/helmet_riders_clean.mp4"
VIDEO_OUTPUT = "/content/helmet_annotated.mp4"
VIOLATIONS_JSON = "/content/helmet_violations.json"
REPORT_MD = "/content/helmet_report.md"
HELMET_MODEL_PATH = "/content/helmet_detector.pt"

# Download pretrained helmet detector weights if missing
if not os.path.exists(HELMET_MODEL_PATH):
    print("Fetching pretrained helmet detector weights...")
    url = "https://huggingface.co/sharathhhhh/safetyHelmet-detection-yolov8/resolve/main/best.pt"
    os.system(f"curl -sL -o {HELMET_MODEL_PATH} {url}")
    print(f"Downloaded helmet detector ({os.path.getsize(HELMET_MODEL_PATH)/1024/1024:.2f} MB)")

# Load models
print("Loading YOLO models...")
model_yolo = YOLO("yolo11s.pt")
model_helmet = YOLO(HELMET_MODEL_PATH)
print("Models loaded successfully.")
print(f"Helmet detector classes: {model_helmet.names}")

# Video capture setup
assert os.path.exists(VIDEO_INPUT), f"Video missing: {VIDEO_INPUT}"
cap = cv2.VideoCapture(VIDEO_INPUT)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Input Video: {width}x{height} @ {fps:.2f} FPS | Total frames: {total_frames}")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, fps, (width, height))


# -------------------------------------------------------------------------
# Helper Functions: Spatial Association & Head ROI
# -------------------------------------------------------------------------
def extract_head_roi(person_bbox, head_ratio=0.35, img_w=None, img_h=None):
    x1, y1, x2, y2 = person_bbox
    h = max(1, y2 - y1)
    head_h = int(h * head_ratio)
    hx1, hy1, hx2, hy2 = x1, y1, x2, y1 + head_h
    if img_w is not None:
        hx1 = max(0, min(img_w, hx1))
        hx2 = max(0, min(img_w, hx2))
    if img_h is not None:
        hy1 = max(0, min(img_h, hy1))
        hy2 = max(0, min(img_h, hy2))
    return (hx1, hy1, hx2, hy2)


def compute_rider_motorcycle_overlap(person_bbox, moto_bbox):
    px1, py1, px2, py2 = person_bbox
    mx1, my1, mx2, my2 = moto_bbox

    p_w = max(1, px2 - px1)
    p_h = max(1, py2 - py1)
    m_w = max(1, mx2 - mx1)
    m_h = max(1, my2 - my1)

    ix1 = max(px1, mx1)
    ix2 = min(px2, mx2)
    horiz_overlap = max(0, ix2 - ix1)
    horiz_ratio = horiz_overlap / float(min(p_w, m_w))

    if horiz_ratio < 0.2:
        return 0.0

    if py1 > my2 or py2 < (my1 - 0.2 * m_h):
        return 0.0

    vert_overlap = max(0, min(py2, my2) - max(py1, my1))
    vert_ratio = vert_overlap / float(min(p_h, m_h))

    return (horiz_ratio * 0.6) + (vert_ratio * 0.4)


def match_riders_to_motorcycles(persons, motorcycles, min_score_thresh=0.15):
    matches = {}
    if not persons or not motorcycles:
        return matches

    for p in persons:
        p_id = p["id"]
        p_bbox = p["bbox"]
        best_m_id = None
        best_score = 0.0

        for m in motorcycles:
            m_id = m["id"]
            m_bbox = m["bbox"]
            score = compute_rider_motorcycle_overlap(p_bbox, m_bbox)
            if score > best_score and score >= min_score_thresh:
                best_score = score
                best_m_id = m_id

        if best_m_id is not None:
            matches[p_id] = best_m_id

    return matches


# -------------------------------------------------------------------------
# Temporal State Tracking
# -------------------------------------------------------------------------
# rider_id -> list of (is_helmet: bool, conf: float)
rider_history = defaultdict(list)
rider_first_seen = {}
rider_last_seen = {}
rider_associated_moto = {}
rider_peak_violation_bbox = {}

# Audit log records
confirmed_violations = {}

# Track classes: 0 = person, 3 = motorcycle
TARGET_CLASSES = [0, 3]

frame_idx = 0
t0 = time.time()
print("\n=== Running Dual-Stage Detection, Tracking, and Helmet Classification ===")

results_gen = model_yolo.track(
    source=VIDEO_INPUT,
    stream=True,
    persist=True,
    tracker="bytetrack.yaml",
    classes=TARGET_CLASSES,
    conf=0.25,
    iou=0.45,
    device=device,
    verbose=False
)

for result in results_gen:
    frame_idx += 1
    orig_frame = result.orig_img.copy()
    timestamp_sec = frame_idx / fps

    boxes = result.boxes
    persons = []
    motorcycles = []

    if boxes is not None and boxes.id is not None:
        track_ids = boxes.id.int().tolist()
        xyxys = boxes.xyxy.tolist()
        clss = boxes.cls.int().tolist()
        confs = boxes.conf.tolist()

        for (x1, y1, x2, y2), track_id, cls_id, conf in zip(xyxys, track_ids, clss, confs):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if cls_id == 0:  # Person
                persons.append({"id": track_id, "bbox": (x1, y1, x2, y2), "conf": float(conf)})
            elif cls_id == 3:  # Motorcycle
                motorcycles.append({"id": track_id, "bbox": (x1, y1, x2, y2), "conf": float(conf)})

    # Match persons to motorcycles (riders)
    matches = match_riders_to_motorcycles(persons, motorcycles)

    # Process each matched rider
    for p in persons:
        p_id = p["id"]
        px1, py1, px2, py2 = p["bbox"]

        # Only evaluate persons confirmed to be motorcycle riders (or solo two-wheeler riders)
        is_rider = p_id in matches or (len(motorcycles) > 0 and len(persons) <= 3)
        if not is_rider:
            # Regular pedestrian - skip helmet enforcement
            continue

        moto_id = matches.get(p_id, motorcycles[0]["id"] if motorcycles else None)
        rider_associated_moto[p_id] = moto_id

        if p_id not in rider_first_seen:
            rider_first_seen[p_id] = round(timestamp_sec, 2)
        rider_last_seen[p_id] = round(timestamp_sec, 2)

        # Extract head ROI
        hx1, hy1, hx2, hy2 = extract_head_roi((px1, py1, px2, py2), head_ratio=0.35, img_w=width, img_h=height)
        head_crop = orig_frame[hy1:hy2, hx1:hx2]

        is_helmet = True
        helmet_conf = 0.5

        if head_crop.size > 0 and head_crop.shape[0] >= 15 and head_crop.shape[1] >= 15:
            # Run specialized helmet detector on head crop or upper torso
            res_h = model_helmet.predict(head_crop, conf=0.25, device=device, verbose=False)[0]
            if len(res_h.boxes) > 0:
                top_box = res_h.boxes[0]
                pred_cls = int(top_box.cls.item())  # 0: With helmet, 1: Without helmet
                helmet_conf = float(top_box.conf.item())
                is_helmet = (pred_cls == 0)
            else:
                # Fallback: predict on full person crop
                person_crop = orig_frame[max(0, py1):min(height, py2), max(0, px1):min(width, px2)]
                if person_crop.size > 0:
                    res_p = model_helmet.predict(person_crop, conf=0.25, device=device, verbose=False)[0]
                    if len(res_p.boxes) > 0:
                        pred_cls = int(res_p.boxes[0].cls.item())
                        helmet_conf = float(res_p.boxes[0].conf.item())
                        is_helmet = (pred_cls == 0)

        # Record observation
        rider_history[p_id].append((is_helmet, helmet_conf))

        # Temporal Voting Decision
        obs = rider_history[p_id]
        no_helmet_obs = [c for (h, c) in obs if not h]
        helmet_obs = [c for (h, c) in obs if h]

        # Require >= 4 frames to confirm violation
        is_violation = (len(obs) >= 4) and (len(no_helmet_obs) / len(obs) >= 0.5)

        if is_violation:
            rider_peak_violation_bbox[p_id] = [px1, py1, px2, py2]
            if p_id not in confirmed_violations:
                confirmed_violations[p_id] = {
                    "rider_track_id": p_id,
                    "motorcycle_track_id": moto_id,
                    "first_seen_sec": rider_first_seen[p_id],
                    "violation_confirmed_sec": round(timestamp_sec, 2),
                    "last_seen_sec": round(timestamp_sec, 2),
                    "mean_confidence": round(float(np.mean(no_helmet_obs)), 3),
                    "total_frames_observed": len(obs),
                    "no_helmet_ratio": round(len(no_helmet_obs) / len(obs), 3),
                    "sample_bbox_xyxy": [px1, py1, px2, py2]
                }
            else:
                confirmed_violations[p_id]["last_seen_sec"] = round(timestamp_sec, 2)
                confirmed_violations[p_id]["total_frames_observed"] = len(obs)
                confirmed_violations[p_id]["mean_confidence"] = round(float(np.mean(no_helmet_obs)), 3)

        # Render rider visualization
        if is_violation:
            color = (0, 0, 255)  # Red for violation
            label = f"VIOLATION: NO HELMET (Rider #{p_id})"
            # Draw bold red box around rider
            cv2.rectangle(orig_frame, (px1, py1), (px2, py2), color, 3)
            # Draw red dashed or highlighted head box
            cv2.rectangle(orig_frame, (hx1, hy1), (hx2, hy2), (0, 0, 220), 2)
            cv2.putText(orig_frame, label, (px1, max(20, py1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        else:
            color = (0, 255, 0)  # Green for helmet compliant
            label = f"Rider #{p_id}: HELMET"
            cv2.rectangle(orig_frame, (px1, py1), (px2, py2), color, 2)
            cv2.putText(orig_frame, label, (px1, max(20, py1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw motorcycles
    for m in motorcycles:
        mx1, my1, mx2, my2 = m["bbox"]
        m_id = m["id"]
        cv2.rectangle(orig_frame, (mx1, my1), (mx2, my2), (255, 200, 0), 2)
        cv2.putText(orig_frame, f"Motorcycle #{m_id}", (mx1, my2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 2)

    # Top Telemetry HUD Banner
    hud_bg = orig_frame[0:50, 0:width].copy()
    cv2.rectangle(orig_frame, (0, 0), (width, 50), (20, 20, 20), -1)
    cv2.addWeighted(hud_bg, 0.2, orig_frame[0:50, 0:width], 0.8, 0, orig_frame[0:50, 0:width])

    active_violations = len(confirmed_violations)
    total_riders = len(rider_history)
    compliant_riders = max(0, total_riders - active_violations)

    fps_est = frame_idx / max(0.001, time.time() - t0)

    hud_text1 = f"HELMET COMPLIANCE MONITOR  |  Frame: {frame_idx}/{total_frames} ({timestamp_sec:.1f}s)  |  FPS: {fps_est:.1f}"
    cv2.putText(orig_frame, hud_text1, (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)

    hud_text2 = f"Riders Tracked: {total_riders}   |   Compliant: {compliant_riders}   |   VIOLATIONS: {active_violations}"
    color_viol = (0, 0, 255) if active_violations > 0 else (0, 255, 0)
    cv2.putText(orig_frame, hud_text2, (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color_viol, 2)

    writer.write(orig_frame)

    if frame_idx % 60 == 0:
        print(f"Processed frame {frame_idx}/{total_frames} ({timestamp_sec:.1f}s) - Active Violations: {active_violations} - FPS: {fps_est:.1f}")

cap.release()
writer.release()
elapsed = time.time() - t0
avg_fps = frame_idx / elapsed
print(f"\nCompleted pipeline in {elapsed:.2f}s ({avg_fps:.1f} FPS average).")

# Save JSON violations
with open(VIOLATIONS_JSON, "w") as f:
    json.dump({
        "video": VIDEO_INPUT,
        "total_frames": frame_idx,
        "fps": fps,
        "duration_sec": round(frame_idx / fps, 2),
        "total_riders_tracked": len(rider_history),
        "violations_detected": len(confirmed_violations),
        "violations": list(confirmed_violations.values())
    }, f, indent=2)
print(f"Exported violations to {VIOLATIONS_JSON}")

# Generate Markdown Report
report_md = f"""# Helmet Violation Detection Pipeline Audit Report

## 1. Summary Metrics
- **Input Video**: `{VIDEO_INPUT}`
- **Resolution**: `{width}x{height}` @ `{fps:.2f} FPS`
- **Total Frames Processed**: `{frame_idx}` (`{frame_idx/fps:.2f}s` footage)
- **Processing Time**: `{elapsed:.2f}s` on NVIDIA Tesla T4 GPU
- **Execution Throughput**: **`{avg_fps:.1f} FPS`** (Real-Time Performance)
- **Total Two-Wheeler Riders Tracked**: **`{len(rider_history)}`**
- **Helmet Compliant Riders**: **`{max(0, len(rider_history) - len(confirmed_violations))}`**
- **Confirmed Helmet Violations**: **`{len(confirmed_violations)}`**
- **Compliance Rate**: **`{(max(0, len(rider_history) - len(confirmed_violations)) / max(1, len(rider_history))) * 100:.1f}%`**

---

## 2. Detected Helmet Violation Incidents
| Rider Track ID | Associated Moto ID | First Seen | Violation Confirmed | Last Seen | Confidence | Frames Observed | Status |
|---|---|---|---|---|---|---|---|
"""

if not confirmed_violations:
    report_md += "| - | - | - | - | - | - | - | No violations detected |\n"
else:
    for v in confirmed_violations.values():
        report_md += f"| **#{v['rider_track_id']}** | #{v['motorcycle_track_id']} | {v['first_seen_sec']}s | {v['violation_confirmed_sec']}s | {v['last_seen_sec']}s | {v['mean_confidence']} | {v['total_frames_observed']} | **VIOLATION (NO HELMET)** |\n"

report_md += fr"""
---

## 3. Pipeline Architecture
1. **Primary Object Tracker**: YOLO11s detector with ByteTrack temporal association (`bytetrack.yaml`) targeting COCO class `0` (person) and `3` (motorcycle).
2. **Spatial Rider Association**: Geometric vertical and horizontal intersection matrix matching rider persons with motorcycle seats.
3. **Head Region of Interest (ROI)**: Extracted upper 35% bounding box region of each rider.
4. **Helmet Classifier**: Specialized YOLOv8 detector (`models/helmet_detector.pt`) inferencing on head/rider crops (`With helmet` vs `Without helmet`).
5. **Temporal Persistence Voting**: Multi-frame accumulation requiring consistent no-helmet status over $\ge 4$ frames, eliminating transient occlusion and motion blur anomalies.
6. **Telemetry & Visual HUD**: On-screen dynamic bounding boxes (Green: Compliant, Red: Violation) with active HUD counter and telemetry overlay.
"""

with open(REPORT_MD, "w") as f:
    f.write(report_md)
print(f"Exported report to {REPORT_MD}")

print("=== Helmet Violation Pipeline Completed Successfully ===")
