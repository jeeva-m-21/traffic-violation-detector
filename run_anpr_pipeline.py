import os
import sys
import json
import time
import math
import re
from collections import defaultdict, deque
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import easyocr

print("=== Starting Automatic Number Plate Recognition (ANPR) Pipeline ===")
print("CUDA Available:", torch.cuda.is_available())
device = 0 if torch.cuda.is_available() else "cpu"
print(f"Using compute device: {device}")

VIDEO_INPUT = "/content/plates_raw.mp4"
VIDEO_OUTPUT = "/content/plates_annotated.mp4"
PLATES_JSON = "/content/detected_plates.json"
REPORT_MD = "/content/anpr_report.md"
PLATE_MODEL_PATH = "/content/plate_detector.pt"

# Download pretrained plate detector if missing
if not os.path.exists(PLATE_MODEL_PATH):
    print("Fetching pretrained license plate detector weights...")
    url = "https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8/raw/main/license_plate_detector.pt"
    os.system(f"curl -sL -o {PLATE_MODEL_PATH} {url}")
    print(f"Downloaded plate detector ({os.path.getsize(PLATE_MODEL_PATH)/1024/1024:.2f} MB)")

# Initialize OCR Engine
print("Initializing EasyOCR reader (GPU-accelerated)...")
ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), verbose=False)
print("EasyOCR initialized.")

# Load models
model_veh = YOLO("yolo11s.pt")
model_plate = YOLO(PLATE_MODEL_PATH)
print("YOLO models loaded successfully.")

# Video setup
assert os.path.exists(VIDEO_INPUT), f"Video missing: {VIDEO_INPUT}"
cap = cv2.VideoCapture(VIDEO_INPUT)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Input Video: {width}x{height} @ {fps:.2f} FPS | Total frames: {total_frames}")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, fps, (width, height))

# Validation dictionaries
CHAR_TO_DIGIT = {'O': '0', 'D': '0', 'Q': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'B': '8'}
DIGIT_TO_CHAR = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B'}
INDIAN_PLATE_REGEX = re.compile(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$')

def format_plate(raw_text: str):
    clean = re.sub(r'[^A-Za-z0-9]', '', raw_text).upper()
    if len(clean) < 6 or len(clean) > 11:
        return clean if len(clean) >= 6 else None
    chars = list(clean)
    for i in range(min(2, len(chars))):
        if chars[i].isdigit() and chars[i] in DIGIT_TO_CHAR:
            chars[i] = DIGIT_TO_CHAR[chars[i]]
    for i in range(max(0, len(chars) - 4), len(chars)):
        if not chars[i].isdigit() and chars[i] in CHAR_TO_DIGIT:
            chars[i] = CHAR_TO_DIGIT[chars[i]]
    return "".join(chars)

# Preprocessing helper
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
def preprocess_plate(crop_bgr):
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    enhanced = clahe.apply(gray)
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    return denoised

# Tracking & Voting state
track_votes = defaultdict(lambda: defaultdict(float))
track_first_seen = {}
track_last_seen = {}
track_classes = {}
track_resolved_plates = {}

VEHICLE_CLASSES = [2, 3, 5, 7]
CLASS_NAMES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

frame_idx = 0
t0 = time.time()
print("\n=== Running Cascaded Vehicle Tracking, Plate Detection & OCR ===")

results_gen = model_veh.track(
    source=VIDEO_INPUT,
    stream=True,
    persist=True,
    tracker="bytetrack.yaml",
    classes=VEHICLE_CLASSES,
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
    if boxes is not None and boxes.id is not None:
        track_ids = boxes.id.int().tolist()
        xyxys = boxes.xyxy.tolist()
        clss = boxes.cls.int().tolist()

        for (vx1, vy1, vx2, vy2), track_id, cls_id in zip(xyxys, track_ids, clss):
            vx1, vy1, vx2, vy2 = int(vx1), int(vy1), int(vx2), int(vy2)
            cls_name = CLASS_NAMES.get(cls_id, "Vehicle")
            track_classes[track_id] = cls_name

            if track_id not in track_first_seen:
                track_first_seen[track_id] = round(timestamp_sec, 2)
            track_last_seen[track_id] = round(timestamp_sec, 2)

            # Crop vehicle ROI
            veh_crop = orig_frame[max(0, vy1):min(height, vy2), max(0, vx1):min(width, vx2)]
            plate_coords = None

            if veh_crop.size > 0 and veh_crop.shape[0] > 40 and veh_crop.shape[1] > 40:
                # Detect plate inside vehicle crop
                plate_res = model_plate.predict(veh_crop, conf=0.35, device=device, verbose=False)[0]
                if len(plate_res.boxes) > 0:
                    px1, py1, px2, py2 = plate_res.boxes.xyxy[0].int().tolist()
                    abs_px1 = vx1 + px1
                    abs_py1 = vy1 + py1
                    abs_px2 = vx1 + px2
                    abs_py2 = vy1 + py2
                    plate_coords = (abs_px1, abs_py1, abs_px2, abs_py2)

                    # Extract plate crop
                    plate_crop = orig_frame[max(0, abs_py1):min(height, abs_py2), max(0, abs_px1):min(width, abs_px2)]
                    if plate_crop.size > 0 and plate_crop.shape[0] >= 12 and plate_crop.shape[1] >= 30:
                        # Preprocess and run OCR every 3rd frame to optimize speed
                        if frame_idx % 3 == 0:
                            prep = preprocess_plate(plate_crop)
                            ocr_results = ocr_reader.readtext(
                                prep,
                                allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                                detail=1,
                                paragraph=False
                            )
                            if ocr_results:
                                text = "".join([res[1] for res in ocr_results])
                                conf = float(np.mean([res[2] for res in ocr_results]))
                                formatted = format_plate(text)
                                if formatted:
                                    track_votes[track_id][formatted] += conf

            # Resolve plate string if voted
            if track_id in track_votes and track_votes[track_id]:
                best_plate = max(track_votes[track_id].keys(), key=lambda p: track_votes[track_id][p])
                track_resolved_plates[track_id] = best_plate

            # Visual Annotations
            plate_str = track_resolved_plates.get(track_id, None)
            veh_color = (255, 180, 0) if plate_str else (0, 220, 0)
            cv2.rectangle(orig_frame, (vx1, vy1), (vx2, vy2), veh_color, 2)

            label = f"ID {track_id} {cls_name}"
            if plate_str:
                label += f" | [{plate_str}]"
            cv2.putText(orig_frame, label, (vx1, vy1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, veh_color, 2)

            # Draw license plate box if visible
            if plate_coords:
                cv2.rectangle(orig_frame, (plate_coords[0], plate_coords[1]), (plate_coords[2], plate_coords[3]), (0, 255, 255), 2)

    # Top HUD Banner
    hud = orig_frame.copy()
    cv2.rectangle(hud, (0, 0), (width, 50), (25, 25, 25), -1)
    cv2.addWeighted(hud, 0.8, orig_frame, 0.2, 0, orig_frame)

    cv2.putText(orig_frame, "ANPR SYSTEM: LICENSE PLATE DETECTION & OCR", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(orig_frame, f"Frame: {frame_idx}/{total_frames}", (640, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(orig_frame, f"Plates Read: {len(track_resolved_plates)}", (width - 240, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    writer.write(orig_frame)

    if frame_idx % 100 == 0 or frame_idx == total_frames:
        print(f"Processed {frame_idx}/{total_frames} frames ({frame_idx/total_frames*100:.1f}%) | Resolved Plates: {len(track_resolved_plates)}")

cap.release()
writer.release()
elapsed = time.time() - t0
fps_achieved = total_frames / elapsed if elapsed > 0 else 0
print(f"\nANPR Processing Finished in {elapsed:.2f}s ({fps_achieved:.1f} FPS)")
print(f"Annotated video saved: {VIDEO_OUTPUT} ({os.path.getsize(VIDEO_OUTPUT)/1024/1024:.2f} MB)")

# Structured JSON Export
plate_records = []
for tid, plate_text in track_resolved_plates.items():
    total_votes = sum(track_votes[tid].values())
    plate_records.append({
        "track_id": tid,
        "vehicle_type": track_classes.get(tid, "Vehicle"),
        "license_plate": plate_text,
        "first_seen_sec": track_first_seen.get(tid, 0.0),
        "last_seen_sec": track_last_seen.get(tid, 0.0),
        "confidence_score": round(track_votes[tid][plate_text], 3)
    })

with open(PLATES_JSON, "w") as f:
    json.dump({
        "video": Path(VIDEO_INPUT).name,
        "total_frames": total_frames,
        "duration_sec": round(total_frames / fps, 2),
        "total_vehicles_tracked": len(track_classes),
        "total_plates_recognized": len(plate_records),
        "vehicles": plate_records
    }, f, indent=2)
print(f"Saved plates log: {PLATES_JSON}")

# Summary Report
report_text = f"""# ANPR Pipeline Audit Report: License Plate Recognition

## Executive Summary
- **Source Footage:** `{Path(VIDEO_INPUT).name}`
- **Resolution:** `{width}x{height}` @ `{fps:.2f} FPS`
- **Total Frames:** `{total_frames}` ({total_frames / fps:.2f}s)
- **Execution Speed:** `{fps_achieved:.1f} FPS` on `{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}`
- **Total Tracked Vehicles:** `{len(track_classes)}`
- **Plates Successfully Read:** `{len(plate_records)}`

## Recognized Vehicles & License Plates
| Track ID | Vehicle Type | License Plate | First Seen | Last Seen | Confidence Score |
|---|---|---|---|---|---|
"""
for p in plate_records:
    report_text += f"| **{p['track_id']}** | {p['vehicle_type']} | **`{p['license_plate']}`** | {p['first_seen_sec']}s | {p['last_seen_sec']}s | {p['confidence_score']} |\n"

report_text += f"""
## Artifacts Generated
- **Annotated Video:** `{VIDEO_OUTPUT}`
- **Structured JSON Audit:** `{PLATES_JSON}`
"""

with open(REPORT_MD, "w") as f:
    f.write(report_text)
print(f"Saved report: {REPORT_MD}")
print("=== ANPR Pipeline Complete! ===")
