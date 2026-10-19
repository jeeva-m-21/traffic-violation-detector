import os
import json
import time
import math
from collections import defaultdict, deque
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

print("=== Starting Traffic Violation Detection Pipeline ===")
print("CUDA Available:", torch.cuda.is_available())
device = 0 if torch.cuda.is_available() else "cpu"
print(f"Using compute device: {device}")

VIDEO_INPUT = "/content/wrongway_motion_raw.mp4"
VIDEO_OUTPUT = "/content/wrongway_annotated.mp4"
VIOLATIONS_JSON = "/content/violations.json"
REPORT_MD = "/content/violation_report.md"

assert os.path.exists(VIDEO_INPUT), f"Video file not found at {VIDEO_INPUT}"

cap = cv2.VideoCapture(VIDEO_INPUT)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Input Video: {width}x{height} @ {fps:.2f} FPS | Total frames: {total_frames}")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, fps, (width, height))

model = YOLO("yolo11s.pt")
# Target vehicle classes: 2: car, 3: motorcycle, 5: bus, 7: truck
VEHICLE_CLASSES = [2, 3, 5, 7]
CLASS_NAMES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

# Trajectory and violation state
track_history = defaultdict(lambda: deque(maxlen=30))
track_classes = {}
track_violations = {}
active_violations_log = {}

# Road flow heuristic for dashcam footage:
# In this dashcam clip, normal traffic moving in the left/oncoming lanes or camera forward flow has a standard pattern.
# For oncoming or wrong-way lanes, we track displacement over a 15-frame window.
# In wrongway_motion_raw.mp4, vehicles traveling in the wrong direction move downward/upward opposite to normal flow.
# We determine dominant traffic flow direction in the lane partition.
TRAJECTORY_WINDOW = 12
MIN_DISPLACEMENT_PX = 20  # minimum movement to consider non-stationary

frame_idx = 0
t0 = time.time()
print("\n=== Running Detection, ByteTrack Tracking & Violation Classifier ===")

# Stream predictions with ByteTrack
results_generator = model.track(
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

for result in results_generator:
    frame_idx += 1
    orig_frame = result.orig_img.copy()
    timestamp_sec = frame_idx / fps
    current_frame_has_violation = False

    boxes = result.boxes
    if boxes is not None and boxes.id is not None:
        track_ids = boxes.id.int().tolist()
        xyxys = boxes.xyxy.tolist()
        confs = boxes.conf.tolist()
        clss = boxes.cls.int().tolist()

        for (x1, y1, x2, y2), track_id, conf, cls_id in zip(xyxys, track_ids, confs, clss):
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            track_history[track_id].append((cx, cy))
            track_classes[track_id] = CLASS_NAMES.get(cls_id, "Vehicle")

            # Check motion vector over trajectory window
            is_wrong_way = False
            history = track_history[track_id]
            if len(history) >= TRAJECTORY_WINDOW:
                start_x, start_y = history[0]
                end_x, end_y = history[-1]
                dx = end_x - start_x
                dy = end_y - start_y
                dist = math.hypot(dx, dy)

                # Direction vector classification:
                # In dashcam perspectives, vehicles driving towards camera in their assigned lane have dy > 0 (downward).
                # Vehicles driving away have dy < 0 (upward).
                # On divided highways / one-way roads:
                # If a vehicle on the left lane is moving backward (dy < -MIN_DISPLACEMENT_PX) when flow is forward,
                # or moving forward (dy > MIN_DISPLACEMENT_PX) in an oncoming lane, flag as wrong-way.
                # In wrongway_motion_raw.mp4, the violator vehicle approaches against traffic or reverses in flow.
                # Check if moving opposite to normal flow in the right/left corridor:
                angle_deg = math.degrees(math.atan2(dy, dx))
                
                # Check for wrong-way heading (moving in opposite direction of travel)
                # In the camera frame: normal travel in this corridor has dy > 0 (moving down/closer)
                # Any vehicle traveling upward (dy < -25) while in the foreground (cy > height * 0.4) is moving against traffic:
                if dy < -25 and cy > int(height * 0.35) and dist > MIN_DISPLACEMENT_PX:
                    is_wrong_way = True
                # Or vehicle driving head-on against camera flow in the reverse lane:
                elif dy > 35 and cy < int(height * 0.45) and abs(dx) < 20 and dist > MIN_DISPLACEMENT_PX:
                    is_wrong_way = True

            if is_wrong_way or track_violations.get(track_id, False):
                track_violations[track_id] = True
                current_frame_has_violation = True

                if track_id not in active_violations_log:
                    active_violations_log[track_id] = {
                        "track_id": track_id,
                        "type": "WRONG_WAY_DRIVING",
                        "vehicle_type": track_classes[track_id],
                        "first_detected_frame": frame_idx,
                        "first_detected_time_sec": round(timestamp_sec, 2),
                        "last_frame": frame_idx,
                        "last_time_sec": round(timestamp_sec, 2),
                        "confidence": round(conf, 3),
                        "bbox_sample": [int(x1), int(y1), int(x2), int(y2)]
                    }
                else:
                    active_violations_log[track_id]["last_frame"] = frame_idx
                    active_violations_log[track_id]["last_time_sec"] = round(timestamp_sec, 2)

                # Draw Violation (Bright Red & Warning label)
                color = (0, 0, 230)
                label = f"VIOLATION: ID {track_id} {track_classes[track_id]} [WRONG WAY]"
                cv2.rectangle(orig_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                
                # Filled header for label
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(orig_frame, (int(x1), int(y1) - 24), (int(x1) + label_size[0] + 8, int(y1)), color, -1)
                cv2.putText(orig_frame, label, (int(x1) + 4, int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            else:
                # Normal Vehicle (Green box)
                color = (0, 200, 0)
                label = f"ID {track_id} {track_classes[track_id]} [NORMAL]"
                cv2.rectangle(orig_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(orig_frame, label, (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            # Draw trajectory path trail
            pts = list(history)
            for i in range(1, len(pts)):
                thickness = int(math.sqrt(64 / float(len(pts) - i + 1)) * 1.5)
                trail_color = (0, 0, 255) if track_violations.get(track_id, False) else (0, 255, 255)
                cv2.line(orig_frame, pts[i - 1], pts[i], trail_color, thickness)

    # Heads-Up Display (HUD) Banner at top
    hud_bg = orig_frame.copy()
    cv2.rectangle(hud_bg, (0, 0), (width, 50), (20, 20, 20), -1)
    cv2.addWeighted(hud_bg, 0.75, orig_frame, 0.25, 0, orig_frame)

    cv2.putText(orig_frame, "TRAFFIC VIOLATION SYSTEM: WRONG-WAY DETECTION", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(orig_frame, f"Frame: {frame_idx}/{total_frames} | {timestamp_sec:.1f}s", (640, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    status_color = (0, 0, 230) if current_frame_has_violation else (0, 220, 0)
    status_text = f"ALERT: {len(active_violations_log)} VIOLATION(S)" if len(active_violations_log) > 0 else "STATUS: NORMAL FLOW"
    cv2.putText(orig_frame, status_text, (width - 320, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    writer.write(orig_frame)

    if frame_idx % 100 == 0 or frame_idx == total_frames:
        print(f"Processed {frame_idx}/{total_frames} frames ({frame_idx/total_frames*100:.1f}%) | Active Violations: {len(active_violations_log)}")

cap.release()
writer.release()
elapsed = time.time() - t0
fps_achieved = total_frames / elapsed if elapsed > 0 else 0
print(f"\nProcessing Complete! Time: {elapsed:.2f}s ({fps_achieved:.1f} FPS)")
print(f"Annotated video saved to: {VIDEO_OUTPUT} ({os.path.getsize(VIDEO_OUTPUT) / 1024 / 1024:.2f} MB)")

# Save violations JSON
violations_list = list(active_violations_log.values())
for v in violations_list:
    v["duration_sec"] = round(v["last_time_sec"] - v["first_detected_time_sec"], 2)

with open(VIOLATIONS_JSON, "w") as f:
    json.dump({
        "video": Path(VIDEO_INPUT).name,
        "total_frames": total_frames,
        "duration_sec": round(total_frames / fps, 2),
        "total_violations_detected": len(violations_list),
        "violations": violations_list
    }, f, indent=2)
print(f"Saved violations log to: {VIOLATIONS_JSON}")

# Generate Violation Report
report_text = f"""# Traffic Violation Analysis Report: Wrong-Way Motion

## Executive Summary
- **Source Video:** `{Path(VIDEO_INPUT).name}`
- **Resolution:** `{width}x{height}` @ `{fps:.2f} FPS`
- **Frames Analyzed:** `{total_frames}` ({total_frames / fps:.2f}s)
- **Processing Speed:** `{fps_achieved:.1f} FPS` on `{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}`
- **Total Violations Detected:** `{len(violations_list)}`

## Violation Incidents
| Track ID | Vehicle Type | Start Time | End Time | Duration | Confidence | Status |
|---|---|---|---|---|---|---|
"""

if violations_list:
    for v in violations_list:
        report_text += f"| **{v['track_id']}** | {v['vehicle_type']} | {v['first_detected_time_sec']}s | {v['last_time_sec']}s | {v['duration_sec']}s | {v['confidence']} | **FLAGGED (WRONG WAY)** |\n"
else:
    report_text += "| None | - | - | - | - | - | No wrong-way violations detected |\n"

report_text += f"""
## Outputs
- **Annotated Video:** `{VIDEO_OUTPUT}`
- **Structured Audit Log:** `{VIOLATIONS_JSON}`
"""

with open(REPORT_MD, "w") as f:
    f.write(report_text)
print(f"Saved report to: {REPORT_MD}")
print("=== Traffic Violation Pipeline Finished Successfully! ===")
