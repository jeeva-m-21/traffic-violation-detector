"""Local Hardware Inference & Verification Runner.

Executes the full computer vision traffic violation pipeline natively on the host CPU:
1. Multi-threaded CPU PyTorch setup (Intel Core i5).
2. Vehicle and Pedestrian Tracking (YOLO11s + ByteTrack).
3. Rider Motorcycle Association & Head ROI extraction.
4. Helmet Compliance Check (YOLOv8 helmet classifier).
5. Targeted License Plate Detection (YOLO plate detector).
6. Local OCR character recognition (EasyOCR CPU mode).
7. Frame-by-frame latency logging and visual video export.
"""

import os
import sys
import time
import json
import re
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import easyocr

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.violations.helmet_detector import match_riders_to_motorcycles, extract_head_roi
from src.violations.helmet_voter import HelmetViolationVoter
from src.anpr.plate_enhancer import (
    PlateCandidateTracker,
    compute_crop_clarity,
    enhance_plate_crop,
    assemble_multiline_plate,
)

# Optimize CPU threads for 13th Gen Intel Core i5
threads = min(8, os.cpu_count() or 4)
torch.set_num_threads(threads)
print(f"=== Running Local Hardware Inference ===")
print(f"Host CPU: 13th Gen Intel Core i5 | PyTorch CPU Threads: {threads}")
print(f"CUDA Available: {torch.cuda.is_available()} (Running strictly on local CPU hardware)")

INPUT_VIDEO = sys.argv[1] if len(sys.argv) > 1 else "/home/jeeva/Downloads/test_clip_5s.mp4"
OUTPUT_DIR = ROOT_DIR / "results" / "local_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_VIDEO = OUTPUT_DIR / "local_annotated.mp4"
OUTPUT_JSON = OUTPUT_DIR / "local_results.json"

MODEL_VEH_PATH = ROOT_DIR / "models" / "yolo11s.pt"
MODEL_HELMET_PATH = ROOT_DIR / "models" / "helmet_detector.pt"
MODEL_PLATE_PATH = ROOT_DIR / "models" / "plate_detector.pt"

assert os.path.exists(INPUT_VIDEO), f"Input video missing: {INPUT_VIDEO}"
assert MODEL_VEH_PATH.exists(), f"{MODEL_VEH_PATH} missing"
assert MODEL_HELMET_PATH.exists(), f"{MODEL_HELMET_PATH} missing"
assert MODEL_PLATE_PATH.exists(), f"{MODEL_PLATE_PATH} missing"

print("\n--- Step 1: Loading Neural Models & OCR Engine into CPU Memory ---")
t_load_start = time.time()
model_veh = YOLO(str(MODEL_VEH_PATH))
model_helmet = YOLO(str(MODEL_HELMET_PATH))
model_plate = YOLO(str(MODEL_PLATE_PATH))
model_helmet = YOLO("models/helmet_detector.pt")
model_plate = YOLO("models/plate_detector.pt")
ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
t_load_end = time.time()
print(f"Models loaded in {t_load_end - t_load_start:.2f}s.")

# Video Reader & Writer
cap = cv2.VideoCapture(INPUT_VIDEO)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Input Video: {INPUT_VIDEO} ({width}x{height} @ {fps:.1f} FPS, {total_frames} frames)")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, fps, (width, height))

# State
helmet_voter = HelmetViolationVoter(min_observations=3, violation_threshold=0.5)
plate_tracker = PlateCandidateTracker(min_clarity=20.0)
confirmed_violations = {}
moto_plates = {}
incident_counter = 1

frame_idx = 0
t_start = time.time()
TARGET_CLASSES = [0, 3]  # Person, Motorcycle

print("\n--- Step 2: Processing Video Frames on Local CPU ---")

# Run tracking generator
results_gen = model_veh.track(
    source=INPUT_VIDEO,
    stream=True,
    persist=True,
    tracker="bytetrack.yaml",
    classes=TARGET_CLASSES,
    conf=0.25,
    device="cpu",
    verbose=False
)

for result in results_gen:
    f_t0 = time.time()
    frame_idx += 1
    frame = result.orig_img.copy()
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
            if cls_id == 0:
                persons.append({"id": track_id, "bbox": (x1, y1, x2, y2), "conf": float(conf)})
            elif cls_id == 3:
                motorcycles.append({"id": track_id, "bbox": (x1, y1, x2, y2), "conf": float(conf)})

    # Match persons to motorcycles
    matches = match_riders_to_motorcycles(persons, motorcycles)

    # Evaluate Riders for Helmet
    for p in persons:
        p_id = p["id"]
        px1, py1, px2, py2 = p["bbox"]

        is_rider = (p_id in matches) or (len(motorcycles) > 0 and len(persons) <= 3)
        if not is_rider:
            continue

        moto_id = matches.get(p_id, motorcycles[0]["id"] if motorcycles else None)

        # Head ROI
        hx1, hy1, hx2, hy2 = extract_head_roi((px1, py1, px2, py2), head_ratio=0.35, img_width=width, img_height=height)
        head_crop = frame[hy1:hy2, hx1:hx2]

        is_helmet = True
        helmet_conf = 0.5

        if head_crop.size > 0 and head_crop.shape[0] >= 12 and head_crop.shape[1] >= 12:
            res_h = model_helmet.predict(head_crop, conf=0.25, device="cpu", verbose=False)[0]
            if len(res_h.boxes) > 0:
                pred_cls = int(res_h.boxes[0].cls.item())
                helmet_conf = float(res_h.boxes[0].conf.item())
                is_helmet = (pred_cls == 0)

        helmet_voter.add_observation(p_id, is_helmet, helmet_conf)
        is_viol = helmet_voter.is_violation(p_id)

        if is_viol:
            if p_id not in confirmed_violations:
                inc_id = f"INC-{incident_counter:03d}"
                incident_counter += 1
                confirmed_violations[p_id] = {
                    "incident_id": inc_id,
                    "rider_id": p_id,
                    "motorcycle_id": moto_id,
                    "first_seen_sec": round(timestamp_sec, 2),
                    "violation_type": "NO_HELMET",
                    "plate_number": "SCANNING...",
                    "plate_confidence": 0.0,
                    "plate_clarity": 0.0,
                    "snapshot_path": None,
                    "plate_crop_path": None,
                    "sample_bbox": [px1, py1, px2, py2]
                }
            # Draw red violation box
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 3)
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 0, 200), 2)
            cv2.putText(frame, f"VIOLATION: NO HELMET (#{p_id})", (px1, max(20, py1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            # Draw green compliant box
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
            cv2.putText(frame, f"Rider #{p_id}: HELMET", (px1, max(20, py1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    # Targeted Plate Detection on Offending Motorcycles
    violating_moto_ids = {v["motorcycle_id"] for v in confirmed_violations.values() if v.get("motorcycle_id") is not None}

    for m in motorcycles:
        m_id = m["id"]
        mx1, my1, mx2, my2 = m["bbox"]
        cv2.rectangle(frame, (mx1, my1), (mx2, my2), (255, 200, 0), 2)
        label_m = f"Motorcycle #{m_id}"
        if m_id in moto_plates:
            label_m += f" | Plate: {moto_plates[m_id]}"
        cv2.putText(frame, label_m, (mx1, my2 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 2)

        if m_id in violating_moto_ids:
            moto_crop = frame[max(0, my1):min(height, my2), max(0, mx1):min(width, mx2)]
            if moto_crop.size > 0 and moto_crop.shape[0] >= 30 and moto_crop.shape[1] >= 30:
                plate_res = model_plate.predict(moto_crop, conf=0.30, device="cpu", verbose=False)[0]
                if len(plate_res.boxes) > 0:
                    px1, py1, px2, py2 = plate_res.boxes.xyxy[0].int().tolist()
                    abs_px1 = mx1 + px1
                    abs_py1 = my1 + py1
                    abs_px2 = mx1 + px2
                    abs_py2 = my1 + py2

                    # Highlight plate
                    cv2.rectangle(frame, (abs_px1, abs_py1), (abs_px2, abs_py2), (0, 255, 255), 2)

                    # Extract plate crop & register candidate in clarity tracker
                    plate_crop = frame[max(0, abs_py1):min(height, abs_py2), max(0, abs_px1):min(width, abs_px2)]
                    if plate_crop.size > 0 and plate_crop.shape[0] >= 12 and plate_crop.shape[1] >= 20:
                        plate_tracker.add_candidate(
                            track_id=m_id,
                            crop=plate_crop,
                            bbox=(abs_px1, abs_py1, abs_px2, abs_py2),
                            frame_idx=frame_idx,
                            timestamp_sec=timestamp_sec
                        )

                        # Run high-clarity OCR periodically on the best available candidate
                        if frame_idx % 3 == 0:
                            best_cand = plate_tracker.get_best_candidate(m_id)
                            if best_cand is not None and best_cand.crop.size > 0:
                                prep = enhance_plate_crop(best_cand.crop, upscale_factor=2.5)
                                ocr_res = ocr_reader.readtext(
                                    prep,
                                    allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-. ',
                                    detail=1,
                                    paragraph=False
                                )
                                full_plate, ocr_conf = assemble_multiline_plate(ocr_res)
                                if full_plate and len(full_plate) >= 4:
                                    moto_plates[m_id] = full_plate
                                    for v in confirmed_violations.values():
                                        if v["motorcycle_id"] == m_id:
                                            v["plate_number"] = full_plate
                                            v["plate_confidence"] = round(float(ocr_conf), 3)
                                            v["plate_clarity"] = round(float(best_cand.clarity_score), 1)

                                            # Save standalone high-clarity plate crop JPG
                                            plate_snap_name = f"{v['incident_id']}_plate.jpg"
                                            plate_snap_path = OUTPUT_DIR / plate_snap_name
                                            cv2.imwrite(str(plate_snap_path), best_cand.crop)
                                            v["plate_crop_path"] = str(plate_snap_path)

                                            # Save composite scene evidence if not saved
                                            if v["snapshot_path"] is None:
                                                r_bbox = v["sample_bbox"]
                                                cx1 = max(0, min(r_bbox[0], mx1) - 20)
                                                cy1 = max(0, min(r_bbox[1], my1) - 20)
                                                cx2 = min(width, max(r_bbox[2], mx2) + 20)
                                                cy2 = min(height, max(r_bbox[3], my2) + 20)
                                                snap_crop = frame[cy1:cy2, cx1:cx2]
                                                if snap_crop.size > 0:
                                                    comp_snap_name = f"{v['incident_id']}.jpg"
                                                    comp_snap_path = OUTPUT_DIR / comp_snap_name
                                                    cv2.imwrite(str(comp_snap_path), snap_crop)
                                                    v["snapshot_path"] = str(comp_snap_path)

    # Top HUD Banner
    cv2.rectangle(frame, (0, 0), (width, 40), (20, 20, 20), -1)
    f_ms = (time.time() - f_t0) * 1000
    hud = f"CPU INFERENCE (Intel Core i5) | Frame {frame_idx}/{total_frames} | Latency: {f_ms:.1f}ms | Violations: {len(confirmed_violations)}"
    cv2.putText(frame, hud, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    writer.write(frame)

    if frame_idx % 10 == 0 or frame_idx == total_frames:
        print(f"Processed Frame {frame_idx:3d}/{total_frames} ({timestamp_sec:.2f}s) | Latency: {f_ms:5.1f}ms | Violations: {len(confirmed_violations)} | Plates: {len(moto_plates)}")

cap.release()
writer.release()
elapsed = time.time() - t_start
avg_fps = frame_idx / elapsed

print("\n--- Step 3: Local Hardware Execution Results ---")
print(f"Hardware: 13th Gen Intel Core i5 (12 vCPUs)")
print(f"Processed: {frame_idx} frames in {elapsed:.2f}s (Average CPU throughput: {avg_fps:.2f} FPS)")
print(f"Confirmed Violations: {len(confirmed_violations)}")
print(f"Extracted License Plates: {moto_plates}")

results_data = {
    "hardware": "13th Gen Intel Core i5-1335U (CPU)",
    "input_video": INPUT_VIDEO,
    "frames_processed": frame_idx,
    "elapsed_time_sec": round(elapsed, 2),
    "fps": round(avg_fps, 2),
    "violations_detected": len(confirmed_violations),
    "violations": list(confirmed_violations.values())
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(results_data, f, indent=2)

print(f"\nSaved local output video to: {OUTPUT_VIDEO}")
print(f"Saved local results to: {OUTPUT_JSON}")
print("=== Local Hardware Inference Completed Successfully ===")
