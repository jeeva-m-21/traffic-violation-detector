"""Integrated Traffic Violation & License Plate Recognition Pipeline.

Performs dual-stage detection and enforcement on video:
1. Vehicle & Pedestrian Tracking (YOLO11s + ByteTrack).
2. Motorcycle Rider Association & Head ROI extraction.
3. Helmet Compliance Classification (YOLOv8 helmet detector).
4. Multi-frame persistence voting.
5. Targeted License Plate Recognition (YOLO plate detector + EasyOCR + Syntax Validator)
   triggered specifically for offending two-wheelers without helmets.
6. Violation Snapshot Evidence Caching and Video Overlay with HUD telemetry.
7. Structured JSON and Markdown Audit Log generation.
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
import easyocr

print("=== Starting Integrated Helmet Violation & Plate OCR Pipeline ===")
print("CUDA Available:", torch.cuda.is_available())
device = 0 if torch.cuda.is_available() else "cpu"
print(f"Using compute device: {device}")

VIDEO_INPUT = os.environ.get("INPUT_VIDEO", "/content/helmet_riders_clean.mp4")
VIDEO_OUTPUT = "/content/integrated_annotated.mp4"
RESULTS_JSON = "/content/integrated_results.json"
REPORT_MD = "/content/integrated_report.md"
SNAPSHOTS_DIR = Path("/content/snapshots")
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

HELMET_MODEL_PATH = "/content/helmet_detector.pt"
PLATE_MODEL_PATH = "/content/plate_detector.pt"

# Ensure models are present
if not os.path.exists(HELMET_MODEL_PATH):
    print("Downloading helmet detector weights...")
    url_h = "https://huggingface.co/sharathhhhh/safetyHelmet-detection-yolov8/resolve/main/best.pt"
    os.system(f"curl -sL -o {HELMET_MODEL_PATH} {url_h}")

if not os.path.exists(PLATE_MODEL_PATH):
    print("Downloading license plate detector weights...")
    url_p = "https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8/raw/main/license_plate_detector.pt"
    os.system(f"curl -sL -o {PLATE_MODEL_PATH} {url_p}")

# Load models
print("Loading YOLO models & EasyOCR...")
model_yolo = YOLO("yolo11s.pt")
model_helmet = YOLO(HELMET_MODEL_PATH)
model_plate = YOLO(PLATE_MODEL_PATH)
ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), verbose=False)
print("All neural network weights and OCR engines loaded.")

# Video setup
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
# OCR & Preprocessing Helpers
# -------------------------------------------------------------------------
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
def preprocess_plate(crop_bgr):
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    enhanced = clahe.apply(gray)
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    return denoised

CHAR_TO_DIGIT = {'O': '0', 'D': '0', 'Q': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'B': '8'}
DIGIT_TO_CHAR = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B'}

def format_plate(raw_text: str):
    clean = re.sub(r'[^A-Za-z0-9]', '', raw_text).upper()
    if len(clean) < 4:
        return None
    # If 6-10 characters, apply Indian positional heuristics if applicable
    if 6 <= len(clean) <= 10:
        chars = list(clean)
        for i in range(min(2, len(chars))):
            if chars[i].isdigit() and chars[i] in DIGIT_TO_CHAR:
                chars[i] = DIGIT_TO_CHAR[chars[i]]
        for i in range(max(0, len(chars) - 4), len(chars)):
            if not chars[i].isdigit() and chars[i] in CHAR_TO_DIGIT:
                chars[i] = CHAR_TO_DIGIT[chars[i]]
        return "".join(chars)
    return clean


# -------------------------------------------------------------------------
# Rider Spatial Matching & Head ROI Helpers
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
# State Tracking
# -------------------------------------------------------------------------
rider_history = defaultdict(list)
rider_first_seen = {}
rider_last_seen = {}
moto_first_seen = {}
moto_last_seen = {}
moto_plate_votes = defaultdict(lambda: defaultdict(float))
moto_resolved_plates = {}
moto_plate_boxes = {}

# Incidents
incidents = {}  # inc_id -> dict
rider_to_inc_id = {}
incident_counter = 1

TARGET_CLASSES = [0, 3]  # Person, Motorcycle

frame_idx = 0
t0 = time.time()
print("\n=== Running Dual-Stage Tracking, Helmet Detection & Targeted ANPR ===")

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
            if cls_id == 0:
                persons.append({"id": track_id, "bbox": (x1, y1, x2, y2), "conf": float(conf)})
            elif cls_id == 3:
                motorcycles.append({"id": track_id, "bbox": (x1, y1, x2, y2), "conf": float(conf)})
                if track_id not in moto_first_seen:
                    moto_first_seen[track_id] = round(timestamp_sec, 2)
                moto_last_seen[track_id] = round(timestamp_sec, 2)

    matches = match_riders_to_motorcycles(persons, motorcycles)

    # 1. Process riders for helmet compliance
    for p in persons:
        p_id = p["id"]
        px1, py1, px2, py2 = p["bbox"]

        is_rider = p_id in matches or (len(motorcycles) > 0 and len(persons) <= 3)
        if not is_rider:
            continue

        moto_id = matches.get(p_id, motorcycles[0]["id"] if motorcycles else None)

        if p_id not in rider_first_seen:
            rider_first_seen[p_id] = round(timestamp_sec, 2)
        rider_last_seen[p_id] = round(timestamp_sec, 2)

        hx1, hy1, hx2, hy2 = extract_head_roi((px1, py1, px2, py2), head_ratio=0.35, img_w=width, img_h=height)
        head_crop = orig_frame[hy1:hy2, hx1:hx2]

        is_helmet = True
        helmet_conf = 0.5

        if head_crop.size > 0 and head_crop.shape[0] >= 15 and head_crop.shape[1] >= 15:
            res_h = model_helmet.predict(head_crop, conf=0.25, device=device, verbose=False)[0]
            if len(res_h.boxes) > 0:
                top_box = res_h.boxes[0]
                pred_cls = int(top_box.cls.item())
                helmet_conf = float(top_box.conf.item())
                is_helmet = (pred_cls == 0)
            else:
                person_crop = orig_frame[max(0, py1):min(height, py2), max(0, px1):min(width, px2)]
                if person_crop.size > 0:
                    res_p = model_helmet.predict(person_crop, conf=0.25, device=device, verbose=False)[0]
                    if len(res_p.boxes) > 0:
                        pred_cls = int(res_p.boxes[0].cls.item())
                        helmet_conf = float(res_p.boxes[0].conf.item())
                        is_helmet = (pred_cls == 0)

        rider_history[p_id].append((is_helmet, helmet_conf))

        obs = rider_history[p_id]
        no_helmet_obs = [c for (h, c) in obs if not h]

        # Multi-frame persistence: >= 4 frames with >= 50% without helmet
        is_violation = (len(obs) >= 4) and (len(no_helmet_obs) / len(obs) >= 0.5)

        if is_violation:
            if p_id not in rider_to_inc_id:
                inc_id = f"INC-{incident_counter:03d}"
                incident_counter += 1
                rider_to_inc_id[p_id] = inc_id

                incidents[inc_id] = {
                    "incident_id": inc_id,
                    "rider_id": p_id,
                    "motorcycle_id": moto_id,
                    "timestamp_sec": round(timestamp_sec, 2),
                    "last_seen_sec": round(timestamp_sec, 2),
                    "violation_type": "NO_HELMET",
                    "helmet_confidence": round(float(np.mean(no_helmet_obs)), 3),
                    "plate_number": "SCANNING...",
                    "plate_confidence": 0.0,
                    "status": "PENDING_PLATE",
                    "snapshot_path": None,
                    "sample_bbox": [px1, py1, px2, py2]
                }
            else:
                inc_id = rider_to_inc_id[p_id]
                incidents[inc_id]["last_seen_sec"] = round(timestamp_sec, 2)
                incidents[inc_id]["helmet_confidence"] = round(float(np.mean(no_helmet_obs)), 3)

            # Draw Violation Rider Box
            cv2.rectangle(orig_frame, (px1, py1), (px2, py2), (0, 0, 255), 3)
            cv2.rectangle(orig_frame, (hx1, hy1), (hx2, hy2), (0, 0, 220), 2)
            cv2.putText(orig_frame, f"VIOLATION: NO HELMET (Rider #{p_id})", (px1, max(20, py1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 255), 2)
        else:
            # Compliant Rider
            cv2.rectangle(orig_frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
            cv2.putText(orig_frame, f"Rider #{p_id}: HELMET", (px1, max(20, py1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 2. Targeted ANPR for motorcycles involved in violations
    violating_moto_ids = {inc["motorcycle_id"] for inc in incidents.values() if inc.get("motorcycle_id") is not None}

    for m in motorcycles:
        m_id = m["id"]
        mx1, my1, mx2, my2 = m["bbox"]

        # Draw motorcycle tracking box
        cv2.rectangle(orig_frame, (mx1, my1), (mx2, my2), (255, 200, 0), 2)
        moto_label = f"Motorcycle #{m_id}"
        if m_id in moto_resolved_plates:
            moto_label += f" | Plate: {moto_resolved_plates[m_id]}"
        cv2.putText(orig_frame, moto_label, (mx1, my2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 2)

        # Trigger plate detector if this motorcycle has a violation
        if m_id in violating_moto_ids:
            moto_crop = orig_frame[max(0, my1):min(height, my2), max(0, mx1):min(width, mx2)]
            if moto_crop.size > 0 and moto_crop.shape[0] >= 30 and moto_crop.shape[1] >= 30:
                plate_res = model_plate.predict(moto_crop, conf=0.30, device=device, verbose=False)[0]
                if len(plate_res.boxes) > 0:
                    px1, py1, px2, py2 = plate_res.boxes.xyxy[0].int().tolist()
                    abs_px1 = mx1 + px1
                    abs_py1 = my1 + py1
                    abs_px2 = mx1 + px2
                    abs_py2 = my1 + py2
                    moto_plate_boxes[m_id] = (abs_px1, abs_py1, abs_px2, abs_py2)

                    # Highlight license plate on motorcycle
                    cv2.rectangle(orig_frame, (abs_px1, abs_py1), (abs_px2, abs_py2), (0, 255, 255), 2)

                    plate_crop = orig_frame[max(0, abs_py1):min(height, abs_py2), max(0, abs_px1):min(width, abs_px2)]
                    if plate_crop.size > 0 and plate_crop.shape[0] >= 12 and plate_crop.shape[1] >= 25:
                        # Run OCR on sample frames
                        if frame_idx % 2 == 0:
                            prep = preprocess_plate(plate_crop)
                            ocr_results = ocr_reader.readtext(
                                prep,
                                allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-',
                                detail=1,
                                paragraph=False
                            )
                            if ocr_results:
                                text = "".join([res[1] for res in ocr_results])
                                ocr_conf = float(np.mean([res[2] for res in ocr_results]))
                                formatted = format_plate(text)
                                if formatted:
                                    moto_plate_votes[m_id][formatted] += ocr_conf

            # Resolve best plate string
            if m_id in moto_plate_votes and moto_plate_votes[m_id]:
                best_plate = max(moto_plate_votes[m_id].keys(), key=lambda p: moto_plate_votes[m_id][p])
                moto_resolved_plates[m_id] = best_plate

                # Update associated incidents
                for inc in incidents.values():
                    if inc["motorcycle_id"] == m_id:
                        inc["plate_number"] = best_plate
                        inc["plate_confidence"] = round(float(np.mean(list(moto_plate_votes[m_id].values()))), 3)
                        inc["status"] = "CONFIRMED"

                        # Save violation snapshot image once resolved
                        if inc["snapshot_path"] is None:
                            snap_filename = f"{inc['incident_id']}.jpg"
                            snap_filepath = SNAPSHOTS_DIR / snap_filename
                            # Extract composite ROI with rider + bike
                            r_bbox = inc["sample_bbox"]
                            crop_x1 = max(0, min(r_bbox[0], mx1) - 20)
                            crop_y1 = max(0, min(r_bbox[1], my1) - 20)
                            crop_x2 = min(width, max(r_bbox[2], mx2) + 20)
                            crop_y2 = min(height, max(r_bbox[3], my2) + 20)
                            snap_crop = orig_frame[crop_y1:crop_y2, crop_x1:crop_x2]
                            if snap_crop.size > 0:
                                cv2.imwrite(str(snap_filepath), snap_crop)
                                inc["snapshot_path"] = f"/static/snapshots/{snap_filename}"

    # Top HUD Banner
    hud_bg = orig_frame[0:52, 0:width].copy()
    cv2.rectangle(orig_frame, (0, 0), (width, 52), (18, 18, 18), -1)
    cv2.addWeighted(hud_bg, 0.2, orig_frame[0:52, 0:width], 0.8, 0, orig_frame[0:52, 0:width])

    total_riders = len(rider_history)
    total_violations = len(incidents)
    confirmed_plates = len([i for i in incidents.values() if i["status"] == "CONFIRMED"])
    compliant = max(0, total_riders - total_violations)
    fps_est = frame_idx / max(0.001, time.time() - t0)

    hud_t1 = f"AI TRAFFIC SURVEILLANCE  |  Frame: {frame_idx}/{total_frames} ({timestamp_sec:.1f}s)  |  Speed: {fps_est:.1f} FPS"
    cv2.putText(orig_frame, hud_t1, (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)

    hud_t2 = f"Riders: {total_riders}  |  Compliant: {compliant}  |  HELMET VIOLATIONS: {total_violations}  |  Plates Extracted: {confirmed_plates}"
    color_viol = (0, 0, 255) if total_violations > 0 else (0, 255, 0)
    cv2.putText(orig_frame, hud_t2, (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color_viol, 2)

    writer.write(orig_frame)

    if frame_idx % 60 == 0:
        print(f"Frame {frame_idx}/{total_frames} ({timestamp_sec:.1f}s) - Violations: {total_violations} - Plates: {confirmed_plates} - FPS: {fps_est:.1f}")

cap.release()
writer.release()
elapsed = time.time() - t0
avg_fps = frame_idx / elapsed
print(f"\nPipeline finished in {elapsed:.2f}s ({avg_fps:.1f} FPS average).")

# Finalize any incidents without snapshots
for inc in incidents.values():
    if inc["snapshot_path"] is None:
        snap_filename = f"{inc['incident_id']}.jpg"
        snap_filepath = SNAPSHOTS_DIR / snap_filename
        r_bbox = inc["sample_bbox"]
        if r_bbox:
            crop_x1 = max(0, r_bbox[0] - 20)
            crop_y1 = max(0, r_bbox[1] - 20)
            crop_x2 = min(width, r_bbox[2] + 20)
            crop_y2 = min(height, r_bbox[3] + 20)
            # Take from last frame
            snap_crop = orig_frame[crop_y1:crop_y2, crop_x1:crop_x2]
            if snap_crop.size > 0:
                cv2.imwrite(str(snap_filepath), snap_crop)
                inc["snapshot_path"] = f"/static/snapshots/{snap_filename}"

# Save JSON results
all_incidents_list = list(incidents.values())
results_data = {
    "video": VIDEO_INPUT,
    "total_frames": frame_idx,
    "fps": fps,
    "duration_sec": round(frame_idx / fps, 2),
    "elapsed_time_sec": round(elapsed, 2),
    "avg_fps": round(avg_fps, 1),
    "total_riders_tracked": len(rider_history),
    "total_motorcycles_tracked": len(moto_first_seen),
    "compliant_riders": max(0, len(rider_history) - len(incidents)),
    "violations_detected": len(incidents),
    "plates_extracted": len([i for i in all_incidents_list if i["status"] == "CONFIRMED"]),
    "incidents": all_incidents_list
}

with open(RESULTS_JSON, "w") as f:
    json.dump(results_data, f, indent=2)
print(f"Exported results to {RESULTS_JSON}")

# Markdown report
report_md = f"""# Integrated Traffic Enforcement: Helmet Violation & License Plate OCR Report

## 1. Summary
- **Input Video**: `{VIDEO_INPUT}`
- **Processed**: `{frame_idx}` frames (`{frame_idx/fps:.2f}s`) at **`{avg_fps:.1f} FPS`** on Tesla T4 GPU
- **Total Riders Tracked**: **`{len(rider_history)}`**
- **Helmet Violations Detected**: **`{len(incidents)}`**
- **Plates Successfully Extracted**: **`{len([i for i in all_incidents_list if i["status"] == "CONFIRMED"])}`**

---

## 2. Violation Incidents & Extracted Number Plates
| Incident ID | Rider ID | Motorcycle ID | Time | Extracted Plate Number | Status | Helmet Conf | Plate Conf |
|---|---|---|---|---|---|---|---|
"""

for inc in all_incidents_list:
    report_md += f"| **{inc['incident_id']}** | #{inc['rider_id']} | #{inc['motorcycle_id']} | {inc['timestamp_sec']}s | **`{inc['plate_number']}`** | {inc['status']} | {inc['helmet_confidence']} | {inc['plate_confidence']} |\n"

with open(REPORT_MD, "w") as f:
    f.write(report_md)
print(f"Exported report to {REPORT_MD}")

print("=== Integrated Pipeline Finished Successfully ===")
