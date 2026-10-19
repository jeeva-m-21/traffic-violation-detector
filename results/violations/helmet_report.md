# Helmet Violation Detection Pipeline Audit Report

## 1. Summary Metrics
- **Input Video**: `/content/helmet_riders_clean.mp4`
- **Resolution**: `1000x522` @ `30.00 FPS`
- **Total Frames Processed**: `1693` (`56.43s` footage)
- **Processing Time**: `61.19s` on NVIDIA Tesla T4 GPU
- **Execution Throughput**: **`27.7 FPS`** (Real-Time Performance)
- **Total Two-Wheeler Riders Tracked**: **`28`**
- **Helmet Compliant Riders**: **`25`**
- **Confirmed Helmet Violations**: **`3`**
- **Compliance Rate**: **`89.3%`**

---

## 2. Detected Helmet Violation Incidents
| Rider Track ID | Associated Moto ID | First Seen | Violation Confirmed | Last Seen | Confidence | Frames Observed | Status |
|---|---|---|---|---|---|---|---|
| **#70** | #71 | 30.67s | 31.97s | 34.2s | 0.462 | 106 | **VIOLATION (NO HELMET)** |
| **#109** | #110 | 45.07s | 45.23s | 45.3s | 0.588 | 8 | **VIOLATION (NO HELMET)** |
| **#112** | #113 | 50.43s | 50.53s | 56.43s | 0.716 | 181 | **VIOLATION (NO HELMET)** |

---

## 3. Pipeline Architecture
1. **Primary Object Tracker**: YOLO11s detector with ByteTrack temporal association (`bytetrack.yaml`) targeting COCO class `0` (person) and `3` (motorcycle).
2. **Spatial Rider Association**: Geometric vertical and horizontal intersection matrix matching rider persons with motorcycle seats.
3. **Head Region of Interest (ROI)**: Extracted upper 35% bounding box region of each rider.
4. **Helmet Classifier**: Specialized YOLOv8 detector (`models/helmet_detector.pt`) inferencing on head/rider crops (`With helmet` vs `Without helmet`).
5. **Temporal Persistence Voting**: Multi-frame accumulation requiring consistent no-helmet status over $\ge 4$ frames, eliminating transient occlusion and motion blur anomalies.
6. **Telemetry & Visual HUD**: On-screen dynamic bounding boxes (Green: Compliant, Red: Violation) with active HUD counter and telemetry overlay.
