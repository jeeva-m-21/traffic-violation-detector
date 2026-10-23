# 🚦 Smart Traffic Violation Detection & ANPR Platform

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![YOLO11](https://img.shields.io/badge/Ultralytics-YOLO11-00FFFF.svg)](https://docs.ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/Tests-36%20Passed-brightgreen.svg)](tests/)

An end-to-end, real-time Computer Vision and Deep Learning traffic enforcement intelligence platform. It features multi-camera dashcam ingestion, ego-motion compensated wrong-way driving detection, motorcycle rider-passenger helmet violation association, and high-clarity Automatic Number Plate Recognition (ANPR) with Indian RTO syntax validation.

---

## 🌟 Key Highlights

- **🛣️ Ego-Compensated Wrong-Way Driving Detection:**
  Eliminates false alerts during lane-overtaking maneuvers by combining road corridor vector projection, ego-motion compensation, and a temporal hysteresis state machine (K-of-N frame confirmation).
- **🛵 Motorcycle Helmet Enforcement Engine:**
  Associates motorcycle track IDs with riders and passengers via spatial IoU intersection, tracks head ROIs, and applies temporal smoothing to confirm non-compliance.
- **🔍 High-Clarity ANPR & Super-Resolution:**
  Scores license plate image clarity (contrast, sharpness, edge density), applies 3x bicubic super-resolution, and reconstructs multi-line Indian license plates with OCR positional correction (e.g. `O` vs `0`, `I` vs `1`).
- **💻 Unified Enforcement Dashboard:**
  Built with FastAPI and modern UI: includes video playback speed controls (0.25x – 2.0x), live evidence feed, interactive threshold tuner, and 1-click audit trail export (JSON & CSV).
- **⚡ Dual-Engine Orchestration:**
  Supports local CPU inference runners as well as automated Google Colab GPU cloud orchestration with automatic VM teardown.

---

## 📐 Architecture Pipeline

```mermaid
flowchart TD
    VideoInput["📹 Input Video / Dashcam Stream"] --> Detector["🎯 YOLO11 Multi-Object Detector & ByteTrack"]
    
    Detector --> Vehicles["🚗 Vehicles (Cars, Trucks, Buses)"]
    Detector --> Motorcycles["🏍️ Motorcycles & Riders"]

    subgraph WrongWayDetection["Road-Aware Wrong-Way Detection"]
        Vehicles --> Kinematics["Kinematics & Optical Flow"]
        Kinematics --> CorridorProj["Road Corridor Projection"]
        CorridorProj --> Hysteresis["K-of-N Hysteresis Filter"]
        Hysteresis --> WWAlert["⚠️ Wrong-Way Violation"]
    end

    subgraph HelmetDetection["Helmet Compliance Pipeline"]
        Motorcycles --> SpatialIoU["Spatial IoU Association"]
        SpatialIoU --> HeadROI["Head ROI Extractor"]
        HeadROI --> HelmetClassifier["Helmet vs No-Helmet Classifier"]
        HelmetClassifier --> TemporalVoter["Temporal Voting Gate"]
        TemporalVoter --> NHAlert["⚠️ No-Helmet Violation"]
    end

    subgraph ANPR["High-Clarity ANPR & OCR"]
        WWAlert --> PlateDetector["License Plate Localizer"]
        NHAlert --> PlateDetector
        PlateDetector --> ClarityScore["Clarity & Sharpness Scoring"]
        ClarityScore --> SuperRes["Bicubic Super-Resolution & CLAHE"]
        SuperRes --> EasyOCR["EasyOCR Engine"]
        EasyOCR --> RTOValidator["Indian RTO Syntax Validator & Disambiguation"]
    end

    ANPR --> EvidenceFeed["📋 Unified Incident Feed & REST API / Dashboard"]
```

---

## 📸 Dashboard Preview

![Unified Traffic Violation & ANPR Dashboard](static/dashboard_preview.png)

---

## 🚀 Quickstart

### 1. Prerequisites
- Python 3.10+ (Python 3.11 recommended)
- `uv` (recommended) or `pip`
- Git

### 2. Installation

Clone repository:
```bash
git clone https://github.com/jeeva-m-21/traffic-violation-detector.git
cd traffic-violation-detector
```

Create virtual environment and install dependencies:
```bash
# Using uv (fastest)
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Or using standard pip
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set up environment variables:
```bash
cp .env.example .env
```

### 3. Run Unit Tests
```bash
python -m unittest discover tests
```
*Expected: 36/36 tests passing covering plate enhancer, syntax validator, wrong-way kinematics, helmet detector, and API.*

### 4. Launch the Web Dashboard
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
Navigate to `http://localhost:8000` in your browser.

---

## 🧪 CLI Pipelines & Inference

### Local End-to-End Inference Runner
Run the CPU-optimized inference pipeline on any test video:
```bash
python run_local_inference.py
```

### Integrated Violation & ANPR Pipeline
Process input video and generate full audit reports and snapshots:
```bash
python run_integrated_pipeline.py
```

### Run Specialized Sub-Pipelines
```bash
# Helmet violation detection
python run_helmet_pipeline.py

# ANPR and license plate recognition
python run_anpr_pipeline.py

# Wrong-way driving detection
python run_violation_pipeline.py
```

---

## 🌐 REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web enforcement UI dashboard |
| `GET` | `/api/violations` | Retrieve detected violation incidents with snapshot URIs |
| `GET` | `/api/metrics` | Summary metrics (total violations, helmet, wrong-way, ANPR confidence) |
| `GET` | `/api/settings` | Retrieve active detector and OCR threshold settings |
| `POST` | `/api/settings` | Dynamically tune thresholds (clarity score, confidence, IoU) |
| `GET` | `/api/export/csv` | Export violations as CSV audit sheet |
| `GET` | `/api/export/json` | Export violations as structured JSON log |
| `POST` | `/api/process` | Upload a video for pipeline analysis |

---

## 📁 Repository Structure

```
├── app.py                      # FastAPI web server and REST API
├── PRD.md                      # Product Requirements & Engineering Spec
├── backlog.md                  # Sprint tracker and backlog items
├── requirements.txt            # Python dependencies
├── .env.example                # Example environment configuration
├── config/
│   └── settings.json           # Runtime thresholds and configurations
├── src/
│   ├── anpr/                   # Plate enhancement, multi-line OCR, RTO syntax
│   ├── violations/             # Helmet detection, wrong-way kinematics, road segmenter
│   └── pipeline/               # Integrated violation-to-plate event engine
├── templates/
│   └── index.html              # Modern dark-mode dashboard interface
├── static/                     # Evidence snapshots, CSS assets, preview images
├── tests/                      # Automated test suite (36 unit tests)
├── results/                    # Validation benchmarks, reports, and JSON audits
└── run_*.py                    # Pipeline execution scripts
```

---

## 📄 License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
