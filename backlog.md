# Project Backlog

## SPRINT-1: Setup & Colab Compute Provisioning
- [x] SPRINT-1.1: Git initialization and project baseline commit
- [x] SPRINT-1.2: Colab VM allocation with T4 GPU runtime
- [x] SPRINT-1.3: Build comprehensive Colab notebook (`traffic_violation_m1.ipynb`)

## SPRINT-2: Dataset Fetching & Caching
- [x] SPRINT-2.1: Fetch & cache COCO annotations on Google Drive / Colab workspace (`traffic-violation-detector/`)
- [x] SPRINT-2.2: Extract sampled 200 vehicle validation images with fixed seed 42

## SPRINT-3: Model Execution & Evaluation
- [x] SPRINT-3.1: Download & verify pretrained YOLO11 detector (`yolo11s.pt`) and save model locally in `models/`
- [x] SPRINT-3.2: Run full inference and map classes to COCO standard format (`xywh`)
- [x] SPRINT-3.3: Evaluate metrics using pycocotools (mAP50: 0.6313, mAP50-95: 0.4763, per-class AP) and vehicle count MAE (1.205)
- [x] SPRINT-3.4: Save all output artifacts (metrics.json, detections.json, per_image_counts.csv, report.md) locally and on Colab

## SPRINT-4: Traffic Violation Pipeline on Dashcam Video
- [x] SPRINT-4.1: Video ingestion & Colab GPU provisioning with auto-stop wrapper
- [x] SPRINT-4.2: Multi-object vehicle tracking pipeline (YOLO11 + ByteTrack)
- [x] SPRINT-4.3: Wrong-way motion vector classification & violation alert engine
- [x] SPRINT-4.4: Render annotated video with trajectory overlays, HUD, and export `violations.json`
- [x] SPRINT-4.5: Download annotated video and violation log locally, verify Colab session shutdown

## SPRINT-5: License Plate Detection Model Weights & Setup
- [x] SPRINT-5.1: Integrate specialized YOLO license plate detector (`models/plate_detector.pt`)
- [x] SPRINT-5.2: Test plate localization on vehicle crops
- [x] SPRINT-5.3: Automated Colab session management and teardown

## SPRINT-6: OCR Preprocessing & Indian RTO Syntax Validator
- [x] SPRINT-6.1: Build OpenCV plate crop enhancement module (CLAHE, bilateral filter, Otsu)
- [x] SPRINT-6.2: Build EasyOCR recognition wrapper with alphanumeric whitelist
- [x] SPRINT-6.3: Implement Indian RTO regex validator and positional confusion matrix (`src/anpr/syntax_validator.py`)
- [x] SPRINT-6.4: Unit test OCR parser (`tests/test_syntax_validator.py`)

## SPRINT-7: End-to-End ANPR Video Pipeline on plates_raw.mp4
- [x] SPRINT-7.1: Build cascaded Vehicle + Plate tracking pipeline (`run_anpr_pipeline.py`)
- [x] SPRINT-7.2: Multi-frame temporal voting per vehicle track ID (`src/anpr/temporal_voter.py`)
- [x] SPRINT-7.3: Render annotated video `plates_annotated.mp4` with on-screen plate tags and HUD
- [x] SPRINT-7.4: Export audit log `results/plates/detected_plates.json`
- [x] SPRINT-7.5: Execute on Colab GPU with verified session shutdown (`orchestrate_anpr.py`)

## SPRINT-8: Motorcycle Rider Helmet Violation Detection Pipeline
- [x] SPRINT-8.1: Motorcycle & Rider spatial association and head ROI extraction (`src/violations/helmet_detector.py`)
- [x] SPRINT-8.2: Temporal helmet violation voter & persistence tracker (`src/violations/helmet_voter.py`)
- [x] SPRINT-8.3: Unit tests for spatial association and temporal voting (`tests/test_helmet_detector.py`)
- [x] SPRINT-8.4: Build end-to-end video pipeline runner with dual-stage detection & HUD overlays (`run_helmet_pipeline.py`)
- [x] SPRINT-8.5: Orchestrate pipeline execution on Colab GPU with verified VM teardown (`orchestrate_helmet.py`)

## SPRINT-9: Integrated Helmet-to-ANPR Pipeline & Web Application
- [x] SPRINT-9.1: Integrated violation event engine linking No-Helmet alerts to plate localization and OCR (`src/pipeline/integrated_event.py`)
- [x] SPRINT-9.2: Unit tests for integrated violation event engine (`tests/test_integrated_event.py`)
- [x] SPRINT-9.3: Build unified video processing pipeline & Colab GPU runner (`run_integrated_pipeline.py`)
- [x] SPRINT-9.4: Build FastAPI backend & REST API (`app.py`) with sample video catalog
- [x] SPRINT-9.5: Build modern HTML5/CSS3 frontend dashboard with video player & violation evidence feed (`templates/index.html`)
- [x] SPRINT-9.6: End-to-end verification, artifacts audit, and walkthrough update

## SPRINT-10: High-Clarity Plate Extraction & Multi-Line Full Format OCR
- [/] SPRINT-10.1: Build clarity evaluator & multi-line plate assembler (`src/anpr/plate_enhancer.py`)
- [ ] SPRINT-10.2: Unit tests for clarity scoring, upscaling, and multi-line plate reconstruction (`tests/test_plate_enhancer.py`)
- [ ] SPRINT-10.3: Update integrated event tracker and pipeline runners to buffer candidate plates, export high-clarity plate JPGs, and perform multi-line OCR
- [ ] SPRINT-10.4: Update Web UI to display standalone high-clarity plate JPGs alongside violation snapshots and show full plate formats
- [ ] SPRINT-10.5: Local pipeline verification, test suite execution, and walkthrough update



