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
