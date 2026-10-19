# Project Backlog

## SPRINT-1: Setup & Colab Compute Provisioning
- [x] SPRINT-1.1: Git initialization and project baseline commit
- [/] SPRINT-1.2: Colab VM allocation with GPU and Drive mount verification
- [ ] SPRINT-1.3: Build comprehensive Colab notebook (`traffic_violation_m1.ipynb`)

## SPRINT-2: Dataset Fetching & Caching
- [ ] SPRINT-2.1: Fetch & cache COCO annotations on Google Drive (`MyDrive/traffic-violation-detector/`)
- [ ] SPRINT-2.2: Extract sampled 200 vehicle validation images with fixed seed

## SPRINT-3: Model Execution & Evaluation
- [ ] SPRINT-3.1: Download & verify pretrained YOLO11 detector (`yolo11s.pt`)
- [ ] SPRINT-3.2: Run full inference and map classes to COCO standard format
- [ ] SPRINT-3.3: Evaluate metrics using pycocotools (mAP50, mAP50-95, per-class AP) and vehicle count MAE
- [ ] SPRINT-3.4: Save all output artifacts (metrics.json, detections.json, per_image_counts.csv, report.md) to Drive and locally
