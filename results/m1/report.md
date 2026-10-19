# Milestone 1 Evaluation Report: COCO Vehicle Detection Benchmark

## 1. Executive Summary
- **Model:** `yolo11s.pt`
- **Device:** `Tesla T4`
- **Dataset:** COCO 2017 Validation Vehicle Subset (200 images, seed 42)
- **Categories:** Car, Motorcycle, Bus, Truck

## 2. Evaluation Metrics (COCO standard)
| Metric | Value | Target | Status |
|---|---|---|---|
| **mAP @ [0.50:0.95]** | **0.4763** | - | - |
| **mAP @ 0.50** | **0.6313** | - | - |
| **Mean Per-Class AP@0.5** | **0.6313** | $\ge$ 0.50 | PASS |
| **Vehicle Count MAE** | **1.205** | $\le$ 1.0 | FAIL |
| **Count Exact Match Rate** | **40.5%** | - | - |

## 3. Per-Class AP@0.5 Breakdown
| Class | Category ID | AP@0.5 |
|---|---|---|
| Car | 3 | 0.5649 |
| Motorcycle | 4 | 0.6737 |
| Bus | 6 | 0.8218 |
| Truck | 8 | 0.4648 |

## 4. Criteria Verdict
Some criteria not met; review class breakdown.
