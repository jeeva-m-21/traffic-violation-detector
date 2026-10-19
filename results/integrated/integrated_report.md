# Integrated Traffic Enforcement: Helmet Violation & License Plate OCR Report

## 1. Summary
- **Input Video**: `/content/helmet_riders_clean.mp4`
- **Processed**: `1693` frames (`56.43s`) at **`27.4 FPS`** on Tesla T4 GPU
- **Total Riders Tracked**: **`28`**
- **Helmet Violations Detected**: **`3`**
- **Plates Successfully Extracted**: **`3`**

---

## 2. Violation Incidents & Extracted Number Plates
| Incident ID | Rider ID | Motorcycle ID | Time | Extracted Plate Number | Status | Helmet Conf | Plate Conf |
|---|---|---|---|---|---|---|---|
| **INC-001** | #70 | #71 | 31.97s | **`8703`** | CONFIRMED | 0.462 | 0.249 |
| **INC-002** | #109 | #110 | 45.23s | **`9495`** | CONFIRMED | 0.588 | 0.345 |
| **INC-003** | #112 | #113 | 50.53s | **`06216`** | CONFIRMED | 0.716 | 0.166 |
