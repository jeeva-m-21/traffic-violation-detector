# Traffic Violation Analysis Report: Wrong-Way Motion

## Executive Summary
- **Source Video:** `wrongway_motion_raw.mp4`
- **Resolution:** `1280x720` @ `14.98 FPS`
- **Frames Analyzed:** `572` (38.17s)
- **Processing Speed:** `25.4 FPS` on `Tesla T4`
- **Total Violations Detected:** `6`

## Violation Incidents
| Track ID | Vehicle Type | Start Time | End Time | Duration | Confidence | Status |
|---|---|---|---|---|---|---|
| **1** | Car | 1.8s | 5.47s | 3.67s | 0.801 | **FLAGGED (WRONG WAY)** |
| **14** | Car | 3.2s | 3.54s | 0.34s | 0.613 | **FLAGGED (WRONG WAY)** |
| **19** | Car | 5.14s | 7.01s | 1.87s | 0.86 | **FLAGGED (WRONG WAY)** |
| **23** | Car | 6.87s | 38.17s | 31.3s | 0.812 | **FLAGGED (WRONG WAY)** |
| **2** | Truck | 9.61s | 10.68s | 1.07s | 0.734 | **FLAGGED (WRONG WAY)** |
| **73** | Truck | 14.75s | 16.22s | 1.47s | 0.924 | **FLAGGED (WRONG WAY)** |

## Outputs
- **Annotated Video:** `/content/wrongway_annotated.mp4`
- **Structured Audit Log:** `/content/violations.json`
