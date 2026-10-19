# ANPR Pipeline Audit Report: License Plate Recognition

## Executive Summary
- **Source Footage:** `plates_raw.mp4`
- **Resolution:** `1280x720` @ `14.98 FPS`
- **Total Frames:** `572` (38.17s)
- **Execution Speed:** `21.0 FPS` on `Tesla T4`
- **Total Tracked Vehicles:** `31`
- **Plates Successfully Read:** `3`

## Recognized Vehicles & License Plates
| Track ID | Vehicle Type | License Plate | First Seen | Last Seen | Confidence Score |
|---|---|---|---|---|---|
| **30** | Car | **`AIE084HHI020512994`** | 6.21s | 38.17s | 0.615 |
| **72** | Truck | **`4AMARNATH`** | 12.35s | 16.28s | 0.96 |
| **157** | Truck | **`NE084HH12051`** | 28.63s | 38.17s | 0.466 |

## Artifacts Generated
- **Annotated Video:** `/content/plates_annotated.mp4`
- **Structured JSON Audit:** `/content/detected_plates.json`
