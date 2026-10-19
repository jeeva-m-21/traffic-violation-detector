# PRD Document

## 1. Overview

This product is an AI agent that completes data science and computer vision tasks by writing, running, and checking code in a Google Colab environment. A user gives the agent a task specification, such as "evaluate a pretrained vehicle detector on this dataset and report metrics." The agent plans the work, writes notebook cells, executes them on a Colab GPU runtime, reads the outputs and errors, retries within limits, and returns a report with artifacts saved to Google Drive.

This PRD covers the agent's core loop and the first milestone, a single easy task. The longer-term goal is a traffic violation pipeline, starting with helmet detection on dashcam footage. The agent is the mechanism for running and iterating on those pipelines, so the foundation has to work before the violation logic is built on top of it.

**Assumptions:** Colab does not provide a public agent API. This PRD assumes an execution bridge (defined in Open Questions) that lets the agent send code to a Colab or Jupyter-compatible runtime and receive outputs. Phase 1 can use a manual bridge, where a human pastes cells, to validate code quality before automating execution.

## 2. Goals

- G1: Execute a multi-step Colab task (install, download, run a model, evaluate, save) from a task spec with no manual code writing.
- G2: Detect and recover from common failures (missing packages, out-of-memory errors, wrong paths) within a bounded retry budget.
- G3: Produce reproducible results: same spec and seed yield metrics within a small tolerance.
- G4: Leave a complete audit trail: every executed cell, its output, and every artifact produced.
- G5: Complete the first milestone task (Section 4 of the task section below) end to end, with measured success criteria.

## 3. Non-Goals

- Building a general-purpose coding agent for arbitrary software projects.
- Training large models from scratch. Milestone 1 uses pretrained weights.
- Autonomous decisions on legal or enforcement outcomes. Any violation logic in later phases requires human review.
- A custom notebook UI. Users interact through the task spec and the report.
- Bypassing Colab limits, quotas, or terms of service. Usage must stay within what Google allows (see Risks).
- Multi-user collaboration, billing, or team permissions in v1.

## 4. Target Users

- **Primary: ML and computer vision engineers** who want to run repeatable experiments and evaluations without writing boilerplate notebook code.
- **Secondary: product managers and analysts** who need a quick, verified metric on a dataset and can read a report but do not write code.
- **Internal: the project team** building the traffic violation pipeline, who need a reliable way to run experiments on Colab GPUs.

## 5. User Stories

- As an ML engineer, I want to submit a task spec describing a dataset, a model, and metrics, so that I get evaluation results without writing the notebook myself.
- As an ML engineer, I want to see every cell the agent ran and its output, so that I can audit and reproduce the result.
- As a user, I want the agent to stop and ask me before large downloads, deletions, or long GPU jobs, so that I don't lose time or quota by accident.
- As a user, I want the agent to recover from a missing package or a runtime disconnect without restarting from scratch, so that long jobs finish.
- As a product manager, I want a short report with pass/fail against the success criteria, so that I can decide whether to proceed to the next milestone.
- As an engineer, I want to re-run the same task with a different model or threshold by changing only the spec, so that comparisons are fast.

## 6. Functional Requirements

**Task intake**
- FR-01: Accept a task spec in YAML or JSON with fields for dataset source, sample size, model, parameters, metrics, output location, and success criteria.
- FR-02: Validate the spec before execution and report any missing or conflicting fields.

**Execution**
- FR-03: Execute code cells on a Colab runtime with a configurable timeout per cell (default 10 minutes; longer for model training).
- FR-04: Capture stdout, stderr, return values, exceptions, and files written to the runtime.
- FR-05: Install Python packages with pinned versions when specified.
- FR-06: Mount Google Drive and read or write only under an approved working folder (for example, `MyDrive/agent_runs/`).
- FR-07: Detect GPU availability and report the runtime type (for example, T4) at the start of each run.

**Reasoning and recovery**
- FR-08: Generate code for each planned step, execute it, and check the output against the step's expected result.
- FR-09: On failure, retry with the error in context, up to a maximum of 3 attempts per step.
- FR-10: After retries are exhausted, stop the step, save the state, and escalate to the user with a summary of what failed.
- FR-11: Resume from the last successful step after a runtime disconnect, using checkpoint files on Drive.

**Approval gates**
- FR-12: Require user approval before downloads larger than 1 GB, deletions, writes outside the approved folder, or GPU jobs estimated to exceed 30 minutes.

**Outputs and reporting**
- FR-13: Write a run log containing every executed cell, its output, timestamps, and the spec version.
- FR-14: Save all artifacts (metrics JSON, CSV, predictions, annotated samples) to the run folder.
- FR-15: Generate a report with a pass/fail table against the success criteria and links to artifacts.

**Configuration**
- FR-16: Support a config cell where thresholds and parameters are set once and reused, so re-running analysis does not require re-running heavy steps.

## 7. Non-Functional Requirements

- **Reproducibility:** Fixed random seeds, pinned package versions, and recorded model file hashes in every run.
- **Auditability:** Every run is fully logged and can be re-created from its log without the agent.
- **Security:** No API keys or credentials written into notebooks or logs. Use Colab's secrets mechanism or environment variables. Only read the paths the spec allows.
- **Cost control:** Track GPU minutes per run and enforce a per-run budget set in the spec.
- **Reliability:** Target completion without human intervention for at least 80% of runs on the Milestone 1 task.
- **Performance:** The Milestone 1 task completes in under 20 minutes of wall-clock time on a T4 runtime, excluding one-time downloads.
- **Usability:** The report is readable by a non-engineer; technical detail sits in appendices.
- **Maintainability:** Each step is a separate function or cell with a clear input and output, so a failing step can be fixed alone.

## 8. Success Metrics

| Metric | Target | How measured |
|---|---|---|
| Task completion rate (Milestone 1 task, no human help) | ≥ 80% over 10 runs | Run log status |
| First-attempt cell success rate | ≥ 70% | Cells that run without error on the first attempt |
| Median time to result | ≤ 20 min | Run timestamps |
| Metric reproducibility | Within ±0.5 AP points across repeat runs with the same seed | Metrics JSON comparison |
| Approval gates triggered correctly | 100% of defined triggers | Test cases with known triggers |
| Artifacts complete per run | 100% | Checklist in the run report |

These targets are proposals to calibrate after the first baseline runs.

## 9. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Colab automation may conflict with Google's terms of service | Blocked access or account action | Review the current terms before automating. Start with a manual bridge. Confirm the approved route with Google before scaling. |
| Runtime disconnects or idle timeouts | Lost progress and state | Checkpoint to Drive after each step. Resume logic (FR-11). |
| GPU unavailable or quota exhausted | Runs stall | Check GPU at start, fall back to CPU for the easy task if needed (slower but workable), and report the runtime type. |
| Agent writes incorrect code (hallucinated APIs, wrong paths) | Wrong results or wasted runs | Step-level output checks, retries with error context, and pinned package versions. |
| Package version drift (for example, Ultralytics changing its API or defaults) | Unreproducible results | Pin versions in the spec and log them. |
| Dataset download blocked or slow | Task fails at step 1 | Cache downloads on Drive. Keep the subset small. Use a retry-with-backoff download function. |
| Dataset licensing limits use | Legal exposure | Check each dataset's license before use. Keep data internal and out of public outputs. |
| Evaluation mismatch (wrong class IDs, bbox format errors) | Misleading metrics | Unit-test the conversion step on one image with known ground truth. Show the mapping table in the report. |
| Privacy when later using dashcam footage | Personal data exposure | Store only required evidence, and plan for face and plate handling before Phase 3 work. |
| Agent acts on destructive commands | Data loss on Drive | Deny-list dangerous commands, restrict writes to the run folder, require approval for deletions. |

## 10. Open Questions

1. **Execution bridge:** Which mechanism will let the agent run cells on Colab? Options include a manual copy-paste loop, a local Jupyter-compatible runtime connected to Colab, or a sanctioned integration. This needs a decision and a terms-of-service check.
2. **Colab tier:** Is a paid tier available for longer sessions and more reliable GPU access? This affects the timeout and budget settings.
3. **Approval granularity:** Should approvals be per step, per run, or once per spec?
4. **Report audience:** Who reads the report: engineers, product managers, or both? This sets the detail level.
5. **Storage:** Is Google Drive the only permitted storage, or is a project bucket acceptable?
6. **Model choice policy:** Is any pretrained model acceptable, or only models with specific licenses?
7. **Success criteria authority:** Who sets the thresholds for passing the Milestone 1 task?

---

# Implementation Plan

## Phase 1: Execution foundation

Goal: the system can run code on a Colab GPU runtime and return structured results.

- Choose and verify the execution bridge (Open Question 1). Document the terms-of-service review.
- Build a `execute(code, timeout)` tool that returns stdout, stderr, exception, and produced files.
- Build helpers for package install with pinned versions, Drive mount, and GPU check.
- Build a manual fallback: the agent produces each cell, a human runs it, and pastes the output back. Use this to test prompts and output parsing before automation.
- Test the execution tool with a trivial notebook: install a package, print the GPU name, write a file to Drive.

## Phase 2: State, safety, and logging

Goal: every run is recoverable, auditable, and bounded.

- Define the run folder layout on Drive: `agent_runs/<run_id>/` with `spec.yaml`, `log.jsonl`, `checkpoints/`, `artifacts/`, and `report.md`.
- Implement the run log: one JSON line per cell with timestamp, step ID, code hash, output, status, and attempt number.
- Implement checkpoint save and load after each successful step.
- Implement approval gates (FR-12) as a function that pauses the run and asks the user. Test each trigger.
- Implement a command deny-list and path allow-list for file operations.
- Implement GPU minute tracking and a per-run budget check.

## Phase 3: Agent loop

Goal: the agent turns a spec into executed, checked steps.

- Build the planner: converts a spec into an ordered list of steps, each with a goal, inputs, expected output, and a check.
- Build the code writer: produces one cell per step, given the step, the previous outputs, and the error history from failed attempts.
- Build the checker: compares a step's output with its expected result. Use deterministic checks where possible (file exists, metric is a number in range, shape matches) and use the model only for judgment calls.
- Implement the retry policy (FR-09, FR-10): at most 3 attempts per step, then escalate.
- Implement the report generator (FR-15) from the run log and artifacts.
- Test the full loop on a toy task (for example, a 5-image evaluation) before the real task.

## Phase 4: Milestone 1 task and evaluation harness

Goal: the easy task runs end to end and produces verifiable metrics.

- Implement the dataset sampling, download, and caching steps (see Step-by-Step Completion Plan).
- Implement the inference step and the COCO-format conversion.
- Implement the evaluation step using pycocotools, restricted to the vehicle categories.
- Add a unit check: convert one image's predictions and confirm the bounding box format and class IDs against the ground truth.
- Run the baseline three times with the same seed and record the variation.
- Calibrate the success thresholds in the spec against the baseline, then lock them.

## Phase 5: Hardening and handoff

Goal: the system is repeatable and ready for the next task.

- Run the Milestone 1 task 10 times and measure the success metrics in Section 8.
- Document failure modes observed and their fixes.
- Write a short runbook: how to submit a spec, read a report, and resume a failed run.
- Write the spec template for the next task (helmet detection on the dashcam clip), with the dataset and license checks included.

## Milestones

| Milestone | Definition of done | Depends on |
|---|---|---|
| M0: Access confirmed | Colab GPU runtime reachable, Drive mounted, a test cell runs and writes a file | Execution bridge decision |
| M1: Agent executes cells | Agent produces and runs cells for a toy task, with logs and checkpoints | Phases 1 and 2 |
| M2: Easy task passes once | Milestone 1 task passes its success criteria in one full run | Phases 3 and 4 |
| M3: Repeatable | 10 runs with at least 80% completion and reproducibility within the stated tolerance | Phase 5 |
| M4: Next task scoped | Helmet-detection spec written, dataset license cleared, approvals agreed | M3 and dataset review |

## Dependencies

- A Google account with Colab access and enough Drive storage (about 2 to 3 GB for the dataset subset, cache, and outputs).
- A confirmed execution bridge and a terms-of-service review.
- Internet access from the Colab runtime to the dataset host and the package index.
- Access to Ultralytics, pycocotools, and PyTorch packages, which install from public indexes.
- The pretrained model weight (`yolo11s.pt`) downloaded by the library on first use.
- For M4: the dashcam footage's usage rights and the license of any plate dataset.

---

# Dataset Recommendation

| Dataset Name | Source | Size | Format | License | Rationale |
|--------------|--------|------|--------|---------|-----------|
| COCO 2017 val (vehicle subset, 200 images) | cocodataset.org; images at images.cocodataset.org/val2017; annotations at images.cocodataset.org/annotations/annotations_trainval2017.zip | Full val: 5,000 images (about 778 MB zipped). Milestone 1 uses 200 sampled images with vehicles, about 40 MB. Annotations zip about 241 MB, of which only one JSON is needed. | JPEG images plus COCO-format JSON annotations (bounding boxes, category IDs, iscrowd flags) | Annotations: CC BY 4.0. Images come from Flickr under varied licenses, so check COCO's terms before redistribution. Fine for internal evaluation. | Public, widely used, and has reliable ground truth for cars, motorcycles, buses, and trucks. Its class labels match the pretrained detector, so no training is needed. Its validation split is separate from the training split the pretrained model used. Its images are mostly not Indian road scenes, so it measures general vehicle detection, not dashcam performance. |
| Own dashcam clip (Milestone 3 onward) | Project footage | Currently one clip, 38 seconds, 1280x720, 29.97 fps | MP4 video, labels to be created | Unknown for the uploaded clip. It appears to come from a public video platform, so confirm rights before use or redistribution. | Needed for the traffic violation work, since COCO does not match dashcam conditions. Use it for evaluation and labeling, not for the first milestone. |
| Indian number plate dataset (Roboflow export in the notebook) | Roboflow Universe export, as used in the prototype | 1,215 train, 78 val, 59 test images | YOLOv8 format (images, labels, data.yaml) | Check the license in the export's metadata before use. Not confirmed in the notebook. | Relevant to the later plate-reading step. Not needed for Milestone 1. Its validation mAP50 of 0.995 comes from a small set and should be re-measured on the test split before it is trusted. |

**Recommendation for Milestone 1:** use the COCO 2017 vehicle subset. It is public, has clean ground truth, needs no labeling, and runs quickly on a T4. Keep the sample at 200 images so runs finish in minutes.

---

# Easy Task Selection

- **Task Name:** Vehicle detection benchmark on a COCO validation subset
- **Description:** The agent loads 200 COCO validation images that contain at least one vehicle, runs a pretrained YOLO11 detector restricted to car, motorcycle, bus, and truck, converts the predictions to COCO format, and evaluates them against the ground truth. It reports average precision per class, overall AP, and a per-image count error. It saves all artifacts to Drive.
- **Inputs:**
  - Task spec (YAML) with: dataset name, annotation URL, image URL pattern, sample size (200), random seed (42), model (`yolo11s.pt`), image size (640), confidence threshold (0.25), vehicle category IDs (3, 4, 6, 8), output folder.
  - Colab GPU runtime (T4 preferred; CPU works but is slower).
- **Expected Outputs:**
  - `detections.json` in COCO results format.
  - `metrics.json` with mAP@[.5:.95], mAP@0.5, per-class AP@0.5, count MAE, and count exact-match rate.
  - `per_image_counts.csv` with ground-truth and predicted vehicle counts per image.
  - `run_log.jsonl` and `report.md`.
- **Success Criteria** (provisional; calibrate after the baseline run):
  1. The run completes without human intervention in at least 80% of attempts.
  2. Mean AP@0.5 across the four vehicle classes is at least 0.50.
  3. Count MAE is at most 1.0 vehicle per image.
  4. Three repeated runs with the same seed produce metrics within ±0.5 AP points.
  5. All four output files exist on Drive and the report lists every executed step.

---

# Step-by-Step Completion Plan

1. **Create the Colab notebook and set the runtime.** Open a new notebook. Go to Runtime → Change runtime type → T4 GPU. Confirm with `!nvidia-smi -L`.
2. **Mount Drive and set paths.** Mount Google Drive and create the run folder `agent_runs/<run_id>/`. Write the spec to that folder.
3. **Install dependencies with pinned versions.** Install `ultralytics` and `pycocotools`. Record the installed versions in the log.
4. **Download the annotation file.** Download `annotations_trainval2017.zip` (about 241 MB), extract only `annotations/instances_val2017.json`, and cache the zip on Drive so it isn't downloaded again.
5. **Select the subset.** Find all validation images with at least one non-crowd vehicle annotation. Sample 200 with seed 42. Save the list of selected image IDs to the run folder. Stop and ask for approval if the sample is not exactly 200.
6. **Download the sampled images.** Download each image from the COCO image host with retries and backoff. Verify the count of files on disk matches the sample. Cache the images on Drive.
7. **Run a one-image sanity check.** Run inference on one image and print its raw boxes, class IDs, and scores. Confirm the class IDs are 2, 3, 5, and 7 and that the boxes are in pixel coordinates. This step catches the most common error in this task.
8. **Run inference on the full subset.** Run the detector on the 200 images with image size 640, confidence 0.25, and the four vehicle classes. Convert each box from `xyxy` to `xywh`. Map each YOLO class index to its COCO category ID (2→3, 3→4, 5→6, 7→8).
9. **Evaluate with pycocotools.** Load the ground truth, load the detections, restrict the evaluation to the 200 image IDs and the four category IDs, and run the evaluation. Extract overall AP and AP@0.5. Extract per-class AP@0.5 from the precision array.
10. **Compute count metrics.** For each image, count ground-truth vehicles (non-crowd) and predicted vehicles. Compute the mean absolute error and exact-match rate.
11. **Check against success criteria.** Compare each metric to its threshold. Record pass or fail in the report.
12. **Save artifacts.** Write `detections.json`, `metrics.json`, `per_image_counts.csv`, and the run log to the run folder on Drive.
13. **Repeat for reproducibility.** Run steps 8 to 12 two more times with the same seed and compare metrics. Record the spread.
14. **Generate the report.** Write `report.md` with the spec, the runtime type, package versions, the step table, pass/fail results, and links to artifacts.

**Failure handling for this task:**
- Download failure in step 4 or 6: retry with backoff, then stop and ask the user.
- Out-of-memory in step 8: reduce batch size to 8, then 4. Log the change.
- Runtime disconnect: resume from the last checkpoint (step 5 output and the image cache are the expensive parts).
- Metric far below expectations: check the class ID mapping and box format before changing the model or threshold.

---

# Code / Colab Notebook Outline

**Notebook structure**

1. Setup: runtime check, install, Drive mount
2. Config: spec loading and constants
3. Data: annotation download, subset selection, image download
4. Sanity check: one-image inference and conversion
5. Inference: full subset run and detection export
6. Evaluation: pycocotools metrics and count metrics
7. Outputs: save artifacts and write the report
8. Repeat runs (optional): loop over seeds

**Cell 1: Setup**

```python
# Cell 1: install pinned dependencies (run once per session)
!pip install -q ultralytics==8.3.0 pycocotools
!nvidia-smi -L
```

Note: `8.3.0` is an example pin. Replace it with the version you verify in Phase 1, and record it in the run log.

**Cell 2: Config and Drive**

```python
# Cell 2: configuration and Drive mount
import json, csv, random, urllib.request, zipfile
from pathlib import Path
from google.colab import drive

drive.mount("/content/drive")

SEED = 42
N_IMAGES = 200
IMGSZ = 640
CONF = 0.25
MODEL_NAME = "yolo11s.pt"

ROOT = Path("/content/coco_vehicle_eval")          # fast local cache
RUN_DIR = Path("/content/drive/MyDrive/agent_runs/coco_vehicle_m1")
IMG_DIR = ROOT / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)
RUN_DIR.mkdir(parents=True, exist_ok=True)

ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMG_URL = "http://images.cocodataset.org/val2017/{}"

VEHICLE_COCO_IDS = [3, 4, 6, 8]                    # car, motorcycle, bus, truck
YOLO_TO_COCO = {2: 3, 3: 4, 5: 6, 7: 8}            # YOLO class index -> COCO category ID
NAMES = {3: "car", 4: "motorcycle", 6: "bus", 8: "truck"}
```

**Cell 3: Annotations and subset selection**

```python
# Cell 3: download annotations, select a seeded subset of vehicle images
ann_zip = ROOT / "annotations_trainval2017.zip"
if not ann_zip.exists():
    urllib.request.urlretrieve(ANN_URL, ann_zip)

with zipfile.ZipFile(ann_zip) as z:
    z.extract("annotations/instances_val2017.json", ROOT)
GT_PATH = ROOT / "annotations" / "instances_val2017.json"
gt = json.load(open(GT_PATH))

imgs_by_id = {im["id"]: im for im in gt["images"]}
veh_img_ids = sorted({a["image_id"] for a in gt["annotations"]
                      if a["category_id"] in VEHICLE_COCO_IDS and not a["iscrowd"]})

random.seed(SEED)
subset_ids = sorted(random.sample(veh_img_ids, N_IMAGES))
subset = [imgs_by_id[i] for i in subset_ids]
subset_set = set(subset_ids)
print(f"{len(veh_img_ids)} candidate images; sampled {len(subset)}")
```

**Cell 4: Download images**

```python
# Cell 4: download sampled images with retry
import time

def fetch(url, dst, tries=3):
    for k in range(tries):
        try:
            urllib.request.urlretrieve(url, dst)
            return True
        except Exception as e:
            print("retry", dst.name, e); time.sleep(2 ** k)
    return False

failed = []
for im in subset:
    dst = IMG_DIR / im["file_name"]
    if not dst.exists() and not fetch(IMG_URL.format(im["file_name"]), dst):
        failed.append(im["file_name"])
print("on disk:", len(list(IMG_DIR.glob("*.jpg"))), "| failed:", failed)
assert not failed, "Download failed; stop and ask the user (see Step 6)."
```

**Cell 5: Sanity check on one image**

```python
# Cell 5: verify class IDs and box format on one image before the full run
from ultralytics import YOLO

model = YOLO(MODEL_NAME)
sample_path = str(IMG_DIR / subset[0]["file_name"])
r = model.predict(sample_path, imgsz=IMGSZ, conf=CONF, classes=list(YOLO_TO_COCO),
                  verbose=False)[0]
print("classes:", r.boxes.cls.int().tolist())
print("xyxy:   ", r.boxes.xyxy.tolist()[:2])
assert set(r.boxes.cls.int().tolist()) <= set(YOLO_TO_COCO), "Unexpected class IDs"
```

**Cell 6: Full inference and COCO-format export**

```python
# Cell 6: run detector on the full subset and convert to COCO results format
id_by_file = {im["file_name"]: im["id"] for im in subset}
paths = [str(IMG_DIR / im["file_name"]) for im in subset]

detections = []
for r in model.predict(source=paths, imgsz=IMGSZ, conf=CONF,
                       classes=list(YOLO_TO_COCO), stream=True,
                       verbose=False, batch=16, device=0):
    img_id = id_by_file[Path(r.path).name]
    xyxy = r.boxes.xyxy.tolist()
    scores = r.boxes.conf.tolist()
    cls = r.boxes.cls.int().tolist()
    for (x1, y1, x2, y2), s, c in zip(xyxy, scores, cls):
        detections.append({
            "image_id": img_id,
            "category_id": YOLO_TO_COCO[c],
            "bbox": [x1, y1, x2 - x1, y2 - y1],     # COCO expects x, y, w, h
            "score": s,
        })
print("detections:", len(detections))
```

**Cell 7: Evaluation**

```python
# Cell 7: pycocotools evaluation restricted to the subset and vehicle classes
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

coco_gt = COCO(str(GT_PATH))
coco_dt = coco_gt.loadRes(detections) if detections else None
assert coco_dt is not None, "No detections produced"

ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
ev.params.imgIds = subset_ids
ev.params.catIds = VEHICLE_COCO_IDS
ev.evaluate(); ev.accumulate(); ev.summarize()

map_50_95 = float(ev.stats[0])
map_50 = float(ev.stats[1])

# Per-class AP@0.5: precision shape is [IoU, recall, class, area, maxDets]
prec = ev.eval["precision"]
per_class = {}
for k, cid in enumerate(ev.params.catIds):
    p = prec[0, :, k, 0, 2]            # IoU=0.5, all areas, maxDets=100
    valid = p[p > -1]
    per_class[NAMES[cid]] = float(valid.mean()) if valid.size else float("nan")
```

**Cell 8: Count metrics and checks**

```python
# Cell 8: per-image count error and success criteria
gt_counts = {i: 0 for i in subset_ids}
for a in gt["annotations"]:
    if a["image_id"] in subset_set and a["category_id"] in VEHICLE_COCO_IDS and not a["iscrowd"]:
        gt_counts[a["image_id"]] += 1

pr_counts = {i: 0 for i in subset_ids}
for d in detections:
    pr_counts[d["image_id"]] += 1

errs = [abs(pr_counts[i] - gt_counts[i]) for i in subset_ids]
count_mae = sum(errs) / len(errs)
count_exact = sum(e == 0 for e in errs) / len(errs)

mean_ap50 = float(np.nanmean(list(per_class.values())))
checks = {
    "mean_AP50_>=_0.50": mean_ap50 >= 0.50,       # provisional threshold
    "count_MAE_<=_1.0": count_mae <= 1.0,         # provisional threshold
}
print(f"mAP50={map_50:.3f}  mAP50-95={map_50_95:.3f}  count MAE={count_mae:.2f}")
print("per class:", {k: round(v, 3) for k, v in per_class.items()})
print("checks:", checks)
```

**Cell 9: Save artifacts and report**

```python
# Cell 9: write artifacts to the Drive run folder
metrics = {
    "model": MODEL_NAME, "imgsz": IMGSZ, "conf": CONF, "seed": SEED,
    "n_images": len(subset), "mAP50_95": map_50_95, "mAP50": map_50,
    "AP50_per_class": per_class, "count_MAE": count_mae,
    "count_exact_match": count_exact, "checks": checks,
}
json.dump(detections, open(RUN_DIR / "detections.json", "w"))
json.dump(metrics, open(RUN_DIR / "metrics.json", "w"), indent=2)
with open(RUN_DIR / "per_image_counts.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["image_id", "file_name", "gt_vehicles", "pred_vehicles"])
    for im in subset:
        w.writerow([im["id"], im["file_name"], gt_counts[im["id"]], pr_counts[im["id"]]])

report = f"""# Run report: COCO vehicle benchmark
- Model: {MODEL_NAME} | imgsz {IMGSZ} | conf {CONF} | seed {SEED}
- Images: {len(subset)}
- mAP@0.5: {map_50:.3f} | mAP@[.5:.95]: {map_50_95:.3f}
- Per-class AP@0.5: {per_class}
- Count MAE: {count_mae:.2f} | exact match: {count_exact:.1%}
- Checks: {checks}
"""
open(RUN_DIR / "report.md", "w").write(report)
print("saved to", RUN_DIR)
```

**Agent control loop (pseudocode)**

```python
# Pseudocode: how the agent drives the notebook above
def run_task(spec):
    run = create_run_folder(spec)                  # FR-01, FR-06
    plan = planner.plan(spec)                      # ordered steps with checks
    for step in plan.steps:
        if step.needs_approval:                    # FR-12
            ask_user(step.approval_reason)
        for attempt in range(1, MAX_ATTEMPTS + 1): # FR-09, max 3
            code = coder.write_cell(step, history=run.errors_for(step))
            result = colab.execute(code, timeout=step.timeout)   # FR-03, FR-04
            log(run, step, attempt, code, result)                # FR-13
            verdict = checker.review(result, step.expected)      # deterministic first
            if verdict.ok:
                checkpoint(run, step)                            # FR-11
                break
        else:
            escalate(run, step)                                  # FR-10
            return run
    write_report(run)                                            # FR-15
    return run
```

`colab.execute` is the execution bridge from Open Questions. Until it exists, the agent produces each cell and a human runs it, which lets you test the code and the checker against real outputs.

---

# Assumptions

- Colab access is available through a standard Google account with a T4 GPU runtime. Colab does not offer a public agent API, so the execution bridge is an open decision, and automated use must be checked against Google's current terms before it is built.
- The Milestone 1 task uses only public data and pretrained weights. No training is required.
- The COCO image and annotation hosts remain available and do not rate-limit the 200 downloads. If they do, cache images in advance or use a mirror that allows the same license terms.
- Ultralytics and pycocotools APIs match the examples above. The version pins in the code are examples and must be verified during Phase 1.
- The pretrained YOLO11 model was trained on COCO train2017, not val2017, so val2017 is a held-out evaluation split for this purpose. The benchmark measures general vehicle detection, not performance on Indian dashcam footage.
- The success thresholds (mean AP@0.5 ≥ 0.50, count MAE ≤ 1.0) are provisional and will be calibrated after the first baseline run.
- Budgets, approval thresholds, and retry limits in this plan are proposed defaults and need confirmation from the project owner.
- The uploaded dashcam clip and the Roboflow plate dataset are not used in Milestone 1. Their licensing needs review before Milestone 4.