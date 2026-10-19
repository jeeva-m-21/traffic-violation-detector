import os
import json
import random
import time
import urllib.request
import zipfile
from pathlib import Path

print("=== Step 1: Initializing Workspace and Storage ===")
DRIVE_DIR = Path("/content/drive/MyDrive/traffic-violation-detector")
LOCAL_DIR = Path("/content/traffic-violation-detector")

# Determine active work directory
if Path("/content/drive/MyDrive").exists():
    WORKDIR = DRIVE_DIR
    print(f"Using Google Drive workspace: {WORKDIR}")
else:
    WORKDIR = LOCAL_DIR
    print(f"Google Drive not yet mounted. Using fast local workspace: {WORKDIR}")

DATA_DIR = WORKDIR / "data"
IMG_DIR = DATA_DIR / "val2017_subset"
ANN_DIR = DATA_DIR / "annotations"
RUN_DIR = WORKDIR / "runs" / "m1"

for p in [DATA_DIR, IMG_DIR, ANN_DIR, RUN_DIR]:
    p.mkdir(parents=True, exist_ok=True)

SEED = 42
N_IMAGES = 200
ANN_ZIP_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMG_URL_PATTERN = "http://images.cocodataset.org/val2017/{}"
VEHICLE_COCO_IDS = [3, 4, 6, 8]  # 3: car, 4: motorcycle, 6: bus, 8: truck

print("\n=== Step 2: Fetching Annotations ===")
ann_zip_path = DATA_DIR / "annotations_trainval2017.zip"
gt_path = ANN_DIR / "instances_val2017.json"

if not gt_path.exists():
    if not ann_zip_path.exists():
        print(f"Downloading COCO annotations zip from {ANN_ZIP_URL}...")
        t0 = time.time()
        urllib.request.urlretrieve(ANN_ZIP_URL, ann_zip_path)
        print(f"Downloaded {ann_zip_path.stat().st_size / 1024 / 1024:.2f} MB in {time.time() - t0:.2f}s")
    
    print("Extracting instances_val2017.json...")
    with zipfile.ZipFile(ann_zip_path, 'r') as z:
        z.extract("annotations/instances_val2017.json", DATA_DIR)
    extracted_path = DATA_DIR / "annotations" / "instances_val2017.json"
    if extracted_path.exists() and extracted_path != gt_path:
        os.replace(extracted_path, gt_path)
    print(f"Extracted instances_val2017.json ({gt_path.stat().st_size / 1024 / 1024:.2f} MB)")
else:
    print(f"Annotation file already exists at {gt_path}")

print("\n=== Step 3: Sampling 200 Vehicle Validation Images ===")
with open(gt_path, "r") as f:
    gt_data = json.load(f)

imgs_by_id = {im["id"]: im for im in gt_data["images"]}
vehicle_img_ids = sorted({
    ann["image_id"] for ann in gt_data["annotations"]
    if ann["category_id"] in VEHICLE_COCO_IDS and not ann.get("iscrowd", 0)
})

print(f"Found {len(vehicle_img_ids)} validation images containing vehicles.")
random.seed(SEED)
subset_ids = sorted(random.sample(vehicle_img_ids, N_IMAGES))
subset_images = [imgs_by_id[img_id] for img_id in subset_ids]

# Save sampled image metadata
subset_meta_path = RUN_DIR / "subset_images.json"
with open(subset_meta_path, "w") as f:
    json.dump({"seed": SEED, "count": len(subset_images), "images": subset_images}, f, indent=2)
print(f"Saved subset metadata ({len(subset_images)} images) to {subset_meta_path}")

print("\n=== Step 4: Fetching Sampled Images ===")
def fetch_url(url, dst, retries=3):
    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(url, dst)
            return True
        except Exception as e:
            time.sleep(1.5 ** attempt)
    return False

downloaded = 0
failed = []
t0 = time.time()
for idx, im in enumerate(subset_images, 1):
    dst = IMG_DIR / im["file_name"]
    if not dst.exists():
        url = IMG_URL_PATTERN.format(im["file_name"])
        if fetch_url(url, dst):
            downloaded += 1
        else:
            failed.append(im["file_name"])
    if idx % 50 == 0:
        print(f"Progress: {idx}/{len(subset_images)} images checked/downloaded...")

total_on_disk = len(list(IMG_DIR.glob("*.jpg")))
print(f"Finished image fetching in {time.time() - t0:.2f}s.")
print(f"Downloaded new: {downloaded}, Total on disk: {total_on_disk}/{len(subset_images)}, Failed: {len(failed)}")
assert len(failed) == 0, f"Failed images: {failed}"
print("=== Step 1-4 Complete: Dataset successfully fetched & cached! ===")
