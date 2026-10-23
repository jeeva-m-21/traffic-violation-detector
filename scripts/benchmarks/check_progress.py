import os
from pathlib import Path

img_dir = Path("/content/traffic-violation-detector/data/val2017_subset")
if img_dir.exists():
    imgs = list(img_dir.glob("*.jpg"))
    print(f"Images downloaded on disk: {len(imgs)} / 200")
else:
    print("img_dir does not exist yet")

ann_path = Path("/content/traffic-violation-detector/data/annotations/instances_val2017.json")
print("Annotation file exists:", ann_path.exists())
if ann_path.exists():
    print("Annotation file size:", ann_path.stat().st_size / 1024 / 1024, "MB")
