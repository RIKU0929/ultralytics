# Normalized float32 `.npy` segmentation smoke test

Use a dataset with `images/{train,val,test}/*.npy`, matching `labels/{train,val,test}/*.txt`, `channels: 1`, and either `npy_padding_value` in `data.yaml` or `zero_gauss_value_npy` in `normalization.json`.

```python
from pathlib import Path

import numpy as np

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data import build_yolo_dataset

DATA = Path("/path/to/PRACTICE_YOLO_SEG_NPY/data.yaml")
PAD = 150.580388879593

# Dataloader inspection
cfg = get_cfg(overrides={"task": "segment", "data": str(DATA), "imgsz": 256, "cache": False, "npy_padding_value": PAD})
data = {
    "path": str(DATA.parent),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "channels": 1,
    "names": {0: "positive", 1: "negative"},
    "npy_padding_value": PAD,
}
for split in ("train", "val", "test"):
    files = sorted((DATA.parent / "images" / split).glob("*.npy"))
    print(split, "count", len(files))
    a = np.load(files[0], allow_pickle=False)
    print("raw", a.shape, a.dtype, float(a.min()), float(a.max()), np.isfinite(a).all())

dataset = build_yolo_dataset(cfg, str(DATA.parent / "images/train"), batch=2, data=data)
sample = dataset[0]
print("padding", dataset.npy_padding_value)
print("sample tensor", sample["img"].shape, sample["img"].dtype, sample["img"].min().item(), sample["img"].max().item())
print("after /255", (sample["img"].float() / 255).min().item(), (sample["img"].float() / 255).max().item())

# 1 epoch training
model = YOLO("yolo11n-seg.pt")
model.train(
    data=str(DATA),
    epochs=1,
    imgsz=256,
    batch=2,
    cache=False,
    npy_padding_value=PAD,
    mosaic=0.0,
    mixup=0.0,
    cutmix=0.0,
    copy_paste=0.0,
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
    fliplr=0.0,
    flipud=0.0,
    degrees=0.0,
    translate=0.0,
    scale=0.0,
    shear=0.0,
    perspective=0.0,
)

# Test evaluation
best = YOLO("runs/segment/train/weights/best.pt")
best.val(data=str(DATA), split="test", npy_padding_value=PAD)

# Prediction
for source in [
    "/path/to/sample.npy",
    ["/path/to/a.npy", "/path/to/b.npy"],
    "/path/to/PRACTICE_YOLO_SEG_NPY/images/test",
]:
    for r in best.predict(source=source, npy_padding_value=PAD):
        print(
            r.path,
            r.boxes,
            None if r.boxes is None else r.boxes.cls,
            None if r.boxes is None else r.boxes.conf,
            r.masks,
        )
```
