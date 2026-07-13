# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path

import numpy as np
import torch

from ultralytics.data.build import build_dataloader
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import DEFAULT_CFG, YAML


def test_float32_grayscale_npy_dataset_and_dataloader_shapes(tmp_path: Path):
    """Test float32 2D .npy segmentation images stay float32 and collate as single-channel tensors."""
    h, w = 16, 16
    images_dir = tmp_path / "images" / "train"
    labels_dir = tmp_path / "labels" / "train"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    for i in range(2):
        np.save(images_dir / f"im{i}.npy", np.full((h, w), i + 0.5, dtype=np.float32), allow_pickle=False)
        (labels_dir / f"im{i}.txt").write_text("0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n", encoding="utf-8")

    data_yaml = tmp_path / "data.yaml"
    YAML.save(
        data={"path": tmp_path, "train": "images/train", "val": "images/train", "channels": 1, "names": {0: "object"}},
        file=data_yaml,
    )
    data = YAML.load(data_yaml)

    dataset = YOLODataset(img_path=images_dir, imgsz=h, augment=False, hyp=DEFAULT_CFG, task="segment", data=data)
    sample = dataset[0]["img"]
    assert sample.shape == (1, h, w)
    assert sample.dtype == torch.float32
    assert sample[0, 0, 0] == 0.5

    dataloader = build_dataloader(dataset, batch=2, workers=0, shuffle=False)
    try:
        batch = next(iter(dataloader))
    finally:
        dataloader.close()
    assert batch["img"].shape == (2, 1, h, w)
    assert batch["img"].dtype == torch.float32
    assert torch.equal(batch["img"][:, 0, 0, 0], torch.tensor([0.5, 1.5]))
