# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ultralytics.cfg import get_cfg
from ultralytics.data import build_yolo_dataset, load_inference_source
from ultralytics.data.augment import LetterBox
from ultralytics.data.utils import load_npy_image
from ultralytics.models.yolo.detect.train import DetectionTrainer


def _write_seg_dataset(root: Path, values: np.ndarray) -> Path:
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        np.save(root / "images" / split / "sample.npy", values.astype(np.float32), allow_pickle=False)
        (root / "labels" / split / "sample.txt").write_text("0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n")
    data = root / "data.yaml"
    data.write_text(
        f"path: {root}\ntrain: images/train\nval: images/val\ntest: images/test\nchannels: 1\n"
        "npy_padding_value: 150.580388879593\nnames:\n  0: item\n"
    )
    return data


def test_load_npy_image_preserves_float32_and_rejects_invalid(tmp_path):
    values = np.array([[-10.5, 0.25], [127.75, 270.5]], dtype=np.float32)
    f = tmp_path / "sample.npy"
    np.save(f, values, allow_pickle=False)
    im = load_npy_image(f)
    assert im.shape == (2, 2, 1)
    assert im.dtype == np.float32
    assert im[0, 0, 0] == pytest.approx(-10.5)

    for bad, msg in [(np.array([[np.nan]], np.float32), "NaN or Inf"), (values.astype(np.float64), "float32")]:
        np.save(f, bad, allow_pickle=False)
        with pytest.raises(ValueError, match=msg):
            load_npy_image(f)
    np.save(f, np.zeros((1, 1, 1), dtype=np.float32), allow_pickle=False)
    with pytest.raises(ValueError, match="2D"):
        load_npy_image(f)


def test_npy_letterbox_padding_preserves_fraction_and_channel():
    im = np.ones((2, 4, 1), dtype=np.float32)
    out = LetterBox(new_shape=(4, 4), padding_value=150.580388879593)(image=im)
    assert out.shape == (4, 4, 1)
    assert out.dtype == np.float32
    assert out[0, 0, 0] == pytest.approx(150.580388879593)

    rgb = np.ones((2, 4, 3), dtype=np.uint8)
    out_rgb = LetterBox(new_shape=(4, 4))(image=rgb)
    assert out_rgb[0, 0, 0] == 114


def test_npy_yolo_dataset_batch_and_single_div255(tmp_path):
    values = np.array([[-10.5, 0.25, 127.75], [255.0, 270.5, 1.5]], dtype=np.float32)
    data = _write_seg_dataset(tmp_path, values)
    cfg = get_cfg(overrides={"task": "segment", "imgsz": 4, "cache": False, "data": str(data)})
    dataset = build_yolo_dataset(
        cfg,
        str(tmp_path / "images" / "train"),
        batch=1,
        data={"path": str(tmp_path), "channels": 1, "names": {0: "item"}, "npy_padding_value": 150.580388879593},
    )
    sample = dataset[0]
    assert sample["img"].shape[0] == 1
    assert sample["img"].dtype == torch.float32
    batch = {"img": sample["img"].unsqueeze(0)}
    trainer = SimpleNamespace(device=torch.device("cpu"), args=SimpleNamespace(multi_scale=0.0))
    out = DetectionTrainer.preprocess_batch(trainer, batch)["img"]
    assert out.shape[1] == 1
    assert out.dtype == torch.float32
    assert out.min() < 0
    assert out.max() > 1
    assert torch.isclose(out, torch.tensor(270.5 / 255)).any()
    assert torch.isclose(out, torch.tensor(-10.5 / 255)).any()


def test_npy_inference_sources(tmp_path):
    values = np.ones((3, 5), dtype=np.float32)
    paths = []
    for name in ("a.npy", "b.npy"):
        f = tmp_path / name
        np.save(f, values, allow_pickle=False)
        paths.append(str(f))
    assert len(load_inference_source(paths, channels=1)) == 2
    batch_paths, images, _ = next(iter(load_inference_source(str(tmp_path), channels=1, batch=2)))
    assert batch_paths == paths
    assert images[0].shape == (3, 5, 1)
    assert images[0].dtype == np.float32
