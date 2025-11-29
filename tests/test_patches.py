"""Tests for monkey-patched utilities."""

import sys
from types import ModuleType

import cv2
import numpy as np

from ultralytics.utils.patches import imread


def test_imread_supports_fits(monkeypatch, tmp_path):
    """Ensure FITS decoding falls back to astropy and outputs scaled float images."""

    class DummyHDU:
        def __init__(self, data, header):
            self.data = data
            self.header = header

    class DummyHDUList:
        def __init__(self, hdu):
            self.hdu = hdu

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __getitem__(self, idx):
            if idx == 0:
                return self.hdu
            raise IndexError

    data = np.array([[1, 2], [3, 4]], dtype=np.uint16)
    header = {"BSCALE": 2, "BZERO": 1}

    def mock_open(*args, **kwargs):
        return DummyHDUList(DummyHDU(data, header))

    astropy = ModuleType("astropy")
    io_module = ModuleType("astropy.io")
    fits_module = ModuleType("astropy.io.fits")
    fits_module.open = mock_open
    io_module.fits = fits_module
    astropy.io = io_module

    monkeypatch.setitem(sys.modules, "astropy", astropy)
    monkeypatch.setitem(sys.modules, "astropy.io", io_module)
    monkeypatch.setitem(sys.modules, "astropy.io.fits", fits_module)

    fits_path = tmp_path / "image.fits"
    fits_path.write_bytes(b"dummy fits content")

    image = imread(str(fits_path))

    assert image.shape == (2, 2, 3)
    assert image.dtype == np.float32
    expected = data.astype(np.float32) * header["BSCALE"] + header["BZERO"]
    assert np.array_equal(image[..., 0], expected)


def test_imread_png(tmp_path):
    """Verify standard image decoding still works for PNG files."""

    array = np.zeros((4, 4, 3), dtype=np.uint8)
    path = tmp_path / "img.png"
    cv2.imwrite(str(path), array)

    image = imread(str(path))

    assert image.shape == array.shape
    assert image.dtype == array.dtype
