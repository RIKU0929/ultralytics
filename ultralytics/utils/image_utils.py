# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Utility helpers for working with specialized image formats."""

from __future__ import annotations

from pathlib import Path

import numpy as np

FITS_EXTENSIONS = {"fts"}
FITS_DTYPE_MAX = 65535.0


def is_fits_file(path: str | Path) -> bool:
    """Return True if a path points to a FITS/FITS-derived image."""
    suffix = Path(path).suffix.lower().lstrip(".")
    return suffix in FITS_EXTENSIONS


def _import_astropy():
    try:
        from astropy.io import fits
    except ImportError as e:
        raise ImportError(
            "Reading FITS images requires the 'astropy' package. Install it with 'pip install astropy'."
        ) from e
    return fits


def load_fits_image(path: str | Path, channels: int = 3) -> np.ndarray:
    """Load a FITS image, apply percentile clipping, and return a contiguous uint8 array."""
    fits = _import_astropy()
    data = fits.getdata(path)
    if data is None:
        raise ValueError(f"Unable to read FITS data from {path}.")

    # --- 【修正点】ここで上下反転を加える ---
    # FITSの標準（左下原点）を 画像処理の標準（左上原点）に合わせる
    data = np.flipud(data)    

    # Convert and sanitize
    arr = np.asarray(data, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # Percentile clipping for robust normalization
    p_low, p_high = np.percentile(arr, [1.0, 99.9])
    if p_high <= p_low:
        arr = np.zeros_like(arr)
    else:
        arr = np.clip(arr, p_low, p_high)
        arr = (arr - p_low) / (p_high - p_low)
        arr *= 255.0  # scale to standard uint8 intensity range for downstream augmentations

    # Ensure channel dimension exists
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3:
        raise ValueError(f"FITS image at {path} must be 2D or 3D, but has shape {arr.shape}.")

    # Tile or trim channels to the requested count
    target = max(1, channels)
    current = arr.shape[2]
    if current == target:
        return np.ascontiguousarray(arr.astype(np.uint8))
    if current == 1:
        arr = np.tile(arr, (1, 1, target))
    elif target == 1:
        arr = arr[..., :1]
    elif current > target:
        arr = arr[..., :target]
    else:
        reps = int(np.ceil(target / current))
        arr = np.concatenate([arr] * reps, axis=2)
        arr = arr[..., :target]
    return np.ascontiguousarray(arr.astype(np.uint8))


__all__ = ("FITS_DTYPE_MAX", "FITS_EXTENSIONS", "is_fits_file", "load_fits_image")
