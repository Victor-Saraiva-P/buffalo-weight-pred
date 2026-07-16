from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import rotate

from buffalo_weight.cnn_mask import find_mask_path, load_mask, resize_mask


def principal_axis_angle(mask: np.ndarray) -> float:
    """Return the silhouette major-axis angle; for example, ``principal_axis_angle(mask)``."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 2:
        return 0.0
    coordinates = np.column_stack((xs - xs.mean(), ys - ys.mean()))
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(coordinates, rowvar=False))
    major_vector = eigenvectors[:, int(np.argmax(eigenvalues))]
    angle = float(np.degrees(np.arctan2(major_vector[1], major_vector[0])))
    return (angle + 90.0) % 180.0 - 90.0


def canonicalize_mask(mask: np.ndarray, image_size: int, resize_mode: str = "letterbox") -> np.ndarray:
    """Align and crop a silhouette; for example, ``canonicalize_mask(mask, 64)``."""
    rotated = rotate(mask.astype(np.uint8), principal_axis_angle(mask), reshape=True, order=0) > 0
    ys, xs = np.nonzero(rotated)
    if not len(xs):
        return np.zeros((image_size, image_size), dtype=np.float32)
    cropped = rotated[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    image = Image.fromarray(cropped.astype(np.uint8) * 255)
    return (np.asarray(resize_mask(image, image_size, resize_mode)) > 0).astype(np.float32)


def load_canonical_masks(
    masks_dir: Path, rows: list[dict[str, str]], image_size: int, resize_mode: str = "letterbox"
) -> np.ndarray:
    """Load aligned silhouettes; for example, ``load_canonical_masks(path, rows, 64)``."""
    masks = []
    for row in rows:
        path = find_mask_path(masks_dir, row["file_name"])
        original = load_mask(path, max(image_size * 4, 256), "letterbox")
        masks.append(canonicalize_mask(original, image_size, resize_mode))
    return np.stack(masks)
