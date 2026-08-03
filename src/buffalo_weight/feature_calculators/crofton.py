from __future__ import annotations

import math

import numpy as np


def crofton_perimeter(mask: np.ndarray) -> float:
    """Estimate four-direction perimeter; for example, a 2x2 block is 6.47375."""
    padded = np.pad(mask.astype(bool), 1, constant_values=False)
    horizontal = np.count_nonzero(padded[:, 1:] != padded[:, :-1])
    vertical = np.count_nonzero(padded[1:, :] != padded[:-1, :])
    diagonal_down = np.count_nonzero(padded[1:, 1:] != padded[:-1, :-1])
    diagonal_up = np.count_nonzero(padded[1:, :-1] != padded[:-1, 1:])
    axial = horizontal + vertical
    diagonal = (diagonal_down + diagonal_up) / math.sqrt(2)
    return float(math.pi * (axial + diagonal) / 8)
