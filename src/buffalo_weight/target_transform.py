from __future__ import annotations

import numpy as np


TARGET_TRANSFORMS = frozenset({"cube_root", "identity", "log"})


def transform_target(values: np.ndarray, transform: str) -> np.ndarray:
    """Transform positive weights; for example, ``transform_target(weights, "log")``."""
    if transform == "identity":
        return values
    if transform == "log":
        return np.log(values)
    if transform == "cube_root":
        return np.cbrt(values)
    raise ValueError(f"target transform was {transform!r}; expected one of {sorted(TARGET_TRANSFORMS)}")


def inverse_target(values: np.ndarray, transform: str) -> np.ndarray:
    """Restore weights to kilograms; for example, ``inverse_target(values, "log")``."""
    if transform == "identity":
        return values
    if transform == "log":
        return np.exp(values)
    if transform == "cube_root":
        return values**3
    raise ValueError(f"target transform was {transform!r}; expected one of {sorted(TARGET_TRANSFORMS)}")


def transform_target_power(values: np.ndarray, power: float) -> np.ndarray:
    """Apply a positive allometric power; for example, ``transform_target_power(values, 0.25)``."""
    if 0.0 < power <= 1.0:
        return values**power
    raise ValueError(f"target power was {power}; expected 0 < target_power <= 1")


def inverse_target_power(values: np.ndarray, power: float) -> np.ndarray:
    """Restore power-transformed weights; for example, ``inverse_target_power(values, 0.25)``."""
    if 0.0 < power <= 1.0:
        return np.maximum(values, 0.0) ** (1.0 / power)
    raise ValueError(f"target power was {power}; expected 0 < target_power <= 1")
