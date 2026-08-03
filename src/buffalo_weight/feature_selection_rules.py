"""Pure decision rules for comparative feature evidence."""

from __future__ import annotations

import hashlib

PRACTICAL_MARGIN_KG = 1.0
BASELINE_NAMES = ("random_forest", "dense")


def classify_mae_delta(delta_mae_kg: float) -> str:
    """Classify removal-minus-full MAE; for example, ``classify_mae_delta(-1.2)`` improves."""
    if delta_mae_kg < -PRACTICAL_MARGIN_KG:
        return "improvement"
    if delta_mae_kg > PRACTICAL_MARGIN_KG:
        return "harm"
    return "neutral"


def conservative_removal_recommendation(deltas_by_baseline: dict[str, float]) -> str:
    """Apply the human-review gate; for example, one improvement can recommend removal."""
    _validate_baseline_deltas(deltas_by_baseline)
    effects = [classify_mae_delta(deltas_by_baseline[name]) for name in BASELINE_NAMES]
    if "harm" in effects:
        return "retain_harm_veto"
    if "improvement" in effects:
        return "recommend_removal"
    return "retain_double_neutral"


def permutation_seed(split_seed: int, fold: int, feature: str, repetition: int) -> int:
    """Derive a stable seed; for example, ``permutation_seed(42, 1, 'area', 0)``."""
    identity = f"{split_seed}:{fold}:{feature}:{repetition}".encode()
    return int.from_bytes(hashlib.sha256(identity).digest()[:4], "big")


def _validate_baseline_deltas(deltas_by_baseline: dict[str, float]) -> None:
    actual = tuple(sorted(deltas_by_baseline))
    expected = tuple(sorted(BASELINE_NAMES))
    if actual != expected:
        raise ValueError(f"baseline names were {actual!r}; expected exactly {expected!r}")
