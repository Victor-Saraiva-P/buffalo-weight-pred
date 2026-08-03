"""Closed vocabulary and ordering shared by feature-selection artifacts."""

from __future__ import annotations

from typing import Literal

FeatureExperiment = Literal["isolated", "removal", "permutation"]
FeatureBaselineName = Literal["random_forest", "dense"]
EvidenceScope = Literal["fold", "oof"]
EvidenceEffect = Literal["improvement", "neutral", "harm"]
FEATURE_EXPERIMENTS: tuple[FeatureExperiment, ...] = ("isolated", "removal", "permutation")
FEATURE_BASELINES: tuple[FeatureBaselineName, ...] = ("random_forest", "dense")
EVIDENCE_SCOPES: tuple[EvidenceScope, ...] = ("fold", "oof")
EVIDENCE_EFFECTS: tuple[EvidenceEffect, ...] = ("improvement", "neutral", "harm")


def canonical_evidence_sort_key(
    experiment: str, baseline: str, target: str, scope: str,
    fold: int | None, repetition: int | None,
) -> tuple[object, ...]:
    """Order evidence; for example, fold rows precede the matching grouped OOF row."""
    scope_rank = 0 if scope == "fold" else 1
    key = (experiment, baseline, target, scope_rank, fold or 0, repetition or 0)
    return key
