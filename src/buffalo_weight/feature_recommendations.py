"""Provisional recommendations derived from OOF removal evidence."""

from __future__ import annotations

from dataclasses import dataclass

from buffalo_weight.feature_evaluation import FeatureEvidence
from buffalo_weight.feature_selection_rules import conservative_removal_recommendation


@dataclass(frozen=True)
class RemovalRecommendation:
    target: str
    random_forest_delta_mae_kg: float
    dense_delta_mae_kg: float
    recommendation: str


def build_removal_recommendations(
    evidence: list[FeatureEvidence], ordered_targets: tuple[str, ...]
) -> list[RemovalRecommendation]:
    """Summarize OOF deltas; for example, recommendations remain provisional."""
    oof_rows = [row for row in evidence if row.experiment == "removal" and row.scope == "oof"]
    indexed = {(row.target, row.baseline): row for row in oof_rows}
    return [_recommendation(target, indexed) for target in ordered_targets]


def provisional_feature_contract(
    candidate_features: tuple[str, ...], recommendations: list[RemovalRecommendation],
    report_sha256: str,
) -> dict[str, object]:
    """Create a non-promoted contract; for example, human fields always remain null."""
    return {
        "schema_version": 1,
        "status": "provisional",
        "candidate_features": list(candidate_features),
        "recommendations": [_recommendation_record(item) for item in recommendations],
        "standardization": "fit within each permitted training partition",
        "report_sha256": report_sha256,
        "selected_features": None,
        "human_decision": None,
    }


def _recommendation(
    target: str, indexed: dict[tuple[str, str], FeatureEvidence]
) -> RemovalRecommendation:
    deltas = {name: _required_delta(indexed.get((target, name)), target, name)
              for name in ("random_forest", "dense")}
    decision = conservative_removal_recommendation(deltas)
    return RemovalRecommendation(target, deltas["random_forest"], deltas["dense"], decision)


def _required_delta(row: FeatureEvidence | None, target: str, baseline: str) -> float:
    if row is None or row.delta_mae_kg is None:
        raise ValueError(
            f"OOF removal evidence was {row!r} for {target}/{baseline}; expected one numeric delta"
        )
    return row.delta_mae_kg


def _recommendation_record(item: RemovalRecommendation) -> dict[str, object]:
    return {
        "target": item.target,
        "random_forest_delta_mae_kg": round(item.random_forest_delta_mae_kg, 6),
        "dense_delta_mae_kg": round(item.dense_delta_mae_kg, 6),
        "recommendation": item.recommendation,
    }
