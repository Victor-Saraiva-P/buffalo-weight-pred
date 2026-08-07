"""Orchestration and validation of the descriptive slice for expanded diagnosis."""

from __future__ import annotations

from dataclasses import dataclass

from buffalo_weight.diagnostic_coverage import (
    DiagnosticCoverageSample,
    DiagnosticCoverageSummary,
    compute_sample_coverage,
)
from buffalo_weight.diagnostic_farm_comparison import (
    FarmComparisonRecord,
    compare_farms_under_approved_subsets,
)
from buffalo_weight.diagnostic_notable_cases import (
    NotableCaseRecord,
    identify_notable_cases,
)
from buffalo_weight.diagnostic_residual_correlations import (
    ResidualCorrelationRecord,
    compute_residual_correlations,
)
from buffalo_weight.diagnostic_stratified import (
    DiagnosticPrediction,
    StratifiedMetricRecord,
    compute_stratified_metrics,
)


PROHIBITED_TERMS = (
    "oracle_model",
    "modelo_oraculo",
    "p_value",
    "p-valor",
    "bootstrap",
    "stratum_ranking",
    "ranking_estrato",
    "causal_claim",
    "causalidade",
)


@dataclass(frozen=True)
class DescriptiveDiagnosticSlice:
    """Consolidated descriptive diagnostic slice containing all 5 characterizations."""

    coverage_summary: DiagnosticCoverageSummary
    stratified_metrics: list[StratifiedMetricRecord]
    farm_comparisons: list[FarmComparisonRecord]
    residual_correlations: list[ResidualCorrelationRecord]
    shared_hard_cases: list[NotableCaseRecord]
    divergent_cases: list[NotableCaseRecord]


def build_descriptive_diagnostic_slice(
    coverage_samples: list[DiagnosticCoverageSample],
    predictions: list[DiagnosticPrediction],
) -> DescriptiveDiagnosticSlice:
    """Build the descriptive diagnostic slice from coverage samples and OOF predictions.

    Example: ``build_descriptive_diagnostic_slice(samples, preds)`` returns ``DescriptiveDiagnosticSlice``.
    """
    coverage = compute_sample_coverage(coverage_samples)
    stratified = compute_stratified_metrics(predictions)
    farms = compare_farms_under_approved_subsets(predictions)
    correlations = compute_residual_correlations(predictions)
    shared_hard, divergent = identify_notable_cases(predictions)
    slice_obj = DescriptiveDiagnosticSlice(
        coverage_summary=coverage,
        stratified_metrics=stratified,
        farm_comparisons=farms,
        residual_correlations=correlations,
        shared_hard_cases=shared_hard,
        divergent_cases=divergent,
    )
    assert_no_prohibited_diagnostic_elements(slice_obj)
    return slice_obj


def assert_no_prohibited_diagnostic_elements(target: object) -> None:
    """Assert that no oracle model, p-values, bootstrap, or causal claims exist in the target.

    Example: ``assert_no_prohibited_diagnostic_elements(slice_obj)`` checks compliance.
    """
    text_repr = str(target).lower()
    for term in PROHIBITED_TERMS:
        if term.lower() in text_repr:
            raise ValueError(
                f"diagnostic contained prohibited term {term!r}; expected no oracle, p-value, bootstrap, or causal claims"
            )
