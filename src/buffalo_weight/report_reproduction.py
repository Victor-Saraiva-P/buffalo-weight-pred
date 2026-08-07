"""Orchestrates full report reproduction across stages and gates."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TextIO

from buffalo_weight.baseline_comparison_provenance import BaselineComparisonProvenance
from buffalo_weight.baseline_comparison_stage import (
    BaselineComparisonUpstreamDependencies,
    run_baseline_comparison_stage,
)
from buffalo_weight.baseline_provenance import BaselineProvenance
from buffalo_weight.baseline_stage import run_random_forest_baseline_stage
from buffalo_weight.compact_cnn_provenance import CompactCnnProvenance
from buffalo_weight.compact_cnn_stage import run_compact_cnn_stage
from buffalo_weight.compact_cnn_types import CompactCnnTrainingAdapter
from buffalo_weight.dense_baseline_stage import DenseBaselineDependencies, run_dense_baseline_stage
from buffalo_weight.diagnostic_descriptive_stage import run_diagnostic_descriptive_stage
from buffalo_weight.diagnostic_learning_stage import run_diagnostic_learning_stage
from buffalo_weight.diagnostic_sensitivity_stage import run_diagnostic_sensitivity_stage
from buffalo_weight.feature_evaluation import FeatureBaseline
from buffalo_weight.feature_selection_provenance import FeatureSelectionProvenance
from buffalo_weight.feature_selection_stage import (
    FeatureEvidenceRunner,
    run_feature_selection_stage,
)
from buffalo_weight.report_inputs import run_inputs_stage
from buffalo_weight.report_provenance import ReportProvenance
from buffalo_weight.report_reproduction_nodes import (
    ReproductionNode,
    plan_reproduction_nodes,
)
from buffalo_weight.reproduction_config import ReportContract
from buffalo_weight.resnet_baseline_provenance import ResNetBaselineProvenance
from buffalo_weight.resnet_baseline_stage import (
    ResNetBaselineRunner,
    run_resnet_baseline_stage,
)
from buffalo_weight.snapshot_io import SnapshotPublisher
from buffalo_weight.tuning_provenance import TuningProvenance
from buffalo_weight.tuning_stage import run_tuning_stage


@dataclass(frozen=True)
class ReproductionDependencies:
    """Inject execution boundaries for full report reproduction."""

    snapshot_publisher: SnapshotPublisher | None = None
    report_provenance: ReportProvenance | None = None
    feature_evidence_runner: FeatureEvidenceRunner | None = None
    feature_selection_provenance: FeatureSelectionProvenance | None = None
    feature_confirmation_environment: object | None = None
    random_forest_baseline: FeatureBaseline | None = None
    baseline_provenance: BaselineProvenance | None = None
    dense_baseline_dependencies: DenseBaselineDependencies | None = None
    compact_cnn_adapter: CompactCnnTrainingAdapter | None = None
    compact_cnn_provenance: CompactCnnProvenance | None = None
    resnet_baseline_runner: ResNetBaselineRunner | None = None
    resnet_baseline_provenance: ResNetBaselineProvenance | None = None
    baseline_comparison_provenance: BaselineComparisonProvenance | None = None
    tuning_provenance: TuningProvenance | None = None


def run_report_reproduction(
    contract: ReportContract,
    dry_run: bool = False,
    deps: ReproductionDependencies | None = None,
    stdout: TextIO = sys.stdout,
) -> int:
    """Orchestrate the 11-node report reproduction pipeline.

    Example: ``run_report_reproduction(contract, dry_run=False)`` returns 0 or 1.
    """
    dependencies = deps or ReproductionDependencies()
    nodes = plan_reproduction_nodes(contract, dependencies)
    _print_reproduction_plan(nodes, stdout)

    if dry_run:
        return _dry_run_exit_code(nodes, stdout)
    return _execute_reproduction_nodes(contract, nodes, dependencies, stdout)


def _print_reproduction_plan(nodes: list[ReproductionNode], stdout: TextIO) -> None:
    stdout.write("reproduction_plan:\n")
    for node in nodes:
        stdout.write(f"  [{node.node_type}] {node.name}: {node.action} ({node.reason})\n")


def _dry_run_exit_code(nodes: list[ReproductionNode], stdout: TextIO) -> int:
    blocked_count = sum(1 for node in nodes if node.action == "blocked")
    if blocked_count > 0:
        stdout.write(f"reproduction: blocked by {blocked_count} node(s)\n")
        return 1
    stdout.write("reproduction: plan valid\n")
    return 0


def _execute_reproduction_nodes(
    contract: ReportContract,
    nodes: list[ReproductionNode],
    deps: ReproductionDependencies,
    stdout: TextIO,
) -> int:
    for node in nodes:
        code = _execute_single_node(contract, node, deps, stdout)
        if code != 0:
            return code
    stdout.write("reproduction: complete\n")
    return 0


def _execute_single_node(
    contract: ReportContract,
    node: ReproductionNode,
    deps: ReproductionDependencies,
    stdout: TextIO,
) -> int:
    if node.action == "reusable":
        stdout.write(f"{node.name}: reusable\n")
        return 0
    if node.action == "released":
        stdout.write(f"{node.name}: released\n")
        return 0
    if node.action == "blocked":
        stdout.write(f"blocked: {node.name} -> {node.reason}\n")
        _print_gate_action_instruction(node.name, stdout)
        return 1
    return _dispatch_stage_rebuild(contract, node.name, deps, stdout)


def _dispatch_stage_rebuild(
    contract: ReportContract,
    stage_name: str,
    deps: ReproductionDependencies,
    stdout: TextIO,
) -> int:
    if stage_name == "inputs":
        res: int | str = run_inputs_stage(
            contract, False, deps.snapshot_publisher, deps.report_provenance
        )
    elif stage_name == "feature-selection":
        res = run_feature_selection_stage(
            contract,
            False,
            deps.feature_evidence_runner,
            deps.snapshot_publisher,
            deps.report_provenance,
            deps.feature_selection_provenance,
        )
    elif stage_name == "baselines":
        res = _rebuild_baselines_stage(contract, deps)
    elif stage_name == "compare-baselines":
        res = _rebuild_compare_baselines_stage(contract, deps)
    else:
        return _dispatch_diagnostic_stage_rebuild(contract, stage_name, deps, stdout)
    return _check_rebuild_result(stage_name, res, stdout)


def _dispatch_diagnostic_stage_rebuild(
    contract: ReportContract,
    stage_name: str,
    deps: ReproductionDependencies,
    stdout: TextIO,
) -> int:
    if stage_name == "tuning":
        res: int | str = _rebuild_tuning_stage(contract, deps)
    elif stage_name == "diagnostics-descriptive":
        res = run_diagnostic_descriptive_stage(contract, False)
    elif stage_name == "diagnostics-learning":
        res = _rebuild_learning_curves_stage(contract, deps)
    elif stage_name == "diagnostics-sensitivity":
        res = run_diagnostic_sensitivity_stage(contract, False, deps.random_forest_baseline)
    else:
        return 1
    return _check_rebuild_result(stage_name, res, stdout)


def _check_rebuild_result(stage_name: str, result: int | str, stdout: TextIO) -> int:
    if result in (0, "rebuilt", "reusable"):
        stdout.write(f"{stage_name}: rebuilt\n")
        return 0
    return 1


def _rebuild_baselines_stage(contract: ReportContract, deps: ReproductionDependencies) -> int:
    code = run_random_forest_baseline_stage(
        contract, False, deps.random_forest_baseline, deps.baseline_provenance
    )
    if code != 0:
        return code
    code = run_dense_baseline_stage(contract, False, deps.dense_baseline_dependencies)
    if code != 0:
        return code
    code = run_compact_cnn_stage(
        contract, False, deps.compact_cnn_adapter, deps.compact_cnn_provenance
    )
    if code != 0:
        return code
    return run_resnet_baseline_stage(
        contract, False, deps.resnet_baseline_runner, deps.resnet_baseline_provenance
    )


def _rebuild_compare_baselines_stage(
    contract: ReportContract, deps: ReproductionDependencies
) -> int:
    upstream = BaselineComparisonUpstreamDependencies(
        baseline_provenance=deps.baseline_provenance,
        dense_baseline_dependencies=deps.dense_baseline_dependencies,
        compact_cnn_provenance=deps.compact_cnn_provenance,
        resnet_baseline_provenance=deps.resnet_baseline_provenance,
    )
    return run_baseline_comparison_stage(
        contract, False, upstream, deps.baseline_comparison_provenance
    )


def _rebuild_tuning_stage(contract: ReportContract, deps: ReproductionDependencies) -> int:
    dense_adapter = (
        deps.dense_baseline_dependencies.adapter
        if deps.dense_baseline_dependencies is not None
        else None
    )
    return run_tuning_stage(
        contract,
        False,
        deps.report_provenance,
        deps.random_forest_baseline,
        deps.baseline_provenance,
        dense_adapter,
        deps.compact_cnn_adapter,
        deps.resnet_baseline_runner,
        deps.tuning_provenance,
    )


def _rebuild_learning_curves_stage(
    contract: ReportContract, deps: ReproductionDependencies
) -> int:
    return run_diagnostic_learning_stage(
        contract,
        False,
        deps.report_provenance,
        deps.random_forest_baseline,
        deps.baseline_provenance,
        deps.dense_baseline_dependencies,
        deps.compact_cnn_adapter,
        deps.compact_cnn_provenance,
        deps.resnet_baseline_runner,
        deps.resnet_baseline_provenance,
    )


def _print_gate_action_instruction(gate_name: str, stdout: TextIO) -> None:
    stdout.write("Action required:\n")
    if gate_name == "confirm-features":
        stdout.write(
            "  Run `python main.py confirm-features --contract <contract.json> --report <report.md>`\n"
        )
    elif gate_name == "confirm-approach":
        stdout.write(
            "  Run `python main.py confirm-approach --contract <contract.json> --report <report.md>`\n"
        )
    elif gate_name == "confirm-diagnostics":
        stdout.write(
            "  Run `python main.py confirm-diagnostics --contract <contract.json> --report <report.md>`\n"
        )
    else:
        stdout.write(f"  Confirm human decision for gate '{gate_name}'\n")
