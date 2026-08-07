"""Graph node definitions and evaluation logic for full report reproduction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TextIO

from buffalo_weight.approach_confirmation import approach_gate_status
from buffalo_weight.baseline_comparison_manifest import baseline_comparison_status
from buffalo_weight.baseline_manifest import baseline_configuration_status
from buffalo_weight.baseline_provenance import SystemBaselineProvenance
from buffalo_weight.baseline_types import BASELINE_DEFINITIONS
from buffalo_weight.compact_cnn_manifest import compact_cnn_status
from buffalo_weight.compact_cnn_provenance import SystemCompactCnnProvenance
from buffalo_weight.compact_cnn_types import COMPACT_CNN_RECIPE
from buffalo_weight.dense_baseline_manifest import dense_baseline_status
from buffalo_weight.dense_baseline_provenance import SystemDenseBaselineProvenance
from buffalo_weight.diagnostic_confirmation import diagnostics_gate_status
from buffalo_weight.feature_confirmation import baselines_gate_status
from buffalo_weight.feature_confirmation_manifest import validate_frozen_feature_contract
from buffalo_weight.feature_selection_manifest import feature_selection_status
from buffalo_weight.feature_selection_provenance import SystemFeatureSelectionProvenance
from buffalo_weight.inputs_manifest import stage_status as inputs_stage_status
from buffalo_weight.report_provenance import SystemReportProvenance
from buffalo_weight.resnet_baseline_provenance import SystemResNetBaselineProvenance
from buffalo_weight.resnet_baseline_stage import plan_resnet_baseline_stage
from buffalo_weight.tuning_manifest import tuning_stage_status
from buffalo_weight.tuning_provenance import SystemTuningProvenance

if TYPE_CHECKING:
    from buffalo_weight.report_reproduction import ReproductionDependencies
    from buffalo_weight.reproduction_config import ReportContract


NodeType = Literal["stage", "gate"]
NodeAction = Literal["reusable", "rebuild", "released", "blocked"]


@dataclass(frozen=True)
class ReproductionNode:
    """Represents a node (stage or gate) in the report reproduction execution plan."""

    name: str
    node_type: NodeType
    action: NodeAction
    reason: str


def plan_reproduction_nodes(
    contract: ReportContract, deps: ReproductionDependencies
) -> list[ReproductionNode]:
    """Build the ordered list of 11 reproduction nodes.

    Example: ``plan_reproduction_nodes(contract, deps)`` returns nodes graph.
    """
    inputs_node = _evaluate_inputs_node(contract, deps)
    feature_sel_node = _evaluate_feature_selection_node(
        contract, deps, inputs_node.action == "rebuild"
    )
    feature_gate_node = _evaluate_feature_gate_node(contract)

    nodes = [inputs_node, feature_sel_node, feature_gate_node]
    downstream = _evaluate_downstream_nodes(
        contract, deps, feature_gate_node.action == "blocked"
    )
    nodes.extend(downstream)
    return nodes


def _evaluate_downstream_nodes(
    contract: ReportContract, deps: ReproductionDependencies, gate1_blocked: bool
) -> list[ReproductionNode]:
    baselines = _evaluate_baselines_node(contract, deps, gate1_blocked)
    comp_base = _evaluate_compare_baselines_node(
        contract, deps, gate1_blocked, baselines.action == "rebuild"
    )
    app_gate = _evaluate_approach_gate_node(contract)
    gate2_blocked = gate1_blocked or (app_gate.action == "blocked")
    up_rebuild = baselines.action == "rebuild" or comp_base.action == "rebuild"

    tuning = _evaluate_tuning_node(contract, deps, gate2_blocked, up_rebuild)
    desc = _evaluate_diag_descriptive_node(contract, gate2_blocked, up_rebuild)
    learn = _evaluate_diag_learning_node(contract, deps, gate2_blocked, up_rebuild)
    sens = _evaluate_diag_sensitivity_node(contract, deps, gate2_blocked, up_rebuild)
    diag_gate = _evaluate_diag_gate_node(contract)

    return [baselines, comp_base, app_gate, tuning, desc, learn, sens, diag_gate]


def _evaluate_inputs_node(
    contract: ReportContract, deps: ReproductionDependencies
) -> ReproductionNode:
    report_prov = deps.report_provenance or SystemReportProvenance()
    try:
        status = inputs_stage_status(contract, report_prov)
    except (OSError, ValueError):
        status = "obsolete"
    if status == "reusable":
        return ReproductionNode("inputs", "stage", "reusable", "inputs snapshot intact")
    return ReproductionNode("inputs", "stage", "rebuild", "inputs snapshot absent or obsolete")


def _evaluate_feature_selection_node(
    contract: ReportContract, deps: ReproductionDependencies, inputs_rebuilding: bool
) -> ReproductionNode:
    if inputs_rebuilding:
        return ReproductionNode(
            "feature-selection", "stage", "rebuild", "upstream inputs stage will be rebuilt"
        )
    sel_prov = deps.feature_selection_provenance or SystemFeatureSelectionProvenance()
    try:
        status = feature_selection_status(contract, sel_prov)
    except (OSError, ValueError):
        status = "obsolete"
    if status == "reusable":
        return ReproductionNode(
            "feature-selection", "stage", "reusable", "feature selection evidence intact"
        )
    return ReproductionNode(
        "feature-selection", "stage", "rebuild", "feature selection evidence absent or obsolete"
    )


def _evaluate_feature_gate_node(contract: ReportContract) -> ReproductionNode:
    status = baselines_gate_status(contract)
    if status == "released":
        return ReproductionNode(
            "confirm-features", "gate", "released", "confirmed feature selection package intact"
        )
    return ReproductionNode("confirm-features", "gate", "blocked", status)


def _evaluate_baselines_node(
    contract: ReportContract, deps: ReproductionDependencies, gate1_blocked: bool
) -> ReproductionNode:
    if gate1_blocked:
        return ReproductionNode(
            "baselines", "stage", "blocked", "blocked by unconfirmed feature selection gate"
        )
    if _are_all_baselines_reusable(contract, deps):
        return ReproductionNode("baselines", "stage", "reusable", "all 4 baseline models intact")
    return ReproductionNode("baselines", "stage", "rebuild", "one or more baselines require rebuild")


def _are_all_baselines_reusable(
    contract: ReportContract, deps: ReproductionDependencies
) -> bool:
    try:
        features = validate_frozen_feature_contract(contract)
        rf_prov = deps.baseline_provenance or SystemBaselineProvenance()
        for defn in BASELINE_DEFINITIONS:
            if baseline_configuration_status(
                contract, defn.configuration, defn.evaluation_role, features, rf_prov
            ) != "reusable":
                return False
        return _are_neural_baselines_reusable(contract, deps, features)
    except (OSError, ValueError):
        return False


def _are_neural_baselines_reusable(
    contract: ReportContract, deps: ReproductionDependencies, features: tuple[str, ...]
) -> bool:
    dense_prov = (
        deps.dense_baseline_dependencies.provenance
        if (deps.dense_baseline_dependencies and deps.dense_baseline_dependencies.provenance)
        else SystemDenseBaselineProvenance()
    )
    if dense_baseline_status(contract, features, dense_prov) != "reusable":
        return False
    compact_prov = deps.compact_cnn_provenance or SystemCompactCnnProvenance()
    if compact_cnn_status(contract, COMPACT_CNN_RECIPE, compact_prov) != "reusable":
        return False
    resnet_prov = deps.resnet_baseline_provenance or SystemResNetBaselineProvenance()
    return plan_resnet_baseline_stage(contract, resnet_prov) == "reusable"


def _evaluate_compare_baselines_node(
    contract: ReportContract,
    deps: ReproductionDependencies,
    gate1_blocked: bool,
    upstream_rebuilding: bool,
) -> ReproductionNode:
    if gate1_blocked:
        return ReproductionNode(
            "compare-baselines", "stage", "blocked", "blocked by unconfirmed feature selection gate"
        )
    if upstream_rebuilding:
        return ReproductionNode(
            "compare-baselines", "stage", "rebuild", "upstream baselines stage will be rebuilt"
        )
    comp_prov = deps.baseline_comparison_provenance or SystemBaselineComparisonProvenance()
    try:
        status = baseline_comparison_status(contract, comp_prov)
    except (OSError, ValueError):
        status = "obsolete"
    if status == "reusable":
        return ReproductionNode(
            "compare-baselines", "stage", "reusable", "baseline comparison report intact"
        )
    return ReproductionNode(
        "compare-baselines", "stage", "rebuild", "baseline comparison report absent or obsolete"
    )


def _evaluate_approach_gate_node(contract: ReportContract) -> ReproductionNode:
    status = approach_gate_status(contract)
    if status == "released":
        return ReproductionNode(
            "confirm-approach", "gate", "released", "confirmed approach selection package intact"
        )
    return ReproductionNode("confirm-approach", "gate", "blocked", status)


def _evaluate_tuning_node(
    contract: ReportContract,
    deps: ReproductionDependencies,
    gate2_blocked: bool,
    upstream_rebuilding: bool,
) -> ReproductionNode:
    if gate2_blocked:
        return ReproductionNode(
            "tuning", "stage", "blocked", "blocked by unconfirmed approach selection gate"
        )
    if upstream_rebuilding:
        return ReproductionNode(
            "tuning", "stage", "rebuild", "upstream compare-baselines stage will be rebuilt"
        )
    tun_prov = deps.tuning_provenance or SystemTuningProvenance()
    try:
        status = tuning_stage_status(contract, tun_prov)
    except (OSError, ValueError):
        status = "obsolete"
    if status in ("reusable", "baseline_maintained"):
        return ReproductionNode("tuning", "stage", "reusable", "tuning artifacts intact")
    return ReproductionNode("tuning", "stage", "rebuild", "tuning evidence absent or obsolete")


def _evaluate_diag_descriptive_node(
    contract: ReportContract, gate2_blocked: bool, upstream_rebuilding: bool
) -> ReproductionNode:
    if gate2_blocked:
        return ReproductionNode(
            "diagnostics-descriptive", "stage", "blocked", "blocked by unconfirmed approach gate"
        )
    if upstream_rebuilding:
        return ReproductionNode(
            "diagnostics-descriptive", "stage", "rebuild", "upstream stage will be rebuilt"
        )
    out_dir = contract.artifacts_root / "diagnostics" / "descriptive"
    if _is_diag_stage_complete(out_dir):
        return ReproductionNode(
            "diagnostics-descriptive", "stage", "reusable", "descriptive diagnostics intact"
        )
    return ReproductionNode(
        "diagnostics-descriptive", "stage", "rebuild", "descriptive diagnostics absent/obsolete"
    )


def _evaluate_diag_learning_node(
    contract: ReportContract,
    deps: ReproductionDependencies,
    gate2_blocked: bool,
    upstream_rebuilding: bool,
) -> ReproductionNode:
    del deps
    if gate2_blocked:
        return ReproductionNode(
            "diagnostics-learning", "stage", "blocked", "blocked by unconfirmed approach gate"
        )
    if upstream_rebuilding:
        return ReproductionNode(
            "diagnostics-learning", "stage", "rebuild", "upstream stage will be rebuilt"
        )
    out_dir = contract.artifacts_root / "diagnostics" / "learning_curves"
    if _is_diag_stage_complete(out_dir):
        return ReproductionNode(
            "diagnostics-learning", "stage", "reusable", "learning curves diagnostics intact"
        )
    return ReproductionNode(
        "diagnostics-learning", "stage", "rebuild", "learning curves absent/obsolete"
    )


def _evaluate_diag_sensitivity_node(
    contract: ReportContract,
    deps: ReproductionDependencies,
    gate2_blocked: bool,
    upstream_rebuilding: bool,
) -> ReproductionNode:
    del deps
    if gate2_blocked:
        return ReproductionNode(
            "diagnostics-sensitivity", "stage", "blocked", "blocked by unconfirmed approach gate"
        )
    if upstream_rebuilding:
        return ReproductionNode(
            "diagnostics-sensitivity", "stage", "rebuild", "upstream stage will be rebuilt"
        )
    out_dir = contract.artifacts_root / "diagnostics" / "sensitivity"
    if _is_diag_stage_complete(out_dir):
        return ReproductionNode(
            "diagnostics-sensitivity", "stage", "reusable", "sensitivity diagnostics intact"
        )
    return ReproductionNode(
        "diagnostics-sensitivity", "stage", "rebuild", "sensitivity diagnostics absent/obsolete"
    )


def _is_diag_stage_complete(output_dir: Path) -> bool:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        import json
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return str(manifest.get("status")) == "complete"
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _evaluate_diag_gate_node(contract: ReportContract) -> ReproductionNode:
    status = diagnostics_gate_status(contract)
    if status == "released":
        return ReproductionNode(
            "confirm-diagnostics", "gate", "released", "confirmed diagnostic package intact"
        )
    return ReproductionNode("confirm-diagnostics", "gate", "blocked", status)
