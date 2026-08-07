"""Public command-line interface for report reproduction."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
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
from buffalo_weight.dense_baseline_stage import (
    DenseBaselineDependencies,
    run_dense_baseline_stage,
)
from buffalo_weight.approach_confirmation import confirm_approach_selection
from buffalo_weight.environment_contract import SetupServices
from buffalo_weight.environment_setup import setup_official_environment
from buffalo_weight.feature_confirmation import (
    baselines_gate_status,
    confirm_feature_selection,
    require_baselines_gate,
)
from buffalo_weight.feature_confirmation_environment import (
    FeatureConfirmationEnvironment,
)
from buffalo_weight.feature_evaluation import FeatureBaseline
from buffalo_weight.feature_selection_provenance import FeatureSelectionProvenance
from buffalo_weight.feature_selection_stage import (
    FeatureEvidenceRunner,
    run_feature_selection_stage,
)
from buffalo_weight.report_inputs import clean_reconstructible_stage, run_inputs_stage
from buffalo_weight.report_provenance import ReportProvenance
from buffalo_weight.reproduction_config import ReportContract, load_report_contract
from buffalo_weight.resnet_baseline_provenance import ResNetBaselineProvenance
from buffalo_weight.resnet_baseline_stage import (
    ResNetBaselineRunner,
    plan_resnet_baseline_stage,
    run_resnet_baseline_stage,
)
from buffalo_weight.snapshot_io import SnapshotPublisher
from buffalo_weight.system_setup import default_setup_services
from buffalo_weight.tuning_provenance import TuningProvenance
from buffalo_weight.tuning_stage import run_tuning_stage


@dataclass(frozen=True)
class _CliDependencies:
    services: SetupServices | None
    snapshot_publisher: SnapshotPublisher | None
    report_provenance: ReportProvenance | None
    feature_evidence_runner: FeatureEvidenceRunner | None
    feature_selection_provenance: FeatureSelectionProvenance | None
    feature_confirmation_environment: FeatureConfirmationEnvironment | None
    random_forest_baseline: FeatureBaseline | None
    baseline_provenance: BaselineProvenance | None
    dense_baseline_dependencies: DenseBaselineDependencies | None
    compact_cnn_adapter: CompactCnnTrainingAdapter | None
    compact_cnn_provenance: CompactCnnProvenance | None
    resnet_baseline_runner: ResNetBaselineRunner | None
    resnet_baseline_provenance: ResNetBaselineProvenance | None
    baseline_comparison_provenance: BaselineComparisonProvenance | None
    tuning_provenance: TuningProvenance | None


def main(
    argv: Sequence[str] | None = None, services: SetupServices | None = None,
    stdout: TextIO = sys.stdout, stderr: TextIO = sys.stderr,
    snapshot_publisher: SnapshotPublisher | None = None, report_provenance: ReportProvenance | None = None,
    feature_evidence_runner: FeatureEvidenceRunner | None = None,
    feature_selection_provenance: FeatureSelectionProvenance | None = None,
    feature_confirmation_environment: FeatureConfirmationEnvironment | None = None,
    random_forest_baseline: FeatureBaseline | None = None,
    baseline_provenance: BaselineProvenance | None = None,
    dense_baseline_dependencies: DenseBaselineDependencies | None = None,
    compact_cnn_adapter: CompactCnnTrainingAdapter | None = None,
    compact_cnn_provenance: CompactCnnProvenance | None = None,
    resnet_baseline_runner: ResNetBaselineRunner | None = None,
    resnet_baseline_provenance: ResNetBaselineProvenance | None = None,
    baseline_comparison_provenance: BaselineComparisonProvenance | None = None,
    tuning_provenance: TuningProvenance | None = None,
) -> int:
    """Run the public CLI; for example, ``main(["setup"])`` prepares the environment."""
    dependencies = _CliDependencies(
        services, snapshot_publisher, report_provenance, feature_evidence_runner,
        feature_selection_provenance, feature_confirmation_environment,
        random_forest_baseline, baseline_provenance,
        dense_baseline_dependencies,
        compact_cnn_adapter, compact_cnn_provenance,
        resnet_baseline_runner, resnet_baseline_provenance,
        baseline_comparison_provenance, tuning_provenance,
    )
    return _execute_cli(argv, dependencies, stdout, stderr)


def _execute_cli(
    argv: Sequence[str] | None, dependencies: _CliDependencies,
    stdout: TextIO, stderr: TextIO,
) -> int:
    arguments = _build_parser().parse_args(argv)
    try:
        return _dispatch(arguments, dependencies, stdout)
    except (OSError, ValueError) as error:
        print(f"rejected: {error}", file=stderr)
        return 1


def _dispatch(
    arguments: argparse.Namespace, dependencies: _CliDependencies, stdout: TextIO,
) -> int:
    if arguments.command == "setup":
        return _run_setup(dependencies.services or default_setup_services(), stdout)
    contract = load_report_contract(Path(arguments.config))
    return _dispatch_report_command(arguments, dependencies, contract, stdout)


def _dispatch_report_command(
    arguments: argparse.Namespace, dependencies: _CliDependencies,
    contract: ReportContract, stdout: TextIO,
) -> int:
    if arguments.command in {"inputs", "feature-selection"}:
        return _dispatch_reconstructible_command(arguments, dependencies, contract, stdout)
    if arguments.command == "confirm-features":
        return _run_confirmation_command(
            arguments, contract, dependencies.feature_confirmation_environment,
            dependencies.feature_selection_provenance, stdout,
        )
    if arguments.command == "confirm-approach":
        return _run_approach_confirmation_command(
            arguments, contract, dependencies.feature_confirmation_environment,
            dependencies.baseline_comparison_provenance, stdout,
        )
    if arguments.command == "baselines":
        return _run_baselines_command(arguments, dependencies, contract, stdout)
    if arguments.command == "compare-baselines":
        return _run_baseline_comparison_command(arguments, dependencies, contract, stdout)
    if arguments.command == "tuning":
        return _run_tuning_command(arguments, dependencies, contract, stdout)
    removed = clean_reconstructible_stage(contract, arguments.stage)
    print(f"cleaned: {', '.join(removed) if removed else 'nothing'}", file=stdout)
    return 0


def _dispatch_reconstructible_command(
    arguments: argparse.Namespace, dependencies: _CliDependencies,
    contract: ReportContract, stdout: TextIO,
) -> int:
    if arguments.command == "inputs":
        return _run_inputs_command(
            arguments, contract, dependencies.snapshot_publisher,
            dependencies.report_provenance, stdout,
        )
    return _run_selection_command(
        arguments, contract, dependencies.feature_evidence_runner,
        dependencies.snapshot_publisher, dependencies.report_provenance,
        dependencies.feature_selection_provenance, stdout,
    )


def _run_inputs_command(
    arguments: argparse.Namespace, contract: ReportContract,
    publisher: SnapshotPublisher | None, provenance: ReportProvenance | None,
    stdout: TextIO,
) -> int:
    status = run_inputs_stage(contract, arguments.dry_run, publisher, provenance)
    print(f"inputs: {status}", file=stdout)
    return 0


def _run_selection_command(
    arguments: argparse.Namespace, contract: ReportContract,
    runner: FeatureEvidenceRunner | None, publisher: SnapshotPublisher | None,
    inputs_provenance: ReportProvenance | None,
    selection_provenance: FeatureSelectionProvenance | None, stdout: TextIO,
) -> int:
    status = run_feature_selection_stage(contract, arguments.dry_run, runner, publisher,
                                         inputs_provenance, selection_provenance)
    print(f"feature_selection: {status}", file=stdout)
    return 0


def _run_setup(services: SetupServices, stdout: TextIO) -> int:
    messages = setup_official_environment(services)
    for message in messages:
        print(message, file=stdout)
    return 0


def _run_confirmation_command(
    arguments: argparse.Namespace, contract: ReportContract,
    environment: FeatureConfirmationEnvironment | None,
    provenance: FeatureSelectionProvenance | None, stdout: TextIO,
) -> int:
    status = confirm_feature_selection(
        contract, Path(arguments.contract), Path(arguments.report), arguments.dry_run,
        environment, provenance,
    )
    print(f"feature_confirmation: {status}", file=stdout)
    return 0


def _run_approach_confirmation_command(
    arguments: argparse.Namespace, contract: ReportContract,
    environment: FeatureConfirmationEnvironment | None,
    provenance: BaselineComparisonProvenance | None, stdout: TextIO,
) -> int:
    status = confirm_approach_selection(
        contract, Path(arguments.contract), Path(arguments.report), arguments.dry_run,
        environment, provenance,
    )
    print(f"approach_confirmation: {status}", file=stdout)
    return 0


def _run_baselines_command(
    arguments: argparse.Namespace, dependencies: _CliDependencies,
    contract: ReportContract, stdout: TextIO,
) -> int:
    status = baselines_gate_status(contract)
    print(f"baselines: {status}", file=stdout)
    if status != "released":
        if not arguments.dry_run:
            require_baselines_gate(contract)
        return 0
    results = run_random_forest_baseline_stage(
        contract, arguments.dry_run, dependencies.random_forest_baseline,
        dependencies.baseline_provenance, inputs_provenance=dependencies.report_provenance,
    )
    for configuration, result in results.items():
        print(f"{configuration}: {result}", file=stdout)
    dense_status = run_dense_baseline_stage(
        contract, arguments.dry_run, dependencies.snapshot_publisher,
        dependencies.dense_baseline_dependencies,
    )
    print(dense_status.removeprefix("released; "), file=stdout)
    compact_status = run_compact_cnn_stage(
        contract, arguments.dry_run, dependencies.compact_cnn_adapter,
        dependencies.compact_cnn_provenance,
    )
    print(f"compact_cnn: {compact_status}", file=stdout)
    plan = plan_resnet_baseline_stage(contract, dependencies.resnet_baseline_provenance)
    print(f"resnet18_baseline: {plan}", file=stdout)
    if arguments.dry_run or plan == "reusable":
        return 0
    baseline_status = run_resnet_baseline_stage(
        contract, False, dependencies.resnet_baseline_runner,
        dependencies.resnet_baseline_provenance,
    )
    print(f"resnet18_baseline: {baseline_status}", file=stdout)
    return 0


def _run_baseline_comparison_command(
    arguments: argparse.Namespace, dependencies: _CliDependencies,
    contract: ReportContract, stdout: TextIO,
) -> int:
    status = run_baseline_comparison_stage(
        contract, arguments.dry_run, dependencies.snapshot_publisher,
        dependencies.baseline_comparison_provenance,
        _comparison_upstream_dependencies(dependencies),
    )
    print(f"baseline_comparison: {status}", file=stdout)
    return 0


def _run_tuning_command(
    arguments: argparse.Namespace, dependencies: _CliDependencies,
    contract: ReportContract, stdout: TextIO,
) -> int:
    dense_adapter = dependencies.dense_baseline_dependencies.adapter if (
        dependencies.dense_baseline_dependencies is not None
    ) else None
    status = run_tuning_stage(
        contract, arguments.dry_run, dependencies.tuning_provenance,
        dependencies.snapshot_publisher, dense_adapter,
        dependencies.compact_cnn_adapter, dependencies.resnet_baseline_runner,
    )
    print(f"tuning: {status}", file=stdout)
    return 0


def _comparison_upstream_dependencies(
    dependencies: _CliDependencies,
) -> BaselineComparisonUpstreamDependencies:
    return BaselineComparisonUpstreamDependencies(
        dependencies.report_provenance, dependencies.baseline_provenance,
        dependencies.dense_baseline_dependencies, dependencies.compact_cnn_provenance,
        dependencies.resnet_baseline_provenance,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce buffalo-weight report evidence")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("setup", help="prepare and audit the official environment")
    _add_reconstructible_stage_parsers(subcommands)
    _add_confirmation_parsers(subcommands)
    clean = subcommands.add_parser("clean", help="remove reconstructible stage artifacts")
    clean.add_argument("stage")
    clean.add_argument("--config", default="configs/report.yaml")
    return parser


def _add_reconstructible_stage_parsers(
    subcommands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    inputs = subcommands.add_parser("inputs", help="validate masks and build features and folds")
    inputs.add_argument("--config", default="configs/report.yaml")
    inputs.add_argument("--dry-run", action="store_true")
    selection = subcommands.add_parser(
        "feature-selection", help="build comparative feature evidence for human review"
    )
    selection.add_argument("--config", default="configs/report.yaml")
    selection.add_argument("--dry-run", action="store_true")
    baseline_comparison_parser = subcommands.add_parser(
        "compare-baselines", help="consolidate current baselines for human review"
    )
    baseline_comparison_parser.add_argument("--config", default="configs/report.yaml")
    baseline_comparison_parser.add_argument("--dry-run", action="store_true")
    tuning_parser = subcommands.add_parser(
        "tuning", help="execute pre-registered configuration tuning for the confirmed approach"
    )
    tuning_parser.add_argument("--config", default="configs/report.yaml")
    tuning_parser.add_argument("--dry-run", action="store_true")


def _add_confirmation_parsers(
    subcommands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    confirmation = subcommands.add_parser(
        "confirm-features", help="promote a reviewed shared-feature decision"
    )
    confirmation.add_argument("--config", default="configs/report.yaml")
    confirmation.add_argument("--contract", required=True)
    confirmation.add_argument("--report", required=True)
    confirmation.add_argument("--dry-run", action="store_true")
    approach_confirmation = subcommands.add_parser(
        "confirm-approach", help="promote a reviewed approach decision"
    )
    approach_confirmation.add_argument("--config", default="configs/report.yaml")
    approach_confirmation.add_argument("--contract", required=True)
    approach_confirmation.add_argument("--report", required=True)
    approach_confirmation.add_argument("--dry-run", action="store_true")
    baselines = subcommands.add_parser(
        "baselines", help="evaluate frozen baselines after the shared-feature gate"
    )
    baselines.add_argument("--config", default="configs/report.yaml")
    baselines.add_argument("--dry-run", action="store_true")
