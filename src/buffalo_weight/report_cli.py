"""Public command-line interface for report reproduction."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from buffalo_weight.environment_contract import SetupServices
from buffalo_weight.environment_setup import setup_official_environment
from buffalo_weight.feature_selection_stage import FeatureEvidenceRunner, run_feature_selection_stage
from buffalo_weight.feature_selection_provenance import FeatureSelectionProvenance
from buffalo_weight.report_inputs import clean_reconstructible_stage, run_inputs_stage
from buffalo_weight.report_provenance import ReportProvenance
from buffalo_weight.reproduction_config import ReportContract, load_report_contract
from buffalo_weight.snapshot_io import SnapshotPublisher
from buffalo_weight.system_setup import default_setup_services


def main(
    argv: Sequence[str] | None = None,
    services: SetupServices | None = None,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
    snapshot_publisher: SnapshotPublisher | None = None,
    report_provenance: ReportProvenance | None = None,
    feature_evidence_runner: FeatureEvidenceRunner | None = None,
    feature_selection_provenance: FeatureSelectionProvenance | None = None,
) -> int:
    """Run the public CLI; for example, ``main(["setup"])`` prepares the environment."""
    arguments = _build_parser().parse_args(argv)
    try:
        return _dispatch(
            arguments, services, snapshot_publisher, report_provenance,
            feature_evidence_runner, feature_selection_provenance, stdout
        )
    except (OSError, ValueError) as error:
        print(f"rejected: {error}", file=stderr)
        return 1


def _dispatch(
    arguments: argparse.Namespace, services: SetupServices | None,
    snapshot_publisher: SnapshotPublisher | None, report_provenance: ReportProvenance | None,
    feature_evidence_runner: FeatureEvidenceRunner | None,
    feature_selection_provenance: FeatureSelectionProvenance | None,
    stdout: TextIO,
) -> int:
    if arguments.command == "setup":
        return _run_setup(services or default_setup_services(), stdout)
    contract = load_report_contract(Path(arguments.config))
    if arguments.command == "inputs":
        return _run_inputs_command(arguments, contract, snapshot_publisher,
                                   report_provenance, stdout)
    if arguments.command == "feature-selection":
        return _run_selection_command(arguments, contract, feature_evidence_runner,
                                      snapshot_publisher, report_provenance,
                                      feature_selection_provenance, stdout)
    removed = clean_reconstructible_stage(contract, arguments.stage)
    print(f"cleaned: {', '.join(removed) if removed else 'nothing'}", file=stdout)
    return 0


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce buffalo-weight report evidence")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("setup", help="prepare and audit the official environment")
    inputs = subcommands.add_parser("inputs", help="validate masks and build features and folds")
    inputs.add_argument("--config", default="configs/report.yaml")
    inputs.add_argument("--dry-run", action="store_true")
    selection = subcommands.add_parser(
        "feature-selection", help="build comparative feature evidence for human review"
    )
    selection.add_argument("--config", default="configs/report.yaml")
    selection.add_argument("--dry-run", action="store_true")
    clean = subcommands.add_parser("clean", help="remove reconstructible stage artifacts")
    clean.add_argument("stage")
    clean.add_argument("--config", default="configs/report.yaml")
    return parser
