"""Manifest creation and verification for confirmed expanded diagnostics evidence.

Reference: GitHub Issue #27.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from buffalo_weight.diagnostic_artifacts import (
    COVERAGE_COLUMNS,
    NOTABLE_CASE_COLUMNS,
    RESIDUAL_CORRELATION_COLUMNS,
    STRATIFIED_COLUMNS,
)
from buffalo_weight.diagnostic_learning_artifacts import SUMMARY_COLUMNS
from buffalo_weight.diagnostic_sensitivity_artifacts import SENSITIVITY_COLUMNS
from buffalo_weight.hashing import sha256_file
from buffalo_weight.reproduction_config import ReportContract, contract_identity

CANONICAL_TABLE_SCHEMAS: dict[str, list[str]] = {
    "sample_coverage.csv": COVERAGE_COLUMNS,
    "stratified_metrics.csv": STRATIFIED_COLUMNS,
    "residual_correlations.csv": RESIDUAL_CORRELATION_COLUMNS,
    "notable_cases.csv": NOTABLE_CASE_COLUMNS,
    "learning_curves_summary.csv": SUMMARY_COLUMNS,
    "sensitivity_perturbations.csv": SENSITIVITY_COLUMNS,
}

CONFIRMED_DIAGNOSTIC_MANIFEST_KEYS = {
    "schema_version",
    "stage",
    "status",
    "execution",
    "repository_commit",
    "contract_identity",
    "report_sha256",
    "human_decision",
    "canonical_tables",
    "outputs",
}


def build_confirmed_diagnostic_manifest(
    package_dir: Path,
    contract: ReportContract,
    human_contract: dict[str, object],
    commit: str = "unknown",
) -> dict[str, object]:
    """Build the confirmed diagnostic package manifest.

    Example: ``build_confirmed_diagnostic_manifest(dir, contract, human)`` returns dict.
    """
    validate_canonical_tables(package_dir, contract.inputs.expected_mask_count)
    table_records = _summarize_canonical_tables(package_dir)
    outputs_record = _summarize_package_outputs(package_dir)
    return {
        "schema_version": 1,
        "stage": "confirmed_diagnostics",
        "status": "confirmed",
        "execution": "cuda_official",
        "repository_commit": commit,
        "contract_identity": contract_identity(contract),
        "report_sha256": sha256_file(package_dir / "expanded_diagnostics_report.md"),
        "human_decision": human_contract.get("human_decision", {}),
        "canonical_tables": table_records,
        "outputs": outputs_record,
    }


def validate_confirmed_diagnostic_manifest(
    manifest: object,
    package_dir: Path,
    human_contract: dict[str, object],
    contract: ReportContract,
) -> None:
    """Validate an existing confirmed diagnostic manifest against package directory.

    Example: ``validate_confirmed_diagnostic_manifest(manifest, dir, human, contract)`` checks manifest.
    """
    if not isinstance(manifest, dict) or set(manifest) != CONFIRMED_DIAGNOSTIC_MANIFEST_KEYS:
        actual_keys = sorted(manifest) if isinstance(manifest, dict) else manifest
        raise ValueError(
            f"confirmed diagnostic manifest keys were {actual_keys!r}; expected exactly "
            f"{sorted(CONFIRMED_DIAGNOSTIC_MANIFEST_KEYS)!r}"
        )
    _validate_manifest_fixed_fields(manifest)
    _validate_manifest_identity(manifest.get("contract_identity"), contract)
    _validate_manifest_report_hash(manifest.get("report_sha256"), package_dir)
    validate_canonical_tables(package_dir, contract.inputs.expected_mask_count)
    _validate_manifest_outputs(manifest.get("outputs"), package_dir)


def validate_canonical_tables(package_dir: Path, expected_sample_count: int) -> None:
    """Validate headers and cross-table integrity for the six canonical tables.

    Example: ``validate_canonical_tables(dir, 132)`` verifies CSV schemas and counts.
    """
    table_rows: dict[str, list[dict[str, str]]] = {}
    for filename, expected_headers in CANONICAL_TABLE_SCHEMAS.items():
        path = package_dir / filename
        if not path.is_file():
            raise ValueError(f"canonical table {filename} was missing from {package_dir}")
        table_rows[filename] = _read_and_validate_csv_headers(path, expected_headers)
    _validate_cross_table_integrity(table_rows, expected_sample_count)


def read_json_mapping(path: Path, label: str) -> dict[str, object]:
    """Read a JSON mapping file; for example, ``read_json_mapping(path, 'manifest')``."""
    if not path.is_file():
        raise ValueError(f"{label} was missing at {path}")
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ValueError(f"failed to parse {label} at {path}: {error}") from error
    if not isinstance(content, dict):
        raise ValueError(f"{label} at {path} was {type(content).__name__}; expected a mapping")
    return content


def _read_and_validate_csv_headers(path: Path, expected_headers: list[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        headers = list(reader.fieldnames or [])
        rows = list(reader)
    if headers[:len(expected_headers)] != expected_headers:
        raise ValueError(
            f"table {path.name} headers were {headers!r}; expected starting with {expected_headers!r}"
        )
    return rows


def _validate_cross_table_integrity(
    table_rows: dict[str, list[dict[str, str]]],
    expected_sample_count: int,
) -> None:
    _validate_coverage_integrity(table_rows["sample_coverage.csv"], expected_sample_count)
    _validate_stratified_integrity(table_rows["stratified_metrics.csv"])
    _validate_correlations_integrity(table_rows["residual_correlations.csv"])
    _validate_notable_cases_integrity(table_rows["notable_cases.csv"])
    _validate_learning_summary_integrity(table_rows["learning_curves_summary.csv"])
    _validate_sensitivity_integrity(table_rows["sensitivity_perturbations.csv"])


def _validate_coverage_integrity(rows: list[dict[str, str]], expected_count: int) -> None:
    cat_sum = sum(int(r["sample_count"]) for r in rows if r["stratum_type"] == "weight_category")
    if cat_sum != expected_count:
        raise ValueError(
            f"sample_coverage category sum was {cat_sum}; expected total {expected_count}"
        )


def _validate_stratified_integrity(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("stratified_metrics.csv was empty; expected evaluated rows")
    for r in rows:
        if not r.get("configuration") or not r.get("mae_kg"):
            raise ValueError(f"invalid stratified metric row: {r!r}")


def _validate_correlations_integrity(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("residual_correlations.csv was empty; expected correlation rows")


def _validate_notable_cases_integrity(rows: list[dict[str, str]]) -> None:
    valid_types = {"shared_hard_case", "divergent_case"}
    for r in rows:
        if r.get("case_type") not in valid_types:
            raise ValueError(
                f"notable case type was {r.get('case_type')!r}; expected one of {sorted(valid_types)!r}"
            )


def _validate_learning_summary_integrity(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("learning_curves_summary.csv was empty; expected summary rows")


def _validate_sensitivity_integrity(rows: list[dict[str, str]]) -> None:
    valid_statuses = {"evaluated", "rejected"}
    for r in rows:
        if r.get("status") not in valid_statuses:
            raise ValueError(
                f"sensitivity row status was {r.get('status')!r}; expected one of {sorted(valid_statuses)!r}"
            )


def _summarize_canonical_tables(package_dir: Path) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for filename in CANONICAL_TABLE_SCHEMAS:
        path = package_dir / filename
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            rows = list(reader)
        summary[filename] = {
            "row_count": len(rows),
            "sha256": sha256_file(path),
        }
    return summary


def _summarize_package_outputs(package_dir: Path) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for path in sorted(package_dir.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            outputs[path.name] = sha256_file(path)
    return outputs


def _validate_manifest_fixed_fields(manifest: dict[str, object]) -> None:
    if manifest.get("status") != "confirmed" or manifest.get("stage") != "confirmed_diagnostics":
        raise ValueError(
            f"manifest stage/status were {manifest.get('stage')!r}/{manifest.get('status')!r}; "
            "expected 'confirmed_diagnostics'/'confirmed'"
        )


def _validate_manifest_identity(candidate: object, contract: ReportContract) -> None:
    expected = contract_identity(contract)
    if candidate != expected:
        raise ValueError(f"manifest contract_identity was {candidate!r}; expected {expected!r}")


def _validate_manifest_report_hash(candidate: object, package_dir: Path) -> None:
    report_path = package_dir / "expanded_diagnostics_report.md"
    expected = sha256_file(report_path)
    if candidate != expected:
        raise ValueError(f"manifest report_sha256 was {candidate!r}; expected {expected!r}")


def _validate_manifest_outputs(candidate: object, package_dir: Path) -> None:
    if not isinstance(candidate, dict):
        raise ValueError(f"manifest outputs were {candidate!r}; expected a dictionary")
    for filename, expected_hash in candidate.items():
        path = package_dir / str(filename)
        if not path.is_file():
            raise ValueError(f"output file {filename} declared in manifest was missing")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"file {filename} hash was {actual_hash!r}; expected {expected_hash!r}"
            )
