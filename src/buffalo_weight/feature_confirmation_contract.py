"""Schema and report checks for the human feature-selection decision."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from buffalo_weight.hashing import sha256_file

CONFIRMED_CONTRACT_KEYS = {
    "schema_version", "status", "selected_features", "standardization",
    "report_sha256", "human_decision",
}
STANDARDIZATION_RULE = "fit within each permitted training partition"
HUMAN_DECISION_KEYS = {"decision_url", "reviewer", "reviewed_at"}


def validate_confirmed_feature_contract(
    contract: object, report_path: Path, candidate_features: tuple[str, ...],
) -> dict[str, object]:
    """Validate one human contract; for example, selected features retain candidate order."""
    if not isinstance(contract, dict) or set(contract) != CONFIRMED_CONTRACT_KEYS:
        actual_keys = sorted(contract) if isinstance(contract, dict) else contract
        raise ValueError(
            f"confirmed contract keys were {actual_keys!r}; expected exactly "
            f"{sorted(CONFIRMED_CONTRACT_KEYS)!r}"
        )
    _validate_fixed_fields(contract, report_path)
    _validate_human_decision(contract.get("human_decision"))
    selected = _selected_features(contract.get("selected_features"))
    _validate_selected_features(selected, report_path.read_text(), candidate_features)
    return contract


def decision_url(contract: dict[str, object]) -> str:
    """Return the recorded decision URL; for example, manifests reference the review."""
    decision = contract.get("human_decision")
    if not isinstance(decision, dict) or not isinstance(decision.get("decision_url"), str):
        raise ValueError(f"human_decision was {decision!r}; expected a recorded decision URL")
    return decision["decision_url"]


def _validate_fixed_fields(contract: dict[str, object], report_path: Path) -> None:
    fixed = (contract.get("schema_version"), contract.get("status"),
             contract.get("standardization"))
    expected = (1, "confirmed", STANDARDIZATION_RULE)
    if fixed != expected:
        raise ValueError(f"confirmed contract fixed fields were {fixed!r}; expected {expected!r}")
    expected_hash = sha256_file(report_path)
    if contract.get("report_sha256") != expected_hash:
        raise ValueError(
            f"report_sha256 was {contract.get('report_sha256')!r}; expected {expected_hash!r} "
            f"for reviewed report {report_path}"
        )
    _validate_reviewed_report(report_path)


def _validate_reviewed_report(report_path: Path) -> None:
    report = report_path.read_text()
    required = "## Registro de revisão humana\n"
    if required not in report or "- Status: revisado" not in report:
        observed = tuple(
            line for line in report.splitlines()
            if "Registro de revisão humana" in line or "Status:" in line
        )
        raise ValueError(
            f"review record in {report_path} was {observed!r}; expected section "
            "'Registro de revisão humana' with status line '- Status: revisado'"
        )
    pending_markers = ("Status: pendente", "não preenchid")
    placeholders = tuple(value for value in pending_markers if value in report)
    if placeholders:
        raise ValueError(
            f"review report placeholders were {placeholders!r}; expected completed human review"
        )


def _validate_human_decision(candidate: object) -> None:
    if not isinstance(candidate, dict) or set(candidate) != HUMAN_DECISION_KEYS:
        keys = sorted(candidate) if isinstance(candidate, dict) else candidate
        raise ValueError(
            f"human_decision keys were {keys!r}; expected exactly {sorted(HUMAN_DECISION_KEYS)!r}"
        )
    for field in ("decision_url", "reviewer", "reviewed_at"):
        value = candidate.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"human_decision {field} was {value!r}; expected non-empty text")
    _validate_review_date(candidate["reviewed_at"])


def _validate_review_date(value: object) -> None:
    try:
        date.fromisoformat(str(value))
    except ValueError as error:
        raise ValueError(
            f"human_decision reviewed_at was {value!r}; expected an ISO date YYYY-MM-DD"
        ) from error


def _selected_features(candidate: object) -> tuple[str, ...]:
    if not isinstance(candidate, list) or not candidate:
        raise ValueError(
            f"selected_features was {candidate!r}; expected a non-empty ordered list of strings"
        )
    invalid = next((value for value in candidate if not isinstance(value, str)), None)
    if invalid is not None:
        raise ValueError(f"selected feature was {invalid!r}; expected a string")
    return tuple(candidate)


def _validate_selected_features(
    selected: tuple[str, ...], report: str, candidates: tuple[str, ...],
) -> None:
    _validate_candidate_membership(selected, candidates)
    repeated = next((feature for feature in selected if selected.count(feature) > 1), None)
    if repeated is not None:
        raise ValueError(f"selected feature was {repeated!r}; expected unique features")
    _validate_candidate_order(selected, candidates)
    missing = next((feature for feature in selected if f"`{feature}`" not in report), None)
    if missing is not None:
        raise ValueError(
            f"selected feature was {missing!r}; expected it to be present in the reviewed report"
        )


def _validate_candidate_membership(
    selected: tuple[str, ...], candidates: tuple[str, ...],
) -> None:
    unknown = next((feature for feature in selected if feature not in candidates), None)
    if unknown is not None:
        raise ValueError(
            f"selected feature was {unknown!r}; expected one of the 26 candidate features "
            f"{candidates!r}"
        )


def _validate_candidate_order(
    selected: tuple[str, ...], candidates: tuple[str, ...],
) -> None:
    positions = tuple(candidates.index(feature) for feature in selected)
    if positions != tuple(sorted(positions)):
        raise ValueError(
            f"selected feature order was {selected!r}; expected candidate order {candidates!r}"
        )
