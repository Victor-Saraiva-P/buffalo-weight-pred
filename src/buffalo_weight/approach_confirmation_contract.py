"""Schema and report checks for the human approach-selection decision."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from buffalo_weight.hashing import sha256_file

CONFIRMED_CONTRACT_KEYS = {
    "schema_version", "status", "selected_approach", "baseline_configuration",
    "maximum_tuning_variations", "source_report_sha256", "human_decision",
}
HUMAN_DECISION_KEYS = {"decision_url", "reviewer", "reviewed_at"}
ELIGIBLE_APPROACHES = {
    "random_forest": "random_forest_baseline",
    "dense_feature_network": "dense",
    "compact_cnn": "compact_cnn",
    "resnet18": "resnet18_pretrained_partial",
}
MAX_TUNING_VARIATIONS_CEILING = 3


def validate_confirmed_approach_contract(
    contract: object, report_path: Path,
) -> dict[str, object]:
    """Validate one human approach contract; for example, matching approach and baseline configuration."""
    if not isinstance(contract, dict) or set(contract) != CONFIRMED_CONTRACT_KEYS:
        actual_keys = sorted(contract) if isinstance(contract, dict) else contract
        raise ValueError(
            f"confirmed contract keys were {actual_keys!r}; expected exactly "
            f"{sorted(CONFIRMED_CONTRACT_KEYS)!r}"
        )
    _validate_fixed_fields(contract, report_path)
    _validate_approach_and_config(contract)
    _validate_tuning_budget(contract.get("maximum_tuning_variations"))
    _validate_human_decision(contract.get("human_decision"))
    _validate_reviewed_report(report_path, str(contract.get("selected_approach")))
    return contract


def decision_url(contract: dict[str, object]) -> str:
    """Return the recorded decision URL; for example, manifests reference the review."""
    decision = contract.get("human_decision")
    if not isinstance(decision, dict) or not isinstance(decision.get("decision_url"), str):
        raise ValueError(f"human_decision was {decision!r}; expected a recorded decision URL")
    return decision["decision_url"]


def _validate_fixed_fields(contract: dict[str, object], report_path: Path) -> None:
    fixed = (contract.get("schema_version"), contract.get("status"))
    expected = (1, "confirmed")
    if fixed != expected:
        raise ValueError(f"confirmed contract fixed fields were {fixed!r}; expected {expected!r}")
    expected_hash = sha256_file(report_path)
    if contract.get("source_report_sha256") != expected_hash:
        raise ValueError(
            f"source_report_sha256 was {contract.get('source_report_sha256')!r}; expected {expected_hash!r} "
            f"for reviewed report {report_path}"
        )


def _validate_approach_and_config(contract: dict[str, object]) -> None:
    approach = contract.get("selected_approach")
    config = contract.get("baseline_configuration")
    if not isinstance(approach, str) or approach not in ELIGIBLE_APPROACHES:
        raise ValueError(
            f"selected_approach was {approach!r}; expected one of {sorted(ELIGIBLE_APPROACHES)!r}"
        )
    expected_config = ELIGIBLE_APPROACHES[approach]
    if config != expected_config:
        raise ValueError(
            f"baseline_configuration was {config!r} for approach {approach!r}; "
            f"expected {expected_config!r}"
        )


def _validate_tuning_budget(candidate: object) -> None:
    if isinstance(candidate, bool) or not isinstance(candidate, int):
        raise ValueError(
            f"maximum_tuning_variations was {candidate!r}; expected an integer between 0 and "
            f"{MAX_TUNING_VARIATIONS_CEILING}"
        )
    if candidate < 0 or candidate > MAX_TUNING_VARIATIONS_CEILING:
        raise ValueError(
            f"maximum_tuning_variations was {candidate}; expected an integer between 0 and "
            f"{MAX_TUNING_VARIATIONS_CEILING}"
        )


def _validate_reviewed_report(report_path: Path, selected_approach: str) -> None:
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
    _validate_report_placeholders(report_path, report)
    _validate_report_approach_mention(report_path, report, selected_approach)


def _validate_report_placeholders(report_path: Path, report: str) -> None:
    pending_markers = ("Status: pendente", "não preenchid")
    placeholders = tuple(value for value in pending_markers if value in report)
    if placeholders:
        raise ValueError(
            f"review report placeholders in {report_path} were {placeholders!r}; expected completed human review"
        )


def _validate_report_approach_mention(
    report_path: Path, report: str, selected_approach: str,
) -> None:
    if selected_approach not in report:
        raise ValueError(
            f"selected_approach {selected_approach!r} was absent from {report_path}; "
            "expected report to mention the selected approach"
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
