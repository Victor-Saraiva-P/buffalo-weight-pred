"""Schema and report checks for human expanded diagnostics decision.

Reference: GitHub Issue #27.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from buffalo_weight.hashing import sha256_file

CONFIRMED_DIAGNOSTIC_CONTRACT_KEYS = {
    "schema_version",
    "status",
    "diagnostic_scope",
    "source_report_sha256",
    "no_decision_reopening",
    "human_decision",
}
HUMAN_DECISION_KEYS = {"decision_url", "reviewer", "reviewed_at"}
ALLOWED_DIAGNOSTIC_SCOPES = {"expanded", "descriptive_learning_sensitivity"}

FORBIDDEN_REPORT_TERMS = (
    "bootstrap",
    "p-valor",
    "p-value",
    "modelo-oráculo",
    "modelo oráculo",
    "ranking por estrato",
    "alegações causais",
    "alegação causal",
    "validação independente",
    "desempenho em animais novos",
)


def validate_confirmed_diagnostic_contract(
    contract: object,
    report_path: Path,
) -> dict[str, object]:
    """Validate human diagnostic contract and report.

    Example: ``validate_confirmed_diagnostic_contract(c, path)`` checks structure.
    """
    if not isinstance(contract, dict) or set(contract) != CONFIRMED_DIAGNOSTIC_CONTRACT_KEYS:
        actual_keys = sorted(contract) if isinstance(contract, dict) else contract
        raise ValueError(
            f"confirmed diagnostic contract keys were {actual_keys!r}; expected exactly "
            f"{sorted(CONFIRMED_DIAGNOSTIC_CONTRACT_KEYS)!r}"
        )
    _validate_fixed_fields(contract, report_path)
    _validate_scope_and_lock(contract)
    _validate_human_decision(contract.get("human_decision"))
    _validate_reviewed_report(report_path)
    return contract


def _validate_fixed_fields(contract: dict[str, object], report_path: Path) -> None:
    fixed = (contract.get("schema_version"), contract.get("status"))
    expected = (1, "confirmed")
    if fixed != expected:
        raise ValueError(f"confirmed diagnostic contract fixed fields were {fixed!r}; expected {expected!r}")
    expected_hash = sha256_file(report_path)
    if contract.get("source_report_sha256") != expected_hash:
        raise ValueError(
            f"source_report_sha256 was {contract.get('source_report_sha256')!r}; expected {expected_hash!r} "
            f"for reviewed report {report_path}"
        )


def _validate_scope_and_lock(contract: dict[str, object]) -> None:
    scope = contract.get("diagnostic_scope")
    if scope not in ALLOWED_DIAGNOSTIC_SCOPES:
        raise ValueError(
            f"diagnostic_scope was {scope!r}; expected one of {sorted(ALLOWED_DIAGNOSTIC_SCOPES)!r}"
        )
    lock = contract.get("no_decision_reopening")
    if lock is not True:
        raise ValueError(
            f"no_decision_reopening was {lock!r}; expected True to confirm decisions are locked"
        )


def _validate_reviewed_report(report_path: Path) -> None:
    report = report_path.read_text(encoding="utf-8")
    _validate_review_section(report_path, report)
    _validate_report_placeholders(report_path, report)
    _validate_oof_identification(report_path, report)
    _validate_forbidden_terms(report_path, report)


def _validate_review_section(report_path: Path, report: str) -> None:
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


def _validate_report_placeholders(report_path: Path, report: str) -> None:
    pending_markers = ("Status: pendente", "não preenchid")
    placeholders = tuple(value for value in pending_markers if value in report)
    if placeholders:
        raise ValueError(
            f"review report placeholders in {report_path} were {placeholders!r}; expected completed human review"
        )


def _validate_oof_identification(report_path: Path, report: str) -> None:
    if "MAE OOF Pós-Seleção" not in report:
        raise ValueError(
            f"report at {report_path} missing required framing 'MAE OOF Pós-Seleção'"
        )


def _validate_forbidden_terms(report_path: Path, report: str) -> None:
    report_lower = report.lower()
    found = [term for term in FORBIDDEN_REPORT_TERMS if term in report_lower]
    if found:
        raise ValueError(
            f"report at {report_path} contains forbidden terms {found!r}; expected clean diagnostic report"
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
