"""Unit tests for diagnostic confirmation contract validation.

Reference: GitHub Issue #27.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from buffalo_weight.diagnostic_confirmation_contract import (
    CONFIRMED_DIAGNOSTIC_CONTRACT_KEYS,
    validate_confirmed_diagnostic_contract,
)
from buffalo_weight.hashing import sha256_file


class TestDiagnosticConfirmationContract(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())
        self.report_path = self.temp_dir / "expanded_diagnostics_report.md"
        self.report_content = (
            "# Relatório Diagnóstico Confirmado\n\n"
            "Resultados avaliados sob MAE OOF Pós-Seleção.\n\n"
            "## Registro de revisão humana\n"
            "- Revisor: Especialista\n"
            "- Status: revisado\n"
        )
        self.report_path.write_text(self.report_content, encoding="utf-8")
        self.valid_contract = {
            "schema_version": 1,
            "status": "confirmed",
            "diagnostic_scope": "expanded",
            "source_report_sha256": sha256_file(self.report_path),
            "no_decision_reopening": True,
            "human_decision": {
                "decision_url": "https://github.com/Victor-Saraiva-P/buffalo-weight-pred/issues/27",
                "reviewer": "Especialista",
                "reviewed_at": "2026-08-06",
            },
        }

    def test_valid_contract_passes(self) -> None:
        result = validate_confirmed_diagnostic_contract(self.valid_contract, self.report_path)
        self.assertEqual(result["status"], "confirmed")

    def test_missing_keys_raises(self) -> None:
        invalid = dict(self.valid_contract)
        invalid.pop("no_decision_reopening")
        with self.assertRaises(ValueError) as ctx:
            validate_confirmed_diagnostic_contract(invalid, self.report_path)
        self.assertIn("confirmed diagnostic contract keys were", str(ctx.exception))

    def test_wrong_scope_raises(self) -> None:
        invalid = dict(self.valid_contract)
        invalid["diagnostic_scope"] = "invalid_scope"
        with self.assertRaises(ValueError) as ctx:
            validate_confirmed_diagnostic_contract(invalid, self.report_path)
        self.assertIn("diagnostic_scope was 'invalid_scope'", str(ctx.exception))

    def test_unlocked_decisions_raises(self) -> None:
        invalid = dict(self.valid_contract)
        invalid["no_decision_reopening"] = False
        with self.assertRaises(ValueError) as ctx:
            validate_confirmed_diagnostic_contract(invalid, self.report_path)
        self.assertIn("no_decision_reopening was False", str(ctx.exception))

    def test_missing_oof_framing_raises(self) -> None:
        self.report_path.write_text(
            "# Relatório\n\nSem o termo obrigatorio.\n\n## Registro de revisão humana\n- Status: revisado\n",
            encoding="utf-8",
        )
        invalid = dict(self.valid_contract)
        invalid["source_report_sha256"] = sha256_file(self.report_path)
        with self.assertRaises(ValueError) as ctx:
            validate_confirmed_diagnostic_contract(invalid, self.report_path)
        self.assertIn("missing required framing 'MAE OOF Pós-Seleção'", str(ctx.exception))

    def test_forbidden_term_raises(self) -> None:
        self.report_path.write_text(
            "# Relatório\n\nMAE OOF Pós-Seleção com p-valor calculado.\n\n## Registro de revisão humana\n- Status: revisado\n",
            encoding="utf-8",
        )
        invalid = dict(self.valid_contract)
        invalid["source_report_sha256"] = sha256_file(self.report_path)
        with self.assertRaises(ValueError) as ctx:
            validate_confirmed_diagnostic_contract(invalid, self.report_path)
        self.assertIn("contains forbidden terms", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
