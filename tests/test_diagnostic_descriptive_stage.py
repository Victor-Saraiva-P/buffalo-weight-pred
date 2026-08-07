from __future__ import annotations

import unittest
from pathlib import Path

from buffalo_weight.diagnostic_descriptive_stage import (
    diagnostic_descriptive_output_dir,
    run_diagnostic_descriptive_stage,
)
from buffalo_weight.reproduction_config import InputsContract, ReportContract


class DiagnosticDescriptiveStageTest(unittest.TestCase):
    def test_output_dir_locates_under_artifacts_root(self) -> None:
        contract = ReportContract(
            inputs=_dummy_inputs_contract(),
            artifacts_root=Path("/repo/generated/report"),
        )
        out_dir = diagnostic_descriptive_output_dir(contract)
        self.assertEqual(out_dir, Path("/repo/generated/report/diagnostics/descriptive"))

    def test_dry_run_returns_status(self) -> None:
        contract = ReportContract(
            inputs=_dummy_inputs_contract(),
            artifacts_root=Path("/repo/generated/report"),
        )
        status = run_diagnostic_descriptive_stage(contract, dry_run=True)
        self.assertEqual(status, "reconstructible")


def _dummy_inputs_contract() -> InputsContract:
    return InputsContract(Path("index.csv"), Path("masks"), 132, 1024, 10, 5, 42)


if __name__ == "__main__":
    unittest.main()
