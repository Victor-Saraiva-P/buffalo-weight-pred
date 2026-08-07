"""Tests for diagnostic learning stage output directory and dry-run planning."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from buffalo_weight.diagnostic_learning_stage import (
    diagnostic_learning_output_dir,
    run_diagnostic_learning_stage,
)
from buffalo_weight.diagnostic_learning_types import LearningCurvesSlice
from buffalo_weight.reproduction_config import InputsContract, ReportContract


class DiagnosticLearningStageTest(unittest.TestCase):
    def test_output_dir_locates_under_artifacts_root(self) -> None:
        contract = ReportContract(
            inputs=_dummy_inputs_contract(),
            artifacts_root=Path("/repo/generated/report"),
        )
        out_dir = diagnostic_learning_output_dir(contract)
        self.assertEqual(out_dir, Path("/repo/generated/report/diagnostics/learning_curves"))

    def test_dry_run_returns_status(self) -> None:
        contract = ReportContract(
            inputs=_dummy_inputs_contract(),
            artifacts_root=Path("/repo/generated/report"),
        )
        status = run_diagnostic_learning_stage(contract, dry_run=True)
        self.assertEqual(status, "reconstructible")

    def test_dry_run_returns_reusable_when_manifest_exists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = ReportContract(
                inputs=_dummy_inputs_contract(),
                artifacts_root=root,
            )
            out_dir = diagnostic_learning_output_dir(contract)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "manifest.json").write_text('{"status": "complete"}\n', encoding="utf-8")

            status = run_diagnostic_learning_stage(contract, dry_run=True)
            self.assertEqual(status, "reusable")

    def test_execution_resolves_default_gpu_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            contract = ReportContract(
                inputs=_dummy_inputs_contract(),
                artifacts_root=Path(directory),
            )
            empty_slice = LearningCurvesSlice((), ())
            with (
                patch(
                    "buffalo_weight.diagnostic_learning_stage.evaluate_learning_curves",
                    return_value=empty_slice,
                ) as evaluate,
                patch("buffalo_weight.diagnostic_learning_stage.write_learning_curves_artifacts"),
                patch("buffalo_weight.diagnostic_learning_stage._write_stage_manifest"),
            ):
                status = run_diagnostic_learning_stage(contract)

            self.assertEqual(status, "rebuilt")
            self.assertIsNotNone(evaluate.call_args.kwargs["compact_adapter"])
            self.assertIsNotNone(evaluate.call_args.kwargs["resnet_runner"])


def _dummy_inputs_contract() -> InputsContract:
    return InputsContract(Path("index.csv"), Path("masks"), 132, 1024, 10, 5, 42)


if __name__ == "__main__":
    unittest.main()
