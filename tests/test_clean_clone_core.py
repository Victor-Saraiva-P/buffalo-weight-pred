"""Tests for clean clone reproducible core integration and verification."""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

from buffalo_weight.report_cli import main
from buffalo_weight.report_provenance import ReportProvenance
from buffalo_weight.reproduction_config import load_report_contract
from tests.fake_report_provenance import FixedReportProvenance
from tests.report_inputs_fixture import CuratedInputsFixture


class CleanCloneCoreTest(unittest.TestCase):
    def test_readme_documents_limitations_and_core_contracts(self) -> None:
        readme_path = Path(__file__).parent.parent / "README.md"
        content = readme_path.read_text(encoding="utf-8")
        self.assertIn("MAE OOF Pós-Seleção", content)
        self.assertIn("animais novos", content)
        self.assertIn("CUDA", content)
        self.assertIn("make test", content)
        self.assertIn("confirm-features", content)
        self.assertIn("confirm-approach", content)
        self.assertIn("confirm-diagnostics", content)

    def test_makefile_delegates_all_shortcuts_to_main_py(self) -> None:
        makefile_path = Path(__file__).parent.parent / "Makefile"
        content = makefile_path.read_text(encoding="utf-8")
        self.assertIn("main.py setup", content)
        self.assertIn("main.py reproduce", content)
        self.assertIn("main.py inputs", content)
        self.assertIn("main.py feature-selection", content)
        self.assertIn("main.py confirm-features", content)
        self.assertIn("main.py baselines", content)
        self.assertIn("main.py compare-baselines", content)
        self.assertIn("main.py confirm-approach", content)
        self.assertIn("main.py tuning", content)
        self.assertIn("main.py confirm-diagnostics", content)

    def test_dry_run_displays_graph_and_blocks_on_unconfirmed_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = CuratedInputsFixture(Path(directory), sample_count=132)
            stdout, stderr = io.StringIO(), io.StringIO()
            code = main(
                ["reproduce", "--dry-run", "--config", str(fixture.config_path)],
                stdout=stdout,
                stderr=stderr,
                report_provenance=FixedReportProvenance(),
            )
            output = stdout.getvalue()
            self.assertEqual(code, 1)
            self.assertIn("reproduction_plan:", output)
            self.assertIn("[stage] inputs: rebuild", output)
            self.assertIn("[gate] confirm-features: blocked", output)
            self.assertIn("reproduction: blocked by", output)


if __name__ == "__main__":
    unittest.main()
