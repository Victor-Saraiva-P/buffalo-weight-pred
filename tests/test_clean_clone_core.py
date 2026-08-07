"""Integration tests for clean clone reproducible core execution."""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

from buffalo_weight.report_cli import main
from buffalo_weight.report_inputs import clean_reconstructible_stage
from buffalo_weight.report_reproduction import run_report_reproduction
from buffalo_weight.reproduction_config import load_report_contract
from tests.fake_baseline_comparison import FixedBaselineComparisonProvenance
from buffalo_weight.environment_contract import APPROVED_DEPENDENCIES
from tests.fake_baseline_comparison import FixedBaselineComparisonProvenance
from tests.fake_baseline_provenance import FixedBaselineProvenance
from tests.fake_compact_cnn import FixedCompactCnnProvenance, RecordingCompactCnnAdapter
from tests.fake_dense_baseline import (
    FixedCudaRuntimeProbe,
    FixedDenseBaselineProvenance,
    FixedDenseBaselineRunner,
)
from tests.fake_feature_confirmation import FixedFeatureConfirmationEnvironment
from tests.fake_feature_evaluation import RecordingFeatureBaseline
from tests.fake_feature_selection import FixedFeatureEvidenceRunner
from tests.fake_report_provenance import FixedReportProvenance
from tests.fake_resnet_baseline import FixedResNetBaselineProvenance, FixedResNetBaselineRunner
from tests.fake_setup_services import FakePackageGateway, setup_services
from tests.fake_tuning_provenance import FixedTuningProvenance
from tests.report_inputs_fixture import CuratedInputsFixture
from tests.test_reproduction_cli import (
    _make_repro_deps,
    _prepare_fully_confirmed_environment,
)


class CleanCloneCoreTest(unittest.TestCase):
    """Verify clean clone setup, inputs, dry-run graph, and CLI seam reproduction lifecycle."""

    def test_setup_command_executes_successfully(self) -> None:
        """Verify main.py setup executes environment audit cleanly.

        Example: ``self.test_setup_command_executes_successfully()``
        """
        services = setup_services(FakePackageGateway(dict(APPROVED_DEPENDENCIES)))
        stdout, stderr = io.StringIO(), io.StringIO()
        code = main(["setup"], services=services, stdout=stdout, stderr=stderr)
        self.assertEqual(code, 0)
        self.assertIn("approved dependencies", stdout.getvalue())

    def test_inputs_command_executes_with_fixture(self) -> None:
        """Verify main.py inputs validates inputs and builds index snapshot.

        Example: ``self.test_inputs_command_executes_with_fixture()``
        """
        with tempfile.TemporaryDirectory() as directory:
            fixture = CuratedInputsFixture(Path(directory), sample_count=132)
            stdout, stderr = io.StringIO(), io.StringIO()
            code = main(
                ["inputs", "--config", str(fixture.config_path)],
                stdout=stdout,
                stderr=stderr,
                report_provenance=FixedReportProvenance(),
            )
            self.assertEqual(code, 0)
            self.assertIn("inputs: rebuilt", stdout.getvalue())

    def test_dry_run_shows_full_graph_and_blocks_at_unconfirmed_gate(self) -> None:
        """Verify reproduce --dry-run shows graph and halts at unconfirmed human gate.

        Example: ``self.test_dry_run_shows_full_graph_and_blocks_at_unconfirmed_gate()``
        """
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
            self.assertIn("[gate] confirm-features: blocked", output)

    def test_cli_seam_full_reproduction_resume_invalidation_and_clean(self) -> None:
        """Verify CLI seam reproduction lifecycle: execution, resume, invalidation, and clean.

        Example: ``self.test_cli_seam_full_reproduction_resume_invalidation_and_clean()``
        """
        with tempfile.TemporaryDirectory() as directory:
            fixture = _prepare_fully_confirmed_environment(Path(directory))
            deps = _make_repro_deps()
            contract = load_report_contract(fixture.config_path)

            stdout_first = io.StringIO()
            code_first = run_report_reproduction(contract, False, deps, stdout_first)
            self.assertEqual(code_first, 0)
            self.assertIn("reproduction: complete", stdout_first.getvalue())

            stdout_second = io.StringIO()
            code_second = run_report_reproduction(contract, False, deps, stdout_second)
            self.assertEqual(code_second, 0)

            sel_manifest = fixture.root / "generated" / "report" / "feature_selection" / "manifest.json"
            sel_manifest.write_text(sel_manifest.read_text() + "tampered\n")
            stdout_third = io.StringIO()
            code_third = run_report_reproduction(contract, False, deps, stdout_third)
            self.assertEqual(code_third, 0)
            self.assertIn("feature-selection: rebuilt", stdout_third.getvalue())

            cleaned = clean_reconstructible_stage(contract, "inputs")
            self.assertTrue(len(cleaned) > 0)



if __name__ == "__main__":
    unittest.main()

