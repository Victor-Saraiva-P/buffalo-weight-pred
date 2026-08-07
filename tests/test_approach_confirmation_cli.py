from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from buffalo_weight.approach_confirmation import (
    approach_gate_status,
    require_approach_gate,
)
from buffalo_weight.dense_baseline_stage import DenseBaselineDependencies
from buffalo_weight.report_cli import main
from buffalo_weight.reproduction_config import load_report_contract
from tests.baseline_comparison_fixture import prepared_comparison_fixture
from tests.fake_baseline_comparison import FixedBaselineComparisonProvenance
from tests.fake_baseline_provenance import FixedBaselineProvenance
from tests.fake_compact_cnn import FixedCompactCnnProvenance
from tests.fake_dense_baseline import (
    FixedCudaRuntimeProbe,
    FixedDenseBaselineProvenance,
)
from tests.fake_feature_confirmation import FixedFeatureConfirmationEnvironment
from tests.fake_report_provenance import FixedReportProvenance
from tests.fake_resnet_baseline import FixedResNetBaselineProvenance
from tests.report_inputs_fixture import CuratedInputsFixture

CANDIDATES = (
    ("random_forest", "random_forest_baseline"),
    ("dense_feature_network", "dense"),
    ("compact_cnn", "compact_cnn"),
    ("resnet18", "resnet18_pretrained_partial"),
)


class ApproachConfirmationCliTest(unittest.TestCase):
    def test_valid_human_contract_for_each_candidate_is_promoted_and_releases_gate(self) -> None:
        for approach, baseline_config in CANDIDATES:
            with self.subTest(approach=approach):
                with tempfile.TemporaryDirectory() as directory:
                    fixture = _prepare_baseline_comparison(Path(directory))
                    contract_path, report_path = _prepare_human_review(
                        fixture, approach, baseline_config,
                    )
                    result, stdout, stderr = _run_confirmation(fixture, contract_path, report_path)
                    self.assertEqual(result, 0, stderr)
                    self.assertIn("approach_confirmation: confirmed", stdout)
                    _assert_confirmed_package(self, fixture, approach, baseline_config)
                    contract = load_report_contract(fixture.config_path)
                    self.assertEqual(approach_gate_status(contract), "released")
                    self.assertEqual(require_approach_gate(contract), (approach, baseline_config, 3))

    def test_approach_gate_reports_blocked_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _prepare_baseline_comparison(Path(directory))
            contract = load_report_contract(fixture.config_path)
            self.assertTrue(approach_gate_status(contract).startswith("blocked"))
            self.assertFalse((fixture.root / "evidence" / "confirmed" / "approach_selection").exists())
            with self.assertRaisesRegex(ValueError, "confirmed approach gate was blocked"):
                require_approach_gate(contract)

    def test_report_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _prepare_baseline_comparison(Path(directory))
            contract_path, report_path = _prepare_human_review(
                fixture, "random_forest", "random_forest_baseline",
            )
            contract = json.loads(contract_path.read_text())
            contract["source_report_sha256"] = "0" * 64
            contract_path.write_text(json.dumps(contract))

            result, _, stderr = _run_confirmation(fixture, contract_path, report_path)
            self.assertEqual(result, 1)
            self.assertIn("source_report_sha256 was", stderr)

    def test_unknown_approach_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _prepare_baseline_comparison(Path(directory))
            contract_path, report_path = _prepare_human_review(
                fixture, "random_forest", "random_forest_baseline",
            )
            contract = json.loads(contract_path.read_text())
            contract["selected_approach"] = "unknown_approach"
            contract_path.write_text(json.dumps(contract))
            result, _, stderr = _run_confirmation(fixture, contract_path, report_path)
            self.assertEqual(result, 1)
            self.assertIn("selected_approach was 'unknown_approach'", stderr)

    def test_incompatible_config_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _prepare_baseline_comparison(Path(directory))
            contract_path, report_path = _prepare_human_review(
                fixture, "random_forest", "random_forest_baseline",
            )
            contract = json.loads(contract_path.read_text())
            contract["baseline_configuration"] = "dense"
            contract_path.write_text(json.dumps(contract))
            result, _, stderr = _run_confirmation(fixture, contract_path, report_path)
            self.assertEqual(result, 1)
            self.assertIn("baseline_configuration was 'dense'", stderr)

    def test_invalid_tuning_budget_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _prepare_baseline_comparison(Path(directory))
            contract_path, report_path = _prepare_human_review(
                fixture, "random_forest", "random_forest_baseline",
            )
            for invalid_budget in (4, -1, "three"):
                with self.subTest(budget=invalid_budget):
                    contract = json.loads(contract_path.read_text())
                    contract["maximum_tuning_variations"] = invalid_budget
                    contract_path.write_text(json.dumps(contract))
                    result, _, stderr = _run_confirmation(fixture, contract_path, report_path)
                    self.assertEqual(result, 1)
                    self.assertIn("maximum_tuning_variations was", stderr)

    def test_unreviewed_status_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _prepare_baseline_comparison(Path(directory))
            contract_path, report_path = _prepare_human_review(
                fixture, "random_forest", "random_forest_baseline",
            )
            pending = report_path.read_text().replace("- Status: revisado", "- Status: pendente")
            report_path.write_text(pending)
            _rehash_report(contract_path, report_path)
            result, _, stderr = _run_confirmation(fixture, contract_path, report_path)
            self.assertEqual(result, 1)
            self.assertIn("Registro de revisão humana", stderr)

    def test_unreviewed_placeholders_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _prepare_baseline_comparison(Path(directory))
            contract_path, report_path = _prepare_human_review(
                fixture, "random_forest", "random_forest_baseline",
            )
            placeholders = report_path.read_text().replace("revisadas", "não preenchidas")
            report_path.write_text(placeholders)
            _rehash_report(contract_path, report_path)
            result, _, stderr = _run_confirmation(fixture, contract_path, report_path)
            self.assertEqual(result, 1)
            self.assertIn("placeholders", stderr)

    def test_dirty_worktree_rejects_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _prepare_baseline_comparison(Path(directory))
            contract_path, report_path = _prepare_human_review(
                fixture, "random_forest", "random_forest_baseline",
            )
            environment = FixedFeatureConfirmationEnvironment((" M reviewed_report.md",))
            result, _, stderr = _run_confirmation(
                fixture, contract_path, report_path, environment=environment,
            )
            self.assertEqual(result, 1)
            self.assertIn("worktree changes were", stderr)

    def test_confirmation_dry_run_reports_gate_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _prepare_baseline_comparison(Path(directory))
            contract_path, report_path = _prepare_human_review(
                fixture, "random_forest", "random_forest_baseline",
            )
            result, stdout, stderr = _run_confirmation(
                fixture, contract_path, report_path, dry_run=True,
            )
            self.assertEqual(result, 0, stderr)
            self.assertIn("approach_confirmation: released", stdout)
            self.assertFalse((fixture.root / "evidence" / "confirmed" / "approach_selection").exists())

    def test_confirmation_dry_run_explains_a_blocked_gate_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _prepare_baseline_comparison(Path(directory))
            contract_path, report_path = _prepare_human_review(
                fixture, "random_forest", "random_forest_baseline",
            )
            contract = json.loads(contract_path.read_text())
            contract["source_report_sha256"] = "0" * 64
            contract_path.write_text(json.dumps(contract))
            result, stdout, stderr = _run_confirmation(
                fixture, contract_path, report_path, dry_run=True,
            )
            self.assertEqual(result, 0, stderr)
            self.assertIn("approach_confirmation: blocked", stdout)
            self.assertIn("source_report_sha256 was", stdout)
            self.assertFalse((fixture.root / "evidence" / "confirmed" / "approach_selection").exists())

    def test_provisional_confirmation_blocks_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _prepare_baseline_comparison(Path(directory))
            contract_path, report_path = _prepare_human_review(
                fixture, "random_forest", "random_forest_baseline",
            )
            self.assertEqual(_run_confirmation(fixture, contract_path, report_path)[0], 0)
            confirmed_dir = _confirmed_approach_dir(fixture)
            confirmed_contract = confirmed_dir / "selected_approach.json"
            contract_dict = json.loads(confirmed_contract.read_text())
            contract_dict["status"] = "provisional"
            confirmed_contract.write_text(json.dumps(contract_dict))
            rep_contract = load_report_contract(fixture.config_path)
            self.assertTrue(approach_gate_status(rep_contract).startswith("blocked"))

    def test_tampered_report_blocks_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _prepare_baseline_comparison(Path(directory))
            contract_path, report_path = _prepare_human_review(
                fixture, "random_forest", "random_forest_baseline",
            )
            self.assertEqual(_run_confirmation(fixture, contract_path, report_path)[0], 0)
            confirmed_dir = _confirmed_approach_dir(fixture)
            report = confirmed_dir / "approach_selection_report.md"
            report.write_text(f"{report.read_text()}\ntampered\n")
            rep_contract = load_report_contract(fixture.config_path)
            self.assertTrue(approach_gate_status(rep_contract).startswith("blocked"))

    def test_source_provenance_tampering_blocks_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _prepare_baseline_comparison(Path(directory))
            contract_path, report_path = _prepare_human_review(
                fixture, "random_forest", "random_forest_baseline",
            )
            self.assertEqual(_run_confirmation(fixture, contract_path, report_path)[0], 0)
            manifest_path = _confirmed_approach_dir(fixture) / "manifest.json"
            original = manifest_path.read_text()

            for field, value in (("source_commit", "g" * 40),
                                 ("source_baseline_comparison_manifest_sha256", "f" * 64)):
                with self.subTest(field=field):
                    manifest = json.loads(original)
                    manifest[field] = value
                    manifest_path.write_text(json.dumps(manifest))
                    rep_contract = load_report_contract(fixture.config_path)
                    self.assertTrue(approach_gate_status(rep_contract).startswith("blocked"))
                    manifest_path.write_text(original)


def _prepare_baseline_comparison(root: Path) -> CuratedInputsFixture:
    fixture = prepared_comparison_fixture(root)
    stdout, stderr = io.StringIO(), io.StringIO()
    result = main(
        ["compare-baselines", "--config", str(fixture.config_path)],
        stdout=stdout, stderr=stderr,
        report_provenance=FixedReportProvenance(),
        baseline_provenance=FixedBaselineProvenance(),
        dense_baseline_dependencies=DenseBaselineDependencies(
            provenance=FixedDenseBaselineProvenance(), runtime_probe=FixedCudaRuntimeProbe(),
        ),
        compact_cnn_provenance=FixedCompactCnnProvenance(),
        resnet_baseline_provenance=FixedResNetBaselineProvenance(),
        baseline_comparison_provenance=FixedBaselineComparisonProvenance(),
    )
    if result != 0:
        raise AssertionError(f"compare-baselines returned {result}; expected success. stderr: {stderr.getvalue()}")
    return fixture


def _prepare_human_review(
    fixture: CuratedInputsFixture, approach: str, baseline_config: str,
) -> tuple[Path, Path]:
    report_path = _write_reviewed_report(fixture, approach)
    contract_path = _write_human_contract(fixture, approach, baseline_config, report_path)
    return contract_path, report_path


def _write_reviewed_report(fixture: CuratedInputsFixture, approach: str) -> Path:
    report_path = fixture.root / "reviewed_report.md"
    source_report = fixture.root / "generated" / "report" / "approach_selection" / (
        "approach_selection_report.md"
    )
    report_text = source_report.read_text().replace("Status: pendente", "Status: revisado")
    report_text = report_text.replace("não preenchidas", "revisadas")
    report_text = report_text.replace("não preenchida", "registrada no contrato")
    report_text += f"\nAbordagem confirmada: {approach}\n"
    report_path.write_text(report_text)
    return report_path


def _write_human_contract(
    fixture: CuratedInputsFixture, approach: str, baseline_config: str, report_path: Path,
) -> Path:
    contract_path = fixture.root / "human_approach_contract.json"
    contract = {
        "schema_version": 1,
        "status": "confirmed",
        "selected_approach": approach,
        "baseline_configuration": baseline_config,
        "maximum_tuning_variations": 3,
        "source_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "human_decision": {
            "decision_url": "https://github.com/example/repository/issues/22#issuecomment-1",
            "reviewer": "Researcher",
            "reviewed_at": "2026-08-06",
        },
    }
    contract_path.write_text(json.dumps(contract))
    return contract_path


def _rehash_report(contract_path: Path, report_path: Path) -> None:
    contract = json.loads(contract_path.read_text())
    contract["source_report_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    contract_path.write_text(json.dumps(contract))


def _confirmed_approach_dir(fixture: CuratedInputsFixture) -> Path:
    """Return confirmed approach selection directory.

    Example: ``_confirmed_approach_dir(fixture)`` points under evidence.
    """
    confirmed_path = fixture.root / "evidence" / "confirmed" / "approach_selection" / "v1"
    return confirmed_path


def _assert_confirmed_package(
    test_case: unittest.TestCase, fixture: CuratedInputsFixture,
    expected_approach: str, expected_config: str,
) -> None:
    confirmed_dir = _confirmed_approach_dir(fixture)
    contract_path = confirmed_dir / "selected_approach.json"
    confirmed_contract = json.loads(contract_path.read_text())
    manifest = json.loads((confirmed_dir / "manifest.json").read_text())
    test_case.assertEqual(confirmed_contract["selected_approach"], expected_approach)
    test_case.assertEqual(confirmed_contract["baseline_configuration"], expected_config)
    test_case.assertEqual(confirmed_contract["maximum_tuning_variations"], 3)
    test_case.assertEqual(manifest["status"], "confirmed")
    test_case.assertEqual(
        manifest["decision"]["sha256"], hashlib.sha256(contract_path.read_bytes()).hexdigest(),
    )
    test_case.assertTrue((confirmed_dir / "source_baseline_comparison_manifest.json").is_file())


def _run_confirmation(
    fixture: CuratedInputsFixture, contract_path: Path, report_path: Path,
    environment: FixedFeatureConfirmationEnvironment | None = None,
    dry_run: bool = False,
) -> tuple[int, str, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    dry_run_arg = ["--dry-run"] if dry_run else []
    result = main(
        ["confirm-approach", *dry_run_arg, "--config", str(fixture.config_path),
         "--contract", str(contract_path), "--report", str(report_path)],
        stdout=stdout, stderr=stderr,
        feature_confirmation_environment=environment or FixedFeatureConfirmationEnvironment(),
        baseline_comparison_provenance=FixedBaselineComparisonProvenance(),
    )
    return result, stdout.getvalue(), stderr.getvalue()


if __name__ == "__main__":
    unittest.main()
