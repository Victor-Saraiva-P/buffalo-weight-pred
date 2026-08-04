from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from buffalo_weight.report_cli import main
from tests.fake_feature_confirmation import FixedFeatureConfirmationEnvironment
from tests.fake_feature_selection import FixedFeatureEvidenceRunner
from tests.fake_report_provenance import FixedFeatureSelectionProvenance, FixedReportProvenance
from tests.report_inputs_fixture import CuratedInputsFixture


class FeatureConfirmationCliTest(unittest.TestCase):
    def test_valid_human_contract_is_promoted_and_releases_baselines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = CuratedInputsFixture(Path(directory))
            contract_path, report_path = _prepare_human_review(fixture, ("area", "perimeter"))
            result, stdout, stderr = _run_confirmation(fixture, contract_path, report_path)
            self.assertEqual(result, 0, stderr)
            self.assertIn("feature_confirmation: confirmed", stdout)
            _assert_confirmed_package(self, fixture)
            status, gate_stdout, gate_stderr = _run_baselines_gate(fixture)
            self.assertEqual(status, 0, gate_stderr)
            self.assertIn("baselines: released", gate_stdout)

    def test_baselines_gate_reports_blocked_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = CuratedInputsFixture(Path(directory))

            result, stdout, stderr = _run_baselines_gate(fixture)

            self.assertEqual(result, 0, stderr)
            self.assertIn("baselines: blocked", stdout)
            self.assertIn("manifest.json", stdout)
            self.assertFalse((fixture.root / "evidence").exists())

            execution_stdout, execution_stderr = io.StringIO(), io.StringIO()
            execution = main(
                ["baselines", "--config", str(fixture.config_path)],
                stdout=execution_stdout, stderr=execution_stderr,
            )
            self.assertEqual(execution, 1)
            self.assertIn("confirmed feature gate was blocked", execution_stderr.getvalue())

    def test_report_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = CuratedInputsFixture(Path(directory))
            contract_path, report_path = _prepare_human_review(fixture, ("area",))
            contract = json.loads(contract_path.read_text())
            contract["report_sha256"] = "0" * 64
            contract_path.write_text(json.dumps(contract))

            result, _, stderr = _run_confirmation(fixture, contract_path, report_path)

            self.assertEqual(result, 1)
            self.assertIn("report_sha256 was", stderr)
            self.assertIn("expected", stderr)

    def test_confirmation_dry_run_reports_gate_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = CuratedInputsFixture(Path(directory))
            contract_path, report_path = _prepare_human_review(fixture, ("area",))
            result, stdout, stderr = _run_confirmation(
                fixture, contract_path, report_path, dry_run=True,
            )
            self.assertEqual(result, 0, stderr)
            self.assertIn("feature_confirmation: released", stdout)
            self.assertFalse((fixture.root / "evidence").exists())

    def test_confirmation_dry_run_explains_a_blocked_gate_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = CuratedInputsFixture(Path(directory))
            contract_path, report_path = _prepare_human_review(fixture, ("area",))
            contract = json.loads(contract_path.read_text())
            contract["report_sha256"] = "0" * 64
            contract_path.write_text(json.dumps(contract))
            result, stdout, stderr = _run_confirmation(
                fixture, contract_path, report_path, dry_run=True,
            )
            self.assertEqual(result, 0, stderr)
            self.assertIn("feature_confirmation: blocked", stdout)
            self.assertIn("report_sha256 was", stdout)
            self.assertFalse((fixture.root / "evidence").exists())

    def test_nonofficial_feature_evidence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = CuratedInputsFixture(Path(directory))
            contract_path, report_path = _prepare_human_review(fixture, ("area",))
            manifest_path = fixture.root / "generated" / "report" / "feature_selection" / (
                "manifest.json"
            )
            manifest = json.loads(manifest_path.read_text())
            manifest["execution"] = {
                "random_forest_device": "cpu", "dense_device": "cpu", "official": False,
            }
            manifest_path.write_text(json.dumps(manifest))

            result, _, stderr = _run_confirmation(fixture, contract_path, report_path)

            self.assertEqual(result, 1)
            self.assertIn("feature-selection stage status was 'obsolete'", stderr)
            self.assertIn("expected reusable reviewed evidence", stderr)

    def test_invalid_selected_features_are_rejected_with_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = CuratedInputsFixture(Path(directory))
            contract_path, report_path = _prepare_human_review(fixture, ("area", "perimeter"))
            original_report = report_path.read_text()
            cases = _invalid_feature_cases(original_report)
            for selected, report_text, offending, expected in cases:
                with self.subTest(selected=selected):
                    _write_human_contract(contract_path, report_path, selected, report_text)
                    result, _, stderr = _run_confirmation(fixture, contract_path, report_path)
                    self.assertEqual(result, 1)
                    self.assertIn(offending, stderr)
                    self.assertIn(expected, stderr)

    def test_dirty_worktree_rejects_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = CuratedInputsFixture(Path(directory))
            contract_path, report_path = _prepare_human_review(fixture, ("area",))
            environment = FixedFeatureConfirmationEnvironment((" M reviewed_report.md",))

            result, _, stderr = _run_confirmation(
                fixture, contract_path, report_path, environment
            )

            self.assertEqual(result, 1)
            self.assertIn("worktree changes were (' M reviewed_report.md',)", stderr)
            self.assertIn("expected a clean worktree", stderr)

    def test_provisional_or_tampered_confirmation_blocks_baselines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = CuratedInputsFixture(Path(directory))
            contract_path, report_path = _prepare_human_review(fixture, ("area",))
            self.assertEqual(_run_confirmation(fixture, contract_path, report_path)[0], 0)
            confirmed_dir = _confirmed_feature_dir(fixture)
            confirmed_contract = confirmed_dir / "shared_feature_contract.json"
            original = confirmed_contract.read_text()
            contract = json.loads(original)
            contract["status"] = "provisional"
            confirmed_contract.write_text(json.dumps(contract))
            self.assertIn("baselines: blocked", _run_baselines_gate(fixture)[1])
            confirmed_contract.write_text(original)
            report = confirmed_dir / "feature_selection_report.md"
            report.write_text(f"{report.read_text()}\ntampered\n")
            self.assertIn("baselines: blocked", _run_baselines_gate(fixture)[1])

    def test_confirmed_source_provenance_tampering_blocks_baselines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = CuratedInputsFixture(Path(directory))
            contract_path, report_path = _prepare_human_review(fixture, ("area",))
            self.assertEqual(_run_confirmation(fixture, contract_path, report_path)[0], 0)
            manifest_path = _confirmed_feature_dir(fixture) / "manifest.json"
            original = manifest_path.read_text()
            for field, value in (("source_commit", "g" * 40),
                                 ("source_feature_selection_manifest_sha256", "f" * 64)):
                with self.subTest(field=field):
                    manifest = json.loads(original)
                    manifest[field] = value
                    manifest_path.write_text(json.dumps(manifest))
                    status, stdout, _ = _run_baselines_gate(fixture)
                    self.assertEqual(status, 0)
                    self.assertIn("baselines: blocked", stdout)
                    manifest_path.write_text(original)


def _prepare_human_review(
    fixture: CuratedInputsFixture, selected_features: tuple[str, ...]
) -> tuple[Path, Path]:
    provenance = FixedReportProvenance()
    _build_provisional_inputs(fixture, provenance)
    _build_provisional_feature_evidence(fixture, provenance)
    return _write_review_inputs(fixture, selected_features)


def _build_provisional_inputs(
    fixture: CuratedInputsFixture, provenance: FixedReportProvenance,
) -> None:
    result = main(
        ["inputs", "--config", str(fixture.config_path)], stdout=io.StringIO(),
        stderr=io.StringIO(), report_provenance=provenance,
    )
    if result != 0:
        raise AssertionError(f"inputs setup returned {result}; expected success")


def _build_provisional_feature_evidence(
    fixture: CuratedInputsFixture, provenance: FixedReportProvenance,
) -> None:
    result = main(
        ["feature-selection", "--config", str(fixture.config_path)],
        stdout=io.StringIO(), stderr=io.StringIO(), report_provenance=provenance,
        feature_evidence_runner=FixedFeatureEvidenceRunner(),
        feature_selection_provenance=FixedFeatureSelectionProvenance(),
    )
    if result != 0:
        raise AssertionError(f"feature-selection setup returned {result}; expected success")


def _write_review_inputs(
    fixture: CuratedInputsFixture, selected_features: tuple[str, ...],
) -> tuple[Path, Path]:
    report_path = fixture.root / "reviewed_report.md"
    source_report = fixture.root / "generated" / "report" / "feature_selection" / (
        "feature_selection_report.md"
    )
    report_text = source_report.read_text().replace("- Status: pendente", "- Status: revisado")
    report_text = report_text.replace("não preenchidas", "revisadas")
    report_text = report_text.replace("não preenchido", "registrado no contrato")
    contract_path = fixture.root / "human_feature_contract.json"
    _write_human_contract(contract_path, report_path, selected_features, report_text)
    return contract_path, report_path


def _write_human_contract(
    contract_path: Path, report_path: Path, selected_features: tuple[str, ...],
    report_text: str,
) -> None:
    report_path.write_text(report_text)
    contract = {
        "schema_version": 1,
        "status": "confirmed",
        "selected_features": list(selected_features),
        "standardization": "fit within each permitted training partition",
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "human_decision": {
            "decision_url": "https://github.com/example/repository/issues/1#issuecomment-1",
            "reviewer": "Researcher",
            "reviewed_at": "2026-08-04",
        },
    }
    contract_path.write_text(json.dumps(contract))


def _reviewed_report(feature_mentions: str) -> str:
    return (
        "# Relatório revisado\n\n"
        f"Features consideradas: {feature_mentions}.\n\n"
        "## Registro de revisão humana\n\n"
        "- Status: revisado\n"
        "- Interpretações aceitas, corrigidas ou rejeitadas: revisadas\n"
        "- Conjunto Compartilhado de Features confirmado: registrado no contrato\n"
    )


def _invalid_feature_cases(
    original_report: str,
) -> tuple[tuple[tuple[str, ...], str, str, str], ...]:
    return (
        (("not_a_feature",), original_report, "not_a_feature", "26 candidate features"),
        (("area", "area"), original_report, "area", "unique features"),
        (("perimeter", "area"), original_report, "perimeter", "candidate order"),
        (("perimeter",), _reviewed_report("`area`"), "perimeter",
         "present in the reviewed report"),
    )


def _confirmed_feature_dir(fixture: CuratedInputsFixture) -> Path:
    """Locate test evidence.

    Example: tampering tests edit the confirmed manifest.
    """
    return fixture.root / "evidence" / "confirmed" / "feature_selection" / "v1"


def _assert_confirmed_package(
    test_case: unittest.TestCase, fixture: CuratedInputsFixture,
) -> None:
    confirmed_dir = _confirmed_feature_dir(fixture)
    contract_path = confirmed_dir / "shared_feature_contract.json"
    confirmed_contract = json.loads(contract_path.read_text())
    manifest = json.loads((confirmed_dir / "manifest.json").read_text())
    test_case.assertEqual(confirmed_contract["selected_features"], ["area", "perimeter"])
    test_case.assertEqual(manifest["status"], "confirmed")
    test_case.assertEqual(
        manifest["decision"]["sha256"], hashlib.sha256(contract_path.read_bytes()).hexdigest(),
    )
    test_case.assertTrue((confirmed_dir / "source_feature_selection_manifest.json").is_file())


def _run_confirmation(
    fixture: CuratedInputsFixture, contract_path: Path, report_path: Path,
    environment: FixedFeatureConfirmationEnvironment | None = None,
    dry_run: bool = False,
) -> tuple[int, str, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    dry_run_argument = ["--dry-run"] if dry_run else []
    result = main(
        ["confirm-features", *dry_run_argument, "--config", str(fixture.config_path),
         "--contract", str(contract_path), "--report", str(report_path)],
        stdout=stdout, stderr=stderr,
        feature_confirmation_environment=environment or FixedFeatureConfirmationEnvironment(),
        feature_selection_provenance=FixedFeatureSelectionProvenance(),
    )
    return result, stdout.getvalue(), stderr.getvalue()


def _run_baselines_gate(fixture: CuratedInputsFixture) -> tuple[int, str, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    result = main(
        ["baselines", "--dry-run", "--config", str(fixture.config_path)],
        stdout=stdout, stderr=stderr,
    )
    return result, stdout.getvalue(), stderr.getvalue()


if __name__ == "__main__":
    unittest.main()
