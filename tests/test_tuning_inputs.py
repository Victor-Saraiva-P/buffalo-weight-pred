from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from buffalo_weight.reproduction_config import load_report_contract
from buffalo_weight.tuning_inputs import validate_tuning_gate_and_contract
from tests.baseline_comparison_fixture import prepared_comparison_fixture
from tests.report_inputs_fixture import CuratedInputsFixture


class TuningInputsTest(unittest.TestCase):
    def test_unconfirmed_approach_gate_raises_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = CuratedInputsFixture(Path(directory))
            contract = load_report_contract(fixture.config_path)
            with self.assertRaisesRegex(ValueError, "confirmed approach gate was blocked"):
                validate_tuning_gate_and_contract(contract)

    def test_confirmed_approach_gate_returns_choice_and_variations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = prepared_comparison_fixture(Path(directory))
            _setup_confirmed_approach(fixture, "random_forest", "random_forest_baseline", 3)
            contract = load_report_contract(fixture.config_path)
            approach, baseline, budget, features, variations = validate_tuning_gate_and_contract(contract)
            self.assertEqual(approach, "random_forest")
            self.assertEqual(baseline, "random_forest_baseline")
            self.assertEqual(budget, 3)
            self.assertIsNotNone(features)
            self.assertEqual(len(variations), 3)

    def test_cnn_approach_does_not_require_features(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = CuratedInputsFixture(Path(directory))
            _setup_confirmed_approach(fixture, "compact_cnn", "compact_cnn", 2)
            contract = load_report_contract(fixture.config_path)
            approach, baseline, budget, features, variations = validate_tuning_gate_and_contract(contract)
            self.assertEqual(approach, "compact_cnn")
            self.assertEqual(baseline, "compact_cnn")
            self.assertEqual(budget, 2)
            self.assertIsNone(features)
            self.assertEqual(len(variations), 2)


def _setup_confirmed_approach(
    fixture: CuratedInputsFixture, approach: str, baseline_config: str, budget: int,
) -> None:
    path = fixture.root / "evidence" / "confirmed" / "approach_selection" / "v1"
    path.mkdir(parents=True, exist_ok=True)
    report_text = f"# Report\n## Registro de revisão humana\n- Status: revisado\nAbordagem confirmada: {approach}\n"
    (path / "approach_selection_report.md").write_text(report_text)
    (path / "baseline_metrics.csv").write_text("configuration,approach,role,scope,fold,population,samples,mae_kg,rmse_kg,bias_kg,r2\n")
    import hashlib
    report_sha = hashlib.sha256(report_text.encode()).hexdigest()
    contract = {
        "schema_version": 1, "status": "confirmed", "selected_approach": approach,
        "baseline_configuration": baseline_config, "maximum_tuning_variations": budget,
        "source_report_sha256": report_sha,
        "human_decision": {
            "decision_url": "https://github.com/example/issues/22#comment",
            "reviewer": "Reviewer", "reviewed_at": "2026-08-06",
        },
    }
    (path / "selected_approach.json").write_text(json.dumps(contract))
    (path / "source_baseline_comparison_manifest.json").write_text(json.dumps({
        "source_commit": "e456bac" + "0" * 33, "inputs": {},
    }))
    baseline_metrics_sha = hashlib.sha256((path / "baseline_metrics.csv").read_bytes()).hexdigest()
    from buffalo_weight.approach_confirmation_manifest import build_confirmed_approach_manifest
    manifest = build_confirmed_approach_manifest(path, load_report_contract(fixture.config_path), contract)
    (path / "manifest.json").write_text(json.dumps(manifest))


def _setup_confirmed_features(fixture: CuratedInputsFixture) -> None:
    path = fixture.root / "evidence" / "confirmed" / "feature_selection" / "v1"
    path.mkdir(parents=True, exist_ok=True)
    report_text = "# Feature Report\n## Registro de revisão humana\n- Status: revisado\nRevisado.\n"
    (path / "feature_selection_report.md").write_text(report_text)
    import hashlib
    report_sha = hashlib.sha256(report_text.encode()).hexdigest()
    contract = {
        "schema_version": 1, "status": "confirmed",
        "selected_features": ["area", "perimeter", "solidity"],
        "standardization": "fit within each permitted training partition",
        "report_sha256": report_sha,
        "human_decision": {
            "decision_url": "https://github.com/example/issues/16#comment",
            "reviewer": "Reviewer", "reviewed_at": "2026-08-04",
        },
    }
    (path / "shared_feature_contract.json").write_text(json.dumps(contract))
    (path / "source_feature_selection_manifest.json").write_text(json.dumps({
        "source_commit": "e456bac" + "0" * 33,
        "execution": {"device": "cpu", "deterministic": True, "official": True, "python_version": "3.10"},
        "inputs": {},
    }))
    (path / "feature_predictive_evidence.csv").write_text("baseline_name,configuration,feature_name,fold,mae_kg,rmse_kg,r2\n")
    (path / "feature_redundancy.csv").write_text("feature_a,feature_b,pearson_r,spearman_rho\n")
    (path / "permutation_effects.png").write_bytes(b"png")
    (path / "redundancy_heatmap.png").write_bytes(b"png")
    (path / "removal_heatmap.png").write_bytes(b"png")
    from buffalo_weight.feature_confirmation_manifest import build_confirmed_manifest
    manifest = build_confirmed_manifest(path, load_report_contract(fixture.config_path), contract)
    (path / "manifest.json").write_text(json.dumps(manifest))


if __name__ == "__main__":
    unittest.main()
