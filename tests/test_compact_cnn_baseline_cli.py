from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from buffalo_weight.report_cli import main
from tests.fake_compact_cnn import (
    FailingCompactCnnAdapter,
    FixedCompactCnnProvenance,
    RecordingCompactCnnAdapter,
)
from buffalo_weight.compact_cnn_types import CompactCnnTrainingAdapter
from tests.test_feature_confirmation_cli import _prepare_human_review, _run_confirmation
from tests.report_inputs_fixture import CuratedInputsFixture


class CompactCnnBaselineCliTest(unittest.TestCase):
    def test_cli_builds_isolated_oof_predictions_and_refits_each_fold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _confirmed_fixture(Path(directory))
            adapter = RecordingCompactCnnAdapter()

            result, stdout, stderr = _run_baselines(fixture, adapter)

            self.assertEqual(result, 0, stderr)
            self.assertIn("baselines: released", stdout)
            self.assertIn("compact_cnn: rebuilt", stdout)
            self.assertEqual(len(adapter.selection_calls), 5)
            self.assertEqual(len(adapter.refit_calls), 5)
            self._assert_training_isolation(adapter)
            self._assert_spatial_input_contract(adapter)
            self._assert_train_only_augmentation(adapter)
            self._assert_public_artifacts(fixture)

    def test_cli_reuses_current_artifact_and_detects_recipe_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _confirmed_fixture(Path(directory))
            adapter = RecordingCompactCnnAdapter()
            provenance = FixedCompactCnnProvenance()
            self.assertEqual(_run_baselines(fixture, adapter, provenance)[0], 0)
            selection_count = len(adapter.selection_calls)

            reused = _run_baselines(fixture, adapter, provenance)

            self.assertEqual(reused[0], 0, reused[2])
            self.assertIn("compact_cnn: reusable", reused[1])
            self.assertEqual(len(adapter.selection_calls), selection_count)
            self._assert_mask_change_invalidates(fixture, adapter, provenance)
            changed = FixedCompactCnnProvenance("7" * 64)
            dry_run = _run_baselines(fixture, adapter, changed, "--dry-run")
            self.assertEqual(dry_run[0], 0, dry_run[2])
            self.assertIn("compact_cnn: obsolete", dry_run[1])
            self._assert_output_change_invalidates(fixture, adapter, provenance)

    def test_obsolete_artifact_is_removed_before_a_failed_retrain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _confirmed_fixture(Path(directory))
            self.assertEqual(_run_baselines(fixture, RecordingCompactCnnAdapter())[0], 0)
            output_dir = fixture.root / "generated/report/baselines/compact_cnn"

            result = _run_baselines(
                fixture, FailingCompactCnnAdapter(), FixedCompactCnnProvenance("7" * 64),
            )

            self.assertEqual(result[0], 1)
            self.assertIn("injected compact CNN training failure", result[2])
            self.assertFalse(output_dir.exists())

    def _assert_training_isolation(self, adapter: RecordingCompactCnnAdapter) -> None:
        for (selection, stopping), (refit, epochs), held_out in zip(
            adapter.selection_calls, adapter.refit_calls, adapter.prediction_calls
        ):
            selected_ids = set(selection.sample_ids)
            stopping_ids = set(stopping.sample_ids)
            held_out_ids = set(held_out.sample_ids)
            self.assertTrue(selected_ids.isdisjoint(stopping_ids))
            self.assertTrue((selected_ids | stopping_ids).isdisjoint(held_out_ids))
            self.assertEqual(set(refit.sample_ids), selected_ids | stopping_ids)
            self.assertEqual(epochs, 2)

    def _assert_mask_change_invalidates(
        self, fixture: CuratedInputsFixture, adapter: RecordingCompactCnnAdapter,
        provenance: FixedCompactCnnProvenance,
    ) -> None:
        mask_path = fixture.mask_path(0)
        original_mask = mask_path.read_bytes()
        mask_path.write_bytes(original_mask + b"changed")
        changed = _run_baselines(fixture, adapter, provenance, "--dry-run")
        self.assertIn("compact_cnn: obsolete", changed[1])
        mask_path.write_bytes(original_mask)

    def _assert_output_change_invalidates(
        self, fixture: CuratedInputsFixture, adapter: RecordingCompactCnnAdapter,
        provenance: FixedCompactCnnProvenance,
    ) -> None:
        output = fixture.root / "generated/report/baselines/compact_cnn/fold_metrics.csv"
        output.write_text(f"{output.read_text()}tampered\n")
        tampered = _run_baselines(fixture, adapter, provenance, "--dry-run")
        self.assertIn("compact_cnn: obsolete", tampered[1])

    def _assert_spatial_input_contract(self, adapter: RecordingCompactCnnAdapter) -> None:
        batches = [call[0] for call in adapter.selection_calls]
        self.assertTrue(all(batch.pixels.shape[1:] == (1, 224, 224) for batch in batches))
        self.assertTrue(all(set(map(float, np.unique(batch.pixels))) <= {0.0, 1.0}
                            for batch in batches))

    def _assert_train_only_augmentation(self, adapter: RecordingCompactCnnAdapter) -> None:
        self.assertEqual(len(adapter.augmented_training), 10)
        self.assertEqual(len(adapter.evaluation_pixels), 10)
        changed = []
        for original, augmented in adapter.augmented_training:
            np.testing.assert_array_equal(original.sum(axis=(1, 2, 3)),
                                          augmented.sum(axis=(1, 2, 3)))
            self.assertLessEqual(set(map(float, np.unique(augmented))), {0.0, 1.0})
            self.assertTrue(all(_is_valid_conservative_transform(before, after)
                                for before, after in zip(original, augmented, strict=True)))
            changed.append(not np.array_equal(original, augmented))
        self.assertTrue(any(changed))
        self.assertTrue(all(set(map(float, np.unique(pixels))) <= {0.0, 1.0}
                            for pixels in adapter.evaluation_pixels))

    def _assert_public_artifacts(self, fixture: CuratedInputsFixture) -> None:
        output_dir = fixture.root / "generated" / "report" / "baselines" / "compact_cnn"
        predictions = _read_rows(output_dir / "predictions.csv")
        metrics = _read_rows(output_dir / "fold_metrics.csv")
        manifest = json.loads((output_dir / "manifest.json").read_text())
        self.assertEqual(len(predictions), fixture.sample_count)
        self.assertEqual(len({row["file_name"] for row in predictions}), fixture.sample_count)
        self.assertEqual({row["model"] for row in predictions}, {"compact_cnn"})
        self.assertEqual({row["scope"] for row in metrics}, {"fold", "oof", "category"})
        self.assertEqual(manifest["recipe"]["image_size"], 224)
        self.assertEqual(manifest["recipe"]["optimizer"], "AdamW")
        self.assertEqual(manifest["status"], "complete")
        self.assertIn("deterministic_cuda", manifest["validations"])
        self.assertIn("decision_url", manifest)
        self.assertEqual(manifest["inputs"]["referenced_masks"]["file_count"], 100)
        self.assertNotIn("shared_feature_contract", manifest["inputs"])


def _confirmed_fixture(root: Path) -> CuratedInputsFixture:
    fixture = CuratedInputsFixture(root, sample_count=100)
    contract_path, report_path = _prepare_human_review(fixture, ("area", "perimeter"))
    result, _, stderr = _run_confirmation(fixture, contract_path, report_path)
    if result != 0:
        raise AssertionError(f"feature confirmation returned {result}: {stderr}")
    return fixture


def _run_baselines(
    fixture: CuratedInputsFixture, adapter: CompactCnnTrainingAdapter,
    provenance: FixedCompactCnnProvenance | None = None, *extra: str,
) -> tuple[int, str, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    result = main(
        ["baselines", "--config", str(fixture.config_path), *extra],
        stdout=stdout, stderr=stderr, compact_cnn_adapter=adapter,
        compact_cnn_provenance=provenance or FixedCompactCnnProvenance(),
    )
    return result, stdout.getvalue(), stderr.getvalue()


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    return rows


def _is_valid_conservative_transform(
    original: NDArray[np.float32], augmented: NDArray[np.float32],
) -> bool:
    augmented_origin = _foreground_origin(augmented)
    for candidate in (original, np.flip(original, axis=2)):
        candidate_origin = _foreground_origin(candidate)
        shift_y = augmented_origin[0] - candidate_origin[0]
        shift_x = augmented_origin[1] - candidate_origin[1]
        if abs(shift_y) > 11 or abs(shift_x) > 11:
            continue
        if np.array_equal(_translate_mask(candidate, shift_y, shift_x), augmented):
            return True
    return False


def _foreground_origin(mask: NDArray[np.float32]) -> tuple[int, int]:
    coordinates = np.argwhere(mask[0] > 0)
    origin = (int(coordinates[:, 0].min()), int(coordinates[:, 1].min()))
    return origin


def _translate_mask(
    mask: NDArray[np.float32], shift_y: int, shift_x: int,
) -> NDArray[np.float32]:
    translated = np.roll(mask, shift=(shift_y, shift_x), axis=(1, 2))
    if shift_y > 0:
        translated[:, :shift_y, :] = 0
    elif shift_y < 0:
        translated[:, shift_y:, :] = 0
    if shift_x > 0:
        translated[:, :, :shift_x] = 0
    elif shift_x < 0:
        translated[:, :, shift_x:] = 0
    return translated


if __name__ == "__main__":
    unittest.main()
