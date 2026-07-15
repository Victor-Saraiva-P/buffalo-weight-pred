from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from buffalo_weight.clean_train import clean_training_outputs, parse_model_names


class CleanTrainTest(unittest.TestCase):
    def test_parses_space_and_comma_separated_model_names(self) -> None:
        names = parse_model_names(["model_a,model_b", "model_c"])

        self.assertEqual(names, ["model_a", "model_b", "model_c"])

    def test_cleans_selected_model_and_invalidates_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "train"
            (output_dir / "model_a").mkdir(parents=True)
            (output_dir / "model_b").mkdir()
            (output_dir / "model_comparison.csv").touch()
            (output_dir / "model_comparison.png").touch()

            clean_training_outputs(output_dir, ["model_a"])

            self.assertFalse((output_dir / "model_a").exists())
            self.assertTrue((output_dir / "model_b").exists())
            self.assertFalse((output_dir / "model_comparison.csv").exists())
            self.assertFalse((output_dir / "model_comparison.png").exists())

    def test_cleans_entire_training_output_without_model_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "train"
            (output_dir / "model_a").mkdir(parents=True)

            clean_training_outputs(output_dir, [])

            self.assertFalse(output_dir.exists())

    def test_rejects_model_name_that_escapes_training_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "train"
            output_dir.mkdir()

            with self.assertRaisesRegex(ValueError, "../model.*single directory name"):
                clean_training_outputs(output_dir, ["../model"])


if __name__ == "__main__":
    unittest.main()
