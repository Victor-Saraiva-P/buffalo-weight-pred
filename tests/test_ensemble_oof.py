from __future__ import annotations

import unittest

from buffalo_weight.ensemble_oof import evaluate_equal_weight_ensembles


def prediction_rows(predictions: list[float]) -> list[dict[str, str]]:
    return [
        {
            "file_name": f"animal-{index}",
            "fold": str(index % 2 + 1),
            "weight": str(actual),
            "y_pred": str(predicted),
        }
        for index, (actual, predicted) in enumerate(
            zip([100.0, 200.0, 300.0, 400.0], predictions, strict=True)
        )
    ]


class EnsembleOofTest(unittest.TestCase):
    def test_equal_weight_ensemble_can_cancel_opposite_errors(self) -> None:
        predictions = {
            "low": prediction_rows([90.0, 190.0, 290.0, 390.0]),
            "high": prediction_rows([110.0, 210.0, 310.0, 410.0]),
        }

        rows = evaluate_equal_weight_ensembles(predictions)

        self.assertEqual(rows[0]["ensemble"], "high+low")
        self.assertEqual(float(rows[0]["mae"]), 0.0)

    def test_rejects_models_with_different_animals(self) -> None:
        predictions = {"one": prediction_rows([100.0] * 4), "two": prediction_rows([100.0] * 4)[:-1]}

        with self.assertRaisesRegex(ValueError, "identical file_name"):
            evaluate_equal_weight_ensembles(predictions)

    def test_rejects_empty_model_collection(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one model"):
            evaluate_equal_weight_ensembles({})


if __name__ == "__main__":
    unittest.main()
