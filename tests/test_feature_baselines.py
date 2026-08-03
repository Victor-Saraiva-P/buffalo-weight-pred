from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from buffalo_weight.dense_feature_adapter import DenseFeatureAdapter
from buffalo_weight.feature_baselines import (
    DENSE_BASELINE_RECIPE,
    DenseFeatureBaseline,
    RandomForestBaseline,
)
from buffalo_weight.feature_evaluation import PredictionPartition, TrainingPartition


class FeatureBaselinesTest(unittest.TestCase):
    def test_random_forest_uses_frozen_recipe_and_is_repeatable(self) -> None:
        partition = training_partition()
        baseline = RandomForestBaseline()
        first = baseline.fit(partition, ("area", "perimeter"))
        second = baseline.fit(partition, ("area", "perimeter"))
        held_out = PredictionPartition(partition.values, partition.sample_ids)
        self.assertEqual(
            baseline.recipe,
            {
                "n_estimators": 500,
                "criterion": "squared_error",
                "bootstrap": True,
                "max_depth": None,
                "min_samples_leaf": 3,
                "min_samples_split": 6,
                "max_features": 0.7,
                "random_state": 44,
            },
        )
        np.testing.assert_array_equal(first.predict(held_out), second.predict(held_out))


@unittest.skipUnless(DenseFeatureAdapter.cuda_available(), "CUDA contract requires a GPU")
class DenseFeatureAdapterCudaTest(unittest.TestCase):
    def test_cuda_step_checkpoint_device_and_repetition(self) -> None:
        adapter = DenseFeatureAdapter()
        first = adapter.contract_probe(input_count=2, seed=44)
        second = adapter.contract_probe(input_count=2, seed=44)

        self.assertEqual(first.device_type, "cuda")
        self.assertGreater(first.loss, 0.0)
        self.assertTrue(first.has_gradients)
        self.assertTrue(first.parameters_updated)
        self.assertEqual(first.predictions, second.predictions)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "dense.pt"
            adapter.save_model(first.model, checkpoint)
            restored = adapter.load_model(checkpoint, input_count=2)
            self.assertEqual(first.predictions, adapter.predict_tuple(restored))

    def test_dense_baseline_isolates_internal_selection_and_external_retrain(self) -> None:
        partition = dense_training_partition()
        recipe = replace(DENSE_BASELINE_RECIPE, max_epochs=2, patience=1)
        baseline = DenseFeatureBaseline(recipe)

        predictor = baseline.fit(partition, ("area", "perimeter"))

        audit = baseline.training_audits[0]
        self.assertTrue(set(audit.selection_ids).isdisjoint(audit.stopping_ids))
        self.assertEqual(set(audit.retrain_ids), set(partition.sample_ids))
        self.assertEqual(set(audit.selection_ids) | set(audit.stopping_ids), set(partition.sample_ids))
        predictions = predictor.predict(PredictionPartition(partition.values[:2], partition.sample_ids[:2]))
        self.assertEqual(predictions.shape, (2,))

    def test_dense_default_recipe_is_frozen(self) -> None:
        self.assertEqual(DENSE_BASELINE_RECIPE.hidden_layers, (64, 32))
        self.assertEqual(DENSE_BASELINE_RECIPE.dropout, 0.20)
        self.assertEqual(DENSE_BASELINE_RECIPE.learning_rate, 0.001)
        self.assertEqual(DENSE_BASELINE_RECIPE.batch_size, 16)
        self.assertEqual(DENSE_BASELINE_RECIPE.weight_decay, 0.0001)
        self.assertEqual(DENSE_BASELINE_RECIPE.max_epochs, 500)
        self.assertEqual(DENSE_BASELINE_RECIPE.patience, 40)
        self.assertEqual(DENSE_BASELINE_RECIPE.minimum_improvement_kg, 0.1)
        self.assertEqual(DENSE_BASELINE_RECIPE.gradient_clip, 5.0)
        self.assertEqual(DENSE_BASELINE_RECIPE.inner_seed, 43)
        self.assertEqual(DENSE_BASELINE_RECIPE.training_seed, 44)


def training_partition() -> TrainingPartition:
    values = np.asarray(
        [[1.0, 5.0], [2.0, 4.0], [3.0, 3.0], [4.0, 2.0], [5.0, 1.0], [6.0, 0.0]],
        dtype=np.float64,
    )
    return TrainingPartition(
        values,
        np.asarray([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], dtype=np.float64),
        tuple(f"sample-{index}" for index in range(6)),
        ("B1", "B1", "B1", "B2", "B2", "B2"),
    )


def dense_training_partition() -> TrainingPartition:
    values = np.asarray([[float(index), float(20 - index)] for index in range(20)])
    return TrainingPartition(
        values,
        np.asarray([80.0 + index * 4 for index in range(20)], dtype=np.float64),
        tuple(f"dense-{index}" for index in range(20)),
        tuple(f"B{1 + index % 2}" for index in range(20)),
    )


if __name__ == "__main__":
    unittest.main()
