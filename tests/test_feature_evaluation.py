from __future__ import annotations

import unittest

from buffalo_weight.feature_evaluation import FeatureSample, evaluate_feature_evidence
from buffalo_weight.feature_selection_stage import ScientificFeatureEvidenceRunner
from buffalo_weight.feature_selection_rules import permutation_seed
from tests.fake_feature_evaluation import RecordingFeatureBaseline


class FeatureEvaluationTest(unittest.TestCase):
    def test_scientific_runner_uses_injected_baseline_seams(self) -> None:
        random_forest = RecordingFeatureBaseline("random_forest")
        dense = RecordingFeatureBaseline("dense")
        runner = ScientificFeatureEvidenceRunner((random_forest, dense))

        evidence = runner.evaluate(
            two_fold_samples(), ("area", "perimeter"), (), 1, 42
        )

        self.assertEqual({row.baseline for row in evidence}, {"random_forest", "dense"})
        self.assertTrue(random_forest.fit_calls)
        self.assertTrue(dense.fit_calls)

    def test_evaluates_only_across_the_reserved_outer_fold(self) -> None:
        samples = two_fold_samples()
        baseline = RecordingFeatureBaseline()
        evidence = evaluate_feature_evidence(
            samples,
            ("area", "perimeter"),
            (),
            (baseline,),
            permutation_count=2,
            split_seed=42,
        )

        training_sets = {call.sample_ids for call in baseline.fit_calls}
        self.assertEqual(training_sets, {("a", "b"), ("c", "d")})
        predicted_sets = {part.sample_ids for part in baseline.predicted_partitions}
        self.assertEqual(predicted_sets, {("a", "b"), ("c", "d")})
        for call in baseline.prediction_calls:
            self.assertTrue(set(call.training_ids).isdisjoint(call.prediction_ids))
        self.assertTrue(any(row.scope == "oof" for row in evidence))

    def test_records_repetition_specific_permutation_seeds(self) -> None:
        samples = two_fold_samples()
        evidence = evaluate_feature_evidence(
            samples, ("area", "perimeter"), (), (RecordingFeatureBaseline(),), 2, 42
        )
        fold_rows = [
            row
            for row in evidence
            if row.experiment == "permutation" and row.scope == "fold"
            and row.fold == 1 and row.target == "area"
        ]
        self.assertEqual(
            [row.permutation_seed for row in fold_rows],
            [permutation_seed(42, 1, "area", 0), permutation_seed(42, 1, "area", 1)],
        )
        oof_rows = [row for row in evidence if row.experiment == "permutation"
                    and row.scope == "oof" and row.target == "area"]
        self.assertEqual(len(oof_rows), 2)
        self.assertEqual([row.permutation_seed for row in oof_rows], [None, None])


def two_fold_samples() -> list[FeatureSample]:
    return [
        FeatureSample("a", 1, "B1", 10.0, {"area": 10.0, "perimeter": 7.0}),
        FeatureSample("b", 1, "B2", 20.0, {"area": 20.0, "perimeter": 8.0}),
        FeatureSample("c", 2, "B1", 30.0, {"area": 30.0, "perimeter": 9.0}),
        FeatureSample("d", 2, "B2", 40.0, {"area": 40.0, "perimeter": 10.0}),
    ]


if __name__ == "__main__":
    unittest.main()
