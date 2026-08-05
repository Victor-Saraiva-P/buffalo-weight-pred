from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from buffalo_weight.dense_baseline_evaluation import (
    DenseBaselineEvaluation,
    DenseFoldAudit,
    DenseOofPrediction,
)
from buffalo_weight.dense_feature_adapter import DenseTargetScale, DenseTrainingRecipe
from buffalo_weight.environment_contract import ComputeEnvironment, PythonRuntime
from buffalo_weight.feature_evaluation import FeatureSample


class FixedDenseBaselineRunner:
    """Return deterministic OOF predictions while recording the frozen feature contract."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    def evaluate(
        self, samples: list[FeatureSample], feature_names: tuple[str, ...],
    ) -> DenseBaselineEvaluation:
        """Predict a known fold offset; for example, CLI tests can verify residual signs."""
        self.calls.append((tuple(sample.file_name for sample in samples), feature_names))
        predictions = tuple(
            DenseOofPrediction(
                sample.file_name, sample.fold, sample.weight_category, sample.weight_kg,
                sample.weight_kg + sample.fold,
            )
            for sample in samples
        )
        audits = tuple(_fixed_fold_audit(samples, fold)
                       for fold in sorted({sample.fold for sample in samples}))
        return DenseBaselineEvaluation(predictions, audits)


def _fixed_fold_audit(samples: list[FeatureSample], fold: int) -> DenseFoldAudit:
    retrain = tuple(sample.file_name for sample in samples if sample.fold != fold)
    stopping_count = math.ceil(len(retrain) * 0.20)
    return DenseFoldAudit(
        fold, retrain[:-stopping_count], retrain[-stopping_count:], retrain,
        tuple(sample.file_name for sample in samples if sample.fold == fold), 2 + fold,
    )


class FailingDenseBaselineRunner:
    """Fail evaluation; for example, tests prove stale artifacts are removed first."""

    def evaluate(
        self, samples: list[FeatureSample], feature_names: tuple[str, ...],
    ) -> DenseBaselineEvaluation:
        """Reject the call; for example, a retraining failure leaves no stale output."""
        raise ValueError(
            f"dense evaluation received {len(samples)} rows and {feature_names!r}; "
            "expected the deliberate test failure"
        )


@dataclass(frozen=True)
class FakeDenseModel:
    """Represent a refitted model without allocating CUDA tensors."""

    standardized_prediction: float = 0.0


class RecordingDenseFeatureAdapter:
    """Record inner selection and full refit calls behind the project-owned adapter seam."""

    def __init__(self) -> None:
        self.selection_sizes: list[tuple[int, int]] = []
        self.refit_sizes: list[int] = []
        self.recipes: list[DenseTrainingRecipe] = []

    def select_epoch_count(
        self, train_values: NDArray[np.float64], train_targets: NDArray[np.float64],
        validation_values: NDArray[np.float64], validation_targets_kg: NDArray[np.float64],
        target_scale: DenseTargetScale, recipe: DenseTrainingRecipe,
    ) -> int:
        """Record selection sizes; for example, outer-fold rows remain unavailable."""
        self.selection_sizes.append((len(train_values), len(validation_values)))
        self.recipes.append(recipe)
        return 7

    def fit_epochs(
        self, values: NDArray[np.float64], targets: NDArray[np.float64],
        epochs: int, recipe: DenseTrainingRecipe,
    ) -> FakeDenseModel:
        """Record full refit size; for example, all outer-training rows are consumed."""
        self.refit_sizes.append(len(values))
        self.recipes.append(recipe)
        if epochs != 7:
            raise ValueError(f"refit epochs were {epochs}; expected selected value 7")
        return FakeDenseModel()

    def predict_array(
        self, model: FakeDenseModel, values: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return fixed standardized predictions; for example, tests avoid CUDA tensors."""
        return np.full(len(values), model.standardized_prediction, dtype=np.float64)


class FixedDenseBaselineProvenance:
    """Provide stable scientific and execution provenance without external discovery."""

    def __init__(
        self, recipe_digit: str = "5", torch_version: str = "2.13.0",
        source_commit: str = "2" * 40,
    ) -> None:
        self._recipe_digit = recipe_digit
        self._torch_version = torch_version
        self._source_commit = source_commit

    def dense_baseline_recipe_hash(self) -> str:
        """Return a fixed hash; for example, one digit models a recipe change."""
        return self._recipe_digit * 64

    def scientific_environment(self) -> dict[str, object]:
        """Return fixed versions; for example, tests can invalidate only Torch."""
        return {"python_series": "3.14", "direct_dependencies": {
            "numpy": "2.4.2", "scikit-learn": "1.8.0", "torch": self._torch_version,
        }}

    def execution_environment(self) -> dict[str, object]:
        """Return deterministic CUDA metadata; for example, manifest validation accepts it."""
        return {
            "device": "cuda", "gpu_name": "Fake GPU", "deterministic_algorithms": True,
            "cudnn_benchmark": False,
        }

    def repository_commit(self) -> str:
        """Return audit-only source identity; for example, commits do not invalidate reuse."""
        return self._source_commit


class ChangedInputsProvenance:
    """Rebuild identical public inputs under a later source identity."""

    def inputs_recipe_hash(self) -> str:
        """Return a later recipe hash; for example, an input rebuild changes its manifest."""
        return "7" * 64

    def dependencies(self) -> dict[str, str]:
        """Return fixed dependencies; for example, the fake performs no discovery."""
        return {"fake-compute": "1.0"}

    def repository_commit(self) -> str:
        """Return a later commit; for example, input tables can remain byte-identical."""
        return "8" * 40


@dataclass
class FixedCudaRuntimeProbe:
    """Expose an available CUDA runtime and count preflight probes."""

    compute_checks: int = 0

    def python_runtime(self) -> PythonRuntime:
        """Return the official series; for example, preflight accepts Python 3.14."""
        return PythonRuntime(3, 14, 6, "CPython")

    def compute_environment(self) -> ComputeEnvironment:
        """Return CUDA metadata; for example, preflight records exactly one probe."""
        self.compute_checks += 1
        return ComputeEnvironment("Fake GPU", "9.0", "13.0", "590.00")

    def platform_description(self) -> str:
        """Return a stable platform; for example, audit metadata is deterministic."""
        return "Fake Linux"


@dataclass
class UnavailableDenseRuntimeProbe:
    """Report no CUDA while recording that the early preflight occurred."""

    compute_checks: int = 0

    def python_runtime(self) -> PythonRuntime:
        """Return the official series; for example, failure is specific to CUDA."""
        return PythonRuntime(3, 14, 6, "CPython")

    def compute_environment(self) -> ComputeEnvironment:
        """Return missing CUDA metadata; for example, evaluation must not begin."""
        self.compute_checks += 1
        return ComputeEnvironment(None, None, "13.0", "590.00")

    def platform_description(self) -> str:
        """Return a stable platform; for example, errors do not depend on the host."""
        return "Fake Linux"
