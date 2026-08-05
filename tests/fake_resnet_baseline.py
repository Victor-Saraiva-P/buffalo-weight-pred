from __future__ import annotations

from buffalo_weight.resnet_baseline_evaluation import ResNetOofPrediction, ResNetSample


class FixedResNetBaselineRunner:
    """Return deterministic OOF values; for example, CLI tests avoid CUDA training."""

    def __init__(self) -> None:
        self.preflight_count = 0
        self.evaluation_count = 0

    def preflight(self) -> None:
        """Record an execution preflight; for example, reuse skips this call."""
        self.preflight_count += 1

    def evaluate(self, samples: tuple[ResNetSample, ...]) -> list[ResNetOofPrediction]:
        """Predict one deterministic value per sample; for example, preserve every fold."""
        self.evaluation_count += 1
        return [
            ResNetOofPrediction(
                sample.file_name,
                sample.fold,
                sample.weight_category,
                sample.weight_kg,
                sample.weight_kg + sample.fold / 10,
            )
            for sample in samples
        ]

    def execution_metadata(self) -> dict[str, object]:
        """Identify an injected run; for example, manifests distinguish test execution."""
        return {"device": "injected", "deterministic": True, "official": False}


class FixedResNetBaselineProvenance:
    """Keep CLI artifact identity stable; for example, source edits do not affect tests."""

    def __init__(self, recipe_hash: str = "a" * 64) -> None:
        self._recipe_hash = recipe_hash

    def recipe_hash(self) -> str:
        """Return fixed recipe identity; for example, reuse sees the same implementation."""
        return self._recipe_hash

    def dependency_versions(self) -> dict[str, str]:
        """Return fixed scientific dependencies; for example, cache identity stays stable."""
        return {
            "numpy": "2.5.0",
            "Pillow": "12.2.0",
            "scikit-learn": "1.9.0",
            "torch": "2.13.0",
            "torchvision": "0.28.0",
        }

    def repository_commit(self) -> str:
        """Return an audit commit; for example, manifests retain source provenance."""
        return "b" * 40
