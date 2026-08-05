"""Selective recipe, dependency, and execution provenance for the compact CNN."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import inspect
import subprocess
from pathlib import Path
from typing import Protocol

from buffalo_weight.system_setup import default_runtime_probe


class CompactCnnProvenance(Protocol):
    """External provenance seam; for example, tests inject fixed scientific identity."""

    def compact_cnn_recipe_hash(self) -> str:
        """Hash pertinent source; for example, augmentation edits invalidate reuse."""
        # Implementations resolve source knowledge rather than trusting manual versions.
        ...

    def compact_cnn_dependencies(self) -> dict[str, str]:
        """Report scientific packages; for example, include exact PyTorch."""
        # Exact versions participate in scientific freshness.
        ...

    def repository_commit(self) -> str:
        """Report source identity; for example, manifests record the full HEAD SHA."""
        # Commit identity remains audit metadata rather than a freshness key.
        ...

    def compact_cnn_execution(self) -> dict[str, str]:
        """Audit CUDA host details; for example, record GPU and driver without invalidation."""
        # Hardware metadata documents a run without invalidating it.
        ...


class SystemCompactCnnProvenance:
    """Discover local provenance; for example, production stages use this adapter."""

    def compact_cnn_recipe_hash(self) -> str:
        """Hash compact-CNN knowledge; for example, unrelated diagnostics are excluded."""
        digest = hashlib.sha256()
        for module_name, symbol_name in _recipe_source_symbols():
            qualified_name = f"{module_name}:{symbol_name}"
            digest.update(qualified_name.encode())
            module = importlib.import_module(module_name)
            digest.update(inspect.getsource(getattr(module, symbol_name)).encode())
        return digest.hexdigest()

    def compact_cnn_dependencies(self) -> dict[str, str]:
        """Read exact packages; for example, Pillow defines nearest-neighbor loading."""
        names = ("numpy", "Pillow", "scikit-learn", "torch")
        return {name: importlib.metadata.version(name) for name in names}

    def repository_commit(self) -> str:
        """Read HEAD; for example, manifests retain the implementation commit."""
        root = Path(__file__).parents[2]
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                                capture_output=True, text=True)
        return result.stdout.strip()

    def compact_cnn_execution(self) -> dict[str, str]:
        """Read CUDA audit fields; for example, missing GPU is rejected by the adapter."""
        runtime = default_runtime_probe()
        python, compute = runtime.python_runtime(), runtime.compute_environment()
        return {
            "device": "cuda", "gpu_name": str(compute.gpu_name),
            "cuda_capability": str(compute.cuda_capability),
            "cuda_version": str(compute.cuda_version),
            "driver_version": str(compute.driver_version),
            "python_version": python.full_version,
        }


def _recipe_source_symbols() -> tuple[tuple[str, str], ...]:
    prefix = "buffalo_weight"
    symbols = (
        (f"{prefix}.compact_cnn_types", "CompactCnnRecipe"),
        (f"{prefix}.compact_cnn_types", "CompactCnnTargetScale"),
        (f"{prefix}.compact_cnn_network", "CompactCnnNetwork"),
        (f"{prefix}.compact_cnn_network", "DeterministicAdaptiveAveragePool4"),
        *(_augmentation_symbols(prefix)), *(_training_symbols(prefix)),
        *(_evaluation_symbols(prefix)),
    )
    return symbols


def _augmentation_symbols(prefix: str) -> tuple[tuple[str, str], ...]:
    module = f"{prefix}.compact_cnn_augmentation"
    names = ("augment_binary_masks", "_augment_one", "_valid_translation",
             "_translation_limits", "_sample_shift", "_translate_without_wrap")
    return tuple((module, name) for name in names)


def _training_symbols(prefix: str) -> tuple[tuple[str, str], ...]:
    module = f"{prefix}.compact_cnn_adapter"
    names = ("CompactCnnAdapter", "TorchCompactCnnPredictor", "_optimizer",
             "_train_epoch", "_validation_mae", "_seed_everything")
    return tuple((module, name) for name in names)


def _evaluation_symbols(prefix: str) -> tuple[tuple[str, str], ...]:
    module = f"{prefix}.compact_cnn_evaluation"
    names = ("evaluate_compact_cnn", "_evaluate_fold", "_inner_samples", "_mask_batch",
             "load_compact_cnn_samples", "load_letterboxed_mask",
             "fit_compact_target_scale", "_sample_from_row", "_nearest_letterbox",
             "_validate_oof_predictions")
    return tuple((module, name) for name in names)
