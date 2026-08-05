"""Selective recipe, dependency, and execution provenance for the compact CNN."""

from __future__ import annotations

import hashlib
import importlib.metadata
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
        source_root = Path(__file__).parent
        digest = hashlib.sha256()
        for name in _recipe_module_names():
            digest.update(name.encode())
            digest.update((source_root / name).read_bytes())
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


def _recipe_module_names() -> tuple[str, ...]:
    names = (
        "compact_cnn_adapter.py", "compact_cnn_artifacts.py",
        "compact_cnn_augmentation.py", "compact_cnn_evaluation.py",
        "compact_cnn_network.py", "compact_cnn_provenance.py",
        "compact_cnn_stage.py", "compact_cnn_types.py",
    )
    return names
