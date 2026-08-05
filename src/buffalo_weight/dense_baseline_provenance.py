"""Selective scientific provenance for the dense baseline."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import inspect
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from buffalo_weight.environment_contract import PYTHON_SERIES, RuntimeProbe
from buffalo_weight.system_setup import default_runtime_probe


class DenseBaselineProvenance(Protocol):
    """Provenance seam; for example, tests provide fixed environment identities."""

    def dense_baseline_recipe_hash(self) -> str:
        """Hash stage knowledge; for example, an adapter edit invalidates reuse."""
        ...

    def scientific_environment(self) -> dict[str, object]:
        """Identify validity inputs; for example, dependency pins control reuse."""
        ...

    def execution_environment(self) -> dict[str, object]:
        """Describe the CUDA run; for example, manifests record GPU and driver."""
        ...

    def repository_commit(self) -> str:
        """Identify source; for example, manifests record the full HEAD SHA."""
        ...


class SystemDenseBaselineProvenance:
    """Discover local identity; for example, production stages use installed packages."""

    def __init__(self, runtime_probe: RuntimeProbe | None = None) -> None:
        self._runtime_probe = runtime_probe or default_runtime_probe()

    def dense_baseline_recipe_hash(self) -> str:
        """Hash owned implementation; for example, metrics changes invalidate artifacts."""
        digest = hashlib.sha256()
        source_root = Path(__file__).parent
        for name in _recipe_module_names():
            digest.update(name.encode())
            digest.update((source_root / name).read_bytes())
        for qualified_name, source in _selected_sources():
            digest.update(qualified_name.encode())
            digest.update(source.encode())
        return digest.hexdigest()

    def scientific_environment(self) -> dict[str, object]:
        """Record validity fields; for example, Python patch does not invalidate reuse."""
        versions = {name: importlib.metadata.version(name) for name in _scientific_dependencies()}
        runtime = self._runtime_probe.python_runtime()
        actual_series = f"{runtime.major}.{runtime.minor}"
        if actual_series != PYTHON_SERIES:
            raise ValueError(
                f"Python series was {actual_series!r}; expected official series {PYTHON_SERIES!r}"
            )
        return {"python_series": actual_series, "direct_dependencies": versions}

    def execution_environment(self) -> dict[str, object]:
        """Record informational fields; for example, hardware remains audit-visible."""
        runtime = self._runtime_probe.python_runtime()
        compute = self._runtime_probe.compute_environment()
        return {
            "device": "cuda", "deterministic_algorithms": True,
            "cudnn_benchmark": False, "python_version": runtime.full_version,
            "python_implementation": runtime.implementation,
            "platform": self._runtime_probe.platform_description(),
            "compute": asdict(compute),
        }

    def repository_commit(self) -> str:
        """Read HEAD; for example, source provenance uses a full Git commit."""
        root = Path(__file__).parents[2]
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True,
        )
        return result.stdout.strip()


def _recipe_module_names() -> tuple[str, ...]:
    return (
        "dense_baseline_artifacts.py", "dense_baseline_evaluation.py",
        "dense_baseline_manifest.py", "dense_baseline_provenance.py",
        "dense_baseline_stage.py",
    )


def _scientific_dependencies() -> tuple[str, ...]:
    return ("numpy", "scikit-learn", "torch")


def _selected_sources() -> list[tuple[str, str]]:
    selected: list[tuple[str, str]] = []
    for module_name, symbol_names in _recipe_symbols():
        module = importlib.import_module(module_name)
        for symbol_name in symbol_names:
            qualified_name = f"{module_name}:{symbol_name}"
            selected.append((qualified_name, inspect.getsource(getattr(module, symbol_name))))
    return selected


def _recipe_symbols() -> tuple[tuple[str, tuple[str, ...]], ...]:
    return (
        ("buffalo_weight.dense_feature_adapter", (
            "DenseTrainingRecipe", "DenseTargetScale", "DenseFeatureNetwork",
            "DenseFeatureAdapter", "_initialize_he", "_seed_everything",
        )),
        ("buffalo_weight.feature_baselines", (
            "DenseTrainingAudit", "DenseFeaturePredictor", "DenseFeatureBaseline",
            "_FeatureScale", "_inner_indices", "_fit_feature_scale",
            "_fit_target_scale", "_training_audit",
        )),
        ("buffalo_weight.feature_selection_io", (
            "load_feature_samples", "_read_expected_csv", "_sample",
            "_finite_numeric_field", "_integer_field", "_required_selection_field",
        )),
    )
