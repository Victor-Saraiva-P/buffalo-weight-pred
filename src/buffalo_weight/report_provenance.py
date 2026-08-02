"""Execution and recipe provenance for reconstructed report stages."""

from __future__ import annotations

import hashlib
import importlib.metadata
import subprocess
from pathlib import Path


def inputs_recipe_hash() -> str:
    """Hash input-stage knowledge.

    Example: ``inputs_recipe_hash()`` changes when a consumed implementation changes.
    """
    source_root = Path(__file__).parent
    modules = [source_root / name for name in _recipe_module_names()]
    calculators = sorted((source_root / "feature_calculators").glob("*.py"))
    return _combined_source_hash(modules + calculators)


def reproduction_dependencies() -> dict[str, str]:
    """Report pertinent package versions.

    Example: ``reproduction_dependencies()['numpy']`` returns the NumPy version.
    """
    distributions = ("numpy", "Pillow", "scipy", "scikit-learn", "PyYAML")
    return {name: importlib.metadata.version(name) for name in distributions}


def repository_commit() -> str:
    """Report the source commit.

    Example: ``repository_commit()`` returns the checkout's full Git SHA.
    """
    repository_root = Path(__file__).parents[2]
    result = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _recipe_module_names() -> tuple[str, ...]:
    return (
        "canonical_split.py", "csv_io.py", "curated_inputs.py", "hashing.py",
        "input_schema.py", "inputs_manifest.py", "report_inputs.py",
        "report_provenance.py", "reproduction_config.py", "snapshot_io.py",
    )


def _combined_source_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    source_root = Path(__file__).parent
    for path in paths:
        digest.update(path.relative_to(source_root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
