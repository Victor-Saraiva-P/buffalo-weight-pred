"""Configuration contract for the report reproduction pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


CANONICAL_FOLD_SEED = 42


@dataclass(frozen=True)
class InputsContract:
    mask_index_path: Path
    masks_dir: Path
    expected_mask_count: int
    canonical_long_side: int
    weight_category_count: int
    fold_count: int
    fold_seed: int


@dataclass(frozen=True)
class ReportContract:
    inputs: InputsContract
    artifacts_root: Path

    @property
    def inputs_output_dir(self) -> Path:
        """Locate reconstructed inputs.

        Example: ``contract.inputs_output_dir`` resolves beneath the artifact root.
        """
        return self.artifacts_root / "inputs"


def load_report_contract(path: Path) -> ReportContract:
    """Load the report contract; for example, ``load_report_contract(Path('report.yaml'))``."""
    loaded = yaml.safe_load(path.read_text())
    if not isinstance(loaded, dict):
        raise ValueError(f"config root was {loaded!r}; expected a mapping")
    inputs = _required_mapping(loaded, "inputs")
    artifacts = _required_mapping(loaded, "artifacts")
    artifacts_root = Path(_required_text(artifacts, "root"))
    _validate_artifacts_root(artifacts_root)
    return ReportContract(_inputs_contract(inputs), artifacts_root)


def contract_identity(contract: ReportContract) -> dict[str, object]:
    """Normalize pertinent settings; for example, ``contract_identity(contract)``."""
    inputs = contract.inputs
    return {
        "mask_index_path": str(inputs.mask_index_path),
        "masks_dir": str(inputs.masks_dir),
        "expected_mask_count": inputs.expected_mask_count,
        "canonical_long_side": inputs.canonical_long_side,
        "weight_category_count": inputs.weight_category_count,
        "fold_count": inputs.fold_count,
        "fold_seed": inputs.fold_seed,
    }


def _inputs_contract(values: dict[object, object]) -> InputsContract:
    contract = InputsContract(
        Path(_required_text(values, "mask_index_path")),
        Path(_required_text(values, "masks_dir")),
        _positive_int(values, "expected_mask_count"),
        _positive_int(values, "canonical_long_side"),
        _positive_int(values, "weight_category_count"),
        _positive_int(values, "fold_count"),
        _integer(values, "fold_seed"),
    )
    if contract.weight_category_count < 2 or contract.fold_count < 2:
        raise ValueError(
            "category/fold counts were "
            f"{contract.weight_category_count}/{contract.fold_count}; expected both at least 2"
        )
    _validate_canonical_fold_seed(contract.fold_seed)
    return contract


def _validate_canonical_fold_seed(fold_seed: int) -> None:
    if fold_seed != CANONICAL_FOLD_SEED:
        raise ValueError(
            f"fold_seed was {fold_seed!r}; expected canonical seed {CANONICAL_FOLD_SEED}"
        )


def _required_mapping(values: dict[object, object], name: str) -> dict[object, object]:
    candidate = values.get(name)
    if not isinstance(candidate, dict):
        raise ValueError(f"config {name} was {candidate!r}; expected a mapping")
    return candidate


def _required_text(values: dict[object, object], name: str) -> str:
    candidate = values.get(name)
    if not isinstance(candidate, str) or not candidate:
        raise ValueError(f"config {name} was {candidate!r}; expected non-empty text")
    return candidate


def _integer(values: dict[object, object], name: str) -> int:
    candidate = values.get(name)
    if isinstance(candidate, bool) or not isinstance(candidate, int):
        raise ValueError(f"config {name} was {candidate!r}; expected an integer")
    return candidate


def _positive_int(values: dict[object, object], name: str) -> int:
    candidate = _integer(values, name)
    if candidate <= 0:
        raise ValueError(f"config {name} was {candidate}; expected an integer greater than 0")
    return candidate


def _validate_artifacts_root(path: Path) -> None:
    parts = path.parts
    if len(parts) < 2 or parts[-2:] != ("generated", "report"):
        raise ValueError(
            f"artifacts root was {path}; expected a path ending in generated/report"
        )
