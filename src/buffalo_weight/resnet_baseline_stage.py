"""Atomic, reusable execution of the ResNet-18 baseline configuration."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from buffalo_weight.environment_contract import (
    RESNET18_CACHE_PATH,
    RESNET18_SHA256,
    RESNET18_WEIGHT_NAME,
)
from buffalo_weight.feature_confirmation import require_baselines_gate
from buffalo_weight.hashing import sha256_file
from buffalo_weight.inputs_manifest import stage_status as inputs_stage_status
from buffalo_weight.report_provenance import ReportProvenance, SystemReportProvenance
from buffalo_weight.reproduction_config import ReportContract
from buffalo_weight.resnet18_weights import require_offline_resnet18_weights
from buffalo_weight.resnet_baseline_adapter import (
    RESNET18_BASELINE_RECIPE,
    ResNet18BaselineAdapter,
)
from buffalo_weight.resnet_baseline_artifacts import (
    MODEL_CONFIG,
    load_resnet_samples,
    output_metadata,
    validate_output_metadata,
    validate_predictions,
    write_manifest,
    write_resnet_outputs,
)
from buffalo_weight.resnet_baseline_evaluation import (
    ResNetBaselineEvaluator,
    ResNetOofPrediction,
    ResNetSample,
)
from buffalo_weight.resnet_baseline_provenance import (
    ResNetBaselineProvenance,
    SystemResNetBaselineProvenance,
)
from buffalo_weight.snapshot_io import FilesystemSnapshotPublisher, SnapshotPublisher
from buffalo_weight.system_setup import default_runtime_probe, require_official_neural_runtime


class ResNetBaselineRunner(Protocol):
    """Expensive execution seam; for example, CLI tests inject deterministic predictions."""

    def preflight(self) -> None:
        """Validate CUDA and weights; for example, failure precedes mask loading."""
        ...

    def evaluate(self, samples: tuple[ResNetSample, ...]) -> list[ResNetOofPrediction]:
        """Produce OOF predictions; for example, one result is returned per mask."""
        ...

    def execution_metadata(self) -> dict[str, object]:
        """Describe execution; for example, record GPU identity without cache invalidation."""
        ...


class ScientificResNetBaselineRunner:
    """Compose official offline CUDA adapters; for example, the public CLI uses this."""

    def preflight(self) -> None:
        """Fail early; for example, missing weights instruct the setup command."""
        require_offline_resnet18_weights(RESNET18_CACHE_PATH, RESNET18_SHA256)
        require_official_neural_runtime(False)

    def evaluate(self, samples: tuple[ResNetSample, ...]) -> list[ResNetOofPrediction]:
        """Evaluate canonical folds; for example, each refit reloads official weights."""
        adapter = ResNet18BaselineAdapter()
        evaluator = ResNetBaselineEvaluator(adapter, RESNET18_BASELINE_RECIPE.inner_seed)
        return evaluator.evaluate(samples)

    def execution_metadata(self) -> dict[str, object]:
        """Record CUDA audit fields; for example, hardware changes do not invalidate reuse."""
        compute = default_runtime_probe().compute_environment()
        return {
            "device": "cuda", "deterministic": True, "official": True,
            "gpu_name": compute.gpu_name, "cuda_capability": compute.cuda_capability,
            "cuda_version": compute.cuda_version, "driver_version": compute.driver_version,
        }


def run_resnet_baseline_stage(
    contract: ReportContract, dry_run: bool = False,
    runner: ResNetBaselineRunner | None = None,
    provenance: ResNetBaselineProvenance | None = None,
    report_provenance: ReportProvenance | None = None,
    publisher: SnapshotPublisher | None = None,
) -> str:
    """Run or reuse the baseline; for example, dry-run never probes CUDA or weights."""
    require_baselines_gate(contract)
    _require_current_inputs(contract, report_provenance or SystemReportProvenance())
    resolved_provenance = provenance or SystemResNetBaselineProvenance()
    status = resnet_baseline_status(contract, resolved_provenance)
    if dry_run or status == "reusable":
        return status
    resolved_runner = runner or ScientificResNetBaselineRunner()
    resolved_runner.preflight()
    _build_snapshot(contract, resolved_runner, resolved_provenance,
                    publisher or FilesystemSnapshotPublisher())
    return "rebuilt"


def resnet_baseline_status(
    contract: ReportContract, provenance: ResNetBaselineProvenance
) -> str:
    """Inspect artifact freshness; for example, a changed predictions hash is obsolete."""
    output_dir = resnet_baseline_output_dir(contract)
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return "absent"
    try:
        manifest = json.loads(manifest_path.read_text())
        _validate_manifest(manifest, output_dir, contract, provenance)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return "obsolete"
    return "reusable"


def resnet_baseline_output_dir(contract: ReportContract) -> Path:
    """Locate one configuration; for example, sibling baselines keep separate artifacts."""
    return contract.artifacts_root / "baselines" / MODEL_CONFIG


def _require_current_inputs(contract: ReportContract, provenance: ReportProvenance) -> None:
    status = inputs_stage_status(contract, provenance)
    if status != "reusable":
        raise ValueError(f"inputs stage status was {status!r}; expected reusable before baselines")


def _build_snapshot(
    contract: ReportContract, runner: ResNetBaselineRunner,
    provenance: ResNetBaselineProvenance, publisher: SnapshotPublisher,
) -> None:
    destination = resnet_baseline_output_dir(contract)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".resnet18-", dir=destination.parent))
    identity = _baseline_identity(contract, provenance)
    try:
        _write_snapshot(temporary, contract, runner, provenance, identity)
        publisher.publish(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _write_snapshot(
    output_dir: Path, contract: ReportContract, runner: ResNetBaselineRunner,
    provenance: ResNetBaselineProvenance, identity: dict[str, object],
) -> None:
    samples = load_resnet_samples(
        contract.inputs_output_dir / "canonical_split.csv", contract.inputs.masks_dir
    )
    predictions = runner.evaluate(samples)
    validate_predictions(predictions, samples)
    write_resnet_outputs(output_dir, predictions)
    if _baseline_identity(contract, provenance) != identity:
        raise ValueError("ResNet baseline identity changed; expected an unchanged evaluation")
    manifest = _complete_manifest(output_dir, identity, provenance, runner)
    write_manifest(output_dir / "manifest.json", manifest)


def _complete_manifest(
    output_dir: Path, identity: dict[str, object], provenance: ResNetBaselineProvenance,
    runner: ResNetBaselineRunner,
) -> dict[str, object]:
    return {
        "schema_version": 1, "status": "complete", "stage": "baselines",
        "model_config": MODEL_CONFIG, "identity": identity,
        "source_commit": provenance.repository_commit(),
        "execution": runner.execution_metadata(), "outputs": output_metadata(output_dir),
    }


def _baseline_identity(
    contract: ReportContract, provenance: ResNetBaselineProvenance
) -> dict[str, object]:
    return {
        "recipe": asdict(RESNET18_BASELINE_RECIPE), "recipe_hash": provenance.recipe_hash(),
        "dependencies": provenance.dependency_versions(),
        "inputs": _input_hashes(contract),
        "weights": {"name": RESNET18_WEIGHT_NAME, "sha256": RESNET18_SHA256},
    }


def _input_hashes(contract: ReportContract) -> dict[str, str]:
    confirmed = contract.confirmed_feature_selection_dir
    return {
        "inputs_manifest": sha256_file(contract.inputs_output_dir / "manifest.json"),
        "canonical_split": sha256_file(contract.inputs_output_dir / "canonical_split.csv"),
        "confirmed_feature_contract": sha256_file(confirmed / "shared_feature_contract.json"),
        "confirmed_feature_manifest": sha256_file(confirmed / "manifest.json"),
        "masks": _mask_collection_hash(contract),
    }


def _mask_collection_hash(contract: ReportContract) -> str:
    digest = hashlib.sha256()
    samples = load_resnet_samples(
        contract.inputs_output_dir / "canonical_split.csv", contract.inputs.masks_dir
    )
    for sample in samples:
        digest.update(sample.file_name.encode())
        digest.update(sha256_file(sample.mask_path).encode())
    return digest.hexdigest()


def _validate_manifest(
    manifest: object, output_dir: Path, contract: ReportContract,
    provenance: ResNetBaselineProvenance,
) -> None:
    if not isinstance(manifest, dict):
        raise ValueError(f"baseline manifest was {manifest!r}; expected a mapping")
    fixed = (manifest.get("schema_version"), manifest.get("status"),
             manifest.get("stage"), manifest.get("model_config"))
    expected = (1, "complete", "baselines", MODEL_CONFIG)
    if fixed != expected:
        raise ValueError(f"baseline manifest fields were {fixed!r}; expected {expected!r}")
    _validate_audit_fields(manifest)
    identity = _baseline_identity(contract, provenance)
    if manifest.get("identity") != identity:
        raise ValueError(f"baseline identity was {manifest.get('identity')!r}; expected {identity!r}")
    validate_output_metadata(output_dir, manifest.get("outputs"),
                             contract.inputs.expected_mask_count, contract.inputs.fold_count)


def _validate_audit_fields(manifest: dict[str, object]) -> None:
    commit = manifest.get("source_commit")
    execution = manifest.get("execution")
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError(f"baseline source commit was {commit!r}; expected a full Git SHA")
    if not isinstance(execution, dict):
        raise ValueError(f"baseline execution was {execution!r}; expected a mapping")
    required = (execution.get("device"), execution.get("deterministic"),
                execution.get("official"))
    if not isinstance(required[0], str) or required[1] is not True or not isinstance(required[2], bool):
        raise ValueError(
            f"baseline execution fields were {required!r}; expected device text, "
            "deterministic true and official boolean"
        )
