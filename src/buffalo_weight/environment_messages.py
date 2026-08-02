"""User-facing plain-text messages for official environment setup."""

from pathlib import Path

from buffalo_weight.environment_contract import (
    APPROVED_DEPENDENCIES,
    RESNET18_WEIGHT_NAME,
    ComputeEnvironment,
    DependencySetupStatus,
    PythonRuntime,
    WeightSetupStatus,
)


def setup_messages(
    runtime: PythonRuntime,
    dependency_status: DependencySetupStatus,
    weight_status: WeightSetupStatus,
    compute: ComputeEnvironment,
    provenance_path: Path,
) -> list[str]:
    """Describe setup results; for example, distinguish installed from reused assets."""
    return [
        f"validated Python {runtime.full_version} (3.14.x required; patch is informational)",
        dependency_message(dependency_status),
        f"{weight_status.value} ResNet-18 {RESNET18_WEIGHT_NAME} weights with SHA-256",
        cuda_message(compute),
        f"recorded environment provenance at {provenance_path}",
    ]


def dependency_message(status: DependencySetupStatus) -> str:
    """Describe dependency status; for example, report all eight approved pins."""
    return (
        f"{status.value} {len(APPROVED_DEPENDENCIES)} approved dependencies "
        "and validated consistency"
    )


def cuda_message(compute: ComputeEnvironment) -> str:
    """Describe CUDA audit status; for example, CPU-only setup remains available."""
    if compute.gpu_name is None:
        return "CUDA unavailable; setup and dry-run remain available"
    return f"recorded CUDA GPU {compute.gpu_name} with capability {compute.cuda_capability}"
