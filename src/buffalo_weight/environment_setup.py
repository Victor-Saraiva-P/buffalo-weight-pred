"""Use case that prepares and audits the official environment."""

from pathlib import Path

from buffalo_weight.environment_contract import (
    APPROVED_DEPENDENCIES,
    PYTHON_SERIES,
    RESNET18_SHA256,
    RESNET18_WEIGHT_NAME,
    ComputeEnvironment,
    DependencySetupStatus,
    EnvironmentProvenance,
    InformationalEnvironment,
    PackageGateway,
    PythonRuntime,
    ScientificValidity,
    SetupPaths,
    SetupServices,
)
from buffalo_weight.environment_messages import setup_messages


def validate_python_runtime(runtime: PythonRuntime) -> None:
    """Reject unsupported Python series; for example, Python 3.13 is rejected."""
    if (runtime.major, runtime.minor) == (3, 14):
        return
    raise ValueError(
        f"Python version was {runtime.full_version!r}; expected the 3.14.x series"
    )


def synchronize_dependencies(
    packages: PackageGateway, requirements: Path
) -> DependencySetupStatus:
    """Install or reuse approved dependencies; for example, matching pins are reused."""
    status = _install_dependency_changes(packages, requirements)
    validated = packages.installed_direct_versions()
    if validated != APPROVED_DEPENDENCIES:
        raise ValueError(
            f"direct dependency versions were {validated!r}; expected {APPROVED_DEPENDENCIES!r}"
        )
    packages.verify_consistency()
    return status


def _install_dependency_changes(
    packages: PackageGateway, requirements: Path
) -> DependencySetupStatus:
    if packages.installed_direct_versions() == APPROVED_DEPENDENCIES:
        return DependencySetupStatus.REUSED
    packages.install_approved(requirements)
    return DependencySetupStatus.INSTALLED


def build_environment_provenance(
    services: SetupServices, runtime: PythonRuntime, compute: ComputeEnvironment
) -> EnvironmentProvenance:
    """Separate validity from audit detail; for example, patch stays informational."""
    validity = ScientificValidity(
        PYTHON_SERIES, dict(APPROVED_DEPENDENCIES), RESNET18_WEIGHT_NAME, RESNET18_SHA256
    )
    informational = InformationalEnvironment(
        runtime.full_version,
        runtime.implementation,
        services.runtime.platform_description(),
        services.packages.resolved_versions(),
        compute,
    )
    return EnvironmentProvenance(validity, informational)


def setup_official_environment(
    services: SetupServices, paths: SetupPaths = SetupPaths()
) -> list[str]:
    """Prepare the official environment; for example, setup remains CPU-safe."""
    runtime = services.runtime.python_runtime()
    validate_python_runtime(runtime)
    dependency_status = synchronize_dependencies(services.packages, paths.requirements)
    weight_status = services.weights.ensure_resnet18_weights(paths.weights_cache, RESNET18_SHA256)
    compute = services.runtime.compute_environment()
    provenance = build_environment_provenance(services, runtime, compute)
    services.provenance.write(paths.provenance, provenance)
    return setup_messages(runtime, dependency_status, weight_status, compute, paths.provenance)
