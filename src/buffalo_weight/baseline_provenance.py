"""Selective provenance boundary for baseline configurations."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import inspect
import json
import subprocess
from pathlib import Path
from typing import Protocol

from buffalo_weight.baseline_artifacts import (
    FOLD_METRIC_COLUMNS,
    GROUPED_METRIC_COLUMNS,
    PREDICTION_COLUMNS,
)
from buffalo_weight.baseline_types import (
    BASELINE_VALIDATIONS,
    BaselineConfiguration,
    baseline_definition,
)
from buffalo_weight.feature_baselines import RandomForestBaseline


RecipeSymbol = tuple[str, str]


class BaselineProvenance(Protocol):
    """Discover configuration identity; for example, tests provide fixed hashes."""

    def baseline_recipe_hash(self, configuration: BaselineConfiguration) -> str:
        """Hash recipe knowledge; for example, RF and reference hashes are independent."""
        # Implementations own discovery so callers cannot pass a stale manual version.
        ...

    def baseline_dependencies(
        self, configuration: BaselineConfiguration,
    ) -> dict[str, str]:
        """Report pertinent packages; for example, RF records scikit-learn."""
        # Resolution at execution time keeps manifests tied to the scientific environment.
        ...

    def repository_commit(self) -> str:
        """Report source identity; for example, a manifest records HEAD."""
        # Git access remains replaceable so tests never depend on the surrounding checkout.
        ...


class BaselineEnvironment(Protocol):
    """External provenance I/O; for example, tests replace source and package reads."""

    def source_text(self, module_name: str, symbol_name: str) -> str:
        """Read one symbol; for example, RF-only source stays out of the reference hash."""
        # Symbol-granular reads keep unrelated configuration logic out of each recipe.
        ...

    def distribution_version(self, name: str) -> str:
        """Read one package version; for example, scikit-learn is resolved at runtime."""
        # Package discovery stays behind this boundary for repeatable tests.
        ...

    def repository_commit(self, root: Path) -> str:
        """Read Git identity; for example, production resolves the current checkout."""
        # Subprocess access stays replaceable by a named fake environment.
        ...


class LocalBaselineEnvironment:
    """Read local source, packages and Git for scientific provenance."""

    def source_text(self, module_name: str, symbol_name: str) -> str:
        """Read one implementation; for example, helpers are hashed independently."""
        module = importlib.import_module(module_name)
        source = inspect.getsource(getattr(module, symbol_name))
        return source

    def distribution_version(self, name: str) -> str:
        """Read an installed version; for example, NumPy is recorded exactly."""
        version = importlib.metadata.version(name)
        return version

    def repository_commit(self, root: Path) -> str:
        """Read HEAD; for example, completed manifests retain a full Git SHA."""
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True,
        )
        return result.stdout.strip()


class SystemBaselineProvenance:
    """Build selective recipe, package and Git identity for baseline manifests."""

    def __init__(self, environment: BaselineEnvironment | None = None) -> None:
        """Inject external I/O; for example, tests use fixed source and versions."""
        resolved_environment = environment or LocalBaselineEnvironment()
        self._environment = resolved_environment

    def baseline_recipe_hash(self, configuration: BaselineConfiguration) -> str:
        """Hash exact knowledge; for example, RF edits preserve the reference cache."""
        digest = hashlib.sha256()
        for module_name, symbol_name in _recipe_symbols(configuration):
            identity = f"{module_name}:{symbol_name}"
            digest.update(identity.encode())
            digest.update(self._environment.source_text(module_name, symbol_name).encode())
        digest.update(_recipe_constants(configuration).encode())
        return digest.hexdigest()

    def baseline_dependencies(
        self, configuration: BaselineConfiguration,
    ) -> dict[str, str]:
        """Report pertinent packages; for example, RF additionally records scikit-learn."""
        definition = baseline_definition(configuration)
        names = list(definition.dependencies)
        return {name: self._environment.distribution_version(name) for name in names}

    def repository_commit(self) -> str:
        """Read HEAD; for example, completed manifests preserve their source commit."""
        root = Path(__file__).parents[2]
        commit = self._environment.repository_commit(root)
        return commit


def _recipe_symbols(configuration: BaselineConfiguration) -> tuple[RecipeSymbol, ...]:
    shared = _shared_recipe_symbols()
    if configuration == "random_forest_baseline":
        return (*shared, *_random_forest_recipe_symbols())
    return (*shared, *_training_mean_recipe_symbols())


def _shared_recipe_symbols() -> tuple[RecipeSymbol, ...]:
    return (
        ("buffalo_weight.baseline_artifacts", "write_baseline_predictions"),
        ("buffalo_weight.baseline_artifacts", "write_baseline_metrics"),
        ("buffalo_weight.baseline_artifacts", "_prediction_record"),
        ("buffalo_weight.baseline_artifacts", "_identity"),
        ("buffalo_weight.baseline_artifacts", "_metric_record"),
        ("buffalo_weight.baseline_metrics", "summarize_predictions"),
        ("buffalo_weight.baseline_metrics", "fold_summaries"),
        ("buffalo_weight.baseline_metrics", "grouped_summaries"),
        ("buffalo_weight.baseline_metrics", "_r2"),
        ("buffalo_weight.baseline_types", "BaselinePrediction"),
        *_shared_io_recipe_symbols(),
        *_shared_manifest_recipe_symbols(),
        ("buffalo_weight.baseline_evaluation", "_outer_fold_partitions"),
        ("buffalo_weight.baseline_evaluation", "_baseline_prediction"),
        *_shared_orchestration_recipe_symbols(),
    )


def _shared_io_recipe_symbols() -> tuple[RecipeSymbol, ...]:
    return (
        ("buffalo_weight.csv_io", "format_csv_number"),
        ("buffalo_weight.csv_io", "write_csv_rows"),
        ("buffalo_weight.feature_selection_io", "load_feature_samples"),
        ("buffalo_weight.feature_selection_io", "_sample"),
        ("buffalo_weight.feature_selection_io", "_finite_numeric_field"),
        ("buffalo_weight.feature_selection_io", "_integer_field"),
        ("buffalo_weight.feature_selection_io", "_required_selection_field"),
        ("buffalo_weight.feature_selection_io", "_read_expected_csv"),
    )


def _shared_manifest_recipe_symbols() -> tuple[RecipeSymbol, ...]:
    return (
        ("buffalo_weight.baseline_manifest", "complete_baseline_manifest"),
        ("buffalo_weight.baseline_manifest", "baseline_configuration_status"),
        ("buffalo_weight.baseline_manifest", "baseline_identity"),
        ("buffalo_weight.baseline_manifest", "_input_records"),
        ("buffalo_weight.baseline_manifest", "_csv_projection_record"),
        ("buffalo_weight.baseline_manifest", "_output_records"),
        ("buffalo_weight.baseline_manifest", "_outputs_match"),
    )


def _shared_orchestration_recipe_symbols() -> tuple[RecipeSymbol, ...]:
    return (
        ("buffalo_weight.baseline_stage", "run_random_forest_baseline_stage"),
        ("buffalo_weight.baseline_stage", "_run_current_baseline_inputs"),
        ("buffalo_weight.baseline_stage", "_rebuild_obsolete_configurations"),
        ("buffalo_weight.baseline_stage", "_configuration_statuses"),
        ("buffalo_weight.baseline_stage", "_inputs_identity_is_current"),
        ("buffalo_weight.baseline_stage", "_blocked_or_raise"),
        ("buffalo_weight.baseline_stage", "_remove_all_obsolete"),
        ("buffalo_weight.baseline_stage", "_publish_configuration"),
    )


def _random_forest_recipe_symbols() -> tuple[RecipeSymbol, ...]:
    return (
        ("buffalo_weight.baseline_evaluation", "evaluate_random_forest_oof"),
        ("buffalo_weight.baseline_evaluation", "_training_partition"),
        ("buffalo_weight.baseline_evaluation", "_prediction_partition"),
        ("buffalo_weight.baseline_evaluation", "_feature_matrix"),
        ("buffalo_weight.baseline_evaluation", "_prediction_rows"),
        ("buffalo_weight.feature_baselines", "RandomForestBaseline"),
        ("buffalo_weight.feature_baselines", "SklearnFeaturePredictor"),
        ("buffalo_weight.baseline_stage", "_random_forest_predictions"),
    )


def _training_mean_recipe_symbols() -> tuple[RecipeSymbol, ...]:
    return (
        ("buffalo_weight.baseline_evaluation", "evaluate_training_mean_reference"),
        ("buffalo_weight.baseline_evaluation", "_reference_rows"),
        ("buffalo_weight.baseline_stage", "_training_mean_predictions"),
    )


def _recipe_constants(configuration: BaselineConfiguration) -> str:
    from buffalo_weight.baseline_stage import baseline_evaluator_symbol

    definition = baseline_definition(configuration)
    recipe_contract: dict[str, object] = {
        "configuration": configuration, "prediction_columns": PREDICTION_COLUMNS,
        "fold_metric_columns": FOLD_METRIC_COLUMNS,
        "grouped_metric_columns": GROUPED_METRIC_COLUMNS,
        "validations": BASELINE_VALIDATIONS, "evaluation_role": definition.evaluation_role,
        "consumes_confirmed_features": definition.consumes_confirmed_features,
        "evaluator_symbol": baseline_evaluator_symbol(configuration),
    }
    if configuration == "random_forest_baseline":
        recipe_contract["model_recipe"] = RandomForestBaseline.recipe
    return json.dumps(recipe_contract, sort_keys=True, separators=(",", ":"))
