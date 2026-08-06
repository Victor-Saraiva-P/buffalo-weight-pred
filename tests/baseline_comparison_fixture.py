from __future__ import annotations

import io
from pathlib import Path

from buffalo_weight.dense_baseline_stage import DenseBaselineDependencies
from buffalo_weight.report_cli import main
from tests.fake_baseline_provenance import FixedBaselineProvenance
from tests.fake_compact_cnn import FixedCompactCnnProvenance, RecordingCompactCnnAdapter
from tests.fake_dense_baseline import (
    FixedCudaRuntimeProbe,
    FixedDenseBaselineProvenance,
    FixedDenseBaselineRunner,
)
from tests.fake_feature_evaluation import RecordingFeatureBaseline
from tests.fake_report_provenance import FixedReportProvenance
from tests.fake_resnet_baseline import FixedResNetBaselineProvenance, FixedResNetBaselineRunner
from tests.report_inputs_fixture import CuratedInputsFixture
from tests.test_feature_confirmation_cli import _prepare_human_review, _run_confirmation


def prepared_comparison_fixture(root: Path) -> CuratedInputsFixture:
    """Build 132 fake baseline inputs; for example, comparison tests avoid CUDA."""
    fixture = CuratedInputsFixture(root, sample_count=132)
    contract_path, report_path = _prepare_human_review(fixture, ("area", "perimeter"))
    result, _, stderr = _run_confirmation(fixture, contract_path, report_path)
    if result != 0:
        raise AssertionError(f"feature confirmation returned {result}: {stderr}")
    _run_fake_baselines(fixture)
    return fixture


def _run_fake_baselines(fixture: CuratedInputsFixture) -> None:
    stdout, stderr = io.StringIO(), io.StringIO()
    result = main(
        ["baselines", "--config", str(fixture.config_path)], stdout=stdout, stderr=stderr,
        random_forest_baseline=RecordingFeatureBaseline(),
        baseline_provenance=FixedBaselineProvenance(),
        report_provenance=FixedReportProvenance(),
        dense_baseline_dependencies=DenseBaselineDependencies(
            FixedDenseBaselineRunner(), FixedDenseBaselineProvenance(), FixedCudaRuntimeProbe(),
        ),
        compact_cnn_adapter=RecordingCompactCnnAdapter(),
        compact_cnn_provenance=FixedCompactCnnProvenance(),
        resnet_baseline_runner=FixedResNetBaselineRunner(),
        resnet_baseline_provenance=FixedResNetBaselineProvenance(),
    )
    if result != 0:
        raise AssertionError(f"baseline command returned {result}: {stderr.getvalue()}")
