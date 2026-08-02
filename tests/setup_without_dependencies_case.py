from __future__ import annotations

import importlib.util
import io
import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from buffalo_weight.report_environment import APPROVED_DEPENDENCIES, WeightSetupStatus
from buffalo_weight.report_cli import main
from tests.fake_setup_services import (
    FakePackageGateway,
    FakeWeightGateway,
    RecordingProvenanceWriter,
    setup_services,
)


class SetupWithoutDependenciesTest(unittest.TestCase):
    def test_public_setup_installs_before_torch_is_importable(self) -> None:
        self.assertIsNone(importlib.util.find_spec("torch"))
        packages = FakePackageGateway({})
        services = setup_services(
            packages,
            weights=FakeWeightGateway(result=WeightSetupStatus.REUSED),
            writer=RecordingProvenanceWriter(),
        )

        result = main(["setup"], services, stdout=io.StringIO(), stderr=io.StringIO())

        self.assertEqual(result, 0)
        self.assertEqual(packages.installed, APPROVED_DEPENDENCIES)


if __name__ == "__main__":
    unittest.main()
