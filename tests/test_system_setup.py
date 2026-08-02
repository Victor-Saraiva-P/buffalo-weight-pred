from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from buffalo_weight.report_environment import (
    ComputeEnvironment,
    EnvironmentProvenance,
    InformationalEnvironment,
    ScientificValidity,
    WeightSetupStatus,
)
from buffalo_weight.system_setup import HttpWeightGateway, JsonProvenanceWriter


class SystemSetupTest(unittest.TestCase):
    def test_weight_gateway_reuses_cache_with_expected_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "weights.pth"
            cache_path.write_bytes(b"official weights")
            expected = hashlib.sha256(b"official weights").hexdigest()

            status = HttpWeightGateway().ensure_resnet18_weights(cache_path, expected)

            self.assertEqual(status, WeightSetupStatus.REUSED)

    def test_weight_gateway_rejects_existing_cache_with_wrong_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "weights.pth"
            cache_path.write_bytes(b"corrupt weights")

            with self.assertRaisesRegex(ValueError, "ResNet-18 cache SHA-256.*expected"):
                HttpWeightGateway().ensure_resnet18_weights(cache_path, "0" * 64)

    def test_provenance_writer_serializes_validity_and_information_separately(self) -> None:
        provenance = EnvironmentProvenance(
            ScientificValidity("3.14", {"numpy": "2.5.0"}, "IMAGENET1K_V1", "abc"),
            InformationalEnvironment(
                "3.14.6",
                "CPython",
                "Linux",
                {"numpy": "2.5.0", "nvidia-cublas": "13.0"},
                ComputeEnvironment("Fake GPU", "9.0", "13.0", "590.00"),
            ),
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "environment.json"
            JsonProvenanceWriter().write(path, provenance)
            record = json.loads(path.read_text())

        self.assertEqual(record["validity"]["python_series"], "3.14")
        self.assertEqual(record["informational"]["python_version"], "3.14.6")
        self.assertEqual(record["informational"]["compute"]["driver_version"], "590.00")


if __name__ == "__main__":
    unittest.main()
