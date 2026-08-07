from __future__ import annotations

import unittest
from pathlib import Path

from buffalo_weight.tuning_provenance import SystemTuningProvenance
from tests.fake_tuning_provenance import FixedTuningProvenance


class FakeEnvironment:
    def read_source(self, path: Path) -> bytes:
        return f"content of {path.name}".encode()

    def distribution_version(self, name: str) -> str:
        return f"1.0.0-{name}"

    def repository_commit(self, root: Path) -> str:
        return "1234567890abcdef1234567890abcdef12345678"


class TuningProvenanceTest(unittest.TestCase):
    def test_fixed_provenance_returns_expected_fields(self) -> None:
        provenance = FixedTuningProvenance()
        self.assertEqual(len(provenance.tuning_recipe_hash()), 64)
        self.assertEqual(len(provenance.repository_commit()), 40)
        self.assertIn("torch", provenance.tuning_dependencies())

    def test_system_provenance_uses_environment(self) -> None:
        env = FakeEnvironment()
        provenance = SystemTuningProvenance(env)
        self.assertEqual(len(provenance.tuning_recipe_hash()), 64)
        self.assertEqual(provenance.repository_commit(), "1234567890abcdef1234567890abcdef12345678")
        self.assertEqual(provenance.tuning_dependencies()["torch"], "1.0.0-torch")


if __name__ == "__main__":
    unittest.main()
