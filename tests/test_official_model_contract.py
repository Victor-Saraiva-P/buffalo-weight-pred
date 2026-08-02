from __future__ import annotations

import unittest

from buffalo_weight.models import ModelConfig
from buffalo_weight.official_model_contract import validate_official_model_configs


class OfficialModelContractTest(unittest.TestCase):
    def test_rejects_xgboost_outside_optional_environment(self) -> None:
        configs = [ModelConfig("xgboost_baseline", "xgboost", {"n_estimators": 100})]

        with self.assertRaisesRegex(
            ValueError, "xgboost_baseline.*xgboost.*eight approved dependencies"
        ):
            validate_official_model_configs(configs)

    def test_accepts_model_backed_by_official_dependencies(self) -> None:
        configs = [ModelConfig("random_forest_baseline", "random_forest", {})]

        validate_official_model_configs(configs)


if __name__ == "__main__":
    unittest.main()
