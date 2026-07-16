from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from buffalo_weight.pca_feature_fusion import PcaFeatureFusionRegressor


class CanonicalPcaFeatureFusionTest(unittest.TestCase):
    def test_fits_original_and_canonical_mask_branches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            masks_dir = Path(directory)
            rows = self._mask_rows(masks_dir)
            params = {
                "image_size": 16,
                "n_components": 2,
                "canonical_components": 2,
                "n_estimators": 10,
                "random_state": 42,
                "target_transform": "log",
            }
            model = PcaFeatureFusionRegressor(masks_dir, params)

            model.fit(rows, ["area"])
            predictions = model.predict(rows, ["area"])

        self.assertEqual(predictions.shape, (8,))
        self.assertTrue(np.isfinite(predictions).all())

    def _mask_rows(self, masks_dir: Path) -> list[dict[str, str]]:
        rows = []
        for index in range(8):
            mask = np.zeros((16, 16), dtype=np.uint8)
            mask[4:12, 2 : index + 6] = 255
            Image.fromarray(mask).save(masks_dir / f"mask-{index}.png")
            rows.append(
                {"file_name": f"mask-{index}", "weight": str(100 + index * 10), "area": str(mask.sum())}
            )
        return rows


if __name__ == "__main__":
    unittest.main()
