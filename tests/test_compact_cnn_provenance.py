from __future__ import annotations

import unittest

from buffalo_weight.compact_cnn_provenance import (
    SystemCompactCnnProvenance,
    _recipe_source_symbols,
)


class CompactCnnProvenanceTest(unittest.TestCase):
    def test_recipe_hash_uses_only_prediction_affecting_symbols(self) -> None:
        symbols = _recipe_source_symbols()
        modules = {module.rsplit(".", maxsplit=1)[-1] for module, _ in symbols}

        self.assertIn("compact_cnn_network", modules)
        self.assertIn("compact_cnn_augmentation", modules)
        self.assertNotIn("compact_cnn_artifacts", modules)
        self.assertNotIn("compact_cnn_manifest", modules)
        self.assertNotIn("compact_cnn_stage", modules)
        self.assertNotIn("compact_cnn_provenance", modules)

    def test_recipe_hash_is_stable_and_hexadecimal(self) -> None:
        provenance = SystemCompactCnnProvenance()

        first = provenance.compact_cnn_recipe_hash()
        second = provenance.compact_cnn_recipe_hash()

        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)
        self.assertTrue(set(first) <= set("0123456789abcdef"))


if __name__ == "__main__":
    unittest.main()
