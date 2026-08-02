from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from buffalo_weight.hashing import sha256_file


class HashingTest(unittest.TestCase):
    def test_hashes_file_bytes_with_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.txt"
            path.write_bytes(b"abc")

            digest = sha256_file(path)

            self.assertEqual(
                digest,
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )


if __name__ == "__main__":
    unittest.main()
