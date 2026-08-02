from __future__ import annotations

import sys
from pathlib import Path


# The project intentionally has no package metadata, so the root CLI exposes src directly.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from buffalo_weight.report_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
