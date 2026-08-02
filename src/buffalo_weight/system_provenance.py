"""JSON persistence adapter for official environment provenance."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from buffalo_weight.report_environment import EnvironmentProvenance


class JsonProvenanceWriter:
    def write(self, path: Path, provenance: EnvironmentProvenance) -> None:
        """Write provenance atomically; for example, replace only complete JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        partial_path = path.with_suffix(f"{path.suffix}.part")
        serialized = json.dumps(asdict(provenance), indent=2, sort_keys=True) + "\n"
        partial_path.write_text(serialized)
        partial_path.replace(path)
