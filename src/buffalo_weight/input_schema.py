"""Machine-readable schemas for reconstructed input artifacts."""

from __future__ import annotations

from buffalo_weight.feature_calculators import APPROVED_FEATURES

FEATURE_COLUMNS = ["file_name", "farm", "weight_kg", *APPROVED_FEATURES]
SPLIT_COLUMNS = ["file_name", "farm", "weight_kg", "weight_category", "fold"]
OUTPUT_SCHEMAS = {
    "feature_index.csv": FEATURE_COLUMNS,
    "canonical_split.csv": SPLIT_COLUMNS,
}
