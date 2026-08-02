---
status: superseded by ADR-0015
---

# Use a focused report-reproduction pipeline

The repository will support only the workflow needed to validate the 132 curated inputs, derive the approved geometry features and OOF split, evaluate the selected Random Forest and mask-pixel ResNet-18, and generate their report-ready evidence. It will no longer act as a generic model-experiment platform, because preserving extensibility for discarded model families adds configuration, cache, documentation, and testing costs without contributing to the final report.
