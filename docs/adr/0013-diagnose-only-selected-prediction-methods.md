---
status: superseded by ADR-0015
---

# Diagnose error using only selected prediction methods

Model-dependent explanations of the high MAE will be recalculated for the selected Random Forest and ResNet-18 rather than inherited from discarded hybrid or ensemble predictions. Dataset-level evidence such as weight coverage, farm confounding, mask similarity, upstream segmentation validation, and manual B10 inspection may remain, while diagnostics whose conclusions depend on removed models will be deleted or replaced.
