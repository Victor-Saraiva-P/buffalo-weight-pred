---
status: superseded by ADR-0015
---

# Preserve geometry and mask-pixel prediction evidence

The Núcleo Reprodutível do Relatório will retain both the Random Forest evaluation using ten pure-geometry features and the ResNet-18 evaluation using pixels derived from the binary mask. The first is an interpretable baseline and the second is the best observed predictor, so their comparison communicates the contribution of learned mask shape without preserving the full history of discarded experiments.
