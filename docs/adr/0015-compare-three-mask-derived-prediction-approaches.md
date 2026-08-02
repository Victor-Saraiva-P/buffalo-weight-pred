---
status: superseded by ADR-0016
---

# Compare three mask-derived prediction approaches

The final controlled evaluation will compare a classical model over automatically calculated mask features, a dense neural network over exactly the same Conjunto Compartilhado de Features, and a convolutional neural network over spatial mask representations. Model-specific feature subsets are outside this comparison because they would mix the effect of feature selection with the effect of model class. This three-approach design replaces the earlier two-method scope, aligns the deep-learning feature approach with the supervisor's direction, and keeps all prediction inputs derived from the same binary masks; model-dependent error diagnostics will cover all three selected approaches.

Rede Densa por Feições is trained from a random Xavier/He initialization; pretrained tabular foundation models such as TabPFN are outside this effort because they define a different model class and reproducibility contract. Pretrained visual backbones remain a question only for Rede Convolucional por Máscara, whose input representation is compatible with transferred image features.
