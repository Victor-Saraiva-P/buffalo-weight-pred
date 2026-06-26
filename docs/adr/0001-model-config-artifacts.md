# Model configuration artifacts

Model evaluations use named Configurações de Modelo as the primary identity for outputs, with the base Modelo Clássico de Predição recorded as metadata. Each evaluation flow writes per-configuration artifacts under `generated/<flow>/<model_config>/` and keeps only general or comparative artifacts at the flow root, so predictions and per-model plots do not mix while the root still contains the single comparison needed to choose between configurations.

**Considered Options**: A single combined CSV was easier to aggregate, but made per-configuration inspection noisier. Directories by base model were rejected because multiple Configurações de Modelo can share the same model type with different hyperparameters.
