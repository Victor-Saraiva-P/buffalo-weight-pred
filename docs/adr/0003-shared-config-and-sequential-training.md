# Shared config and sequential training

Training uses one shared config for data, generated artifacts, split settings, and the training output root, plus separate model configs for Modelo Clássico de Predição and Modelo de Predição por Máscara. `make train` runs those families sequentially and then writes one aggregated comparison under `generated/train/`, because both families must share the same Divisão Estratificada while keeping their input contracts separate: classical models use the Índice de Features, and mask models read Máscara Binarizada pixels directly.

**Considered Options**: A single mixed config was rejected because it made `cnn_mask` appear to depend on `feature_columns`. Fully duplicated configs were rejected because the shared `split_path` could drift and make comparisons unfair. Separate top-level commands without orchestration were rejected because they made the final comparison easy to forget or generate from stale model outputs.
