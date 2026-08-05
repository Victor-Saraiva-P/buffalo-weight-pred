# Dense feature baseline artifacts

The Rede Densa por Feições baseline is stored under
`generated/report/baselines/dense/`. The directory is reconstructible and is
complete only when its `manifest.json` validates the two CSV outputs below.

## `predictions.csv`

One row represents one OOF prediction for one Máscara Válida. `file_name` is
the unique key and rows sort by it. `model_config` is `dense`, `approach` is
`dense_feature_network`, `fold` is an integer from 1 through 5, and
`weight_category` is B1 through B10. Observed and predicted weights,
`residual_kg` (prediction minus observation), and `absolute_error_kg` are
finite kilograms written with six decimal places.

## `fold_metrics.csv`

One row represents one metric population within either an external `fold` or
the grouped `oof` predictions. Rows sort by fold before OOF and by population
`all`, `B1`, then `B10`. `fold` and `selected_epochs` are null only for OOF
rows. `n` is a positive integer; MAE, RMSE, and signed bias are finite
kilograms with six decimal places. `r2` is nullable when fewer than two
observations or zero target variance make it undefined.

## `manifest.json`

The manifest is written last. It records the ordered confirmed features, the
complete frozen training recipe, scientific environment, source and input
hashes, output schemas and hashes, deterministic CUDA execution, and per-fold
hashes and counts for inner selection, stopping, full external refit, and the
reserved fold. Hardware details are audit information; the scientific
environment and recipe participate in cache validity.
