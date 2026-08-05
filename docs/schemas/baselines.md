# Baseline evaluation schemas

The `baselines` stage writes one reconstructible artifact per configuration below
`generated/report/baselines/`. `random_forest_baseline` has
`evaluation_role=candidate`; `training_mean_reference` has
`evaluation_role=reference` and is not a fifth candidate. Both consume the same
Divisão Estratificada Canônica. The Random Forest consumes the ordered
`selected_features` from the confirmed human contract without a model-level
feature override.

## `predictions.csv`

The unit is one Predição OOF for one Máscara Válida and configuration. The key
is `file_name`; rows are ordered by that key. Every file appears exactly once in
each configuration.

| Column | Type / unit | Contract |
| --- | --- | --- |
| `configuration` | enum | `random_forest_baseline` or `training_mean_reference` |
| `evaluation_role` | enum | `candidate` or `reference` |
| `file_name` | text | Non-empty unique PNG basename |
| `weight_category` | enum | `B1` through `B10` |
| `fold` | integer | `1` through `5` |
| `observed_weight_kg` | kg | Finite, six decimal places |
| `predicted_weight_kg` | kg | Finite, six decimal places |
| `residual_kg` | kg | Prediction minus observed weight; positive means overestimation |
| `absolute_error_kg` | kg | Absolute value of `residual_kg` |

## `fold_metrics.csv`

The unit is one configuration and canonical fold, ordered by `fold`. Columns are
`configuration`, `evaluation_role`, `fold`, `n`, `mae_kg`, `rmse_kg`,
`bias_kg`, and `r2`. `bias_kg` is the mean signed residual. Numeric derived
fields use six decimal places.

## `grouped_metrics.csv`

The unit is one configuration and OOF population. Columns are `configuration`,
`evaluation_role`, `population`, `n`, `mae_kg`, `rmse_kg`, `bias_kg`, and `r2`.
Rows are ordered as `all`, `B1`, and `B10`. The `all` row is calculated directly
from all 132 pooled OOF predictions, never as an average of fold metrics.

## `manifest.json`

The manifest is written last. It records configuration and role, the ordered
confirmed feature contract, normalized report contract, fold seed 42, training
seed 44, pertinent input, recipe and dependency identities, and the SHA-256,
schema and row count of all three CSV outputs. Input hashes project only the
rows and columns consumed by each configuration. A missing or divergent
identity, schema, row count or output hash makes only that configuration
obsolete. The obsolete snapshot is removed before retraining, so a failed run
cannot leave stale evidence published. Changing Random Forest-specific
knowledge or features does not invalidate the training-mean reference.

## Controlled comparison package

`python main.py compare-baselines` reads only complete, reusable artifacts for
the four candidates and the training-mean reference. It refuses to train or
repair a configuration. The provisional package is written atomically to
`generated/report/approach_selection/` and contains the files below.

### `baseline_metrics.csv`

The unit is one configuration, metric scope and population. The key is
`configuration`, `scope`, `fold`, `population`. Rows are ordered by Random
Forest, Rede Densa por Feições, compact CNN, ResNet-18 and finally the
training-mean reference. Within each configuration, folds 1 through 5 precede
the pooled OOF rows `all`, `B1` and `B10`.

| Column | Type / unit | Contract |
| --- | --- | --- |
| `configuration` | enum | `random_forest_baseline`, `dense`, `compact_cnn`, `resnet18_pretrained_partial`, `training_mean_reference` |
| `approach` | enum | `random_forest`, `dense_feature_network`, `compact_cnn`, `resnet18`, `training_mean` |
| `evaluation_role` | enum | `candidate` or `reference`; only the latter applies to `training_mean_reference` |
| `scope` | enum | `fold` or `oof` |
| `fold` | nullable integer | `1` through `5` for `fold`; empty for `oof` |
| `population` | enum | `all`, `B1` or `B10`; fold rows use only `all` |
| `n` | integer | Number of Predições OOF in the row |
| `mae_kg` | kg | Direct mean absolute error, six decimal places |
| `rmse_kg` | nullable kg | Direct RMSE for `all`; empty for B1/B10 |
| `bias_kg` | kg | Mean of prediction minus observed weight; positive means overestimation |
| `r2` | nullable ratio | Direct R² for `all`; empty for B1/B10 |

The pooled `all` row uses all 132 public, six-decimal predictions and is never
an average of fold metrics. B1/B10 expose only the predeclared descriptive MAE
and bias.

### Review artifacts

`global_mae.png`, `predicted_vs_observed.png` and
`residuals_vs_observed.png` are 2400×1800 PNG figures at 300 DPI. They show only
the four candidates; the reference remains in `baseline_metrics.csv` and the
report. `approach_selection_report.md` uses the term MAE OOF Pós-Seleção,
records limitations and a revisable lowest-MAE recommendation, and leaves the
human decision unfilled.

`selected_approach.json` is a provisional review template with
`schema_version=1`, the four compatible approach/configuration pairs, a tuning
budget ceiling of three, the report hash and `human_decision=null`. It is not a
confirmed approach contract.

### Provisional `manifest.json`

The manifest has `package_type=provisional_evidence`,
`stage=approach_selection`, `status=provisional`, `revision=1` and
`decision_url=null`. It binds the confirmed feature package and every consumed
baseline manifest and prediction table by SHA-256, plus the comparison recipe,
dependencies, source commit, output schemas, row counts and hashes. It is
written last and remains reconstructible until the separate human gate promotes
a reviewed package.
