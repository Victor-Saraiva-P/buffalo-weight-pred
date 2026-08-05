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
