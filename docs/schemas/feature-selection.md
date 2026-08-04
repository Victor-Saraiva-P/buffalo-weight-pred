# Feature-selection evidence schemas

The reconstructible package lives at
`generated/report/feature_selection/`. It is provisional until a separate
human review confirms the Conjunto Compartilhado de Features.

`feature_redundancy.csv` has one deterministically ordered row for each of the
325 unordered pairs in the 26-feature candidate universe. Its columns are
`feature_a`, `feature_b`, `structural_relation`, `pearson`, `spearman`, and
`removal_group`. Correlations are dimensionless, use six decimal places, and
are empty only when the statistic is undefined. `structural_relation` is
`none` or a deterministic `|`-separated list of the frozen mathematical
relations (area transformations, Bounding Rectangle, Elipse Equivalente,
Ocupação Vertical Regional, Fecho Convexo da Máscara, and area–contour).
`removal_group` is one of the six frozen group names or `none`.

`feature_predictive_evidence.csv` has one row per fold result and one grouped
OOF row per experimental result. Its ordered columns are `experiment`,
`baseline`, `target`, `scope`, `fold`, `repetition`, `permutation_seed`, `n`,
`reference_mae_kg`, `result_mae_kg`, `delta_mae_kg`, and `effect`.
`experiment` is `isolated`, `removal`, or `permutation`; `baseline` is
`random_forest` or `dense`; `scope` is `fold` or `oof`. OOF rows have an empty
fold and permutation seed. Repetition is populated only for permutation rows.
MAE fields are kilograms with six decimals. `delta_mae_kg` is result minus the
complete-feature reference, so a negative value improves MAE. `effect` is
`improvement`, `neutral`, or `harm`, with exactly ±1 kg classified as neutral.

`shared_feature_contract.json` records all 26 candidates, every provisional
removal recommendation, the within-training standardization rule, and the
SHA-256 of the report draft. Its `status` is `provisional`; both
`selected_features` and `human_decision` are JSON null. The stage has no public
operation that can populate those fields automatically.

`redundancy_heatmap.png`, `removal_heatmap.png`, and
`permutation_effects.png` are fixed 2400×1800 PNG images at 300 DPI. The
Markdown report includes every recommended, neutral, and rejected withdrawal
and ends with an unfilled human-review record.

`manifest.json` is written last. It records the input-manifest hash, recipe and
dependency identity, source commit, exact output hashes, CSV schemas and row
counts, JSON keys, PNG dimensions and DPI, and the validations performed. Its
status is also `provisional`; a missing or divergent output makes the stage
absent or obsolete rather than reusable.

## Human confirmation

A human-authored contract and reviewed report are promoted together to
`evidence/confirmed/feature_selection/v1/`. Promotion never derives
`selected_features` from recommendations. The contract has exactly this shape:

```json
{
  "schema_version": 1,
  "status": "confirmed",
  "selected_features": ["area", "perimeter"],
  "standardization": "fit within each permitted training partition",
  "report_sha256": "<64 lowercase hexadecimal characters>",
  "human_decision": {
    "decision_url": "https://github.com/owner/repository/issues/16#issuecomment-...",
    "reviewer": "<reviewer name>",
    "reviewed_at": "2026-08-04"
  }
}
```

The selected list is non-empty, unique, limited to the 26 candidates, and a
subsequence of their canonical order. Every selected name must occur in
backticks in the reviewed report. The report must retain the
`Registro de revisão humana` section, set its status to `revisado`, and replace
all pending placeholders. `report_sha256` is the SHA-256 of that exact report.

Promotion accepts only an intact provisional package produced by the official
CPU/CUDA execution and a clean Git worktree. It copies the human contract
without filling fields, writes the confirmed manifest last, and records the
decision URL and contract hash. The baseline gate revalidates the confirmed
contract, report, evidence schemas, output hashes, and current input snapshot.
