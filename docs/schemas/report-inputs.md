# Report inputs schemas

The versioned operational input is `data/mask_index.csv`, encoded as UTF-8 with
the exact columns `file_name`, `farm`, and `weight_kg`. It has one row for each
of the 132 manually approved Máscaras Válidas. `file_name` is a unique PNG
basename, `farm` is non-empty text, and `weight_kg` is a finite number greater
than zero. `data/indice.xlsx` is retained only as immutable upstream history;
the report pipeline does not read it.

`generated/report/inputs/feature_index.csv` has one row per `file_name`, ordered
by that key. Its first three columns reproduce the operational index. They are
followed, in glossary order, by the 26 features in the Universo de Features
Candidatas. Numeric derived values use six decimal places. Lengths use canonical
pixels, areas canonical pixels squared, geometric volume proxies canonical
pixels cubed, and ratios and invariants are dimensionless.

`generated/report/inputs/canonical_split.csv` has one row per `file_name`, in the
same deterministic order, with `file_name`, `farm`, `weight_kg`,
`weight_category`, and `fold`. Categories are `B1` through `B10`; folds are the
strings `1` through `5` generated with seed 42.

`manifest.json` is written after both CSVs pass their schema, row-count,
one-to-one, hash, and fold-distribution checks. It records the normalized
contract, recipe hash, hashes of every operational input and output, output row
counts, and validation names. A missing or divergent manifest makes the stage
absent or obsolete rather than reusable.
