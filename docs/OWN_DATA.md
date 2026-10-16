# Use the existing workflow on your own data

[Open the own-data notebook in Colab](https://colab.research.google.com/github/niloofarmaani/proteomics-imputation-benchmark-cvd-association/blob/main/notebooks/impute_your_own_data.ipynb)

The notebook accepts CSV/TSV files with samples in rows and numeric variables
in columns. Set the optional ID column and list metadata/label columns to keep
outside imputation. Variable values are used on their supplied scale. There
is no additional normalization, categorical encoding, or automatic variable
selection.

## Existing workflow retained

| Stage | Existing implementation called |
|---|---|
| Dependency installation | `scripts/install_colab.sh` and `scripts/install_r.R` |
| Execution budget | `reproducibility.pipeline.Settings` / `Settings.paper_budget()` |
| Missingness and validation masks | Existing `corrupt`, `validation_mask`, `combo_seed` |
| Hyperparameter optimization | Existing `reproducibility.pipeline.tune()` |
| Imputation algorithms/search spaces | Existing `reproducibility.methods` and `legacy/` |
| Frozen final reconstruction | Existing `reproducibility.pipeline.evaluate()` |
| Paper-style comparison | Existing `legacy/Evaluation/paper_seed_rmse_boxplots.py` |
| Fit selected method on original input | Existing `reproducibility.methods.impute()` |

The existing workflow source files are unchanged by this companion. New upload,
reference, selection, and export helpers are isolated in
`reproducibility/user_data.py`. Clinical association analysis is omitted.

Tuning remains MAR/MNAR at 10%/30%, with seed 42 and additional observed-value
validation holdouts. One configuration per method is frozen using the existing
objective. Final comparison remains MCAR/MAR/MNAR at 10%/30%/50%. Rank-block
missingness retains the existing public workflow's documented limitations.

`demo` uses two trials per method, 30 neural epochs, and one final seed. Those
trials remain within Optuna's random startup phase and are intended for an
execution check. `paper_budget` uses the existing method-specific trial counts,
300 epochs, and final seeds 42, 100, 2024, 777, 999. The optional five-seed quick
budget changes only the explicitly configured seed list.

## Complete reference measurements are required

The original simulation benchmark needs true values at artificially masked
entries. It cannot calculate reconstruction error at an unknown native missing
value. This requirement is preserved.

Choose either:

1. Fully observed rows across **all selected variables** of the uploaded
   dataset. Every such row is used; no smaller panel is silently selected.
2. A separate complete reference table with exactly the same variables,
   measurement units, and preprocessing. Headers are matched by name, so its
   variable order may differ.

The companion requires at least 50 complete reference rows. Incomplete reference
tables or insufficient complete rows produce a clear error. Reference values
are never pre-imputed or replaced with predictions before benchmarking.

This restriction matters for sparse data with no complete rows: supply a genuine
complete reference table to use this unchanged workflow. Supporting such data
through native observed-value cross-validation would be a different benchmark
protocol and is deliberately not substituted here.

Complete-case selection may be unrepresentative, and a separate reference may
not match the target population. Rankings apply to that reference, its scales,
and the selected tuning budget. They do not prove accuracy at unknown values in
the original input. Larger-scale variables can dominate the unchanged RMSE
objective; prepare compatible scales before upload when scientifically appropriate.

## Selecting and exporting the imputed dataset

The method with the lowest frozen validation RMSE is selected **before** final
reconstruction evaluation. Final benchmark scores are presented for comparison
and do not choose the winner. Ties follow the configured method order.

The selected original method is then refitted on all rows and variables of the
uploaded dataset, using its actual native missing values, the frozen parameters,
and seed 42. This final fit uses the complete original input shape, even if only
complete rows were used as the benchmark reference. No test/reference truth or
metadata is supplied to the imputer.

`best_imputed_dataset.csv` preserves original row/column order, sample IDs,
excluded metadata, and the strings of observed measurements. Only originally
missing numeric measurements are filled. An internal identifier mapping protects
leading-zero IDs and arbitrary headers from legacy CSV/R name handling.

If the original input is already complete, its completed CSV is unchanged and
the export record states that no extra final fit was needed. A fully missing
variable is rejected because the unchanged adapters require observed values.
Errors from the original methods are retained rather than replaced with a
fallback estimator.

## Downloadable outputs

- `best_imputed_dataset.csv`: original table with missing measurements filled.
- `best_method.json`: method, validation score, frozen parameters, fit seed,
  filled-cell count, and output checksum.
- `method_selection.csv`: validation-based method ranking.
- Original tuning trials, condition scores, Optuna database, and frozen settings.
- Original final reconstruction scores, exact masks, completed benchmark matrices,
  logs, per-seed values, and paper-style PDF/PNG figures.
- Input/reference audit, variable-name mapping, source hashes and environment.
- A ZIP containing the complete run, with an optional second download control.

The notebook creates results only in its runtime. User inputs, completed datasets,
and run archives are not committed to GitHub by the notebook.

## Testing

The companion has a labelled synthetic example with 256 samples and 60 numeric
variables. The first 160 rows are kept complete and native omissions are introduced
into later rows, allowing the reference benchmark and final original-data
imputation to be tested separately. IDs have leading zeros. The synthetic
example is for execution validation, not evidence about imputation performance.

For local execution, use the repository's existing in-process notebook runner:

```bash
IMPUTATION_INPUT_MODE=example python scripts/execute_in_process.py \
  --input notebooks/impute_your_own_data.ipynb \
  --output notebooks/impute_your_own_data_executed.ipynb
```

Fresh Colab upload/download and full-budget runs still need separate validation.

The original Transformer can reject a training batch containing no observed
values under heavy simulated missingness. This was observed in the first
small example (97 reference rows, 12 variables, MNAR at 50%). Its implementation
was not modified. The companion retains original method errors and logs; it
does not silently drop rows, change batching, or substitute another estimator.
