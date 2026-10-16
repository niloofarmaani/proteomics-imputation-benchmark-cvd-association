# Validation record

Validated on 16 October 2026 in the supplied Linux workspace.

## Own-data companion validation

The separate `notebooks/impute_your_own_data.ipynb` calls the existing tuning,
evaluation, method adapters, and original paper renderer without editing them.
Its only additional stages prepare uploaded data/reference values, select a
method by frozen validation RMSE, and refit/export the original input. No
clinical association stage is called.

The notebook's explicit synthetic example was executed through the same local
in-process IPython route. Saved results are in
`notebooks/impute_your_own_data_executed.ipynb`; summary outputs and a synthetic
completed CSV are in `examples/own_data_demo/`.

- Input: 256 samples, 60 numeric variables, 1,054 native missing cells.
- Reference: 160 fully observed samples across the same 60 variables.
- All nine original implementations; 18 complete tuning trials, 72 validation
  condition scores, and 81 frozen final reconstruction fits.
- Iterative was selected using frozen validation RMSE before final scoring.
- Its final refit used all 256 input samples and filled all 1,054 missing
  measurements. The output retained all 61 columns including sample IDs.
- Observed measurement strings, leading-zero IDs, row order and column order
  were independently compared with the original input.
- Thirteen code cells completed without errors. The additional display-only
  filled-cell preview was executed against the same saved CSV and appended to
  the saved notebook without changing any numeric stages.
- Fourteen checks passed, including input/header checks, missing-reference
  handling, reference-column alignment, validation-only method selection,
  original-data refitting, preservation of observed values/metadata, and
  rejection of invalid completed matrices.
- The exported PNG and notebook figure were visually inspected; their labels
  identify the uploaded-data reference and the single-seed limitation.
- Existing files under `legacy/`, the core `reproducibility` workflow modules,
  and the installer/execution scripts were unchanged by this companion.

The first small synthetic example reached 18 trials and 80 reconstruction fits
but stopped on the original Transformer's `Training batch contains no observed
cells` check (97 reference rows, 12 variables, MNAR at 50%). This failure was
retained rather than patched. The final example uses the previously validated
160-by-60 reference size. Original method failures on other input configurations
remain visible and are not replaced with fallback algorithms.

Fresh Colab upload/download UI execution and the full-budget own-data run
remain unverified. Inputs without enough complete rows require a separate
complete reference table to retain the original benchmark protocol.

## User-saved Colab run reviewed

The notebook saved on GitHub in commit
`f65cb56148778b1a93f00b426aec0f255c28e303` was inspected, including its figures
and captured outputs. An unchanged copy is preserved as
`notebooks/benchmark_TCGA_Colab_run.ipynb`.

- The recorded source checkout is `f9e440444b0ea59803ccf81cc926c62e2b06af33`.
- All 12 code cells have execution counts and no saved error outputs.
- It records 18 completed tuning trials, 81/81 final reconstruction scores,
  81 association comparisons, and 60/60 successful reference protein models.
- The outcome balance is 97/63, as in the local demo. One reference feature
  passes FDR correction.
- The recorded PyTorch build is `2.11.0+cpu`. Complete installation logs and
  its output archive were not committed, so those cannot be independently
  audited from the notebook alone.

This is evidence of successful execution in the user's Colab session. It is
distinct from independently starting a new Colab runtime here.

## Figure 2 five-seed local validation

The updated tutorial was executed top-to-bottom with the same in-process
IPython route described below. Its saved outputs are in
`notebooks/benchmark_TCGA_executed.ipynb`; summary files and figures are in
`examples/five_seed_demo/`.

- 160 patients, 60 RPPA features, all nine methods, two tuning trials per
  method and a 30-epoch neural training limit.
- 18 completed tuning trials and 72 condition-specific validation scores.
- 405 completed final fits across seeds 42, 100, 2024, 777, 999.
- 81 method/mechanism/fraction boxes, each containing five actual final RMSE
  values. Counts and finite sample SDs were checked against the saved grid.
- 81 downstream comparisons at seed 42; zero failed imputed association fits.
- All 13 code cells executed; no saved error outputs.
- Seven integrity/plot checks passed, including rejection of incomplete and
  duplicate grids, sample-SD calculation, and undefined SD for one seed.
- Independent spot checks recalculated saved RMSE from the imputed matrices
  and masks, checked identifiers and unchanged observed entries, and confirmed
  that different seeds produce different masks.
- The exported PNG layout and saved notebook figure were inspected.
  Colours and ordering match the original renderer; public-data and seed labels
  are visible, and the shared RMSE range contains all plotted values.
- Original-source audit now has two documented changes: `config_paths.py`
  and optional title/note arguments in `paper_seed_rmse_boxplots.py`. No
  imputation algorithm was changed for this update. Plot provenance records
  its final renderer and adapter hashes separately from the run's initial
  source snapshot.

This is a five-seed reduced-budget public demo, not the full-panel
paper-budget run. The new five-seed version still requires a fresh Colab
execution check. The user's earlier one-seed Colab run is preserved above.

## Initial completed local public-data demo

- TCGA-UCEC: 160 patients, 60 RPPA features, recorded hypertension outcome.
- Hypertension classes: 97 yes, 63 no.
- All nine original imputation implementations completed.
- 18 completed Optuna trials, with 72 condition-specific validation scores.
- 81 final reconstruction fits: nine methods x three mechanisms x three rates.
- 81 downstream association-preservation comparisons at seed 42.
- 60/60 reference association models fitted; zero failed imputed association
  models across the comparison grid.
- One reference FDR-significant feature in this demonstration panel. FDR-set
  metrics are therefore coarse and depend heavily on that single feature.
- All completed matrices were finite, sample/protein identifiers remained
  aligned, and observed values were preserved. Final scoring excluded all
  observed entries and used only each saved simulation mask.
- Four integrity tests passed: deterministic/disjoint masks, strict handling
  of failed predictions, known association-metric identities and undefined
  denominators, and association input alignment.
- A separate focused integration run verified relative output paths,
  relocation of a run directory, completion checksums, and association rescoring.
- Original source hash audit: only `legacy/config_paths.py` differs on disk
  from the 99 supplied source files. The R identifier/seed fixes are applied
  to temporary adapter copies.
- Three rendered figures were inspected. Heatmap labels were adjusted to
  separate mechanism groups from fraction labels. Saved notebook image outputs
  were refreshed from the rerendered, unchanged numerical results.

The full common panel has 195 features. In the 60-feature demo, independent
random top-50 sets already have expected overlap 50/60 = 83.3%. High top-50
overlap in this small panel is not strong evidence by itself. Prefer signed
z-Spearman and top-20 overlap for interpreting the short example, and use the
larger panel when assessing top-50 preservation.

## Execution environment

Python 3.12; CPU PyTorch 2.14.1+cpu; NumPy 2.3.5; pandas 2.2.3; SciPy 1.17.0;
scikit-learn 1.8.0; Optuna 5.0.0; statsmodels 0.15.0; Matplotlib 3.10.8.

R 4.3.3; Metrics 0.1.4; impute 1.76.0; missRanger 2.6.1; ranger 0.16.0;
FNN 1.1.4.1; Rcpp 1.0.12; RcppEigen 0.3.4.0.0.
The full installed-package record is retained with the example outputs.

The workspace does not permit the sockets needed by a normal Jupyter kernel.
Both TCP and IPC startup were blocked. The notebook was therefore executed
cell by cell, in order, in a fresh in-process IPython session using
`scripts/execute_in_process.py`. Actual computation, tables, and figures are
saved in `notebooks/benchmark_TCGA_executed.ipynb`. This is a real sequential
execution of the notebook's Python cells, but is not a clean Jupyter-kernel or
Google Colab validation. Standard nbconvert execution is documented in README.

After the main demo, output-path serialization was made relative and verified
by the focused relocation test; this does not alter the numerical results.
Summary-table presentation and figure labels were refreshed without refitting.

## Remaining validation gaps

1. The new Figure 2/five-seed notebook changes have not been rerun in a fresh
   Google Colab session here. The user's earlier successful Colab run is
   preserved separately, as described above.
2. The full-panel, paper-budget five-seed search/evaluation has not been run.
3. The restricted MyoVasc inputs were not used or redistributed. No numerical
   reproduction of the MyoVasc CVD/plaque findings is claimed.
4. The unavailable R-library ZIP was not inspected. R dependencies were built
   separately for this validation environment. Compiled library directories
   are not a portable substitute for environment specification.
5. The dependency installer and a cross-runtime R version lock have not been
   independently verified on Colab. Actual tested versions are recorded above.

The clean notebook, source, and tests were prepared for the initial GitHub upload.
The source repository is `niloofarmaani/proteomics-imputation-benchmark-cvd-association`.


## Git-clone setup update

The clean notebook now clones the public source repository directly. Its bootstrap
was tested in a fresh local directory against the live GitHub repository, then
rerun to verify that the same checkout and commit are reused and local files
are preserved. The user's later saved run validates that bootstrap in Colab;
use `benchmark_TCGA_reproducible.ipynb` for current setup.
