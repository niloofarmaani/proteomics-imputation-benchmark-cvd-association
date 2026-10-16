# Validation record

Validated on 16 October 2026 in the supplied Linux workspace.

## Completed public-data demo

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

1. No clean Google Colab session has been run in this delivery environment.
   Run the clean notebook from a fresh Colab runtime before a public release.
2. The paper-budget five-seed search/evaluation has not been run.
3. The restricted MyoVasc inputs were not used or redistributed. No numerical
   reproduction of the MyoVasc CVD/plaque findings is claimed.
4. The unavailable R-library ZIP was not inspected. R dependencies were built
   separately for this validation environment. Compiled library directories
   are not a portable substitute for environment specification.
5. The dependency installer and a cross-runtime R version lock have not been
   independently verified on Colab. Actual tested versions are recorded above.

The clean notebook, source, and tests were prepared for the initial GitHub upload.
The private repository is `niloofarmaani/proteomics-imputation-benchmark-cvd-association`.
A fresh Google Colab run and a public release remain pending.
