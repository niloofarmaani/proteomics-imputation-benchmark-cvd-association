# Five-seed public-data demo

TCGA-UCEC, 160 patients and 60 RPPA features; two tuning trials per method,
30 neural epochs; final seeds 42, 100, 2024, 777, 999.

All 18 tuning trials, 405 final reconstruction fits, and 81 association
comparisons completed. Associations use seed 42. Each of the 81 Figure 2
method/condition boxes contains five actual final reconstruction scores.
The full-panel paper-budget search has not been run.

See `../../notebooks/benchmark_TCGA_executed.ipynb` and
`../../docs/VALIDATION.md`. The figure uses the original paper renderer and
is labelled as TCGA-UCEC. These are not MyoVasc numerical results.

This directory contains summary tables and figures. Matrix paths in
reconstruction.csv refer to the full run archive exported by the notebook;
completed matrices and masks are not bundled here. run.json preserves the
source snapshot at initialization and separately records the plotting source
used for final postprocessing.
