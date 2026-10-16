# Method and reproducibility decisions

## What this package reproduces

The public notebook executes the supplied nine method implementations and the
same two-stage benchmark structure: select one hyperparameter configuration
per method using additional observed-cell holdouts, freeze those settings,
evaluate reconstruction, and measure association preservation. It does not
claim numerical reproduction of the MyoVasc paper.

## Original source lineage

The supplied `imputation-benchmark-bmc-submission (1).zip` was the source for
99 Python/R files retained in `legacy/`. Their original SHA-256 hashes are
recorded in `original_source_sha256.json`. The only on-disk source edit is
`legacy/config_paths.py`: its hardcoded scratch path was replaced by the
`IMPUTATION_DATA_ROOT` environment variable, defaulting to a local `work/`
directory. Empty package initializers were added where needed.

Two R adapters apply narrowly scoped patches to temporary copies at runtime:

- KNN: preserve antibody identifiers with `check.names=FALSE` and pass the
  explicit seed to `impute.knn(rng.seed=...)`.
- MissForest: restore original antibody identifiers after `as.data.frame()`
  sanitizes them. The estimator remains `missRanger`.

These avoid silent column mismatches. The original source files remain
unchanged for comparison. Archived preprocessing, evaluation, and optimization
scripts remain available but the public notebook calls the new runner.

The original benchmark notebooks are not included: their saved outputs and
associated archive files contain private-cohort data/results. The new notebook
is an additional public companion, not an overwrite of those originals.

## Public input construction

- Study: cBioPortal `ucec_tcga`, Firehose Legacy; non-z-score RPPA source values.
- Patient phenotypes: GDC TCGA-UCEC Biotab patient file plus follow-up versions
  1.7, 2.0, and 4.0, all identified by UUID in `configs/data_sources.json`.
- One primary tumor sample per patient, selected by sorted sample ID.
- Of 208 antibody features, 13 have native batch-related omissions. Retain the
  195-feature common complete panel before selecting any demo subset.
- Clinical yes/no values map to 1/0. All available recorded values must agree;
  discordant patients are excluded for that phenotype. Missing records do not
  imply absence of disease. Five patients have discordant hypertension records
  across the complete clinical source, and two have discordant diabetes records.
- 380 of 440 RPPA patients have a usable hypertension status with this rule.
- Demo selection: deterministic sample subset (160, seed 42) and first 60
  alphabetically ordered retained antibody identifiers. No association testing
  is used to select features or patients.
- Tumor RPPA, including phosphoproteins, is not circulating Olink proteomics.
  Timing across baseline and follow-up is unharmonized. No CVD, plaque, myocardial
  infarction, stroke, or cardiovascular mortality endpoint has been verified.

The notebook reports class balance, covariate missingness, discarded features,
and exclusions. Data availability and complete-panel selection limit transport
to the original population and assay panel. Downloads are checksum locked.
An upstream data change triggers an error and requires an explicit source-lock
review rather than a silent refresh.

## Missingness adaptation

The paper describes Jenga 0.0.1a1. Inspection of that implementation shows that
MAR chooses another column from a Python `set`, making the order dependent on
process hash randomization. The supplied preprocessor also lets the corruptor
see metadata columns and previously corrupted columns when selecting/sorting a
MAR driver.

The new implementation retains Jenga's MCAR/rank-block procedures but uses a
stable protein-column order and immutable protein-only reference values for
sorting. Exactly `floor(n*f)` cells per protein are removed. This avoids IDs,
outcomes, and previous NaNs affecting the missingness generator. It changes
the generated masks, so this public adaptation is not bitwise paper-mask
reproduction. Every generated mask is saved.

The MAR-labelled source protein can itself become incomplete. The rank-block
procedure alone does not prove Rubin MAR conditional on the final observed
matrix. MNAR rank blocks do not necessarily select low-abundance values and
are not an explicit below-limit-of-detection model. Interpret labels according
to their implemented definitions.

## Tuning and fitting

The four tuning conditions are MAR/MNAR x 10%/30%, seed 42. Extra validation
omissions are sampled only from still-observed values, quantile-stratified per
protein. Every method/trial uses identical masks. Bin-wise ceiling causes the
realized validation fraction to exceed 5% in small cohorts. The objective is
the equally weighted mean RMSE over these four conditions.

Search spaces are from the original `Optimization/method_spaces.py`, except
infeasible SVD ranks are removed for small panels. The paper trial budgets are:
KNN/AE/DAE/VAE/MissForest 20; Iterative 12; SoftImpute/GAIN 15; Transformer 6.
Optuna's default ten-trial startup means the two-trial demo and six-trial
Transformer search are random-startup searches, despite the TPE sampler.

Neural internal early stopping/refit remains as implemented in the archived
source. The imputer adapter accepts only the incomplete protein matrix. Hidden
simulation/tuning values and outcomes cannot be passed through that interface.
Because legacy APIs require a complete reference argument for diagnostics, the
adapter supplies column-mean surrogates derived from its observed inputs.
Scores in those legacy logs are diagnostic surrogate scores. The new pipeline
calculates all reported validation/final RMSE externally from true heldout
values. Final association metrics never select hyperparameters.

The benchmark is transductive reconstruction in a retained cohort, as in the
paper. It does not estimate prediction on new patients. Neural settings,
algorithm seeds, original source, environment, and masks are recorded. GPU
floating-point behavior may differ from the CPU validation environment.

## Association models and preservation

The public hypertension models adjust for age, diabetes, and BMI. Missing age
and BMI use their median; missing diabetes uses its most frequent category.
Sex is constant, and hypertension cannot be both outcome and adjustment.
Other paper covariates were not verified for this example. Diabetes is a
recorded-status covariate with the same timing limitation as hypertension.

One fixed design and identical patients are used in all reference/imputed
scans. Proteins are standardized separately within each matrix, with sample
standard deviation. Binomial GLM, logit link, and Wald inference follow the
paper's approach. At least 50 patients and both outcome classes are required.
Rank-deficient, nonconverged, and separated models produce explicit failure
statuses. BH adjustment uses finite p-values within each scan; failed fits
remain visible. Metrics disclose the evaluable and failed protein counts.

The metrics include beta/z Spearman correlations, top-20/top-50 overlap,
coefficient-sign concordance among reference top-k proteins, beta RMSE,
FDR-set Jaccard, and reference-FDR retention. A top-k metric is undefined if
either scan has fewer than k fitted proteins. Empty FDR-set denominators are
undefined, not zero or one. Association analyses use only final simulation
seed 42. Averaging conditions does not give replicate uncertainty.

## Exact paper reproduction requires additional inputs

Exact MyoVasc numerical reproduction requires authorized original protein and
phenotype inputs, verified column mappings, original masks or original process
hash seed, all frozen settings, the original dependency environment, and the
final analysis scripts. Those inputs cannot be reconstructed from a public TCGA
cohort. No supplied private data or cached paper outputs are redistributed here.

## R environments

The public workflow needs `Metrics`, `impute`, `missRanger`, and their dependencies.
The previous Colab notebook's `glmnet`, `mice`, `caret`, `VIM`, `pcaMethods`,
`randomForest`, and other packages are not needed by these nine adapters.
The missing `colab_R_libs2-20261016T074758Z-1-001.zip` attachment was not inspected.
The installer therefore does not rely on its compiled packages. Installers and
recorded versions are provided, but a cross-runtime R lockfile validated in
Google Colab remains a release task. `R-session.txt` records the actual run's
R version and package versions.
