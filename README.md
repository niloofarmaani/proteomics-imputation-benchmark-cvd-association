# Proteomics imputation: reproducible public-data companion

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/niloofarmaani/proteomics-imputation-benchmark-cvd-association/blob/main/notebooks/benchmark_TCGA_reproducible.ipynb)

Nine imputation methods, validation-based tuning, reconstruction benchmarking,
and downstream protein-association preservation in a portable Jupyter/Colab workflow.

Start with [`notebooks/benchmark_TCGA_reproducible.ipynb`](notebooks/benchmark_TCGA_reproducible.ipynb).
The accompanying executed notebook records the validation run described in
[`docs/VALIDATION.md`](docs/VALIDATION.md).

## Public example and scientific scope

The example joins **TCGA-UCEC primary tumor RPPA** measurements to GDC recorded
hypertension status. The locked source files yield **380 eligible patients and
195 features** in the common fully observed assay panel. Hypertension records
are combined across baseline and follow-up; patients with discordant yes/no
records are excluded. Timing is not harmonized. This is an exploratory
recorded-status association, not an incident cardiovascular endpoint.

Hypertension is a cardiovascular risk factor. Tumor RPPA differs from plasma
Olink proteomics. This public example demonstrates the methods and metrics;
it **does not reproduce the MyoVasc CVD/plaque findings**. Missing clinical
statuses are not classified as controls, and tumor/normal labels are not used
as cardiovascular labels.

## Open in Colab

1. Click **Open in Colab** above.
2. Run the first code cell. It clones the public repository into
   `/content/imputation-reproducibility` and prints the source commit.
3. Run the dependency setup cell. If it reports that already loaded packages
   changed, restart the Colab session once, then rerun from the beginning.
4. Continue through the demo cells in order. The notebook downloads verified
   public data, tunes all nine methods, compares reconstruction and associations,
   and exports its results.

No ZIP download/upload, Google Drive mount, or saved R-library archive is required.
Rerunning the first cell reuses the current checkout; it does not overwrite
local changes or silently pull new code into an active experiment. Start a
fresh Colab session to clone a later repository version. See [COLAB.md](COLAB.md).

The user's completed Colab demo is preserved with its outputs in
[`benchmark_TCGA_Colab_run.ipynb`](notebooks/benchmark_TCGA_Colab_run.ipynb).
It completed 18 tuning trials, 81 reconstruction fits, and 81 association
comparisons at seed 42, with no saved cell errors. The current tutorial also
generates Figure 2 style plots with the paper's five seeds; see
[docs/VALIDATION.md](docs/VALIDATION.md).

## Impute your own data

For your own CSV/TSV, use the separate
[**Impute your own data** Colab notebook](https://colab.research.google.com/github/niloofarmaani/proteomics-imputation-benchmark-cvd-association/blob/main/notebooks/impute_your_own_data.ipynb).
It calls the existing nine-method tuning, frozen reconstruction, and plotting
workflow unchanged, omits clinical associations, and downloads the original
table completed by the validation-selected method. It uses complete rows or a
separate complete reference table for the existing benchmark. See
[docs/OWN_DATA.md](docs/OWN_DATA.md) for input structure and this reference
requirement. User uploads/results are not committed to GitHub by the notebook.

## Local run

Use Python 3.12, R, and a C/C++ toolchain for R dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
Rscript scripts/install_r.R
python -m jupyterlab notebooks/benchmark_TCGA_reproducible.ipynb
```

Install `jupyterlab` separately if you prefer that editor; it is not required
for command-line notebook execution. On Linux, install R with your package
manager, for example `sudo apt-get install r-base r-base-dev`.

## Modes

| Setting | Demo, default | Paper-budget public example |
|---|---|---|
| Patients/features | 160 / 60 | 380 / 195 |
| Tuning panel | MAR/MNAR, 10%/30%, seed 42 | Same |
| Trials per method | 2 | 6-20, as specified in the paper |
| Neural epoch limit | 30 | 300 |
| Final conditions | MCAR/MAR/MNAR, 10%/30%/50% | Same |
| Final simulation seeds | 42, 100, 2024, 777, 999 | Same |
| Association seed | 42 | 42 |

Demo results are execution examples, not a publication-strength method ranking.
The notebook's `FINAL_SEEDS=(42,)` option restores the earlier quick single-seed
check. The Python API's `Settings()` still defaults to one seed; the notebook
sets its five-seed budget explicitly.
The paper-budget mode has not been executed as part of this delivery. Model
architecture, simulation details, and public cohort limitations are documented
in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Code layout

```text
notebooks/          Clear tutorial notebook and executed example
reproducibility/    Public data loader, deterministic masks, adapters, pipeline, figures
legacy/             Original Python/R source implementations from the supplied archive
configs/            Source URLs, fixed GDC UUIDs, SHA-256 hashes
scripts/            Installers, notebook checks, GitHub badge configuration
tests/              Scientific integrity checks
docs/               Method mapping, source lineage, validation and limitations
```

The adapter retains KNN, Bayesian-ridge Iterative, missRanger MissForest,
rank-truncated-SVD SoftImpute, and the five original neural methods. No fallback
estimator silently replaces a missing method. The archived source's `SoftImpute`
is not standard singular-value-shrinkage SoftImpute.

## Outputs

Section **8a** calls `legacy/Evaluation/paper_seed_rmse_boxplots.py`, the actual
Figure 2 renderer, using the same colours and method order. It exports labelled
TCGA-UCEC PDF/PNG figures, individual seed values, mean/SD summaries, a caption,
and plot provenance. Axis limits adapt to the public data. One-seed previews
are explicitly labelled and do not estimate between-seed variability.

To plot an existing run without refitting:

```bash
python scripts/plot_paper_rmse.py --run results/YOUR_RUN_DIRECTORY
```

See [docs/FIGURE2.md](docs/FIGURE2.md) for the full plotting protocol.

Every run records source/phenotype hashes, settings, dependency versions,
validation trials and conditions, SQLite Optuna studies, frozen parameters,
exact simulation/validation masks, final completed matrices, reconstruction
scores and timing, protein-level association scans, preservation metrics,
figures, and per-fit logs. Each method gets identical evaluation inputs.

The adapter's `impute` interface accepts no hidden truth or clinical labels.
Legacy APIs receive a dummy label and a mean-filled observed-data surrogate;
their diagnostic RMSE values are not benchmark scores. True RMSE is calculated
outside the imputer, only on the saved evaluation masks.

## Private MyoVasc data

Use `reproducibility.data.load_private_cohort()` with a protein CSV
(`eid`, optional `label`, then proteins) and a separate phenotype CSV (`eid`,
verified binary outcomes, covariates). `label` is removed from protein inputs.
Run the same stages with explicit outcome/covariate names. For exact paper
reproduction, the original authorized data, masks, original environment, and
analysis settings are required; see the reproducibility notes.

The source archive's patient-demography tables, spreadsheets, cached experiment
results, and notebook outputs have been excluded from this distribution.
Do not commit participant data, credentials, private result matrices, or
compiled R-library archives. The `.gitignore` excludes these working paths.

## Checks and attribution

```bash
python -m unittest discover -s tests -v
python scripts/validate_notebook.py
python -m nbconvert --execute --to notebook --output benchmark_TCGA_local_run.ipynb notebooks/benchmark_TCGA_reproducible.ipynb
```

Original source hashes are in `docs/original_source_sha256.json`.
The original source archive did not include a license. No new license or author
identity has been imposed; choose the repository license and citation metadata
with the project authors before public release. Cite TCGA, the RPPA provider,
and the imputation methods, in addition to the benchmark paper.

Sources: [TCGA-UCEC publication/data](https://gdc.cancer.gov/about-data/publications/ucec_2013),
[cBioPortal Datahub](https://github.com/cBioPortal/datahub),
[GDC clinical field documentation](https://gdc.cancer.gov/resources-tcga-users/legacy-archive-tcga-tag-descriptions).
