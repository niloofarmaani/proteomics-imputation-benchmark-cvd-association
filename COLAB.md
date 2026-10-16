[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/niloofarmaani/proteomics-imputation-benchmark-cvd-association/blob/main/notebooks/benchmark_TCGA_reproducible.ipynb)

1. Open the notebook using the badge.
2. Run the first code cell. It clones the public GitHub repository directly into `/content/imputation-reproducibility` and prints the exact commit.
3. Run the dependency setup cell. Restart the runtime once if it reports changed loaded packages, then rerun from the beginning.
4. Keep `MODE='demo'` for the first full test. Save the exported result archive before the runtime ends.

The current notebook uses five final simulation seeds for Figure 2 style
boxplots. For the shortest execution check, set `FINAL_SEEDS=(42,)` in section
3. Section 8a renders the paper-style figure. No imputation is rerun by that
plotting cell. See [docs/FIGURE2.md](docs/FIGURE2.md) to plot a saved run.

No repository ZIP download or upload is needed. The first cell reuses an existing checkout on reruns; it does not pull updates or overwrite files. Start a fresh session when you want to clone a later repository version.

The repository must remain publicly readable for cloning without GitHub credentials.
