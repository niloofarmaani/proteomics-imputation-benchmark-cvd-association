# Figure 2 style reconstruction plots

The notebook's section 8a calls the paper's original
`legacy/Evaluation/paper_seed_rmse_boxplots.py`. The portable adapter only maps
the reconstruction table to that module's input schema, verifies the complete
run grid, and supplies public-data labels. It does not train models.

- Panels A/B/C: MCAR, MAR, MNAR, sharing one RMSE scale.
- Fractions: 10%, 30%, 50%, from left to right in each panel.
- Methods: the original paper's order and colours, shared with the RMSE curves.
- Replicates: seeds 42, 100, 2024, 777, 999 by default in the notebook.
- Boxes: IQR; line: median; whiskers: most extreme observed values within
  1.5 IQR; circles: the actual seed-level RMSE values.
- RMSE units: source RPPA units, measured only at simulated missing cells.
- Summary SD: sample standard deviation (`ddof=1`), not a confidence interval.

Each replicate includes a newly simulated missingness mask and the corresponding
algorithm seed. Spread therefore includes both sources of randomness, as in the
paper's repeated-seed reconstruction protocol.

The two-trial/30-epoch public demo is deliberately smaller than the paper's
optimization budget. Five demo seeds reproduce the figure structure and seed
protocol, not the full optimization budget or MyoVasc numerical results.
Association preservation continues to use only seed 42.

## Existing Colab run

After explicitly updating the source checkout with `git pull --ff-only`, run
this in the still-active notebook session (where `ROOT` and `OUT` exist):

```python
subprocess.run([sys.executable, str(ROOT/'scripts/plot_paper_rmse.py'),
                '--run', str(OUT)], check=True)
from IPython.display import Image, display
display(Image(filename=str(OUT/'figures/figure2_TCGA_UCEC/Figure2_TCGA_UCEC_RMSE.png')))
```

The completed earlier demo has only seed 42. Its preview shows collapsed boxes
and individual values with an explicit one-seed label. It cannot supply an IQR
or SD across independent replicates. Generating a five-seed distribution
requires the additional final fits; changing the seed list only at plotting
time is rejected as an incomplete grid.

For a new run, use the updated notebook from a fresh Colab runtime and leave
`FINAL_SEEDS=(42, 100, 2024, 777, 999)`. This yields 405 final reconstruction
fits with nine methods, three mechanisms, and three fractions. For the full
public-data tuning budget and panel, also select `MODE='paper_budget'`.

## Files saved with each run

Under `figures/figure2_TCGA_UCEC/`:

- `Figure2_TCGA_UCEC_RMSE.pdf`: vector figure.
- `Figure2_TCGA_UCEC_RMSE.png`: 300-dpi figure.
- `paper_seed_rmse_values.csv`: plotted individual replicate values.
- `paper_seed_rmse_summary.csv`: counts, means, SDs, medians, minima/maxima.
- `caption.txt`: dataset, scope, seeds, and plot definitions.
- `plot_manifest.json`: hashes of plotting inputs and renderer source.

The figure itself identifies TCGA-UCEC tumor RPPA. Its axis limits follow the
observed TCGA scores, so it cannot be visually mistaken for the paper's lower
MyoVasc RMSE range. Unavailable or duplicate results stop plotting.
