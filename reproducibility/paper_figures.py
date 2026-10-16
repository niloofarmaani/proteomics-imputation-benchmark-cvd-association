"""Use the paper's actual renderer on the public benchmark's saved RMSE grid."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import numpy as np

from .methods import METHODS  # Also makes the original legacy modules importable.
from Evaluation import paper_seed_rmse_boxplots as paper


PAPER_SEEDS = (42, 100, 2024, 777, 999)
PAPER_METHOD_ORDER = (
    "Iterative", "MissForest", "SoftImpute", "KNN", "AE", "DAE", "GAIN", "VAE", "Transformer"
)


def paper_rmse_figure(results, out, *, dataset_label="TCGA-UCEC tumor RPPA", show=True):
    """Plot existing scores only, without tuning or fitting any imputation model.

    The saved run.json defines the required methods, conditions and seeds. An
    incomplete or duplicated grid is an error. Single-seed previews are explicitly
    labelled: a collapsed box does not estimate between-seed variability.
    """
    out = Path(out).resolve()
    run = json.loads((out / "run.json").read_text())
    settings = run["settings"]
    required = {"method", "mechanism", "fraction", "seed", "rmse"}
    if not required.issubset(results.columns):
        raise ValueError(f"Missing reconstruction columns: {sorted(required - set(results.columns))}")
    if results.empty:
        raise ValueError("No completed reconstruction results to plot.")
    if set(settings["methods"]) - set(METHODS):
        raise ValueError("Run contains an unsupported method.")
    keys = ["method", "mechanism", "fraction", "seed"]
    expected = {
        (method, mechanism, float(fraction), int(seed))
        for method in settings["methods"]
        for mechanism in settings["mechanisms"]
        for fraction in settings["fractions"]
        for seed in settings["final_seeds"]
    }
    actual = set(results[keys].itertuples(index=False, name=None))
    if results.duplicated(keys).any() or actual != expected:
        raise ValueError("Reconstruction grid is incomplete, duplicated, or differs from run.json.")
    scores = pd.to_numeric(results["rmse"], errors="raise").to_numpy(float)
    if not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("RMSE values must be finite and nonnegative.")

    directory = out / "figures" / "figure2_TCGA_UCEC"
    directory.mkdir(parents=True, exist_ok=True)
    values = results[list(keys) + ["rmse"]].rename(columns={"mechanism": "missingness"})
    source_csv = directory / "reconstruction_for_paper_plot.csv"
    values.to_csv(source_csv, index=False)
    seeds = list(settings["final_seeds"])
    n_seeds = len(seeds)
    note = (
        "One seed: collapsed boxes show individual values; replicate variability is not estimated."
        if n_seeds == 1 else
        "Boxes: IQR; line: median; whiskers: 1.5 IQR; circles: individual simulation seeds."
    )
    outputs, seed_values, summary, _ = paper.make_paper_seed_rmse_boxplots(
        columns_num="public",
        methods_to_compare=[m for m in PAPER_METHOD_ORDER if m in settings["methods"]],
        fractions=settings["fractions"], missingness_types=settings["mechanisms"], seeds=seeds,
        output_dir=directory, seed_values_csv=source_csv,
        strict=True, sort_by="input", y_range=None,
        show_seed_points=True, save_png=True, show=show,
        make_bayes_tables=False, make_individual_panels=False, make_mechanism_panels=False,
        print_latex_tables=False, print_markdown_tables=False,
        figure_height_mm=190,
        figure_title=f"{dataset_label}\nFinal reconstruction RMSE | {n_seeds} simulation seed(s)",
        figure_note=note,
    )
    # Use a descriptive exported name; the legacy renderer's name predates Figure 2.
    for extension, old_path in list(outputs["main_all_conditions_figure"].items()):
        target = directory / f"Figure2_TCGA_UCEC_RMSE.{extension}"
        Path(old_path).replace(target)
        outputs["main_all_conditions_figure"][extension] = str(target)
    caption = (
        f"Figure 2 style public-data reconstruction: {dataset_label}. "
        f"Panels show MCAR, MAR and MNAR, with fractions grouped left to right and "
        f"the original paper's method colours and ordering. Seeds: {', '.join(map(str, seeds))}. "
        f"{note} RMSE is in source RPPA units and is scored only on simulated omissions. "
        "Hyperparameters were frozen after validation-only tuning. "
        "These TCGA results do not reproduce the paper's MyoVasc numerical results."
    )
    (directory / "caption.txt").write_text(caption + "\n")
    manifest = {
        "dataset_label": dataset_label, "seeds": seeds, "n_seeds": n_seeds,
        "input_sha256": run.get("input_sha256"),
        "plot_values_sha256": hashlib.sha256(source_csv.read_bytes()).hexdigest(),
        "renderer": "legacy/Evaluation/paper_seed_rmse_boxplots.py",
        "renderer_sha256": hashlib.sha256(Path(paper.__file__).read_bytes()).hexdigest(),
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "uncertainty": "not estimated" if n_seeds == 1 else "spread across simulation seeds, not a confidence interval",
        "outputs": {
            "pdf": str(Path(outputs["main_all_conditions_figure"]["pdf"]).relative_to(out)),
            "png": str(Path(outputs["main_all_conditions_figure"]["png"]).relative_to(out)),
            "values_csv": str(Path(outputs["values_csv"]).relative_to(out)),
            "summary_csv": str(Path(outputs["summary_csv"]).relative_to(out)),
        },
    }
    (directory / "plot_manifest.json").write_text(json.dumps(manifest, indent=2))
    return outputs, seed_values, summary
