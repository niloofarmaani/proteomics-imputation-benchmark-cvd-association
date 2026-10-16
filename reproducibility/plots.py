"""Portable figures with stable method order and explicit uncertainty labels."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .methods import METHODS
from Evaluation.paper_seed_rmse_boxplots import METHOD_COLORS

COLORS = {method: METHOD_COLORS[method] for method in METHODS}


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 125, "savefig.dpi": 220})


def reconstruction_plot(results, out):
    style()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, mechanism in zip(axes, ["MCAR", "MAR", "MNAR"]):
        panel = results[results.mechanism == mechanism]
        for method in METHODS:
            group = panel[panel.method == method].groupby("fraction").rmse.agg(["mean", "std", "count"])
            if group.empty:
                continue
            ax.plot(group.index * 100, group["mean"], marker="o", label=method, color=COLORS[method], linewidth=1.5)
            if group["count"].min() > 1:
                ax.errorbar(group.index * 100, group["mean"], yerr=group["std"], fmt="none", color=COLORS[method], capsize=2)
        ax.set(title=mechanism, xlabel="Simulated missingness (%)", xticks=[10, 30, 50])
        ax.grid(axis="y", alpha=0.18)
    axes[0].set_ylabel("RMSE on simulated missing entries\n(source RPPA units)")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.03, 1), frameon=False)
    n = results.seed.nunique()
    fig.suptitle(f"Frozen-parameter reconstruction | {n} simulation seed(s)" + ("; error bars = SD" if n > 1 else "; no replicate uncertainty estimate"), fontsize=12)
    fig.tight_layout()
    directory = Path(out) / "figures"
    directory.mkdir(exist_ok=True)
    fig.savefig(directory / "reconstruction.png", bbox_inches="tight")
    return fig


def preservation_plot(metrics, out):
    style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.1))
    cols = [(m, f) for m in ["MCAR", "MAR", "MNAR"] for f in sorted(metrics.fraction.unique())]
    methods = [m for m in METHODS if m in set(metrics.method)]
    for ax, metric, title in zip(axes, ["z_spearman", "top50_overlap", "fdr_jaccard"], ["Signed z-statistic Spearman", "Top-50 overlap", "FDR-set Jaccard"]):
        table = metrics.pivot(index="method", columns=["mechanism", "fraction"], values=metric).reindex(index=methods, columns=cols)
        cmap = plt.get_cmap("cividis").copy()
        cmap.set_bad("#E5E7EB")
        image = ax.imshow(table, vmin=-1 if metric == "z_spearman" else 0, vmax=1, aspect="auto", cmap=cmap)
        for i in range(len(methods)):
            for j in range(len(cols)):
                value = table.iloc[i, j]
                ax.text(j, i, f"{value:.2f}" if np.isfinite(value) else "NA", ha="center", va="center", fontsize=7.5,
                        color="white" if np.isfinite(value) and value < (0 if metric == "z_spearman" else .5) else "#151A22")
        ax.set(xticks=range(len(cols)), xticklabels=[f"{f:.0%}" for m, f in cols], yticks=range(len(methods)), yticklabels=methods, title=title)
        for mechanism in ["MCAR", "MAR", "MNAR"]:
            positions = [j for j, (m, _) in enumerate(cols) if m == mechanism]
            if positions:
                ax.text(float(np.mean(positions)), -.105, mechanism, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9)
        for boundary in [j - .5 for j in range(1, len(cols)) if cols[j][0] != cols[j-1][0]]:
            ax.axvline(boundary, color="white", linewidth=1)
        ax.tick_params(axis="x", labelsize=8)
        fig.colorbar(image, ax=ax, fraction=.045, pad=.03)
    fig.suptitle("Hypertension association preservation | fixed seed 42", fontsize=12)
    fig.text(.5, .015, "NA = undefined, including two empty FDR-significant sets. Grey does not mean zero preservation.", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, .11, 1, 1])
    directory = Path(out) / "figures"
    directory.mkdir(exist_ok=True)
    fig.savefig(directory / "association_preservation.png", bbox_inches="tight")
    return fig


def volcano(reference, out):
    style()
    fig, ax = plt.subplots(figsize=(7, 4.1))
    valid = reference.dropna(subset=["beta", "q"])
    significant = valid.q < .05
    ax.scatter(valid.beta, -np.log10(valid.q.clip(lower=1e-300)), c=np.where(significant, "#A9527E", "#637381"), s=24, alpha=.8)
    ax.axhline(-np.log10(.05), color="#444444", linestyle="--", linewidth=1)
    ax.set(xlabel="Adjusted log odds per protein SD", ylabel="-log10(BH-adjusted p-value)",
           title=f"Reference hypertension associations | {int(significant.sum())} FDR-significant feature(s)")
    fig.tight_layout()
    directory = Path(out) / "figures"
    directory.mkdir(exist_ok=True)
    fig.savefig(directory / "reference_associations.png", bbox_inches="tight")
    return fig
