from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# =============================================================================
# Helpers
# =============================================================================
def _metric_mean_col(metric_prefix: str) -> str:
    return f"{metric_prefix}_mean"


def _metric_ci_cols(metric_prefix: str) -> Tuple[str, str]:
    return f"{metric_prefix}_ci95_low", f"{metric_prefix}_ci95_high"


def _method_colors(methods: List[str]) -> Dict[str, str]:
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    if not colors:
        colors = [f"C{i}" for i in range(10)]

    return {
        method: colors[i % len(colors)]
        for i, method in enumerate(methods)
    }


def _ordered_missingness(values) -> List[str]:
    order = ["MCAR", "MAR", "MNAR"]
    values = [str(v) for v in values]
    present = [m for m in order if m in set(values)]
    extras = sorted([m for m in set(values) if m not in present])
    return present + extras


def _save_figure(
    fig,
    out_dir: Path,
    filename_prefix: str,
    save_individual: bool = True,
    pdf_pages: Optional[PdfPages] = None,
    show: bool = True,
) -> Dict[str, Optional[Path]]:
    outputs = {
        "png": None,
        "pdf": None,
    }

    if save_individual:
        out_png = out_dir / f"{filename_prefix}.png"
        out_pdf = out_dir / f"{filename_prefix}.pdf"

        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        fig.savefig(out_pdf, bbox_inches="tight")

        outputs["png"] = out_png
        outputs["pdf"] = out_pdf

    if pdf_pages is not None:
        pdf_pages.savefig(fig, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)

    return outputs


def _safe_read_summary(summary_csv: Union[str, Path]) -> pd.DataFrame:
    summary_csv = Path(summary_csv).expanduser().resolve()

    if not summary_csv.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")

    df = pd.read_csv(summary_csv)

    if df.empty:
        raise ValueError(f"Summary CSV is empty: {summary_csv}")

    return df


def _safe_read_fold(fold_csv: Union[str, Path]) -> pd.DataFrame:
    fold_csv = Path(fold_csv).expanduser().resolve()

    if not fold_csv.exists():
        raise FileNotFoundError(f"Fold CSV not found: {fold_csv}")

    df = pd.read_csv(fold_csv)

    if df.empty:
        raise ValueError(f"Fold CSV is empty: {fold_csv}")

    return df


# =============================================================================
# Title and summary pages
# =============================================================================
def add_title_page(
    pdf_pages: PdfPages,
    *,
    title: str,
    summary_csv: Union[str, Path],
    fold_csv: Optional[Union[str, Path]] = None,
) -> None:
    summary_csv = Path(summary_csv).expanduser().resolve()

    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")

    lines = [
        title,
        "",
        "CVD downstream evaluation report",
        "",
        f"Summary CSV:",
        str(summary_csv),
    ]

    if fold_csv is not None:
        lines.extend(
            [
                "",
                "Fold-level CSV:",
                str(Path(fold_csv).expanduser().resolve()),
            ]
        )

    lines.extend(
        [
            "",
            "Included visualizations:",
            "1. ROC-AUC across missing fractions",
            "2. PR-AUC across missing fractions",
            "3. Brier score across missing fractions",
            "4. Log loss across missing fractions",
            "5. Imputation RMSE versus ROC-AUC",
            "6. Imputation RMSE versus PR-AUC",
            "7. Delta ROC-AUC versus baseline, if baseline is available",
            "8. Delta Brier improvement versus baseline, if baseline is available",
            "9. Calibration slope",
            "10. Calibration intercept",
            "11. Fold-level boxplots for selected condition, if fold CSV is provided",
            "",
            "Notes:",
            "Positive delta means improvement over the selected baseline.",
            "For Brier score and log loss, lower values are better.",
            "For ROC-AUC and PR-AUC, higher values are better.",
        ]
    )

    y = 0.95
    for i, line in enumerate(lines):
        if i == 0:
            ax.text(
                0.05,
                y,
                line,
                fontsize=18,
                fontweight="bold",
                va="top",
                ha="left",
                transform=ax.transAxes,
            )
            y -= 0.055
        elif line.endswith(":"):
            ax.text(
                0.05,
                y,
                line,
                fontsize=11,
                fontweight="bold",
                va="top",
                ha="left",
                transform=ax.transAxes,
            )
            y -= 0.028
        else:
            ax.text(
                0.05,
                y,
                line,
                fontsize=10,
                va="top",
                ha="left",
                transform=ax.transAxes,
            )
            y -= 0.028

    pdf_pages.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_results_table_page(
    pdf_pages: PdfPages,
    *,
    summary_csv: Union[str, Path],
    title: str = "Best method summary by missingness and fraction",
    metric: str = "roc_auc_mean",
    top_n: int = 20,
) -> None:
    df = _safe_read_summary(summary_csv)

    required = {"method", "missingness", "fraction", "rmse", metric}
    missing = required - set(df.columns)

    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")

    if missing:
        ax.text(
            0.05,
            0.95,
            f"Could not create summary table. Missing columns: {sorted(missing)}",
            fontsize=12,
            va="top",
            ha="left",
            transform=ax.transAxes,
        )
        pdf_pages.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        return

    tmp = df.copy()
    tmp["fraction"] = pd.to_numeric(tmp["fraction"], errors="coerce")
    tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
    tmp["rmse"] = pd.to_numeric(tmp["rmse"], errors="coerce")

    tmp = tmp.dropna(subset=["fraction", metric]).copy()

    tmp = tmp.sort_values(["missingness", "fraction", metric], ascending=[True, True, False])

    rows = []

    for (missingness, fraction), g in tmp.groupby(["missingness", "fraction"]):
        best = g.iloc[0]

        rows.append(
            {
                "missingness": missingness,
                "fraction": fraction,
                "best_method": best["method"],
                metric: best[metric],
                "rmse": best["rmse"],
            }
        )

    table_df = pd.DataFrame(rows).head(top_n)

    ax.set_title(title, fontsize=15, fontweight="bold", pad=20)

    table_data = table_df.copy()
    if not table_data.empty:
        for col in table_data.columns:
            if pd.api.types.is_numeric_dtype(table_data[col]):
                table_data[col] = table_data[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")

    tbl = ax.table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.4)

    pdf_pages.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main plot functions
# =============================================================================
def plot_metric_panels(
    summary_csv,
    metric_prefix: str = "roc_auc",
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    show_ci: bool = True,
    show: bool = True,
    save_individual: bool = True,
    pdf_pages: Optional[PdfPages] = None,
):
    """
    Line plot across missing fractions.
    One panel per missingness type.
    """

    summary_csv = Path(summary_csv).expanduser().resolve()
    out_dir = summary_csv.parent

    df = _safe_read_summary(summary_csv)

    mean_col = _metric_mean_col(metric_prefix)
    low_col, high_col = _metric_ci_cols(metric_prefix)

    required = ["method", "missingness", "fraction", mean_col]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns in summary CSV: {missing}")

    df["fraction"] = pd.to_numeric(df["fraction"], errors="coerce")
    df[mean_col] = pd.to_numeric(df[mean_col], errors="coerce")
    df = df.dropna(subset=["fraction", mean_col]).copy()

    missingness_types = _ordered_missingness(df["missingness"].astype(str).unique())
    methods = sorted(df["method"].astype(str).unique())
    colors = _method_colors(methods)

    fig, axes = plt.subplots(
        1,
        len(missingness_types),
        figsize=(5.3 * len(missingness_types), 4.2),
        sharey=True,
        sharex=True,
    )

    if len(missingness_types) == 1:
        axes = [axes]

    for ax, missingness in zip(axes, missingness_types):
        sub_miss = df[df["missingness"].astype(str) == missingness].copy()

        for method in methods:
            g = sub_miss[sub_miss["method"].astype(str) == method].sort_values("fraction")

            if g.empty:
                continue

            x = g["fraction"].to_numpy(dtype=float)
            y = g[mean_col].to_numpy(dtype=float)

            ax.plot(
                x,
                y,
                marker="o",
                linewidth=1.8,
                color=colors[method],
                label=method,
            )

            if show_ci and low_col in g.columns and high_col in g.columns:
                low = pd.to_numeric(g[low_col], errors="coerce").to_numpy(dtype=float)
                high = pd.to_numeric(g[high_col], errors="coerce").to_numpy(dtype=float)

                ax.fill_between(
                    x,
                    low,
                    high,
                    color=colors[method],
                    alpha=0.12,
                )

        ax.set_title(missingness)
        ax.set_xlabel("Missing fraction")
        ax.grid(True, alpha=0.3)

    if ylabel is None:
        ylabel = mean_col

    axes[0].set_ylabel(ylabel)

    if title is None:
        title = f"{ylabel} by missingness mechanism"

    fig.suptitle(title, y=1.02)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Method",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )

    fig.tight_layout(rect=[0, 0, 0.86, 1])

    return _save_figure(
        fig=fig,
        out_dir=out_dir,
        filename_prefix=f"cvd_{metric_prefix}_panels",
        save_individual=save_individual,
        pdf_pages=pdf_pages,
        show=show,
    )


def plot_rmse_vs_metric(
    summary_csv,
    metric_prefix: str = "roc_auc",
    ylabel: Optional[str] = None,
    show: bool = True,
    save_individual: bool = True,
    pdf_pages: Optional[PdfPages] = None,
):
    """
    Scatterplot:
        x = imputation RMSE
        y = downstream metric
    One panel per missingness type.
    """

    summary_csv = Path(summary_csv).expanduser().resolve()
    out_dir = summary_csv.parent

    df = _safe_read_summary(summary_csv)

    mean_col = _metric_mean_col(metric_prefix)

    required = ["method", "missingness", "fraction", "rmse", mean_col]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns in summary CSV: {missing}")

    df["fraction"] = pd.to_numeric(df["fraction"], errors="coerce")
    df["rmse"] = pd.to_numeric(df["rmse"], errors="coerce")
    df[mean_col] = pd.to_numeric(df[mean_col], errors="coerce")
    df = df.dropna(subset=["fraction", "rmse", mean_col]).copy()

    missingness_types = _ordered_missingness(df["missingness"].astype(str).unique())
    methods = sorted(df["method"].astype(str).unique())
    colors = _method_colors(methods)

    fraction_values = sorted(df["fraction"].unique())
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    marker_map = {
        f: markers[i % len(markers)]
        for i, f in enumerate(fraction_values)
    }

    fig, axes = plt.subplots(
        1,
        len(missingness_types),
        figsize=(5.3 * len(missingness_types), 4.2),
        sharey=True,
    )

    if len(missingness_types) == 1:
        axes = [axes]

    for ax, missingness in zip(axes, missingness_types):
        sub = df[df["missingness"].astype(str) == missingness].copy()

        for _, row in sub.iterrows():
            method = str(row["method"])
            fraction = row["fraction"]

            ax.scatter(
                row["rmse"],
                row[mean_col],
                color=colors[method],
                marker=marker_map[fraction],
                s=55,
                alpha=0.85,
            )

        ax.set_title(missingness)
        ax.set_xlabel("Imputation RMSE")
        ax.grid(True, alpha=0.3)

    if ylabel is None:
        ylabel = mean_col

    axes[0].set_ylabel(ylabel)

    method_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=colors[m],
            markersize=7,
            label=m,
        )
        for m in methods
    ]

    fraction_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=marker_map[f],
            color="black",
            linestyle="none",
            markersize=7,
            label=f"fraction={f:g}",
        )
        for f in fraction_values
    ]

    fig.legend(
        handles=method_handles + fraction_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        title="Legend",
    )

    fig.suptitle(f"Imputation RMSE versus downstream {ylabel}", y=1.02)
    fig.tight_layout(rect=[0, 0, 0.84, 1])

    return _save_figure(
        fig=fig,
        out_dir=out_dir,
        filename_prefix=f"cvd_rmse_vs_{metric_prefix}",
        save_individual=save_individual,
        pdf_pages=pdf_pages,
        show=show,
    )


def plot_delta_vs_baseline(
    summary_csv,
    metric_prefix: str = "roc_auc",
    baseline_method: str = "Mean",
    higher_is_better: bool = True,
    ylabel: Optional[str] = None,
    show: bool = True,
    save_individual: bool = True,
    pdf_pages: Optional[PdfPages] = None,
):
    """
    Plot improvement over a baseline method.

    For higher-is-better metrics:
        delta = method - baseline

    For lower-is-better metrics:
        delta = baseline - method

    Positive values always mean improvement over baseline.
    """

    summary_csv = Path(summary_csv).expanduser().resolve()
    out_dir = summary_csv.parent

    df = _safe_read_summary(summary_csv)

    mean_col = _metric_mean_col(metric_prefix)

    required = ["method", "missingness", "fraction", mean_col]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns in summary CSV: {missing}")

    df["fraction"] = pd.to_numeric(df["fraction"], errors="coerce")
    df[mean_col] = pd.to_numeric(df[mean_col], errors="coerce")
    df = df.dropna(subset=["fraction", mean_col]).copy()

    df["method_lower"] = df["method"].astype(str).str.lower()
    baseline_lower = baseline_method.lower()

    baseline = df[df["method_lower"] == baseline_lower][
        ["missingness", "fraction", mean_col]
    ].rename(columns={mean_col: "baseline_value"})

    if baseline.empty:
        raise ValueError(f"Could not find baseline method '{baseline_method}' in summary CSV.")

    merged = df.merge(
        baseline,
        on=["missingness", "fraction"],
        how="inner",
    )

    merged = merged[merged["method_lower"] != baseline_lower].copy()

    if higher_is_better:
        merged["delta"] = merged[mean_col] - merged["baseline_value"]
    else:
        merged["delta"] = merged["baseline_value"] - merged[mean_col]

    missingness_types = _ordered_missingness(merged["missingness"].astype(str).unique())
    methods = sorted(merged["method"].astype(str).unique())
    colors = _method_colors(methods)

    fig, axes = plt.subplots(
        1,
        len(missingness_types),
        figsize=(5.3 * len(missingness_types), 4.2),
        sharey=True,
        sharex=True,
    )

    if len(missingness_types) == 1:
        axes = [axes]

    for ax, missingness in zip(axes, missingness_types):
        sub = merged[merged["missingness"].astype(str) == missingness].copy()

        for method in methods:
            g = sub[sub["method"].astype(str) == method].sort_values("fraction")

            if g.empty:
                continue

            ax.plot(
                g["fraction"],
                g["delta"],
                marker="o",
                linewidth=1.8,
                color=colors[method],
                label=method,
            )

        ax.axhline(0, color="black", linestyle="--", linewidth=1.0)
        ax.set_title(missingness)
        ax.set_xlabel("Missing fraction")
        ax.grid(True, alpha=0.3)

    if ylabel is None:
        ylabel = f"Delta {metric_prefix} versus {baseline_method}"

    axes[0].set_ylabel(ylabel)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Method",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )

    fig.suptitle(ylabel, y=1.02)
    fig.tight_layout(rect=[0, 0, 0.86, 1])

    safe_baseline = baseline_method.replace(" ", "_")
    return _save_figure(
        fig=fig,
        out_dir=out_dir,
        filename_prefix=f"cvd_delta_{metric_prefix}_vs_{safe_baseline}",
        save_individual=save_individual,
        pdf_pages=pdf_pages,
        show=show,
    )


def plot_calibration_panels(
    summary_csv,
    value: str = "calibration_slope",
    ideal_value: float = 1.0,
    ylabel: Optional[str] = None,
    show: bool = True,
    save_individual: bool = True,
    pdf_pages: Optional[PdfPages] = None,
):
    """
    Plot calibration slope or calibration intercept.

    For slope:
        ideal_value = 1

    For intercept:
        ideal_value = 0
    """

    summary_csv = Path(summary_csv).expanduser().resolve()
    out_dir = summary_csv.parent

    df = _safe_read_summary(summary_csv)

    required = ["method", "missingness", "fraction", value]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns in summary CSV: {missing}")

    df["fraction"] = pd.to_numeric(df["fraction"], errors="coerce")
    df[value] = pd.to_numeric(df[value], errors="coerce")
    df = df.dropna(subset=["fraction", value]).copy()

    missingness_types = _ordered_missingness(df["missingness"].astype(str).unique())
    methods = sorted(df["method"].astype(str).unique())
    colors = _method_colors(methods)

    fig, axes = plt.subplots(
        1,
        len(missingness_types),
        figsize=(5.3 * len(missingness_types), 4.2),
        sharey=True,
        sharex=True,
    )

    if len(missingness_types) == 1:
        axes = [axes]

    for ax, missingness in zip(axes, missingness_types):
        sub = df[df["missingness"].astype(str) == missingness].copy()

        for method in methods:
            g = sub[sub["method"].astype(str) == method].sort_values("fraction")

            if g.empty:
                continue

            ax.plot(
                g["fraction"],
                g[value],
                marker="o",
                linewidth=1.8,
                color=colors[method],
                label=method,
            )

        ax.axhline(ideal_value, color="black", linestyle="--", linewidth=1.0)
        ax.set_title(missingness)
        ax.set_xlabel("Missing fraction")
        ax.grid(True, alpha=0.3)

    if ylabel is None:
        ylabel = value

    axes[0].set_ylabel(ylabel)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Method",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )

    fig.suptitle(ylabel, y=1.02)
    fig.tight_layout(rect=[0, 0, 0.86, 1])

    return _save_figure(
        fig=fig,
        out_dir=out_dir,
        filename_prefix=f"cvd_{value}",
        save_individual=save_individual,
        pdf_pages=pdf_pages,
        show=show,
    )


def plot_fold_metric_boxplot(
    fold_csv,
    metric: str = "roc_auc",
    fraction: float = 0.3,
    missingness: str = "MNAR",
    sort_by: str = "median",
    show: bool = True,
    save_individual: bool = True,
    pdf_pages: Optional[PdfPages] = None,
):
    """
    Boxplot over repeated-CV fold metrics for one condition.
    """

    fold_csv = Path(fold_csv).expanduser().resolve()
    out_dir = fold_csv.parent

    df = _safe_read_fold(fold_csv)

    required = ["method", "missingness", "fraction", metric]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns in fold CSV: {missing}")

    df["fraction"] = pd.to_numeric(df["fraction"], errors="coerce")
    df[metric] = pd.to_numeric(df[metric], errors="coerce")

    sub = df[
        (df["missingness"].astype(str).str.upper() == str(missingness).upper())
        & (np.isclose(df["fraction"], float(fraction)))
    ].copy()

    sub = sub.dropna(subset=[metric])

    if sub.empty:
        raise ValueError(f"No fold-level data found for fraction={fraction}, missingness={missingness}")

    if sort_by == "median":
        order = sub.groupby("method")[metric].median().sort_values().index.tolist()
    elif sort_by == "mean":
        order = sub.groupby("method")[metric].mean().sort_values().index.tolist()
    else:
        order = sorted(sub["method"].unique())

    data = [
        sub.loc[sub["method"] == method, metric].to_numpy(dtype=float)
        for method in order
    ]

    colors = _method_colors(order)

    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(order)), 4.8))

    try:
        bp = ax.boxplot(
            data,
            tick_labels=order,
            patch_artist=True,
            showmeans=False,
            showfliers=False,
        )
    except TypeError:
        bp = ax.boxplot(
            data,
            labels=order,
            patch_artist=True,
            showmeans=False,
            showfliers=False,
        )

    for patch, method in zip(bp["boxes"], order):
        patch.set_facecolor(colors[method])
        patch.set_alpha(0.45)

    ax.set_title(f"Fold-level {metric}: fraction={fraction}, missingness={missingness}")
    ax.set_xlabel("Imputation method")
    ax.set_ylabel(metric)
    ax.grid(True, axis="y", alpha=0.3)

    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    frac_token = str(fraction).replace(".", "p")
    filename = f"fold_boxplot_{metric}_fraction_{frac_token}_{missingness}"

    return _save_figure(
        fig=fig,
        out_dir=out_dir,
        filename_prefix=filename,
        save_individual=save_individual,
        pdf_pages=pdf_pages,
        show=show,
    )


# =============================================================================
# Combined report
# =============================================================================
def make_all_recommended_cvd_plots(
    summary_csv,
    fold_csv: Optional[str] = None,
    baseline_method: str = "Mean",
    show: bool = True,
    save_individual: bool = True,
    combined_pdf_name: str = "cvd_downstream_report.pdf",
    include_title_page: bool = True,
    include_summary_table: bool = True,
):
    """
    Makes all recommended downstream plots and saves them into one combined PDF.

    Returns
    -------
    outputs : dict
        Includes individual plot paths and combined PDF path.
    """

    summary_csv = Path(summary_csv).expanduser().resolve()
    out_dir = summary_csv.parent
    combined_pdf_path = out_dir / combined_pdf_name

    outputs = {
        "combined_pdf": combined_pdf_path,
        "individual_plots": {},
    }

    with PdfPages(combined_pdf_path) as pdf:
        if include_title_page:
            add_title_page(
                pdf,
                title="CVD downstream evaluation",
                summary_csv=summary_csv,
                fold_csv=fold_csv,
            )

        if include_summary_table:
            add_results_table_page(
                pdf,
                summary_csv=summary_csv,
                title="Best method by condition using ROC-AUC",
                metric="roc_auc_mean",
            )

        outputs["individual_plots"]["roc_auc_panels"] = plot_metric_panels(
            summary_csv,
            metric_prefix="roc_auc",
            ylabel="ROC-AUC",
            title="CVD prediction ROC-AUC",
            show_ci=True,
            show=show,
            save_individual=save_individual,
            pdf_pages=pdf,
        )

        outputs["individual_plots"]["pr_auc_panels"] = plot_metric_panels(
            summary_csv,
            metric_prefix="pr_auc",
            ylabel="PR-AUC",
            title="CVD prediction PR-AUC",
            show_ci=True,
            show=show,
            save_individual=save_individual,
            pdf_pages=pdf,
        )

        outputs["individual_plots"]["brier_panels"] = plot_metric_panels(
            summary_csv,
            metric_prefix="brier",
            ylabel="Brier score",
            title="CVD prediction Brier score",
            show_ci=True,
            show=show,
            save_individual=save_individual,
            pdf_pages=pdf,
        )

        outputs["individual_plots"]["log_loss_panels"] = plot_metric_panels(
            summary_csv,
            metric_prefix="log_loss",
            ylabel="Log loss",
            title="CVD prediction log loss",
            show_ci=True,
            show=show,
            save_individual=save_individual,
            pdf_pages=pdf,
        )

        outputs["individual_plots"]["rmse_vs_roc_auc"] = plot_rmse_vs_metric(
            summary_csv,
            metric_prefix="roc_auc",
            ylabel="ROC-AUC",
            show=show,
            save_individual=save_individual,
            pdf_pages=pdf,
        )

        outputs["individual_plots"]["rmse_vs_pr_auc"] = plot_rmse_vs_metric(
            summary_csv,
            metric_prefix="pr_auc",
            ylabel="PR-AUC",
            show=show,
            save_individual=save_individual,
            pdf_pages=pdf,
        )

        try:
            outputs["individual_plots"]["delta_roc_auc_vs_baseline"] = plot_delta_vs_baseline(
                summary_csv,
                metric_prefix="roc_auc",
                baseline_method=baseline_method,
                higher_is_better=True,
                ylabel=f"Delta ROC-AUC versus {baseline_method}",
                show=show,
                save_individual=save_individual,
                pdf_pages=pdf,
            )
        except Exception as e:
            print(f"Skipping delta ROC-AUC versus {baseline_method}: {e}")

        try:
            outputs["individual_plots"]["delta_brier_vs_baseline"] = plot_delta_vs_baseline(
                summary_csv,
                metric_prefix="brier",
                baseline_method=baseline_method,
                higher_is_better=False,
                ylabel=f"Delta Brier improvement versus {baseline_method}",
                show=show,
                save_individual=save_individual,
                pdf_pages=pdf,
            )
        except Exception as e:
            print(f"Skipping delta Brier versus {baseline_method}: {e}")

        outputs["individual_plots"]["calibration_slope"] = plot_calibration_panels(
            summary_csv,
            value="calibration_slope",
            ideal_value=1.0,
            ylabel="Calibration slope",
            show=show,
            save_individual=save_individual,
            pdf_pages=pdf,
        )

        outputs["individual_plots"]["calibration_intercept"] = plot_calibration_panels(
            summary_csv,
            value="calibration_intercept",
            ideal_value=0.0,
            ylabel="Calibration intercept",
            show=show,
            save_individual=save_individual,
            pdf_pages=pdf,
        )

        if fold_csv is not None:
            try:
                outputs["individual_plots"]["fold_roc_auc_boxplot_mnar_30"] = plot_fold_metric_boxplot(
                    fold_csv,
                    metric="roc_auc",
                    fraction=0.3,
                    missingness="MNAR",
                    sort_by="median",
                    show=show,
                    save_individual=save_individual,
                    pdf_pages=pdf,
                )
            except Exception as e:
                print(f"Skipping fold-level ROC-AUC boxplot: {e}")

            try:
                outputs["individual_plots"]["fold_pr_auc_boxplot_mnar_30"] = plot_fold_metric_boxplot(
                    fold_csv,
                    metric="pr_auc",
                    fraction=0.3,
                    missingness="MNAR",
                    sort_by="median",
                    show=show,
                    save_individual=save_individual,
                    pdf_pages=pdf,
                )
            except Exception as e:
                print(f"Skipping fold-level PR-AUC boxplot: {e}")

    print(f"Saved combined CVD downstream PDF report: {combined_pdf_path}")

    return outputs