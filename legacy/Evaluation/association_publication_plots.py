# Evaluation/association_publication_plots.py
"""
Publication plots from completed association-preservation results.

This module does not refit association models.

Main changes:
- Figures sized for a 131 mm manuscript text width.
- Larger fonts and vertically stacked outcome panels.
- Fixed method order, names, and colours.
- Shared heatmap colour limits across outcomes.
- Improved volcano annotations with connector lines.
- Dot plots for average preservation performance.
- Publication summaries restricted to 10%, 30%, and 50% missingness.
"""

from pathlib import Path
import re
import textwrap
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


# =====================================================================
# Publication settings
# =====================================================================

# Match the width used by \includegraphics in the manuscript.
FIGURE_WIDTH_MM = 131.0
FIGURE_WIDTH_IN = FIGURE_WIDTH_MM / 25.4

PUBLICATION_FRACTIONS = (0.10, 0.30, 0.50)
MISSINGNESS_ORDER = ("MCAR", "MAR", "MNAR")

METHOD_ORDER = (
    "iterative",
    "mf",
    "softImpute",
    "KNN",
    "AE",
    "DAE",
    "GAIN",
    "VAE",
    "Transformer",
)

METHOD_LABELS = {
    "iterative": "Iterative",
    "mf": "MissForest",
    "softImpute": "SoftImpute",
}

OUTCOME_LABELS = {
    "cvd": "CVD",
    "plaque_yn": "Plaque status",
}

# Optional mapping from exact dataset identifiers to verified protein names.
# Unmapped identifiers are displayed unchanged.
#
# Example structure:
# PROTEIN_LABELS = {
#     "exact_identifier_from_your_csv": "Verified protein name",
# }
PROTEIN_LABELS = {}

DEFAULT_RANK_METRICS = (
    "z_spearman",
    "beta_spearman",
    "top50_overlap",
    "fdr_jaccard",
    "beta_rmse",
)

DEFAULT_MAIN_METRICS = (
    "z_spearman",
    "top50_overlap",
    "fdr_jaccard",
)

LOWER_IS_BETTER = {
    "rmse",
    "beta_rmse",
    "mean_rank",
}

METRIC_LABELS = {
    "z_spearman": "$z$-statistic Spearman correlation",
    "beta_spearman": "Effect-size Spearman correlation",
    "top50_overlap": "Top-50 overlap",
    "top20_overlap": "Top-20 overlap",
    "fdr_jaccard": "FDR-set Jaccard similarity",
    "beta_rmse": "Effect-size RMSE",
    "rmse": "Imputation RMSE",
    "mean_rank": "Composite rank (lower is better)",
}


# =====================================================================
# Shared helpers
# =====================================================================

def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
    })


def _path(value):
    return Path(value).expanduser().resolve()


def _outdir(source, output_dir):
    if output_dir is None:
        directory = _path(source).parent / "publication_plots"
    else:
        directory = _path(output_dir)

    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _safe_name(value):
    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(value),
    ).strip("_")


def _metric_label(metric):
    return METRIC_LABELS.get(
        metric,
        metric.replace("_", " "),
    )


def _methods(df):
    present = set(df["method"].astype(str))

    return (
        [method for method in METHOD_ORDER if method in present]
        + sorted(present - set(METHOD_ORDER))
    )


def _colors(methods):
    order = (
        list(METHOD_ORDER)
        + sorted(set(methods) - set(METHOD_ORDER))
    )

    palette = plt.get_cmap("tab10").colors

    return {
        method: palette[index % len(palette)]
        for index, method in enumerate(order)
    }


def _outcomes(df, outcomes):
    available = set(df["outcome"].astype(str))

    if outcomes is None:
        selected = (
            [
                outcome
                for outcome in OUTCOME_LABELS
                if outcome in available
            ]
            + sorted(available - set(OUTCOME_LABELS))
        )
    else:
        selected = list(outcomes)

    if not selected or not set(selected).issubset(available):
        raise ValueError(
            f"Requested outcomes: {selected}. "
            f"Available outcomes: {sorted(available)}"
        )

    return selected


# =====================================================================
# Data loading and validation
# =====================================================================

def _load_summary(summary_csv):
    df = pd.read_csv(_path(summary_csv))

    required = {
        "method",
        "missingness",
        "fraction",
        "outcome",
    }

    if not required.issubset(df.columns):
        raise ValueError(
            "Missing summary columns: "
            f"{sorted(required - set(df.columns))}"
        )

    df["method"] = df["method"].astype(str)
    df["outcome"] = df["outcome"].astype(str)

    df["missingness"] = (
        df["missingness"]
        .astype(str)
        .str.upper()
    )

    df["fraction"] = pd.to_numeric(
        df["fraction"],
        errors="coerce",
    )

    # Filter before computing any publication averages or ranks.
    keep_fraction = np.isclose(
        df["fraction"].to_numpy()[:, None],
        PUBLICATION_FRACTIONS,
    ).any(axis=1)

    keep_mechanism = df["missingness"].isin(
        MISSINGNESS_ORDER
    )

    df = df.loc[
        keep_fraction & keep_mechanism
    ].copy()

    if df.empty:
        raise ValueError(
            "No rows match the publication missingness conditions."
        )

    # Canonical values avoid floating-point grouping mismatches.
    df["fraction"] = [
        min(
            PUBLICATION_FRACTIONS,
            key=lambda target: abs(target - fraction),
        )
        for fraction in df["fraction"]
    ]

    keys = [
        "outcome",
        "method",
        "missingness",
        "fraction",
    ]

    if df.duplicated(keys).any():
        raise ValueError(
            "Duplicate outcome/method/condition rows were found. "
            "Supply results from one association seed only."
        )

    numeric_columns = {
        "rmse",
        "best_value_validation_rmse",
        "n_overlap",
        "beta_rmse",
        "n_ref_fdr",
        "n_imp_fdr",
    }

    metric_prefixes = (
        "beta_",
        "z_",
        "top20_",
        "top50_",
        "fdr_",
        "sign_concordance_",
    )

    for column in df.columns:
        if (
            column in numeric_columns
            or column.startswith(metric_prefixes)
        ):
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    # Prevent averages based on different condition panels.
    expected = pd.MultiIndex.from_product(
        [
            sorted(df["outcome"].unique()),
            _methods(df),
            MISSINGNESS_ORDER,
            PUBLICATION_FRACTIONS,
        ],
        names=keys,
    )

    observed = pd.MultiIndex.from_frame(df[keys])
    absent = expected.difference(observed)

    if len(absent):
        raise ValueError(
            f"The publication panel has {len(absent)} missing rows. "
            f"First examples: {list(absent[:4])}"
        )

    return df


def _load_reference(reference_csv):
    ref = pd.read_csv(_path(reference_csv))

    required = {
        "outcome",
        "protein",
        "beta",
        "p",
        "q",
        "z",
        "abs_z",
    }

    if ref.empty or not required.issubset(ref.columns):
        raise ValueError(
            "The reference table is empty or lacks required columns: "
            f"{sorted(required - set(ref.columns))}"
        )

    ref["outcome"] = ref["outcome"].astype(str)

    for column in ("beta", "p", "q", "z", "abs_z"):
        ref[column] = pd.to_numeric(
            ref[column],
            errors="coerce",
        )

    return ref


# =====================================================================
# Figure export
# =====================================================================

def _save_figure(
    fig,
    out_dir,
    filename_prefix,
    *,
    save_individual=True,
    pdf_pages=None,
    show=False,
):
    out_dir = _path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "png": None,
        "pdf": None,
    }

    # Preserve the physical figure width.
    # constrained_layout handles labels and margins.
    with plt.rc_context({"savefig.bbox": None}):
        fig.canvas.draw()

        if save_individual:
            for extension in ("png", "pdf"):
                target = (
                    out_dir
                    / f"{filename_prefix}.{extension}"
                )

                fig.savefig(
                    target,
                    dpi=300,
                    facecolor="white",
                )

                result[extension] = target

        if pdf_pages is not None:
            pdf_pages.savefig(
                fig,
                facecolor="white",
            )

    if show:
        plt.show()

    plt.close(fig)
    return result


# =====================================================================
# Publication summary tables
# =====================================================================

def compute_method_summary(
    summary_csv,
    output_dir=None,
):
    df = _load_summary(summary_csv)

    metrics = [
        metric
        for metric in (
            "rmse",
            "beta_spearman",
            "z_spearman",
            "top50_overlap",
            "fdr_jaccard",
            "beta_rmse",
        )
        if metric in df.columns
    ]

    values = df[metrics].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Non-finite publication metrics were found. "
            "Resolve or explicitly report missing results "
            "before averaging."
        )

    summary = (
        df.groupby(
            ["outcome", "method"],
            as_index=False,
        )[metrics]
        .mean()
    )

    target = (
        _outdir(summary_csv, output_dir)
        / "association_method_summary_for_publication.csv"
    )

    summary.to_csv(target, index=False)
    return target


def compute_composite_ranks(
    summary_csv,
    output_dir=None,
    rank_metrics=DEFAULT_RANK_METRICS,
):
    df = _load_summary(summary_csv)

    missing = set(rank_metrics) - set(df.columns)

    if missing or not rank_metrics:
        raise ValueError(
            "Invalid rank metrics. Missing columns: "
            f"{sorted(missing)}"
        )

    rows = []

    group_keys = [
        "outcome",
        "missingness",
        "fraction",
    ]

    for _, sub in df.groupby(group_keys):
        sub = sub.copy()
        rank_columns = []

        for metric in rank_metrics:
            if not sub[metric].notna().any():
                continue

            column = f"rank_{metric}"

            sub[column] = sub[metric].rank(
                ascending=metric in LOWER_IS_BETTER,
                method="average",
                na_option="bottom",
            )

            rank_columns.append(column)

        if not rank_columns:
            continue

        sub["mean_rank"] = sub[rank_columns].mean(
            axis=1
        )

        keep_columns = (
            group_keys
            + ["method", "mean_rank"]
            + list(rank_metrics)
            + rank_columns
        )

        rows.append(sub[keep_columns])

    if not rows:
        raise ValueError(
            "No composite ranks could be computed."
        )

    detail = pd.concat(
        rows,
        ignore_index=True,
    )

    # Alphabetical method order resolves exact ties deterministically.
    best = (
        detail
        .sort_values(
            group_keys + ["mean_rank", "method"]
        )
        .drop_duplicates(group_keys)
        .reset_index(drop=True)
    )

    directory = _outdir(summary_csv, output_dir)

    detail_path = (
        directory
        / "association_composite_rank_detail.csv"
    )

    best_path = (
        directory
        / "association_best_composite_rank_by_condition.csv"
    )

    detail.to_csv(detail_path, index=False)
    best.to_csv(best_path, index=False)

    return detail, best, detail_path, best_path


# =====================================================================
# Reference volcano plots
# =====================================================================

def plot_reference_volcano(
    reference_csv,
    output_dir=None,
    outcomes=None,
    fdr_threshold=0.05,
    top_n_labels=5,
    show=False,
    save_individual=True,
    pdf_pages=None,
):
    """Side-by-side reference volcano plots at publication width."""
    _style()

    if not 0 < fdr_threshold < 1:
        raise ValueError(
            "fdr_threshold must lie between zero and one."
        )

    if not 0 <= top_n_labels <= 5:
        raise ValueError(
            "Use 0 to 5 labels per outcome "
            "at this publication width."
        )

    ref = _load_reference(reference_csv)
    outcomes = _outcomes(ref, outcomes)
    n_out = len(outcomes)

    # One row of outcome panels, with a shared legend underneath.
    fig = plt.figure(
        figsize=(FIGURE_WIDTH_IN, 3.65),
        constrained_layout=True,
    )

    grid = fig.add_gridspec(
        2,
        n_out,
        height_ratios=[1, 0.12],
    )

    axes = [
        fig.add_subplot(grid[0, index])
        for index in range(n_out)
    ]

    for index, (ax, outcome) in enumerate(
        zip(axes, outcomes)
    ):
        sub = ref.loc[
            ref["outcome"] == outcome
        ].copy()

        valid = (
            np.isfinite(sub["beta"])
            & np.isfinite(sub["q"])
            & sub["q"].between(0, 1)
        )

        if not valid.all():
            warnings.warn(
                f"{outcome}: omitted "
                f"{int((~valid).sum())} invalid volcano points."
            )

        sub = sub.loc[valid].copy()

        if sub.empty:
            raise ValueError(
                f"No valid reference associations for {outcome}."
            )

        sub["logq"] = -np.log10(
            sub["q"].clip(lower=1e-300)
        )

        significant = sub["q"] < fdr_threshold

        groups = (
            (
                ~significant,
                "0.70",
                "Not significant",
            ),
            (
                significant,
                "#BB3434",
                "FDR-significant",
            ),
        )

        for mask, colour, label in groups:
            ax.scatter(
                sub.loc[mask, "beta"],
                sub.loc[mask, "logq"],
                s=13,
                color=colour,
                alpha=0.85,
                edgecolors="none",
                label=label,
                zorder=2,
            )

        # Zero-effect line and FDR significance threshold.
        ax.axvline(
            0,
            color="0.55",
            linewidth=0.7,
            zorder=0,
        )

        ax.axhline(
            -np.log10(fdr_threshold),
            color="0.4",
            linewidth=0.8,
            linestyle="--",
            zorder=0,
        )

        # Short titles fit comfortably within each panel.
        # Significant-protein counts remain in the manuscript caption.
        ax.set_title(
            f"{chr(97 + index)}  "
            f"{OUTCOME_LABELS.get(outcome, outcome)}",
            loc="left",
            fontsize=11,
        )

        ax.set_xlabel(
            r"Effect estimate ($\beta$)",
            fontsize=10,
        )

        ax.set_ylabel(
            r"$-\log_{10}(q)$",
            fontsize=10,
        )

        ax.xaxis.set_major_locator(
            MaxNLocator(nbins=3)
        )

        ax.yaxis.set_major_locator(
            MaxNLocator(nbins=4)
        )

        ax.tick_params(labelsize=9)

        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        # Select annotations using the same absolute-z ranking as before.
        top = (
            sub.loc[np.isfinite(sub["abs_z"])]
            .nlargest(top_n_labels, "abs_z")
        )

        # Reserve annotation space while avoiding excessive empty margins.
        lower = min(
            float(sub["beta"].min()),
            0.0,
        )

        upper = max(
            float(sub["beta"].max()),
            0.0,
        )

        span = max(
            upper - lower,
            0.10,
        )

        left_padding = (
            0.45
            if (top["beta"] < 0).any()
            else 0.10
        )

        right_padding = (
            0.60
            if (top["beta"] >= 0).any()
            else 0.10
        )

        ax.set_xlim(
            lower - left_padding * span,
            upper + right_padding * span,
        )

        ymax = max(
            float(sub["logq"].max()),
            -np.log10(fdr_threshold),
        )

        ax.set_ylim(
            -0.03,
            ymax * 1.18 + 0.1,
        )

        # Distribute protein labels vertically on either side.
        for side in ("left", "right"):
            if side == "left":
                group = top.loc[
                    top["beta"] < 0
                ]
            else:
                group = top.loc[
                    top["beta"] >= 0
                ]

            group = group.sort_values("logq")

            if len(group) > 1:
                positions = np.linspace(
                    0.18,
                    0.88,
                    len(group),
                )
            else:
                positions = [0.84]

            for (_, row), ypos in zip(
                group.iterrows(),
                positions,
            ):
                protein_id = str(row["protein"])

                name = str(
                    PROTEIN_LABELS.get(
                        protein_id,
                        protein_id,
                    )
                )

                # Wrap long identifiers without changing their content.
                name = "\n".join(
                    textwrap.wrap(
                        name,
                        width=14,
                        break_long_words=True,
                    )
                )

                ax.annotate(
                    name,
                    xy=(
                        row["beta"],
                        row["logq"],
                    ),
                    xytext=(
                        0.02 if side == "left" else 0.98,
                        ypos,
                    ),
                    textcoords="axes fraction",
                    ha=(
                        "left"
                        if side == "left"
                        else "right"
                    ),
                    va="center",
                    fontsize=9,
                    zorder=4,
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.9,
                        "pad": 1,
                    },
                    arrowprops={
                        "arrowstyle": "-",
                        "color": "0.35",
                        "lw": 0.6,
                    },
                )

    # Shared legend spanning both outcome panels.
    legend_ax = fig.add_subplot(grid[1, :])
    legend_ax.axis("off")

    handles, labels = (
        axes[0].get_legend_handles_labels()
    )

    legend_ax.legend(
        handles,
        labels,
        loc="center",
        ncol=2,
        fontsize=9,
        frameon=False,
    )

    return _save_figure(
        fig,
        _outdir(reference_csv, output_dir),
        "association_reference_volcano",
        save_individual=save_individual,
        pdf_pages=pdf_pages,
        show=show,
    )

# =====================================================================
# Shared heatmap renderer
# =====================================================================

def _heatmaps(
    df,
    metric,
    output_dir,
    filename,
    outcomes,
    show,
    save_individual,
    pdf_pages,
):
    _style()

    if metric not in df.columns:
        raise ValueError(
            f"Missing metric: {metric}"
        )

    outcomes = _outcomes(df, outcomes)

    df = df.loc[
        df["outcome"].isin(outcomes)
    ]

    methods = _methods(df)

    conditions = pd.MultiIndex.from_product(
        [
            MISSINGNESS_ORDER,
            PUBLICATION_FRACTIONS,
        ],
        names=[
            "missingness",
            "fraction",
        ],
    )

    values = df[metric].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]

    if not len(finite):
        raise ValueError(
            f"No finite values for {metric}."
        )

    # Compute one colour range for all displayed outcomes.
    if metric == "mean_rank":
        vmin = 1.0
        vmax = max(2, len(methods))

    elif metric.endswith(("spearman", "pearson")):
        vmin = min(
            0.95,
            np.floor(finite.min() * 20) / 20,
        )
        vmax = 1.0

    elif metric in LOWER_IS_BETTER:
        vmin = 0.0
        vmax = max(
            float(finite.max()),
            0.01,
        )

    else:
        vmin = 0.0
        vmax = 1.0

    cmap_name = (
        "viridis_r"
        if metric in LOWER_IS_BETTER
        else "viridis"
    )

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("#EEEEEE")

    fig, axes = plt.subplots(
        len(outcomes),
        1,
        squeeze=False,
        figsize=(
            FIGURE_WIDTH_IN,
            2.8 * len(outcomes),
        ),
        constrained_layout=True,
    )

    for index, (ax, outcome) in enumerate(
        zip(axes[:, 0], outcomes)
    ):
        mat = (
            df.loc[df["outcome"] == outcome]
            .pivot(
                index="method",
                columns=[
                    "missingness",
                    "fraction",
                ],
                values=metric,
            )
            .reindex(
                index=methods,
                columns=conditions,
            )
        )

        data = mat.to_numpy(dtype=float)

        image = ax.imshow(
            np.ma.masked_invalid(data),
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_yticks(
            np.arange(len(methods))
        )

        ax.set_yticklabels(
            [
                METHOD_LABELS.get(method, method)
                for method in methods
            ],
            fontsize=9,
        )

        ax.set_xticks(
            np.arange(len(conditions))
        )

        ax.set_xticklabels(
            [
                f"{fraction:.0%}"
                for _, fraction in conditions
            ],
            fontsize=9,
        )

        ax.tick_params(length=0)

        ax.set_title(
            f"{chr(97 + index)}  "
            f"{OUTCOME_LABELS.get(outcome, outcome)}",
            loc="left",
            pad=23,
        )

        # Mechanism headings above each block of fraction columns.
        block_size = len(PUBLICATION_FRACTIONS)

        for block, mechanism in enumerate(
            MISSINGNESS_ORDER
        ):
            centre = (
                block * block_size
                + (block_size - 1) / 2
            )

            ax.text(
                centre,
                1.02,
                mechanism,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=9,
            )

            if block:
                ax.axvline(
                    block * block_size - 0.5,
                    color="white",
                    linewidth=2,
                )

        ax.set_xticks(
            np.arange(
                -0.5,
                len(conditions),
                1,
            ),
            minor=True,
        )

        ax.set_yticks(
            np.arange(
                -0.5,
                len(methods),
                1,
            ),
            minor=True,
        )

        ax.grid(
            which="minor",
            color="white",
            linewidth=0.5,
        )

        ax.tick_params(
            which="minor",
            bottom=False,
            left=False,
        )

        for row in range(data.shape[0]):
            for column in range(data.shape[1]):
                value = data[row, column]

                if np.isfinite(value):
                    number_format = (
                        ".1f"
                        if metric == "mean_rank"
                        else ".2f"
                    )

                    label = format(
                        value,
                        number_format,
                    )

                    rgb = image.cmap(
                        image.norm(value)
                    )[:3]

                else:
                    label = "NA"
                    rgb = (1, 1, 1)

                brightness = np.dot(
                    rgb,
                    [0.299, 0.587, 0.114],
                )

                ax.text(
                    column,
                    row,
                    label,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=(
                        "white"
                        if brightness < 0.5
                        else "black"
                    ),
                )

        colourbar = fig.colorbar(
            image,
            ax=ax,
            fraction=0.04,
            pad=0.025,
        )

        colourbar.ax.tick_params(
            labelsize=9
        )

        colourbar.locator = MaxNLocator(
            nbins=4,
            integer=metric == "mean_rank",
        )

        colourbar.update_ticks()

    return _save_figure(
        fig,
        output_dir,
        filename,
        save_individual=save_individual,
        pdf_pages=pdf_pages,
        show=show,
    )


def plot_metric_heatmaps(
    summary_csv,
    metric,
    output_dir=None,
    outcomes=None,
    method_order_metric="z_spearman",
    show=False,
    save_individual=True,
    pdf_pages=None,
):
    # Retained for compatibility with previous callers.
    # Display order now follows METHOD_ORDER consistently.
    return _heatmaps(
        _load_summary(summary_csv),
        metric,
        _outdir(summary_csv, output_dir),
        f"association_heatmap_{_safe_name(metric)}",
        outcomes,
        show,
        save_individual,
        pdf_pages,
    )


def plot_composite_rank_heatmap(
    summary_csv,
    output_dir=None,
    outcomes=None,
    rank_metrics=DEFAULT_RANK_METRICS,
    show=False,
    save_individual=True,
    pdf_pages=None,
):
    detail, _, _, _ = compute_composite_ranks(
        summary_csv,
        output_dir,
        rank_metrics,
    )

    return _heatmaps(
        detail,
        "mean_rank",
        _outdir(summary_csv, output_dir),
        "association_heatmap_composite_rank",
        outcomes,
        show,
        save_individual,
        pdf_pages,
    )


# =====================================================================
# Average preservation: horizontal dot plots
# =====================================================================

def plot_method_average_summary(
    summary_csv,
    metrics=DEFAULT_MAIN_METRICS,
    output_dir=None,
    outcomes=None,
    show=False,
    save_individual=True,
    pdf_pages=None,
):
    _style()

    df = _load_summary(summary_csv)
    outcomes = _outcomes(df, outcomes)

    if (
        not metrics
        or not set(metrics).issubset(df.columns)
    ):
        raise ValueError(
            "Requested average-summary metrics are unavailable."
        )

    df = df.loc[
        df["outcome"].isin(outcomes)
    ]

    values = df[list(metrics)].to_numpy(
        dtype=float
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "Cannot average incomplete or non-finite metric results."
        )

    methods = _methods(df)
    colours = _colors(methods)

    means = (
        df.groupby(
            ["outcome", "method"]
        )[list(metrics)]
        .mean()
    )

    fig, axes = plt.subplots(
        len(outcomes),
        len(metrics),
        squeeze=False,
        figsize=(
            FIGURE_WIDTH_IN,
            2.65 * len(outcomes),
        ),
        constrained_layout=True,
    )

    short_titles = {
        "z_spearman": "$z$-statistic\nSpearman",
        "top50_overlap": "Top-50\noverlap",
        "fdr_jaccard": "FDR-set\nJaccard",
    }

    for row, outcome in enumerate(outcomes):
        for column, metric in enumerate(metrics):
            ax = axes[row, column]

            average = (
                means
                .xs(outcome)
                .reindex(methods)[metric]
            )

            ax.scatter(
                average,
                np.arange(len(methods)),
                c=[
                    colours[method]
                    for method in methods
                ],
                s=28,
                zorder=3,
                edgecolors="white",
                linewidths=0.4,
            )

            ax.set_yticks(
                np.arange(len(methods))
            )

            if column == 0:
                ax.set_yticklabels(
                    [
                        METHOD_LABELS.get(method, method)
                        for method in methods
                    ]
                )

                ax.set_ylabel(
                    OUTCOME_LABELS.get(outcome, outcome),
                    fontweight="bold",
                )

            else:
                ax.set_yticklabels([])

            ax.set_ylim(
                len(methods) - 0.5,
                -0.5,
            )

            ax.set_title(
                short_titles.get(
                    metric,
                    _metric_label(metric),
                ),
                fontsize=10,
            )

            ax.set_xlabel(
                "Mean",
                fontsize=9,
            )

            # Same limits for the same metric across both outcomes.
            all_values = means[metric].to_numpy()

            padding = max(
                float(np.ptp(all_values)) * 0.12,
                0.01,
            )

            low = float(all_values.min()) - padding
            high = float(all_values.max()) + padding

            if metric.endswith(("spearman", "pearson")):
                low = max(-1, low)
                high = min(1, high)

            elif metric not in LOWER_IS_BETTER:
                low = max(0, low)
                high = min(1, high)

            else:
                low = max(0, low)

            ax.set_xlim(low, high)

            ax.xaxis.set_major_locator(
                MaxNLocator(nbins=2)
            )

            ax.grid(
                axis="x",
                color="0.88",
                linewidth=0.6,
            )

            ax.tick_params(
                axis="y",
                length=0,
            )

            ax.spines[
                ["top", "right", "left"]
            ].set_visible(False)

    return _save_figure(
        fig,
        _outdir(summary_csv, output_dir),
        "association_method_average_summary",
        save_individual=save_individual,
        pdf_pages=pdf_pages,
        show=show,
    )


# =====================================================================
# Reconstruction RMSE versus association preservation
# =====================================================================

def plot_rmse_vs_metric(
    summary_csv,
    metric="z_spearman",
    output_dir=None,
    outcomes=None,
    show=False,
    save_individual=True,
    pdf_pages=None,
):
    _style()

    df = _load_summary(summary_csv)

    if not {"rmse", metric}.issubset(df.columns):
        raise ValueError(
            f"The scatter plot requires rmse and {metric}."
        )

    outcomes = _outcomes(df, outcomes)
    methods = _methods(df)
    colours = _colors(methods)

    markers = {
        "MCAR": "o",
        "MAR": "s",
        "MNAR": "^",
    }

    sizes = {
        fraction: (
            15
            + 55 * fraction / max(PUBLICATION_FRACTIONS)
        )
        for fraction in PUBLICATION_FRACTIONS
    }

    fig = plt.figure(
        figsize=(
            FIGURE_WIDTH_IN,
            2.25 * len(outcomes) + 1.2,
        ),
        constrained_layout=True,
    )

    grid = fig.add_gridspec(
        len(outcomes) + 1,
        1,
        height_ratios=(
            [2.25] * len(outcomes) + [1.2]
        ),
    )

    first_axis = None

    for index, outcome in enumerate(outcomes):
        ax = fig.add_subplot(
            grid[index, 0],
            sharex=first_axis,
            sharey=first_axis,
        )

        if first_axis is None:
            first_axis = ax

        sub = df.loc[
            df["outcome"] == outcome
        ]

        for _, result in sub.iterrows():
            ax.scatter(
                result["rmse"],
                result[metric],
                s=sizes[result["fraction"]],
                marker=markers[result["missingness"]],
                color=colours[result["method"]],
                alpha=0.8,
                edgecolors="0.25",
                linewidths=0.25,
            )

        ax.set_title(
            f"{chr(97 + index)}  "
            f"{OUTCOME_LABELS.get(outcome, outcome)}",
            loc="left",
        )

        ax.set_xlabel(
            "Reconstruction RMSE"
        )

        if metric == "z_spearman":
            ax.set_ylabel(
                "$z$-statistic\nSpearman correlation"
            )
        else:
            ax.set_ylabel(
                _metric_label(metric)
            )

        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=colours[method],
            label=METHOD_LABELS.get(method, method),
            markersize=5,
        )
        for method in methods
    ]

    handles += [
        Line2D(
            [],
            [],
            marker=markers[mechanism],
            linestyle="",
            color="0.3",
            label=mechanism,
            markersize=5,
        )
        for mechanism in MISSINGNESS_ORDER
    ]

    handles += [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color="0.6",
            label=f"{fraction:.0%} missing",
            markersize=np.sqrt(sizes[fraction]),
        )
        for fraction in PUBLICATION_FRACTIONS
    ]

    legend_ax = fig.add_subplot(grid[-1, 0])
    legend_ax.axis("off")

    legend_ax.legend(
        handles=handles,
        ncol=3,
        loc="center",
        frameon=False,
        fontsize=8.5,
        columnspacing=1,
        handletextpad=0.5,
    )

    return _save_figure(
        fig,
        _outdir(summary_csv, output_dir),
        f"association_rmse_vs_{_safe_name(metric)}",
        save_individual=save_individual,
        pdf_pages=pdf_pages,
        show=show,
    )


# =====================================================================
# Main report function: compatible with the existing runner
# =====================================================================

def make_association_publication_report(
    summary_csv,
    reference_csv,
    output_dir=None,
    outcomes=None,
    heatmap_metrics=DEFAULT_MAIN_METRICS,
    rank_metrics=DEFAULT_RANK_METRICS,
    combined_pdf_name="association_publication_report.pdf",
    show=False,
    save_individual=True,
):
    directory = _outdir(
        summary_csv,
        output_dir,
    )

    df = _load_summary(summary_csv)
    selected_outcomes = _outcomes(df, outcomes)

    # Confirm that the requested outcomes exist in the reference table.
    _outcomes(
        _load_reference(reference_csv),
        selected_outcomes,
    )

    # Record exactly which aggregate results entered the publication plots.
    selected_csv = (
        directory
        / "association_publication_selected_summary.csv"
    )

    if selected_csv.resolve() == _path(summary_csv):
        raise ValueError(
            "Use the original association summary as input, "
            "not this report's selected-summary output."
        )

    df.loc[
        df["outcome"].isin(selected_outcomes)
    ].to_csv(
        selected_csv,
        index=False,
    )

    method_summary = compute_method_summary(
        selected_csv,
        directory,
    )

    _, _, rank_detail_path, best_rank_path = (
        compute_composite_ranks(
            selected_csv,
            directory,
            rank_metrics,
        )
    )

    outputs = {
        "combined_pdf": directory / combined_pdf_name,
        "plots": {},
        "tables": {
            "selected_summary": selected_csv,
            "method_summary": method_summary,
            "composite_rank_detail": rank_detail_path,
            "best_composite_rank": best_rank_path,
        },
    }

    with PdfPages(outputs["combined_pdf"]) as pdf:
        common = {
            "output_dir": directory,
            "outcomes": selected_outcomes,
            "show": show,
            "save_individual": save_individual,
            "pdf_pages": pdf,
        }

        outputs["plots"]["reference_volcano"] = (
            plot_reference_volcano(
                reference_csv,
                **common,
            )
        )

        for metric in heatmap_metrics:
            outputs["plots"][f"heatmap_{metric}"] = (
                plot_metric_heatmaps(
                    selected_csv,
                    metric,
                    **common,
                )
            )

        outputs["plots"]["composite_rank_heatmap"] = (
            plot_composite_rank_heatmap(
                selected_csv,
                rank_metrics=rank_metrics,
                **common,
            )
        )

        outputs["plots"]["rmse_vs_z_spearman"] = (
            plot_rmse_vs_metric(
                selected_csv,
                **common,
            )
        )

        outputs["plots"]["method_average_summary"] = (
            plot_method_average_summary(
                selected_csv,
                **common,
            )
        )

    print(
        "Saved publication association report: "
        f"{outputs['combined_pdf']}"
    )

    return outputs