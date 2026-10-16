from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.exceptions import ConvergenceWarning


import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, cross_validate

from sklearn.exceptions import ConvergenceWarning
# ============================================================
# Constants
# ============================================================

_MISS_TYPES = ("MCAR", "MAR", "MNAR")
_BAD_NAME_HINTS = ("rmse", "mse", "mae", "metric", "score", "summary", "log")


# ============================================================
# Utilities
# ============================================================

def _coerce_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _normalize_missingness(s: str) -> Optional[str]:
    if s is None:
        return None
    up = str(s).strip().upper()
    for m in _MISS_TYPES:
        if m == up or m in up:
            return m
    return None


def _looks_like_xy_features_csv(p: Path) -> bool:
    if any(b in p.name.lower() for b in _BAD_NAME_HINTS):
        return False
    try:
        df = pd.read_csv(p, nrows=20)
    except Exception:
        return False
    if df.shape[1] < 3:
        return False
    y = pd.to_numeric(df.iloc[:, 1], errors="coerce").dropna()
    if y.empty:
        return False
    return set(y.unique()).issubset({0, 1, -1})


# ============================================================
# Log parsing (NEW FORMAT)
# ============================================================

def _extract_record_from_line(line: str) -> Optional[Dict]:
    raw = line.strip()
    if not raw:
        return None

    miss_m = re.search(r"\b(MCAR|MAR|MNAR)\b", raw, flags=re.IGNORECASE)
    if not miss_m:
        return None

    method_m = re.search(r"method\s*:\s*([^,]+)", raw, flags=re.IGNORECASE)
    frac_m = re.search(r"fraction\s*:\s*([0-9.eE+-]+)", raw, flags=re.IGNORECASE)
    rmse_m = re.search(r"rmse\s*:\s*([0-9.eE+-]+)", raw, flags=re.IGNORECASE)
    file_m = re.search(r"file\s*:\s*([^,\s]+\.csv)", raw, flags=re.IGNORECASE)

    if not (method_m and frac_m and rmse_m):
        return None

    return {
        "method": method_m.group(1).strip().rstrip(","),
        "missingness": _normalize_missingness(miss_m.group(1)),
        "fraction": _coerce_float(frac_m.group(1)),
        "rmse": _coerce_float(rmse_m.group(1)),
        "rmse_file": file_m.group(1) if file_m else None,
    }


def parse_best_rmse_log(log_file_path: Union[str, Path]) -> pd.DataFrame:
    log_file_path = Path(log_file_path).expanduser().resolve()
    records: List[Dict] = []

    with log_file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            rec = _extract_record_from_line(line)
            if rec:
                records.append(rec)

    if not records:
        raise ValueError("No valid records parsed from log file")

    df = pd.DataFrame(records)
    return df.dropna(subset=["method", "missingness", "fraction", "rmse"]).reset_index(drop=True)


# ============================================================
# Imputed file resolver (FIXED FOR YOUR DATA)
# ============================================================

def _resolve_imputed_file_for_row(row: pd.Series, imputed_root: Path) -> Path:
    method = row["method"]
    missingness = row["missingness"]
    frac = float(row["fraction"])

    folder = imputed_root / f"fraction_{frac}_missingness_{missingness}"

    if not folder.exists():
        raise FileNotFoundError(f"Missing folder: {folder}")

    candidates = []
    for p in folder.glob("imputed_*.csv"):
        if method in p.name and _looks_like_xy_features_csv(p):
            candidates.append(p)

    if not candidates:
        raise FileNotFoundError(
            "No matching imputed CSV found.\n"
            f"  folder      = {folder}\n"
            f"  method      = {method}\n"
            f"  missingness = {missingness}\n"
            f"  fraction    = {frac}\n"
            f"  expected    = imputed_{method}_*.csv"
        )

    candidates.sort()
    return candidates[0].resolve()


def add_imputed_paths(log_file_path: Union[str, Path], imputed_root: Union[str, Path]) -> pd.DataFrame:
    imputed_root = Path(imputed_root).expanduser().resolve()
    df = parse_best_rmse_log(log_file_path)

    df = (
        df.sort_values(["missingness", "fraction", "method", "rmse"])
        .groupby(["missingness", "fraction", "method"], as_index=False)
        .first()
    )

    paths = []
    for _, row in df.iterrows():
        paths.append(str(_resolve_imputed_file_for_row(row, imputed_root)))

    df["imputed_file"] = paths
    return df.reset_index(drop=True)


# ============================================================
# CVD baseline model
# ============================================================

def load_X_y(csv_path: Union[str, Path]) -> Tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(csv_path)
    df = df.rename(columns={df.columns[0]: "eid", df.columns[1]: "label"})
    df = df.sort_values("eid").reset_index(drop=True)
    y = df["label"].astype(int).to_numpy()
    X = df.drop(columns=["eid", "label"]).apply(pd.to_numeric, errors="coerce")
    return X, y


def make_baseline_model(random_state: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("var", VarianceThreshold(0.0)),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegressionCV(
                    penalty="elasticnet",
                    solver="saga",
                    l1_ratios=[0.1, 0.5, 0.9],
                    Cs=np.logspace(-3, 3, 10),
                    cv=5,
                    scoring="roc_auc",
                    class_weight="balanced",
                    max_iter=10000,
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )


# ============================================================
# Main runner
# ============================================================

def run_cvd_baseline_from_best_rmse_log(
    log_file_path: Union[str, Path],
    imputed_root: Union[str, Path],
    random_state: int = 42,
) -> Path:
    meta_df = add_imputed_paths(log_file_path, imputed_root)

    X0, y0 = load_X_y(meta_df.loc[0, "imputed_file"])
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    splits = list(outer_cv.split(X0, y0))

    results = []

    for i, row in meta_df.iterrows():
        print(f"[{i+1}/{len(meta_df)}] {row['missingness']} frac={row['fraction']} method={row['method']}")

        X, y = load_X_y(row["imputed_file"])
        model = make_baseline_model(random_state=random_state)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            cv = cross_validate(
                model,
                X,
                y,
                cv=splits,
                scoring="roc_auc",
                n_jobs=1,
            )

        results.append(
            {
                "method": row["method"],
                "missingness": row["missingness"],
                "fraction": row["fraction"],
                "rmse": row["rmse"],
                "dataset": row["imputed_file"],
                "roc_auc_mean": float(np.mean(cv["test_score"])),
                "roc_auc_std": float(np.std(cv["test_score"])),
            }
        )

    out_csv = Path(log_file_path).with_name("cvd_baseline_elasticnet_summary.csv")
    pd.DataFrame(results).to_csv(out_csv, index=False)
    return out_csv


# ============================================================
# CLI entry point
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run CVD baseline from best RMSE log (new format supported)"
    )
    parser.add_argument("--log-file", required=True, help="Path to best_rmse_log.txt")
    parser.add_argument(
        "--imputed-root",
        required=True,
        help="Root directory containing fraction_*_missingness_* folders",
    )

    args = parser.parse_args()

    out_csv = run_cvd_baseline_from_best_rmse_log(
        log_file_path=Path(args.log_file),
        imputed_root=Path(args.imputed_root),
    )

    print("Done. Results saved to:", out_csv)
# -----------------------------
# Plotting function (saves next to results_csv and also shows)
# -----------------------------

def plot_cvd_results_rocauc_panels(
    *,
    results_csv: Union[str, Path],
    n_folds: int = 5,
    err_mode: str = "std",   # "std" or "ci95"
    show: bool = True,
    verbose: bool = True,
) -> Dict[str, Path]:
    """
    Reads results_csv and generates these figures next to it:
      - rocauc_by_missingness.png
      - rocauc_by_missingness_dodged.pdf + .png
      - rocauc_by_missingness_dodged_colormatch.pdf + .png

    Also displays the plots (unless show=False).
    """
    results_csv = Path(results_csv).expanduser().resolve()
    out_dir = results_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_csv)

    required = {"missingness", "fraction", "method", "roc_auc_mean", "roc_auc_std"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"results_csv is missing required columns: {sorted(missing)}")

    df["fraction"] = pd.to_numeric(df["fraction"], errors="coerce")
    df["roc_auc_mean"] = pd.to_numeric(df["roc_auc_mean"], errors="coerce")
    df["roc_auc_std"] = pd.to_numeric(df["roc_auc_std"], errors="coerce")
    df["missingness"] = df["missingness"].astype(str)
    df["method"] = df["method"].astype(str)
    df = df.dropna(subset=["fraction", "roc_auc_mean", "roc_auc_std", "missingness", "method"]).copy()

    if err_mode.lower() == "ci95":
        df["roc_auc_err"] = 1.96 * df["roc_auc_std"] / np.sqrt(float(n_folds))
        err_col = "roc_auc_err"
    else:
        err_col = "roc_auc_std"

    preferred_order = ["MCAR", "MAR", "MNAR"]
    present = [m for m in preferred_order if m in df["missingness"].unique()]
    if not present:
        present = sorted(df["missingness"].unique())
    df["missingness"] = pd.Categorical(df["missingness"], categories=present, ordered=True)
    df = df.sort_values(["missingness", "method", "fraction"])

    outputs: Dict[str, Path] = {}

    # Figure 1: simple (no dodge)
    n_panels = len(present)
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(5.5 * n_panels, 4.2),
        sharey=True,
        sharex=True,
    )
    if n_panels == 1:
        axes = [axes]

    for ax, miss in zip(axes, present):
        sub = df[df["missingness"] == miss]
        for method, g in sub.groupby("method"):
            g = g.sort_values("fraction")
            ax.errorbar(
                g["fraction"],
                g["roc_auc_mean"],
                yerr=g[err_col],
                fmt="o-",
                capsize=3,
                linewidth=1.5,
                markersize=5,
                label=method,
            )
        ax.set_title(miss)
        ax.set_xlabel("Fraction")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("ROC-AUC")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="method", loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout(rect=[0, 0, 0.85, 1])

    out_png = (out_dir / "rocauc_by_missingness.png").resolve()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    outputs["rocauc_by_missingness_png"] = out_png
    if verbose:
        print("Saved:", out_png)

    # Setup for dodged figures
    methods = sorted(df["method"].unique())
    x_unique = np.sort(df["fraction"].unique())
    min_step = float(np.min(np.diff(x_unique))) if len(x_unique) > 1 else 0.1

    marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    markers = {m: marker_cycle[i % len(marker_cycle)] for i, m in enumerate(methods)}

    # Figure 2: dodged (grayscale-friendly)
    dodge = 0.16 * min_step
    offsets = {m: (i - (len(methods) - 1) / 2.0) * dodge for i, m in enumerate(methods)}

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(5.5 * n_panels, 4.2),
        sharex=True,
        sharey=True,
    )
    if n_panels == 1:
        axes = [axes]

    for ax, miss in zip(axes, present):
        sub = df[df["missingness"] == miss].sort_values(["method", "fraction"])
        for method, g in sub.groupby("method"):
            g = g.sort_values("fraction")
            x = g["fraction"].to_numpy()
            y = g["roc_auc_mean"].to_numpy()
            e = g[err_col].to_numpy()
            x_off = x + offsets[method]

            ax.plot(
                x_off, y,
                marker=markers[method],
                linestyle="-",
                linewidth=1.8,
                markersize=6,
                alpha=0.95,
                label=method,
            )
            ax.errorbar(
                x_off, y,
                yerr=e,
                fmt="none",
                capsize=3,
                elinewidth=1.2,
                alpha=0.35,
            )

        ax.set_title(miss)
        ax.set_xlabel("Fraction")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("ROC-AUC")

    xmin, xmax = float(x_unique.min()), float(x_unique.max())
    pad = 4.0 * dodge
    for ax in axes:
        ax.set_xlim(xmin - pad, xmax + pad)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="method", loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout(rect=[0, 0, 0.85, 1])

    out_pdf = (out_dir / "rocauc_by_missingness_dodged.pdf").resolve()
    out_png2 = (out_dir / "rocauc_by_missingness_dodged.png").resolve()

    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png2, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    outputs["rocauc_dodged_pdf"] = out_pdf
    outputs["rocauc_dodged_png"] = out_png2
    if verbose:
        print("Saved:", out_pdf)
        print("Saved:", out_png2)

    # Figure 3: dodged + color matched (matplotlib default cycle)
    default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not default_colors:
        default_colors = ["C0", "C1", "C2", "C3", "C4", "C5"]
    color_map = {m: default_colors[i % len(default_colors)] for i, m in enumerate(methods)}

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(5.5 * n_panels, 4.2),
        sharex=True,
        sharey=True,
    )
    if n_panels == 1:
        axes = [axes]

    for ax, miss in zip(axes, present):
        sub = df[df["missingness"] == miss].sort_values(["method", "fraction"])
        for method, g in sub.groupby("method"):
            g = g.sort_values("fraction")
            x = g["fraction"].to_numpy()
            y = g["roc_auc_mean"].to_numpy()
            e = g[err_col].to_numpy()

            x_off = x + offsets[method]
            c = color_map[method]

            ax.errorbar(
                x_off, y, yerr=e,
                fmt="none",
                ecolor=c,
                elinewidth=1.2,
                capsize=3,
                alpha=0.30,
                zorder=1,
            )

            ax.plot(
                x_off, y,
                marker=markers[method],
                linestyle="-",
                linewidth=2.0,
                markersize=6,
                color=c,
                alpha=0.95,
                label=method,
                zorder=2,
            )

        ax.set_title(miss)
        ax.set_xlabel("Fraction")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("ROC-AUC")

    for ax in axes:
        ax.set_xlim(xmin - pad, xmax + pad)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="method", loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout(rect=[0, 0, 0.85, 1])

    out_pdf2 = (out_dir / "rocauc_by_missingness_dodged_colormatch.pdf").resolve()
    out_png3 = (out_dir / "rocauc_by_missingness_dodged_colormatch.png").resolve()

    plt.savefig(out_pdf2, bbox_inches="tight")
    plt.savefig(out_png3, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    outputs["rocauc_dodged_colormatch_pdf"] = out_pdf2
    outputs["rocauc_dodged_colormatch_png"] = out_png3
    if verbose:
        print("Saved:", out_pdf2)
        print("Saved:", out_png3)

    return outputs
