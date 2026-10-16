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
