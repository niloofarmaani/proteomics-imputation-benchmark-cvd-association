# cvd_baseline_tools.py
#
# Expected imputed CSV structure:
#   column 0 = sample ID
#   column 1 = binary CVD label
#   columns 2 onward = imputed protein features
#
# Main workflow:
#   from Evaluation.cvd_baseline_tools import run_cvd_baseline_from_best_rmse_log
#
#   results_csv = run_cvd_baseline_from_best_rmse_log(
#       log_file_path=log_path,
#       imputed_root=main_dir,
#       out_csv_name="cvd_baseline_elasticnet_repeated_summary.csv",
#       n_splits=5,
#       n_repeats=5,
#       inner_cv=5,
#       random_state=42,
#       debug_print_labels=True,
#       allow_drop_missing_labels=False,
#       verbose=True,
#   )

from __future__ import annotations

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

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from sklearn.exceptions import ConvergenceWarning


# =============================================================================
# Parsing and resolving imputed CSV paths
# =============================================================================

_MISS_TYPES = ("MCAR", "MAR", "MNAR")
_BAD_NAME_HINTS = ("rmse", "mse", "mae", "metric", "score", "summary", "log")


def _coerce_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _normalize_missingness(s: str) -> Optional[str]:
    if s is None:
        return None

    up = str(s).strip().upper()

    for m in _MISS_TYPES:
        if m == up:
            return m

    for m in _MISS_TYPES:
        if m in up:
            return m

    return None


def rmse_name_to_imputed_name(rmse_filename: str) -> str:
    """
    Convert RMSE_*.csv to imputed_*.csv using the convention:
      RMSE_X.csv -> imputed_X.csv
    """
    name = Path(rmse_filename).name

    if name.startswith("RMSE_"):
        return "imputed_" + name[len("RMSE_"):]

    return name.replace("RMSE_", "imputed_", 1)


def _looks_like_xy_features_csv(
    p: Path,
    *,
    min_cols: int = 3,
    sample_rows: int = 50,
) -> bool:
    """
    Heuristic guard to avoid accidentally using metric CSVs.

    Expected format:
      column 0 = sample ID
      column 1 = binary label
      columns 2 onward = features
    """
    name_low = p.name.lower()

    if any(b in name_low for b in _BAD_NAME_HINTS):
        return False

    try:
        df = pd.read_csv(p, nrows=sample_rows)
    except Exception:
        return False

    if df.shape[1] < min_cols:
        return False

    y = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    y = y.replace([np.inf, -np.inf], np.nan).dropna()

    if y.empty:
        return False

    uniq = set(y.unique())

    if len(uniq) > 2:
        return False

    if not (uniq.issubset({0, 1}) or uniq.issubset({-1, 1})):
        return False

    return True


def _pick_best_csv_token(tokens: List[str]) -> Optional[str]:
    """
    If multiple CSV tokens appear in one log line, prefer imputed datasets over RMSE files.
    """
    if not tokens:
        return None

    imputed = [t for t in tokens if "imputed" in Path(t).name.lower()]

    if imputed:
        return imputed[0]

    non_bad = [
        t for t in tokens
        if not any(b in Path(t).name.lower() for b in _BAD_NAME_HINTS)
    ]

    if non_bad:
        return non_bad[0]

    return tokens[0]


def _try_read_as_table(log_file_path: Union[str, Path]) -> Optional[pd.DataFrame]:
    """
    Try to parse best_rmse_log.txt as a CSV/TSV/table.

    Expected columns can include:
      method, missingness, fraction, rmse, imputed_file
    """
    log_file_path = Path(log_file_path).expanduser().resolve()

    readers = [
        dict(sep=None, engine="python"),
        dict(sep="\t"),
        dict(sep=","),
        dict(delim_whitespace=True),
    ]

    for kwargs in readers:
        try:
            df = pd.read_csv(log_file_path, **kwargs)
        except Exception:
            continue

        if df.empty:
            continue

        lower_cols = {c.lower().strip(): c for c in df.columns}

        method_col = None
        missing_col = None
        fraction_col = None
        rmse_col = None
        file_col = None

        for key, orig in lower_cols.items():
            if method_col is None and key in {"method", "imputer", "imputation_method"}:
                method_col = orig

            if missing_col is None and key in {"missingness", "missing_type", "missingness_type"}:
                missing_col = orig

            if fraction_col is None and key in {"fraction", "missing_fraction", "frac"}:
                fraction_col = orig

            if rmse_col is None and key in {"rmse", "best_rmse", "rmse_vae", "rmse_value"}:
                rmse_col = orig

            if file_col is None and key in {"imputed_file", "file", "csv", "path", "rmse_file"}:
                file_col = orig

        if method_col and missing_col and fraction_col and rmse_col:
            out = pd.DataFrame(
                {
                    "method": df[method_col].astype(str),
                    "missingness": df[missing_col].map(_normalize_missingness),
                    "fraction": pd.to_numeric(df[fraction_col], errors="coerce"),
                    "rmse": pd.to_numeric(df[rmse_col], errors="coerce"),
                }
            )

            if file_col is not None:
                out["imputed_file"] = df[file_col].astype(str)
            else:
                out["imputed_file"] = None

            out = out.dropna(subset=["method", "missingness", "fraction", "rmse"]).reset_index(drop=True)

            if not out.empty:
                return out

    return None


def _extract_record_from_line(raw: str) -> Optional[Dict]:
    """
    Parse one free-form log line.

    Supports patterns like:
      Method: KNN, Missingness: MAR, Fraction: 0.3, RMSE: 0.31, File: RMSE_...
      KNN MAR 0.3 0.31 RMSE_...
    """
    raw = str(raw).strip()

    if not raw:
        return None

    missingness = _normalize_missingness(raw)

    if missingness is None:
        return None

    method = None
    fraction = None
    rmse = None

    m_method = re.search(
        r"(?:method|imputer|imputation_method)\s*[:=]\s*([A-Za-z0-9_\-]+)",
        raw,
        flags=re.IGNORECASE,
    )
    if m_method:
        method = m_method.group(1).strip()

    m_frac = re.search(
        r"(?:fraction|frac|missing_fraction)\s*[:=]\s*([0-9.]+)",
        raw,
        flags=re.IGNORECASE,
    )
    if m_frac:
        fraction = _coerce_float(m_frac.group(1))

    m_rmse = re.search(
        r"(?:rmse|best_rmse|score)\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        raw,
        flags=re.IGNORECASE,
    )
    if m_rmse:
        rmse = _coerce_float(m_rmse.group(1))

    csv_tokens: List[str] = []

    abs_paths = re.findall(r"(/[^ \t\r\n,;]+\.csv)", raw)
    csv_tokens.extend(abs_paths)

    m_file = re.search(
        r"(?:file|csv|csv_path|path)\s*[:=]\s*([^,\s]+\.csv)",
        raw,
        flags=re.IGNORECASE,
    )
    if m_file:
        csv_tokens.append(m_file.group(1).strip())

    imputed_file = _pick_best_csv_token(csv_tokens)

    if method is None or fraction is None or rmse is None:
        toks = raw.split()
        miss_idx = None

        for i, t in enumerate(toks):
            if _normalize_missingness(t) == missingness:
                miss_idx = i
                break

        def is_float_token(t: str) -> bool:
            try:
                float(t)
                return True
            except Exception:
                return False

        if miss_idx is not None:
            if method is None and miss_idx - 1 >= 0 and not is_float_token(toks[miss_idx - 1]):
                method = toks[miss_idx - 1]

            if fraction is None and miss_idx + 1 < len(toks) and is_float_token(toks[miss_idx + 1]):
                fraction = float(toks[miss_idx + 1])

            if rmse is None and miss_idx + 2 < len(toks) and is_float_token(toks[miss_idx + 2]):
                rmse = float(toks[miss_idx + 2])

            if imputed_file is None:
                for t in toks:
                    if t.endswith(".csv"):
                        imputed_file = t
                        break

        if rmse is None:
            floats = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw)
            floats = [f for f in floats if f not in ("", ".", "+", "-")]

            if floats:
                cand = _coerce_float(floats[-1])
                rmse = rmse if rmse is not None else cand

    if method is None or fraction is None or rmse is None:
        return None

    return {
        "method": str(method),
        "missingness": missingness,
        "fraction": float(fraction),
        "rmse": float(rmse),
        "imputed_file": imputed_file,
    }


def parse_best_rmse_log(
    log_file_path: Union[str, Path],
) -> pd.DataFrame:
    """
    Parse best_rmse_log.txt into a dataframe with columns:
      method, missingness, fraction, rmse, imputed_file
    """
    log_file_path = Path(log_file_path).expanduser().resolve()

    if not log_file_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_file_path}")

    df_table = _try_read_as_table(log_file_path)

    if df_table is not None and not df_table.empty:
        return df_table

    records: List[Dict] = []

    with log_file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            rec = _extract_record_from_line(line)
            if rec is not None:
                records.append(rec)

    if not records:
        raise ValueError(
            "Could not parse any records from the log. "
            "If your log is a table, make sure it has columns like method, missingness, fraction, rmse."
        )

    df = pd.DataFrame(records)
    df["missingness"] = df["missingness"].map(_normalize_missingness)
    df["fraction"] = pd.to_numeric(df["fraction"], errors="coerce")
    df["rmse"] = pd.to_numeric(df["rmse"], errors="coerce")

    return df.dropna(subset=["method", "missingness", "fraction", "rmse"]).reset_index(drop=True)


def _fraction_tokens(frac: float) -> List[str]:
    """
    Build filename-friendly representations of the fraction.
    """
    frac = float(frac)
    toks = set()

    toks.add(f"{frac:g}")
    toks.add(f"{frac:.2f}")
    toks.add(f"{frac:.3f}")
    toks.add(f"{frac:g}".replace(".", "_"))
    toks.add(f"{frac:g}".replace(".", "p"))

    pct = int(round(frac * 100))
    toks.add(str(pct))
    toks.add(f"{pct}pct")
    toks.add(f"{pct}percent")

    toks = {
        t.lstrip("0") if t.startswith("0") and len(t) > 1 else t
        for t in toks
    }

    return sorted(toks, key=len, reverse=True)


def _resolve_imputed_file_for_row(
    row: pd.Series,
    *,
    imputed_root: Path,
    check_exists: bool = True,
    csv_index: Optional[List[Path]] = None,
) -> Path:
    """
    Resolve a usable imputed CSV path for a row.

    Priority:
      1) If row.imputed_file points to a valid dataset CSV, use it
      2) If row.imputed_file is an RMSE_*.csv, map to imputed_*.csv
      3) Else search under imputed_root for a matching imputed CSV
    """
    method = str(row["method"])
    missingness = str(row["missingness"]).strip()
    frac = float(row["fraction"])

    raw_path = row.get("imputed_file", None)

    if isinstance(raw_path, str) and raw_path.strip():
        raw_path = raw_path.strip()
        p = Path(raw_path)

        if not p.is_absolute():
            p = (imputed_root / p).resolve()

        if (not check_exists or p.exists()) and _looks_like_xy_features_csv(p):
            return p

        name_low = p.name.lower()

        if ("rmse" in name_low) or p.name.startswith("RMSE_") or any(b in name_low for b in _BAD_NAME_HINTS):
            imputed_name = rmse_name_to_imputed_name(p.name)

            cand_a = p.with_name(imputed_name)
            if (not check_exists or cand_a.exists()) and _looks_like_xy_features_csv(cand_a):
                return cand_a.resolve()

            for tok in _fraction_tokens(frac):
                subfolder = f"fraction_{tok}_missingness_{missingness}"
                cand_b = (imputed_root / subfolder / imputed_name).resolve()

                if (not check_exists or cand_b.exists()) and _looks_like_xy_features_csv(cand_b):
                    return cand_b

    if csv_index is None:
        csv_index = list(imputed_root.rglob("*.csv"))

    m_low = method.lower()
    miss_low = missingness.lower()
    frac_toks = _fraction_tokens(frac)

    scored: List[Tuple[int, Path]] = []

    for p in csv_index:
        s = str(p).lower()

        if m_low not in s:
            continue

        if miss_low not in s:
            continue

        if not any(tok in s for tok in frac_toks):
            continue

        if any(b in p.name.lower() for b in _BAD_NAME_HINTS):
            continue

        score = 0
        name_low = p.name.lower()

        if "imput" in name_low:
            score += 3

        if "best" in name_low:
            score += 1

        score -= len(p.parts)
        scored.append((score, p))

    if not scored:
        raise FileNotFoundError(
            "Could not resolve an imputed CSV for row:\n"
            f"  method={method}, missingness={missingness}, fraction={frac}\n"
            f"Searched under: {imputed_root}\n"
            "Tip: ensure imputed dataset CSVs exist under folders like "
            "'fraction_<frac>_missingness_<MCAR|MAR|MNAR>' and are named like 'imputed_*.csv'."
        )

    scored.sort(key=lambda x: x[0], reverse=True)

    for _, candidate in scored:
        candidate = candidate.resolve()

        if check_exists and not candidate.exists():
            continue

        if _looks_like_xy_features_csv(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not resolve a valid imputed dataset CSV for row:\n"
        f"  method={method}, missingness={missingness}, fraction={frac}\n"
        f"Searched under: {imputed_root}"
    )


def add_imputed_paths_preserve_methods(
    *,
    log_file_path: Union[str, Path],
    imputed_root: Optional[Union[str, Path]] = None,
    check_exists: bool = True,
) -> pd.DataFrame:
    """
    Returns meta_df with:
      method, missingness, fraction, rmse, imputed_file
    """
    log_file_path = Path(log_file_path).expanduser().resolve()
    log_dir = log_file_path.parent
    imputed_root = Path(imputed_root).expanduser().resolve() if imputed_root is not None else log_dir

    meta_df = parse_best_rmse_log(log_file_path)

    meta_df = (
        meta_df.sort_values(["missingness", "fraction", "method", "rmse"])
        .groupby(["missingness", "fraction", "method"], as_index=False)
        .first()
    )

    csv_index = list(imputed_root.rglob("*.csv"))

    resolved_paths: List[str] = []

    for _, row in meta_df.iterrows():
        p = _resolve_imputed_file_for_row(
            row,
            imputed_root=imputed_root,
            check_exists=check_exists,
            csv_index=csv_index,
        )
        resolved_paths.append(str(p))

    meta_df["imputed_file"] = resolved_paths
    meta_df = meta_df.sort_values(["missingness", "fraction", "method"]).reset_index(drop=True)

    return meta_df


# =============================================================================
# CVD label loading and diagnostics
# =============================================================================

def _coerce_binary_label(
    y: pd.Series,
    *,
    source_file: Union[str, Path] = "",
    label_col_name: str = "",
    allow_drop_missing_labels: bool = False,
    debug_print: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts column 1 label to binary 0/1 safely.

    Returns
    -------
    y_int : np.ndarray
        Clean binary labels.
    valid_mask : np.ndarray
        Boolean mask of rows kept.

    If allow_drop_missing_labels=False, this raises a clear error when labels
    contain NaN/inf instead of crashing with pandas IntCastingNaNError.
    """

    y_num = pd.to_numeric(y, errors="coerce")
    y_num = y_num.replace([np.inf, -np.inf], np.nan)

    n_total = len(y_num)
    n_missing = int(y_num.isna().sum())
    valid_mask = y_num.notna().to_numpy()

    finite_values = sorted(y_num.dropna().unique().tolist())

    if debug_print:
        print("\n" + "=" * 80)
        print("CVD label diagnostic")
        print("=" * 80)
        print(f"Source file: {source_file}")
        print(f"Label column: {label_col_name}")
        print(f"Total rows: {n_total}")
        print(f"Missing or non-finite labels: {n_missing}")
        print(f"Unique finite label values: {finite_values[:20]}")
        print("=" * 80 + "\n")

    if n_missing > 0 and not allow_drop_missing_labels:
        bad_rows = np.where(~valid_mask)[0][:10].tolist()
        raise ValueError(
            "The binary CVD label column contains NaN or infinite values.\n"
            f"File: {source_file}\n"
            f"Label column: {label_col_name}\n"
            f"Number of bad labels: {n_missing}\n"
            f"First bad row indices: {bad_rows}\n"
            "Either fix the label column in the imputed CSVs, or call "
            "run_cvd_baseline_from_best_rmse_log(..., allow_drop_missing_labels=True)."
        )

    vals = sorted(y_num[valid_mask].unique().tolist())

    if len(vals) != 2:
        raise ValueError(
            "The label column is not binary after removing missing values.\n"
            f"File: {source_file}\n"
            f"Label column: {label_col_name}\n"
            f"Unique finite values found: {vals[:20]}"
        )

    if set(vals) == {0, 1}:
        y_clean = y_num[valid_mask].to_numpy(dtype=float)
    elif set(vals) == {-1, 1}:
        y_clean = y_num[valid_mask].map({-1: 0, 1: 1}).to_numpy(dtype=float)
    else:
        mapping = {vals[0]: 0, vals[1]: 1}
        if debug_print:
            print(f"Mapping binary labels as: {mapping}")
        y_clean = y_num[valid_mask].map(mapping).to_numpy(dtype=float)

    y_int = y_clean.astype(int)

    if debug_print:
        print("Class counts after cleaning:")
        print(pd.Series(y_int).value_counts().sort_index().to_string())
        print("=" * 80 + "\n")

    return y_int, valid_mask


def load_X_y(
    csv_path: Union[str, Path],
    *,
    sort_by_eid: bool = True,
    debug_print: bool = True,
    allow_drop_missing_labels: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Backward-compatible loader.

    Expected file structure:
        column 0 = sample ID
        column 1 = binary CVD label
        columns 2 onward = protein features
    """
    X, y, _ = load_X_y_ids(
        csv_path,
        sort_by_eid=sort_by_eid,
        debug_print=debug_print,
        allow_drop_missing_labels=allow_drop_missing_labels,
    )
    return X, y


def load_X_y_ids(
    csv_path: Union[str, Path],
    *,
    sort_by_eid: bool = True,
    debug_print: bool = True,
    allow_drop_missing_labels: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray, pd.Series]:
    """
    Loads X, y, and sample IDs.

    Expected file structure:
        column 0 = sample ID
        column 1 = binary CVD label
        columns 2 onward = features
    """

    csv_path = Path(csv_path).expanduser().resolve()
    df = pd.read_csv(csv_path)

    if df.shape[1] < 3:
        raise ValueError(
            f"Expected at least 3 columns: sample ID, binary label, features. "
            f"Got {df.shape[1]} columns in {csv_path}"
        )

    id_col = df.columns[0]
    label_col = df.columns[1]
    feature_cols = list(df.columns[2:])

    if debug_print:
        print("\n" + "#" * 80)
        print("Loading imputed dataset for CVD downstream task")
        print("#" * 80)
        print(f"File: {csv_path}")
        print(f"Shape: {df.shape}")
        print(f"Column 0 interpreted as sample ID: {id_col}")
        print(f"Column 1 interpreted as binary CVD label: {label_col}")
        print(f"Number of feature columns from column 2 onward: {len(feature_cols)}")
        print("First 8 columns:")
        print(list(df.columns[:8]))
        print("First 5 values of label column:")
        print(df[label_col].head().to_string(index=False))
        print("#" * 80 + "\n")

    ids_all = df[id_col].astype(str).copy()
    y_raw = df[label_col].copy()

    y, valid_mask = _coerce_binary_label(
        y_raw,
        source_file=csv_path,
        label_col_name=label_col,
        allow_drop_missing_labels=allow_drop_missing_labels,
        debug_print=debug_print,
    )

    X = df.iloc[:, 2:].copy()
    X = X.apply(pd.to_numeric, errors="coerce")

    X = X.loc[valid_mask].copy()
    ids = ids_all.loc[valid_mask].copy()

    if sort_by_eid:
        order = np.argsort(ids.to_numpy())
        X = X.iloc[order].reset_index(drop=True)
        y = y[order]
        ids = ids.iloc[order].reset_index(drop=True)

    if debug_print:
        print("\nFinal loaded downstream matrix:")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        print(f"ID count: {len(ids)}")
        print(f"Feature NaN count after loading: {int(X.isna().sum().sum())}")
        print(f"Class balance: {dict(pd.Series(y).value_counts().sort_index())}")
        print("#" * 80 + "\n")

    return X, y, ids


# =============================================================================
# CVD baseline model and repeated CV runner
# =============================================================================

def make_baseline_model(
    *,
    random_state: int = 42,
    inner_cv: int = 5,
    n_jobs_inner: int = -1,
) -> Pipeline:
    return Pipeline(
        [
            ("var", VarianceThreshold(threshold=0.0)),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegressionCV(
                    penalty="elasticnet",
                    solver="saga",
                    l1_ratios=[0.1, 0.5, 0.9],
                    Cs=np.logspace(-3, 3, 10),
                    cv=inner_cv,
                    scoring="roc_auc",
                    class_weight="balanced",
                    max_iter=10000,
                    n_jobs=n_jobs_inner,
                    refit=True,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _safe_binary_metrics(y_true: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    prob = np.clip(prob, 1e-6, 1.0 - 1e-6)

    pred = (prob >= 0.5).astype(int)

    out = {
        "roc_auc": np.nan,
        "pr_auc": np.nan,
        "log_loss": np.nan,
        "brier": np.nan,
        "balanced_acc": np.nan,
        "accuracy": np.nan,
    }

    if len(np.unique(y_true)) >= 2:
        out["roc_auc"] = float(roc_auc_score(y_true, prob))
        out["pr_auc"] = float(average_precision_score(y_true, prob))
        out["log_loss"] = float(log_loss(y_true, prob, labels=[0, 1]))
        out["brier"] = float(brier_score_loss(y_true, prob))
        out["balanced_acc"] = float(balanced_accuracy_score(y_true, pred))
        out["accuracy"] = float(accuracy_score(y_true, pred))

    return out


def _calibration_slope_intercept(y_true: np.ndarray, prob: np.ndarray) -> Tuple[float, float]:
    """
    Calibration model:
      y ~ intercept + slope * logit(predicted probability)

    Ideal:
      intercept = 0
      slope = 1
    """
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    prob = np.clip(prob, 1e-6, 1.0 - 1e-6)

    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan

    logit_prob = np.log(prob / (1.0 - prob)).reshape(-1, 1)

    try:
        cal_model = LogisticRegression(
            C=1e6,
            solver="lbfgs",
            max_iter=10000,
        )
        cal_model.fit(logit_prob, y_true)

        slope = float(cal_model.coef_[0, 0])
        intercept = float(cal_model.intercept_[0])

        return slope, intercept

    except Exception:
        return np.nan, np.nan


def _metric_summary(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "se": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
        }

    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    se = std / np.sqrt(arr.size) if arr.size > 1 else 0.0

    return {
        "mean": mean,
        "std": std,
        "se": se,
        "ci95_low": mean - 1.96 * se,
        "ci95_high": mean + 1.96 * se,
    }


def run_cvd_baseline_from_best_rmse_log(
    *,
    log_file_path: Union[str, Path],
    imputed_root: Optional[Union[str, Path]] = None,
    check_exists: bool = True,
    out_csv_name: str = "cvd_baseline_elasticnet_summary.csv",
    out_fold_csv_name: Optional[str] = None,
    random_state: int = 42,
    n_splits: int = 5,
    n_repeats: int = 5,
    inner_cv: int = 5,
    n_jobs_inner: int = -1,
    outer_n_jobs: int = 1,
    sort_by_eid: bool = True,
    debug_print_labels: bool = True,
    allow_drop_missing_labels: bool = False,
    verbose: bool = True,
) -> Path:
    """
    Main entry point.

    Reads best_rmse_log.txt, resolves imputed CSV paths, runs repeated stratified
    outer CV for each imputed dataset, saves summary CSV, and also writes a
    fold-level metrics CSV.

    The expected imputed CSV layout is:
      column 0 = sample ID
      column 1 = binary CVD label
      columns 2 onward = protein features
    """
    log_file_path = Path(log_file_path).expanduser().resolve()
    log_dir = log_file_path.parent
    log_dir.mkdir(parents=True, exist_ok=True)

    if out_fold_csv_name is None:
        stem = Path(out_csv_name).stem
        out_fold_csv_name = f"{stem}_fold_metrics.csv"

    meta_df = add_imputed_paths_preserve_methods(
        log_file_path=log_file_path,
        imputed_root=imputed_root,
        check_exists=check_exists,
    )

    if meta_df.empty:
        raise ValueError("No datasets found after parsing the log.")

    X0, y0, ids0 = load_X_y_ids(
        meta_df.loc[0, "imputed_file"],
        sort_by_eid=sort_by_eid,
        debug_print=debug_print_labels,
        allow_drop_missing_labels=allow_drop_missing_labels,
    )

    outer_cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )

    splits = list(outer_cv.split(np.zeros_like(y0), y0))

    summary_rows: List[Dict] = []
    fold_rows: List[Dict] = []

    for i, row in meta_df.iterrows():
        csv_path = row["imputed_file"]

        X, y, ids = load_X_y_ids(
            csv_path,
            sort_by_eid=sort_by_eid,
            debug_print=debug_print_labels,
            allow_drop_missing_labels=allow_drop_missing_labels,
        )

        if len(y) != len(y0):
            raise ValueError(
                "Sample count mismatch across datasets. Fixed splits cannot be reused safely.\n"
                f"First dataset n={len(y0)} from {meta_df.loc[0, 'imputed_file']}\n"
                f"Current dataset n={len(y)} from {csv_path}\n"
                "If datasets contain different rows, align them first."
            )

        if not np.array_equal(ids.to_numpy(), ids0.to_numpy()):
            raise ValueError(
                "Sample ID order mismatch across datasets after sorting. "
                "Fixed splits cannot be reused safely."
            )

        if verbose:
            print(
                f"[{i + 1}/{len(meta_df)}] "
                f"{row['missingness']} frac={row['fraction']} method={row['method']}"
            )

        pooled_y = []
        pooled_prob = []
        this_dataset_fold_rows = []

        for split_id, (train_idx, test_idx) in enumerate(splits):
            repeat_id = split_id // n_splits
            fold_id = split_id % n_splits

            model = make_baseline_model(
                random_state=random_state + split_id,
                inner_cv=inner_cv,
                n_jobs_inner=n_jobs_inner,
            )

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)

                model.fit(X.iloc[train_idx], y[train_idx])
                prob = model.predict_proba(X.iloc[test_idx])[:, 1]

            metrics = _safe_binary_metrics(y[test_idx], prob)

            fold_row = {
                "method": row["method"],
                "missingness": row["missingness"],
                "fraction": row["fraction"],
                "rmse": row["rmse"],
                "dataset": csv_path,
                "repeat": repeat_id,
                "fold": fold_id,
                "split_id": split_id,
            }
            fold_row.update(metrics)

            fold_rows.append(fold_row)
            this_dataset_fold_rows.append(fold_row)

            pooled_y.append(y[test_idx])
            pooled_prob.append(prob)

        pooled_y = np.concatenate(pooled_y)
        pooled_prob = np.concatenate(pooled_prob)

        calibration_slope, calibration_intercept = _calibration_slope_intercept(pooled_y, pooled_prob)

        fold_df_one = pd.DataFrame(this_dataset_fold_rows)

        summary_row = {
            "method": row["method"],
            "missingness": row["missingness"],
            "fraction": row["fraction"],
            "rmse": row["rmse"],
            "dataset": csv_path,
            "n_splits": n_splits,
            "n_repeats": n_repeats,
            "n_outer_evaluations": len(splits),
            "outcome_prevalence": float(np.mean(y)),
            "calibration_slope": calibration_slope,
            "calibration_intercept": calibration_intercept,
        }

        for metric in ["roc_auc", "pr_auc", "log_loss", "brier", "balanced_acc", "accuracy"]:
            stats = _metric_summary(fold_df_one[metric].to_numpy(dtype=float))

            summary_row[f"{metric}_mean"] = stats["mean"]
            summary_row[f"{metric}_std"] = stats["std"]
            summary_row[f"{metric}_se"] = stats["se"]
            summary_row[f"{metric}_ci95_low"] = stats["ci95_low"]
            summary_row[f"{metric}_ci95_high"] = stats["ci95_high"]

        summary_rows.append(summary_row)

    results_df = pd.DataFrame(summary_rows)
    fold_df = pd.DataFrame(fold_rows)

    out_csv = (log_dir / out_csv_name).resolve()
    out_fold_csv = (log_dir / out_fold_csv_name).resolve()

    results_df.to_csv(out_csv, index=False)
    fold_df.to_csv(out_fold_csv, index=False)

    if verbose:
        print("Saved results_csv:", out_csv)
        print("Saved fold_metrics_csv:", out_fold_csv)

    return out_csv


# =============================================================================
# Plotting
# =============================================================================

def _metric_error_columns(metric_prefix: str, err_mode: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Returns low_col, high_col, symmetric_err_col.

    If ci95 columns exist, use low/high.
    If std/se columns exist, use symmetric error.
    """
    err_mode = err_mode.lower()

    if err_mode == "ci95":
        return f"{metric_prefix}_ci95_low", f"{metric_prefix}_ci95_high", None

    if err_mode == "se":
        return None, None, f"{metric_prefix}_se"

    return None, None, f"{metric_prefix}_std"


def plot_cvd_metric_panels(
    *,
    results_csv: Union[str, Path],
    metric_prefix: str = "roc_auc",
    y_label: Optional[str] = None,
    err_mode: str = "ci95",
    show: bool = True,
    verbose: bool = True,
    filename_prefix: Optional[str] = None,
) -> Dict[str, Path]:
    """
    Generic downstream plotting function.

    Examples
    --------
    metric_prefix="roc_auc" -> uses roc_auc_mean
    metric_prefix="pr_auc"  -> uses pr_auc_mean
    metric_prefix="brier"   -> uses brier_mean
    metric_prefix="log_loss" -> uses log_loss_mean
    """
    results_csv = Path(results_csv).expanduser().resolve()
    out_dir = results_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_csv)

    mean_col = f"{metric_prefix}_mean"
    required = {"missingness", "fraction", "method", mean_col}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"results_csv is missing required columns: {sorted(missing)}")

    df["fraction"] = pd.to_numeric(df["fraction"], errors="coerce")
    df[mean_col] = pd.to_numeric(df[mean_col], errors="coerce")
    df["missingness"] = df["missingness"].astype(str)
    df["method"] = df["method"].astype(str)
    df = df.dropna(subset=["fraction", mean_col, "missingness", "method"]).copy()

    low_col, high_col, symmetric_err_col = _metric_error_columns(metric_prefix, err_mode)

    if low_col in df.columns:
        df[low_col] = pd.to_numeric(df[low_col], errors="coerce")

    if high_col in df.columns:
        df[high_col] = pd.to_numeric(df[high_col], errors="coerce")

    if symmetric_err_col in df.columns:
        df[symmetric_err_col] = pd.to_numeric(df[symmetric_err_col], errors="coerce")

    preferred_order = ["MCAR", "MAR", "MNAR"]
    present = [m for m in preferred_order if m in df["missingness"].unique()]

    if not present:
        present = sorted(df["missingness"].unique())

    df["missingness"] = pd.Categorical(df["missingness"], categories=present, ordered=True)
    df = df.sort_values(["missingness", "method", "fraction"])

    methods = sorted(df["method"].unique())
    x_unique = np.array(sorted(df["fraction"].dropna().unique()))

    if len(x_unique) > 1:
        min_step = float(np.min(np.diff(x_unique)))
    else:
        min_step = 0.05

    dodge = 0.16 * min_step
    offsets = {
        m: (i - (len(methods) - 1) / 2.0) * dodge
        for i, m in enumerate(methods)
    }

    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    marker_map = {
        m: markers[i % len(markers)]
        for i, m in enumerate(methods)
    }

    default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not default_colors:
        default_colors = ["C0", "C1", "C2", "C3", "C4", "C5"]

    color_map = {
        m: default_colors[i % len(default_colors)]
        for i, m in enumerate(methods)
    }

    n_panels = len(present)

    fig, axes = plt.subplots(
        1,
        n_panels,
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
            x = g["fraction"].to_numpy(dtype=float)
            y = g[mean_col].to_numpy(dtype=float)
            x_off = x + offsets[method]

            c = color_map[method]

            yerr = None
            if low_col is not None and high_col is not None and low_col in g.columns and high_col in g.columns:
                low = g[low_col].to_numpy(dtype=float)
                high = g[high_col].to_numpy(dtype=float)
                yerr = np.vstack([y - low, high - y])
            elif symmetric_err_col is not None and symmetric_err_col in g.columns:
                yerr = g[symmetric_err_col].to_numpy(dtype=float)

            if yerr is not None:
                ax.errorbar(
                    x_off,
                    y,
                    yerr=yerr,
                    fmt="none",
                    ecolor=c,
                    elinewidth=1.2,
                    capsize=3,
                    alpha=0.30,
                    zorder=1,
                )

            ax.plot(
                x_off,
                y,
                marker=marker_map[method],
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

    if y_label is None:
        y_label = mean_col

    axes[0].set_ylabel(y_label)

    if len(x_unique) > 0:
        xmin, xmax = float(x_unique.min()), float(x_unique.max())
        pad = 4.0 * dodge

        for ax in axes:
            ax.set_xlim(xmin - pad, xmax + pad)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="method", loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout(rect=[0, 0, 0.85, 1])

    if filename_prefix is None:
        filename_prefix = f"{metric_prefix}_by_missingness"

    out_pdf = (out_dir / f"{filename_prefix}.pdf").resolve()
    out_png = (out_dir / f"{filename_prefix}.png").resolve()

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)

    outputs = {
        f"{metric_prefix}_pdf": out_pdf,
        f"{metric_prefix}_png": out_png,
    }

    if verbose:
        print("Saved:", out_pdf)
        print("Saved:", out_png)

    return outputs


def plot_cvd_results_rocauc_panels(
    *,
    results_csv: Union[str, Path],
    n_folds: int = 5,
    err_mode: str = "ci95",
    show: bool = True,
    verbose: bool = True,
) -> Dict[str, Path]:
    """
    Backward-compatible ROC-AUC plotting function.

    n_folds is kept for compatibility but is no longer needed if the results CSV
    contains ci95 or std columns from repeated CV.
    """
    return plot_cvd_metric_panels(
        results_csv=results_csv,
        metric_prefix="roc_auc",
        y_label="ROC-AUC",
        err_mode=err_mode,
        show=show,
        verbose=verbose,
        filename_prefix="rocauc_by_missingness_repeated",
    )


def plot_cvd_all_key_metrics(
    *,
    results_csv: Union[str, Path],
    err_mode: str = "ci95",
    show: bool = True,
    verbose: bool = True,
) -> Dict[str, Path]:
    outputs = {}

    outputs.update(
        plot_cvd_metric_panels(
            results_csv=results_csv,
            metric_prefix="roc_auc",
            y_label="ROC-AUC",
            err_mode=err_mode,
            show=show,
            verbose=verbose,
            filename_prefix="rocauc_by_missingness_repeated",
        )
    )

    outputs.update(
        plot_cvd_metric_panels(
            results_csv=results_csv,
            metric_prefix="pr_auc",
            y_label="PR-AUC",
            err_mode=err_mode,
            show=show,
            verbose=verbose,
            filename_prefix="prauc_by_missingness_repeated",
        )
    )

    outputs.update(
        plot_cvd_metric_panels(
            results_csv=results_csv,
            metric_prefix="brier",
            y_label="Brier score",
            err_mode=err_mode,
            show=show,
            verbose=verbose,
            filename_prefix="brier_by_missingness_repeated",
        )
    )

    outputs.update(
        plot_cvd_metric_panels(
            results_csv=results_csv,
            metric_prefix="log_loss",
            y_label="Log loss",
            err_mode=err_mode,
            show=show,
            verbose=verbose,
            filename_prefix="logloss_by_missingness_repeated",
        )
    )

    return outputs