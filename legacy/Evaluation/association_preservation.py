# Evaluation/association_preservation.py

from __future__ import annotations

import re
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import sys

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from config_paths import experiments_dir

try:
    import statsmodels.api as sm
except Exception as e:
    sm = None
    _STATSMODELS_ERROR = e


# =============================================================================
# Method aliases and matching
# =============================================================================
METHOD_ALIASES = {
    "VAE": ["vae"],
    "DAE": ["dae"],
    "AE": ["ae"],
    "GAIN": ["gain"],
    "TRF": ["trf", "transformer"],
    "TRANSFORMER": ["trf", "transformer"],
    "KNN": ["knn"],
    "MEAN": ["mean"],
    "MF": ["mf", "missforest"],
    "MISSFOREST": ["mf", "missforest"],
    "ITERATIVE": ["iterative", "iterative_imputer"],
    "SOFTIMPUTE": ["softimpute", "softimpute_py"],
}


def _canonical_method_name(method: str) -> str:
    m = str(method).strip()

    aliases = {
        "mf": "MF",
        "missforest": "MF",
        "iterative_imputer": "ITERATIVE",
        "iterative": "ITERATIVE",
        "softimpute": "SOFTIMPUTE",
        "softImpute": "SOFTIMPUTE",
        "softImpute_py": "SOFTIMPUTE",
        "trf": "TRF",
        "transformer": "TRF",
        "mean": "MEAN",
    }

    return aliases.get(m, aliases.get(m.lower(), m.upper()))


def _tokenize_text(text: Any) -> str:
    text = str(text).lower()
    text = text.replace(".", "p")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return f"_{text}_" if text else "_"


def _method_matches_text(method: str, text: Any) -> bool:
    canonical = _canonical_method_name(method)
    aliases = METHOD_ALIASES.get(canonical, [canonical.lower()])
    token_text = _tokenize_text(text)

    for alias in aliases:
        alias_token = _tokenize_text(alias)
        if alias_token in token_text:
            return True

    return False


def _safe_name(x: Any) -> str:
    x = str(x)
    x = re.sub(r"[^A-Za-z0-9_.-]+", "_", x)
    x = re.sub(r"_+", "_", x)
    return x.strip("_")


def _require_statsmodels() -> None:
    if sm is None:
        raise ImportError(
            "statsmodels is required for association preservation analysis. "
            f"Original import error: {_STATSMODELS_ERROR}"
        )


def normalize_id_series(s: pd.Series) -> pd.Series:
    out = s.astype(str).str.strip()
    out = out.str.replace(r"\.0$", "", regex=True)
    out = out.replace(
        {
            "nan": np.nan,
            "NaN": np.nan,
            "None": np.nan,
            "<NA>": np.nan,
            "": np.nan,
        }
    )
    return out


# =============================================================================
# Final frozen manifest from Bayesian best-parameter JSONs
# =============================================================================
def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _find_best_json_for_method(
    *,
    best_params_dir: Path,
    method: str,
    study_name_contains: Optional[str] = "balanced",
    exclude_debug: bool = True,
) -> Optional[Path]:
    if not best_params_dir.exists():
        return None

    candidates = sorted(best_params_dir.glob("*_best_params.json"))
    matched = []

    for p in candidates:
        name_low = p.name.lower()

        if exclude_debug and "debug" in name_low:
            continue

        if study_name_contains is not None:
            if str(study_name_contains).lower() not in name_low:
                continue

        try:
            payload = _load_json(p)
            method_from_json = str(payload.get("method", ""))
            study_from_json = str(payload.get("study_name", ""))
        except Exception:
            method_from_json = ""
            study_from_json = ""

        text = " ".join([p.stem, method_from_json, study_from_json])

        if _method_matches_text(method, text):
            matched.append(p)

    if len(matched) == 0:
        return None

    matched = sorted(matched, key=lambda x: x.stat().st_mtime, reverse=True)
    return matched[0]


def _flatten_param_tokens(params: Dict[str, Any]) -> List[str]:
    tokens = []

    def add_token(x: Any):
        if x is None:
            return

        if isinstance(x, float):
            vals = [
                str(x),
                f"{x:g}",
                f"{x:.6g}",
                str(x).replace(".", "p"),
            ]
        elif isinstance(x, (list, tuple)):
            vals = [
                "-".join(map(str, x)),
                "_".join(map(str, x)),
            ]

            for item in x:
                add_token(item)

            for v in vals:
                tokens.append(v)

            return
        else:
            vals = [str(x)]

        for v in vals:
            v = str(v).strip()
            if v:
                tokens.append(v)

    for k, v in params.items():
        tokens.append(str(k))
        add_token(v)

    clean = []

    for t in tokens:
        t = str(t).lower()
        t = t.replace(".", "p")
        t = re.sub(r"[^a-z0-9]+", "_", t)
        t = t.strip("_")

        if len(t) >= 1:
            clean.append(t)

    return sorted(set(clean), key=len, reverse=True)


def _filename_tokenize(name: str) -> str:
    x = str(name).lower()
    x = x.replace(".", "p")
    x = re.sub(r"[^a-z0-9]+", "_", x)
    x = x.strip("_")
    return f"_{x}_" if x else "_"


def _score_candidate_by_params(
    path: Path,
    method: str,
    params: Dict[str, Any],
) -> float:
    name_tok = _filename_tokenize(path.name)
    score = 0.0
    canonical = _canonical_method_name(method)

    if _method_matches_text(method, path.name):
        score += 100.0

    if canonical == "KNN":
        k = params.get("k", params.get("kn", None))

        if k is not None:
            for pat in [f"kn_{k}", f"k_{k}", f"kn{k}", f"k{k}"]:
                if _filename_tokenize(pat).strip("_") in name_tok:
                    score += 40.0

    if canonical == "SOFTIMPUTE":
        rank = params.get("rank", params.get("max_rank", None))
        maxit = params.get("maxit", params.get("max_iter", None))

        if rank is not None and f"_rank_{rank}_" in name_tok:
            score += 30.0

        if maxit is not None:
            if f"_maxit_{maxit}_" in name_tok or f"_max_iter_{maxit}_" in name_tok:
                score += 30.0

    if canonical in {"MF", "MISSFOREST"}:
        for key in [
            "niter",
            "maxiter",
            "ntree",
            "num_trees",
            "tree",
            "max_depth",
            "sample_fraction",
        ]:
            if key in params:
                val_tok = _filename_tokenize(str(params[key])).strip("_")
                if val_tok in name_tok:
                    score += 12.0

    if canonical == "ITERATIVE":
        for key in ["max_iter", "maxit", "n_iter"]:
            if key in params:
                val_tok = _filename_tokenize(str(params[key])).strip("_")
                if val_tok in name_tok:
                    score += 30.0

    if canonical in {"AE", "DAE", "VAE", "GAIN", "TRF", "TRANSFORMER"}:
        for key in ["batch_size", "bs"]:
            if key in params:
                val = params[key]
                if f"_bs{val}_" in name_tok or f"_bs_{val}_" in name_tok:
                    score += 18.0

        for key in ["lr", "learning_rate"]:
            if key in params:
                val_tok = _filename_tokenize(str(params[key])).strip("_")
                if val_tok in name_tok:
                    score += 18.0

        for key in ["n_neurons", "hidden_dims", "architecture"]:
            if key in params:
                val = params[key]

                if isinstance(val, (list, tuple)):
                    arch1 = "-".join(map(str, val))
                    arch2 = "_".join(map(str, val))

                    if _filename_tokenize(arch1).strip("_") in name_tok:
                        score += 18.0

                    if _filename_tokenize(arch2).strip("_") in name_tok:
                        score += 18.0

        for key in ["dim_latent", "latent_dim", "dl"]:
            if key in params:
                val = params[key]
                if f"_dl{val}_" in name_tok or f"_dl_{val}_" in name_tok:
                    score += 18.0

        for key in ["alpha", "hint_rate", "pseudo_rate", "pseudo_beta"]:
            if key in params:
                val_tok = _filename_tokenize(str(params[key])).strip("_")
                if val_tok in name_tok:
                    score += 8.0

    tokens = _flatten_param_tokens(params)

    for tok in tokens:
        if len(tok) <= 1:
            continue

        if tok in name_tok:
            score += 2.0

    if path.name.lower().startswith("imputed_"):
        score += 10.0

    if "bak" in path.name.lower():
        score -= 1000.0

    return score


def _read_rmse_value(path: Optional[Path]) -> Optional[float]:
    if path is None or not path.exists():
        return None

    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    preferred_cols = [
        "rmse_vae",
        "rmse",
        "RMSE",
        "rmse_knn",
        "rmse_mf",
        "rmse_iterative",
        "rmse_softimpute",
        "rmse_mean",
    ]

    for col in preferred_cols:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()

            if len(vals) > 0:
                return float(vals.iloc[0])

    try:
        vals = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()

        if len(vals) > 0:
            return float(vals.iloc[0])
    except Exception:
        return None

    return None


def _expected_rmse_from_imputed(imputed_path: Path) -> Optional[Path]:
    name = imputed_path.name

    if name.startswith("imputed_"):
        rmse_name = "RMSE_" + name[len("imputed_"):]
        candidate = imputed_path.with_name(rmse_name)

        if candidate.exists():
            return candidate

    candidate = imputed_path.with_name(name.replace("imputed", "RMSE", 1))

    if candidate.exists():
        return candidate

    return None


def _filename_has_seed(path: Path, final_seed: Optional[int]) -> bool:
    """
    Strictly match seed-specific output filenames.

    This intentionally checks the file *stem* instead of the full filename.
    The previous implementation tokenized the full filename and converted
    the extension dot to the letter ``p``. Therefore a valid filename such as

        imputed_softImpute_..._seed_42.csv

    became a token containing ``seed_42pcsv`` and was incorrectly rejected as
    not containing ``_seed_42_``.

    Accepted examples for seed=42:
        *_seed_42.csv
        *_seed42.csv
        *_seed_42_extra.csv
        *_seed42_extra.csv

    If final_seed is None, no seed filtering is applied.
    """
    if final_seed is None:
        return True

    seed = int(final_seed)
    stem = Path(path).stem.lower()

    patterns = [
        rf"(?:^|[_-])seed[_-]{seed}(?:$|[_-])",
        rf"(?:^|[_-])seed{seed}(?:$|[_-])",
    ]

    return any(re.search(pattern, stem) is not None for pattern in patterns)


def _normalize_exact_study_names(exact_study_names: Optional[Dict[str, str]]) -> Dict[str, str]:
    """
    Allow exact study mapping keys to be either display names or canonical names.

    Examples accepted:
        {"mf": "MissForest_..."}
        {"MF": "MissForest_..."}
        {"Transformer": "Transformer_..."}
        {"TRF": "Transformer_..."}
    """
    if exact_study_names is None:
        return {}

    out: Dict[str, str] = {}
    for key, value in exact_study_names.items():
        canonical = _canonical_method_name(str(key))
        out[canonical] = str(value)
        out[str(key)] = str(value)
    return out


def _find_exact_best_json_for_method(
    *,
    best_params_dir: Path,
    method: str,
    exact_study_names: Optional[Dict[str, str]],
) -> Optional[Path]:
    """
    Resolve best-parameter JSON by exact study name.

    This avoids accidentally selecting an older non-seeded or failed study just
    because it also contains the word 'balanced'.
    """
    study_map = _normalize_exact_study_names(exact_study_names)
    canonical = _canonical_method_name(method)
    study_name = study_map.get(str(method), study_map.get(canonical))

    if study_name is None:
        return None

    path = best_params_dir / f"{study_name}_best_params.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Exact best-parameter JSON not found for method={method}.\n"
            f"Study name: {study_name}\n"
            f"Expected file: {path}"
        )

    return path


def _canonical_seed_result_csv(main_dir: Path, canonical_method: str) -> Optional[Path]:
    """
    Canonical method-specific repeated-seed CSV generated by the seeded evaluators.

    This is the safest source for association preservation because it records the
    exact imputed file selected from the validation-frozen best hyperparameters.
    """
    mapping = {
        "KNN": ("KNN_seeded_pipeline", "KNN_final_rmse_by_seed.csv"),
        "SOFTIMPUTE": ("SoftImpute_seeded_pipeline", "SoftImpute_final_rmse_by_seed.csv"),
        "VAE": ("VAE_seeded_pipeline", "VAE_final_rmse_by_seed.csv"),
        "DAE": ("DAE_seeded_pipeline", "DAE_final_rmse_by_seed.csv"),
        "AE": ("AE_seeded_pipeline", "AE_final_rmse_by_seed.csv"),
        "GAIN": ("GAIN_seeded_pipeline", "GAIN_final_rmse_by_seed.csv"),
        "RNN": ("RNN_seeded_pipeline", "RNN_final_rmse_by_seed.csv"),
        "TRF": ("Transformer_seeded_pipeline", "Transformer_final_rmse_by_seed.csv"),
        "TRANSFORMER": ("Transformer_seeded_pipeline", "Transformer_final_rmse_by_seed.csv"),
        "MF": ("MissForest_seeded_pipeline", "MissForest_final_rmse_by_seed.csv"),
        "MISSFOREST": ("MissForest_seeded_pipeline", "MissForest_final_rmse_by_seed.csv"),
        "ITERATIVE": ("Iterative_seeded_pipeline", "Iterative_final_rmse_by_seed.csv"),
    }

    if canonical_method not in mapping:
        return None

    folder, filename = mapping[canonical_method]
    return main_dir / "Evaluation" / folder / filename


def _path_from_seed_result_value(value: Any) -> Optional[Path]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    s = str(value).strip()
    if not s:
        return None
    return Path(s).expanduser().resolve()


def _find_final_from_seed_result_csv(
    *,
    main_dir: Path,
    method: str,
    canonical_method: str,
    final_seed: int,
    fraction: float,
    missingness: str,
    require_seed_result_csv: bool,
) -> Optional[Tuple[Path, Optional[Path], float, str]]:
    """
    Resolve final imputed output from the method-specific repeated-seed CSV.

    Returns None only when require_seed_result_csv=False and the canonical CSV is
    absent. If the CSV exists but the requested row is absent or invalid, this
    raises because that indicates the method-specific seeded evaluator was not
    run for the requested grid.
    """
    seed_csv = _canonical_seed_result_csv(main_dir, canonical_method)

    if seed_csv is None:
        return None

    if not seed_csv.exists():
        if require_seed_result_csv:
            raise FileNotFoundError(
                f"Seed-result CSV not found for method={method}:\n{seed_csv}\n"
                "Run the method-specific seeded evaluator first."
            )
        return None

    df = pd.read_csv(seed_csv)
    required = {"fraction", "missingness", "seed", "rmse"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise KeyError(
            f"Seed-result CSV for method={method} is missing required columns {sorted(missing_cols)}.\n"
            f"File: {seed_csv}\nColumns: {list(df.columns)}"
        )

    if "imputed_path" not in df.columns and "imputed_file" not in df.columns:
        raise KeyError(
            f"Seed-result CSV for method={method} must contain imputed_path or imputed_file.\n"
            f"File: {seed_csv}\nColumns: {list(df.columns)}"
        )

    df = df.copy()
    df["fraction"] = pd.to_numeric(df["fraction"], errors="coerce")
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
    df["missingness"] = df["missingness"].astype(str).str.upper()

    sub = df[
        np.isclose(df["fraction"].astype(float), float(fraction))
        & (df["missingness"] == str(missingness).upper())
        & (df["seed"].astype(float) == float(final_seed))
    ].copy()

    if sub.empty:
        raise FileNotFoundError(
            f"No row in seed-result CSV for method={method}, fraction={fraction}, "
            f"missingness={missingness}, seed={final_seed}.\nFile: {seed_csv}"
        )

    if len(sub) > 1:
        raise ValueError(
            f"Multiple rows in seed-result CSV for method={method}, fraction={fraction}, "
            f"missingness={missingness}, seed={final_seed}.\nFile: {seed_csv}"
        )

    row = sub.iloc[0]
    imp_col = "imputed_path" if "imputed_path" in sub.columns else "imputed_file"
    imputed_path = _path_from_seed_result_value(row[imp_col])
    rmse_path = _path_from_seed_result_value(row["rmse_path"]) if "rmse_path" in sub.columns else None

    if imputed_path is None or not imputed_path.exists():
        raise FileNotFoundError(
            f"Imputed file recorded in seed-result CSV does not exist.\n"
            f"Method: {method}\nCSV: {seed_csv}\nValue: {row[imp_col]}"
        )

    if rmse_path is not None and not rmse_path.exists():
        raise FileNotFoundError(
            f"RMSE file recorded in seed-result CSV does not exist.\n"
            f"Method: {method}\nCSV: {seed_csv}\nValue: {row['rmse_path']}"
        )

    if not _filename_has_seed(imputed_path, final_seed):
        raise ValueError(
            f"Imputed file from seed-result CSV does not contain requested seed={final_seed}:\n"
            f"{imputed_path}"
        )

    rmse = float(row["rmse"])
    if not np.isfinite(rmse):
        raise ValueError(f"Non-finite RMSE in seed-result CSV for method={method}: {rmse}")

    return imputed_path, rmse_path, rmse, str(seed_csv)


def _find_matching_rmse_file(
    *,
    folder: Path,
    method: str,
    params: Dict[str, Any],
    final_seed: Optional[int] = None,
) -> Optional[Path]:
    candidates = [
        p
        for p in folder.glob("RMSE*.csv")
        if _method_matches_text(method, p.name)
        and "bak" not in p.name.lower()
        and _filename_has_seed(p, final_seed)
    ]

    if len(candidates) == 0:
        return None

    scored = [
        (_score_candidate_by_params(p, method, params), p)
        for p in candidates
    ]

    scored = sorted(scored, key=lambda x: (x[0], x[1].stat().st_mtime), reverse=True)

    return scored[0][1]


def _find_final_imputed_file(
    *,
    folder: Path,
    method: str,
    params: Dict[str, Any],
    best_json_path: Optional[Path],
    require_fresh_final_outputs: bool,
    allow_stale_with_warning: bool,
    final_seed: Optional[int] = None,
) -> Tuple[Path, Optional[Path], Optional[float], bool, str]:
    candidates = [
        p
        for p in folder.glob("imputed*.csv")
        if _method_matches_text(method, p.name)
        and "bak" not in p.name.lower()
        and _filename_has_seed(p, final_seed)
    ]

    if len(candidates) == 0:
        seed_msg = "" if final_seed is None else f" with seed={int(final_seed)}"
        raise FileNotFoundError(
            f"No imputed files found for method={method}{seed_msg} in folder={folder}.\n"
            "For the seeded pipeline, association preservation must use exact frozen-best outputs "
            "with filenames containing '_seed_<seed>'."
        )

    scored = [
        (_score_candidate_by_params(p, method, params), p)
        for p in candidates
    ]

    scored = sorted(scored, key=lambda x: (x[0], x[1].stat().st_mtime), reverse=True)

    imputed_path = scored[0][1]
    rmse_path = _expected_rmse_from_imputed(imputed_path)

    if rmse_path is not None and not _filename_has_seed(rmse_path, final_seed):
        rmse_path = None

    if rmse_path is None:
        rmse_path = _find_matching_rmse_file(
            folder=folder,
            method=method,
            params=params,
            final_seed=final_seed,
        )

    if rmse_path is None:
        raise FileNotFoundError(
            f"Could not find matching RMSE file for selected imputed file:\n{imputed_path}"
        )

    test_rmse = _read_rmse_value(rmse_path)

    if test_rmse is None or not np.isfinite(test_rmse):
        raise ValueError(f"Could not read finite RMSE value from {rmse_path}")

    is_fresh = True
    message = "freshness_not_checked"

    if best_json_path is not None:
        json_mtime = best_json_path.stat().st_mtime
        imp_mtime = imputed_path.stat().st_mtime
        rmse_mtime = rmse_path.stat().st_mtime

        is_fresh = (imp_mtime >= json_mtime) and (rmse_mtime >= json_mtime)

        if is_fresh:
            message = "final_outputs_newer_than_best_params"
        else:
            message = (
                "final output appears older than best-params JSON. "
                "Rerun run_final_with_best for this method and seed before downstream analysis."
            )

            if require_fresh_final_outputs and not allow_stale_with_warning:
                raise RuntimeError(
                    f"Stale final output detected for method={method}, folder={folder}\n"
                    f"final_seed: {final_seed}\n"
                    f"best_json: {best_json_path}\n"
                    f"imputed_path: {imputed_path}\n"
                    f"rmse_path: {rmse_path}\n"
                    f"{message}"
                )

    return imputed_path, rmse_path, test_rmse, is_fresh, message


def build_final_frozen_manifest_from_best_params(
    *,
    columns_num: str,
    methods_to_compare: Sequence[str],
    fractions: Sequence[float],
    missingness_types: Sequence[str],
    study_name_contains: Optional[str] = "balanced",
    exact_study_names: Optional[Dict[str, str]] = None,
    final_seed: Optional[int] = None,
    require_seed_result_csv: bool = True,
    out_dir: Optional[Union[str, Path]] = None,
    require_fresh_final_outputs: bool = True,
    allow_stale_with_warning: bool = False,
    exclude_debug: bool = True,
    strict: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Path]:
    main_dir = Path(experiments_dir(columns_num))

    if out_dir is None:
        out_dir = main_dir / "association_preservation"
    else:
        out_dir = Path(out_dir).expanduser().resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    best_params_dir = main_dir / "Optimization" / "best_params"

    rows = []
    errors = []

    for method in methods_to_compare:
        canonical = _canonical_method_name(method)

        if canonical == "MEAN":
            best_json_path = None
            best_payload = {
                "method": "Mean",
                "study_name": "Mean_no_optimization",
                "params": {},
                "best_value_validation_rmse": np.nan,
            }
        else:
            best_json_path = _find_exact_best_json_for_method(
                best_params_dir=best_params_dir,
                method=method,
                exact_study_names=exact_study_names,
            )

            if best_json_path is None:
                best_json_path = _find_best_json_for_method(
                    best_params_dir=best_params_dir,
                    method=method,
                    study_name_contains=study_name_contains,
                    exclude_debug=exclude_debug,
                )

            if best_json_path is None:
                msg = (
                    f"No best_params JSON found for method={method} in {best_params_dir}. "
                    f"Provide exact_study_names for seeded analyses or adjust study_name_contains={study_name_contains}."
                )

                if strict:
                    errors.append(msg)
                    continue

                print("WARNING:", msg)
                continue

            best_payload = _load_json(best_json_path)

        params = best_payload.get("params", {})

        if params is None:
            params = {}

        study_name = best_payload.get("study_name", "")
        best_value_validation_rmse = best_payload.get("best_value_validation_rmse", np.nan)

        for fraction in fractions:
            for missingness in missingness_types:
                missingness_upper = str(missingness).upper()
                folder = main_dir / f"fraction_{fraction}_missingness_{missingness_upper}"

                if not folder.exists():
                    msg = f"Folder not found: {folder}"

                    if strict:
                        errors.append(msg)
                        continue

                    print("WARNING:", msg)
                    continue

                try:
                    seed_result_source = ""

                    if final_seed is not None and canonical != "MEAN":
                        from_seed_csv = _find_final_from_seed_result_csv(
                            main_dir=main_dir,
                            method=method,
                            canonical_method=canonical,
                            final_seed=int(final_seed),
                            fraction=float(fraction),
                            missingness=missingness_upper,
                            require_seed_result_csv=require_seed_result_csv,
                        )
                    else:
                        from_seed_csv = None

                    if from_seed_csv is not None:
                        imputed_path, rmse_path, test_rmse, seed_result_source = from_seed_csv
                        is_fresh = True
                        freshness_message = "selected_from_seed_result_csv"

                        if best_json_path is not None and require_fresh_final_outputs:
                            json_mtime = best_json_path.stat().st_mtime
                            imp_mtime = imputed_path.stat().st_mtime
                            rmse_mtime = rmse_path.stat().st_mtime if rmse_path is not None and rmse_path.exists() else imp_mtime
                            is_fresh = (imp_mtime >= json_mtime) and (rmse_mtime >= json_mtime)
                            if not is_fresh and not allow_stale_with_warning:
                                raise RuntimeError(
                                    f"Stale final output detected from seed-result CSV for method={method}.\n"
                                    f"final_seed: {final_seed}\n"
                                    f"best_json: {best_json_path}\n"
                                    f"imputed_path: {imputed_path}\n"
                                    f"rmse_path: {rmse_path}"
                                )
                    else:
                        (
                            imputed_path,
                            rmse_path,
                            test_rmse,
                            is_fresh,
                            freshness_message,
                        ) = _find_final_imputed_file(
                            folder=folder,
                            method=method,
                            params=params,
                            best_json_path=best_json_path,
                            require_fresh_final_outputs=require_fresh_final_outputs,
                            allow_stale_with_warning=allow_stale_with_warning,
                            final_seed=final_seed,
                        )

                    rows.append(
                        {
                            "method": method,
                            "canonical_method": canonical,
                            "study_name": study_name,
                            "fraction": float(fraction),
                            "missingness": missingness_upper,
                            "imputed_file": str(imputed_path),
                            "rmse_file": str(rmse_path) if rmse_path is not None else "",
                            "seed_result_csv": seed_result_source,
                            "test_rmse": test_rmse,
                            "rmse": test_rmse,
                            "final_seed": int(final_seed) if final_seed is not None else np.nan,
                            "best_value_validation_rmse": best_value_validation_rmse,
                            "best_params_json": str(best_json_path) if best_json_path is not None else "fixed_no_optimization",
                            "params_json": json.dumps(params, default=str),
                            "final_output_is_fresh": bool(is_fresh),
                            "freshness_message": freshness_message,
                        }
                    )

                    if verbose:
                        print(
                            f"Manifest row: method={method}, missingness={missingness_upper}, "
                            f"fraction={fraction}, seed={final_seed}, rmse={test_rmse}, imputed={imputed_path.name}"
                        )

                except Exception as e:
                    msg = (
                        f"Could not resolve final output for method={method}, "
                        f"missingness={missingness_upper}, fraction={fraction}: {e}"
                    )

                    if strict:
                        errors.append(msg)
                    else:
                        print("WARNING:", msg)

    if errors:
        error_text = "\n\n".join(errors[:20])
        raise RuntimeError(
            "Manifest construction failed. First errors:\n\n"
            f"{error_text}\n\n"
            "If outputs are stale, rerun run_final_with_best for the relevant methods."
        )

    if len(rows) == 0:
        raise ValueError("No manifest rows were created.")

    manifest_df = pd.DataFrame(rows)
    manifest_path = out_dir / "final_frozen_manifest_from_best_params.csv"
    manifest_df.to_csv(manifest_path, index=False)

    if verbose:
        print("\nSaved fresh final frozen manifest:")
        print(manifest_path)

    return manifest_df, manifest_path


# =============================================================================
# Phenotype and protein data loading
# =============================================================================
def load_phenotypes(
    phenotype_xlsx: Union[str, Path],
    *,
    phenotype_id_col: str = "sid_ext",
    sheet_name: Union[str, int] = 0,
) -> pd.DataFrame:
    phenotype_xlsx = Path(phenotype_xlsx).expanduser().resolve()

    if not phenotype_xlsx.exists():
        raise FileNotFoundError(f"Phenotype file not found: {phenotype_xlsx}")

    pheno = pd.read_excel(phenotype_xlsx, sheet_name=sheet_name)

    if phenotype_id_col not in pheno.columns:
        raise ValueError(
            f"'{phenotype_id_col}' not found in phenotype file. "
            f"Available columns include: {list(pheno.columns[:40])}"
        )

    pheno = pheno.copy()
    pheno["_id_norm"] = normalize_id_series(pheno[phenotype_id_col])

    if pheno["_id_norm"].isna().any():
        bad = pheno.loc[pheno["_id_norm"].isna(), phenotype_id_col].head(10).tolist()
        raise ValueError(
            f"Phenotype file contains missing IDs in {phenotype_id_col}. "
            f"Examples: {bad}"
        )

    if pheno["_id_norm"].duplicated().any():
        dupes = pheno.loc[pheno["_id_norm"].duplicated(), phenotype_id_col].head(10).tolist()
        raise ValueError(
            f"Phenotype file contains duplicated IDs in {phenotype_id_col}. "
            f"Examples: {dupes}"
        )

    return pheno


def load_protein_matrix(
    protein_csv: Union[str, Path],
    *,
    protein_id_col: str = "eid",
    protein_start_col: int = 2,
) -> Tuple[pd.DataFrame, List[str]]:
    protein_csv = Path(protein_csv).expanduser().resolve()

    if not protein_csv.exists():
        raise FileNotFoundError(f"Protein CSV not found: {protein_csv}")

    df = pd.read_csv(protein_csv)

    if df.shape[1] <= protein_start_col:
        raise ValueError(
            f"Protein file has too few columns for protein_start_col={protein_start_col}: "
            f"{protein_csv}"
        )

    if protein_id_col in df.columns:
        id_series = df[protein_id_col]
    else:
        id_series = df.iloc[:, 0]

    out = pd.DataFrame({"_id_norm": normalize_id_series(id_series)})

    if out["_id_norm"].isna().any():
        bad = id_series[out["_id_norm"].isna()].head(10).tolist()
        raise ValueError(
            f"Protein file contains missing IDs. File: {protein_csv}. "
            f"Examples: {bad}"
        )

    protein_df = df.iloc[:, protein_start_col:].copy()
    protein_df = protein_df.apply(pd.to_numeric, errors="coerce")

    protein_cols = list(protein_df.columns)

    out = pd.concat([out, protein_df.reset_index(drop=True)], axis=1)

    return out, protein_cols


def merge_proteins_with_phenotypes(
    *,
    protein_csv: Union[str, Path],
    phenotype_xlsx: Union[str, Path],
    phenotype_id_col: str = "sid_ext",
    protein_id_col: str = "eid",
    protein_start_col: int = 2,
    sheet_name: Union[str, int] = 0,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    protein_df, protein_cols = load_protein_matrix(
        protein_csv,
        protein_id_col=protein_id_col,
        protein_start_col=protein_start_col,
    )

    pheno = load_phenotypes(
        phenotype_xlsx,
        phenotype_id_col=phenotype_id_col,
        sheet_name=sheet_name,
    )

    merged = pheno.merge(
        protein_df,
        on="_id_norm",
        how="inner",
        validate="one_to_one",
    )

    if verbose:
        print("\nMerge check")
        print("=" * 80)
        print(f"Protein CSV: {protein_csv}")
        print(f"Phenotype file: {phenotype_xlsx}")
        print(f"Phenotype rows: {pheno.shape[0]}")
        print(f"Protein rows: {protein_df.shape[0]}")
        print(f"Merged rows: {merged.shape[0]}")
        print(f"Proteins: {len(protein_cols)}")
        print("=" * 80)

    if merged.empty:
        raise ValueError(
            "No samples matched between proteomics and phenotype data. "
            "Check eid and sid_ext formatting."
        )

    return merged, protein_cols


# =============================================================================
# Association model helpers
# =============================================================================
def _bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    q_values = np.full_like(p_values, np.nan, dtype=float)

    mask = np.isfinite(p_values)
    p = p_values[mask]

    if p.size == 0:
        return q_values

    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    ranked = np.minimum(ranked, 1.0)

    q = np.empty_like(p)
    q[order] = ranked
    q_values[mask] = q

    return q_values


def _binary_mapping_from_series(y: pd.Series) -> Tuple[bool, pd.Series, Dict[Any, int]]:
    y_raw = y.copy()

    y_num = pd.to_numeric(y_raw, errors="coerce")
    y_num = y_num.replace([np.inf, -np.inf], np.nan)

    numeric_vals = sorted(y_num.dropna().unique().tolist())

    if len(numeric_vals) == 2:
        if set(numeric_vals) == {0, 1}:
            return True, y_num, {0: 0, 1: 1}

        if set(numeric_vals) == {-1, 1}:
            return True, y_num.map({-1: 0, 1: 1}), {-1: 0, 1: 1}

        mapping = {numeric_vals[0]: 0, numeric_vals[1]: 1}
        return True, y_num.map(mapping), mapping

    y_str = y_raw.astype("string").str.strip()
    y_str = y_str.replace(
        {
            "": pd.NA,
            "nan": pd.NA,
            "NaN": pd.NA,
            "None": pd.NA,
            "<NA>": pd.NA,
        }
    )

    y_low = y_str.str.lower()

    token_map = {
        "0": 0,
        "0.0": 0,
        "false": 0,
        "f": 0,
        "no": 0,
        "n": 0,
        "control": 0,
        "ctrl": 0,
        "negative": 0,
        "neg": 0,
        "absent": 0,
        "none": 0,
        "no disease": 0,
        "no cvd": 0,
        "non-cvd": 0,
        "noncvd": 0,
        "1": 1,
        "1.0": 1,
        "true": 1,
        "t": 1,
        "yes": 1,
        "y": 1,
        "case": 1,
        "positive": 1,
        "pos": 1,
        "present": 1,
        "disease": 1,
        "cvd": 1,
    }

    mapped = y_low.map(token_map)

    nonmissing_original = y_low.notna()
    nonmissing_mapped = mapped.notna()

    if nonmissing_original.sum() > 0 and int(nonmissing_original.sum()) == int(nonmissing_mapped.sum()):
        vals = sorted(mapped.dropna().unique().tolist())

        if set(vals) == {0, 1}:
            used_mapping = {
                str(k): v
                for k, v in token_map.items()
                if k in set(y_low.dropna().unique().tolist())
            }
            return True, mapped.astype(float), used_mapping

    string_vals = sorted(y_low.dropna().unique().tolist())

    if len(string_vals) == 2:
        mapping = {string_vals[0]: 0, string_vals[1]: 1}
        return True, y_low.map(mapping).astype(float), mapping

    return False, y_num, {}


def _prepare_covariates(df: pd.DataFrame, covariates: Sequence[str]) -> pd.DataFrame:
    if len(covariates) == 0:
        return pd.DataFrame(index=df.index)

    missing = [c for c in covariates if c not in df.columns]
    if missing:
        raise ValueError(f"Missing covariates in phenotype file: {missing}")

    cov_df = df[list(covariates)].copy()

    for col in cov_df.columns:
        numeric = pd.to_numeric(cov_df[col], errors="coerce")

        if numeric.notna().sum() >= max(10, int(0.5 * len(numeric))):
            cov_df[col] = numeric
            med = cov_df[col].median()
            cov_df[col] = cov_df[col].replace([np.inf, -np.inf], np.nan).fillna(med)
        else:
            cov_df[col] = cov_df[col].astype(str)
            cov_df[col] = cov_df[col].replace({"nan": np.nan, "None": np.nan, "<NA>": np.nan})
            mode = cov_df[col].mode(dropna=True)
            fill_val = mode.iloc[0] if len(mode) else "missing"
            cov_df[col] = cov_df[col].fillna(fill_val)

    cov_df = pd.get_dummies(cov_df, drop_first=True, dummy_na=False)
    cov_df = cov_df.apply(pd.to_numeric, errors="coerce")

    for col in cov_df.columns:
        med = cov_df[col].median()
        cov_df[col] = cov_df[col].replace([np.inf, -np.inf], np.nan).fillna(med)

    zero_var_cols = [c for c in cov_df.columns if cov_df[c].nunique(dropna=True) <= 1]
    if zero_var_cols:
        cov_df = cov_df.drop(columns=zero_var_cols)

    return cov_df.astype(float)


def _fit_one_protein_association(
    *,
    y: pd.Series,
    protein_values: pd.Series,
    cov_design: pd.DataFrame,
    binary: bool,
    min_n: int = 50,
) -> Dict[str, float]:
    _require_statsmodels()

    y_model = pd.to_numeric(y, errors="coerce")
    y_model = y_model.replace([np.inf, -np.inf], np.nan)

    x = pd.to_numeric(protein_values, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)

    valid = y_model.notna() & x.notna()

    if int(valid.sum()) < min_n:
        return {"beta": np.nan, "se": np.nan, "z": np.nan, "p": np.nan, "n": int(valid.sum())}

    yv = y_model.loc[valid].astype(float)
    xv = x.loc[valid].astype(float)

    if binary and len(np.unique(yv)) < 2:
        return {"beta": np.nan, "se": np.nan, "z": np.nan, "p": np.nan, "n": int(valid.sum())}

    x_std = float(xv.std())

    if not np.isfinite(x_std) or x_std <= 1e-12:
        return {"beta": np.nan, "se": np.nan, "z": np.nan, "p": np.nan, "n": int(valid.sum())}

    protein_std = (xv - float(xv.mean())) / x_std
    protein_std.name = "protein_value"

    X = pd.concat([protein_std, cov_design.loc[valid].copy()], axis=1)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    design = pd.concat([yv.rename("y"), X], axis=1).dropna()

    if design.shape[0] < min_n:
        return {"beta": np.nan, "se": np.nan, "z": np.nan, "p": np.nan, "n": int(design.shape[0])}

    y_fit = design["y"].astype(float)
    X_fit = design.drop(columns=["y"]).astype(float)
    X_fit = sm.add_constant(X_fit, has_constant="add")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if binary:
                fit = sm.GLM(y_fit, X_fit, family=sm.families.Binomial()).fit(maxiter=100)
            else:
                fit = sm.OLS(y_fit, X_fit).fit()

        beta = float(fit.params.get("protein_value", np.nan))
        se = float(fit.bse.get("protein_value", np.nan))
        p = float(fit.pvalues.get("protein_value", np.nan))
        z = beta / se if np.isfinite(se) and se > 0 else np.nan

        return {"beta": beta, "se": se, "z": float(z), "p": p, "n": int(design.shape[0])}

    except Exception:
        return {"beta": np.nan, "se": np.nan, "z": np.nan, "p": np.nan, "n": int(design.shape[0])}


# =============================================================================
# Association scan
# =============================================================================
def association_scan(
    *,
    protein_csv: Union[str, Path],
    phenotype_xlsx: Union[str, Path],
    outcome: str,
    covariates: Sequence[str],
    dataset_id: str,
    method: str = "reference",
    missingness: str = "reference",
    fraction: float = 0.0,
    phenotype_id_col: str = "sid_ext",
    protein_id_col: str = "eid",
    protein_start_col: int = 2,
    sheet_name: Union[str, int] = 0,
    model_type: str = "auto",
    min_n: int = 50,
    verbose: bool = True,
) -> pd.DataFrame:
    _require_statsmodels()

    merged, protein_cols = merge_proteins_with_phenotypes(
        protein_csv=protein_csv,
        phenotype_xlsx=phenotype_xlsx,
        phenotype_id_col=phenotype_id_col,
        protein_id_col=protein_id_col,
        protein_start_col=protein_start_col,
        sheet_name=sheet_name,
        verbose=verbose,
    )

    if outcome not in merged.columns:
        raise ValueError(f"Outcome '{outcome}' not found after merge.")

    y_raw = merged[outcome]
    is_binary_detected, y_binary, binary_mapping = _binary_mapping_from_series(y_raw)

    if model_type == "auto":
        binary = is_binary_detected
    elif model_type == "logistic":
        binary = True
    elif model_type == "linear":
        binary = False
    else:
        raise ValueError("model_type must be one of: 'auto', 'logistic', 'linear'.")

    if binary:
        is_binary, y, mapping = _binary_mapping_from_series(y_raw)

        if not is_binary:
            raise ValueError(
                f"Outcome '{outcome}' was requested as logistic but could not be mapped to 0/1. "
                f"Unique values: {pd.Series(y_raw).dropna().unique()[:20]}"
            )
    else:
        y = pd.to_numeric(y_raw, errors="coerce")
        y = y.replace([np.inf, -np.inf], np.nan)

    cov_design = _prepare_covariates(merged, covariates)

    if verbose:
        y_preview = pd.Series(y_raw).value_counts(dropna=False).head(10)

        print("\nAssociation scan")
        print("=" * 80)
        print(f"dataset_id: {dataset_id}")
        print(f"method: {method}")
        print(f"missingness: {missingness}")
        print(f"fraction: {fraction}")
        print(f"outcome: {outcome}")
        print("raw outcome value counts, first 10:")
        print(y_preview.to_string())
        print(f"binary detected: {is_binary_detected}")
        print(f"binary mapping: {binary_mapping if is_binary_detected else 'not binary'}")
        print(f"model: {'logistic' if binary else 'linear'}")
        print(f"n samples after merge: {merged.shape[0]}")
        print(f"n proteins: {len(protein_cols)}")
        print(f"n covariate columns after encoding: {cov_design.shape[1]}")
        print("=" * 80)

    rows = []

    for j, protein in enumerate(protein_cols, start=1):
        if verbose and (j == 1 or j % 100 == 0 or j == len(protein_cols)):
            print(f"  protein {j}/{len(protein_cols)}")

        res = _fit_one_protein_association(
            y=y,
            protein_values=merged[protein],
            cov_design=cov_design,
            binary=binary,
            min_n=min_n,
        )

        rows.append(
            {
                "dataset_id": dataset_id,
                "method": method,
                "missingness": missingness,
                "fraction": float(fraction),
                "outcome": outcome,
                "model_type": "logistic" if binary else "linear",
                "protein": protein,
                **res,
            }
        )

    out = pd.DataFrame(rows)
    out["q"] = _bh_fdr(out["p"].to_numpy(dtype=float))
    out["abs_z"] = np.abs(out["z"])
    out["direction"] = np.sign(out["beta"])

    return out


# =============================================================================
# Preservation metrics
# =============================================================================
def _topk_set(df: pd.DataFrame, k: int, score_col: str = "abs_z") -> set:
    sub = df.dropna(subset=[score_col]).sort_values(score_col, ascending=False)
    return set(sub["protein"].head(k).astype(str))


def _safe_spearman(x: pd.Series, y: pd.Series) -> float:
    try:
        return float(pd.Series(x).corr(pd.Series(y), method="spearman"))
    except Exception:
        return np.nan


def _safe_pearson(x: pd.Series, y: pd.Series) -> float:
    try:
        return float(pd.Series(x).corr(pd.Series(y), method="pearson"))
    except Exception:
        return np.nan


def compare_association_to_reference(
    *,
    ref_df: pd.DataFrame,
    imp_df: pd.DataFrame,
    outcome: str,
    top_k_values: Sequence[int] = (20, 50),
    fdr_threshold: float = 0.05,
) -> Dict[str, float]:
    ref = ref_df[ref_df["outcome"] == outcome].copy()
    imp = imp_df[imp_df["outcome"] == outcome].copy()

    merged = ref.merge(imp, on="protein", suffixes=("_ref", "_imp"), how="inner")
    merged = merged.dropna(subset=["beta_ref", "beta_imp", "z_ref", "z_imp"]).copy()

    out: Dict[str, float] = {}

    if merged.empty:
        out.update(
            {
                "n_overlap": 0,
                "beta_spearman": np.nan,
                "beta_pearson": np.nan,
                "z_spearman": np.nan,
                "z_pearson": np.nan,
                "sign_concordance_all": np.nan,
                "beta_rmse": np.nan,
                "fdr_jaccard": np.nan,
                "fdr_precision_vs_ref": np.nan,
                "fdr_recall_vs_ref": np.nan,
                "n_ref_fdr": np.nan,
                "n_imp_fdr": np.nan,
            }
        )

        for k in top_k_values:
            out[f"top{k}_overlap"] = np.nan
            out[f"top{k}_jaccard"] = np.nan
            out[f"sign_concordance_top{k}_ref"] = np.nan

        return out

    out["n_overlap"] = int(merged.shape[0])
    out["beta_spearman"] = _safe_spearman(merged["beta_ref"], merged["beta_imp"])
    out["beta_pearson"] = _safe_pearson(merged["beta_ref"], merged["beta_imp"])
    out["z_spearman"] = _safe_spearman(merged["z_ref"], merged["z_imp"])
    out["z_pearson"] = _safe_pearson(merged["z_ref"], merged["z_imp"])

    out["sign_concordance_all"] = float(
        np.mean(np.sign(merged["beta_ref"]) == np.sign(merged["beta_imp"]))
    )

    out["beta_rmse"] = float(
        np.sqrt(np.mean((merged["beta_ref"] - merged["beta_imp"]) ** 2))
    )

    for k in top_k_values:
        ref_top = _topk_set(ref, k)
        imp_top = _topk_set(imp, k)

        if len(ref_top) == 0 or len(imp_top) == 0:
            overlap = np.nan
            jaccard = np.nan
        else:
            overlap = len(ref_top & imp_top) / float(k)
            union = ref_top | imp_top
            jaccard = len(ref_top & imp_top) / float(len(union)) if len(union) else np.nan

        out[f"top{k}_overlap"] = float(overlap) if np.isfinite(overlap) else np.nan
        out[f"top{k}_jaccard"] = float(jaccard) if np.isfinite(jaccard) else np.nan

        top_ref_merged = merged[merged["protein"].astype(str).isin(ref_top)].copy()

        if top_ref_merged.empty:
            out[f"sign_concordance_top{k}_ref"] = np.nan
        else:
            out[f"sign_concordance_top{k}_ref"] = float(
                np.mean(
                    np.sign(top_ref_merged["beta_ref"])
                    == np.sign(top_ref_merged["beta_imp"])
                )
            )

    ref_sig = set(ref.loc[ref["q"] < fdr_threshold, "protein"].astype(str))
    imp_sig = set(imp.loc[imp["q"] < fdr_threshold, "protein"].astype(str))

    inter = ref_sig & imp_sig
    union = ref_sig | imp_sig

    out["n_ref_fdr"] = len(ref_sig)
    out["n_imp_fdr"] = len(imp_sig)
    out["fdr_jaccard"] = len(inter) / float(len(union)) if len(union) else np.nan
    out["fdr_precision_vs_ref"] = len(inter) / float(len(imp_sig)) if len(imp_sig) else np.nan
    out["fdr_recall_vs_ref"] = len(inter) / float(len(ref_sig)) if len(ref_sig) else np.nan

    return out


# =============================================================================
# Main association preservation runner
# =============================================================================
def run_association_preservation_from_best_params(
    *,
    columns_num: str,
    methods_to_compare: Sequence[str],
    fractions: Sequence[float],
    missingness_types: Sequence[str],
    phenotype_xlsx: Union[str, Path],
    reference_protein_csv: Union[str, Path],
    outcomes: Sequence[str],
    covariates: Sequence[str],
    study_name_contains: Optional[str] = "balanced",
    exact_study_names: Optional[Dict[str, str]] = None,
    final_seed: Optional[int] = None,
    require_seed_result_csv: bool = True,
    out_dir: Optional[Union[str, Path]] = None,
    phenotype_id_col: str = "sid_ext",
    protein_id_col: str = "eid",
    protein_start_col: int = 2,
    sheet_name: Union[str, int] = 0,
    model_type: str = "auto",
    top_k_values: Sequence[int] = (20, 50),
    fdr_threshold: float = 0.05,
    min_n: int = 50,
    require_fresh_final_outputs: bool = True,
    allow_stale_with_warning: bool = False,
    exclude_debug: bool = True,
    strict_manifest: bool = True,
    cache_scans: bool = True,
    verbose: bool = True,
) -> Tuple[Path, Path, Path, Path]:
    _require_statsmodels()

    main_dir = Path(experiments_dir(columns_num))

    if out_dir is None:
        out_dir = main_dir / "association_preservation"
    else:
        out_dir = Path(out_dir).expanduser().resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_df, manifest_csv = build_final_frozen_manifest_from_best_params(
        columns_num=columns_num,
        methods_to_compare=methods_to_compare,
        fractions=fractions,
        missingness_types=missingness_types,
        study_name_contains=study_name_contains,
        exact_study_names=exact_study_names,
        final_seed=final_seed,
        require_seed_result_csv=require_seed_result_csv,
        out_dir=out_dir,
        require_fresh_final_outputs=require_fresh_final_outputs,
        allow_stale_with_warning=allow_stale_with_warning,
        exclude_debug=exclude_debug,
        strict=strict_manifest,
        verbose=verbose,
    )

    scans_dir = out_dir / "per_dataset_scans"
    scans_dir.mkdir(parents=True, exist_ok=True)

    reference_scans = []
    imputed_scans = []
    summary_rows = []

    for outcome in outcomes:
        outcome_safe = _safe_name(outcome)
        ref_cache = scans_dir / f"reference_{outcome_safe}.csv"

        if cache_scans and ref_cache.exists():
            if verbose:
                print(f"\nLoading cached reference scan: {ref_cache}")

            ref_scan = pd.read_csv(ref_cache)
        else:
            if verbose:
                print("\n" + "=" * 100)
                print(f"REFERENCE ASSOCIATION SCAN | outcome={outcome}")
                print("=" * 100)

            ref_scan = association_scan(
                protein_csv=reference_protein_csv,
                phenotype_xlsx=phenotype_xlsx,
                outcome=outcome,
                covariates=covariates,
                dataset_id="reference",
                method="reference",
                missingness="reference",
                fraction=0.0,
                phenotype_id_col=phenotype_id_col,
                protein_id_col=protein_id_col,
                protein_start_col=protein_start_col,
                sheet_name=sheet_name,
                model_type=model_type,
                min_n=min_n,
                verbose=verbose,
            )

            ref_scan.to_csv(ref_cache, index=False)

        reference_scans.append(ref_scan)

        for _, row in manifest_df.iterrows():
            method = str(row["method"])
            missingness = str(row["missingness"]).upper()
            fraction = float(row["fraction"])

            if final_seed is None:
                dataset_id = f"{method}_{missingness}_{fraction:g}"
            else:
                dataset_id = f"{method}_{missingness}_{fraction:g}_seed{int(final_seed)}"
            dataset_safe = _safe_name(dataset_id)
            imp_cache = scans_dir / f"{dataset_safe}_{outcome_safe}.csv"

            if cache_scans and imp_cache.exists():
                if verbose:
                    print(f"\nLoading cached imputed scan: {imp_cache}")

                imp_scan = pd.read_csv(imp_cache)
            else:
                if verbose:
                    print("\n" + "-" * 100)
                    print(
                        f"IMPUTED ASSOCIATION SCAN | outcome={outcome} | "
                        f"method={method} | missingness={missingness} | fraction={fraction:g}"
                    )
                    print("-" * 100)

                imp_scan = association_scan(
                    protein_csv=row["imputed_file"],
                    phenotype_xlsx=phenotype_xlsx,
                    outcome=outcome,
                    covariates=covariates,
                    dataset_id=dataset_id,
                    method=method,
                    missingness=missingness,
                    fraction=fraction,
                    phenotype_id_col=phenotype_id_col,
                    protein_id_col=protein_id_col,
                    protein_start_col=protein_start_col,
                    sheet_name=sheet_name,
                    model_type=model_type,
                    min_n=min_n,
                    verbose=verbose,
                )

                imp_scan.to_csv(imp_cache, index=False)

            imputed_scans.append(imp_scan)

            metrics = compare_association_to_reference(
                ref_df=ref_scan,
                imp_df=imp_scan,
                outcome=outcome,
                top_k_values=top_k_values,
                fdr_threshold=fdr_threshold,
            )

            summary_rows.append(
                {
                    "method": method,
                    "missingness": missingness,
                    "fraction": fraction,
                    "rmse": row.get("test_rmse", np.nan),
                    "best_value_validation_rmse": row.get("best_value_validation_rmse", np.nan),
                    "outcome": outcome,
                    "imputed_file": row["imputed_file"],
                    **metrics,
                }
            )

    reference_df = pd.concat(reference_scans, ignore_index=True)
    imputed_df = pd.concat(imputed_scans, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    reference_csv = out_dir / "association_reference_results.csv"
    imputed_csv = out_dir / "association_imputed_results.csv"
    summary_csv = out_dir / "association_preservation_summary.csv"

    reference_df.to_csv(reference_csv, index=False)
    imputed_df.to_csv(imputed_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    if verbose:
        print("\nSaved association preservation outputs:")
        print("Manifest:", manifest_csv)
        print("Reference:", reference_csv)
        print("Imputed:", imputed_csv)
        print("Summary:", summary_csv)

    return manifest_csv, reference_csv, imputed_csv, summary_csv


# =============================================================================
# Plotting and PDF report
# =============================================================================
def _method_color_map(methods: List[str]) -> Dict[str, str]:
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    if not colors:
        colors = [f"C{i}" for i in range(10)]

    return {m: colors[i % len(colors)] for i, m in enumerate(methods)}


def _ordered_missingness(values: Sequence[str]) -> List[str]:
    preferred = ["MCAR", "MAR", "MNAR"]
    values = [str(v).upper() for v in values]
    present = [m for m in preferred if m in set(values)]
    extra = sorted([m for m in set(values) if m not in present])
    return present + extra


def plot_association_preservation_metric(
    *,
    summary_csv: Union[str, Path],
    metric: str = "beta_spearman",
    output_dir: Optional[Union[str, Path]] = None,
    show: bool = True,
    pdf_pages: Optional[PdfPages] = None,
) -> Dict[str, Path]:
    summary_csv = Path(summary_csv).expanduser().resolve()

    if output_dir is None:
        output_dir = summary_csv.parent / "plots"
    else:
        output_dir = Path(output_dir).expanduser().resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_csv)

    required = {"method", "missingness", "fraction", "outcome", metric}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Summary CSV is missing required columns: {sorted(missing)}")

    df["fraction"] = pd.to_numeric(df["fraction"], errors="coerce")
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=["fraction", metric]).copy()

    outputs = {}

    for outcome in sorted(df["outcome"].astype(str).unique()):
        sub_out = df[df["outcome"].astype(str) == outcome].copy()

        missingness_types = _ordered_missingness(sub_out["missingness"].astype(str).unique())
        methods = sorted(sub_out["method"].astype(str).unique())
        colors = _method_color_map(methods)

        fig, axes = plt.subplots(
            1,
            len(missingness_types),
            figsize=(5.2 * len(missingness_types), 4.2),
            sharey=True,
            sharex=True,
        )

        if len(missingness_types) == 1:
            axes = [axes]

        for ax, missingness in zip(axes, missingness_types):
            sub = sub_out[sub_out["missingness"].astype(str).str.upper() == missingness].copy()

            for method in methods:
                g = sub[sub["method"].astype(str) == method].sort_values("fraction")

                if g.empty:
                    continue

                ax.plot(
                    g["fraction"],
                    g[metric],
                    marker="o",
                    linewidth=1.8,
                    color=colors[method],
                    label=method,
                )

            ax.set_title(f"{outcome} | {missingness}")
            ax.set_xlabel("Missing fraction")
            ax.grid(True, alpha=0.30)

        axes[0].set_ylabel(metric)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            title="method",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
        )

        fig.suptitle(f"Association preservation: {metric}", y=1.02)
        fig.tight_layout(rect=[0, 0, 0.86, 1])

        out_png = output_dir / f"association_preservation_{_safe_name(outcome)}_{_safe_name(metric)}.png"
        out_pdf = output_dir / f"association_preservation_{_safe_name(outcome)}_{_safe_name(metric)}.pdf"

        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        fig.savefig(out_pdf, bbox_inches="tight")

        if pdf_pages is not None:
            pdf_pages.savefig(fig, bbox_inches="tight")

        if show:
            plt.show()

        plt.close(fig)

        outputs[f"{outcome}_{metric}_png"] = out_png
        outputs[f"{outcome}_{metric}_pdf"] = out_pdf

    return outputs


def plot_rmse_vs_association_metric(
    *,
    summary_csv: Union[str, Path],
    metric: str = "beta_spearman",
    output_dir: Optional[Union[str, Path]] = None,
    show: bool = True,
    pdf_pages: Optional[PdfPages] = None,
) -> Dict[str, Path]:
    summary_csv = Path(summary_csv).expanduser().resolve()

    if output_dir is None:
        output_dir = summary_csv.parent / "plots"
    else:
        output_dir = Path(output_dir).expanduser().resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_csv)

    required = {"method", "missingness", "fraction", "outcome", "rmse", metric}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Summary CSV is missing required columns: {sorted(missing)}")

    df["rmse"] = pd.to_numeric(df["rmse"], errors="coerce")
    df["fraction"] = pd.to_numeric(df["fraction"], errors="coerce")
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=["rmse", "fraction", metric]).copy()

    outputs = {}

    for outcome in sorted(df["outcome"].astype(str).unique()):
        sub_out = df[df["outcome"].astype(str) == outcome].copy()

        missingness_types = _ordered_missingness(sub_out["missingness"].astype(str).unique())
        methods = sorted(sub_out["method"].astype(str).unique())
        colors = _method_color_map(methods)

        fractions = sorted(sub_out["fraction"].dropna().unique())
        markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
        marker_map = {f: markers[i % len(markers)] for i, f in enumerate(fractions)}

        fig, axes = plt.subplots(
            1,
            len(missingness_types),
            figsize=(5.2 * len(missingness_types), 4.2),
            sharey=True,
        )

        if len(missingness_types) == 1:
            axes = [axes]

        for ax, missingness in zip(axes, missingness_types):
            sub = sub_out[sub_out["missingness"].astype(str).str.upper() == missingness].copy()

            for _, row in sub.iterrows():
                method = str(row["method"])
                fraction = row["fraction"]

                ax.scatter(
                    row["rmse"],
                    row[metric],
                    color=colors[method],
                    marker=marker_map[fraction],
                    s=55,
                    alpha=0.85,
                )

            ax.set_title(f"{outcome} | {missingness}")
            ax.set_xlabel("Imputation RMSE")
            ax.grid(True, alpha=0.30)

        axes[0].set_ylabel(metric)

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
            for f in fractions
        ]

        fig.legend(
            handles=method_handles + fraction_handles,
            title="Legend",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
        )

        fig.suptitle(f"Imputation RMSE versus association preservation: {metric}", y=1.02)
        fig.tight_layout(rect=[0, 0, 0.84, 1])

        out_png = output_dir / f"rmse_vs_association_{_safe_name(outcome)}_{_safe_name(metric)}.png"
        out_pdf = output_dir / f"rmse_vs_association_{_safe_name(outcome)}_{_safe_name(metric)}.pdf"

        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        fig.savefig(out_pdf, bbox_inches="tight")

        if pdf_pages is not None:
            pdf_pages.savefig(fig, bbox_inches="tight")

        if show:
            plt.show()

        plt.close(fig)

        outputs[f"{outcome}_{metric}_png"] = out_png
        outputs[f"{outcome}_{metric}_pdf"] = out_pdf

    return outputs


def make_association_preservation_report(
    *,
    summary_csv: Union[str, Path],
    metrics: Sequence[str] = (
        "beta_spearman",
        "z_spearman",
        "sign_concordance_top50_ref",
        "top50_overlap",
        "fdr_jaccard",
    ),
    combined_pdf_name: str = "association_preservation_report.pdf",
    show: bool = False,
) -> Dict[str, Any]:
    summary_csv = Path(summary_csv).expanduser().resolve()
    out_dir = summary_csv.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    combined_pdf_path = out_dir / combined_pdf_name

    outputs: Dict[str, Any] = {
        "combined_pdf": combined_pdf_path,
        "plots": {},
    }

    with PdfPages(combined_pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")

        lines = [
            "Protein-disease association preservation report",
            "",
            f"Summary CSV: {summary_csv}",
            "",
            "Metrics included:",
            *[f"  - {m}" for m in metrics],
            "",
            "Interpretation:",
            "Higher beta/z correlations indicate stronger preservation of protein-disease association structure.",
            "Higher top-k overlap indicates better recovery of top reference-associated proteins.",
            "Higher sign concordance indicates better preservation of association direction.",
            "FDR Jaccard summarizes overlap of FDR-significant protein sets.",
        ]

        y = 0.95

        for i, line in enumerate(lines):
            ax.text(
                0.05,
                y,
                line,
                fontsize=16 if i == 0 else 10,
                fontweight="bold" if i == 0 else "normal",
                va="top",
                ha="left",
                transform=ax.transAxes,
            )

            y -= 0.04 if i == 0 else 0.028

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        for metric in metrics:
            try:
                outputs["plots"][f"{metric}_lines"] = plot_association_preservation_metric(
                    summary_csv=summary_csv,
                    metric=metric,
                    output_dir=out_dir,
                    show=show,
                    pdf_pages=pdf,
                )
            except Exception as e:
                print(f"Skipping line plot for {metric}: {e}")

            try:
                outputs["plots"][f"{metric}_rmse_scatter"] = plot_rmse_vs_association_metric(
                    summary_csv=summary_csv,
                    metric=metric,
                    output_dir=out_dir,
                    show=show,
                    pdf_pages=pdf,
                )
            except Exception as e:
                print(f"Skipping RMSE scatter for {metric}: {e}")

    print(f"Saved association preservation PDF report: {combined_pdf_path}")

    return outputs