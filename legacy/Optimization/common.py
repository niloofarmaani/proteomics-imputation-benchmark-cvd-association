# Optimization/common.py

from __future__ import annotations

import gc
import hashlib
import importlib
import json
import math
import os
import re
import shutil
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import torch
except Exception:
    torch = None


# =============================================================================
# Project/config helpers
# =============================================================================
def _find_code_dir() -> Path:
    here = Path(__file__).resolve()

    candidates = [
        here.parents[1],
        here.parents[2] if len(here.parents) > 2 else here.parents[1],
        Path.cwd(),
        Path.cwd() / "code",
    ]

    for entry in sys.path:
        if entry:
            candidates.append(Path(entry))

    seen = set()

    for candidate in candidates:
        try:
            candidate = candidate.resolve()
        except Exception:
            continue

        if candidate in seen:
            continue

        seen.add(candidate)

        if (candidate / "config_paths.py").exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate config_paths.py. "
        "Run from the project root or ensure the project code directory is on sys.path."
    )


CODE_DIR = _find_code_dir()

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import config_paths  # noqa: E402


def ensure_runtime_paths() -> None:
    if str(CODE_DIR) not in sys.path:
        sys.path.insert(0, str(CODE_DIR))

    if not hasattr(config_paths, "IMPUTATION_DIR"):
        raise AttributeError("config_paths does not define IMPUTATION_DIR.")

    if not Path(config_paths.IMPUTATION_DIR).exists():
        raise FileNotFoundError(
            f"config_paths.IMPUTATION_DIR does not exist:\n{config_paths.IMPUTATION_DIR}"
        )


ensure_runtime_paths()


def sanitize_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))


@contextmanager
def temporary_experiments_name(name: str):
    old_name = getattr(config_paths, "EXPERIMENTS_NAME", None)

    if old_name is None:
        raise AttributeError("config_paths.EXPERIMENTS_NAME is not set.")

    config_paths.set_experiments_name(name)

    try:
        yield
    finally:
        config_paths.set_experiments_name(old_name)


def get_current_experiments_name() -> str:
    name = str(getattr(config_paths, "EXPERIMENTS_NAME", ""))

    if not name:
        raise ValueError("Current config_paths.EXPERIMENTS_NAME is empty.")

    return name


def get_experiments_root() -> Path:
    if not hasattr(config_paths, "BIOBANK_ROOT"):
        raise AttributeError("config_paths does not define BIOBANK_ROOT.")

    root = Path(getattr(config_paths, "BIOBANK_ROOT")).resolve()

    if not root.exists():
        raise FileNotFoundError(f"BIOBANK_ROOT does not exist:\n{root}")

    return root


def get_experiment_dir(experiments_name: str, columns_num: str) -> Path:
    experiments_name = str(experiments_name)

    if not experiments_name:
        raise ValueError("experiments_name cannot be empty.")

    return get_experiments_root() / experiments_name / str(columns_num)


# =============================================================================
# Validation-mask construction
# =============================================================================
def stable_combo_seed(seed: int, fraction: float, missingness: str) -> int:
    """
    Deterministic seed for a fraction/missingness combination.

    Do not use Python's built-in hash(), because it can change across processes.
    """
    key = f"{int(seed)}|{float(fraction):.12g}|{str(missingness).upper()}"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    offset = int(digest[:8], 16) % 1_000_000
    return int(seed) + offset


def make_validation_mask(
    data: np.ndarray,
    observed_mask: np.ndarray,
    val_rate: float,
    seed: int,
    mode: str = "stratified_quantile",
    n_quantiles: int = 5,
) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    observed_mask = np.asarray(observed_mask, dtype=bool)

    if data.shape != observed_mask.shape:
        raise ValueError(
            f"data and observed_mask must have same shape. "
            f"data={data.shape}, observed_mask={observed_mask.shape}"
        )

    if not (0.0 < float(val_rate) < 1.0):
        raise ValueError(f"val_rate must be between 0 and 1. Got {val_rate}")

    rng = np.random.default_rng(int(seed))
    val_mask = np.zeros_like(observed_mask, dtype=bool)

    if mode == "random":
        obs_indices = np.argwhere(observed_mask)

        if len(obs_indices) == 0:
            raise ValueError("No observed entries available for validation-mask construction.")

        rng.shuffle(obs_indices)
        n_val = int(np.ceil(len(obs_indices) * float(val_rate)))

        if n_val <= 0:
            raise ValueError("Validation mask would contain zero cells.")

        coords = obs_indices[:n_val]
        val_mask[coords[:, 0], coords[:, 1]] = True

        return val_mask

    if mode == "stratified_quantile":
        n_cols = data.shape[1]

        for j in range(n_cols):
            rows = np.where(observed_mask[:, j] & np.isfinite(data[:, j]))[0]

            if len(rows) == 0:
                continue

            values = data[rows, j]

            try:
                q = min(int(n_quantiles), len(np.unique(values)))

                if q <= 1:
                    labels = np.zeros_like(values, dtype=int)
                else:
                    labels = pd.qcut(values, q=q, labels=False, duplicates="drop")
                    labels = np.asarray(labels)
            except Exception:
                labels = np.zeros_like(values, dtype=int)

            for lab in np.unique(labels):
                lab_rows = rows[labels == lab]

                if len(lab_rows) == 0:
                    continue

                n_val_q = int(np.ceil(len(lab_rows) * float(val_rate)))
                n_val_q = min(n_val_q, len(lab_rows))

                if n_val_q <= 0:
                    continue

                chosen = rng.choice(lab_rows, size=n_val_q, replace=False)
                val_mask[chosen, j] = True

        if val_mask.sum() == 0:
            raise ValueError("Validation mask has zero cells.")

        return val_mask

    raise ValueError("mode must be 'random' or 'stratified_quantile'.")


def rmse_on_mask(original: np.ndarray, imputed: np.ndarray, mask: np.ndarray) -> float:
    original = np.asarray(original, dtype=float)
    imputed = np.asarray(imputed, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    if original.shape != imputed.shape or original.shape != mask.shape:
        raise ValueError(
            f"original, imputed, and mask must have same shape. "
            f"original={original.shape}, imputed={imputed.shape}, mask={mask.shape}"
        )

    valid = mask & np.isfinite(original) & np.isfinite(imputed)

    if valid.sum() == 0:
        raise ValueError("No finite cells available for RMSE calculation on validation mask.")

    return float(np.sqrt(np.mean((original[valid] - imputed[valid]) ** 2)))


@dataclass
class PreparedTrialData:
    tmp_experiments_name: str
    tmp_columns_dir: Path
    validation_masks: Dict[Tuple[float, str], np.ndarray]


# =============================================================================
# Temporary validation experiment creation
# =============================================================================
def prepare_validation_experiment(
    *,
    columns_num: str,
    fractions: Sequence[float],
    missingness_types: Sequence[str],
    base_experiments_name: str,
    tmp_experiments_name: str,
    val_rate: float,
    validation_mode: str,
    seed: int,
) -> PreparedTrialData:
    base_columns_dir = get_experiment_dir(base_experiments_name, columns_num)
    tmp_columns_dir = get_experiment_dir(tmp_experiments_name, columns_num)

    if not base_columns_dir.exists():
        raise FileNotFoundError(
            f"Base experiment directory does not exist:\n{base_columns_dir}"
        )

    if tmp_columns_dir.exists():
        shutil.rmtree(tmp_columns_dir)

    tmp_columns_dir.mkdir(parents=True, exist_ok=True)

    validation_masks: Dict[Tuple[float, str], np.ndarray] = {}

    for fraction in fractions:
        fraction = float(fraction)

        for missingness in missingness_types:
            missingness = str(missingness).upper()

            sub = f"fraction_{fraction}_missingness_{missingness}"
            base_dir = base_columns_dir / sub
            tmp_dir = tmp_columns_dir / sub
            tmp_dir.mkdir(parents=True, exist_ok=True)

            if not base_dir.exists():
                raise FileNotFoundError(f"Base fraction directory does not exist:\n{base_dir}")

            corr_name = (
                f"Corrupted_data_fraction_{fraction}_missingness_{missingness}_col_{columns_num}_seed_{int(seed)}.csv"
            )

            norm_name = f"normalized_data_col_{columns_num}.csv"

            corr_path = base_dir / corr_name
            norm_path = base_dir / norm_name

            if not corr_path.exists():
                raise FileNotFoundError(
                    f"Missing required seeded corrupted file:\n{corr_path}\n"
                    "Run preprocessing for this exact fraction, missingness type, and seed."
                )

            if not norm_path.exists():
                raise FileNotFoundError(f"Missing normalized/reference file:\n{norm_path}")

            df_corr = pd.read_csv(corr_path)
            df_norm = pd.read_csv(norm_path)

            if df_corr.shape != df_norm.shape:
                raise ValueError(
                    f"Corrupted and normalized files have different shapes for {sub}.\n"
                    f"Corrupted: {df_corr.shape}\n"
                    f"Normalized: {df_norm.shape}"
                )

            if list(df_corr.columns) != list(df_norm.columns):
                raise ValueError(f"Corrupted and normalized columns differ for {sub}.")

            if list(df_corr.columns[:2]) != ["eid", "label"]:
                raise ValueError(
                    f"First two columns must be ['eid', 'label'] for {corr_path}."
                )

            if not df_corr["eid"].astype(str).equals(df_norm["eid"].astype(str)):
                raise ValueError(f"eid mismatch between corrupted and normalized files for {sub}.")

            if not df_corr["label"].astype(str).equals(df_norm["label"].astype(str)):
                raise ValueError(f"label mismatch between corrupted and normalized files for {sub}.")

            numeric = df_corr.iloc[:, 2:].to_numpy(dtype=float)
            observed_mask = np.isfinite(numeric)

            combo_seed = stable_combo_seed(seed, fraction, missingness)

            val_mask = make_validation_mask(
                data=numeric,
                observed_mask=observed_mask,
                val_rate=val_rate,
                seed=combo_seed,
                mode=validation_mode,
            )

            if val_mask.sum() == 0:
                raise ValueError(f"Validation mask has zero cells for {sub}.")

            numeric_val = numeric.copy()
            numeric_val[val_mask] = np.nan

            df_val = df_corr.copy()
            df_val.iloc[:, 2:] = numeric_val

            df_val.to_csv(tmp_dir / corr_name, index=False)
            df_norm.to_csv(tmp_dir / norm_name, index=False)

            val_mask_df = pd.DataFrame(val_mask, columns=df_corr.columns[2:])
            val_mask_df.to_csv(tmp_dir / f"missing_mask_col_{columns_num}.csv", index=False)
            val_mask_df.to_csv(tmp_dir / f"optuna_validation_mask_col_{columns_num}.csv", index=False)

            validation_masks[(fraction, missingness)] = val_mask

    return PreparedTrialData(
        tmp_experiments_name=tmp_experiments_name,
        tmp_columns_dir=tmp_columns_dir,
        validation_masks=validation_masks,
    )


# =============================================================================
# Imputed file resolution and scoring
# =============================================================================
def find_imputed_csv(folder: Path) -> Path:
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist:\n{folder}")

    candidates = []

    for path in folder.glob("*.csv"):
        low = path.name.lower()

        if low.startswith("imputed") and not any(x in low for x in ["rmse", "metric", "summary", "mask"]):
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(f"No imputed*.csv file found in {folder}")

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    return candidates[0]


def score_trial_outputs(
    *,
    prepared: PreparedTrialData,
    columns_num: str,
    fractions: Sequence[float],
    missingness_types: Sequence[str],
) -> Tuple[float, List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    scores: List[float] = []

    for fraction in fractions:
        fraction = float(fraction)

        for missingness in missingness_types:
            missingness = str(missingness).upper()

            sub = f"fraction_{fraction}_missingness_{missingness}"
            folder = prepared.tmp_columns_dir / sub

            imputed_path = find_imputed_csv(folder)
            norm_path = folder / f"normalized_data_col_{columns_num}.csv"

            if not norm_path.exists():
                raise FileNotFoundError(f"Missing normalized/reference file:\n{norm_path}")

            imputed_df = pd.read_csv(imputed_path)
            original_df = pd.read_csv(norm_path)

            if imputed_df.shape != original_df.shape:
                raise ValueError(
                    f"Imputed and reference files have different shapes for {sub}.\n"
                    f"Imputed: {imputed_df.shape}\n"
                    f"Reference: {original_df.shape}"
                )

            if list(imputed_df.columns) != list(original_df.columns):
                raise ValueError(f"Imputed and reference columns differ for {sub}.")

            imputed = imputed_df.iloc[:, 2:].to_numpy(dtype=float)
            original = original_df.iloc[:, 2:].to_numpy(dtype=float)

            key = (fraction, missingness)

            if key not in prepared.validation_masks:
                raise KeyError(f"Validation mask not found for key={key}")

            mask = prepared.validation_masks[key]

            score = rmse_on_mask(original, imputed, mask)

            scores.append(score)

            rows.append(
                {
                    "fraction": fraction,
                    "missingness": missingness,
                    "validation_rmse": float(score),
                    "n_validation_cells": int(mask.sum()),
                    "imputed_path": str(imputed_path),
                }
            )

    if len(scores) == 0:
        raise RuntimeError("No validation scores were computed.")

    if not np.isfinite(scores).all():
        raise FloatingPointError(f"Non-finite validation scores found: {scores}")

    return float(np.mean(scores)), rows


# =============================================================================
# Dynamic import and method calling
# =============================================================================
def import_method_module(module_name: str):
    ensure_runtime_paths()

    errors = []

    for pkg in ("imputation", "Imputation"):
        try:
            return importlib.import_module(f"{pkg}.{module_name}")
        except Exception as e:
            errors.append(f"{pkg}.{module_name}: {e}")

    try:
        return importlib.import_module(module_name)
    except Exception as e:
        errors.append(f"{module_name}: {e}")

    raise ImportError("Could not import method module. Tried:\n" + "\n".join(errors))


def call_existing_method(
    *,
    method: str,
    params: Dict[str, Any],
    file_path: str,
    columns_num: str,
    fractions: Sequence[float],
    missingness_types: Sequence[str],
    epochs: int,
    patience: int,
    warmup_epochs: int,
    seed: int,
    verbose: bool = False,
) -> None:
    m = method.upper()



    if m == "AE":
        mod = import_method_module("AE")
        mod.ae(
            file_path,
            columns_num,
            list(fractions),
            list(missingness_types),
            params["batch_size"],
            params["learning_rate"],
            params["n_neurons"],
            params["dim_latent"],
            params.get("freebits", 0.1),
            epochs,
            patience,
            warmup_epochs,
            seed,
        )
        return

    if m == "DAE":
        mod = import_method_module("DAE")
        mod.dae(
            file_path,
            columns_num,
            list(fractions),
            list(missingness_types),
            params["batch_size"],
            params["learning_rate"],
            params["n_neurons"],
            params["dim_latent"],
            params.get("freebits", 0.1),
            params["noise_std"],
            params["feature_dropout"],
            epochs,
            patience,
            warmup_epochs,
            seed,
        )
        return

    if m == "VAE":
        mod = import_method_module("VAE")
        mod.vae(
            file_path,
            columns_num,
            list(fractions),
            list(missingness_types),
            params["batch_size"],
            params["learning_rate"],
            params["n_neurons"],
            params["dim_latent"],
            params["freebits"],
            epochs,
            patience,
            warmup_epochs,
            seed,
        )
        return

    if m == "GAIN":
        mod = import_method_module("GAIN")
        mod.gain(
            columns_num=columns_num,
            fractions=list(fractions),
            missingness_types=list(missingness_types),
            n_neurons=params["n_neurons"],
            batch_size=params["batch_size"],
            lr=params["lr"],
            alpha=params["alpha"],
            hint_rate=params["hint_rate"],
            epochs=epochs,
            patience=patience,
            seed=seed,
            val_rate=params["val_rate"],
            validation_mode=params["validation_mode"],
            refit_final=params["refit_final"],
            mc_samples_validation=params["mc_samples_validation"],
            mc_samples_final=params["mc_samples_final"],
            output_activation=params["output_activation"],
            noise_type=params["noise_type"],
            noise_scale=params["noise_scale"],
            dropout_rate=params["dropout_rate"],
            weight_decay=params["weight_decay"],
            grad_clip=params["grad_clip"],
            pseudo_rate=params["pseudo_rate"],
            pseudo_beta=params["pseudo_beta"],
            hint_mode=params["hint_mode"],
            verbose=verbose,
        )
        return

    if m == "GAN":
        mod = import_method_module("GAN")
        mod.gan(
            file_path,
            columns_num,
            list(fractions),
            list(missingness_types),
            params["batch_size"],
            params["learning_rate"],
            params["freebits"],
            params["gen_hidden"],
            params["disc_hidden"],
            params["hint_rate"],
            params["alpha"],
            epochs,
            patience,
            warmup_epochs,
            seed,
        )
        return

    if m == "RNN":
        mod = import_method_module("RNN")
        mod.rnn(
            file_path,
            columns_num,
            list(fractions),
            list(missingness_types),
            params["batch_size"],
            params["learning_rate"],
            params["hidden_size"],
            params["num_layers"],
            params["dropout"],
            params["pseudo_mask_rate"],
            epochs,
            patience,
            seed,
        )
        return

    if m in ("TRANSFORMER", "TRF"):
        mod = import_method_module("Transformer")
        mod.transformer(
            file_path=file_path,
            columns_num=columns_num,
            fractions=list(fractions),
            missingness_types=list(missingness_types),
            batch_size=params["batch_size"],
            learning_rate=params["learning_rate"],
            d_model=params["d_model"],
            nhead=params["nhead"],
            num_layers=params["num_layers"],
            dim_feedforward=params["dim_feedforward"],
            dropout=params["dropout"],
            pseudo_mask_rate=params["pseudo_mask_rate"],
            weight_decay=params["weight_decay"],
            grad_clip=params["grad_clip"],
            internal_val_rate=params["internal_val_rate"],
            epochs=epochs,
            patience=patience,
            seed=seed,
        )
        return

    if m == "KNN":
        mod = import_method_module("knn")
        mod.knn(
            file_path,
            columns_num,
            list(fractions),
            list(missingness_types),
            [params["k"]],
            seed=seed,
        )
        return

    if m in ("MISSFOREST", "MF"):
        mod = import_method_module("missforest")
        mod.missforest(
            file_path,
            columns_num,
            list(fractions),
            list(missingness_types),
            params["niter"],
            [params["ntree"]],
            params["max_depth"],
            params["sample_fraction"],
            seed=seed,
            show_r_output=verbose,
            print_command=verbose,
            print_rmse_after_run=True,
        )
        return

    if m in ("SOFTIMPUTE", "SOFTIMPUTE_PY"):
        mod = import_method_module("Softimpute")
        mod.softImpute(
            file_path,
            columns_num,
            list(fractions),
            list(missingness_types),
            [params["rank_max"]],
            params["maxit"],
            seed=seed,
            show_output=verbose,
            print_command=verbose,
            print_rmse_after_run=True,
        )
        return

    if m in ("ITERATIVE", "ITERATIVE_IMPUTER"):
        mod = import_method_module("iterative")
        mod.ITERATIVE_IMPUTER(
            file_path,
            columns_num,
            list(fractions),
            list(missingness_types),
            params["max_iter"],
            params["tol"],
            params["n_nearest_features"],
            params["imputation_order"],
            params["initial_strategy"],
            params["estimator_name"],
            params["sample_posterior"],
            params["skip_complete"],
            seed=seed,
            show_output=verbose,
            print_command=verbose,
            print_rmse_after_run=True,
        )
        return

    if m == "MICE":
        mod = import_method_module("MICE")
        mod.MICE(
            file_path,
            columns_num,
            list(fractions),
            list(missingness_types),
            params["maxit"],
            [params["m"]],
        )
        return

    if m == "BPCA":
        mod = import_method_module("BPCA")
        mod.BPCA(
            file_path,
            columns_num,
            list(fractions),
            list(missingness_types),
            [params["nPcs"]],
            params["maxSteps"],
        )
        return

    raise ValueError(f"Unsupported method: {method}")


# =============================================================================
# Cleanup and JSON helpers
# =============================================================================
def cleanup_torch() -> None:
    gc.collect()

    if torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)