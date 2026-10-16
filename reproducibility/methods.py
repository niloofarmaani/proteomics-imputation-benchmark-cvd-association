"""Adapters to the supplied implementations, without passing hidden truth to fit."""
from __future__ import annotations
import contextlib
import copy
import importlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "legacy"
if str(LEGACY) not in sys.path:
    sys.path.insert(0, str(LEGACY))

METHODS = ["KNN", "Iterative", "MissForest", "SoftImpute", "AE", "DAE", "VAE", "GAIN", "Transformer"]
PAPER_TRIALS = dict(zip(METHODS, [20, 12, 20, 15, 20, 20, 20, 15, 6]))


def preflight(methods):
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")
    import optuna  # noqa: F401
    import torch
    torch.set_num_threads(2)
    if {"KNN", "MissForest"} & set(methods):
        if not shutil.which("Rscript"):
            raise RuntimeError("Rscript missing. Run scripts/install_colab.sh before benchmarking all nine methods.")
        packages = (["Metrics", "impute"] if "KNN" in methods else []) + (["Metrics", "missRanger"] if "MissForest" in methods else [])
        script = "stopifnot(" + ",".join(f'requireNamespace("{p}", quietly=TRUE)' for p in sorted(set(packages))) + ")"
        subprocess.run(["Rscript", "-e", script], check=True)
    return {"torch": torch.__version__, "device": "cuda" if torch.cuda.is_available() else "cpu", "methods": methods}


def suggest(method, trial, n_features):
    from Optimization.method_spaces import suggest_params
    if method == "SoftImpute":
        # Original rank candidates can exceed the demonstration's feature count.
        ranks = [r for r in [5, 10, 20, 40, 80] if r < n_features]
        if not ranks:
            ranks = [max(1, n_features - 1)]
        return {"rank_max": trial.suggest_categorical("rank_max", ranks),
                "maxit": trial.suggest_categorical("maxit", [50, 100, 200])}
    return suggest_params(method, trial)


def impute(incomplete, method, params, *, seed, epochs, patience, warmup_epochs, log_path):
    """No reference or phenotype argument: only observed protein data reach models.

    Legacy APIs require eid/label and a complete reference CSV. Supply a constant
    dummy label and observed-column-mean surrogate. Legacy RMSE log values refer
    to that surrogate and are not benchmark scores. True scoring is external.
    """
    from Optimization.common import call_existing_method
    import config_paths
    a = incomplete.to_numpy(dtype=float)
    if np.isinf(a).any() or np.isnan(a).all(axis=0).any():
        raise ValueError("Infinite or fully missing protein inputs.")
    frame = incomplete.reset_index(drop=True).copy()
    frame.insert(0, "label", 0)
    frame.insert(0, "eid", incomplete.index.astype(str))
    surrogate = frame.copy()
    surrogate.iloc[:, 2:] = incomplete.fillna(incomplete.mean()).to_numpy()
    previous = config_paths.BIOBANK_ROOT, config_paths.EXPERIMENTS_NAME
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="imputation-") as tmp:
            config_paths.set_biobank_root(tmp)
            config_paths.set_experiments_name("active")
            folder = config_paths.fraction_dir("all", 0.1, "MCAR")
            folder.mkdir(parents=True)
            frame.to_csv(folder / f"Corrupted_data_fraction_0.1_missingness_MCAR_col_all_seed_{seed}.csv", index=False)
            surrogate.to_csv(folder / "normalized_data_col_all.csv", index=False)
            with log_path.open("w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                log.write("Legacy diagnostic RMSE uses a surrogate, not hidden ground truth. See external metrics.csv.\n")
                if method == "KNN":
                    # Preserve algorithm while fixing R name mangling and its unpassed seed.
                    source = (LEGACY / "imputation/knn_imputer.R").read_text()
                    source = source.replace("read.csv(corrupted_data_path)", "read.csv(corrupted_data_path, check.names=FALSE)")
                    source = source.replace("read.csv(normalized_data_path)", "read.csv(normalized_data_path, check.names=FALSE)")
                    source = source.replace("k = kn,", f"k = kn, rng.seed = {int(seed)},")
                    source = "set.seed(" + str(int(seed)) + ")\n" + source
                    rfile = Path(tmp) / "knn_seeded.R"
                    rfile.write_text(source)
                    subprocess.run(["Rscript", str(rfile), str(folder / f"Corrupted_data_fraction_0.1_missingness_MCAR_col_all_seed_{seed}.csv"), str(folder / "normalized_data_col_all.csv"), str(params["k"]), str(folder / "diagnostic_rmse.csv"), str(folder / "imputed_KNN.csv")], stdout=log, stderr=log, check=True)
                elif method == "MissForest":
                    source = (LEGACY / "imputation/missforest_imputer.R").read_text()
                    # as.data.frame(lapply(...)) otherwise sanitizes antibody names.
                    source = source.replace("  missing_indices <-", "  names(df_with_na_numeric) <- names(df_with_na)[-c(1,2)]\n  names(original_df_numeric) <- names(original_df)[-c(1,2)]\n  missing_indices <-")
                    rfile = Path(tmp) / "missforest_names.R"
                    rfile.write_text(source)
                    subprocess.run(["Rscript", str(rfile), str(folder / f"Corrupted_data_fraction_0.1_missingness_MCAR_col_all_seed_{seed}.csv"), str(folder / "normalized_data_col_all.csv"), str(params["niter"]), str(params["ntree"]), str(params["max_depth"]), str(params["sample_fraction"]), str(seed), str(folder / "diagnostic_rmse.csv"), str(folder / "imputed_MissForest.csv")], stdout=log, stderr=log, check=True)
                else:
                    call_existing_method(method=method, params=copy.deepcopy(params), file_path=str(folder / "normalized_data_col_all.csv"), columns_num="all", fractions=[0.1], missingness_types=["MCAR"], epochs=epochs, patience=patience, warmup_epochs=warmup_epochs, seed=seed, verbose=False)
            outputs = list(folder.glob("imputed*.csv"))
            if len(outputs) != 1:
                raise RuntimeError(f"Expected exactly one completed matrix, found {outputs}. See {log_path}.")
            result = pd.read_csv(outputs[0])
            if result.columns.tolist() != frame.columns.tolist() or not result.eid.astype(str).equals(frame.eid.astype(str)):
                raise ValueError(f"Method {method} changed sample order or protein identifiers.")
            b = result.iloc[:, 2:].to_numpy(float)
            if b.shape != a.shape or not np.isfinite(b).all():
                raise ValueError(f"Method {method} returned an invalid completion.")
            observed = np.isfinite(a)
            if not np.allclose(a[observed], b[observed], rtol=1e-6, atol=1e-7):
                raise ValueError(f"Method {method} changed observed values.")
            b[observed] = a[observed]  # restore CSV round-trip precision exactly
            return pd.DataFrame(b, index=incomplete.index, columns=incomplete.columns)
    finally:
        config_paths.set_biobank_root(previous[0])
        config_paths.set_experiments_name(previous[1])
