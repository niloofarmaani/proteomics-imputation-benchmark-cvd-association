"""Separate tuning, frozen reconstruction, and downstream association analysis."""
from __future__ import annotations
import dataclasses
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
from .missingness import corrupt, validation_mask, combo_seed, masked_rmse
from .methods import METHODS, PAPER_TRIALS, preflight, suggest, impute
from .associations import covariate_design, scan, preservation


@dataclasses.dataclass
class Settings:
    mode: str = "demo"
    methods: tuple = tuple(METHODS)
    fractions: tuple = (0.1, 0.3, 0.5)
    mechanisms: tuple = ("MCAR", "MAR", "MNAR")
    tuning_fractions: tuple = (0.1, 0.3)
    tuning_mechanisms: tuple = ("MAR", "MNAR")
    final_seeds: tuple = (42,)
    tuning_seed: int = 42
    association_seed: int = 42
    trials: int = 2
    epochs: int = 30
    patience: int = 8
    warmup_epochs: int = 10

    @classmethod
    def paper_budget(cls):
        return cls(mode="paper_budget", final_seeds=(42, 100, 2024, 777, 999), trials=20, epochs=300, patience=50, warmup_epochs=100)


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=lambda v: int(v) if isinstance(v, np.integer) else str(v)))


def fingerprint(x, settings):
    return hashlib.sha256(x.to_csv().encode() + json.dumps(dataclasses.asdict(settings), sort_keys=True).encode()).hexdigest()


def initialize_run(out, x, metadata, settings, outcome, covariates):
    preflight(list(settings.methods))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run.json").exists():
        raise FileExistsError(f"{out} already contains a run. Choose a new output directory to avoid mixing results.")
    if not x.index.equals(metadata.index) or x.index.has_duplicates or x.columns.has_duplicates:
        raise ValueError("Unique sample/protein identifiers and aligned metadata required.")
    if not np.isfinite(x.to_numpy(float)).all():
        raise ValueError("Complete reference protein measurements required.")
    if settings.association_seed not in settings.final_seeds:
        raise ValueError("association_seed must be among final_seeds.")
    if max(settings.fractions) >= 1 or not settings.methods:
        raise ValueError("Invalid benchmark settings.")
    payload = {"settings": dataclasses.asdict(settings), "input_sha256": fingerprint(x, settings),
               "phenotype_sha256": hashlib.sha256(metadata.to_csv().encode()).hexdigest(),
               "outcome": outcome, "covariates": list(covariates), "python": sys.version,
               "platform": platform.platform(), "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
               "versions": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "torch", "optuna", "statsmodels"]}}
    root = Path(__file__).resolve().parents[1]
    payload["source_sha256"] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for directory in [root / "reproducibility", root / "legacy"]
                                for p in directory.rglob("*") if p.suffix in {".py", ".R"}}
    dump(out / "run.json", payload)
    (out / "environment.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
    if __import__("shutil").which("Rscript"):
        with (out / "R-session.txt").open("w") as log:
            subprocess.run(["Rscript", "-e", 'sessionInfo(); print(installed.packages()[,c("Package","Version")])'], stdout=log, stderr=log, check=True)
    x.to_csv(out / "reference_proteins.csv")
    metadata.to_csv(out / "phenotypes.csv")
    return out


def tune(x, out, settings):
    """The objective only sees additional observed-cell holdouts, never test masks."""
    out = Path(out)
    if (out / "frozen_parameters.json").exists():
        raise FileExistsError("Frozen parameters already exist. Load them for evaluation or start a new run.")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    panel, trials_rows, condition_rows = [], [], []
    masks_dir = out / "masks"
    masks_dir.mkdir(exist_ok=True)
    for mechanism in settings.tuning_mechanisms:
        for fraction in settings.tuning_fractions:
            incomplete, test_mask = corrupt(x, mechanism, fraction, settings.tuning_seed)
            holdout = validation_mask(incomplete, combo_seed(settings.tuning_seed, mechanism, fraction))
            assert not (holdout & test_mask).any()
            training = incomplete.mask(holdout)
            tag = f"tune_{mechanism}_{fraction}_seed{settings.tuning_seed}"
            np.savez_compressed(masks_dir / f"{tag}.npz", simulated=test_mask, validation=holdout)
            panel.append((mechanism, fraction, training, holdout))
    frozen = {}
    for method in settings.methods:
        print(f"Tuning {method}", flush=True)
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=settings.tuning_seed),
                                    study_name=method, storage=f"sqlite:///{(out / 'optuna.sqlite3').resolve()}")
        def objective(trial):
            params = suggest(method, trial, min(x.shape))
            trial.set_user_attr("method_params", params)
            scores = []
            for mechanism, fraction, training, holdout in panel:
                started = time.perf_counter()
                completed = impute(training, method, params, seed=settings.tuning_seed,
                                   epochs=settings.epochs, patience=settings.patience, warmup_epochs=settings.warmup_epochs,
                                   log_path=out / "logs" / f"tune_{method}_{trial.number}_{mechanism}_{fraction}.log")
                score = masked_rmse(x, completed, holdout)
                scores.append(score)
                condition_rows.append(dict(method=method, trial=trial.number, mechanism=mechanism, fraction=fraction,
                                           validation_rmse=score, scored_cells=int(holdout.sum()), seconds=time.perf_counter() - started))
                pd.DataFrame(condition_rows).to_csv(out / "validation_conditions.csv", index=False)
            return float(np.mean(scores))
        n_trials = PAPER_TRIALS[method] if settings.mode == "paper_budget" else settings.trials
        # Fail visibly instead of silently ranking a subset of requested methods.
        study.optimize(objective, n_trials=n_trials, n_jobs=1)
        frozen[method] = {"params": study.best_trial.user_attrs["method_params"], "validation_rmse": study.best_value,
                          "best_trial": study.best_trial.number, "completed_trials": n_trials}
        for trial in study.trials:
            trials_rows.append(dict(method=method, trial=trial.number, state=trial.state.name, objective=trial.value,
                                    params=json.dumps(trial.user_attrs.get("method_params", {}))))
        pd.DataFrame(trials_rows).to_csv(out / "validation_trials.csv", index=False)
        dump(out / "tuning_progress.json", frozen)
    dump(out / "frozen_parameters.json", frozen)
    return frozen


def evaluate(x, out, settings, frozen):
    out = Path(out)
    if set(frozen) != set(settings.methods):
        raise ValueError("Frozen parameters do not cover the exact requested method set.")
    results = []
    for seed in settings.final_seeds:
        for mechanism in settings.mechanisms:
            for fraction in settings.fractions:
                incomplete, mask = corrupt(x, mechanism, fraction, seed)
                tag = f"{mechanism}_{fraction}_seed{seed}"
                (out / "masks").mkdir(exist_ok=True)
                np.savez_compressed(out / "masks" / f"final_{tag}.npz", simulated=mask)
                for method in settings.methods:
                    print(f"Final {method}: {tag}", flush=True)
                    path = out / "imputed" / f"{method}_{tag}.csv"
                    path.parent.mkdir(exist_ok=True)
                    started = time.perf_counter()
                    completed = impute(incomplete, method, frozen[method]["params"], seed=seed,
                                       epochs=settings.epochs, patience=settings.patience, warmup_epochs=settings.warmup_epochs,
                                       log_path=out / "logs" / f"final_{method}_{tag}.log")
                    completed.to_csv(path)
                    results.append(dict(method=method, mechanism=mechanism, fraction=fraction, seed=seed,
                                        rmse=masked_rmse(x, completed, mask), scored_cells=int(mask.sum()),
                                        seconds=time.perf_counter() - started, imputed_path=str(path.relative_to(out)),
                                        imputed_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                    pd.DataFrame(results).to_csv(out / "reconstruction.csv", index=False)
    expected = len(settings.final_seeds) * len(settings.mechanisms) * len(settings.fractions) * len(settings.methods)
    if len(results) != expected:
        raise RuntimeError("Incomplete reconstruction grid.")
    return pd.DataFrame(results)


def association_analysis(x, metadata, outcome, covariates, out, settings, reconstruction):
    out = Path(out)
    directory = out / "associations"
    directory.mkdir(exist_ok=True)
    design = covariate_design(metadata, covariates)
    design.to_csv(directory / "covariate_design.csv")
    reference = scan(x, metadata, outcome, design)
    reference.to_csv(directory / "reference.csv")
    rows = []
    selected = reconstruction[reconstruction.seed == settings.association_seed]
    for record in selected.to_dict("records"):
        path = Path(record["imputed_path"])
        if not path.is_absolute():
            path = out / path
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["imputed_sha256"]:
            raise ValueError("Completed matrix changed since reconstruction scoring.")
        completed = pd.read_csv(path, index_col=0)
        if not completed.index.equals(x.index) or not completed.columns.equals(x.columns):
            raise ValueError("Association matrix identifiers changed.")
        associations = scan(completed, metadata, outcome, design)
        associations.to_csv(directory / path.name)
        rows.append({k: record[k] for k in ["method", "mechanism", "fraction", "seed", "rmse"]} | preservation(reference, associations))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "association_preservation.csv", index=False)
    return reference, metrics
