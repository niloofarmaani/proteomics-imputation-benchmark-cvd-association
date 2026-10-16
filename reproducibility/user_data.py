"""Upload/export helpers around the unchanged tuning and evaluation workflow.

No imputation algorithm, search space, missingness generator, tuning objective,
or final benchmark step is replaced here.
"""
from __future__ import annotations

import csv
import dataclasses
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from .methods import METHODS, impute, preflight
from .pipeline import dump, fingerprint

ROOT = Path(__file__).resolve().parents[1]
MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none"}
ID_NAMES = {"eid", "sample_id", "patient_id", "id", "unnamed: 0"}


@dataclasses.dataclass
class UploadedData:
    path: Path
    original: pd.DataFrame
    matrix: pd.DataFrame
    feature_columns: list
    id_column: str | None
    metadata_columns: list


def load_uploaded_data(path, *, id_column="auto", metadata_columns=(), separator=None):
    """Read samples x numeric variables, preserving metadata and original strings.

    Internal row/variable identifiers prevent legacy CSV/R handling from changing
    leading-zero IDs or arbitrary user headers. Only identifiers are mapped;
    numeric measurements and their order are unchanged.
    """
    path = Path(path).expanduser().resolve()
    separator = separator or ("\t" if path.suffix.lower() in {".tsv", ".txt"} else ",")
    if len(separator) != 1:
        raise ValueError("Use a single-character CSV/TSV separator.")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream, delimiter=separator), [])
    if not header or any(not h.strip() for h in header) or len(set(header)) != len(header):
        raise ValueError("Every column must have a unique, nonempty header.")
    frame = pd.read_csv(path, sep=separator, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if frame.empty:
        raise ValueError("The uploaded dataset has no sample rows.")
    if id_column == "auto":
        candidates = [c for c in frame if c.strip().lower() in ID_NAMES]
        if len(candidates) > 1:
            raise ValueError(f"Multiple possible ID columns {candidates}. Set ID_COLUMN explicitly.")
        id_column = candidates[0] if candidates else None
    metadata_columns = list(metadata_columns)
    requested = metadata_columns + ([id_column] if id_column is not None else [])
    if len(set(requested)) != len(requested) or set(requested) - set(frame.columns):
        raise ValueError("ID/metadata columns must exist and must not be repeated.")
    if id_column is not None:
        ids = frame[id_column].str.strip()
        if ids.eq("").any() or ids.duplicated().any():
            raise ValueError("Sample IDs must be nonempty and unique; their original formatting is preserved.")
    features = [c for c in frame if c not in requested]
    if len(features) < 2:
        raise ValueError("At least two numeric variable columns are required by the benchmark.")
    values = {}
    for j, name in enumerate(features):
        source = frame[name].str.strip()
        missing = source.str.lower().isin(MISSING_TOKENS)
        numeric = pd.to_numeric(source.mask(missing), errors="coerce")
        if (numeric.isna() & ~missing).any():
            raise ValueError(f"Column {name!r} contains nonnumeric values. Put metadata in METADATA_COLUMNS.")
        if np.isinf(numeric.to_numpy(float)).any():
            raise ValueError(f"Column {name!r} contains infinite measurements.")
        if numeric.notna().sum() == 0:
            raise ValueError(f"Column {name!r} has no observed measurements; the original methods require observed values.")
        values[f"variable_{j:05d}"] = numeric.to_numpy(float)
    matrix = pd.DataFrame(values, index=[f"sample_{i:08d}" for i in range(len(frame))])
    matrix.index.name = "eid"
    return UploadedData(path, frame, matrix, features, id_column, metadata_columns)


def reference_for_benchmark(data, reference=None, *, min_rows=50):
    """Obtain true complete measurements without filling unknown reference values."""
    if reference is None:
        complete = data.matrix.notna().all(axis=1)
        x = data.matrix.loc[complete].copy()
        source = "complete rows of the uploaded dataset, across every selected variable"
    else:
        if set(reference.feature_columns) != set(data.feature_columns):
            raise ValueError("Reference and uploaded data must contain exactly the same numeric variables.")
        actual_to_internal = dict(zip(reference.feature_columns, reference.matrix.columns))
        x = reference.matrix[[actual_to_internal[c] for c in data.feature_columns]].copy()
        x.columns = data.matrix.columns
        if x.isna().any().any():
            raise ValueError("The supplied reference contains missing values. Supply measured complete reference values.")
        source = "separate complete reference table; variables matched by their original names"
    if len(x) < min_rows:
        raise ValueError(
            f"Only {len(x)} complete reference rows are available; this companion requires at least {min_rows}. "
            "Upload a separate complete reference CSV with the same variables and units. "
            "Unknown values cannot be used as benchmark truth. No rows/variables were silently filled or dropped."
        )
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError("The reference must contain only finite measured values.")
    return x, {"reference_source": source, "reference_samples": len(x),
               "input_samples": len(data.matrix), "variables": len(data.feature_columns),
               "native_missing_cells": int(data.matrix.isna().sum().sum()),
               "native_missing_fraction": float(data.matrix.isna().to_numpy().mean())}


def initialize_uploaded_run(data, reference, settings, out, audit, reference_data=None):
    """Write provenance without fabricating clinical outcomes or running scans."""
    preflight(list(settings.methods))
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    settings_dict = dataclasses.asdict(settings)
    protected = [*list((ROOT / "legacy").rglob("*.py")), *list((ROOT / "legacy").rglob("*.R")),
                 *list((ROOT / "reproducibility").glob("*.py"))]
    run = {"settings": settings_dict, "input_sha256": fingerprint(reference, settings),
           "uploaded_file_sha256": hashlib.sha256(data.path.read_bytes()).hexdigest(),
           "created_utc": pd.Timestamp.now(tz="UTC").isoformat(), "python": sys.version,
           "platform": platform.platform(), "workflow": "unchanged pipeline.tune and pipeline.evaluate; no association stage",
           "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected},
           "versions": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "torch", "optuna", "scikit-learn"]},
           "selection_rule": "lowest mean validation RMSE from frozen_parameters.json; final reconstruction scores do not select the winner"}
    if (ROOT / ".git").exists():
        run["source_commit"] = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    if reference_data is not None:
        run["reference_file_sha256"] = hashlib.sha256(reference_data.path.read_bytes()).hexdigest()
    dump(out / "run.json", run)
    dump(out / "input_audit.json", audit)
    dump(out / "column_mapping.json", {"id_column": data.id_column, "metadata_columns": data.metadata_columns,
         "variables": dict(zip(data.matrix.columns, data.feature_columns))})
    reference.to_csv(out / "reference_proteins.csv")
    data.original.to_csv(out / "original_uploaded_dataset.csv", index=False)
    np.savez_compressed(out / "original_missing_mask.npz", missing=data.matrix.isna().to_numpy())
    (out / "environment.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
    if shutil.which("Rscript"):
        with (out / "R-session.txt").open("w") as log:
            subprocess.run(["Rscript", "-e", "sessionInfo(); print(installed.packages()[,c('Package','Version')])"],
                           stdout=log, stderr=log, check=True)
    return out


def select_by_validation(frozen, methods):
    if set(frozen) != set(methods):
        raise ValueError("All requested methods must have frozen tuning results before selection.")
    table = pd.DataFrame([{"method": m, "validation_rmse": frozen[m]["validation_rmse"],
                           "best_trial": frozen[m]["best_trial"], "trials": frozen[m]["completed_trials"]} for m in methods])
    if not np.isfinite(table.validation_rmse).all() or (table.validation_rmse < 0).any():
        raise ValueError("Every method must have a finite nonnegative validation RMSE.")
    table = table.sort_values("validation_rmse", kind="stable").reset_index(drop=True)
    table.insert(0, "validation_rank", range(1, len(table) + 1))
    return str(table.iloc[0].method), table


def impute_selected_dataset(data, out, settings, frozen, selected_method):
    """Refit the validation-selected original implementation on all original rows."""
    expected, _ = select_by_validation(frozen, settings.methods)
    if selected_method != expected:
        raise ValueError("The export method must be the frozen validation-selected winner.")
    out = Path(out)
    missing = data.matrix.isna().to_numpy()
    if missing.any():
        completed = impute(data.matrix, selected_method, frozen[selected_method]["params"],
                           seed=settings.tuning_seed, epochs=settings.epochs, patience=settings.patience,
                           warmup_epochs=settings.warmup_epochs, log_path=out / "logs" / "selected_method_original_data.log")
    else:
        completed = data.matrix.copy()
    if not completed.index.equals(data.matrix.index) or not completed.columns.equals(data.matrix.columns):
        raise ValueError("The completed dataset changed row or variable identifiers.")
    values, original_values = completed.to_numpy(float), data.matrix.to_numpy(float)
    if not np.isfinite(values).all() or not np.array_equal(values[~missing], original_values[~missing]):
        raise ValueError("Final imputation must fill every missing measurement and preserve every observed measurement.")
    result = data.original.copy()
    for j, column in enumerate(data.feature_columns):
        rows = np.flatnonzero(missing[:, j])
        result.loc[rows, column] = [format(v, ".17g") for v in values[rows, j]]
    path = out / "best_imputed_dataset.csv"
    result.to_csv(path, index=False)
    dump(out / "best_method.json", {"method": selected_method,
         "selection": "lowest validation RMSE; never selected using final reconstruction scores",
         "validation_rmse": frozen[selected_method]["validation_rmse"],
         "parameters": frozen[selected_method]["params"], "refit_seed": settings.tuning_seed,
         "native_missing_cells_filled": int(missing.sum()), "original_samples": len(result),
         "variables": data.feature_columns, "refitted_on_original_data": bool(missing.any()),
         "output_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return path, result


def plot_uploaded_benchmark(reconstruction, out, settings, *, show=True):
    """Call the same original paper renderer with dataset-appropriate labels."""
    from Evaluation.paper_seed_rmse_boxplots import make_paper_seed_rmse_boxplots
    from .paper_figures import PAPER_METHOD_ORDER
    directory = Path(out) / "figures"
    directory.mkdir(exist_ok=True)
    source = directory / "uploaded_reference_seed_rmse.csv"
    reconstruction.rename(columns={"mechanism": "missingness"}).to_csv(source, index=False)
    n = len(settings.final_seeds)
    outputs, values, summary, _ = make_paper_seed_rmse_boxplots(
        columns_num="uploaded_data", methods_to_compare=[m for m in PAPER_METHOD_ORDER if m in settings.methods],
        fractions=settings.fractions, missingness_types=settings.mechanisms, seeds=settings.final_seeds,
        seed_values_csv=source, output_dir=directory, strict=True, y_range=None, sort_by="input",
        make_bayes_tables=False, make_individual_panels=False, make_mechanism_panels=False,
        print_latex_tables=False, print_markdown_tables=False, save_png=True, show=show,
        figure_height_mm=190,
        figure_title=f"Uploaded-data complete reference\nFrozen-parameter reconstruction | {n} simulation seed(s)",
        figure_note=("RMSE in supplied data units. One seed: collapsed boxes; no replicate variability estimate."
                     if n == 1 else "RMSE in supplied data units. Boxes: IQR; line: median; circles: individual seeds."))
    return outputs, values, summary


def write_worked_example(path):
    """An explicitly synthetic execution example, not a scientific benchmark claim."""
    rng = np.random.default_rng(20261016)
    latent = rng.normal(size=(256, 4))
    values = latent @ rng.normal(size=(4, 60)) + rng.normal(scale=.25, size=(256, 60))
    values = (values - values.mean(axis=0)) / values.std(axis=0)
    # Match the existing demo's reference size. Every later row is incomplete.
    mask = rng.random(size=values[160:].shape) < .18
    empty_rows = np.flatnonzero(~mask.any(axis=1))
    mask[empty_rows, 0] = True
    values[160:][mask] = np.nan
    frame = pd.DataFrame(values, columns=[f"measurement {j+1:02d}" for j in range(60)])
    frame.insert(0, "sample_id", [f"{i:05d}" for i in range(256)])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return Path(path)
