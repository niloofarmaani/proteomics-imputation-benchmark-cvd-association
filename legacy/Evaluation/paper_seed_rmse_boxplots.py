from __future__ import annotations

import ast
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, MultipleLocator

import config_paths


PLOT_MODULE_VERSION = "2026-10-15-method-colours-taller-v4"

# Increase figure height by 50% without changing the runner.
# Applied equally to all missingness-mechanism panels.
FIGURE_VERTICAL_SCALE = 1




# =====================================================================
# Method names, aliases, colours and result-file locations
# =====================================================================

_METHOD_ALIASES = {
    "KNN": ["knn"],
    "SoftImpute": ["softImpute", "softimpute"],
    "VAE": ["vae"],
    "DAE": ["dae"],
    "AE": ["ae"],
    "GAIN": ["gain"],
    "Transformer": ["transformer", "trf"],
    "RNN": ["rnn"],
    "MissForest": ["mf", "missforest"],
    "Iterative": ["iterative", "IterativeImputer"],
    "Mean": ["mean"],
}

METHOD_CONFIGS = {
    method: {
        "label": method,
        "eval_dir": f"{method}_seeded_pipeline",
        "csv_prefix": method,
        "aliases": [method] + aliases,
    }
    for method, aliases in _METHOD_ALIASES.items()
}

METHOD_COLORS = {
    "Iterative": "#332288",
    "MissForest": "#117733",
    "SoftImpute": "#CC6677",
    "KNN": "#88CCEE",
    "AE": "#44AA99",
    "DAE": "#999933",
    "GAIN": "#DDCC77",
    "VAE": "#AA4499",
    "Transformer": "#882255",
    "Mean": "#666666",
    "RNN": "#AA7744",
}


# =====================================================================
# General helpers
# =====================================================================

def canonical_method_name(method):
    value = str(method).strip().lower()

    for canonical, config in METHOD_CONFIGS.items():
        aliases = [
            alias.lower()
            for alias in config["aliases"]
        ]
        if value in aliases:
            return canonical

    raise KeyError(
        f"Unknown method '{method}'. "
        f"Supported methods: {list(METHOD_CONFIGS)}"
    )


def label_for_method(method):
    return METHOD_CONFIGS[
        canonical_method_name(method)
    ]["label"]


def ordered_missingness(values):
    preferred = ["MCAR", "MAR", "MNAR"]

    values = [
        str(value).strip().upper()
        for value in values
    ]

    present = [
        mechanism
        for mechanism in preferred
        if mechanism in set(values)
    ]

    return present + sorted(set(values) - set(present))


def safe_fraction_token(fraction):
    return str(float(fraction)).replace(".", "p")


def format_fraction_percent(fraction):
    percentage = float(fraction) * 100.0

    if percentage.is_integer():
        return f"{int(percentage)}%"

    return f"{percentage:g}%"


def mean_sd_string(mean, sd, digits=3):
    if not np.isfinite(mean):
        return ""

    if not np.isfinite(sd):
        return f"{mean:.{digits}f}"

    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def compact_json(obj, max_len=360):
    text = json.dumps(
        obj,
        sort_keys=True,
        default=str,
    )

    if len(text) <= max_len:
        return text

    return text[:max_len - 3] + "..."


def _integer_values(values, name):
    numbers = pd.to_numeric(
        pd.Series(values),
        errors="raise",
    ).to_numpy(dtype=float)

    valid = (
        np.isfinite(numbers).all()
        and np.equal(numbers, np.floor(numbers)).all()
    )

    if not valid:
        raise ValueError(
            f"{name} must contain finite integers."
        )

    return numbers.astype(int)


# =====================================================================
# Load completed repeated-seed RMSE results
# =====================================================================

def _read_seed_csv(path, method_label):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            "Required repeated-seed RMSE CSV does not exist "
            f"for {method_label}:\n{path}\n\n"
            "Run the method-specific seeded evaluator first."
        )

    frame = pd.read_csv(path)

    required = {
        "missingness",
        "fraction",
        "seed",
        "rmse",
    }

    missing = required - set(frame.columns)

    if missing:
        raise KeyError(
            f"Seed RMSE CSV for {method_label} is missing "
            f"columns: {sorted(missing)}\n"
            f"File: {path}\n"
            f"Columns: {list(frame.columns)}"
        )

    frame = frame.copy()

    frame["missingness"] = (
        frame["missingness"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    frame["fraction"] = pd.to_numeric(
        frame["fraction"],
        errors="raise",
    )

    frame["seed"] = _integer_values(
        frame["seed"],
        "seed",
    )

    frame["rmse"] = pd.to_numeric(
        frame["rmse"],
        errors="raise",
    )

    finite = np.isfinite(
        frame["rmse"].to_numpy(dtype=float)
    )

    if not finite.all():
        bad = frame.loc[~finite]

        raise ValueError(
            f"Non-finite RMSE values in {path}:\n"
            f"{bad.head(20).to_string(index=False)}"
        )

    frame["method"] = method_label
    frame["source_csv"] = str(path)

    return frame


def load_seed_rmse_values(
    *,
    columns_num,
    methods_to_compare,
    fractions,
    missingness_types,
    seeds,
    seed_values_csv=None,
    csv_overrides=None,
    strict=True,
):
    """
    Load existing final RMSE values.

    Default input for each method:
        Evaluation/<Method>_seeded_pipeline/
        <Method>_final_rmse_by_seed.csv

    Alternatively, seed_values_csv may point to a combined CSV
    containing:
        method, missingness, fraction, seed, rmse

    This function does not train or rerun imputation models.
    """
    main_dir = Path(
        config_paths.experiments_dir(columns_num)
    )

    fractions = [
        float(value)
        for value in fractions
    ]

    missingness_types = [
        str(value).strip().upper()
        for value in missingness_types
    ]

    seeds = _integer_values(
        seeds,
        "seeds",
    ).tolist()

    method_labels = [
        label_for_method(method)
        for method in methods_to_compare
    ]

    if len(method_labels) != len(set(method_labels)):
        raise ValueError(
            "methods_to_compare contains duplicate "
            "methods or aliases."
        )

    if seed_values_csv is not None:
        path = (
            Path(seed_values_csv)
            .expanduser()
            .resolve()
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Combined seed values CSV not found:\n{path}"
            )

        frame = pd.read_csv(path)

        required = {
            "method",
            "missingness",
            "fraction",
            "seed",
            "rmse",
        }

        missing = required - set(frame.columns)

        if missing:
            raise KeyError(
                "Combined seed CSV is missing columns "
                f"{sorted(missing)}:\n{path}"
            )

        frame = frame.copy()

        frame["method"] = (
            frame["method"]
            .astype(str)
            .map(label_for_method)
        )

        frame["missingness"] = (
            frame["missingness"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        frame["fraction"] = pd.to_numeric(
            frame["fraction"],
            errors="raise",
        )

        frame["seed"] = _integer_values(
            frame["seed"],
            "seed",
        )

        frame["rmse"] = pd.to_numeric(
            frame["rmse"],
            errors="raise",
        )

        frame = frame[
            frame["method"].isin(method_labels)
            & frame["missingness"].isin(missingness_types)
            & frame["fraction"].isin(fractions)
            & frame["seed"].isin(seeds)
        ].copy()

    else:
        override_map = {
            canonical_method_name(method): (
                Path(path).expanduser().resolve()
            )
            for method, path in (csv_overrides or {}).items()
        }

        frames = []

        for method in methods_to_compare:
            canonical = canonical_method_name(method)
            config = METHOD_CONFIGS[canonical]

            csv_path = override_map.get(
                canonical,
                main_dir
                / "Evaluation"
                / config["eval_dir"]
                / f"{config['csv_prefix']}_final_rmse_by_seed.csv",
            )

            method_frame = _read_seed_csv(
                csv_path,
                method_label=config["label"],
            )

            method_frame = method_frame[
                method_frame["fraction"].isin(fractions)
                & method_frame["missingness"].isin(
                    missingness_types
                )
                & method_frame["seed"].isin(seeds)
            ].copy()

            frames.append(method_frame)

        if not frames:
            raise ValueError(
                "No repeated-seed RMSE values were loaded."
            )

        frame = pd.concat(
            frames,
            ignore_index=True,
        )

    if frame.empty:
        raise ValueError(
            "No data match the requested methods, "
            "conditions and seeds."
        )

    if not np.isfinite(
        frame["rmse"].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Selected seed data contain non-finite RMSE values."
        )

    if (frame["rmse"] < 0).any():
        raise ValueError(
            "RMSE values must be nonnegative."
        )

    key_columns = [
        "method",
        "missingness",
        "fraction",
        "seed",
    ]

    duplicated = frame.duplicated(
        subset=key_columns,
        keep=False,
    )

    if duplicated.any():
        bad = frame.loc[
            duplicated,
            key_columns + ["rmse"],
        ]

        raise ValueError(
            "Duplicate repeated-seed RMSE rows:\n"
            f"{bad.head(40).to_string(index=False)}"
        )

    expected = {
        (method, mechanism, fraction, seed)
        for method in method_labels
        for mechanism in missingness_types
        for fraction in fractions
        for seed in seeds
    }

    observed = set(
        frame[key_columns].itertuples(
            index=False,
            name=None,
        )
    )

    missing_rows = sorted(expected - observed)

    if missing_rows:
        preview = "\n".join(
            str(row)
            for row in missing_rows[:40]
        )

        message = (
            "Missing repeated-seed RMSE rows.\n"
            "Rows are (method, missingness, fraction, seed):\n"
            f"{preview}"
        )

        if strict:
            raise ValueError(message)

        warnings.warn(
            f"Incomplete seed grid: {len(missing_rows)} "
            "requested rows are absent.",
            stacklevel=2,
        )

    frame["method"] = pd.Categorical(
        frame["method"],
        categories=method_labels,
        ordered=True,
    )

    frame = (
        frame.sort_values(
            ["missingness", "fraction", "method", "seed"]
        )
        .reset_index(drop=True)
    )

    frame["method"] = frame["method"].astype(str)

    return frame


def summarize_seed_rmse(seed_df):
    """Summarize RMSE across seeds; standard deviation uses ddof=1."""
    return (
        seed_df.groupby(
            ["missingness", "fraction", "method"]
        )["rmse"]
        .agg(
            ["count", "mean", "std", "median", "min", "max"]
        )
        .reset_index()
    )


# =====================================================================
# Optional optimization-table helpers
# =====================================================================

def _read_json(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"JSON file not found:\n{path}"
        )

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _study_name_map(study_names):
    return {
        canonical_method_name(method): str(study_name)
        for method, study_name in study_names.items()
    }


def _condition_scores_path(main_dir, study_name):
    return (
        Path(main_dir)
        / "Optimization"
        / "trial_tables"
        / f"{study_name}_condition_scores.csv"
    )


def _best_params_path(main_dir, study_name):
    return (
        Path(main_dir)
        / "Optimization"
        / "best_params"
        / f"{study_name}_best_params.json"
    )


def _load_condition_scores(main_dir, study_name):
    path = _condition_scores_path(
        main_dir,
        study_name,
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Condition score table not found:\n{path}"
        )

    frame = pd.read_csv(path)

    required = {
        "trial_number",
        "state",
        "fraction",
        "missingness",
        "validation_rmse",
    }

    missing = required - set(frame.columns)

    if missing:
        raise KeyError(
            "Condition score table is missing "
            f"{sorted(missing)}:\n{path}"
        )

    frame = frame[
        frame["state"].astype(str).str.upper()
        == "COMPLETE"
    ].copy()

    if frame.empty:
        raise ValueError(
            "No COMPLETE condition-score rows found:\n"
            f"{path}"
        )

    frame["trial_number"] = _integer_values(
        frame["trial_number"],
        "trial_number",
    )

    frame["fraction"] = pd.to_numeric(
        frame["fraction"],
        errors="raise",
    )

    frame["missingness"] = (
        frame["missingness"]
        .astype(str)
        .str.upper()
    )

    frame["validation_rmse"] = pd.to_numeric(
        frame["validation_rmse"],
        errors="raise",
    )

    return frame


def _try_parse_dict(raw):
    if raw is None:
        return {}

    if isinstance(raw, dict):
        return raw

    try:
        if pd.isna(raw):
            return {}
    except (TypeError, ValueError):
        pass

    text = str(raw).strip()

    if not text or text.lower() in {"nan", "none", "null"}:
        return {}

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)

            if isinstance(parsed, dict):
                return parsed

        except (ValueError, SyntaxError, TypeError):
            continue

    return {}


def _collect_param_values(condition_df):
    values = {}

    if "method_params_json" in condition_df.columns:
        for raw in condition_df["method_params_json"]:
            for key, value in _try_parse_dict(raw).items():
                values.setdefault(
                    str(key), set()
                ).add(str(value))

    for column in condition_df.columns:
        if column.startswith("params_"):
            key = column.replace("params_", "", 1)

        elif (
            column.startswith("method_params_")
            and column != "method_params_json"
        ):
            key = column.replace("method_params_", "", 1)

        else:
            continue

        for value in condition_df[column].dropna():
            values.setdefault(
                str(key), set()
            ).add(str(value))

    return {
        key: sorted(value_set)
        for key, value_set in values.items()
    }


def _format_value_set(values):
    if not values:
        return ""

    numeric = pd.to_numeric(
        pd.Series(values),
        errors="coerce",
    )

    if numeric.notna().all() and len(values) > 6:
        return (
            f"{numeric.min():.3g} "
            f"to {numeric.max():.3g}"
        )

    if len(values) <= 8:
        return "{" + ", ".join(values) + "}"

    return f"{len(values)} values"


def _search_space_summary(condition_df):
    """
    Summarize parameter values recorded in completed trials.

    These need not represent the full configured search space.
    """
    varying = []
    constant = []

    for key, values in sorted(
        _collect_param_values(condition_df).items()
    ):
        description = (
            f"{key}: {_format_value_set(values)}"
        )

        if len(values) > 1:
            varying.append(description)
        else:
            constant.append(description)

    return (
        "; ".join(varying),
        "; ".join(constant),
    )


def _best_validation_by_condition(
    best_payload,
    missingness,
    fraction,
):
    for row in best_payload.get("score_rows", []):
        try:
            matches = (
                str(row.get("missingness")).upper()
                == str(missingness).upper()
                and np.isclose(
                    float(row.get("fraction")),
                    float(fraction),
                )
            )

            if matches:
                return float(
                    row.get("validation_rmse")
                )

        except (TypeError, ValueError):
            continue

    return None


def _final_rmse_cell(
    summary_df,
    method,
    missingness,
    fraction,
):
    subset = summary_df[
        (summary_df["method"].astype(str) == str(method))
        & (
            summary_df["missingness"]
            .astype(str)
            .str.upper()
            == str(missingness).upper()
        )
        & np.isclose(
            summary_df["fraction"].astype(float),
            float(fraction),
        )
    ]

    if subset.empty:
        return ""

    row = subset.iloc[0]

    return mean_sd_string(
        float(row["mean"]),
        float(row["std"]),
    )


def _table_to_latex(table):
    """Export a booktabs tabular without requiring Jinja2."""
    substitutions = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "±": r"$\pm$",
        "–": "--",
        "\n": " ",
    }

    def escape(value):
        if value is None:
            return ""

        if np.isscalar(value) and pd.isna(value):
            return ""

        return "".join(
            substitutions.get(character, character)
            for character in str(value)
        )

    end_row = " " + chr(92) * 2

    lines = [
        r"\begin{tabular}{"
        + "l" * len(table.columns)
        + "}",
        r"\toprule",
        " & ".join(map(escape, table.columns)) + end_row,
        r"\midrule",
    ]

    for row in table.itertuples(index=False, name=None):
        lines.append(
            " & ".join(map(escape, row)) + end_row
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
    ])

    return "\n".join(lines) + "\n"


def make_bayesian_optimization_tables(
    *,
    columns_num,
    methods_to_compare,
    study_names,
    seed_summary,
    optimization_seed=42,
    final_seeds=(42, 100, 2024, 777, 999),
    tuning_missingness_types=("MAR", "MNAR"),
    tuning_fractions=(0.1, 0.3),
    main_missingness="MAR",
    main_fractions=(0.1, 0.3, 0.5),
    supplementary_missingness_types=("MCAR", "MNAR"),
    supplementary_fractions=(0.1, 0.3, 0.5),
    output_dir=None,
    print_latex=True,
    print_markdown=True,
):
    main_dir = Path(
        config_paths.experiments_dir(columns_num)
    )

    study_map = _study_name_map(study_names)

    if output_dir is None:
        output_dir = (
            main_dir
            / "paper_figures"
            / "seed_rmse_boxplots_final"
            / "tables"
        )
    else:
        output_dir = (
            Path(output_dir)
            .expanduser()
            .resolve()
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    design_table = pd.DataFrame(
        [
            (
                "Optimizer",
                "Optuna Tree-structured Parzen Estimator sampler",
            ),
            (
                "Optimization seed",
                str(int(optimization_seed)),
            ),
            (
                "Tuning missingness mechanisms",
                ", ".join(tuning_missingness_types),
            ),
            (
                "Tuning missingness fractions",
                ", ".join(
                    format_fraction_percent(f)
                    for f in tuning_fractions
                ),
            ),
            (
                "Validation objective",
                "Mean validation RMSE across the tuning panel",
            ),
            (
                "Final benchmark strategy",
                "Selected hyperparameters frozen before evaluation",
            ),
            (
                "Final benchmark mechanisms",
                ", ".join(
                    ordered_missingness(
                        seed_summary["missingness"].unique()
                    )
                ),
            ),
            (
                "Final benchmark fractions",
                ", ".join(
                    format_fraction_percent(f)
                    for f in main_fractions
                ),
            ),
            (
                "Final seeds",
                ", ".join(
                    str(int(seed))
                    for seed in final_seeds
                ),
            ),
        ],
        columns=["Item", "Specification"],
    )

    main_rows = []
    search_rows = []
    per_condition_rows = []

    available_mechanisms = ordered_missingness(
        seed_summary["missingness"].unique()
    )

    for method in methods_to_compare:
        canonical = canonical_method_name(method)
        label = label_for_method(method)

        if canonical not in study_map:
            raise KeyError(
                f"No study name supplied for method '{method}'."
            )

        study_name = study_map[canonical]

        best_path = _best_params_path(
            main_dir,
            study_name,
        )

        condition_path = _condition_scores_path(
            main_dir,
            study_name,
        )

        best_payload = _read_json(best_path)

        if (
            best_payload.get("study_name", study_name)
            != study_name
        ):
            raise ValueError(
                f"Study name mismatch in {best_path}"
            )

        if (
            "seed" in best_payload
            and int(best_payload["seed"])
            != int(optimization_seed)
        ):
            raise ValueError(
                f"Optimization seed mismatch in {best_path}"
            )

        condition_df = _load_condition_scores(
            main_dir,
            study_name,
        )

        varying, constant = _search_space_summary(
            condition_df
        )

        selected_parameters = json.dumps(
            best_payload.get("params", {}),
            sort_keys=True,
            default=str,
        )

        search_rows.append({
            "Method": label,
            "Study name": study_name,
            "Completed trials": int(
                condition_df["trial_number"].nunique()
            ),
            "Tuned hyperparameters": varying,
            "Fixed settings recorded in trial table": constant,
            "Selected hyperparameters": selected_parameters,
            "Best mean validation RMSE": best_payload.get(
                "best_value_validation_rmse",
                np.nan,
            ),
            "Best trial": best_payload.get(
                "best_trial_number",
                "",
            ),
            "Condition score table": str(condition_path),
            "Best-parameter JSON": str(best_path),
        })

        # Main reconstruction table contains only the method
        # and final RMSE for each MAR missingness fraction.
        row = {"Method": label}

        for fraction in main_fractions:
            column = (
                f"{main_missingness} "
                f"{format_fraction_percent(fraction)}"
            )

            row[column] = _final_rmse_cell(
                seed_summary,
                label,
                main_missingness,
                fraction,
            )

        main_rows.append(row)

        for mechanism in available_mechanisms:
            for fraction in main_fractions:
                per_condition_rows.append({
                    "Method": label,
                    "Missingness": mechanism,
                    "Fraction": format_fraction_percent(
                        fraction
                    ),
                    "Final RMSE, mean ± SD": _final_rmse_cell(
                        seed_summary,
                        label,
                        mechanism,
                        fraction,
                    ),
                })

    supplementary_rows = []

    for method in methods_to_compare:
        label = label_for_method(method)
        row = {"Method": label}

        for mechanism in ordered_missingness(
            supplementary_missingness_types
        ):
            for fraction in supplementary_fractions:
                column = (
                    f"{mechanism} "
                    f"{format_fraction_percent(fraction)} "
                    "final RMSE, mean ± SD"
                )

                row[column] = _final_rmse_cell(
                    seed_summary,
                    label,
                    mechanism,
                    fraction,
                )

        supplementary_rows.append(row)

    materials_index = pd.DataFrame([
        {
            "Item": "Main Figure",
            "File": "Figure1_RMSE_All_Conditions.pdf",
            "Content": (
                "Final RMSE across seeds; colours identify methods "
                "and groups identify missingness fractions."
            ),
        },
        {
            "Item": "Main Table",
            "File": "main_mar_bayesian_and_rmse_table.tex",
            "Content": "Final MAR RMSE, mean ± SD.",
        },
        {
            "Item": "Supplementary Table",
            "File": "supplementary_non_mar_rmse_table.tex",
            "Content": (
                "Final RMSE for the remaining mechanisms."
            ),
        },
        {
            "Item": "Supplementary Table",
            "File": (
                "supplementary_hyperparameter_search_space_table.tex"
            ),
            "Content": (
                "Completed trials, recorded parameter values "
                "and selected settings."
            ),
        },
        {
            "Item": "Source data",
            "File": "paper_seed_rmse_values.csv",
            "Content": (
                "Individual final RMSE values by condition and seed."
            ),
        },
        {
            "Item": "Source data",
            "File": "paper_seed_rmse_summary.csv",
            "Content": "Summary statistics across seeds.",
        },
    ])

    tables = {
        "bayesian_optimization_design": design_table,
        "main_mar_bayesian_and_rmse_table": pd.DataFrame(
            main_rows
        ),
        "supplementary_non_mar_rmse_table": pd.DataFrame(
            supplementary_rows
        ),
        "supplementary_hyperparameter_search_space_table": (
            pd.DataFrame(search_rows)
        ),
        "supplementary_all_conditions_long_rmse_table": (
            pd.DataFrame(per_condition_rows)
        ),
        "supplementary_materials_index": materials_index,
    }

    for name, table in tables.items():
        table.to_csv(
            output_dir / f"{name}.csv",
            index=False,
        )

        latex = _table_to_latex(table)

        (output_dir / f"{name}.tex").write_text(
            latex,
            encoding="utf-8",
        )

        if print_markdown:
            print(f"\n{name}")

            try:
                print(table.to_markdown(index=False))
            except ImportError:
                print(table.to_string(index=False))

        if print_latex:
            print(f"\nLaTeX: {name}")
            print(latex)

    return tables


# =====================================================================
# Plot ordering and shared y-axis limits
# =====================================================================

def _resolve_order(
    seed_df,
    methods_to_compare,
    sort_by,
):
    labels = [
        label_for_method(method)
        for method in methods_to_compare
    ]

    if sort_by == "input":
        return labels

    if sort_by not in {"mean", "median", "min"}:
        raise ValueError(
            "sort_by must be input, mean, median or min."
        )

    basis = seed_df[
        seed_df["missingness"] == "MAR"
    ]

    if basis.empty:
        basis = seed_df

    scores = (
        basis.groupby("method")["rmse"]
        .agg(sort_by)
        .reindex(labels)
    )

    return (
        scores.sort_values(
            kind="mergesort",
            na_position="last",
        )
        .index.tolist()
    )


def _resolve_limits(seed_df, y_range):
    values = seed_df["rmse"].to_numpy(dtype=float)

    if not len(values) or not np.isfinite(values).all():
        raise ValueError(
            "Finite RMSE values are required for plotting."
        )

    minimum = float(values.min())
    maximum = float(values.max())

    if y_range is not None:
        lower, upper = map(float, y_range)

        if (
            not np.isfinite([lower, upper]).all()
            or lower >= upper
        ):
            raise ValueError(
                "y_range must be finite and increasing."
            )

        if lower >= minimum or upper <= maximum:
            raise ValueError(
                "y_range would hide or clip seed results. "
                "Use y_range=None."
            )

        return lower, upper

    padding = max(
        0.004,
        0.08 * (maximum - minimum),
    )

    return minimum - padding, maximum + padding


# =====================================================================
# Publication plots
# =====================================================================

def _plot_mechanism_panels(
    *,
    seed_df,
    mechanisms,
    fractions,
    method_order,
    seeds,
    output_dir,
    filename,
    y_limits,
    y_tick_step=None,
    figure_width_mm=170.0,
    figure_height_mm=175.0,
    show_seed_points=True,
    show_means=False,
    show_condition_titles=True,
    show_panel_labels=True,
    show_method_separators=True,
    repeat_method_labels=False,
    box_fill_alpha=0.90,
    save_png=False,
    show=True,
):
    """
    Draw one panel per missingness mechanism.

    Colours identify methods.
    Horizontal groups identify missingness fractions.

    Boxes represent the interquartile range.
    Horizontal lines indicate medians.
    Whiskers extend to observations within 1.5 IQR.
    Points show individual simulation seeds.

    All panels use equal heights and the same RMSE limits.
    Physical figure height is multiplied by FIGURE_VERTICAL_SCALE.

    repeat_method_labels is retained for compatibility.
    Method names appear only in the shared legend.
    """
    if not mechanisms or not fractions or not method_order:
        raise ValueError(
            "Mechanisms, fractions and methods must be nonempty."
        )

    if figure_width_mm <= 0 or figure_height_mm <= 0:
        raise ValueError(
            "Figure dimensions must be positive."
        )

    if y_tick_step is not None and y_tick_step <= 0:
        raise ValueError(
            "y_tick_step must be positive or None."
        )

    if not 0 <= box_fill_alpha <= 1:
        raise ValueError(
            "box_fill_alpha must be between zero and one."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_methods = len(method_order)
    n_fractions = len(fractions)

    # Horizontal dimensions remain unchanged.
    box_width = 0.88
    group_gap = 1.55

    group_starts = (
        np.arange(n_fractions) * (n_methods + group_gap)
    )

    group_centres = (
        group_starts + (n_methods - 1) / 2
    )

    left = -0.72
    right = (
        group_starts[-1] + n_methods - 1 + 0.72
    )

    # Horizontal jitter is deterministic.
    # Vertical coordinates remain the actual RMSE values.
    unique_seeds = sorted(
        set(int(seed) for seed in seeds)
    )

    if len(unique_seeds) > 1:
        offsets = (
            np.linspace(-0.28, 0.28, len(unique_seeds))
            * box_width
        )
    else:
        offsets = np.zeros(1)

    seed_offsets = dict(
        zip(unique_seeds, offsets)
    )

    style = {
        "font.family": "DejaVu Sans",
        "font.size": 9.0,
        "axes.labelsize": 10.0,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 9.0,
        "axes.linewidth": 0.75,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": None,
    }

    with plt.rc_context(style):
        # Increase physical height by 50%.
        # For the existing runner: 175 mm -> 262.5 mm.
        fig = plt.figure(
            figsize=(
                figure_width_mm / 25.4,
                (
                    figure_height_mm
                    * FIGURE_VERTICAL_SCALE
                    / 25.4
                ),
            ),
            constrained_layout=True,
        )

        fig.set_constrained_layout_pads(
            w_pad=0.04,
            h_pad=0.04,
            hspace=0.04,
        )

        legend_columns = min(3, n_methods)

        legend_rows = int(
            np.ceil(n_methods / legend_columns)
        )

        # Allocate additional space primarily to the plot panels.
        # Every mechanism receives an equal-height panel.
        grid = fig.add_gridspec(
            len(mechanisms) + 1,
            1,
            height_ratios=(
                [0.21 * legend_rows + 0.16]
                + [
                    1.8 * FIGURE_VERTICAL_SCALE
                ] * len(mechanisms)
            ),
        )

        # ---------------------------------------------------------
        # Shared method legend
        # ---------------------------------------------------------
        legend_ax = fig.add_subplot(grid[0])
        legend_ax.axis("off")

        handles = [
            Patch(
                facecolor=to_rgba(
                    METHOD_COLORS[method],
                    box_fill_alpha,
                ),
                edgecolor="0.23",
                linewidth=0.8,
                label=method,
            )
            for method in method_order
        ]

        # Matplotlib fills legend columns first.
        # Reorder so legend rows follow the box order.
        legend_order = [
            row * legend_columns + col
            for col in range(legend_columns)
            for row in range(legend_rows)
            if row * legend_columns + col < n_methods
        ]

        legend_ax.legend(
            handles=[
                handles[index]
                for index in legend_order
            ],
            loc="center",
            ncol=legend_columns,
            frameon=False,
            title="Imputation method",
            title_fontsize=9.5,
            handlelength=1.7,
            handleheight=0.85,
            columnspacing=2.1,
            labelspacing=0.45,
            borderaxespad=0,
        )

        # ---------------------------------------------------------
        # Mechanism panels
        # ---------------------------------------------------------
        axes = []

        for panel_index, mechanism in enumerate(mechanisms):
            ax = fig.add_subplot(
                grid[panel_index + 1]
            )
            axes.append(ax)

            panel = seed_df[
                seed_df["missingness"] == mechanism
            ]

            # Dashed separators between fraction groups.
            if show_method_separators:
                for group_index in range(n_fractions - 1):
                    boundary = (
                        group_starts[group_index]
                        + n_methods - 1
                        + group_starts[group_index + 1]
                    ) / 2

                    ax.axvline(
                        boundary,
                        color="0.78",
                        linewidth=0.7,
                        linestyle=(0, (3, 3)),
                        zorder=0,
                    )

            for fraction_index, fraction in enumerate(fractions):
                for method_index, method in enumerate(method_order):
                    subset = panel[
                        (panel["method"] == method)
                        & (
                            panel["fraction"] == float(fraction)
                        )
                    ].sort_values("seed")

                    if subset.empty:
                        continue

                    values = subset["rmse"].to_numpy(
                        dtype=float
                    )

                    position = (
                        group_starts[fraction_index]
                        + method_index
                    )

                    color = METHOD_COLORS[method]

                    ax.boxplot(
                        [values],
                        positions=[position],
                        widths=box_width,
                        manage_ticks=False,
                        patch_artist=True,
                        whis=1.5,
                        showmeans=show_means,
                        showfliers=not show_seed_points,
                        boxprops={
                            "facecolor": to_rgba(
                                color,
                                box_fill_alpha,
                            ),
                            "edgecolor": "0.23",
                            "linewidth": 1.0,
                        },
                        medianprops={
                            "color": "0.12",
                            "linewidth": 1.25,
                        },
                        whiskerprops={
                            "color": "0.30",
                            "linewidth": 0.9,
                        },
                        capprops={
                            "color": "0.30",
                            "linewidth": 0.9,
                        },
                        meanprops={
                            "marker": "D",
                            "markerfacecolor": "white",
                            "markeredgecolor": "0.10",
                            "markersize": 3.5,
                            "markeredgewidth": 0.65,
                            "zorder": 6,
                        },
                        flierprops={
                            "marker": "o",
                            "markerfacecolor": color,
                            "markeredgecolor": "0.2",
                            "markersize": 3,
                        },
                    )

                    if show_seed_points:
                        point_positions = position + np.array([
                            seed_offsets[int(seed)]
                            for seed in subset["seed"]
                        ])

                        ax.scatter(
                            point_positions,
                            values,
                            s=8,
                            marker="o",
                            facecolors=color,
                            edgecolors="white",
                            linewidths=0.35,
                            alpha=0.95,
                            zorder=5,
                        )

            title_parts = []

            if show_panel_labels:
                title_parts.append(
                    chr(ord("A") + panel_index)
                )

            if show_condition_titles:
                title_parts.append(mechanism)

            if title_parts:
                ax.set_title(
                    "  ".join(title_parts),
                    loc="left",
                    fontweight="bold",
                    pad=5,
                )

            ax.set_ylabel("RMSE")
            ax.set_xlim(left, right)

            # Shared, unchanged numerical RMSE range.
            ax.set_ylim(*y_limits)

            ax.set_xticks(group_centres)
            ax.set_xticklabels([
                format_fraction_percent(fraction)
                for fraction in fractions
            ])

            ax.tick_params(
                axis="x",
                length=0,
                pad=5,
            )

            if y_tick_step is None:
                locator = MaxNLocator(nbins=5)
            else:
                locator = MultipleLocator(y_tick_step)

            ax.yaxis.set_major_locator(locator)

            ax.grid(
                axis="y",
                color="0.89",
                linewidth=0.6,
            )

            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        axes[-1].set_xlabel(
            "Missingness fraction",
            labelpad=6,
        )

        fig.canvas.draw()

        paths = {
            "pdf": str(
                output_dir / f"{filename}.pdf"
            )
        }

        # Preserve the taller physical dimensions.
        fig.savefig(
            paths["pdf"],
            facecolor="white",
        )

        if save_png:
            paths["png"] = str(
                output_dir / f"{filename}.png"
            )

            fig.savefig(
                paths["png"],
                dpi=300,
                facecolor="white",
            )

        if show:
            plt.show()

        plt.close(fig)

    return paths


# =====================================================================
# Public entry point
# =====================================================================

def make_paper_seed_rmse_boxplots(
    *,
    columns_num,
    methods_to_compare,
    fractions,
    missingness_types,
    seeds,
    study_names=None,
    output_dir=None,
    seed_values_csv=None,
    csv_overrides=None,
    strict=True,
    sort_by="input",
    shared_method_order=True,
    y_range=None,
    y_tick_step=None,
    show_seed_points=True,
    show_means=False,
    show=True,
    save_png=False,
    show_condition_titles=True,
    show_panel_labels=True,
    make_legacy_split_figures=False,
    make_individual_panels=True,
    make_bayes_tables=True,
    print_latex_tables=True,
    print_markdown_tables=True,
    figure_width_mm=170.0,
    figure_height_mm=175.0,
    make_mechanism_panels=True,
    optimization_seed=42,
    show_method_separators=True,
    repeat_method_labels=False,
    box_fill_alpha=0.90,
):
    """
    Generate final repeated-seed RMSE figures and optional tables.

    Returns:
        outputs, seed_df, summary, tables

    The main figure has one panel per missingness mechanism.
    Methods are identified by colour and grouped within fractions.

    figure_height_mm is the base height. Exported height is
    multiplied internally by FIGURE_VERTICAL_SCALE.
    """
    if not shared_method_order:
        raise ValueError(
            "The publication figure requires "
            "shared_method_order=True."
        )

    if (
        not methods_to_compare
        or not fractions
        or not missingness_types
        or not seeds
    ):
        raise ValueError(
            "Methods, fractions, mechanisms and seeds "
            "must be nonempty."
        )

    fractions = sorted(
        float(f)
        for f in fractions
    )

    if (
        len(fractions) != len(set(fractions))
        or any(not 0 < f < 1 for f in fractions)
    ):
        raise ValueError(
            "fractions must be unique values "
            "between zero and one."
        )

    mechanisms = ordered_missingness(
        missingness_types
    )

    seeds = _integer_values(
        seeds,
        "seeds",
    ).tolist()

    if len(seeds) != len(set(seeds)):
        raise ValueError(
            "seeds must not contain duplicates."
        )

    main_dir = Path(
        config_paths.experiments_dir(columns_num)
    )

    if output_dir is None:
        output_path = (
            main_dir
            / "paper_figures"
            / "seed_rmse_boxplots_final"
        )
    else:
        output_path = (
            Path(output_dir)
            .expanduser()
            .resolve()
        )

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    seed_df = load_seed_rmse_values(
        columns_num=columns_num,
        methods_to_compare=methods_to_compare,
        fractions=fractions,
        missingness_types=mechanisms,
        seeds=seeds,
        seed_values_csv=seed_values_csv,
        csv_overrides=csv_overrides,
        strict=strict,
    )

    summary = summarize_seed_rmse(seed_df)

    method_order = _resolve_order(
        seed_df,
        methods_to_compare,
        sort_by,
    )

    y_limits = _resolve_limits(
        seed_df,
        y_range,
    )

    tables = {}

    if make_bayes_tables:
        if study_names is None:
            raise ValueError(
                "study_names are required when "
                "make_bayes_tables=True."
            )

        tables = make_bayesian_optimization_tables(
            columns_num=columns_num,
            methods_to_compare=methods_to_compare,
            study_names=study_names,
            seed_summary=summary,
            optimization_seed=optimization_seed,
            final_seeds=seeds,
            tuning_missingness_types=("MAR", "MNAR"),
            tuning_fractions=(0.1, 0.3),
            main_missingness="MAR",
            main_fractions=fractions,
            supplementary_missingness_types=[
                mechanism
                for mechanism in mechanisms
                if mechanism != "MAR"
            ],
            supplementary_fractions=fractions,
            output_dir=output_path / "tables",
            print_latex=print_latex_tables,
            print_markdown=print_markdown_tables,
        )

    values_csv = (
        output_path / "paper_seed_rmse_values.csv"
    )

    summary_csv = (
        output_path / "paper_seed_rmse_summary.csv"
    )

    seed_df.to_csv(
        values_csv,
        index=False,
    )

    summary.to_csv(
        summary_csv,
        index=False,
    )

    outputs = {
        "output_dir": str(output_path),
        "values_csv": str(values_csv),
        "summary_csv": str(summary_csv),
        "main_all_conditions_figure": None,
        "main_mar_figure": None,
        "supplementary_non_mar_figure": None,
        "supplementary_all_conditions_figure": None,
        "individual_panels": {},
        "mechanism_panels": {},
        "method_order": method_order,
        "method_colors": {
            method: METHOD_COLORS[method]
            for method in method_order
        },
        "y_range": y_limits,
        "module_version": PLOT_MODULE_VERSION,
        "tables": {
            name: str(
                output_path
                / "tables"
                / f"{name}.csv"
            )
            for name in tables
        },
    }

    common_plot_arguments = {
        "seed_df": seed_df,
        "method_order": method_order,
        "seeds": seeds,
        "y_limits": y_limits,
        "y_tick_step": y_tick_step,
        "figure_width_mm": figure_width_mm,
        "show_seed_points": show_seed_points,
        "show_means": show_means,
        "show_condition_titles": show_condition_titles,
        "show_panel_labels": show_panel_labels,
        "show_method_separators": show_method_separators,
        "repeat_method_labels": repeat_method_labels,
        "box_fill_alpha": box_fill_alpha,
        "save_png": save_png,
        "show": show,
    }

    # Combined figure containing all requested mechanisms.
    outputs["main_all_conditions_figure"] = (
        _plot_mechanism_panels(
            **common_plot_arguments,
            mechanisms=mechanisms,
            fractions=fractions,
            output_dir=output_path,
            filename="Figure1_RMSE_All_Conditions",
            figure_height_mm=figure_height_mm,
        )
    )

    outputs["supplementary_all_conditions_figure"] = (
        outputs["main_all_conditions_figure"]
    )

    # Separate PDFs for MCAR, MAR and MNAR.
    if make_mechanism_panels:
        for mechanism in mechanisms:
            outputs["mechanism_panels"][mechanism] = (
                _plot_mechanism_panels(
                    **common_plot_arguments,
                    mechanisms=[mechanism],
                    fractions=fractions,
                    output_dir=(
                        output_path / "mechanism_panels"
                    ),
                    filename=f"Panel_RMSE_{mechanism}",
                    figure_height_mm=85.0,
                )
            )

    # Optional individual mechanism/fraction PDFs.
    if make_individual_panels:
        for mechanism in mechanisms:
            for fraction in fractions:
                key = f"{mechanism}_{fraction:g}"

                filename = (
                    f"Panel_RMSE_{mechanism}_"
                    f"{safe_fraction_token(fraction)}"
                )

                outputs["individual_panels"][key] = (
                    _plot_mechanism_panels(
                        **common_plot_arguments,
                        mechanisms=[mechanism],
                        fractions=[fraction],
                        output_dir=(
                            output_path / "individual_panels"
                        ),
                        filename=filename,
                        figure_height_mm=85.0,
                    )
                )

    # Optional filenames retained for older manuscript layouts.
    if make_legacy_split_figures:
        if "MAR" in mechanisms:
            outputs["main_mar_figure"] = (
                _plot_mechanism_panels(
                    **common_plot_arguments,
                    mechanisms=["MAR"],
                    fractions=fractions,
                    output_dir=output_path,
                    filename="Figure_RMSE_MAR_seed_boxplots",
                    figure_height_mm=85.0,
                )
            )

        non_mar = [
            mechanism
            for mechanism in mechanisms
            if mechanism != "MAR"
        ]

        if non_mar:
            outputs["supplementary_non_mar_figure"] = (
                _plot_mechanism_panels(
                    **common_plot_arguments,
                    mechanisms=non_mar,
                    fractions=fractions,
                    output_dir=output_path,
                    filename="Supplementary_RMSE_MCAR_MNAR_seed_boxplots",
                    figure_height_mm=(
                        50.0 * len(non_mar) + 25.0
                    ),
                )
            )

    return outputs, seed_df, summary, tables