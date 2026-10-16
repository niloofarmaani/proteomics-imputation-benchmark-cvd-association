# repair_eid.py

from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import config_paths


# =============================================================================
# Configuration
# =============================================================================
config_paths.set_biobank_root("/home/maani/niloo/imputationproject")
config_paths.set_experiments_name("Experiments-mayovasc-proteomic-ccs")

DATA_FILE = "/scratch/biobank/niloo/mayoVasc/cureASciD_01_data_olinkTarget96_20241030.csv"

columns_num = "all"
fractions = [0.01, 0.1, 0.3, 0.5]
missingness_types = ["MCAR", "MAR", "MNAR"]

main_dir = Path(config_paths.experiments_dir(columns_num))

# First run with DRY_RUN=True.
# If the report looks safe, change to False and rerun.
DRY_RUN = False

MAKE_BACKUP = True

# If non-missing IDs in a file do not match the reference row order,
# skip that file instead of repairing it.
SKIP_IF_ID_ORDER_MISMATCH = True

# Repair these file types.
REPAIR_NORMALIZED = True
REPAIR_CORRUPTED = True
REPAIR_IMPUTED = True

# Your project always uses these metadata columns.
ID_COL = "eid"
LABEL_COL = "label"


# =============================================================================
# Helpers
# =============================================================================
def normalize_id_series(s: pd.Series) -> pd.Series:
    """
    Normalize IDs for matching and comparison.

    Handles:
        123
        123.0
        "123 "
        missing strings
    """
    out = s.astype(str).str.strip()
    out = out.str.replace(r"\.0$", "", regex=True)
    out = out.replace(
        {
            "nan": np.nan,
            "NaN": np.nan,
            "None": np.nan,
            "": np.nan,
            "<NA>": np.nan,
        }
    )
    return out


def read_true_metadata_map(data_file: str) -> tuple:
    """
    Reads DATA_FILE and builds:
        normalized eid -> true eid
        normalized eid -> true binary label
    """

    df = pd.read_csv(data_file)

    if ID_COL not in df.columns:
        raise ValueError(
            f"'{ID_COL}' not found in DATA_FILE. "
            f"Available columns include: {list(df.columns[:30])}"
        )

    if LABEL_COL not in df.columns:
        raise ValueError(
            f"'{LABEL_COL}' not found in DATA_FILE. "
            f"Available columns include: {list(df.columns[:30])}"
        )

    ids_raw = df[ID_COL].copy()
    ids_norm = normalize_id_series(ids_raw)

    labels = pd.to_numeric(df[LABEL_COL], errors="coerce")
    labels = labels.replace([np.inf, -np.inf], np.nan)

    print("\n" + "=" * 100)
    print("DATA_FILE metadata check")
    print("=" * 100)
    print(f"DATA_FILE: {data_file}")
    print(f"ID column: {ID_COL}")
    print(f"Label column: {LABEL_COL}")
    print(f"Rows in DATA_FILE: {len(df)}")
    print(f"Missing IDs: {int(ids_norm.isna().sum())}")
    print(f"Missing labels: {int(labels.isna().sum())}")
    print("Label value counts:")
    print(labels.value_counts(dropna=False).to_string())
    print("=" * 100 + "\n")

    if ids_norm.isna().any():
        bad_idx = np.where(ids_norm.isna())[0][:10].tolist()
        raise ValueError(
            "DATA_FILE has missing eid values. Cannot safely repair labels.\n"
            f"First bad row indices: {bad_idx}"
        )

    if labels.isna().any():
        bad_idx = np.where(labels.isna())[0][:10].tolist()
        raise ValueError(
            "DATA_FILE has missing label values. Cannot safely repair labels from this file.\n"
            f"First bad row indices: {bad_idx}"
        )

    if ids_norm.duplicated().any():
        dupes = ids_norm[ids_norm.duplicated()].head(10).tolist()
        raise ValueError(
            "DATA_FILE has duplicated eid values. Cannot safely build one-to-one label map.\n"
            f"First duplicated IDs: {dupes}"
        )

    unique_labels = sorted(labels.unique().tolist())

    if set(unique_labels) == {0, 1}:
        labels_int = labels.astype(int)
    elif set(unique_labels) == {-1, 1}:
        labels_int = labels.map({-1: 0, 1: 1}).astype(int)
    elif len(unique_labels) == 2:
        mapping = {unique_labels[0]: 0, unique_labels[1]: 1}
        print(f"Mapping labels to binary 0/1 using: {mapping}")
        labels_int = labels.map(mapping).astype(int)
    else:
        raise ValueError(
            f"Label column is not binary. Unique values found: {unique_labels[:20]}"
        )

    id_to_raw_id = dict(zip(ids_norm, ids_raw))
    id_to_label = dict(zip(ids_norm, labels_int))

    return id_to_raw_id, id_to_label


def get_reference_metadata(
    folder: Path,
    id_to_label: dict,
    columns_num: str,
) -> pd.DataFrame:
    """
    Reads normalized_data_col_all.csv in a folder and returns two reference columns:

        column 0 = correct eid in the same row order
        column 1 = correct binary label in the same row order

    Row order is taken from normalized_data_col_all.csv and is never changed.
    """

    normalized_path = folder / f"normalized_data_col_{columns_num}.csv"

    if not normalized_path.exists():
        raise FileNotFoundError(f"Reference normalized file not found: {normalized_path}")

    ref = pd.read_csv(normalized_path)

    if ref.shape[1] < 2:
        raise ValueError(f"Reference file has fewer than 2 columns: {normalized_path}")

    ref_ids_raw = ref.iloc[:, 0].copy()
    ref_ids_norm = normalize_id_series(ref_ids_raw)

    if ref_ids_norm.isna().any():
        bad_idx = np.where(ref_ids_norm.isna())[0][:10].tolist()
        raise ValueError(
            f"Reference normalized file has missing eid values: {normalized_path}\n"
            f"First bad row indices: {bad_idx}"
        )

    mapped_labels = ref_ids_norm.map(id_to_label)

    if mapped_labels.isna().any():
        bad_ids = ref_ids_norm[mapped_labels.isna()].head(10).tolist()
        raise ValueError(
            f"Some reference eids are not found in DATA_FILE label map.\n"
            f"File: {normalized_path}\n"
            f"First unmapped IDs: {bad_ids}"
        )

    # Compare existing labels to DATA_FILE labels where existing labels are present.
    existing_labels = pd.to_numeric(ref.iloc[:, 1], errors="coerce")
    existing_labels = existing_labels.replace([np.inf, -np.inf], np.nan)

    comparable = existing_labels.notna()
    if comparable.any():
        disagreements = int(
            (
                existing_labels[comparable].astype(int).to_numpy()
                != mapped_labels[comparable].astype(int).to_numpy()
            ).sum()
        )
    else:
        disagreements = 0

    if disagreements > 0:
        print(
            f"Warning: normalized label column disagrees with DATA_FILE for "
            f"{disagreements} rows.\n"
            f"Using DATA_FILE labels as truth.\n"
            f"File: {normalized_path}"
        )

    reference_meta = pd.DataFrame(
        {
            ID_COL: ref_ids_raw,
            LABEL_COL: mapped_labels.astype(int),
        }
    )

    return reference_meta


def collect_files_to_repair(
    folder: Path,
    fraction: float,
    missingness: str,
    columns_num: str,
) -> list:
    """
    Collect normalized, corrupted, and imputed files to repair.
    """
    paths = []

    if REPAIR_NORMALIZED:
        paths.append(folder / f"normalized_data_col_{columns_num}.csv")

    if REPAIR_CORRUPTED:
        paths.append(
            folder / f"Corrupted_data_fraction_{fraction}_missingness_{missingness}_col_{columns_num}.csv"
        )

    if REPAIR_IMPUTED:
        paths.extend(sorted(folder.glob("imputed_*.csv")))

    clean_paths = []

    for p in paths:
        if not p.exists():
            continue

        name_low = p.name.lower()

        if "rmse" in name_low or "metric" in name_low or "score" in name_low:
            continue

        clean_paths.append(p)

    return sorted(set(clean_paths))


def repair_one_file(
    path: Path,
    reference_meta: pd.DataFrame,
    dry_run: bool,
    make_backup: bool,
) -> dict:
    """
    Replace first two columns in `path` using `reference_meta`, preserving row order.

    Safety checks:
      - row count must match reference
      - non-missing current IDs must match reference IDs position-by-position
      - protein feature columns must remain unchanged
    """

    df = pd.read_csv(path)

    rec = {
        "path": str(path),
        "file": path.name,
        "status": None,
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "n_missing_id_before": np.nan,
        "n_missing_label_before": np.nan,
        "n_id_order_mismatch_nonmissing": np.nan,
        "n_label_disagree_before": np.nan,
        "backup_path": None,
    }

    if df.shape[1] < 2:
        rec["status"] = "skipped_fewer_than_2_columns"
        return rec

    if df.shape[0] != reference_meta.shape[0]:
        rec["status"] = "skipped_row_count_mismatch"
        rec["reference_rows"] = int(reference_meta.shape[0])
        return rec

    # Save features before repair.
    features_before = df.iloc[:, 2:].copy()

    current_id_raw = df.iloc[:, 0].copy()
    current_label_raw = df.iloc[:, 1].copy()

    reference_id_raw = reference_meta[ID_COL].copy()
    reference_label = pd.to_numeric(reference_meta[LABEL_COL], errors="coerce").astype(int)

    current_id_norm = normalize_id_series(current_id_raw)
    reference_id_norm = normalize_id_series(reference_id_raw)

    current_label = pd.to_numeric(current_label_raw, errors="coerce")
    current_label = current_label.replace([np.inf, -np.inf], np.nan)

    rec["n_missing_id_before"] = int(current_id_norm.isna().sum())
    rec["n_missing_label_before"] = int(current_label.isna().sum())

    # Check non-missing current IDs against reference IDs at the same row positions.
    comparable_ids = current_id_norm.notna()

    if comparable_ids.any():
        id_mismatch = (
            current_id_norm[comparable_ids].to_numpy()
            != reference_id_norm[comparable_ids].to_numpy()
        )
        n_id_mismatch = int(np.sum(id_mismatch))
    else:
        id_mismatch = np.array([])
        n_id_mismatch = 0

    rec["n_id_order_mismatch_nonmissing"] = n_id_mismatch

    if n_id_mismatch > 0 and SKIP_IF_ID_ORDER_MISMATCH:
        mismatch_positions = np.where(comparable_ids.to_numpy())[0][np.where(id_mismatch)[0]]

        examples = []
        for idx in mismatch_positions[:5]:
            examples.append(
                {
                    "row_index": int(idx),
                    "current_id": str(current_id_norm.iloc[idx]),
                    "reference_id": str(reference_id_norm.iloc[idx]),
                }
            )

        rec["status"] = "skipped_id_order_mismatch"
        rec["mismatch_examples"] = examples
        return rec

    # Check label disagreement before repair, for non-missing labels only.
    comparable_labels = current_label.notna()

    if comparable_labels.any():
        n_label_disagree = int(
            (
                current_label[comparable_labels].astype(int).to_numpy()
                != reference_label[comparable_labels].astype(int).to_numpy()
            ).sum()
        )
    else:
        n_label_disagree = 0

    rec["n_label_disagree_before"] = n_label_disagree

    if dry_run:
        rec["status"] = "dry_run_would_repair"
        return rec

    if make_backup:
        backup_path = path.with_suffix(path.suffix + ".bak_eid_label")
        if not backup_path.exists():
            shutil.copy2(path, backup_path)
        rec["backup_path"] = str(backup_path)

    # Replace first two columns by position.
    # This does not reorder rows.
    df.iloc[:, 0] = reference_id_raw.to_numpy()
    df.iloc[:, 1] = reference_label.to_numpy()

    # Standardize the first two column names.
    cols = list(df.columns)
    cols[0] = ID_COL
    cols[1] = LABEL_COL
    df.columns = cols

    # Safety: features must be unchanged.
    features_after = df.iloc[:, 2:]

    if not features_before.equals(features_after):
        rec["status"] = "error_features_changed"
        return rec

    # Safety: labels complete and binary after repair.
    repaired_labels = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    repaired_labels = repaired_labels.replace([np.inf, -np.inf], np.nan)

    if repaired_labels.isna().any():
        rec["status"] = "error_labels_still_missing"
        return rec

    unique_labels = sorted(repaired_labels.unique().tolist())

    if set(unique_labels) != {0, 1}:
        rec["status"] = f"warning_repaired_labels_not_binary_{unique_labels}"
    else:
        rec["status"] = "repaired"

    df.to_csv(path, index=False)

    return rec


# =============================================================================
# Main
# =============================================================================
def main():
    id_to_raw_id, id_to_label = read_true_metadata_map(DATA_FILE)

    all_records = []

    print("Main experiment directory:")
    print(main_dir)
    print(f"DATA_FILE ID column: {ID_COL}")
    print(f"DATA_FILE label column: {LABEL_COL}")
    print(f"DRY_RUN = {DRY_RUN}")
    print(f"MAKE_BACKUP = {MAKE_BACKUP}")
    print(f"SKIP_IF_ID_ORDER_MISMATCH = {SKIP_IF_ID_ORDER_MISMATCH}")
    print()

    for fraction in fractions:
        for missingness in missingness_types:
            folder = main_dir / f"fraction_{fraction}_missingness_{missingness}"

            if not folder.exists():
                print(f"Skipping missing folder: {folder}")
                continue

            print("\n" + "=" * 100)
            print(f"Checking folder: fraction={fraction}, missingness={missingness}")
            print("=" * 100)

            try:
                reference_meta = get_reference_metadata(
                    folder=folder,
                    id_to_label=id_to_label,
                    columns_num=columns_num,
                )
            except Exception as e:
                print(f"Could not build reference metadata for {folder}: {e}")

                all_records.append(
                    {
                        "folder": str(folder),
                        "fraction": fraction,
                        "missingness": missingness,
                        "status": "error_building_reference_metadata",
                        "error": str(e),
                    }
                )
                continue

            files = collect_files_to_repair(
                folder=folder,
                fraction=fraction,
                missingness=missingness,
                columns_num=columns_num,
            )

            print(f"Files to check or repair: {len(files)}")

            for path in files:
                rec = repair_one_file(
                    path=path,
                    reference_meta=reference_meta,
                    dry_run=DRY_RUN,
                    make_backup=MAKE_BACKUP,
                )

                rec["fraction"] = fraction
                rec["missingness"] = missingness
                rec["folder"] = str(folder)

                all_records.append(rec)

                print(
                    f"{str(rec.get('status')):<40} | "
                    f"missing_id_before={rec.get('n_missing_id_before')} | "
                    f"missing_label_before={rec.get('n_missing_label_before')} | "
                    f"id_mismatch={rec.get('n_id_order_mismatch_nonmissing')} | "
                    f"label_disagree={rec.get('n_label_disagree_before')} | "
                    f"{path.name}"
                )

    report = pd.DataFrame(all_records)

    report_path = main_dir / (
        "eid_label_repair_report_DRYRUN.csv"
        if DRY_RUN
        else "eid_label_repair_report_REPAIRED.csv"
    )

    report.to_csv(report_path, index=False)

    print("\n" + "=" * 100)
    print("Repair report saved to:")
    print(report_path)
    print("=" * 100)

    if not report.empty and "status" in report.columns:
        print("\nStatus counts:")
        print(report["status"].value_counts(dropna=False).to_string())

        print("\nFiles with missing labels before repair:")
        cols_to_show = [
            "fraction",
            "missingness",
            "status",
            "n_missing_id_before",
            "n_missing_label_before",
            "n_id_order_mismatch_nonmissing",
            "n_label_disagree_before",
            "file",
            "path",
        ]
        cols_to_show = [c for c in cols_to_show if c in report.columns]

        if "n_missing_label_before" in report.columns:
            sub = report[
                pd.to_numeric(report["n_missing_label_before"], errors="coerce")
                .fillna(0)
                > 0
            ].copy()
        else:
            sub = pd.DataFrame()

        if sub.empty:
            print("No files had missing labels.")
        else:
            print(sub[cols_to_show].to_string(index=False))

        print("\nFiles skipped due to ID order mismatch:")
        skipped = report[report["status"].astype(str) == "skipped_id_order_mismatch"].copy()
        if skipped.empty:
            print("No files were skipped due to ID order mismatch.")
        else:
            print(skipped[cols_to_show].to_string(index=False))

    if DRY_RUN:
        print("\nThis was a dry run. No files were changed.")
        print("If the report looks safe, set DRY_RUN = False and rerun this script.")
    else:
        print("\nRepair completed.")
        print("Backups were created with suffix '.bak_eid_label' where applicable.")


if __name__ == "__main__":
    main()