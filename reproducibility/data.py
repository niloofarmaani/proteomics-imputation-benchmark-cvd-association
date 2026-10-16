"""Download verified public inputs and construct a patient-aligned RPPA cohort."""
from __future__ import annotations
import hashlib
import json
import re
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download_sources(cache):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    lock = json.loads((ROOT / "configs/data_sources.json").read_text())
    paths = {}
    for name, item in lock.items():
        dest = cache / name
        if not dest.exists():
            tmp = dest.with_suffix(dest.suffix + ".part")
            request = urllib.request.Request(item["url"], headers={"User-Agent": "imputation-reproducibility/0.1"})
            with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as stream:
                while block := response.read(1024 * 1024):
                    stream.write(block)
            tmp.replace(dest)
        if sha256(dest) != item["sha256"]:
            raise ValueError(f"Checksum mismatch: {dest}. Review the upstream change before updating the source lock.")
        paths[name] = dest
    return paths


def build_tcga_cohort(paths, *, max_samples=None, max_proteins=None):
    """TCGA-UCEC: primary tumor RPPA, recorded hypertension, one sample/patient.

    No cancer/normal, vital-status, or age surrogate is used as a CVD endpoint.
    Missing outcomes are excluded. Complete cases are selected only after the
    explicit protein panel has been chosen, without looking at associations.
    """
    raw = pd.read_csv(paths["ucec_rppa.tsv"], sep="\t", comment="#")
    sample_cols = [c for c in raw if re.fullmatch(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-01[A-Z0-9]*", c)]
    if not sample_cols:
        raise ValueError("No primary tumor sample columns in the RPPA input.")
    proteins = raw.iloc[:, 0].astype(str)
    if proteins.duplicated().any():
        raise ValueError("Duplicate RPPA antibody identifiers; do not silently merge distinct assays.")
    x = raw[sample_cols].apply(pd.to_numeric, errors="coerce").T
    x.columns = proteins.tolist()
    x = x.sort_index().sort_index(axis=1)
    x.index.name = "eid"
    x = x.replace([np.inf, -np.inf], np.nan)
    all_missing = x.columns[x.isna().all()].tolist()
    constant = x.columns[x.nunique(dropna=True) <= 1].tolist()
    # Different assay batches measured different panels. Avoid fabricated truth:
    # retain the common, fully observed panel before complete-case experiments.
    native_incomplete = x.columns[x.isna().any()].tolist()
    x = x.drop(columns=sorted(set(all_missing + constant + native_incomplete)))
    if max_proteins is not None:
        x = x.iloc[:, :int(max_proteins)]
    n_primary = len(x)
    patients = x.index.to_series().str.slice(0, 12)
    duplicate_samples = int(patients.duplicated().sum())
    x = x.loc[~patients.duplicated()]

    # GDC Biotab includes two descriptive rows after its header.
    clinical = pd.read_csv(paths["ucec_patient.tsv"], sep="\t", dtype=str)
    clinical = clinical[clinical["bcr_patient_barcode"].str.fullmatch(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", na=False)].copy()
    if clinical["bcr_patient_barcode"].duplicated().any():
        raise ValueError("Clinical data contain duplicate patients.")
    clinical = clinical.set_index("bcr_patient_barcode")
    aligned = clinical.reindex(x.index.str.slice(0, 12)).copy()
    aligned.index = x.index
    phenotype_records = [clinical.reset_index()]
    for name, path in paths.items():
        if name.startswith("ucec_followup"):
            followup = pd.read_csv(path, sep="\t", dtype=str)
            followup = followup[followup.bcr_patient_barcode.str.fullmatch(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", na=False)]
            phenotype_records.append(followup)
    records = pd.concat(phenotype_records, ignore_index=True)
    conflicts = {}
    def consistent_recorded_binary(column):
        values = records[column].str.strip().str.lower().map({"yes": 1, "no": 0})
        grouped = pd.DataFrame({"patient": records.bcr_patient_barcode, "value": values}).groupby("patient").value
        unique = grouped.nunique()
        conflicts[column] = int((unique > 1).sum())
        resolved = grouped.first().where(unique <= 1)
        return pd.Series(x.index.str.slice(0, 12).map(resolved), index=x.index)
    y = consistent_recorded_binary("hypertension_diagnosis")
    yraw = aligned["hypertension_diagnosis"].str.strip().str.lower()
    age_column = next((c for c in ["age_at_initial_pathologic_diagnosis", "age_at_diagnosis"] if c in aligned), None)
    if age_column is None:
        raise ValueError("No supported diagnosis-age column.")
    metadata = pd.DataFrame({"hypertension": y, "age": pd.to_numeric(aligned[age_column], errors="coerce")}, index=x.index)
    # Diabetes is a secondary endpoint and an adjustment variable for hypertension.
    metadata["diabetes"] = consistent_recorded_binary("diabetes_diagnosis_indicator")
    if "height_cm_at_diagnosis" in aligned and "weight_kg_at_diagnosis" in aligned:
        h = pd.to_numeric(aligned["height_cm_at_diagnosis"], errors="coerce") / 100
        w = pd.to_numeric(aligned["weight_kg_at_diagnosis"], errors="coerce")
        metadata["bmi"] = (w / h.pow(2)).where((h > 1) & (h < 2.5) & (w > 20) & (w < 400))
    complete = x.notna().all(axis=1)
    eligible = complete & metadata.hypertension.notna()
    audit = {
        "study": "TCGA-UCEC / cBioPortal ucec_tcga Firehose Legacy",
        "assay": "tumor reverse-phase protein array (RPPA); antibody/phosphosite features",
        "endpoint": "GDC recorded hypertension across baseline and follow-up: consistent yes=1/no=0; discordant or unrecorded=missing",
        "primary_tumor_samples": n_primary,
        "duplicate_patient_samples_removed": duplicate_samples,
        "all_missing_or_constant_features_removed": sorted(set(all_missing + constant)),
        "native_incomplete_features_removed": native_incomplete,
        "discordant_patient_records_excluded_per_phenotype": conflicts,
        "selected_proteins": x.shape[1],
        "patients_before_complete_case_selection": len(x),
        "protein_complete_cases": int(complete.sum()),
        "missing_hypertension": int(metadata.hypertension.isna().sum()),
        "eligible_complete_cases": int(eligible.sum()),
        "raw_hypertension_values": yraw.fillna("MISSING").value_counts().to_dict(),
        "native_missing_fraction_selected_panel": float(x.isna().to_numpy().mean()),
    }
    x, metadata = x.loc[eligible], metadata.loc[eligible]
    if max_samples is not None:
        # Deterministic random subset independent of phenotype and protein values.
        selected = x.sample(n=min(len(x), int(max_samples)), random_state=42).index.sort_values()
        x, metadata = x.loc[selected], metadata.loc[selected]
    if len(x) < 50 or metadata.hypertension.nunique() != 2 or metadata.hypertension.value_counts().min() < 10:
        raise ValueError(f"Insufficient eligible patients/classes: n={len(x)}, {metadata.hypertension.value_counts().to_dict()}")
    if x.shape[1] < 3:
        raise ValueError("At least three usable protein features are required.")
    audit["analysis_samples"] = len(x)
    audit["hypertension_counts"] = metadata.hypertension.astype(int).value_counts().to_dict()
    audit["covariate_missing_counts"] = metadata.drop(columns="hypertension").isna().sum().to_dict()
    return x, metadata, audit


def load_private_cohort(proteins_csv, phenotypes_csv, outcome, covariates):
    """Optional MyoVasc/local route, with explicit one-to-one sample alignment."""
    x = pd.read_csv(proteins_csv, index_col="eid").drop(columns=["label"], errors="ignore")
    meta = pd.read_csv(phenotypes_csv, index_col="eid")
    if x.index.has_duplicates or meta.index.has_duplicates:
        raise ValueError("Sample identifiers must be unique in both files.")
    x = x.apply(pd.to_numeric, errors="raise").replace([np.inf, -np.inf], np.nan)
    meta = meta.reindex(x.index)[[outcome, *covariates]]
    y = meta[outcome].replace({"yes": 1, "no": 0, "Yes": 1, "No": 0})
    meta[outcome] = pd.to_numeric(y, errors="coerce")
    if not set(meta[outcome].dropna().unique()).issubset({0, 1}):
        raise ValueError("Use an explicitly verified binary 0/1 outcome.")
    keep = x.notna().all(axis=1) & meta[outcome].notna()
    if keep.sum() < 50 or meta.loc[keep, outcome].nunique() != 2:
        raise ValueError("Too few complete cases or a single outcome class.")
    return x.loc[keep], meta.loc[keep]
