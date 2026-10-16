# run_association_preservation_seeded.py

from pathlib import Path
import config_paths

config_paths.set_biobank_root("/home/maani/niloo/imputationproject")
config_paths.set_experiments_name("Experiments-mayovasc-proteomic-ccs")

from Evaluation.association_preservation import (
    run_association_preservation_from_best_params,
    make_association_preservation_report,
)


# =============================================================================
# Main settings
# =============================================================================
DEBUG_RUN = False

PHENOTYPE_FILE = "/scratch/myovasc/cureASciD_01_data.xlsx"
columns_num = "all"
main_dir = Path(config_paths.experiments_dir(columns_num))

# Use the same final seed for every method. Do not choose the seed by performance.
# Use 42 unless you have deliberately generated all final imputed files and
# method-specific seed CSVs for another seed, such as 40.
ASSOCIATION_FINAL_SEED = 42

# Set True only if you have generated actual seeded Mean imputed files.
# Association preservation needs a completed matrix, not only a calculated RMSE.
INCLUDE_MEAN = False

# Reference complete matrix. This file is not seed-specific in your pipeline.
reference_protein_csv = (
    main_dir
    / "fraction_0.1_missingness_MAR"
    / f"normalized_data_col_{columns_num}.csv"
)

covariates = [
    "age",
    "sex",
    "bmi",
    "hyper",
    "diab",
    "dyslip",
    "nic",
    "formernic",
]

# Exact seeded study names. This avoids accidentally selecting old non-seeded
# studies or failed studies with similar names.
exact_study_names = {
    "KNN": "KNN_balanced_MAR_MNAR_10_30_seeded_v1",
    "softImpute": "SoftImpute_balanced_MAR_MNAR_10_30_seeded_v1",
    "VAE": "VAE_balanced_MAR_MNAR_10_30_seeded_v1",
    "DAE": "DAE_balanced_MAR_MNAR_10_30_seeded_v1",
    "AE": "AE_balanced_MAR_MNAR_10_30_seeded_v1",
    "GAIN": "GAIN_balanced_MAR_MNAR_10_30_seeded_v1",
    "Transformer": "Transformer_balanced_MAR_MNAR_10_30_seeded_v2",
    "mf": "MissForest_balanced_MAR_MNAR_10_30_seeded_v1",
    "iterative": "Iterative_balanced_MAR_MNAR_10_30_seeded_v1",
    # Add this only if you include RNN in the association-preservation benchmark:
    # "RNN": "RNN_balanced_MAR_MNAR_10_30_seeded_v1",
}


if DEBUG_RUN:
    out_dir = main_dir / f"association_preservation_seed{ASSOCIATION_FINAL_SEED}_DEBUG"

    methods_to_compare = [
        "KNN",
        "mf",
        "AE",
    ]

    fractions = [0.3]
    missingness_types = ["MNAR"]
    outcomes = ["cvd"]
    combined_pdf_name = f"association_preservation_seed{ASSOCIATION_FINAL_SEED}_DEBUG_report.pdf"

else:
    out_dir = main_dir / f"association_preservation_seed{ASSOCIATION_FINAL_SEED}"

    methods_to_compare = [
        "KNN",
        "softImpute",
        "VAE",
        "DAE",
        "AE",
        "GAIN",
        "Transformer",
        "mf",
        "iterative",
        # Add "RNN" here only after RNN seeded final outputs and evaluator CSV exist.
        # "RNN",
    ]

    if INCLUDE_MEAN:
        methods_to_compare.append("Mean")

    fractions = [0.1, 0.3, 0.5]
    missingness_types = ["MCAR", "MAR", "MNAR"]
    outcomes = ["cvd", "plaque_yn"]
    combined_pdf_name = f"association_preservation_seed{ASSOCIATION_FINAL_SEED}_report.pdf"


# =============================================================================
# Strict preflight checks
# =============================================================================
if not Path(PHENOTYPE_FILE).exists():
    raise FileNotFoundError(f"Phenotype file does not exist: {PHENOTYPE_FILE}")

if not reference_protein_csv.exists():
    raise FileNotFoundError(f"Reference protein CSV does not exist: {reference_protein_csv}")

for method in methods_to_compare:
    if method == "Mean":
        continue
    if method not in exact_study_names:
        raise KeyError(
            f"No exact study name supplied for method={method}. "
            "Add it to exact_study_names or remove the method from methods_to_compare."
        )

print("\nAssociation preservation seeded run")
print("=" * 80)
print(f"Final seed: {ASSOCIATION_FINAL_SEED}")
print(f"Methods: {methods_to_compare}")
print(f"Out dir: {out_dir}")
print("=" * 80)


# =============================================================================
# Run analysis
# =============================================================================
manifest_csv, reference_csv, imputed_csv, summary_csv = run_association_preservation_from_best_params(
    columns_num=columns_num,
    methods_to_compare=methods_to_compare,
    fractions=fractions,
    missingness_types=missingness_types,
    phenotype_xlsx=PHENOTYPE_FILE,
    reference_protein_csv=reference_protein_csv,
    outcomes=outcomes,
    covariates=covariates,

    # Important for the seeded pipeline:
    exact_study_names=exact_study_names,
    final_seed=ASSOCIATION_FINAL_SEED,

    # Use the method-specific seeded evaluator CSVs as the source of truth.
    # This prevents accidental selection of old files or wrong hyperparameters.
    require_seed_result_csv=True,

    # This is only used as fallback when exact_study_names is absent.
    study_name_contains="seeded",

    out_dir=out_dir,
    phenotype_id_col="sid_ext",
    protein_id_col="eid",
    protein_start_col=2,
    sheet_name=0,
    model_type="logistic",
    top_k_values=(20, 50),
    fdr_threshold=0.05,
    min_n=50,

    require_fresh_final_outputs=True,
    allow_stale_with_warning=False,
    exclude_debug=True,
    strict_manifest=True,

    # Keep False when changing seed/methods to avoid cached scans from an older manifest.
    cache_scans=False,
    verbose=True,
)


report = make_association_preservation_report(
    summary_csv=summary_csv,
    metrics=[
        "beta_spearman",
        "z_spearman",
        "sign_concordance_top50_ref",
        "top50_overlap",
        "fdr_jaccard",
    ],
    combined_pdf_name=combined_pdf_name,
    show=False,
)


print("\nOutputs")
print("=" * 80)
print("Manifest:")
print(manifest_csv)
print("Reference association results:")
print(reference_csv)
print("Imputed association results:")
print(imputed_csv)
print("Association preservation summary:")
print(summary_csv)
print("Report:")
print(report["combined_pdf"])
