# run_association_preservation.py

import config_paths
from pathlib import Path

config_paths.set_biobank_root("/home/maani/niloo/imputationproject")
config_paths.set_experiments_name("Experiments-mayovasc-proteomic-ccs")

from Evaluation.association_preservation import (
    run_association_preservation_from_best_params,
    make_association_preservation_report,
)


# =============================================================================
# Debug or full run
# =============================================================================
DEBUG_RUN = False

PHENOTYPE_FILE = "/scratch/myovasc/cureASciD_01_data.xlsx"

columns_num = "all"
main_dir = Path(config_paths.experiments_dir(columns_num))

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


if DEBUG_RUN:
    out_dir = main_dir / "association_preservation_DEBUG"

    methods_to_compare = [
        "mf",
        "GAIN",
        "AE",
    ]

    fractions = [0.3]
    missingness_types = ["MNAR"]

    outcomes = [
        "cvd",
    ]

    combined_pdf_name = "association_preservation_DEBUG_report.pdf"

else:
    out_dir = main_dir / "association_preservation"

    methods_to_compare = [
        "KNN",
        "softImpute",
        "VAE",
        "DAE",
        "AE",
        "mf",
        "GAIN",
        "iterative",
        "Mean",
    ]

    fractions = [0.1, 0.3, 0.5]
    missingness_types = ["MCAR", "MAR", "MNAR"]

    outcomes = [
        "cvd",
        "plaque_yn",
    ]

    combined_pdf_name = "association_preservation_report.pdf"


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
    study_name_contains="balanced",
    out_dir=out_dir,
    phenotype_id_col="sid_ext",
    protein_id_col="eid",
    protein_start_col=2,
    sheet_name=0,

    # Your current selected outcomes are binary yes/no.
    # Use "auto" only later if you mix binary and continuous outcomes.
    model_type="logistic",

    top_k_values=(20, 50),
    fdr_threshold=0.05,
    min_n=50,

    # Reviewer-safe: stop if final outputs are older than best parameter JSONs.
    require_fresh_final_outputs=True,
    allow_stale_with_warning=False,

    exclude_debug=True,
    strict_manifest=True,

    # Important after fixing yes/no outcome mapping:
    # keep False for first debug/full rerun so old cached linear-model scans are not reused.
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


if DEBUG_RUN:
    print("\nDEBUG_RUN=True completed.")
    print("Check that the terminal output says: model: logistic.")
    print("If everything looks correct, set DEBUG_RUN=False and rerun.")
else:
    print("\nFull association preservation run completed.")