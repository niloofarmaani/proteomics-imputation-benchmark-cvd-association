# --------------------------------------------------
# ppca_imputer.R
# --------------------------------------------------

# Optional: persistent library path (Colab / Drive)
lib_path <- Sys.getenv("R_LIBS_USER")
if (nzchar(lib_path)) {
  .libPaths(c(lib_path, .libPaths()))
}

suppressPackageStartupMessages(library(Metrics))
suppressPackageStartupMessages(library(pcaMethods))

ppca_imputer <- function(
  df_with_na,        # corrupted data (contains NAs)
  original_df,       # normalized complete data (ground truth for RMSE)
  missing_mask_df,   # mask CSV (may be empty or misaligned)
  nPcs,
  maxSteps,
  rmse_path,
  imputed_data_path
) {
  cat("Initial dimensions inside ppca_imputer:\n")
  cat("  corrupted   :", nrow(df_with_na),   "rows,", ncol(df_with_na),   "cols\n")
  cat("  normalized  :", nrow(original_df),  "rows,", ncol(original_df),  "cols\n")
  if (!is.null(missing_mask_df)) {
    cat("  missingMask :", nrow(missing_mask_df), "rows,", ncol(missing_mask_df), "cols\n")
  } else {
    cat("  missingMask : NULL (will derive mask from corrupted NA pattern)\n")
  }

  # --------------------------------------------------
  # 1) Align columns between corrupted and normalized
  # --------------------------------------------------
  common_cols <- intersect(colnames(df_with_na), colnames(original_df))
  if (length(common_cols) == 0) {
    stop("No common columns between corrupted and normalized data.")
  }
  df_with_na  <- df_with_na[, common_cols, drop = FALSE]
  original_df <- original_df[, common_cols, drop = FALSE]

  # Drop id/label if present (do not impute them)
  drop_cols <- intersect(c("eid", "label"), common_cols)
  keep_cols <- setdiff(common_cols, drop_cols)

  # Keep only numeric columns
  is_num_cor  <- sapply(df_with_na[, keep_cols, drop = FALSE], is.numeric)
  is_num_orig <- sapply(original_df[, keep_cols, drop = FALSE], is.numeric)
  feat_cols <- keep_cols[is_num_cor & is_num_orig]

  cat("Feature columns used for PPCA:", length(feat_cols), "\n")
  if (length(feat_cols) < 2) {
    stop("Not enough numeric feature columns for PPCA after excluding eid/label.")
  }

  X_na   <- as.matrix(df_with_na[,  feat_cols, drop = FALSE])
  X_true <- as.matrix(original_df[, feat_cols, drop = FALSE])
  storage.mode(X_na) <- "double"
  storage.mode(X_true) <- "double"

  # --------------------------------------------------
  # 2) Determine evaluation mask (where RMSE is computed)
  # --------------------------------------------------
  eval_mask <- NULL

  if (!is.null(missing_mask_df) && ncol(missing_mask_df) > 0) {
    # Try to align mask to feature columns
    if (nrow(missing_mask_df) == nrow(df_with_na) &&
        all(feat_cols %in% colnames(missing_mask_df))) {
      eval_mask <- as.matrix(missing_mask_df[, feat_cols, drop = FALSE])
      eval_mask <- as.logical(eval_mask)
      cat("Using provided missing_mask for RMSE.\n")
    } else {
      cat("Mask present but misaligned. Will derive mask from corrupted NA pattern.\n")
    }
  } else {
    cat("Mask missing/empty. Will derive mask from corrupted NA pattern.\n")
  }

  if (is.null(eval_mask)) {
    # Fallback: mask is where corrupted has NA (and original is not NA)
    eval_mask <- is.na(X_na) & !is.na(X_true)
  }

  n_eval <- sum(eval_mask)
  cat("PPCA matrix dims:", nrow(X_na), "x", ncol(X_na), "\n")
  cat("Missing entries in corrupted:", sum(is.na(X_na)), "\n")
  cat("Eval positions (RMSE):", n_eval, "\n")

  if (n_eval == 0) {
    stop("No masked positions found for RMSE. Check that corruption introduced NA values.")
  }

  # --------------------------------------------------
  # 3) Run PPCA (pcaMethods)
  # --------------------------------------------------
  cat("Performing PPCA with nPcs =", nPcs, "and maxSteps =", maxSteps, "...\n")

  ppca_result <- tryCatch(
    {
      pca(
        object   = X_na,
        method   = "ppca",
        nPcs     = as.integer(nPcs),
        maxSteps = as.integer(maxSteps),
        center   = TRUE,
        scale    = "none"
      )
    },
    error = function(e) {
      cat("\nPPCA failed with error:\n")
      cat(e$message, "\n")
      stop(e)
    }
  )

  X_imp <- completeObs(ppca_result)
  X_imp <- as.matrix(X_imp)
  colnames(X_imp) <- feat_cols

  # --------------------------------------------------
  # 4) RMSE on masked positions
  # --------------------------------------------------
  rmse_ppca <- rmse(
    as.numeric(X_true[eval_mask]),
    as.numeric(X_imp[eval_mask])
  )
  cat("RMSE (PPCA):", rmse_ppca, "\n")

  # --------------------------------------------------
  # 5) Save outputs
  # --------------------------------------------------
  write.csv(
    data.frame(rmse_ppca = rmse_ppca),
    file = rmse_path,
    row.names = FALSE
  )

  # Save only the imputed feature matrix (same columns as used in PPCA)
  write.csv(
    X_imp,
    file = imputed_data_path,
    row.names = FALSE
  )

  cat("Saved RMSE to:", rmse_path, "\n")
  cat("Saved imputed data to:", imputed_data_path, "\n")
}

# --------------------------------------------------
# Main Execution (same 7 args as your pipeline)
# --------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)

corrupted_data_path  <- args[1]
normalized_data_path <- args[2]
missing_mask_path    <- args[3]
nPcs                 <- args[4]
maxSteps             <- args[5]
rmse_path            <- args[6]
imputed_data_path    <- args[7]

df_with_na  <- read.csv(corrupted_data_path,  check.names = FALSE)
original_df <- read.csv(normalized_data_path, check.names = FALSE)

# Mask can be empty or malformed, so read it safely
missing_mask_df <- tryCatch(
  read.csv(missing_mask_path, check.names = FALSE),
  error = function(e) NULL
)

cat("Dimensions (before ppca_imputer):\n")
cat("  corrupted   :", nrow(df_with_na),   "rows,", ncol(df_with_na),   "cols\n")
cat("  normalized  :", nrow(original_df),  "rows,", ncol(original_df),  "cols\n")
if (!is.null(missing_mask_df)) {
  cat("  missingMask :", nrow(missing_mask_df), "rows,", ncol(missing_mask_df), "cols\n")
} else {
  cat("  missingMask : NULL (read failed)\n")
}

ppca_imputer(
  df_with_na,
  original_df,
  missing_mask_df,
  nPcs,
  maxSteps,
  rmse_path,
  imputed_data_path
)
