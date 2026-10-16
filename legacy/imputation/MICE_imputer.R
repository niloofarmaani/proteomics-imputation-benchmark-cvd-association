suppressPackageStartupMessages(library(mice))
library(Metrics)
library(caret)  # For findCorrelation

# --------------------------------------------------
# 1) Define MICE Imputation Function
# --------------------------------------------------
mice_imputer <- function(
  df_with_na,        # Data frame with missing values
  original_df,       # Complete (original) data frame
  maxit,             # Number of iterations per MICE chain
  m,                 # Number of multiple imputations
  rmse_path,         # Path to save the RMSE CSV
  imputed_data_path  # Path to save the best completed dataset
) {
  cat("Performing MICE imputation...\n")

  # Separate the first two columns (assumed eid, label)
  non_imputed_cols <- df_with_na[, 1:2]
  df_impute <- df_with_na[, -c(1,2)]
  
  # Prepare original numeric data to match shapes
  original_df_numeric <- original_df[, -c(1,2)]

  # Check for and remove highly correlated columns
  cat("Checking for highly correlated columns...\n")
  cor_matrix <- cor(df_impute, use = "complete.obs")
  highly_correlated <- findCorrelation(cor_matrix, cutoff = 0.9)
  if (length(highly_correlated) > 0) {
    cat("Removing", length(highly_correlated), "highly correlated columns...\n")
    df_impute <- df_impute[, -highly_correlated]
    # IMPORTANT: Apply the same removal to original data to maintain dimensional alignment
    original_df_numeric <- original_df_numeric[, -highly_correlated] 
  }

  # Check for columns with zero variance and remove them
  cat("Checking for columns with zero variance...\n")
  zero_variance_cols <- which(apply(df_impute, 2, var, na.rm = TRUE) == 0)
  if (length(zero_variance_cols) > 0) {
    cat("Removing", length(zero_variance_cols), "columns with zero variance...\n")
    df_impute <- df_impute[, -zero_variance_cols]
    original_df_numeric <- original_df_numeric[, -zero_variance_cols]
  }

  # Debugging: Check the dimensions of the dataset after preprocessing
  cat("Dataset dimensions after preprocessing:", dim(df_impute), "\n")

  # Ensure there are enough columns left for imputation
  if (ncol(df_impute) < 2) {
    stop("Not enough columns left for imputation after preprocessing. Check your dataset.")
  }

  # --- DYNAMIC MASK CALCULATION ---
  # Find exactly where the NA values are in the remaining columns
  missing_indices <- which(is.na(df_impute))

  # Run MICE only on the selected columns
  cat("Running MICE imputation...\n")
  imputed <- mice(
    data      = df_impute,
    m         = as.integer(m),     # Number of multiple imputations
    maxit     = as.integer(maxit), # Number of iterations for each chain
    method    = "pmm",             # Predictive Mean Matching (common for numeric data)
    printFlag = FALSE,
    seed      = 123,
    remove.lindep = TRUE           # Automatically remove linearly dependent columns
  )

  # Initialize a vector to store RMSE values for each imputed dataset
  rmse_values <- numeric(m)

  # Loop over the m completed datasets and compute RMSE
  cat("Calculating RMSE for each imputed dataset...\n")
  for (i in seq_len(m)) {
    completed_data_i <- complete(imputed, i)

    # Ensure the completed data and original data have the same dimensions
    if (ncol(completed_data_i) != ncol(original_df_numeric)) {
      stop("Completed data and original data have different dimensions. Check preprocessing steps.")
    }

    # Calculate RMSE strictly on the cells that were dynamically identified as missing
    rmse_values[i] <- rmse(
      as.numeric(as.matrix(original_df_numeric))[missing_indices],
      as.numeric(as.matrix(completed_data_i))[missing_indices]
    )
  }

  # Find which dataset has the lowest RMSE
  i_best <- which.min(rmse_values)
  best_rmse <- rmse_values[i_best]

  # Extract the "best" imputed dataset
  best_dataset <- complete(imputed, i_best)

  # Reattach the first two columns back to the imputed dataset
  best_dataset <- cbind(non_imputed_cols, best_dataset)

  cat("RMSE values for each imputed dataset:\n", rmse_values, "\n")
  cat("Best dataset is dataset #", i_best, "with RMSE =", best_rmse, "\n")

  # Save the best RMSE in a CSV
  write.csv(
    data.frame(best_rmse = best_rmse),
    file = rmse_path,
    row.names = FALSE
  )

  # Save the best imputed dataset
  write.csv(best_dataset, file = imputed_data_path, row.names = FALSE)

  cat("Best imputed dataset saved to", imputed_data_path, "\n")
}

# --------------------------------------------------
# 2) Main Execution
# --------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)

# We expect 6 arguments now:
# 1) corrupted_data_path  - CSV with missing values
# 2) normalized_data_path - CSV of the original complete data
# 3) maxit                - MICE iterations per chain
# 4) m                    - Number of multiple imputations
# 5) rmse_path            - Where to save the best RMSE file
# 6) imputed_data_path    - Where to save the best completed dataset

corrupted_data_path  <- args[1]
normalized_data_path <- args[2]
maxit                <- args[3]
m                    <- args[4]
rmse_path            <- args[5]
imputed_data_path    <- args[6]

# Read the CSV inputs
df_with_na   <- read.csv(corrupted_data_path)
original_df  <- read.csv(normalized_data_path)

# Perform MICE imputation
mice_imputer(
  df_with_na        = df_with_na,
  original_df       = original_df,
  maxit             = as.integer(maxit),
  m                 = as.integer(m),
  rmse_path         = rmse_path,
  imputed_data_path = imputed_data_path
)