# Load necessary libraries

suppressPackageStartupMessages(library(glmnet))

# Define the function for LASSO feature selection
lasso_feature_selection <- function(input_file) {

  # Load the data
  data <- read.csv(input_file, header = TRUE)

  # Remove the first column (Patient ID)
  data <- data[, -1]

  # Extract the labels (second column)
  labels <- data[, 1]  # Binary labels (0/1)

  # Extract the features (remaining columns)
  features <- data[, -1]

  # Set a random seed for reproducibility
  set.seed(42)

  # Create indices for the training set
  train_indices <- sample(1:nrow(features), size = 0.7 * nrow(features))

  # Training set
  X_train <- as.matrix(features[train_indices, ])
  Y_train <- as.numeric(labels[train_indices])

  # Test set
  X_test <- as.matrix(features[-train_indices, ])
  Y_test <- as.numeric(labels[-train_indices])

  # Define a sequence of alpha values to test
  alpha_values <- seq(0, 1, by = 0.1)  # From 0 (Ridge) to 1 (LASSO)

  # Initialize variables to store results
  results <- data.frame(alpha = numeric(), lambda = numeric(), auc = numeric())

  # Grid search for alpha and lambda
  for (alpha in alpha_values) {
    # Perform cross-validation for the given alpha
    cv_model <- cv.glmnet(
      x = X_train, 
      y = Y_train, 
      family = "binomial", 
      alpha = alpha,  # Regularization mix (0 = Ridge, 1 = LASSO)
      type.measure = "auc"  # Use AUC as the performance metric
    )
    
    # Store results
    results <- rbind(
      results,
      data.frame(alpha = alpha, lambda = cv_model$lambda.min, auc = max(cv_model$cvm))
    )
  }

  # Find the best alpha and lambda
  best_result <- results[which.max(results$auc), ]
  best_alpha <- best_result$alpha
  best_lambda <- best_result$lambda

  # Fit the final model with the best alpha and lambda
  lasso_model <- glmnet(
    x = X_train, 
    y = Y_train, 
    family = "binomial", 
    alpha = best_alpha, 
    lambda = best_lambda
  )

  # View the coefficients
  coef_lasso <- coef(lasso_model)

  # Convert coefficients to a matrix
  coef_matrix <- as.matrix(coef_lasso)

  # Identify non-zero coefficients
  non_zero_features <- rownames(coef_matrix)[coef_matrix != 0]

  # Exclude the intercept from the count
  selected_features <- non_zero_features[non_zero_features != "(Intercept)"]

  # Return the selected features as a list
  return(selected_features)
}

# Main execution
args <- commandArgs(trailingOnly = TRUE)
input_file <- args[1]
output_file <- args[2]

# Call the function for LASSO feature selection
selected_features <- lasso_feature_selection(input_file)

# Save the selected features to a CSV file
write.csv(data.frame(selected_features = selected_features), file = output_file, row.names = FALSE)

cat("Selected features saved to", output_file, "\n")
