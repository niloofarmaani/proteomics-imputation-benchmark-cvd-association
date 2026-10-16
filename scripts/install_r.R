# Minimal packages actually used by the nine-method public example.
options(repos = c(CRAN = "https://cloud.r-project.org"))
lib <- Sys.getenv("R_LIBS_USER")
if (nzchar(lib)) {
  dir.create(lib, recursive = TRUE, showWarnings = FALSE)
  .libPaths(c(lib, .libPaths()))
}
cran <- c("Metrics", "missRanger")
missing <- cran[!vapply(cran, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) install.packages(missing, dependencies = NA)
if (!requireNamespace("impute", quietly = TRUE)) {
  if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
  BiocManager::install("impute", ask = FALSE, update = FALSE)
}
stopifnot(all(vapply(c(cran, "impute"), requireNamespace, logical(1), quietly = TRUE)))
sessionInfo()
print(installed.packages()[, c("Package", "Version")])
