"""
Bayesian optimization layer for the imputation benchmark.

This package is intentionally separate from imputation/ and Evaluation/.
It calls existing imputation wrappers, scores a validation-only mask, and
stores Optuna studies plus frozen best-parameter JSON files.
"""
