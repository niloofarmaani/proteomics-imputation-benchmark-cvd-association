# Optimization/Iterative_bayes_opt.py

from .run_bayes_opt import run_bayes_opt


def run_iterative_bayes_opt(**kwargs):
    return run_bayes_opt(method="Iterative", **kwargs)
