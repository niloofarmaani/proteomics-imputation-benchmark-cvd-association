from .run_bayes_opt import run_bayes_opt


def run_bpca_bayes_opt(**kwargs):
    return run_bayes_opt(method="BPCA", **kwargs)
