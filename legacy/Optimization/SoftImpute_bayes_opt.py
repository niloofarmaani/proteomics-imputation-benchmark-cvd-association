from .run_bayes_opt import run_bayes_opt


def run_softimpute_bayes_opt(**kwargs):
    return run_bayes_opt(method="SoftImpute", **kwargs)
