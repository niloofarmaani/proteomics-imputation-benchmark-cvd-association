from .run_bayes_opt import run_bayes_opt


def run_gan_bayes_opt(**kwargs):
    return run_bayes_opt(method="GAN", **kwargs)
