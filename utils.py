import torch


def log1mexp(x):
    # Computes log(1-exp(-|x|))
    # See https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    x = -x.abs()
    return torch.where(x > -0.693, torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))


# Normalizes between -1 and 1
def range_norm(x: float, min_v: float, max_v: float) -> float:
    assert max_v > min_v
    return 2.0 * (x - min_v) / (max_v - min_v) - 1.0
