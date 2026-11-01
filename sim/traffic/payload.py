from __future__ import annotations
from math import erf, exp, log, sqrt


def mean_payload(model: dict) -> float:
    """Expected bytes after clipping a lognormal draw, before integer rounding."""
    if model["type"] == "fixed":
        return float(model["bytes"])
    if model["type"] != "clipped_lognormal":
        raise ValueError("Unsupported packet-size model")
    sigma = model["sigma"]
    mu = log(model["mean_bytes"]) - sigma**2 / 2
    lo, hi = model["min_bytes"], model["max_bytes"]
    cdf = lambda x: (1 + erf(x / sqrt(2))) / 2
    a, b = (log(lo)-mu)/sigma, (log(hi)-mu)/sigma
    return lo*cdf(a) + exp(mu+sigma**2/2)*(cdf(b-sigma)-cdf(a-sigma)) + hi*(1-cdf(b))
