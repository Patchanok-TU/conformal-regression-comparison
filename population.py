"""Population intervals for the constant-variance Gaussian mixture."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq
from scipy.special import ndtr

METHODS = ("Split", "MA", "RMS", "RCP", "CQR")
ALPHA = 0.10


def nodes(n, lower=0.0, upper=1.0):
    x, w = leggauss(n)
    return lower + (upper-lower)*(x+1)/2, (upper-lower)*w/2


def abs_cdf(t, eps):
    t, eps = np.broadcast_arrays(np.asarray(t), np.asarray(eps))
    a = np.sqrt(1+eps**2)*np.maximum(t, 0)
    return np.where(t >= 0, ndtr((a-1)/eps)-ndtr((-a-1)/eps), 0.0)


def targets(x):
    eps = 1-0.95*np.asarray(x)
    ma = (eps*np.sqrt(2/np.pi)*np.exp(-0.5/eps**2)+1-2*ndtr(-1/eps))/np.sqrt(1+eps**2)
    q = np.array([brentq(lambda t: float(abs_cdf(t, e))-0.9, 0, 12, xtol=5e-14) for e in eps])
    return {"Split": np.ones(len(x)), "MA": ma, "RMS": np.ones(len(x)), "RCP": q, "CQR": q}


def excess(h, eps):
    a = h*np.sqrt(1+eps**2)
    z1, z2 = (1-a)/eps, (-1-a)/eps
    value = ((1-a)*ndtr(z1)+eps*np.exp(-z1**2/2)/np.sqrt(2*np.pi)
             +(-1-a)*ndtr(z2)+eps*np.exp(-z2**2/2)/np.sqrt(2*np.pi))/np.sqrt(1+eps**2)
    return np.maximum(value, 0)


def calculate(output):
    output.mkdir(parents=True, exist_ok=True)
    x, w = nodes(512)
    eps = 1-0.95*x
    thresholds, summary = {}, []
    for method, scale in targets(x).items():
        if method == "CQR":
            c = brentq(lambda c: float(w@abs_cdf(scale+c, eps))-0.9,
                       -0.99*float(scale.min()), 12, xtol=5e-14)
            half = scale+c
        else:
            c = brentq(lambda c: float(w@abs_cdf(c*scale, eps))-0.9, 0, 12, xtol=5e-14)
            half = c*scale
        thresholds[method] = c
        summary.append(dict(method=method, alpha=ALPHA, threshold=c,
                            coverage=float(w@abs_cdf(half, eps)), width=float(w@(2*half)),
                            interval_score=float(w@(2*half+2/ALPHA*excess(half, eps)))))
    profile, local = [], []
    x = np.linspace(0, 1, 501)
    for method, scale in targets(x).items():
        c = thresholds[method]
        half = scale+c if method == "CQR" else c*scale
        coverage = abs_cdf(half, 1-0.95*x)
        for xx, ss, hh, cc in zip(x, scale, half, coverage):
            profile.append(dict(x=xx, method=method, alpha=ALPHA, target=ss,
                                halfwidth=hh, conditional_coverage=cc))
    for group in range(5):
        x, w = nodes(256, group/5, (group+1)/5)
        eps = 1-0.95*x
        for method, scale in targets(x).items():
            c = thresholds[method]
            half = scale+c if method == "CQR" else c*scale
            local.append(dict(group=group+1, method=method, alpha=ALPHA,
                              coverage=float(5*w@abs_cdf(half, eps)), width=float(5*w@(2*half)),
                              interval_score=float(5*w@(2*half+2/ALPHA*excess(half, eps)))))
    for name, rows in (("summary", summary), ("profiles", profile), ("local", local)):
        pd.DataFrame(rows).to_csv(output/f"population_{name}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/population"))
    calculate(parser.parse_args().output)
