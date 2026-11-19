"""Quantile regression forest baselines with training-only tuning parameters.

Both targets use the same fixed forest specification and bootstrap random state.
All in-bag observations within a reached leaf are retained, with bootstrap
multiplicity; inverse leaf-size weights give each tree equal weight in the
aggregated conditional distribution. No additional leaf subsampling is used.
No calibration or test responses enter this module.

Implementation: quantile-forest 1.4.2, RandomForestQuantileRegressor.
https://zillow.github.io/quantile-forest/generated/quantile_forest.RandomForestQuantileRegressor.html
Method: Meinshausen (2006), Quantile Regression Forests, JMLR 7:983--999.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np
import quantile_forest
from quantile_forest import RandomForestQuantileRegressor


PREDICT_SETTINGS = {
    "interpolation": "linear",
    "weighted_quantile": True,
    "weighted_leaves": True,
    "aggregate_leaves_first": True,
    "oob_score": False,
}


def fit_forests(
    x, y, abs_oof_residuals, query, alphas, random_state,
    leaf_grid=(5, 10, 20, 40), n_estimators=200,
):
    """Return forest candidates indexed by alpha and method/leaf-size tuple.

    ``abs_oof_residuals`` must be the externally computed absolute out-of-fold
    residuals used by the other residual-scale methods. The caller performs
    parameter selection, independent conformal calibration and evaluation.

    Returns
    -------
    candidates_by_alpha : dict
        ``out[alpha][('CQR', 'forest', leaf)]`` contains lower, upper and
        crossed arrays. ``('RCP', 'forest', leaf)`` contains scale and
        floor_active arrays. All arrays follow the order of ``query``.
    diagnostics : dict
        Forest specification, target quantiles, training-derived floor and
        fit/predict timings. Timings are computational diagnostics only.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    residuals = np.asarray(abs_oof_residuals, dtype=float)
    query = np.asarray(query, dtype=float)
    alphas = tuple(float(a) for a in alphas)
    leaves = tuple(int(v) for v in leaf_grid)
    if x.ndim != 2 or query.ndim != 2 or x.shape[1] != query.shape[1]:
        raise ValueError("Training and query covariates must have matching columns.")
    if y.shape != (len(x),) or residuals.shape != (len(x),):
        raise ValueError("Responses and absolute residuals must match training rows.")
    if not all(np.all(np.isfinite(v)) for v in (x, y, residuals, query)):
        raise ValueError("All input arrays must be finite.")
    if len(x) < 2 or len(query) == 0 or np.any(residuals < 0):
        raise ValueError("At least two training rows, query rows and nonnegative residuals are required.")
    if not alphas or len(set(alphas)) != len(alphas) or not all(0 < a < 1 for a in alphas):
        raise ValueError("Alphas must be distinct probabilities strictly between zero and one.")
    if not leaves or len(set(leaves)) != len(leaves) or min(leaves) < 1:
        raise ValueError("Leaf sizes must be distinct positive integers.")
    if not isinstance(random_state, (int, np.integer)):
        raise TypeError("An integer random_state is required for identical bootstrap draws.")
    if int(n_estimators) != n_estimators or n_estimators < 1:
        raise ValueError("n_estimators must be a positive integer.")

    forest_settings = {
        "n_estimators": int(n_estimators),
        "criterion": "squared_error",
        "max_depth": None,
        "max_features": 1.0,
        "max_samples_leaf": None,
        "bootstrap": True,
        "n_jobs": 1,
        "random_state": int(random_state),
    }
    response_quantiles = sorted(set([a / 2 for a in alphas] + [1 - a / 2 for a in alphas]))
    residual_quantiles = sorted(set(1 - a for a in alphas))
    response_index = {q: j for j, q in enumerate(response_quantiles)}
    residual_index = {q: j for j, q in enumerate(residual_quantiles)}
    floor = max(0.01 * float(np.median(residuals)), np.finfo(float).tiny)
    candidates = {a: {} for a in alphas}
    records = []
    for leaf in leaves:
        for method, target, quantiles in (
            ("CQR", y, response_quantiles),
            ("RCP", residuals, residual_quantiles),
        ):
            model = RandomForestQuantileRegressor(min_samples_leaf=leaf, **forest_settings)
            start = perf_counter()
            model.fit(x, target)
            fit_seconds = perf_counter() - start
            start = perf_counter()
            predicted = np.asarray(model.predict(query, quantiles=quantiles, **PREDICT_SETTINGS))
            predict_seconds = perf_counter() - start
            predicted = predicted.reshape(len(query), len(quantiles))
            if not np.all(np.isfinite(predicted)):
                raise RuntimeError("Quantile forest returned nonfinite predictions.")
            if np.any(np.diff(predicted, axis=1) < -1e-12):
                raise RuntimeError("Quantile forest returned nonmonotone quantiles.")
            for alpha in alphas:
                key = (method, "forest", leaf)
                if method == "CQR":
                    lower = predicted[:, response_index[alpha / 2]].copy()
                    upper = predicted[:, response_index[1 - alpha / 2]].copy()
                    candidates[alpha][key] = {
                        "lower": lower, "upper": upper, "crossed": lower > upper,
                    }
                else:
                    raw = predicted[:, residual_index[1 - alpha]].copy()
                    candidates[alpha][key] = {
                        "scale": np.maximum(raw, floor), "floor_active": raw < floor,
                    }
            records.append({
                "method": method, "leaf": leaf,
                "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
                "training_size": len(x), "query_size": len(query),
            })

    diagnostics = {
        "package": "quantile-forest", "package_version": quantile_forest.__version__,
        "forest_settings": forest_settings, "prediction_settings": dict(PREDICT_SETTINGS),
        "leaf_grid": list(leaves), "response_quantiles": response_quantiles,
        "residual_quantiles": residual_quantiles, "floor": floor, "fits": records,
        "leaf_sample_convention": "all in-bag draws with bootstrap multiplicity; inverse leaf-size weights",
    }
    return candidates, diagnostics

