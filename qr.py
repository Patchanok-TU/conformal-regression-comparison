"""Ridge-penalized linear quantile regression solved as a convex QP.

The intercept is unpenalized. Features and outcomes are standardized using
only the observations supplied to the fit. No constraints are placed on
coefficient signs, and predictions are not truncated in this module.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import clarabel
import numpy as np
from scipy import sparse


def _prepare(features, y, probability, response_scale):
    x = np.asarray(features, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("features must be a nonempty two-dimensional array")
    target = np.asarray(y, dtype=float)
    if target.ndim != 1 or target.shape[0] != x.shape[0]:
        raise ValueError("y must be a vector with one entry per feature row")
    if not np.isfinite(x).all() or not np.isfinite(target).all():
        raise ValueError("features and y must be finite")
    probability = float(probability)
    if not math.isfinite(probability) or not 0.0 < probability < 1.0:
        raise ValueError("probability must lie strictly between zero and one")

    feature_center = x.mean(axis=0)
    feature_spread = x.std(axis=0, ddof=0)
    feature_spread[feature_spread == 0.0] = 1.0
    response_center = float(np.median(target))
    if response_scale is None:
        response_scale = max(float(target.std(ddof=0)), 1e-10)
    else:
        response_scale = float(response_scale)
        if not math.isfinite(response_scale) or response_scale <= 0.0:
            raise ValueError("response_scale must be finite and strictly positive")

    z = np.column_stack((np.ones(x.shape[0]),
                         (x - feature_center) / feature_spread))
    yn = (target - response_center) / response_scale
    if not np.isfinite(z).all() or not np.isfinite(yn).all():
        raise ValueError("standardization produced nonfinite values")
    n, p = z.shape
    z_sparse = sparse.csc_matrix(z)
    negative_identity = -sparse.eye(n, format="csc")
    a = sparse.vstack([
        sparse.hstack([-probability * z_sparse, negative_identity], format="csc"),
        sparse.hstack([(1.0 - probability) * z_sparse, negative_identity],
                      format="csc"),
    ], format="csc")
    b = np.concatenate([-probability * yn, (1.0 - probability) * yn])
    q = np.concatenate([np.zeros(p), np.full(n, 1.0 / n)])
    metadata = {
        "feature_center": feature_center.tolist(),
        "feature_spread": feature_spread.tolist(),
        "response_center": response_center,
        "response_scale": response_scale,
        "probability": probability,
        "n_training": int(n),
        "n_features": int(x.shape[1]),
    }
    return z, yn, a, b, q, metadata


def _fit_prepared(prepared, lambda_value):
    penalty = float(lambda_value)
    if not math.isfinite(penalty) or penalty < 0.0:
        raise ValueError("lambda_value must be finite and nonnegative")
    z, yn, a, b, q, metadata = prepared
    n, p = z.shape
    # Clarabel uses 0.5 * w' P w + q' w. Only the slopes are penalized.
    diagonal = np.zeros(p + n)
    diagonal[1:p] = penalty
    quadratic = sparse.diags(diagonal, format="csc")
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_iter = 200
    settings.tol_gap_abs = 1e-8
    settings.tol_gap_rel = 1e-8
    settings.tol_feas = 1e-8
    solver = clarabel.DefaultSolver(
        quadratic, q, a, b, [clarabel.NonnegativeConeT(2 * n)], settings
    )
    solution = solver.solve()
    status = str(solution.status)
    objective = float(solution.obj_val)
    dual_objective = float(solution.obj_val_dual)
    primal_residual = float(solution.r_prim)
    dual_residual = float(solution.r_dual)
    gap = abs(objective - dual_objective)
    certificate = max(gap, primal_residual, dual_residual)
    numbers = [objective, dual_objective, primal_residual, dual_residual, gap]
    if not all(math.isfinite(value) for value in numbers):
        raise RuntimeError(f"Quantile regression returned nonfinite diagnostics: {status}")
    if status != "Solved" and not (status == "AlmostSolved" and certificate <= 1e-6):
        raise RuntimeError(
            f"Quantile regression failed: status={status}, gap={gap:.3g}, "
            f"primal_residual={primal_residual:.3g}, dual_residual={dual_residual:.3g}"
        )

    coefficients = np.asarray(solution.x[:p], dtype=float)
    if not np.isfinite(coefficients).all():
        raise RuntimeError("Quantile regression returned nonfinite coefficients")
    residual = yn - z @ coefficients
    tau = metadata["probability"]
    pinball = np.maximum(tau * residual, (tau - 1.0) * residual)
    computed_objective = float(pinball.mean() +
                               0.5 * penalty * (coefficients[1:] @ coefficients[1:]))
    objective_discrepancy = abs(computed_objective - objective)
    if objective_discrepancy > 1e-6 * (1.0 + abs(objective)):
        raise RuntimeError(
            "Quantile regression objective verification failed: "
            f"computed={computed_objective:.12g}, solver={objective:.12g}"
        )
    model = dict(metadata)
    model.update({
        "coefficients": coefficients.tolist(),
        "lambda_value": penalty,
        "diagnostics": {
            "solver": "clarabel",
            "solver_version": clarabel.__version__,
            "status": status,
            "iterations": int(solution.iterations),
            "objective": objective,
            "dual_objective": dual_objective,
            "computed_objective": computed_objective,
            "objective_discrepancy": objective_discrepancy,
            "absolute_objective_gap": gap,
            "primal_residual": primal_residual,
            "dual_residual": dual_residual,
            "solve_time_seconds": float(solution.solve_time),
        },
    })
    return model


def fit_qr(features, y, probability, lambda_value, response_scale=None):
    """Fit mean pinball loss plus lambda/2 times squared standardized slopes.

    ``response_scale`` is an optional positive scale supplied by the caller;
    otherwise the training population standard deviation is used, floored
    at 1e-10. The outcome is centered at its training median in either case.
    The returned model consists entirely of JSON-safe Python values.
    """
    return _fit_prepared(
        _prepare(features, y, probability, response_scale), lambda_value
    )


def fit_path(features, y, probability, lambdas: Iterable[float], response_scale=None):
    """Fit a penalty path, sharing preprocessing and constraint construction."""
    prepared = _prepare(features, y, probability, response_scale)
    result = {}
    for value in lambdas:
        key = float(value)
        if key in result:
            raise ValueError("lambdas must not contain duplicates")
        result[key] = _fit_prepared(prepared, key)
    if not result:
        raise ValueError("lambdas must not be empty")
    return result


def predict_qr(model, features):
    """Return predictions in the original outcome units without truncation."""
    x = np.asarray(features, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[1] != model["n_features"]:
        raise ValueError("features have the wrong number of columns")
    if not np.isfinite(x).all():
        raise ValueError("prediction features must be finite")
    coefficients = np.asarray(model["coefficients"], dtype=float)
    center = np.asarray(model["feature_center"], dtype=float)
    spread = np.asarray(model["feature_spread"], dtype=float)
    prediction = model["response_center"] + model["response_scale"] * (
        coefficients[0] + ((x - center) / spread) @ coefficients[1:]
    )
    if not np.isfinite(prediction).all():
        raise RuntimeError("Quantile regression produced nonfinite predictions")
    return np.asarray(prediction, dtype=float)
