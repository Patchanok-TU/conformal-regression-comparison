"""MA, RMS and residual quantiles from common forest weights."""

from __future__ import annotations

import hashlib
from time import perf_counter

import numpy as np
import quantile_forest
from quantile_forest import RandomForestQuantileRegressor


REQUIRED_QUANTILE_FOREST_VERSION = "1.4.2"
PREDICT_SETTINGS = {
    "interpolation": "linear",
    "weighted_quantile": True,
    "weighted_leaves": True,
    "aggregate_leaves_first": True,
    "oob_score": False,
}


def _alphas(values):
    values = tuple(float(value) for value in values)
    if not values or len(set(values)) != len(values) or not all(0 < a < 1 for a in values):
        raise ValueError("Alphas must be distinct probabilities strictly between zero and one.")
    return values


class CommonResidualForest:
    """Fit a residual forest once and derive MA, RMS, and RCP from it."""

    def __init__(self, min_samples_leaf=5, n_estimators=200, random_state=0):
        if int(min_samples_leaf) != min_samples_leaf or min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be a positive integer.")
        if int(n_estimators) != n_estimators or n_estimators < 1:
            raise ValueError("n_estimators must be a positive integer.")
        if not isinstance(random_state, (int, np.integer)):
            raise TypeError("An integer random_state is required.")
        self.min_samples_leaf = int(min_samples_leaf)
        self.n_estimators = int(n_estimators)
        self.random_state = int(random_state)

    @property
    def forest_settings(self):
        return {
            "n_estimators": self.n_estimators,
            "criterion": "squared_error",
            "max_depth": None,
            "max_features": 1.0,
            "max_samples_leaf": None,
            "bootstrap": True,
            "n_jobs": 1,
            "random_state": self.random_state,
            "min_samples_leaf": self.min_samples_leaf,
        }

    def fit(self, x, abs_oof_residuals):
        if quantile_forest.__version__ != REQUIRED_QUANTILE_FOREST_VERSION:
            raise RuntimeError("The common-forest control requires quantile-forest 1.4.2.")
        x = np.asarray(x, dtype=float)
        residuals = np.asarray(abs_oof_residuals, dtype=float)
        if x.ndim != 2 or residuals.shape != (len(x),) or len(x) < 2:
            raise ValueError("At least two training covariate rows and matching residuals are required.")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(residuals)):
            raise ValueError("Training arrays must be finite.")
        if np.any(residuals < 0):
            raise ValueError("Absolute out-of-fold residuals must be nonnegative.")

        start = perf_counter()
        self.model_ = RandomForestQuantileRegressor(**self.forest_settings).fit(x, residuals)
        self.fit_seconds_ = perf_counter() - start
        self.n_features_in_ = x.shape[1]
        self.n_train_ = len(x)
        self.floor_ = max(0.01 * float(np.median(residuals)), np.finfo(float).tiny)
        geometry_digest = hashlib.sha256()
        for tree in self.model_.estimators_:
            for values in (tree.tree_.children_left, tree.tree_.children_right,
                           tree.tree_.feature, tree.tree_.threshold):
                geometry_digest.update(np.ascontiguousarray(values).tobytes())
        self.geometry_sha256_ = geometry_digest.hexdigest()
        bootstrap_seeds = np.asarray([tree.random_state for tree in self.model_.estimators_], dtype=np.int64)
        self.bootstrap_seed_sha256_ = hashlib.sha256(bootstrap_seeds.tobytes()).hexdigest()

        start = perf_counter()
        sorted_residuals = np.asarray(self.model_.forest_.y_train, dtype=float)
        occurrence_indices = np.asarray(self.model_.forest_.y_train_leaves)
        if sorted_residuals.shape != (1, len(x)) or occurrence_indices.shape[2] != 1:
            raise RuntimeError("Unexpected single-output quantile-forest storage layout.")
        # Stored indices are 1-based ranks in sorted r; zero denotes padding.
        padded_r = np.concatenate(([0.0], sorted_residuals[0]))
        padded_r_squared = padded_r**2
        leaf_shape = occurrence_indices.shape[:2]
        self.leaf_means_ = np.zeros(leaf_shape, dtype=float)
        self.leaf_second_moments_ = np.zeros(leaf_shape, dtype=float)
        self.leaf_occurrence_counts_ = np.zeros(leaf_shape, dtype=np.int64)
        for tree in range(self.n_estimators):
            indices = occurrence_indices[tree, :, 0, :]
            counts = np.count_nonzero(indices, axis=1)
            self.leaf_occurrence_counts_[tree] = counts
            nonempty = counts > 0
            self.leaf_means_[tree, nonempty] = padded_r[indices[nonempty]].sum(axis=1) / counts[nonempty]
            self.leaf_second_moments_[tree, nonempty] = padded_r_squared[indices[nonempty]].sum(axis=1) / counts[nonempty]
        self.moment_setup_seconds_ = perf_counter() - start
        return self

    def predict_raw(self, query, alphas=(0.10,)):
        """Return unfloored targets as out[alpha][method], sharing query leaves."""
        if not hasattr(self, "model_"):
            raise RuntimeError("Fit the common residual forest before prediction.")
        query = np.asarray(query, dtype=float)
        alphas = _alphas(alphas)
        if query.ndim != 2 or query.shape[1] != self.n_features_in_ or len(query) < 1:
            raise ValueError("Query rows must have the training covariate columns.")
        if not np.all(np.isfinite(query)):
            raise ValueError("Query covariates must be finite.")
        start = perf_counter()
        query_leaves = np.ascontiguousarray(self.model_.apply(query), dtype=np.intp)
        trees = np.arange(self.n_estimators)[None, :]
        if np.any(self.leaf_occurrence_counts_[trees, query_leaves] == 0):
            raise RuntimeError("A query reached a leaf with no retained bootstrap occurrences.")
        ma = self.leaf_means_[trees, query_leaves].mean(axis=1)
        rms = np.sqrt(np.maximum(self.leaf_second_moments_[trees, query_leaves].mean(axis=1), 0.0))
        probabilities = sorted(set(1.0 - alpha for alpha in alphas))
        probability_index = {p: index for index, p in enumerate(probabilities)}
        # This is the same compiled call made by the package's public predict.
        quantiles = np.asarray(self.model_.forest_.predict(
            probabilities, query_leaves, None, b"linear", True, True, True
        )).reshape(len(query), len(probabilities))
        self.predict_seconds_ = perf_counter() - start
        if not np.all(np.isfinite(quantiles)) or np.any(rms + 1e-12 < ma):
            raise RuntimeError("Invalid common-forest targets.")
        return {
            alpha: {
                "MA": ma.copy(),
                "RMS": rms.copy(),
                "RCP": quantiles[:, probability_index[1.0 - alpha]].copy(),
            }
            for alpha in alphas
        }


def fit_common_forests(
    x, abs_residuals, query, alpha, random_state,
    leaf_grid=(5, 10, 20, 40), n_estimators=200,
):
    """Return candidates using the original caller's method/learner/leaf keys.

    ``out[(method, 'forest', leaf)]`` contains ``scale`` and
    ``floor_active`` arrays. Fitting, preprocessing, and residual calculation
    precede this function. The caller tunes candidates and performs conformal
    calibration on separate data. All three targets share one fitted forest
    at each leaf-size candidate.
    """
    alpha = _alphas((alpha,))[0]
    leaves = tuple(int(value) for value in leaf_grid)
    if not leaves or len(set(leaves)) != len(leaves) or min(leaves) < 1:
        raise ValueError("Leaf sizes must be distinct positive integers.")
    candidates = {}
    records = []
    for leaf in leaves:
        fitted = CommonResidualForest(leaf, n_estimators, random_state).fit(x, abs_residuals)
        raw_predictions = fitted.predict_raw(query, (alpha,))[alpha]
        for method, raw in raw_predictions.items():
            candidates[(method, "forest", leaf)] = {
                "scale": np.maximum(raw, fitted.floor_),
                "floor_active": raw < fitted.floor_,
            }
        records.append({
            "leaf": leaf,
            "forest_fit_count": 1,
            "fit_target": "common_absolute_out_of_fold_residuals",
            "fit_seconds": fitted.fit_seconds_,
            "moment_setup_seconds": fitted.moment_setup_seconds_,
            "predict_seconds": fitted.predict_seconds_,
            "training_size": fitted.n_train_,
            "query_size": len(query),
            "shared_geometry_sha256": fitted.geometry_sha256_,
            "shared_bootstrap_seed_sha256": fitted.bootstrap_seed_sha256_,
        })
    diagnostics = {
        "package": "quantile-forest",
        "package_version": quantile_forest.__version__,
        "forest_settings": fitted.forest_settings,
        "prediction_settings": dict(PREDICT_SETTINGS),
        "leaf_grid": list(leaves),
        "alpha": alpha,
        "floor": fitted.floor_,
        "fits": records,
        "forest_fit_count": len(leaves),
        "shared_distribution": "one residual forest per leaf size; same split locations, bootstrap occurrences and query leaves for MA/RMS/RCP",
        "moment_weights": "each bootstrap occurrence in reached leaf b has normalized weight 1/(B*n_b)",
        "quantile_convention": "native quantile-forest 1.4.2 weighted-linear quantile; frequency weights scaled by total reached-leaf occurrences",
    }
    return candidates, diagnostics
