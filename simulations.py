#!/usr/bin/env python3
"""Simulation comparisons. Example: python simulations.py --study main --smoke."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

import methods as base
from common_forest import fit_common_forests

MASTER_SEED = 202609301031
ALPHA = 0.10
N_TEST = 3000
SETTINGS = [(s, d) for s in ("constant_gaussian", "heterogeneous_gaussian", "heterogeneous_t5")
            for d in (1, 5)] + [("varying_shape_unit_variance", 1), ("heterogeneous_gamma", 1)]
METRICS = ["coverage", "width", "interval_score", "worst_group_undercoverage", "max_group_absolute_error"]


def generate(rng, n, d, scenario):
    x = rng.uniform(0.0, 1.0, size=(n, d))
    z = 2 * x[:, 0] - 1
    mu = 1 + 2 * z + z * z
    if scenario == "varying_shape_unit_variance":
        eps = 1 - 0.95 * x[:, 0]
        sign = 2 * rng.integers(0, 2, size=n) - 1
        error = (sign + eps * rng.standard_normal(n)) / np.sqrt(1 + eps * eps)
    elif scenario == "heterogeneous_gamma":
        shape = 1 + 4 * x[:, 0]
        error = (rng.gamma(shape=shape, scale=1.0) - shape) / np.sqrt(shape)
        error *= 0.25 + 1.5 * x[:, 0]
    else:
        sigma = np.ones(n) if scenario == "constant_gaussian" else 0.25 + 1.5 * x[:, 0]
        noise = (rng.standard_t(5, size=n) * np.sqrt(3 / 5)
                 if scenario == "heterogeneous_t5" else rng.standard_normal(n))
        error = sigma * noise
    return x, mu + error


def common_candidates(x, y, seed, query):
    cf_seed, point_seed, forest_seed = seed.spawn(3)
    residuals, _, _ = base.crossfit_training(x, y, cf_seed, point_model="ols")
    absolute = np.abs(residuals)
    mu = base.point_predict(base.point_fit(x, y, "ols", int(point_seed.generate_state(1)[0])), query)
    center, spread = x.mean(axis=0), x.std(axis=0, ddof=0)
    spread = np.where(spread > 0, spread, 1.0)
    distances = cdist((query - center) / spread, (x - center) / spread, metric="sqeuclidean")
    nearest = np.argsort(distances, axis=1, kind="stable")[:, :max(base.K_GRID)]
    magnitudes = absolute[nearest]
    sums = np.cumsum(magnitudes, axis=1)
    squares = np.cumsum(magnitudes * magnitudes, axis=1)
    floor = max(0.01 * float(np.median(absolute)), np.finfo(float).tiny)
    candidates = {}
    for k in base.K_GRID:
        estimates = {"MA": sums[:, k - 1] / k, "RMS": np.sqrt(squares[:, k - 1] / k),
                     "RCP": np.quantile(magnitudes[:, :k], 1 - ALPHA, axis=1, method="linear")}
        for method, raw in estimates.items():
            candidates[method, "knn", k] = {"scale": np.maximum(raw, floor), "floor_active": raw < floor}
    forests, _ = fit_common_forests(x, absolute, query, ALPHA, int(forest_seed.generate_state(1)[0]),
                                   leaf_grid=base.LEAF_GRID, n_estimators=200)
    candidates.update(forests)
    return candidates, mu


def common_selection(x, y, seed):
    rows = []
    for repeat, inner_seed in enumerate(seed.spawn(5), 1):
        partition_seed, fit_seed = inner_seed.spawn(2)
        fit, cal, val = base.partition_groups(len(y), partition_seed, fractions=(0.6, 0.2, 0.2))
        candidates, mu = common_candidates(x[fit], y[fit], fit_seed, np.vstack((x[cal], x[val])))
        for (method, learner, param), candidate in candidates.items():
            lower, upper, _ = base.calibrate_predict(candidate, mu, y[cal], len(cal), ALPHA)
            rows.append({"inner_repeat": repeat, "method": method, "learner": learner,
                         "param": param, "loss": float(base.interval_score(lower, upper, y[val], ALPHA).mean())})
    averages = pd.DataFrame(rows).groupby(["learner", "param"], sort=True).loss.mean().reset_index()
    choices = {}
    for learner in ("knn", "forest"):
        best = averages[averages.learner.eq(learner)].sort_values(["loss", "param"], ascending=[True, False]).iloc[0]
        for method in ("MA", "RMS", "RCP"):
            choices[method, learner] = {"param": int(best.param), "loss": float(best.loss)}
    return choices, rows


def one_repetition(task):
    study, setting_index, n_train, rep = task
    scenario, dimension = SETTINGS[setting_index]
    spawn_key = (8, setting_index, n_train, rep) if study == "sample-size" and n_train != 600 else (setting_index, rep)
    outer_seed, inner_seed, final_seed = np.random.SeedSequence(MASTER_SEED, spawn_key=spawn_key).spawn(3)
    rng = np.random.default_rng(outer_seed)
    x, y = generate(rng, n_train, dimension, scenario)
    cx, cy = generate(rng, n_train // 3, dimension, scenario)
    tx, ty = generate(rng, N_TEST, dimension, scenario)
    query = np.vstack((cx, tx))
    if study == "control":
        choices, tuning = common_selection(x, y, inner_seed)
        candidates, mu = common_candidates(x, y, final_seed, query)
    else:
        k_grid = tuple(max(1, int(np.floor(v * n_train / 600 + 0.5))) for v in base.K_GRID)
        leaf_grid = tuple(max(1, int(np.floor(v * n_train / 600 + 0.5))) for v in base.LEAF_GRID)
        include_linear = study == "main" and setting_index < 7
        choices, tuning, _, _, _, _ = base.select_parameters(
            x, y, inner_seed, alpha=ALPHA, point_model="ols", include_linear=include_linear,
            k_values=k_grid, leaf_grid=leaf_grid)
        candidates, mu, _, _, _ = base.fit_candidates(
            x, y, final_seed, query, alpha=ALPHA, point_model="ols", include_linear=include_linear,
            k_values=k_grid, leaf_grid=leaf_grid)
    fields = {"study": study, "scenario": scenario, "dimension": dimension, "n_train": n_train,
              "n_cal": len(cy), "rep": rep, "alpha": ALPHA}
    groups = np.minimum((5 * tx[:, 0]).astype(int), 4) + 1
    selected, local, fixed, parameters = [], [], [], []
    for (method, learner, param), candidate in candidates.items():
        lower, upper, diagnostics = base.calibrate_predict(candidate, mu, cy, len(cy), ALPHA)
        metrics, group_metrics = base.evaluate(lower, upper, ty, ALPHA, groups)
        row = dict(fields, method=method, learner=learner, param=param, **metrics, **diagnostics)
        fixed.append(row)
        if param == choices[method, learner]["param"]:
            selected.append(row)
            local.extend(dict(fields, method=method, learner=learner, param=param, **g) for g in group_metrics)
            parameters.append(dict(fields, method=method, learner=learner, selected_param=param,
                                   validation_score=choices[method, learner]["loss"]))
    return {"results": selected, "local_results": local, "candidate_results": fixed,
            "selected_parameters": parameters, "tuning_results": [dict(fields, **r) for r in tuning]}


def save_results(chunks, output):
    output.mkdir(parents=True, exist_ok=True)
    frames = {}
    for name in ("results", "local_results", "candidate_results", "selected_parameters", "tuning_results"):
        frame = pd.DataFrame([row for chunk in chunks for row in chunk[name]])
        frame.to_csv(output / f"{name}.csv", index=False)
        frames[name] = frame
    by = ["study", "scenario", "dimension", "n_train", "n_cal", "method", "learner", "alpha"]
    base.summarize_frame(frames["results"], by, METRICS).to_csv(output / "summary.csv", index=False)
    base.summarize_frame(frames["local_results"], by + ["group"],
                         ["coverage", "width", "interval_score"]).to_csv(output / "local_summary.csv", index=False)
    base.summarize_frame(frames["candidate_results"], by + ["param"], METRICS).to_csv(output / "candidate_summary.csv", index=False)
    keys = ["study", "scenario", "dimension", "n_train", "n_cal", "rep", "alpha"]
    pairs = []
    rows = frames["results"]
    for _, cell in rows.groupby(["study", "scenario", "dimension", "n_train", "n_cal"], sort=False):
        order = [("Split", "fixed"), ("MA", "knn"), ("RMS", "knn"), ("RCP", "forest"),
                 ("CQR", "forest"), ("RCP", "quadratic"), ("CQR", "quadratic"),
                 ("MA", "forest"), ("RMS", "forest"), ("RCP", "knn")]
        present = set(cell[["method", "learner"]].itertuples(index=False, name=None))
        variants = [variant for variant in order if variant in present]
        for j, (method, learner) in enumerate(variants):
            a = cell[cell.method.eq(method) & cell.learner.eq(learner)].set_index(keys)
            for baseline, baseline_learner in variants[:j]:
                b = cell[cell.method.eq(baseline) & cell.learner.eq(baseline_learner)].set_index(keys)
                pairs.append((a[METRICS] - b[METRICS]).reset_index().assign(
                    method=method, learner=learner, baseline=baseline, baseline_learner=baseline_learner))
    differences = pd.concat(pairs, ignore_index=True)
    differences.to_csv(output / "paired_differences.csv", index=False)
    base.summarize_frame(differences, by + ["baseline", "baseline_learner"], METRICS).to_csv(
        output / "paired_summary.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("main", "control", "asymmetric", "sample-size", "all"), default="main")
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("results/simulation"))
    parser.add_argument("--smoke", action="store_true", help="Run repetition 1 in each selected setting.")
    args = parser.parse_args()
    if args.repetitions < 1 or args.workers < 1:
        parser.error("Repetitions and workers must be positive.")
    if args.output.exists() and not args.output.is_dir():
        parser.error("The output path must be a directory.")
    # Residuals are recomputed; machine-level linear algebra differences can affect tree splits.
    repetitions = 1 if args.smoke else args.repetitions
    studies = ("main", "control", "asymmetric", "sample-size") if args.study == "all" else (args.study,)
    folders = {"main": "simulations", "control": "control", "asymmetric": "asymmetric", "sample-size": "sample_size"}
    for study in studies:
        destination = args.output / folders[study]
        if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
            parser.error(f"Choose a new or empty study directory: {destination}")
    for study in studies:
        settings = range(7) if study == "main" else (7,) if study == "asymmetric" else (3, 5, 6) if study == "control" else (3, 6)
        sizes = (300, 600, 1200) if study == "sample-size" else (600,)
        tasks = [(study, si, n, rep) for si in settings for n in sizes for rep in range(1, repetitions + 1)]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            chunks = list(pool.map(one_repetition, tasks, chunksize=1))
        folder = folders[study]
        save_results(chunks, args.output / folder)
        print(f"{study}: {len(tasks)} repetitions; results in {args.output / folder}", flush=True)


if __name__ == "__main__":
    main()
