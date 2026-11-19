"""Fit and assess the five interval constructions in Concrete and Fish."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from data import load_data
import methods as base

METHODS = ("Split", "MA", "RMS", "RCP", "CQR")
LEARNERS = {"Split": "fixed", "MA": "knn", "RMS": "knn", "RCP": "forest", "CQR": "forest"}
MASTER, ALPHA = 260930, 0.10


def residual_statistics(absolute, signed, denominator=None):
    values = {
        "MA": float(np.mean(absolute)), "RMS": float(np.sqrt(np.mean(absolute**2))),
        "q90_abs": float(np.quantile(absolute, 0.9, method="linear")),
        "signed_q05": float(np.quantile(signed, 0.05, method="linear")),
        "signed_q50": float(np.quantile(signed, 0.50, method="linear")),
        "signed_q95": float(np.quantile(signed, 0.95, method="linear")),
    }
    denominator = values if denominator is None else denominator
    values.update(MA_relative=values["MA"] / denominator["MA"],
                  RMS_relative=values["RMS"] / denominator["RMS"],
                  q90_relative=values["q90_abs"] / denominator["q90_abs"],
                  q90_over_MA=values["q90_abs"] / values["MA"],
                  q90_over_RMS=values["q90_abs"] / values["RMS"])
    return values


def one_partition(task):
    dataset, rep, data_dir, partitions = task
    data = load_data(dataset, data_dir)
    features = list(data.columns[:-3])
    x, y = data[features].to_numpy(float), data.iloc[:, -3].to_numpy(float)
    groups, source = data.split_group.to_numpy(int), data.source_row.to_numpy(int)
    assignments = pd.read_csv(partitions)
    assignments = assignments[assignments.dataset.eq(dataset) & assignments.rep.eq(rep)].set_index("source_row")
    if assignments.index.has_duplicates or len(assignments) != len(data):
        raise ValueError(f"Incomplete or duplicated partition: {dataset}, {rep}")
    assignments = assignments.loc[source]
    np.testing.assert_array_equal(assignments.group.to_numpy(int), groups)
    roles = assignments.role.to_numpy()
    train, calibration, test = [np.flatnonzero(roles == role) for role in ("train", "calibration", "test")]
    if min(map(len, (train, calibration, test))) == 0 or len(train) + len(calibration) + len(test) != len(data):
        raise ValueError("The partition must contain training, calibration and test observations.")
    role_groups = [set(groups[index]) for index in (train, calibration, test)]
    if any(role_groups[i] & role_groups[j] for i in range(3) for j in range(i)):
        raise ValueError("A record group occurs in more than one sample.")
    ident = 0 if dataset == "concrete" else 1
    tune_seed = np.random.SeedSequence([MASTER, ident, rep, 1])
    choices, _, averaged, _, _, _ = base.select_parameters(
        x[train], y[train], tune_seed, alpha=ALPHA, point_model="rf", groups=groups[train], include_linear=False)
    options = [choices[method, LEARNERS[method]] for method in METHODS]
    chosen = min(options, key=lambda row: (row["loss"], METHODS.index(row["method"])))
    selected_method = chosen["method"]

    fit_seed = np.random.SeedSequence([MASTER, ident, rep, 2])
    candidates, mu, signature, _, _ = base.fit_candidates(
        x[train], y[train], fit_seed, np.vstack((x[calibration], x[test])),
        alpha=ALPHA, point_model="rf", groups=groups[train], include_linear=False)
    cf_seed = np.random.SeedSequence([MASTER, ident, rep, 2]).spawn(3)[0]
    signed, _, folds = base.crossfit_training(x[train], y[train], cf_seed, point_model="rf", groups=groups[train])
    absolute = np.asarray(signature["absolute_oof_residuals"])
    np.testing.assert_allclose(np.abs(signed), absolute, atol=2e-12, rtol=2e-12)
    oof_predictions = y[train] - signed
    if dataset == "concrete":
        diagnostic_values, test_values = x[train, features.index("age")], x[test, features.index("age")]
        cuts, side = np.array([7.0, 28.0]), "left"
    else:
        diagnostic_values, test_values = oof_predictions, mu[len(calibration):]
        cuts, side = np.quantile(oof_predictions, [1 / 3, 2 / 3], method="linear"), "right"
    train_groups = np.searchsorted(cuts, diagnostic_values, side=side) + 1
    test_groups = np.searchsorted(cuts, test_values, side=side) + 1
    common = dict(dataset=dataset, rep=rep, alpha=ALPHA)
    sizes = dict(n_train=len(train), n_cal=len(calibration), n_test=len(test))
    results, local = [], []
    for method in METHODS:
        option = choices[method, LEARNERS[method]]
        lower, upper, _ = base.calibrate_predict(
            candidates[method, LEARNERS[method], option["param"]], mu, y[calibration], len(calibration), ALPHA)
        metrics, local_metrics = base.evaluate(lower, upper, y[test], ALPHA, test_groups)
        labels = (method, "Selected") if method == selected_method else (method,)
        for label in labels:
            row = dict(common, **sizes, method=label, construction=method, learner=LEARNERS[method],
                       param=option["param"], **metrics)
            results.append(row)
            for value in local_metrics:
                group_row = dict(value)
                group_row["n_group_test"] = group_row.pop("n_test")
                local.append(dict(common, **sizes, method=label, construction=method, **group_row))
    global_values = residual_statistics(absolute, signed)
    diagnostics = []
    for group in (1, 2, 3):
        use = train_groups == group
        if not use.any():
            raise ValueError("An application training group is empty.")
        diagnostics.append(dict(common, group=group, n_train_group=int(use.sum()),
                                **residual_statistics(absolute[use], signed[use], global_values)))
    residuals = [dict(common, source_row=int(source[index]), record_group=int(groups[index]),
                      oof_fold=int(folds[j]), oof_prediction=float(oof_predictions[j]),
                      signed_residual=float(signed[j]), absolute_residual=float(absolute[j]),
                      diagnostic_group=int(train_groups[j]), diagnostic_value=float(diagnostic_values[j]))
                 for j, index in enumerate(train)]
    choice = dict(common, method=selected_method, learner=LEARNERS[selected_method],
                  param=chosen["param"], validation_score=chosen["loss"])
    return dict(results=results, local_results=local, choices=[choice],
                method_validation=[dict(common, **row) for row in averaged],
                training_global=[dict(common, **global_values)],
                training_diagnostics=diagnostics, training_residuals=residuals)


def summarize(frame, keys, columns):
    rows = []
    for key, part in frame.groupby(keys, sort=False):
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        row["partitions"] = len(part)
        for name in columns:
            row[name] = float(part[name].mean())
            row[name + "_split_sd"] = float(part[name].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--partitions", type=Path,
                        default=Path(__file__).with_name("CSAM_Application_Partitions.csv.gz"))
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/applications"))
    args = parser.parse_args()
    if not args.partitions.is_file():
        parser.error("Provide the original fixed-partition input with --partitions.")
    if not 1 <= args.repetitions <= 50 or args.workers < 1:
        parser.error("Use 1–50 repetitions and at least one worker.")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("Use a new or empty output directory.")
    for name in ("concrete", "fish"):
        load_data(name, args.data_dir)
    repetitions = 1 if args.smoke else args.repetitions
    tasks = [(name, rep, args.data_dir.resolve(), args.partitions.resolve())
             for name in ("concrete", "fish") for rep in range(1, repetitions + 1)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        chunks = list(pool.map(one_partition, tasks))
    args.output.mkdir(parents=True, exist_ok=True)
    frames = {}
    for name in chunks[0]:
        frames[name] = pd.DataFrame([row for chunk in chunks for row in chunk[name]])
        filename = name + (".csv.gz" if name == "training_residuals" else ".csv")
        frames[name].to_csv(args.output / filename, index=False)
    summarize(frames["results"], ["dataset", "method", "alpha"],
              ["coverage", "width", "interval_score", "worst_group_undercoverage"]).to_csv(args.output / "summary.csv", index=False)
    summarize(frames["local_results"], ["dataset", "method", "alpha", "group"],
              ["coverage", "width", "interval_score", "n_group_test"]).to_csv(args.output / "local_summary.csv", index=False)
    print(f"Application results: {args.output}")


if __name__ == "__main__":
    main()
