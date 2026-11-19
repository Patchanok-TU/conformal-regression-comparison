"""Plot the computed simulation, population and application results."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHODS = ("Split", "MA", "RMS", "RCP", "CQR")
COLORS = dict(zip((*METHODS, "Selected"), ("#666666", "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#000000")))
MARKERS = dict(zip((*METHODS, "Selected"), ("o", "s", "^", "D", "x", "P")))
LABELS = {"constant_gaussian": "Constant Gaussian", "heterogeneous_gaussian": "Varying Gaussian",
          "heterogeneous_t5": r"Varying $t_5$", "varying_shape_unit_variance": "Gaussian mixture"}
SETTINGS = [(s, d) for s in tuple(LABELS)[:3] for d in (1, 5)] + [("varying_shape_unit_variance", 1)]


def primary(frame):
    return frame[frame.learner.ne("quadratic")] if "learner" in frame else frame


def subset(frame, setting):
    scenario, dimension = setting
    return frame[frame.scenario.eq(scenario) & frame.dimension.eq(dimension)]


def coverage(ax, frame, methods=METHODS):
    for offset, method in zip(np.linspace(-0.055, 0.055, len(methods)), methods):
        rows = frame[frame.method.eq(method)].sort_values("group")
        if rows.empty:
            continue
        ax.plot(rows.group+offset, rows.coverage, marker=MARKERS[method], color=COLORS[method], label=method)
        if {"coverage_low", "coverage_high"}.issubset(rows) and np.isfinite(rows.coverage_low).all():
            error = np.vstack((rows.coverage-rows.coverage_low, rows.coverage_high-rows.coverage))
            ax.errorbar(rows.group+offset, rows.coverage, yerr=error, color=COLORS[method], fmt="none", capsize=2)
    ax.axhline(0.9, color="#999999", linestyle="--", linewidth=0.7)
    ax.set(xlabel="Covariate group", ylabel="Coverage")
    ax.grid(axis="y", color="#dddddd", linewidth=0.4)


def save(fig, output, number):
    fig.savefig(output/f"Fig{number}.pdf", bbox_inches="tight")
    plt.close(fig)


def simulation_figures(root, output):
    local = primary(pd.read_csv(root/"simulations/local_summary.csv"))
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.4), constrained_layout=True)
    for ax, setting in zip(axes.flat, (SETTINGS[2], SETTINGS[3], SETTINGS[4], SETTINGS[5])):
        coverage(ax, subset(local, setting))
        ax.set_title(f"{LABELS[setting[0]]}, d={setting[1]}")
    axes[0, 0].legend(fontsize=7)
    save(fig, output, 1)

    paired = pd.read_csv(root/"simulations/paired_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.6), constrained_layout=True)
    for panel, ax in enumerate(axes):
        for shift, method in zip(np.linspace(-0.22, 0.22, 4 if panel == 0 else 2), METHODS[1:] if panel == 0 else ("RCP", "CQR")):
            frame = paired[paired.method.eq(method)]
            if panel == 0:
                frame = frame[frame.baseline.eq("Split") & frame.baseline_learner.eq("fixed") & frame.learner.ne("quadratic")]
            else:
                frame = frame[frame.baseline.eq(method) & frame.baseline_learner.eq("forest") & frame.learner.eq("quadratic")]
            rows = pd.concat([subset(frame, setting) for setting in SETTINGS])
            sign = 1 if panel == 0 else -1
            values = sign*rows.interval_score.to_numpy()
            ax.plot(values, np.arange(len(rows))+shift, linestyle="none", marker=MARKERS[method], color=COLORS[method], label=method)
            lo, hi = rows.interval_score_low.to_numpy(), rows.interval_score_high.to_numpy()
            if np.isfinite(lo).all():
                error = np.vstack((rows.interval_score-lo, hi-rows.interval_score))
                ax.errorbar(values, np.arange(len(rows))+shift, xerr=error if panel == 0 else error[::-1], fmt="none", color=COLORS[method], capsize=2)
        ax.axvline(0, color="#999999", linestyle="--", linewidth=0.7)
        ax.set_yticks(range(7), [f"{LABELS[s]}, d={d}" for s, d in SETTINGS])
        ax.invert_yaxis()
        ax.set(xlabel="Mean interval-score difference", title="Method minus Split" if panel == 0 else "Forest minus quadratic")
        ax.legend(fontsize=7)
    save(fig, output, 2)

    candidates = pd.read_csv(root/"simulations/candidate_summary.csv")
    fig, axes = plt.subplots(4, 2, figsize=(6.4, 8.5), constrained_layout=True)
    for ax, setting in zip(axes.flat, SETTINGS):
        rows = subset(candidates, setting)
        for method in ("MA", "RMS"):
            part = rows[rows.method.eq(method)].sort_values("param")
            ax.plot(part.param, part.interval_score, marker=MARKERS[method], color=COLORS[method], label=method)
        ax.set(xlabel="Number of neighbors", ylabel="Mean interval score", title=f"{LABELS[setting[0]]}, d={setting[1]}")
    axes.flat[-1].axis("off")
    axes.flat[0].legend()
    save(fig, output, 3)

    if (root/"population/population_profiles.csv").exists():
        profiles = pd.read_csv(root/"population/population_profiles.csv")
        fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.5), constrained_layout=True)
        for ax, column, title in ((axes[0, 0], "target", "Population scales and quantile"),
                                  (axes[0, 1], "halfwidth", "Population half-widths"),
                                  (axes[1, 1], "conditional_coverage", "Population conditional coverage")):
            for method in METHODS:
                rows = profiles[profiles.method.eq(method)]
                ax.plot(rows.x, rows[column], color=COLORS[method], label=method)
            ax.set(xlabel="x", title=title)
            ax.legend(fontsize=7)
        coverage(axes[1, 0], subset(local, SETTINGS[-1]))
        axes[1, 0].set_title("Fitted group coverage")
        axes[1, 1].axhline(0.9, color="#999999", linestyle="--")
        save(fig, output, 4)


def additional_figures(root, output):
    if (root/"control/local_summary.csv").exists():
        local = pd.read_csv(root/"control/local_summary.csv")
        fig, axes = plt.subplots(3, 2, figsize=(6.4, 8), constrained_layout=True)
        for row, setting in enumerate((SETTINGS[3], SETTINGS[5], SETTINGS[6])):
            for col, learner in enumerate(("knn", "forest")):
                part = subset(local, setting)
                coverage(axes[row, col], part[part.learner.eq(learner)], ("MA", "RMS", "RCP"))
                axes[row, col].set_title(f"{LABELS[setting[0]]}: {learner}")
        axes[0, 0].legend(fontsize=7)
        save(fig, output, 5)
    if (root/"asymmetric/local_summary.csv").exists():
        fig, ax = plt.subplots(figsize=(6, 3.8), constrained_layout=True)
        coverage(ax, pd.read_csv(root/"asymmetric/local_summary.csv"))
        ax.legend(ncol=5, fontsize=7)
        ax.set_title("Asymmetric Gamma errors")
        save(fig, output, 6)


def application_figure(root, output):
    directory = root/"applications"
    residuals = pd.read_csv(directory/"training_residuals.csv.gz")
    diagnostics = pd.read_csv(directory/"training_diagnostics.csv")
    local = pd.read_csv(directory/"local_summary.csv")
    fig, axes = plt.subplots(3, 2, figsize=(6.4, 8.5), constrained_layout=True)
    for col, dataset in enumerate(("concrete", "fish")):
        points = residuals[residuals.dataset.eq(dataset) & residuals.rep.eq(1)]
        axes[0, col].scatter(points.diagnostic_value, points.signed_residual, s=4, color="#444444")
        axes[0, col].axhline(0, color="#999999", linewidth=0.7)
        if dataset == "concrete":
            axes[0, col].set_xscale("log")
        axes[0, col].set(xlabel="Curing age" if dataset == "concrete" else "Out-of-fold prediction",
                         ylabel="Residual", title=dataset.capitalize())
        part = diagnostics[diagnostics.dataset.eq(dataset) & diagnostics.rep.eq(1)].sort_values("group")
        for field, method in (("MA_relative", "MA"), ("RMS_relative", "RMS"), ("q90_relative", "RCP")):
            axes[1, col].plot(part.group, part[field], marker=MARKERS[method], color=COLORS[method],
                              label="0.90 quantile" if method == "RCP" else method)
        axes[1, col].set(xlabel="Training group", ylabel="Relative value")
        axes[1, col].legend(fontsize=7)
        coverage(axes[2, col], local[local.dataset.eq(dataset)], (*METHODS, "Selected"))
        axes[2, col].set_xticks([1, 2, 3], ["≤7", "8–28", ">28"] if dataset == "concrete" else ["Low", "Middle", "High"])
    axes[2, 0].legend(fontsize=6, ncol=3)
    save(fig, output, 7)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/figures"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "lines.linewidth": 1})
    if (args.input/"simulations/local_summary.csv").exists():
        simulation_figures(args.input, args.output)
    additional_figures(args.input, args.output)
    if (args.input/"applications/local_summary.csv").exists():
        application_figure(args.input, args.output)
    tables = {4: "simulations", 5: "simulations", 6: "control", 7: "asymmetric", 8: "sample_size", 9: "applications"}
    for number, study in tables.items():
        path = args.input/study/"summary.csv"
        if path.exists():
            frame = pd.read_csv(path)
            if number == 4:
                frame = primary(frame)
            elif number == 5:
                frame = frame[frame.method.isin(("RCP", "CQR"))]
            frame.to_csv(args.output/f"Table{number}.csv", index=False)


if __name__ == "__main__":
    main()
