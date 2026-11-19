"""Conformal regression analyses. Run python main.py --help for options."""
import argparse
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys

VERSIONS = {"numpy": "2.3.5", "scipy": "1.17.0", "pandas": "2.2.3", "clarabel": "0.11.1",
            "scikit-learn": "1.8.0", "quantile-forest": "1.4.2", "matplotlib": "3.10.8", "xlrd": "2.0.1"}
ROOT = Path(__file__).resolve().parent


def execute(script, *arguments):
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MPLBACKEND="Agg")
    subprocess.run([sys.executable, str(ROOT/script), *map(str, arguments)], env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog="Recorded environment: Python 3.12.14; use --versions for package versions.")
    parser.add_argument("study", nargs="?", choices=("simulations", "control", "asymmetric", "sample-size",
                                                   "applications", "population", "figures", "all"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--smoke", action="store_true", help="Run one repetition per setting.")
    parser.add_argument("--partitions", type=Path, default=Path("CSAM_Application_Partitions.csv.gz"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--versions", action="store_true")
    args = parser.parse_args()
    if args.versions:
        for name, recorded in VERSIONS.items():
            try:
                installed = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                installed = "not installed"
            print(f"{name}: {installed} (recorded {recorded})")
        return
    if args.study is None:
        parser.error("Choose an analysis or use --versions.")
    if args.workers < 1 or (args.repetitions is not None and args.repetitions < 1):
        parser.error("Workers and repetitions must be positive.")
    if args.study in ("simulations", "control", "asymmetric", "sample-size", "all"):
        selection = "main" if args.study == "simulations" else args.study
        options = ["--study", selection, "--workers", args.workers, "--output", args.output]
        if args.smoke:
            options.append("--smoke")
        if args.repetitions is not None:
            options.extend(("--repetitions", args.repetitions))
        execute("simulations.py", *options)
    if args.study in ("applications", "all"):
        options = ["--partitions", args.partitions, "--data-dir", args.data_dir,
                   "--output", args.output/"applications", "--workers", args.workers]
        if args.smoke:
            options.append("--smoke")
        if args.repetitions is not None:
            options.extend(("--repetitions", args.repetitions))
        execute("applications.py", *options)
    if args.study in ("population", "all"):
        execute("population.py", "--output", args.output/"population")
    if args.study in ("figures", "all"):
        execute("plots.py", "--input", args.output, "--output", args.output/"figures")


if __name__ == "__main__":
    main()
