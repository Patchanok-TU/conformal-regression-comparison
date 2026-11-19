"""UCI Concrete and Fish data, licensed under CC BY 4.0.

Yeh (1998), doi:10.24432/C5PK67; Ballabio et al. (2015), doi:10.24432/C5JG7B.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path
import urllib.request
import zipfile

import pandas as pd

DATASETS = {
    "concrete": {
        "url": "https://archive.ics.uci.edu/static/public/165/concrete%2Bcompressive%2Bstrength.zip",
        "file": "Concrete_Data.xls",
        "columns": ["cement", "slag", "fly_ash", "water", "superplasticizer",
                    "coarse_aggregate", "fine_aggregate", "age", "strength_mpa"],
        "group_columns": 7, "offset": 2, "rows": 1005,
    },
    "fish": {
        "url": "https://archive.ics.uci.edu/static/public/504/qsar%2Bfish%2Btoxicity.zip",
        "file": "qsar_fish_toxicity.csv",
        "columns": ["CIC0", "SM1_DzZ", "GATS1i", "NdsCH", "NdssC", "MLOGP",
                    "LC50_neglog_mol_L"],
        "group_columns": 6, "offset": 1, "rows": 907,
    },
}


def raw_file(name, directory):
    spec = DATASETS[name]
    directory = Path(directory)
    path = directory / name / spec["file"]
    if path.exists():
        return path
    for candidate in (directory / spec["file"], directory / "raw" / name / spec["file"]):
        if candidate.exists():
            return candidate
    with urllib.request.urlopen(spec["url"], timeout=60) as response:
        content = response.read()
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        member = next(item for item in archive.namelist() if Path(item).name == spec["file"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(archive.read(member))
    return path


def load_data(name, directory="data"):
    spec = DATASETS[name]
    path = raw_file(name, directory)
    frame = (pd.read_excel(path, engine="xlrd") if name == "concrete"
             else pd.read_csv(path, sep=";", header=None))
    frame.columns = spec["columns"]
    if frame.isna().any().any():
        raise ValueError(f"Missing values in {name} data.")
    frame = frame.drop_duplicates(keep="first").copy()
    frame["source_row"] = frame.index + spec["offset"]
    group_columns = spec["columns"][:spec["group_columns"]]
    frame["split_group"] = frame.groupby(group_columns, sort=True, dropna=False).ngroup()
    if len(frame) != spec["rows"]:
        raise ValueError(f"Unexpected number of {name} observations: {len(frame)}")
    # Use the same CSV conversion as the analysis inputs.
    return pd.read_csv(io.StringIO(frame.to_csv(index=False, lineterminator="\n")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    for dataset in DATASETS:
        frame = load_data(dataset, args.data_dir)
        print(f"{dataset}: {len(frame)} observations, {frame.split_group.nunique()} record groups")
