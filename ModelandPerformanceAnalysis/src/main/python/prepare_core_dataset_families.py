from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from effitrack_eval.data import (
    _balance_with_smote,
    _balanced_downsample_indices,
    load_csv_dataset,
    write_csv_dataset,
)


DEFAULT_FAMILIES: Dict[str, Dict[str, str]] = {
    "optimized": {
        "source": "hrss_anomalous_optimized",
        "undersample": "Data/HRSS_undersample_optimized.csv",
        "smote": "Data/HRSS_SMOTE_optimized.csv",
    },
    "standard": {
        "source": "hrss_anomalous_standard",
        "undersample": "Data/HRSS_undersample_standard.csv",
        "smote": "Data/HRSS_SMOTE_standard.csv",
    },
}


def _prepare_family(source_dataset: str, undersample_path: Path, smote_path: Path, seed: int) -> None:
    dataset = load_csv_dataset(source_dataset)
    X = dataset["X"]
    y = dataset["y"]
    feature_names = dataset["feature_names"]

    undersample_indices = _balanced_downsample_indices(y=y, seed=seed)
    write_csv_dataset(
        path=undersample_path,
        feature_names=feature_names,
        X=X[undersample_indices],
        y=y[undersample_indices],
    )

    X_smote, y_smote = _balance_with_smote(X=X, y=y, seed=seed)
    write_csv_dataset(
        path=smote_path,
        feature_names=feature_names,
        X=X_smote,
        y=y_smote,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare optimized and standard dataset families.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    for family_name, spec in DEFAULT_FAMILIES.items():
        undersample_path = Path(spec["undersample"]).resolve()
        smote_path = Path(spec["smote"]).resolve()
        _prepare_family(
            source_dataset=spec["source"],
            undersample_path=undersample_path,
            smote_path=smote_path,
            seed=args.seed,
        )
        print("{} -> {}, {}".format(family_name, undersample_path, smote_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
