from __future__ import annotations

import argparse
from pathlib import Path

from effitrack_eval.data import prepare_selected_dataset_variants


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the real three-dataset benchmark suite")
    parser.add_argument(
        "--real-source",
        default="real_imbalanced",
        help="Real imbalanced dataset source used for the benchmark suite",
    )
    parser.add_argument(
        "--output-dir",
        default="Data/benchmark_suite",
        help="Directory where generated benchmark files will be written",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generated = prepare_selected_dataset_variants(
        real_source_dataset=args.real_source,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    for name, path in sorted(generated.items()):
        print("{}: {}".format(name, Path(path).resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
