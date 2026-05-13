from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from effitrack_eval.config import DEFAULT_HOLDOUT_SEEDS, available_models, family_for
from effitrack_eval.data import (
    _balance_with_smote,
    find_repo_root,
    load_csv_dataset,
    write_csv_dataset,
    write_json,
    write_summary_csv,
)
from effitrack_eval.runner import (
    _build_evaluation_payload,
    _build_summary_row,
    _load_runtime_overrides,
    _resolve_runtime_config,
    _resolve_selection,
    evaluate_model_run_from_split,
    evaluate_non_classical_model_run_from_split,
)


DEFAULT_MODELS = ["all"]

DEFAULT_SCENARIOS = [
    {
        "name": "train_standard_test_optimized",
        "train_key": "standard",
        "test_key": "optimized",
    },
    {
        "name": "train_optimized_test_standard",
        "train_key": "optimized",
        "test_key": "standard",
    },
]

CLASSICAL_GENERALIZATION_SEEDS = (42, 52, 62, 72, 82)


def _resolve_output_path(path_like: str) -> Path:
    repo_root = find_repo_root()
    path = Path(path_like)
    if not path.is_absolute():
        path = repo_root / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_output_dir(path_like: str) -> Path:
    path = _resolve_output_path(path_like)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _validate_feature_alignment(left_dataset: Dict[str, Any], right_dataset: Dict[str, Any]) -> List[str]:
    left_features = list(left_dataset["feature_names"])
    right_features = list(right_dataset["feature_names"])
    if left_features != right_features:
        raise ValueError(
            "Standard and optimized datasets must expose the same feature columns in the same order."
        )
    return left_features


def _load_generalization_source_datasets() -> Dict[str, Dict[str, Any]]:
    standard = load_csv_dataset("hrss_anomalous_standard")
    optimized = load_csv_dataset("hrss_anomalous_optimized")
    feature_names = _validate_feature_alignment(standard, optimized)
    return {
        "standard": standard,
        "optimized": optimized,
        "feature_names": feature_names,
    }


def _write_dataset_snapshot(
    output_dir: Path,
    scenario_name: str,
    feature_names: Sequence[str],
    train_dataset: Dict[str, Any],
    test_dataset: Dict[str, Any],
    X_train_smote: np.ndarray,
    y_train_smote: np.ndarray,
) -> None:
    snapshot_dir = output_dir / "data" / scenario_name
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    write_csv_dataset(
        path=snapshot_dir / "train_raw.csv",
        feature_names=list(feature_names),
        X=train_dataset["X"],
        y=train_dataset["y"],
        timestamps=train_dataset["timestamps"],
    )
    write_csv_dataset(
        path=snapshot_dir / "train_smote.csv",
        feature_names=list(feature_names),
        X=X_train_smote,
        y=y_train_smote,
    )
    write_csv_dataset(
        path=snapshot_dir / "test_raw.csv",
        feature_names=list(feature_names),
        X=test_dataset["X"],
        y=test_dataset["y"],
        timestamps=test_dataset["timestamps"],
    )


def _evaluate_single_model(
    model_name: str,
    scenario: Dict[str, str],
    feature_names: Sequence[str],
    train_dataset: Dict[str, Any],
    test_dataset: Dict[str, Any],
    seed: int,
    runtime_config: Dict[str, Any] | None,
) -> Dict[str, Any]:
    family = family_for(model_name)
    run_seeds = CLASSICAL_GENERALIZATION_SEEDS if family == "classical_ml" else DEFAULT_HOLDOUT_SEEDS
    run_metrics: List[Dict[str, Any]] = []
    baseline_metrics: List[Dict[str, Any]] = []

    train_samples_after_smote = 0
    positive_rate_after_smote = 0.0

    for run_index, run_seed in enumerate(run_seeds, start=1):
        X_train_smote, y_train_smote = _balance_with_smote(
            train_dataset["X"],
            train_dataset["y"],
            seed=run_seed,
        )
        train_samples_after_smote = int(y_train_smote.shape[0])
        positive_rate_after_smote = float(y_train_smote.mean())

        if family == "classical_ml":
            metrics, baseline = evaluate_model_run_from_split(
                model_name=model_name,
                X_train=X_train_smote,
                y_train=y_train_smote,
                X_test=test_dataset["X"],
                y_test=test_dataset["y"],
                runtime_config=runtime_config,
                seed=run_seed,
            )
            baseline_seed = run_seed + 10_000
        else:
            metrics, baseline = evaluate_non_classical_model_run_from_split(
                model_name=model_name,
                X_train=X_train_smote,
                y_train=y_train_smote,
                X_test=test_dataset["X"],
                y_test=test_dataset["y"],
                runtime_config=runtime_config,
                seed=run_seed,
            )
            baseline_seed = run_seed + 20_000

        metrics["run"] = run_index
        metrics["seed"] = run_seed
        baseline["run"] = run_index
        baseline["seed"] = baseline_seed
        run_metrics.append(metrics)
        baseline_metrics.append(baseline)

    return _build_evaluation_payload(
        dataset_payload={
            "name": scenario["name"],
            "path": "train={} | test={}".format(train_dataset["dataset_path"], test_dataset["dataset_path"]),
            "samples": int(train_dataset["n_samples"] + test_dataset["n_samples"]),
            "features": int(len(feature_names)),
            "train_samples_raw": int(train_dataset["n_samples"]),
            "train_samples_after_smote": train_samples_after_smote,
            "test_samples": int(test_dataset["n_samples"]),
            "train_source": scenario["train_key"],
            "test_source": scenario["test_key"],
        },
        model_name=model_name,
        protocol={
            "strategy": "cross-regime hold-out ({} runs)".format(len(run_seeds)),
            "train_size": 1.0,
            "test_size": 1.0,
            "runs": len(run_seeds),
            "seed": seed,
            "preprocessing": "original dataset loading",
            "smote": "train-only SMOTE",
            "train_source": scenario["train_key"],
            "test_source": scenario["test_key"],
        },
        run_metrics=run_metrics,
        baseline_metrics=baseline_metrics,
        runtime_config=runtime_config,
    )


def run_generalization_suite(
    output_dir: str,
    summary_output: str,
    models: Sequence[str],
    seed: int,
    write_snapshots: bool,
    runtime_overrides: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    output_root = _resolve_output_dir(output_dir)
    summary_path = _resolve_output_path(summary_output)

    datasets = _load_generalization_source_datasets()
    feature_names = list(datasets["feature_names"])
    source_datasets = {
        "standard": datasets["standard"],
        "optimized": datasets["optimized"],
    }

    summary_rows: List[Dict[str, Any]] = []

    for scenario in DEFAULT_SCENARIOS:
        train_dataset = source_datasets[scenario["train_key"]]
        test_dataset = source_datasets[scenario["test_key"]]
        _validate_feature_alignment(train_dataset, test_dataset)
        if write_snapshots:
            snapshot_X_train_smote, snapshot_y_train_smote = _balance_with_smote(
                train_dataset["X"],
                train_dataset["y"],
                seed=seed,
            )
            _write_dataset_snapshot(
                output_root,
                scenario["name"],
                feature_names,
                train_dataset,
                test_dataset,
                snapshot_X_train_smote,
                snapshot_y_train_smote,
            )

        for model_name in models:
            runtime_config = _resolve_runtime_config(model_name, runtime_overrides=runtime_overrides)
            payload = _evaluate_single_model(
                model_name=model_name,
                scenario=scenario,
                feature_names=feature_names,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                seed=seed,
                runtime_config=runtime_config,
            )

            scenario_dir = output_root / scenario["name"]
            write_json(scenario_dir / f"{model_name}.json", payload)

            summary_row = _build_summary_row(payload)
            summary_row["scenario"] = scenario["name"]
            summary_row["train_source"] = scenario["train_key"]
            summary_row["test_source"] = scenario["test_key"]
            summary_row["train_dataset"] = train_dataset["dataset_name"]
            summary_row["test_dataset"] = test_dataset["dataset_name"]
            summary_row["train_samples_raw"] = int(train_dataset["n_samples"])
            summary_row["train_samples_after_smote"] = int(payload["dataset"]["train_samples_after_smote"])
            summary_row["test_samples"] = int(test_dataset["n_samples"])
            summary_row["positive_rate_train_raw"] = float(train_dataset["y"].mean())
            summary_row["positive_rate_train_after_smote"] = 0.5
            summary_row["positive_rate_test"] = float(test_dataset["y"].mean())
            summary_row["feature_count"] = int(len(feature_names))
            summary_row["result_source"] = "generalization_suite"
            summary_rows.append(summary_row)

    summary_rows.sort(key=lambda item: (str(item["dataset"]), str(item["model"])))
    write_summary_csv(output_root / "summary.csv", summary_rows)
    write_summary_csv(summary_path, summary_rows)
    return summary_rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the EffiTrack generalization benchmark suite")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS, help="Model names or 'all'")
    parser.add_argument(
        "--output-dir",
        default="ModelandPerformanceAnalysis/results/experiments/results_generalization_suite",
        help="Directory where per-model JSON reports and the local summary.csv will be written",
    )
    parser.add_argument(
        "--summary-output",
        default="ModelandPerformanceAnalysis/results/summaries/generalization_suite_summary_all.csv",
        help="Summary CSV output path used by the workbook exporter",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split and SMOTE generation")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument(
        "--runtime-config",
        default=None,
        help="Optional JSON file with runtime overrides keyed by model name",
    )
    parser.add_argument(
        "--skip-snapshots",
        action="store_true",
        help="Do not write train/test dataset snapshot CSV files under the output directory",
    )
    args = parser.parse_args(argv)

    if args.list_models:
        for model_name in available_models():
            print(model_name)
        return 0

    models = _resolve_selection(list(args.models), available_models())
    runtime_overrides = _load_runtime_overrides(args.runtime_config)

    summary_rows = run_generalization_suite(
        output_dir=args.output_dir,
        summary_output=args.summary_output,
        models=models,
        seed=args.seed,
        write_snapshots=not args.skip_snapshots,
        runtime_overrides=runtime_overrides,
    )
    print(f"Saved {len(summary_rows)} generalization rows under {_resolve_output_dir(args.output_dir)}")
    print(_resolve_output_path(args.summary_output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
