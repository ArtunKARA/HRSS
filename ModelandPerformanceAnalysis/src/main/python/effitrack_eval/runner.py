from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import (
    AUTOENCODER_MODELS,
    DEFAULT_HOLDOUT_SEEDS,
    DEFAULT_VALIDATION_SIZE,
    MODEL_METADATA,
    SEQUENCE_INPUT_MODELS,
    TUNING_CANDIDATES,
    available_datasets,
    available_models,
    family_for,
    protocol_for,
)
from .data import (
    find_repo_root,
    load_csv_dataset,
    resolve_dataset_path,
    write_json,
    write_summary_csv,
)
from .metrics import compute_binary_metrics, summarise_metrics
from .models import (
    build_classical_model,
    build_neural_model,
    model_runtime_metadata,
    require_sklearn,
    require_tensorflow,
    resolve_model_metadata,
    set_global_seed,
)
from .statistics import build_random_baseline, exact_wilcoxon_signed_rank


def _require_experiment_dependencies() -> Dict[str, Any]:
    try:
        from sklearn.model_selection import StratifiedKFold, train_test_split
        from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "scikit-learn is required for the experiment runner. "
            "Install ModelandPerformanceAnalysis/requirements.txt first."
        ) from exc

    return {
        "StratifiedKFold": StratifiedKFold,
        "train_test_split": train_test_split,
        "MinMaxScaler": MinMaxScaler,
        "RobustScaler": RobustScaler,
        "StandardScaler": StandardScaler,
    }


def _resolve_output_dir(output_dir: str) -> Path:
    repo_root = find_repo_root()
    path = Path(output_dir)
    if not path.is_absolute():
        path = repo_root / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def _validate_binary_dataset(dataset: Dict[str, Any]) -> None:
    classes = np.unique(dataset["y"])
    if classes.size < 2:
        raise ValueError(
            f"Dataset '{dataset['dataset_name']}' has only one class and cannot be used for binary evaluation."
        )


def _choose_scaler(model_name: str, runtime_config: Optional[Dict[str, Any]] = None) -> Any:
    modules = _require_experiment_dependencies()
    metadata = runtime_config or MODEL_METADATA[model_name]
    scaler_name = str(metadata.get("scaler", MODEL_METADATA[model_name].get("scaler", "standard")))

    if scaler_name == "minmax":
        return modules["MinMaxScaler"]()
    if scaler_name == "robust":
        return modules["RobustScaler"](quantile_range=(10.0, 90.0))
    if scaler_name == "standard":
        return modules["StandardScaler"]()
    return None


def _load_runtime_overrides(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}

    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = find_repo_root() / config_path

    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {str(key): dict(value) for key, value in payload.items()}


def _resolve_runtime_config(
    model_name: str,
    runtime_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    overrides = {}
    if runtime_overrides and model_name in runtime_overrides:
        overrides = runtime_overrides[model_name]
    return resolve_model_metadata(model_name, runtime_overrides=overrides)


def _build_class_weight(y_train: np.ndarray) -> Optional[Dict[int, float]]:
    values, counts = np.unique(y_train, return_counts=True)
    if len(values) < 2:
        return None

    majority = int(np.max(counts))
    minority = int(np.min(counts))
    if majority <= 0 or minority <= 0:
        return None

    imbalance_ratio = float(majority) / float(minority)
    if imbalance_ratio < 1.2:
        return None

    total = float(np.sum(counts))
    weights: Dict[int, float] = {}
    for value, count in zip(values.tolist(), counts.tolist()):
        weights[int(value)] = total / (float(len(values)) * float(count))
    return weights


def _scale_inputs(
    model_name: str,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    runtime_config: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaler = _choose_scaler(model_name, runtime_config=runtime_config)
    if scaler is None:
        return X_train.copy(), X_val.copy(), X_test.copy()

    return (
        scaler.fit_transform(X_train),
        scaler.transform(X_val),
        scaler.transform(X_test),
    )


def _prepare_model_input(
    model_name: str,
    X: np.ndarray,
    runtime_config: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    if model_name in SEQUENCE_INPUT_MODELS:
        metadata = runtime_config or MODEL_METADATA[model_name]
        group_size = int(metadata.get("feature_group_size", 1))
        sequence_layout = str(metadata.get("sequence_layout", "flat_features"))
        if sequence_layout == "flat_vector":
            return X
        if sequence_layout == "sensor_groups" and group_size > 1 and X.shape[1] % group_size == 0:
            return X.reshape((X.shape[0], X.shape[1] // group_size, group_size))
        return X.reshape((X.shape[0], X.shape[1], 1))
    return X


def _select_threshold(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, Dict[str, Any]]:
    best_threshold = 0.5
    best_metrics = compute_binary_metrics(y_true, (scores >= 0.5).astype(int))
    best_signature = (
        best_metrics["f1"],
        best_metrics["precision"],
        best_metrics["recall"],
        -best_metrics["error_rate"],
    )

    for threshold in np.linspace(0.05, 0.95, 37):
        metrics = compute_binary_metrics(y_true, (scores >= threshold).astype(int))
        signature = (
            metrics["f1"],
            metrics["precision"],
            metrics["recall"],
            -metrics["error_rate"],
        )
        if signature > best_signature:
            best_signature = signature
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics


def _select_reconstruction_threshold(
    y_true: np.ndarray,
    errors: np.ndarray,
) -> Tuple[float, Dict[str, Any]]:
    lower = float(errors.min())
    upper = float(errors.max())
    if np.isclose(lower, upper):
        threshold = lower
        return threshold, compute_binary_metrics(y_true, (errors >= threshold).astype(int))

    best_threshold = lower
    best_metrics = compute_binary_metrics(y_true, (errors >= lower).astype(int))
    best_signature = (
        best_metrics["f1"],
        best_metrics["precision"],
        best_metrics["recall"],
        -best_metrics["error_rate"],
    )

    for threshold in np.linspace(lower, upper, 41):
        metrics = compute_binary_metrics(y_true, (errors >= threshold).astype(int))
        signature = (
            metrics["f1"],
            metrics["precision"],
            metrics["recall"],
            -metrics["error_rate"],
        )
        if signature > best_signature:
            best_signature = signature
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics


def _training_callbacks(tf: Any) -> List[Any]:
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=0,
        ),
    ]


def _evaluate_classical_model(
    model_name: str,
    dataset: Dict[str, Any],
    runtime_config: Optional[Dict[str, Any]] = None,
    cv_folds: Optional[int] = None,
) -> Dict[str, Any]:
    _validate_binary_dataset(dataset)
    modules = _require_experiment_dependencies()
    StratifiedKFold = modules["StratifiedKFold"]

    X = dataset["X"]
    y = dataset["y"]
    protocol = dict(protocol_for(model_name))
    if cv_folds is not None:
        protocol["folds"] = int(cv_folds)
        protocol["strategy"] = "{}-fold cross-validation".format(int(cv_folds))
    cross_validator = StratifiedKFold(
        n_splits=int(protocol["folds"]),
        shuffle=True,
        random_state=42,
    )

    fold_metrics: List[Dict[str, Any]] = []
    baseline_metrics: List[Dict[str, Any]] = []

    for fold_index, (train_indices, test_indices) in enumerate(cross_validator.split(X, y), start=1):
        estimator = build_classical_model(model_name, seed=42 + fold_index)
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        estimator.fit(X_train, y_train)
        predictions = np.asarray(estimator.predict(X_test)).astype(int)
        metrics = compute_binary_metrics(y_test, predictions)
        metrics["fold"] = fold_index
        metrics["seed"] = 42 + fold_index
        fold_metrics.append(metrics)

        baseline_predictions = build_random_baseline(
            y_test,
            positive_rate=float(y_train.mean()),
            seed=10_000 + fold_index,
        )
        baseline = compute_binary_metrics(y_test, baseline_predictions)
        baseline["fold"] = fold_index
        baseline_metrics.append(baseline)

    return _build_evaluation_payload(
        dataset_payload={
            "name": dataset["dataset_name"],
            "path": dataset["dataset_path"],
            "samples": dataset["n_samples"],
            "features": dataset["n_features"],
        },
        model_name=model_name,
        protocol=protocol,
        run_metrics=fold_metrics,
        baseline_metrics=baseline_metrics,
        runtime_config=runtime_config,
    )


def _build_evaluation_payload(
    dataset_payload: Dict[str, Any],
    model_name: str,
    protocol: Dict[str, Any],
    run_metrics: List[Dict[str, Any]],
    baseline_metrics: List[Dict[str, Any]],
    runtime_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    summary = summarise_metrics(run_metrics)
    baseline_summary = summarise_metrics(baseline_metrics)
    statistical_test = exact_wilcoxon_signed_rank(
        [float(item["f1"]) for item in run_metrics],
        [float(item["f1"]) for item in baseline_metrics],
    )

    return {
        "dataset": dataset_payload,
        "model": {
            "name": model_name,
            "family": family_for(model_name),
            "metadata": model_runtime_metadata(model_name, runtime_overrides=runtime_config),
        },
        "protocol": protocol,
        "runs": run_metrics,
        "baseline_runs": baseline_metrics,
        "summary": summary,
        "baseline_summary": baseline_summary,
        "statistical_test": statistical_test,
    }


def evaluate_classical_model_holdout_from_split(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    dataset_name: str,
    dataset_path: str,
    runtime_config: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    protocol_overrides: Optional[Dict[str, Any]] = None,
    dataset_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metrics, baseline = evaluate_model_run_from_split(
        model_name=model_name,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        runtime_config=runtime_config,
        seed=seed,
    )
    metrics["run"] = 1
    metrics["seed"] = seed
    baseline["run"] = 1
    baseline["seed"] = seed + 10_000

    dataset_payload = {
        "name": dataset_name,
        "path": dataset_path,
        "samples": int(np.asarray(y_train).shape[0] + np.asarray(y_test).shape[0]),
        "features": int(np.asarray(X_train).shape[1]),
    }
    if dataset_overrides:
        dataset_payload.update(dataset_overrides)

    protocol = {
        "strategy": "hold-out",
        "train_size": float(np.asarray(y_train).shape[0]) / float(np.asarray(y_train).shape[0] + np.asarray(y_test).shape[0]),
        "test_size": float(np.asarray(y_test).shape[0]) / float(np.asarray(y_train).shape[0] + np.asarray(y_test).shape[0]),
        "runs": 1,
        "seed": seed,
    }
    if protocol_overrides:
        protocol.update(protocol_overrides)

    return _build_evaluation_payload(
        dataset_payload=dataset_payload,
        model_name=model_name,
        protocol=protocol,
        run_metrics=[metrics],
        baseline_metrics=[baseline],
        runtime_config=runtime_config,
    )


def evaluate_model_run_from_split(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    runtime_config: Optional[Dict[str, Any]] = None,
    seed: int = 42,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    y_train = np.asarray(y_train).astype(np.int64).reshape(-1)
    y_test = np.asarray(y_test).astype(np.int64).reshape(-1)
    classes = np.unique(np.concatenate([y_train, y_test]))
    if classes.size < 2:
        raise ValueError(
            "The provided split does not contain both classes and cannot be used for binary evaluation."
        )

    estimator = build_classical_model(model_name, seed=seed)
    estimator.fit(np.asarray(X_train), y_train)
    predictions = np.asarray(estimator.predict(np.asarray(X_test))).astype(int)

    metrics = compute_binary_metrics(y_test, predictions)

    baseline_predictions = build_random_baseline(
        y_test,
        positive_rate=float(y_train.mean()),
        seed=seed + 10_000,
    )
    baseline = compute_binary_metrics(y_test, baseline_predictions)
    return metrics, baseline


def evaluate_model_holdout_from_split(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    dataset_name: str,
    dataset_path: str,
    runtime_config: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    protocol_overrides: Optional[Dict[str, Any]] = None,
    dataset_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    family = family_for(model_name)
    if family == "classical_ml":
        return evaluate_classical_model_holdout_from_split(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            runtime_config=runtime_config,
            seed=seed,
            protocol_overrides=protocol_overrides,
            dataset_overrides=dataset_overrides,
        )

    metrics, baseline = evaluate_non_classical_model_run_from_split(
        model_name=model_name,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        runtime_config=runtime_config,
        seed=seed,
    )
    metrics["run"] = 1
    metrics["seed"] = seed
    baseline["run"] = 1
    baseline["seed"] = seed + 20_000

    dataset_payload = {
        "name": dataset_name,
        "path": dataset_path,
        "samples": int(np.asarray(y_train).shape[0] + np.asarray(y_test).shape[0]),
        "features": int(np.asarray(X_train).shape[1]),
    }
    if dataset_overrides:
        dataset_payload.update(dataset_overrides)

    protocol = dict(protocol_for(model_name))
    protocol["runs"] = 1
    protocol["seed"] = seed
    if protocol_overrides:
        protocol.update(protocol_overrides)

    return _build_evaluation_payload(
        dataset_payload=dataset_payload,
        model_name=model_name,
        protocol=protocol,
        run_metrics=[metrics],
        baseline_metrics=[baseline],
        runtime_config=runtime_config,
    )


def evaluate_non_classical_model_run_from_split(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    runtime_config: Optional[Dict[str, Any]] = None,
    seed: int = 42,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    modules = _require_experiment_dependencies()
    train_test_split = modules["train_test_split"]

    metadata = runtime_config or resolve_model_metadata(model_name)
    y_train = np.asarray(y_train).astype(np.int64).reshape(-1)
    y_test = np.asarray(y_test).astype(np.int64).reshape(-1)
    if np.unique(np.concatenate([y_train, y_test])).size < 2:
        raise ValueError(
            "The provided split does not contain both classes and cannot be used for binary evaluation."
        )

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)

    set_global_seed(seed)
    stratify_train = y_train if np.unique(y_train).size > 1 else None
    X_train_inner, X_val, y_train_inner, y_val = train_test_split(
        X_train,
        y_train,
        test_size=DEFAULT_VALIDATION_SIZE,
        stratify=stratify_train,
        random_state=seed,
    )

    X_train_scaled, X_val_scaled, X_test_scaled = _scale_inputs(
        model_name,
        X_train_inner,
        X_val,
        X_test,
        runtime_config=metadata,
    )

    if model_name in AUTOENCODER_MODELS:
        return _evaluate_autoencoder_holdout(
            model_name,
            X_train_scaled,
            y_train_inner,
            X_val_scaled,
            y_val,
            X_test_scaled,
            y_test,
            seed,
            metadata,
        )

    return _evaluate_classifier_holdout(
        model_name,
        X_train_scaled,
        y_train_inner,
        X_val_scaled,
        y_val,
        X_test_scaled,
        y_test,
        seed,
        metadata,
    )


def _evaluate_classifier_holdout(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    runtime_config: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    tf = require_tensorflow()
    metadata = runtime_config

    train_input = _prepare_model_input(model_name, X_train, runtime_config=metadata)
    validation_input = _prepare_model_input(model_name, X_val, runtime_config=metadata)
    test_input = _prepare_model_input(model_name, X_test, runtime_config=metadata)

    fit_kwargs: Dict[str, Any] = {}
    class_weight_mode = str(metadata.get("class_weight_mode", "none"))
    if class_weight_mode == "auto_if_imbalanced":
        class_weight = _build_class_weight(y_train)
        if class_weight:
            fit_kwargs["class_weight"] = class_weight

    model = build_neural_model(model_name, train_input.shape[1:], runtime_overrides=metadata)
    history = model.fit(
        train_input,
        y_train,
        validation_data=(validation_input, y_val),
        epochs=int(metadata["epochs"]),
        batch_size=int(metadata["batch_size"]),
        verbose=0,
        callbacks=_training_callbacks(tf),
        **fit_kwargs,
    )

    validation_scores = np.asarray(model.predict(validation_input, verbose=0)).reshape(-1)
    threshold, validation_metrics = _select_threshold(y_val, validation_scores)
    test_scores = np.asarray(model.predict(test_input, verbose=0)).reshape(-1)
    predictions = (test_scores >= threshold).astype(int)
    metrics = compute_binary_metrics(y_test, predictions)
    metrics["threshold"] = threshold
    metrics["validation_f1"] = float(validation_metrics["f1"])
    metrics["epochs_trained"] = int(len(history.history.get("loss", [])))

    baseline_predictions = build_random_baseline(
        y_test,
        positive_rate=float(y_train.mean()),
        seed=seed + 20_000,
    )
    baseline = compute_binary_metrics(y_test, baseline_predictions)
    return metrics, baseline


def _evaluate_autoencoder_holdout(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    runtime_config: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    tf = require_tensorflow()
    metadata = runtime_config
    model = build_neural_model(model_name, (X_train.shape[1],), runtime_overrides=metadata)

    normal_train = X_train[y_train == 0]
    if normal_train.shape[0] == 0:
        normal_train = X_train

    history = model.fit(
        normal_train,
        normal_train,
        validation_data=(X_val, X_val),
        epochs=int(metadata["epochs"]),
        batch_size=int(metadata["batch_size"]),
        verbose=0,
        callbacks=_training_callbacks(tf),
    )

    val_reconstruction = np.asarray(model.predict(X_val, verbose=0))
    val_errors = np.mean(np.square(X_val - val_reconstruction), axis=1)
    threshold, validation_metrics = _select_reconstruction_threshold(y_val, val_errors)

    test_reconstruction = np.asarray(model.predict(X_test, verbose=0))
    test_errors = np.mean(np.square(X_test - test_reconstruction), axis=1)
    predictions = (test_errors >= threshold).astype(int)
    metrics = compute_binary_metrics(y_test, predictions)
    metrics["threshold"] = threshold
    metrics["validation_f1"] = float(validation_metrics["f1"])
    metrics["epochs_trained"] = int(len(history.history.get("loss", [])))

    baseline_predictions = build_random_baseline(
        y_test,
        positive_rate=float(y_train.mean()),
        seed=seed + 20_000,
    )
    baseline = compute_binary_metrics(y_test, baseline_predictions)
    return metrics, baseline


def _evaluate_holdout_model(
    model_name: str,
    dataset: Dict[str, Any],
    holdout_seeds: List[int],
    runtime_config: Dict[str, Any],
) -> Dict[str, Any]:
    _validate_binary_dataset(dataset)
    modules = _require_experiment_dependencies()
    train_test_split = modules["train_test_split"]

    X = dataset["X"]
    y = dataset["y"]
    protocol = dict(protocol_for(model_name))
    protocol["runs"] = len(holdout_seeds)

    run_metrics: List[Dict[str, Any]] = []
    baseline_metrics: List[Dict[str, Any]] = []

    for run_index, seed in enumerate(holdout_seeds, start=1):
        set_global_seed(seed)

        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X,
            y,
            test_size=float(protocol["test_size"]),
            stratify=y,
            random_state=seed,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full,
            y_train_full,
            test_size=float(protocol["validation_size"]),
            stratify=y_train_full,
            random_state=seed,
        )

        X_train_scaled, X_val_scaled, X_test_scaled = _scale_inputs(
            model_name,
            X_train,
            X_val,
            X_test,
            runtime_config=runtime_config,
        )

        if model_name in AUTOENCODER_MODELS:
            metrics, baseline = _evaluate_autoencoder_holdout(
                model_name,
                X_train_scaled,
                y_train,
                X_val_scaled,
                y_val,
                X_test_scaled,
                y_test,
                seed,
                runtime_config,
            )
        else:
            metrics, baseline = _evaluate_classifier_holdout(
                model_name,
                X_train_scaled,
                y_train,
                X_val_scaled,
                y_val,
                X_test_scaled,
                y_test,
                seed,
                runtime_config,
            )

        metrics["run"] = run_index
        metrics["seed"] = seed
        baseline["run"] = run_index
        baseline["seed"] = seed
        run_metrics.append(metrics)
        baseline_metrics.append(baseline)

    return _build_evaluation_payload(
        dataset_payload={
            "name": dataset["dataset_name"],
            "path": dataset["dataset_path"],
            "samples": dataset["n_samples"],
            "features": dataset["n_features"],
        },
        model_name=model_name,
        protocol=protocol,
        run_metrics=run_metrics,
        baseline_metrics=baseline_metrics,
        runtime_config=runtime_config,
    )


def _build_summary_row(payload: Dict[str, Any]) -> Dict[str, Any]:
    summary = payload["summary"]
    baseline_summary = payload["baseline_summary"]
    statistical_test = payload["statistical_test"]

    return {
        "dataset": payload["dataset"]["name"],
        "model": payload["model"]["name"],
        "family": payload["model"]["family"],
        "evaluation_strategy": payload["protocol"]["strategy"],
        "accuracy_mean": summary["accuracy_mean"],
        "accuracy_std": summary["accuracy_std"],
        "precision_mean": summary["precision_mean"],
        "precision_std": summary["precision_std"],
        "recall_mean": summary["recall_mean"],
        "recall_std": summary["recall_std"],
        "f1_mean": summary["f1_mean"],
        "f1_std": summary["f1_std"],
        "baseline_f1_mean": baseline_summary["f1_mean"],
        "baseline_f1_std": baseline_summary["f1_std"],
        "p_value": statistical_test["p_value"],
        "significant": statistical_test["significant"],
        "direction": statistical_test["direction"],
        "learning_rate": payload["model"]["metadata"]["learning_rate"],
        "batch_size": payload["model"]["metadata"]["batch_size"],
        "epochs": payload["model"]["metadata"]["epochs"],
        "optimizer": payload["model"]["metadata"]["optimizer"],
        "loss_function": payload["model"]["metadata"]["loss_function"],
        "class_weight_mode": payload["model"]["metadata"].get("class_weight_mode", "N/A"),
        "tuning_profile": payload["model"]["metadata"].get("tuning_profile", "N/A"),
        "hardware": payload["model"]["metadata"]["hardware"],
    }


def tune_non_classical_models(
    model_names: List[str],
    dataset_names: List[str],
    output_dir: str,
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    output_root = _resolve_output_dir(output_dir)
    tuning_rows: List[Dict[str, Any]] = []
    selected_overrides: Dict[str, Dict[str, Any]] = {}

    for model_name in model_names:
        if model_name not in TUNING_CANDIDATES:
            continue

        best_candidate: Optional[Dict[str, Any]] = None
        best_signature: Optional[Tuple[float, float]] = None

        for candidate in TUNING_CANDIDATES[model_name]:
            validation_scores: List[float] = []
            test_scores: List[float] = []

            for dataset_name in dataset_names:
                dataset = load_csv_dataset(dataset_name)
                runtime_config = resolve_model_metadata(model_name, runtime_overrides=dict(candidate))
                payload = _evaluate_holdout_model(
                    model_name,
                    dataset,
                    holdout_seeds=[42],
                    runtime_config=runtime_config,
                )
                run_payload = payload["runs"][0]
                validation_scores.append(float(run_payload.get("validation_f1", run_payload["f1"])))
                test_scores.append(float(run_payload["f1"]))

            mean_validation_f1 = float(np.mean(validation_scores))
            mean_test_f1 = float(np.mean(test_scores))
            signature = (mean_validation_f1, mean_test_f1)

            tuning_rows.append(
                {
                    "model": model_name,
                    "tuning_profile": candidate.get("tuning_profile", "candidate"),
                    "learning_rate": candidate.get("learning_rate"),
                    "batch_size": candidate.get("batch_size"),
                    "epochs": candidate.get("epochs"),
                    "mean_validation_f1": mean_validation_f1,
                    "mean_test_f1": mean_test_f1,
                }
            )

            if best_signature is None or signature > best_signature:
                best_signature = signature
                best_candidate = dict(candidate)

        if best_candidate is not None:
            selected_overrides[model_name] = best_candidate

    write_summary_csv(output_root / "tuning_summary.csv", tuning_rows)
    write_json(output_root / "tuned_runtime_config.json", selected_overrides)
    return selected_overrides, tuning_rows


def run_experiments(
    model_names: List[str],
    dataset_names: List[str],
    output_dir: str,
    holdout_runs: int = len(DEFAULT_HOLDOUT_SEEDS),
    runtime_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    cv_folds: Optional[int] = None,
) -> List[Dict[str, Any]]:
    output_root = _resolve_output_dir(output_dir)
    holdout_seeds = [42 + (index * 10) for index in range(holdout_runs)]

    summary_rows: List[Dict[str, Any]] = []

    for dataset_name in dataset_names:
        dataset = load_csv_dataset(dataset_name)

        if np.unique(dataset["y"]).size < 2:
            print(f"Skipping {dataset['dataset_name']}: only one class is available.")
            continue

        for model_name in model_names:
            family = family_for(model_name)
            runtime_config = _resolve_runtime_config(model_name, runtime_overrides=runtime_overrides)
            if family == "classical_ml":
                payload = _evaluate_classical_model(
                    model_name,
                    dataset,
                    runtime_config=runtime_config,
                    cv_folds=cv_folds,
                )
            else:
                payload = _evaluate_holdout_model(
                    model_name,
                    dataset,
                    holdout_seeds=holdout_seeds,
                    runtime_config=runtime_config,
                )

            result_path = output_root / dataset["dataset_name"] / f"{model_name}.json"
            write_json(result_path, payload)
            summary_rows.append(_build_summary_row(payload))

    write_summary_csv(output_root / "summary.csv", summary_rows)
    return summary_rows


def _resolve_selection(requested: List[str], available: List[str]) -> List[str]:
    if not requested or requested == ["all"] or "all" in requested:
        return available

    unknown = []
    for item in requested:
        if item in available:
            continue
        try:
            _, candidate_path = resolve_dataset_path(item)
            if candidate_path.exists():
                continue
        except Exception:
            pass
        unknown.append(item)

    if unknown:
        raise KeyError(f"Unknown selections: {', '.join(sorted(unknown))}")
    return requested


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="EffiTrack experiment runner")
    parser.add_argument("--models", nargs="*", default=["all"], help="Model names or 'all'")
    parser.add_argument("--datasets", nargs="*", default=["hrss_smote_optimized"], help="Dataset names or 'all'")
    parser.add_argument(
        "--output-dir",
        default="ModelandPerformanceAnalysis/results/experiments",
        help="Directory where experiment reports will be stored",
    )
    parser.add_argument("--runs", type=int, default=len(DEFAULT_HOLDOUT_SEEDS), help="Hold-out repetition count")
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=None,
        help="Optional override for classical machine learning cross-validation fold count",
    )
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--list-datasets", action="store_true", help="List available datasets")
    parser.add_argument(
        "--runtime-config",
        default=None,
        help="Optional JSON file with runtime overrides keyed by model name",
    )
    parser.add_argument(
        "--tune-non-classical",
        action="store_true",
        help="Tune deep learning and hybrid runtime parameters before final runs",
    )
    parser.add_argument(
        "--tune-datasets",
        nargs="*",
        default=["hrss_anomalous_optimized"],
        help="Datasets used for non-classical tuning",
    )
    parser.add_argument(
        "--tuning-output-dir",
        default="ModelandPerformanceAnalysis/results/tuning/tuning_results",
        help="Directory where tuning summaries and selected configs are stored",
    )
    args = parser.parse_args(argv)

    if args.list_models:
        for model_name in available_models():
            print(model_name)
        return 0

    if args.list_datasets:
        for dataset_name in available_datasets():
            _, dataset_path = resolve_dataset_path(dataset_name)
            print(f"{dataset_name}: {dataset_path}")
        return 0

    models = _resolve_selection(args.models, available_models())
    datasets = _resolve_selection(args.datasets, available_datasets())
    runtime_overrides = _load_runtime_overrides(args.runtime_config)

    if args.tune_non_classical:
        tuning_models = [model_name for model_name in models if model_name in TUNING_CANDIDATES]
        tuning_datasets = _resolve_selection(args.tune_datasets, available_datasets())
        selected_overrides, tuning_rows = tune_non_classical_models(
            tuning_models,
            tuning_datasets,
            args.tuning_output_dir,
        )
        runtime_overrides.update(selected_overrides)
        print(
            "Saved {} tuning rows under {}".format(
                len(tuning_rows),
                _resolve_output_dir(args.tuning_output_dir),
            )
        )

    summary_rows = run_experiments(
        models,
        datasets,
        args.output_dir,
        holdout_runs=args.runs,
        runtime_overrides=runtime_overrides,
        cv_folds=args.cv_folds,
    )
    print(f"Saved {len(summary_rows)} experiment reports under {_resolve_output_dir(args.output_dir)}")
    return 0
