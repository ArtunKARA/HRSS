from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import DATASET_REGISTRY


def find_repo_root(start: Optional[Path] = None) -> Path:
    current = (start or Path(__file__).resolve()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "README.md").exists() and (candidate / "Data").exists():
            return candidate
    raise FileNotFoundError("Repo root could not be located from the current file path.")


def resolve_dataset_path(dataset_name_or_path: str) -> Tuple[str, Path]:
    repo_root = find_repo_root()
    if dataset_name_or_path in DATASET_REGISTRY:
        spec = DATASET_REGISTRY[dataset_name_or_path]
        return spec.name, repo_root / spec.relative_path

    candidate = Path(dataset_name_or_path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.stem, candidate


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_csv_dataset(dataset_name_or_path: str) -> Dict[str, Any]:
    dataset_name, dataset_path = resolve_dataset_path(dataset_name_or_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with dataset_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if "Labels" not in fieldnames:
            raise ValueError(f"'Labels' column is required in dataset: {dataset_path}")

        feature_names = [column for column in fieldnames if column not in {"Timestamp", "Labels"}]
        features: List[List[float]] = []
        labels: List[int] = []
        timestamps: List[float] = []

        for row in reader:
            if not row:
                continue

            feature_row = [float(row[name]) for name in feature_names]
            features.append(feature_row)
            labels.append(int(float(row["Labels"])))

            timestamp_value = row.get("Timestamp", "")
            if timestamp_value == "":
                timestamps.append(float(len(timestamps)))
            else:
                timestamps.append(float(timestamp_value))

    return {
        "dataset_name": dataset_name,
        "dataset_path": str(dataset_path),
        "feature_names": feature_names,
        "X": np.asarray(features, dtype=np.float64),
        "y": np.asarray(labels, dtype=np.int64),
        "timestamps": np.asarray(timestamps, dtype=np.float64),
        "n_samples": len(labels),
        "n_features": len(feature_names),
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def write_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_directory(path.parent)
    if not rows:
        return

    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_csv_dataset(
    path: Path,
    feature_names: List[str],
    X: np.ndarray,
    y: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
) -> None:
    ensure_directory(path.parent)
    if timestamps is None:
        timestamps = np.arange(X.shape[0], dtype=np.float64)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Timestamp", "Labels", *feature_names])
        for timestamp, label, feature_row in zip(timestamps.tolist(), y.tolist(), X.tolist()):
            writer.writerow([float(timestamp), int(label), *feature_row])


def _balanced_downsample_indices(y: np.ndarray, seed: int) -> np.ndarray:
    unique_labels, counts = np.unique(y, return_counts=True)
    if unique_labels.size < 2:
        return np.arange(y.shape[0], dtype=np.int64)

    rng = np.random.default_rng(seed)
    target_per_class = int(np.min(counts))
    selected_chunks: List[np.ndarray] = []

    for label in unique_labels.tolist():
        label_indices = np.where(y == label)[0]
        if int(label_indices.shape[0]) <= target_per_class:
            selected = np.asarray(label_indices, dtype=np.int64)
        else:
            selected = rng.choice(label_indices, size=target_per_class, replace=False)
            selected = np.asarray(selected, dtype=np.int64)
        selected_chunks.append(selected)

    selected_indices = np.concatenate(selected_chunks)
    rng.shuffle(selected_indices)
    return selected_indices


def _generate_smote_samples(
    minority_features: np.ndarray,
    synthetic_count: int,
    seed: int,
    k_neighbors: int = 5,
) -> np.ndarray:
    if synthetic_count <= 0:
        return np.empty((0, minority_features.shape[1]), dtype=np.float64)

    rng = np.random.default_rng(seed)
    minority_features = np.asarray(minority_features, dtype=np.float64)

    if minority_features.shape[0] == 1:
        noise = rng.normal(loc=0.0, scale=1e-3, size=(synthetic_count, minority_features.shape[1]))
        return minority_features[0:1] + noise

    try:
        from sklearn.neighbors import NearestNeighbors
    except ModuleNotFoundError as exc:
        raise RuntimeError("scikit-learn is required for SMOTE dataset preparation.") from exc

    neighbor_count = min(max(2, k_neighbors + 1), int(minority_features.shape[0]))
    model = NearestNeighbors(n_neighbors=neighbor_count)
    model.fit(minority_features)
    neighbor_indices = model.kneighbors(return_distance=False)

    synthetic_rows: List[np.ndarray] = []
    for _ in range(synthetic_count):
        base_index = int(rng.integers(0, minority_features.shape[0]))
        candidates = [index for index in neighbor_indices[base_index].tolist() if index != base_index]
        if not candidates:
            candidates = [base_index]
        neighbor_index = int(rng.choice(candidates))
        gap = float(rng.random())
        synthetic = minority_features[base_index] + gap * (
            minority_features[neighbor_index] - minority_features[base_index]
        )
        synthetic_rows.append(synthetic.astype(np.float64, copy=False))

    return np.vstack(synthetic_rows)


def _balance_with_smote(X: np.ndarray, y: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    labels, counts = np.unique(y, return_counts=True)
    if labels.size < 2:
        return X.copy(), y.copy()

    majority_label = int(labels[np.argmax(counts)])
    minority_label = int(labels[np.argmin(counts)])
    majority_count = int(np.max(counts))
    minority_count = int(np.min(counts))
    synthetic_count = majority_count - minority_count
    if synthetic_count <= 0:
        return X.copy(), y.copy()

    minority_features = X[y == minority_label]
    synthetic_features = _generate_smote_samples(
        minority_features=minority_features,
        synthetic_count=synthetic_count,
        seed=seed,
    )

    X_balanced = np.vstack([X, synthetic_features])
    y_balanced = np.concatenate(
        [y, np.full(synthetic_count, minority_label, dtype=np.int64)]
    )

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(X_balanced.shape[0])
    return X_balanced[permutation], y_balanced[permutation]


def prepare_selected_dataset_variants(
    real_source_dataset: str = "hrss_anomalous_optimized",
    output_dir: str = "Data/benchmark_suite",
    seed: int = 42,
) -> Dict[str, str]:
    repo_root = find_repo_root()
    target_dir = Path(output_dir)
    if not target_dir.is_absolute():
        target_dir = repo_root / target_dir

    generated_paths: Dict[str, str] = {}
    data = load_csv_dataset(real_source_dataset)

    balanced_indices = _balanced_downsample_indices(
        y=data["y"],
        seed=seed,
    )
    X_balanced_subset = data["X"][balanced_indices]
    y_balanced_subset = data["y"][balanced_indices]
    balanced_path = target_dir / "HRSS_real_downsample_balanced.csv"
    write_csv_dataset(
        path=balanced_path,
        feature_names=data["feature_names"],
        X=X_balanced_subset,
        y=y_balanced_subset,
    )
    generated_paths["real_downsample_balanced"] = str(balanced_path)

    X_smote, y_smote = _balance_with_smote(data["X"], data["y"], seed=seed)
    smote_path = target_dir / "HRSS_real_smote_balanced.csv"
    write_csv_dataset(
        path=smote_path,
        feature_names=data["feature_names"],
        X=X_smote,
        y=y_smote,
    )
    generated_paths["real_smote_balanced"] = str(smote_path)

    return generated_paths
