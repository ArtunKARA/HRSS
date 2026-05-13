from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    relative_path: str
    description: str


DEFAULT_CV_FOLDS = 5
DEFAULT_HOLDOUT_TEST_SIZE = 0.2
DEFAULT_VALIDATION_SIZE = 0.2
DEFAULT_HOLDOUT_SEEDS = (42, 52, 62)

DATASET_REGISTRY: Dict[str, DatasetSpec] = {
    "hrss_smote_optimized": DatasetSpec(
        name="hrss_smote_optimized",
        relative_path="Data/HRSS_SMOTE_optimized.csv",
        description="Balanced optimized HRSS dataset created with SMOTE.",
    ),
    "hrss_undersample_optimized": DatasetSpec(
        name="hrss_undersample_optimized",
        relative_path="Data/HRSS_undersample_optimized.csv",
        description="Balanced optimized HRSS dataset created with undersampling.",
    ),
    "hrss_anomalous_optimized": DatasetSpec(
        name="hrss_anomalous_optimized",
        relative_path="Data/HRSS_anomalous_optimized.csv",
        description="Original optimized HRSS dataset with anomaly labels.",
    ),
    "hrss_anomalous_standard": DatasetSpec(
        name="hrss_anomalous_standard",
        relative_path="Data/HRSS_anomalous_standard.csv",
        description="Original standard HRSS dataset with anomaly labels.",
    ),
    "hrss_undersample_standard": DatasetSpec(
        name="hrss_undersample_standard",
        relative_path="Data/HRSS_undersample_standard.csv",
        description="Balanced standard HRSS dataset created with undersampling.",
    ),
    "hrss_normal_optimized": DatasetSpec(
        name="hrss_normal_optimized",
        relative_path="Data/HRSS_normal_optimized.csv",
        description="Optimized HRSS normal-only observations.",
    ),
    "hrss_normal_standard": DatasetSpec(
        name="hrss_normal_standard",
        relative_path="Data/HRSS_normal_standard.csv",
        description="Standard HRSS normal-only observations.",
    ),
    "hrss_smote_standard": DatasetSpec(
        name="hrss_smote_standard",
        relative_path="Data/HRSS_SMOTE_standard.csv",
        description="Balanced standard HRSS dataset created with SMOTE.",
    ),
    "hrss_smote_standard_legacy": DatasetSpec(
        name="hrss_smote_standard_legacy",
        relative_path="Data/HRSS_SMOTE_standard.csv",
        description="Backward-compatible alias that points to the current SMOTE standard dataset path.",
    ),
    "real_reference_full": DatasetSpec(
        name="real_reference_full",
        relative_path="Data/HRSS_anomalous_optimized.csv",
        description="Full real reference dataset used in the real benchmark pack.",
    ),
    "real_imbalanced": DatasetSpec(
        name="real_imbalanced",
        relative_path="Data/HRSS_anomalous_optimized.csv",
        description="Full imbalanced real dataset used in the real benchmark suite.",
    ),
    "real_selected_imbalanced": DatasetSpec(
        name="real_selected_imbalanced",
        relative_path="Data/selected_benchmarks/HRSS_real_selected_imbalanced.csv",
        description="Reduced imbalanced benchmark dataset selected from the real optimized HRSS data.",
    ),
    "real_selected_smote": DatasetSpec(
        name="real_selected_smote",
        relative_path="Data/selected_benchmarks/HRSS_real_selected_smote.csv",
        description="SMOTE-augmented benchmark dataset derived from the selected real HRSS subset.",
    ),
    "real_selected_balanced": DatasetSpec(
        name="real_selected_balanced",
        relative_path="Data/selected_benchmarks/HRSS_real_selected_balanced.csv",
        description="Downsampled class-balanced benchmark dataset derived from the real optimized HRSS data.",
    ),
    "real_downsample_balanced": DatasetSpec(
        name="real_downsample_balanced",
        relative_path="Data/benchmark_suite/HRSS_real_downsample_balanced.csv",
        description="Downsampled class-balanced dataset where normal rows are reduced to match the anomaly count in the real dataset.",
    ),
    "real_smote_balanced": DatasetSpec(
        name="real_smote_balanced",
        relative_path="Data/benchmark_suite/HRSS_real_smote_balanced.csv",
        description="SMOTE-balanced dataset where anomaly rows are increased to match the normal count in the real dataset.",
    ),
}

PROTOCOLS: Dict[str, Dict[str, object]] = {
    "classical_ml": {
        "strategy": "5-fold cross-validation",
        "folds": DEFAULT_CV_FOLDS,
    },
    "deep_learning": {
        "strategy": "hold-out",
        "test_size": DEFAULT_HOLDOUT_TEST_SIZE,
        "validation_size": DEFAULT_VALIDATION_SIZE,
        "runs": len(DEFAULT_HOLDOUT_SEEDS),
    },
    "transformer": {
        "strategy": "hold-out",
        "test_size": DEFAULT_HOLDOUT_TEST_SIZE,
        "validation_size": DEFAULT_VALIDATION_SIZE,
        "runs": len(DEFAULT_HOLDOUT_SEEDS),
    },
    "hybrid": {
        "strategy": "hold-out",
        "test_size": DEFAULT_HOLDOUT_TEST_SIZE,
        "validation_size": DEFAULT_VALIDATION_SIZE,
        "runs": len(DEFAULT_HOLDOUT_SEEDS),
    },
}

MODEL_METADATA: Dict[str, Dict[str, object]] = {
    "logistic_regression": {
        "family": "classical_ml",
        "learning_rate": "N/A",
        "batch_size": "N/A",
        "epochs": "N/A",
        "optimizer": "N/A",
        "loss_function": "log_loss",
        "scaler": "standard",
    },
    "random_forest": {
        "family": "classical_ml",
        "learning_rate": "N/A",
        "batch_size": "N/A",
        "epochs": "N/A",
        "optimizer": "N/A",
        "loss_function": "gini",
        "scaler": "none",
    },
    "svm": {
        "family": "classical_ml",
        "learning_rate": "N/A",
        "batch_size": "N/A",
        "epochs": "N/A",
        "optimizer": "N/A",
        "loss_function": "hinge",
        "scaler": "standard",
    },
    "decision_tree": {
        "family": "classical_ml",
        "learning_rate": "N/A",
        "batch_size": "N/A",
        "epochs": "N/A",
        "optimizer": "N/A",
        "loss_function": "gini",
        "scaler": "none",
    },
    "knn": {
        "family": "classical_ml",
        "learning_rate": "N/A",
        "batch_size": "N/A",
        "epochs": "N/A",
        "optimizer": "N/A",
        "loss_function": "distance_vote",
        "scaler": "standard",
    },
    "naive_bayes": {
        "family": "classical_ml",
        "learning_rate": "N/A",
        "batch_size": "N/A",
        "epochs": "N/A",
        "optimizer": "N/A",
        "loss_function": "gaussian_likelihood",
        "scaler": "standard",
    },
    "cnn": {
        "family": "deep_learning",
        "learning_rate": 0.0003,
        "batch_size": 64,
        "epochs": 120,
        "optimizer": "Adam",
        "loss_function": "binary_crossentropy",
        "scaler": "robust",
        "class_weight_mode": "auto_if_imbalanced",
        "tuning_profile": "data_driven_v1",
        "sequence_layout": "sensor_groups",
        "feature_group_size": 3,
    },
    "rnn": {
        "family": "deep_learning",
        "learning_rate": 0.0004,
        "batch_size": 64,
        "epochs": 100,
        "optimizer": "Adam",
        "loss_function": "binary_crossentropy",
        "scaler": "robust",
        "class_weight_mode": "auto_if_imbalanced",
        "tuning_profile": "data_driven_v1",
        "sequence_layout": "sensor_groups",
        "feature_group_size": 3,
    },
    "lstm": {
        "family": "deep_learning",
        "learning_rate": 0.0003,
        "batch_size": 64,
        "epochs": 120,
        "optimizer": "Adam",
        "loss_function": "binary_crossentropy",
        "scaler": "robust",
        "class_weight_mode": "auto_if_imbalanced",
        "tuning_profile": "data_driven_v1",
        "sequence_layout": "sensor_groups",
        "feature_group_size": 3,
    },
    "gru": {
        "family": "deep_learning",
        "learning_rate": 0.0003,
        "batch_size": 64,
        "epochs": 120,
        "optimizer": "Adam",
        "loss_function": "binary_crossentropy",
        "scaler": "robust",
        "class_weight_mode": "auto_if_imbalanced",
        "tuning_profile": "data_driven_v1",
        "sequence_layout": "sensor_groups",
        "feature_group_size": 3,
    },
    "autoencoder": {
        "family": "deep_learning",
        "learning_rate": 0.0005,
        "batch_size": 64,
        "epochs": 80,
        "optimizer": "Adam",
        "loss_function": "mse",
        "scaler": "minmax",
        "class_weight_mode": "normal_only",
        "tuning_profile": "data_driven_v1",
    },
    "vanilla_transformer": {
        "family": "transformer",
        "learning_rate": 0.0003,
        "batch_size": 64,
        "epochs": 140,
        "optimizer": "Adam",
        "loss_function": "binary_crossentropy",
        "scaler": "robust",
        "class_weight_mode": "auto_if_imbalanced",
        "tuning_profile": "data_driven_v1",
        "sequence_layout": "sensor_groups",
        "feature_group_size": 3,
    },
    "encoder_decoder_transformer": {
        "family": "transformer",
        "learning_rate": 0.0003,
        "batch_size": 64,
        "epochs": 140,
        "optimizer": "Adam",
        "loss_function": "binary_crossentropy",
        "scaler": "robust",
        "class_weight_mode": "auto_if_imbalanced",
        "tuning_profile": "data_driven_v1",
        "sequence_layout": "sensor_groups",
        "feature_group_size": 3,
    },
    "temporal_fusion_transformer": {
        "family": "transformer",
        "learning_rate": 0.0003,
        "batch_size": 64,
        "epochs": 140,
        "optimizer": "Adam",
        "loss_function": "binary_crossentropy",
        "scaler": "robust",
        "class_weight_mode": "auto_if_imbalanced",
        "tuning_profile": "data_driven_v1",
        "sequence_layout": "sensor_groups",
        "feature_group_size": 3,
    },
    "cnn_lstm_hybrid": {
        "family": "hybrid",
        "learning_rate": 0.0003,
        "batch_size": 64,
        "epochs": 120,
        "optimizer": "Adam",
        "loss_function": "binary_crossentropy",
        "scaler": "robust",
        "class_weight_mode": "auto_if_imbalanced",
        "tuning_profile": "data_driven_v1",
        "sequence_layout": "sensor_groups",
        "feature_group_size": 3,
    },
}

TUNING_CANDIDATES: Dict[str, List[Dict[str, object]]] = {
    "cnn": [
        {
            "tuning_profile": "data_driven_v1",
            "learning_rate": 0.0003,
            "batch_size": 64,
            "epochs": 120,
            "scaler": "robust",
            "sequence_layout": "sensor_groups",
            "feature_group_size": 3,
        },
        {
            "tuning_profile": "data_driven_v2",
            "learning_rate": 0.0002,
            "batch_size": 64,
            "epochs": 160,
            "scaler": "robust",
            "sequence_layout": "sensor_groups",
            "feature_group_size": 3,
        },
    ],
    "rnn": [
        {
            "tuning_profile": "data_driven_v1",
            "learning_rate": 0.0004,
            "batch_size": 64,
            "epochs": 100,
            "scaler": "robust",
            "sequence_layout": "sensor_groups",
            "feature_group_size": 3,
        },
        {
            "tuning_profile": "data_driven_v2",
            "learning_rate": 0.0003,
            "batch_size": 32,
            "epochs": 140,
            "scaler": "robust",
            "sequence_layout": "sensor_groups",
            "feature_group_size": 3,
        },
    ],
    "lstm": [
        {
            "tuning_profile": "data_driven_v1",
            "learning_rate": 0.0003,
            "batch_size": 64,
            "epochs": 120,
            "scaler": "robust",
            "sequence_layout": "sensor_groups",
            "feature_group_size": 3,
        },
        {
            "tuning_profile": "data_driven_v2",
            "learning_rate": 0.0002,
            "batch_size": 64,
            "epochs": 160,
            "scaler": "robust",
            "sequence_layout": "sensor_groups",
            "feature_group_size": 3,
        },
    ],
    "gru": [
        {
            "tuning_profile": "data_driven_v1",
            "learning_rate": 0.0003,
            "batch_size": 64,
            "epochs": 120,
            "scaler": "robust",
            "sequence_layout": "sensor_groups",
            "feature_group_size": 3,
        },
        {
            "tuning_profile": "data_driven_v2",
            "learning_rate": 0.0002,
            "batch_size": 64,
            "epochs": 160,
            "scaler": "robust",
            "sequence_layout": "sensor_groups",
            "feature_group_size": 3,
        },
    ],
    "autoencoder": [
        {"tuning_profile": "data_driven_v1", "learning_rate": 0.0005, "batch_size": 64, "epochs": 80},
        {
            "tuning_profile": "data_driven_v2",
            "learning_rate": 0.0003,
            "batch_size": 64,
            "epochs": 120,
        },
    ],
    "vanilla_transformer": [
        {
            "tuning_profile": "data_driven_v1",
            "learning_rate": 0.0003,
            "batch_size": 64,
            "epochs": 140,
            "scaler": "robust",
            "sequence_layout": "sensor_groups",
            "feature_group_size": 3,
        },
        {
            "tuning_profile": "data_driven_v2",
            "learning_rate": 0.0002,
            "batch_size": 64,
            "epochs": 180,
            "scaler": "robust",
            "sequence_layout": "sensor_groups",
            "feature_group_size": 3,
        },
    ],
    "encoder_decoder_transformer": [
        {
            "tuning_profile": "data_driven_v1",
            "learning_rate": 0.0003,
            "batch_size": 64,
            "epochs": 140,
            "scaler": "robust",
            "sequence_layout": "sensor_groups",
            "feature_group_size": 3,
        },
        {
            "tuning_profile": "data_driven_v2",
            "learning_rate": 0.0002,
            "batch_size": 64,
            "epochs": 180,
            "scaler": "robust",
            "sequence_layout": "sensor_groups",
            "feature_group_size": 3,
        },
    ],
    "temporal_fusion_transformer": [
        {
            "tuning_profile": "data_driven_v1",
            "learning_rate": 0.0003,
            "batch_size": 64,
            "epochs": 140,
            "scaler": "robust",
            "sequence_layout": "sensor_groups",
            "feature_group_size": 3,
        },
        {
            "tuning_profile": "data_driven_v2",
            "learning_rate": 0.0002,
            "batch_size": 64,
            "epochs": 180,
            "scaler": "robust",
            "sequence_layout": "sensor_groups",
            "feature_group_size": 3,
        },
    ],
    "cnn_lstm_hybrid": [
        {
            "tuning_profile": "data_driven_v1",
            "learning_rate": 0.0003,
            "batch_size": 64,
            "epochs": 120,
            "scaler": "robust",
            "sequence_layout": "sensor_groups",
            "feature_group_size": 3,
        },
        {
            "tuning_profile": "data_driven_v2",
            "learning_rate": 0.0002,
            "batch_size": 64,
            "epochs": 160,
            "scaler": "robust",
            "sequence_layout": "sensor_groups",
            "feature_group_size": 3,
        },
    ],
}

SEQUENCE_INPUT_MODELS: Set[str] = {
    "cnn",
    "rnn",
    "lstm",
    "gru",
    "vanilla_transformer",
    "encoder_decoder_transformer",
    "temporal_fusion_transformer",
    "cnn_lstm_hybrid",
}

AUTOENCODER_MODELS: Set[str] = {"autoencoder"}


def available_models() -> List[str]:
    return sorted(MODEL_METADATA.keys())


def available_datasets() -> List[str]:
    return sorted(DATASET_REGISTRY.keys())


def family_for(model_name: str) -> str:
    return str(MODEL_METADATA[model_name]["family"])


def protocol_for(model_name: str) -> Dict[str, object]:
    return PROTOCOLS[family_for(model_name)]
