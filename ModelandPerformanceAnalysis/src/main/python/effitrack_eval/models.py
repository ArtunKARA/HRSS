from __future__ import annotations

import importlib.util
import random
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .config import MODEL_METADATA


EXTERNAL_MODEL_SOURCES: Dict[str, str] = {
    "cnn": "ModelandPerformanceAnalysis/src/main/scala/DeepLearning/CNN.py",
    "rnn": "ModelandPerformanceAnalysis/src/main/scala/DeepLearning/RNN.py",
    "lstm": "ModelandPerformanceAnalysis/src/main/scala/DeepLearning/lstm.py",
    "gru": "ModelandPerformanceAnalysis/src/main/scala/DeepLearning/GRU.py",
    "autoencoder": "ModelandPerformanceAnalysis/src/main/scala/DeepLearning/autoencoder.py",
    "cnn_lstm_hybrid": "ModelandPerformanceAnalysis/src/main/scala/DeepLearning/CNNLSTMHybrid.py",
    "vanilla_transformer": "ModelandPerformanceAnalysis/src/main/scala/Transformer/VanilaTransformer.py",
    "encoder_decoder_transformer": "ModelandPerformanceAnalysis/src/main/scala/Transformer/EncoderDecoderTransformer.py",
    "temporal_fusion_transformer": "ModelandPerformanceAnalysis/src/main/scala/Transformer/TemporalFusionTransformer.py",
}

_MODULE_CACHE: Dict[str, ModuleType] = {}


def require_sklearn() -> Dict[str, Any]:
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import LinearSVC
        from sklearn.tree import DecisionTreeClassifier
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "scikit-learn is required for the experiment runner. "
            "Install ModelandPerformanceAnalysis/requirements.txt first."
        ) from exc

    return {
        "RandomForestClassifier": RandomForestClassifier,
        "LogisticRegression": LogisticRegression,
        "GaussianNB": GaussianNB,
        "KNeighborsClassifier": KNeighborsClassifier,
        "Pipeline": Pipeline,
        "StandardScaler": StandardScaler,
        "LinearSVC": LinearSVC,
        "DecisionTreeClassifier": DecisionTreeClassifier,
    }


def require_tensorflow() -> Any:
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TensorFlow is required for deep learning, transformer, and hybrid experiments. "
            "Install ModelandPerformanceAnalysis/requirements.txt first."
        ) from exc

    return tf


def _find_repo_root(start: Optional[Path] = None) -> Path:
    current = (start or Path(__file__).resolve()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "README.md").exists() and (candidate / "Data").exists():
            return candidate
    raise FileNotFoundError("Repo root could not be located from the current file path.")


def _load_external_model_module(model_name: str) -> ModuleType:
    if model_name in _MODULE_CACHE:
        return _MODULE_CACHE[model_name]

    if model_name not in EXTERNAL_MODEL_SOURCES:
        raise KeyError("Unknown external model source: {}".format(model_name))

    source_path = _find_repo_root() / EXTERNAL_MODEL_SOURCES[model_name]
    spec = importlib.util.spec_from_file_location("effitrack_external_{}".format(model_name), source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load external model module from {}".format(source_path))

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _MODULE_CACHE[model_name] = module
    return module


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        tf = require_tensorflow()
        tf.keras.utils.set_random_seed(seed)
    except RuntimeError:
        pass


def runtime_hardware(model_name: str) -> str:
    family = str(MODEL_METADATA[model_name]["family"])
    if family == "classical_ml":
        return "CPU"

    try:
        tf = require_tensorflow()
    except RuntimeError:
        return "CPU"

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return "CPU"
    return "GPU"


def resolve_model_metadata(
    model_name: str,
    runtime_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = dict(MODEL_METADATA[model_name])
    if runtime_overrides:
        payload.update(runtime_overrides)
    return payload


def model_runtime_metadata(
    model_name: str,
    runtime_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = resolve_model_metadata(model_name, runtime_overrides=runtime_overrides)
    payload["hardware"] = runtime_hardware(model_name)
    return payload


def build_classical_model(model_name: str, seed: int) -> Any:
    modules = require_sklearn()
    Pipeline = modules["Pipeline"]
    StandardScaler = modules["StandardScaler"]

    if model_name == "logistic_regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", modules["LogisticRegression"](max_iter=2000, random_state=seed)),
            ]
        )
    if model_name == "random_forest":
        return modules["RandomForestClassifier"](n_estimators=500, random_state=seed, n_jobs=-1)
    if model_name == "svm":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", modules["LinearSVC"](max_iter=5000, random_state=seed)),
            ]
        )
    if model_name == "decision_tree":
        return modules["DecisionTreeClassifier"](random_state=seed)
    if model_name == "knn":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", modules["KNeighborsClassifier"](n_neighbors=5)),
            ]
        )
    if model_name == "naive_bayes":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", modules["GaussianNB"]()),
            ]
        )

    raise KeyError(f"Unknown classical model: {model_name}")


def _compile_binary_classifier(model: Any, metadata: Dict[str, Any]) -> Any:
    tf = require_tensorflow()
    learning_rate = float(metadata["learning_rate"])
    optimizer_name = str(metadata.get("optimizer", "Adam")).lower()
    if optimizer_name == "rmsprop":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    loss_name = str(metadata.get("loss_function", "binary_crossentropy"))
    if loss_name == "binary_focal_crossentropy" and hasattr(tf.keras.losses, "BinaryFocalCrossentropy"):
        loss = tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0)
    else:
        loss = "binary_crossentropy"

    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model


def _wrap_external_classifier_model(
    model_name: str,
    base_model: Any,
    input_shape: Tuple[int, ...],
) -> Any:
    tf = require_tensorflow()
    if model_name not in {"encoder_decoder_transformer", "temporal_fusion_transformer"}:
        return base_model

    # The source model files stay faithful to the original definitions.
    # The experiment runner only adapts them to a single-input/single-score interface.
    inputs = tf.keras.Input(shape=input_shape)
    if model_name == "encoder_decoder_transformer":
        outputs = base_model([inputs, inputs])
    else:
        outputs = base_model(inputs)
    outputs = tf.keras.layers.Lambda(
        lambda tensor: tf.reduce_mean(tensor, axis=-1, keepdims=True),
        name="score_adapter",
    )(outputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)


def build_neural_model(
    model_name: str,
    input_shape: Tuple[int, ...],
    runtime_overrides: Optional[Dict[str, Any]] = None,
) -> Any:
    tf = require_tensorflow()
    metadata = resolve_model_metadata(model_name, runtime_overrides=runtime_overrides)
    if model_name not in EXTERNAL_MODEL_SOURCES:
        raise KeyError("Unknown neural model: {}".format(model_name))

    module = _load_external_model_module(model_name)
    if not hasattr(module, "build_model"):
        raise RuntimeError(
            "External model module for '{}' does not expose build_model().".format(model_name)
        )

    model = module.build_model(input_shape, metadata)
    if model_name == "autoencoder":
        optimizer = tf.keras.optimizers.Adam(learning_rate=float(metadata["learning_rate"]))
        model.compile(optimizer=optimizer, loss="mse")
        return model
    model = _wrap_external_classifier_model(model_name, model, input_shape)
    return _compile_binary_classifier(model, metadata)
