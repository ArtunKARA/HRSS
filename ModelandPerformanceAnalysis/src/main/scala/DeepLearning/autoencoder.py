from __future__ import annotations

from typing import Any, Dict, Tuple


MODEL_NAME = "autoencoder"


def build_model(input_shape: Tuple[int, ...], metadata: Dict[str, Any]) -> Any:
    import tensorflow as tf

    del metadata

    layers = tf.keras.layers
    Model = tf.keras.Model

    input_dim = int(input_shape[0])

    inputs = layers.Input(shape=(input_dim,))
    x = layers.GaussianNoise(0.05)(inputs)
    x = layers.Dense(64, activation="swish")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(32, activation="swish")(x)
    bottleneck = layers.Dense(12, activation="swish", name="bottleneck")(x)
    x = layers.Dense(32, activation="swish")(bottleneck)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation="swish")(x)
    outputs = layers.Dense(input_dim, activation="sigmoid")(x)
    return Model(inputs, outputs)
