from __future__ import annotations

from typing import Any, Dict, Tuple


MODEL_NAME = "cnn"


def build_model(input_shape: Tuple[int, ...], metadata: Dict[str, Any]) -> Any:
    import tensorflow as tf

    del metadata

    layers = tf.keras.layers
    Model = tf.keras.Model

    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(48, kernel_size=2, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.SpatialDropout1D(0.1)(x)
    x = layers.Conv1D(80, kernel_size=3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)
    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    x = layers.Concatenate()([avg_pool, max_pool])
    x = layers.Dense(48, activation="swish")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return Model(inputs, outputs)
