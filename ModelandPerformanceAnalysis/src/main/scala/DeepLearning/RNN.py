from __future__ import annotations

from typing import Any, Dict, Tuple


MODEL_NAME = "rnn"


def build_model(input_shape: Tuple[int, ...], metadata: Dict[str, Any]) -> Any:
    import tensorflow as tf

    del metadata

    layers = tf.keras.layers
    Model = tf.keras.Model

    inputs = layers.Input(shape=input_shape)
    x = layers.SimpleRNN(48, activation="tanh", return_sequences=True)(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.SimpleRNN(32, activation="tanh")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="swish")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return Model(inputs, outputs)
