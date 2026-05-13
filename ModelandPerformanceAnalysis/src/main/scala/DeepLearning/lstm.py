from __future__ import annotations

from typing import Any, Dict, Tuple


MODEL_NAME = "lstm"


def build_model(input_shape: Tuple[int, ...], metadata: Dict[str, Any]) -> Any:
    import tensorflow as tf

    del metadata

    layers = tf.keras.layers
    Model = tf.keras.Model

    inputs = layers.Input(shape=input_shape)
    x = layers.Bidirectional(
        layers.LSTM(48, return_sequences=True, dropout=0.15)
    )(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(64, activation="swish")(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return Model(inputs, outputs)
