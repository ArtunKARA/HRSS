from __future__ import annotations

from typing import Any, Dict, Tuple


MODEL_NAME = "gru"


def build_model(input_shape: Tuple[int, ...], metadata: Dict[str, Any]) -> Any:
    import tensorflow as tf

    del metadata

    layers = tf.keras.layers
    Model = tf.keras.Model

    inputs = layers.Input(shape=input_shape)
    x = layers.Bidirectional(
        layers.GRU(64, return_sequences=True, dropout=0.1)
    )(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.Bidirectional(
        layers.GRU(32, return_sequences=True, dropout=0.1)
    )(x)
    x = layers.LayerNormalization()(x)
    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    x = layers.Concatenate()([avg_pool, max_pool])
    x = layers.Dense(64, activation="swish")(x)
    x = layers.Dropout(0.15)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return Model(inputs, outputs)
