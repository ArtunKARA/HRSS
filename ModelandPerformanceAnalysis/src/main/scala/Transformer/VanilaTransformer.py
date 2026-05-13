from __future__ import annotations

from typing import Any, Dict, Tuple


MODEL_NAME = "vanilla_transformer"


def build_model(input_shape: Tuple[int, ...], metadata: Dict[str, Any]) -> Any:
    import tensorflow as tf

    del metadata

    layers = tf.keras.layers
    Model = tf.keras.Model

    inputs = layers.Input(shape=input_shape)
    x = layers.Dense(32, use_bias=False)(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("swish")(x)
    attention_output = layers.MultiHeadAttention(
        num_heads=4,
        key_dim=8,
        dropout=0.1,
    )(x, x)
    x = layers.Add()([x, attention_output])
    x = layers.LayerNormalization()(x)
    ff_output = layers.Dense(64, activation="swish")(x)
    ff_output = layers.Dropout(0.1)(ff_output)
    ff_output = layers.Dense(32)(ff_output)
    x = layers.Add()([x, ff_output])
    x = layers.LayerNormalization()(x)
    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    x = layers.Concatenate()([avg_pool, max_pool])
    x = layers.Dense(48, activation="swish")(x)
    x = layers.Dropout(0.15)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return Model(inputs, outputs)
