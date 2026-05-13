from __future__ import annotations

from typing import Any, Dict, Tuple


MODEL_NAME = "temporal_fusion_transformer"


def build_model(input_shape: Tuple[int, ...], metadata: Dict[str, Any]) -> Any:
    import tensorflow as tf

    del metadata

    layers = tf.keras.layers
    Model = tf.keras.Model

    inputs = layers.Input(shape=input_shape)
    x = layers.Bidirectional(
        layers.LSTM(48, return_sequences=True, dropout=0.1)
    )(inputs)
    x = layers.Dense(48, use_bias=False)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("swish")(x)
    attention_output = layers.MultiHeadAttention(
        num_heads=4,
        key_dim=12,
        dropout=0.1,
    )(x, x)
    attention_output = layers.Add()([x, attention_output])
    attention_output = layers.LayerNormalization()(attention_output)
    dense_output = layers.Dense(96, activation="swish")(attention_output)
    dense_output = layers.Dropout(0.1)(dense_output)
    dense_output = layers.Dense(48)(dense_output)
    dense_output = layers.Add()([attention_output, dense_output])
    dense_output = layers.LayerNormalization()(dense_output)
    avg_pool = layers.GlobalAveragePooling1D()(dense_output)
    max_pool = layers.GlobalMaxPooling1D()(dense_output)
    dense_output = layers.Concatenate()([avg_pool, max_pool])
    dense_output = layers.Dense(64, activation="swish")(dense_output)
    dense_output = layers.Dropout(0.1)(dense_output)
    outputs = layers.Dense(1, activation="sigmoid")(dense_output)
    return Model(inputs, outputs)
