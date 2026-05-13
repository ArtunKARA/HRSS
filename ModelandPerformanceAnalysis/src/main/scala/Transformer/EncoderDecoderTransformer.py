from __future__ import annotations

from typing import Any, Dict, Tuple


MODEL_NAME = "encoder_decoder_transformer"


def build_model(input_shape: Tuple[int, ...], metadata: Dict[str, Any]) -> Any:
    import tensorflow as tf

    del metadata

    layers = tf.keras.layers
    Model = tf.keras.Model

    def encoder_block(inputs: Any) -> Any:
        attention_output = layers.MultiHeadAttention(
            num_heads=4,
            key_dim=8,
            dropout=0.1,
        )(inputs, inputs)
        attention_output = layers.Dropout(0.1)(attention_output)
        attention_output = layers.Add()([inputs, attention_output])
        attention_output = layers.LayerNormalization()(attention_output)
        ff_output = layers.Dense(64, activation="swish")(attention_output)
        ff_output = layers.Dense(int(inputs.shape[-1]))(ff_output)
        ff_output = layers.Dropout(0.1)(ff_output)
        ff_output = layers.Add()([attention_output, ff_output])
        return layers.LayerNormalization()(ff_output)

    def decoder_block(encoder_outputs: Any, decoder_inputs: Any) -> Any:
        attention_output = layers.MultiHeadAttention(
            num_heads=4,
            key_dim=8,
            dropout=0.1,
        )(decoder_inputs, decoder_inputs)
        attention_output = layers.Dropout(0.1)(attention_output)
        attention_output = layers.Add()([decoder_inputs, attention_output])
        attention_output = layers.LayerNormalization()(attention_output)
        cross_attention_output = layers.MultiHeadAttention(
            num_heads=4,
            key_dim=8,
            dropout=0.1,
        )(attention_output, encoder_outputs)
        cross_attention_output = layers.Dropout(0.1)(cross_attention_output)
        cross_attention_output = layers.Add()([attention_output, cross_attention_output])
        cross_attention_output = layers.LayerNormalization()(cross_attention_output)
        ff_output = layers.Dense(64, activation="swish")(cross_attention_output)
        ff_output = layers.Dense(int(decoder_inputs.shape[-1]))(ff_output)
        ff_output = layers.Dropout(0.1)(ff_output)
        ff_output = layers.Add()([cross_attention_output, ff_output])
        return layers.LayerNormalization()(ff_output)

    encoder_inputs = layers.Input(shape=input_shape)
    encoder_projection = layers.Dense(32, use_bias=False)(encoder_inputs)
    encoder_projection = layers.LayerNormalization()(encoder_projection)
    encoder_projection = layers.Activation("swish")(encoder_projection)
    encoder_outputs = encoder_block(encoder_projection)
    decoder_inputs = layers.Input(shape=input_shape)
    decoder_projection = layers.Dense(32, use_bias=False)(decoder_inputs)
    decoder_projection = layers.LayerNormalization()(decoder_projection)
    decoder_projection = layers.Activation("swish")(decoder_projection)
    decoder_outputs = decoder_block(encoder_outputs, decoder_projection)
    avg_pool = layers.GlobalAveragePooling1D()(decoder_outputs)
    max_pool = layers.GlobalMaxPooling1D()(decoder_outputs)
    decoder_outputs = layers.Concatenate()([avg_pool, max_pool])
    decoder_outputs = layers.Dense(48, activation="swish")(decoder_outputs)
    decoder_outputs = layers.Dropout(0.15)(decoder_outputs)
    output_layer = layers.Dense(1, activation="sigmoid")(decoder_outputs)
    return Model(inputs=[encoder_inputs, decoder_inputs], outputs=output_layer)
