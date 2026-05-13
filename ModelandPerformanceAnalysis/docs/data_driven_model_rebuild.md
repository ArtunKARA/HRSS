# Data-Driven Model Rebuild Notes

## Dataset observations

- The experiment runner excludes `Timestamp` from the model input and keeps the `18` numeric sensor features only.
- The remaining features form a natural `6 sensor groups x 3 channels` layout:
  - `Weg`
  - `power`
  - `voltage`
- The main anomalous dataset is imbalanced (`15117` normal, `4517` anomalous).
- There are no missing values in the main datasets.
- Multiple feature pairs are strongly correlated:
  - `O_w_HR_voltage` vs `O_w_HL_voltage`
  - `I_w_HR_Weg` vs `I_w_HL_Weg`
  - several `power` vs `voltage` pairs within the same sensor group
- Feature scales differ heavily, especially for `power` variables, which makes outlier-robust scaling more appropriate than plain min-max scaling for most classifier models.

## Design consequences

- Sequence-style models should consume the input as `6` grouped sensor tokens with `3` channels each, not as a flat `18 x 1` pseudo-sequence.
- Because the effective sequence length is only `6`, shallow architectures are preferred over deep stacks.
- Early aggressive pooling and naive flattening waste too much structure on such short grouped inputs.
- `swish` is preferred in dense / convolutional blocks because the features are continuous and strongly scaled.
- Recurrent models keep their natural gated / `tanh` dynamics internally, but they now aggregate grouped sensor context instead of raw feature order.
- Transformer variants now operate on grouped sensor tokens and summarize with pooling instead of flattening token logits.

## Revised model families

- `cnn`
  - Two short-kernel grouped convolutions
  - Batch normalization
  - Global average pooling instead of flattening
  - Dense head with dropout

- `rnn`
  - Two-stage SimpleRNN stack
  - Layer normalization between recurrent stages
  - Small dense head

- `lstm`
  - Bidirectional LSTM over grouped sensor tokens
  - Layer normalization
  - Global max pooling

- `gru`
  - Bidirectional GRU over grouped sensor tokens
  - Layer normalization
  - Global max pooling

- `autoencoder`
  - Denoising autoencoder with Gaussian noise
  - Wider encoder and smaller bottleneck
  - Symmetric reconstruction path

- `vanilla_transformer`
  - Token projection to a compact model dimension
  - Single encoder-style self-attention block
  - Residual feed-forward block
  - Global average pooling head

- `encoder_decoder_transformer`
  - Compact encoder / decoder attention blocks
  - Shared grouped sensor layout
  - Pooling-based scalar output instead of flattening token scores

- `temporal_fusion_transformer`
  - Bidirectional LSTM context encoder
  - Compact self-attention refinement
  - Residual feed-forward block
  - Global average pooling head

- `cnn_lstm_hybrid`
  - Grouped convolution front-end
  - Bidirectional LSTM context layer
  - Global max pooling and dense head

## Training defaults

- Classifier models now default to:
  - `robust` scaling
  - grouped input layout (`feature_group_size = 3`) where applicable
  - lower learning rates (`2e-4` to `4e-4`)
  - larger batch size (`64`)
  - longer but early-stopped training windows

- The autoencoder keeps `minmax` scaling because its decoder still reconstructs into a bounded feature space.

## Practical implication

The rebuilt models are no longer shaped around an arbitrary flat-feature ordering. They now match the actual HRSS sensor structure much more closely, which makes the resulting architecture choices easier to defend in a paper.
