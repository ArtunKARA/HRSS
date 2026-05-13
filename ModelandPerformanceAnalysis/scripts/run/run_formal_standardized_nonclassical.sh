#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/mnt/c/Users/Artun/anaconda3/envs/pyspark_env/python.exe}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYSIS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$ANALYSIS_ROOT/.." && pwd)"
cd "$REPO_ROOT"

DATASETS=(
  hrss_anomalous_optimized
  hrss_smote_optimized
  hrss_undersample_optimized
)

MODELS=(
  cnn
  rnn
  lstm
  gru
  autoencoder
  vanilla_transformer
  encoder_decoder_transformer
  temporal_fusion_transformer
  cnn_lstm_hybrid
)

for model in "${MODELS[@]}"; do
  summary_path="ModelandPerformanceAnalysis/results/experiments/results_formal_standardized/$model/summary.csv"
  if [[ -f "$summary_path" ]]; then
    echo "Skipping $model because summary already exists at $summary_path"
    continue
  fi

  "$PYTHON_BIN" ModelandPerformanceAnalysis/src/main/python/run_experiments.py \
    --models "$model" \
    --datasets "${DATASETS[@]}" \
    --runs 3 \
    --runtime-config ModelandPerformanceAnalysis/configs/runtime/formal_runtime_config.json \
    --output-dir "ModelandPerformanceAnalysis/results/experiments/results_formal_standardized/$model"
done
