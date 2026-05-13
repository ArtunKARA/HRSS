#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYSIS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$ANALYSIS_ROOT/.." && pwd)"
cd "$REPO_ROOT"

python3 ModelandPerformanceAnalysis/src/main/python/merge_experiment_results.py \
  --summary-csv ModelandPerformanceAnalysis/results/experiments/results_formal_new/classical/summary.csv \
  --json-root ModelandPerformanceAnalysis/results/experiments/results_formal_models \
  --json-root ModelandPerformanceAnalysis/results/experiments/results_formal_standardized/cnn \
  --json-root ModelandPerformanceAnalysis/results/experiments/results_formal_standardized/rnn \
  --json-root ModelandPerformanceAnalysis/results/experiments/results_formal_standardized/lstm \
  --json-root ModelandPerformanceAnalysis/results/experiments/results_formal_standardized/gru \
  --json-root ModelandPerformanceAnalysis/results/experiments/results_formal_standardized/autoencoder \
  --json-root ModelandPerformanceAnalysis/results/experiments/results_formal_standardized/vanilla_transformer \
  --json-root ModelandPerformanceAnalysis/results/experiments/results_formal_standardized/encoder_decoder_transformer \
  --json-root ModelandPerformanceAnalysis/results/experiments/results_formal_standardized/temporal_fusion_transformer \
  --json-root ModelandPerformanceAnalysis/results/experiments/results_formal_standardized/cnn_lstm_hybrid \
  --output ModelandPerformanceAnalysis/results/summaries/formal_standardized_summary_all.csv

python3 ModelandPerformanceAnalysis/src/main/python/export_results_workbook.py \
  --summary ModelandPerformanceAnalysis/results/summaries/formal_standardized_summary_all.csv \
  --output ModelandPerformanceAnalysis/results/workbooks/ModelResultsNew.xlsx
