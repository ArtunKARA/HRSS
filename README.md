# EffiTrack

EffiTrack is a data-centric anomaly detection and evaluation framework for high-rack storage system (HRSS) sensor data. The project combines Apache Spark based preprocessing, classical machine learning, deep learning, transformer-style models, and a Spark/Kafka live anomaly detection path.

The repository package intentionally excludes the LaTeX manuscript/report sources and generated paper PDFs. It contains the runnable project code, datasets, configuration files, selected result summaries, and the model artifact needed by the live Random Forest Kafka pipeline.

## Project Structure

```text
Data/
  HRSS datasets and prepared benchmark variants
DatasetAndPreprocessing/
  Scala/Spark preprocessing, anomaly labeling, SMOTE, outlier handling, undersampling
ModelandPerformanceAnalysis/
  Python and Scala evaluation code, runtime configs, result exporters, selected summaries
SparkKafkaStreaming/
  Spark/Kafka producer-consumer flow and saved Random Forest pipeline artifact
Visualization/
  Python visualization scripts and generated plot images
```

## Main Capabilities

- HRSS anomaly dataset preparation with Scala/Spark.
- Class balancing with undersampling and SMOTE.
- Classical ML evaluation: Logistic Regression, Random Forest, SVM, Decision Tree, KNN, Naive Bayes.
- Non-classical evaluation: CNN, RNN, LSTM, GRU, Autoencoder, Transformer variants, CNN-LSTM hybrid.
- Publication-oriented metric exports in CSV and Excel workbook formats.
- Live Random Forest anomaly routing with Spark Streaming and Kafka.
- Visualization scripts for correlation, distributions, time-series anomaly views, pair plots, and power components.

## Requirements

- Java and sbt for the Scala/Spark modules.
- Scala `2.11.8`.
- Apache Spark `2.3.1` dependencies are declared in each `build.sbt`.
- Python `3.10` to `3.12` for the main experiment runner.
- Kafka and ZooKeeper for the live streaming workflow.

Install Python dependencies from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r ModelandPerformanceAnalysis/requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r ModelandPerformanceAnalysis\requirements.txt
```

## Dataset Source

The HRSS data used by this project is based on the public Kaggle dataset:

```text
https://www.kaggle.com/datasets/inIT-OWL/high-storage-system-data-for-energy-optimization
```

Prepared datasets are included under `Data/`, including original normal/anomalous files and balanced benchmark variants.

## Running Experiments

List available models and datasets:

```bash
python ModelandPerformanceAnalysis/src/main/python/run_experiments.py --list-models
python ModelandPerformanceAnalysis/src/main/python/run_experiments.py --list-datasets
```

Run a compact Random Forest benchmark:

```bash
python ModelandPerformanceAnalysis/src/main/python/run_experiments.py \
  --datasets real_imbalanced real_downsample_balanced real_smote_balanced \
  --models random_forest \
  --runs 3 \
  --output-dir ModelandPerformanceAnalysis/results/experiments/manual_random_forest
```

Run the full configured data-driven suite with the provided Windows script:

```powershell
ModelandPerformanceAnalysis\scripts\run\run_selected_data_driven_v2.ps1
```

## Scala/Spark Modules

Compile each Scala/Spark module separately:

```bash
cd DatasetAndPreprocessing
sbt compile

cd ../ModelandPerformanceAnalysis
sbt compile

cd ../SparkKafkaStreaming
sbt compile
```

## Live Kafka Path

The live path uses:

- Training/model family: Random Forest
- Saved pipeline artifact: `SparkKafkaStreaming/model_artifacts/random_forest_smote_standard_pipeline`
- Training data: `Data/HRSS_SMOTE_standard.csv`
- Live input data: `Data/HRSS_normal_standard.csv` and `Data/HRSS_anomalous_standard.csv`
- Kafka input topic: `model-input`
- Kafka output topics: `anomalies3`, `normal_data`, `uncertain_data`

After Kafka and ZooKeeper are running locally, use:

```cmd
ModelandPerformanceAnalysis\scripts\run\run_live_bv_kafka_suite.cmd
```

## Included Results

Selected result artifacts are included to document the latest project state without committing large intermediate logs:

- `ModelandPerformanceAnalysis/results/summaries/`
- `ModelandPerformanceAnalysis/results/tables/`
- `ModelandPerformanceAnalysis/results/workbooks/`
- `ModelandPerformanceAnalysis/results/bv_suite/BV_BigData_Test_Suite.xlsx`
- `ModelandPerformanceAnalysis/results/bv_suite/bv_bigdata_suite_results.json`
- `SparkKafkaStreaming/results/live_metrics/live_kafka_metrics_20260503-124414.jsonl`

Key project findings represented by these files:

- On the real imbalanced HRSS benchmark, Random Forest reached high anomaly coverage while substantially reducing manual review volume.
- On the SMOTE-balanced real benchmark, Random Forest produced the strongest balanced score among the evaluated model families.
- The strongest operational models are CPU-friendly classical ML models, which lowers deployment complexity compared with GPU-heavy alternatives.
- The benchmark suite preserves classical ML cross-validation and repeated hold-out evaluation for deep, transformer, and hybrid models.

## GitHub Packaging Notes

This package is prepared for code-focused publication. It does not include:

- LaTeX manuscript/report sources.
- Generated manuscript PDFs.
- IDE metadata.
- sbt `target/` build outputs.
- Spark warehouse folders.
- Bulk experiment logs and temporary probe outputs.
