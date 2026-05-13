from __future__ import annotations

import json
import os
from pathlib import Path

from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType, IntegerType


REPO_ROOT = Path(__file__).resolve().parents[3]


def env_path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default))


def probability_one(value) -> float | None:
    if value is None:
        return None
    return float(value[1])


def main() -> int:
    output_path = env_path(
        "ROUTING_SCORE_OUTPUT_JSONL",
        str(REPO_ROOT / "ModelandPerformanceAnalysis" / "results" / "bv_suite" / "rf_routing_scores.jsonl"),
    )
    model_path = env_path(
        "RF_PIPELINE_ARTIFACT_PATH",
        str(REPO_ROOT / "SparkKafkaStreaming" / "model_artifacts" / "random_forest_smote_standard_pipeline"),
    )
    normal_path = env_path("STREAM_SOURCE_NORMAL_CSV_PATH", str(REPO_ROOT / "Data" / "HRSS_normal_standard.csv"))
    anomalous_path = env_path("STREAM_SOURCE_ANOMALOUS_CSV_PATH", str(REPO_ROOT / "Data" / "HRSS_anomalous_standard.csv"))

    if not model_path.exists():
        raise FileNotFoundError(f"Missing RF pipeline artifact: {model_path}")
    if not normal_path.exists():
        raise FileNotFoundError(f"Missing raw normal dataset: {normal_path}")
    if not anomalous_path.exists():
        raise FileNotFoundError(f"Missing raw anomalous dataset: {anomalous_path}")

    spark = SparkSession.builder.appName("RandomForestRoutingScoreProbe").getOrCreate()

    try:
        normal_df = spark.read.option("header", True).option("inferSchema", True).csv(str(normal_path))
        anomalous_df = spark.read.option("header", True).option("inferSchema", True).csv(str(anomalous_path))
        combined_df = normal_df.union(anomalous_df.select(normal_df.columns))

        model = PipelineModel.load(str(model_path))
        transformed = model.transform(combined_df)

        probability_udf = udf(probability_one, DoubleType())
        scored_rows = (
            transformed
            .select(
                col("Labels").cast(IntegerType()).alias("label"),
                probability_udf(col("probability")).alias("prediction"),
            )
            .collect()
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for row in scored_rows:
                handle.write(
                    json.dumps(
                        {
                            "label": int(row["label"]),
                            "prediction": float(row["prediction"]),
                        },
                        separators=(",", ":"),
                    )
                )
                handle.write("\n")

        print(
            json.dumps(
                {
                    "status": "ok",
                    "model_path": str(model_path),
                    "normal_path": str(normal_path),
                    "anomalous_path": str(anomalous_path),
                    "output_path": str(output_path),
                    "sample_count": len(scored_rows),
                }
            )
        )
        return 0
    finally:
        spark.stop()


if __name__ == "__main__":
    raise SystemExit(main())
