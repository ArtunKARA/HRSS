from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from pathlib import Path

from pyspark import SparkConf, SparkContext
from py4j.protocol import Py4JError


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def env_path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default))


def parse_numeric(value: str) -> int | float | None:
    value = value.strip()
    if not value:
        return None
    numeric = float(value)
    if numeric.is_integer():
        return int(numeric)
    return numeric


def load_rows(path: Path, required_label: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = int(float(row.get("Labels", "-1")))
            if label == required_label:
                rows.append(row)
    return rows


def interleave_rows(normal_rows: list[dict[str, str]], anomalous_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    normal_index = 0
    anomalous_index = 0
    normal_per_anomaly = max(1, len(normal_rows) // max(1, len(anomalous_rows)))

    while normal_index < len(normal_rows) or anomalous_index < len(anomalous_rows):
        emitted_normals = 0
        while normal_index < len(normal_rows) and emitted_normals < normal_per_anomaly:
            merged.append(normal_rows[normal_index])
            normal_index += 1
            emitted_normals += 1
        if anomalous_index < len(anomalous_rows):
            merged.append(anomalous_rows[anomalous_index])
            anomalous_index += 1
        if normal_index >= len(normal_rows):
            merged.extend(anomalous_rows[anomalous_index:])
            anomalous_index = len(anomalous_rows)
        if anomalous_index >= len(anomalous_rows):
            merged.extend(normal_rows[normal_index:])
            normal_index = len(normal_rows)

    return merged


def build_payloads(normal_path: Path, anomalous_path: Path) -> list[str]:
    normal_rows = load_rows(normal_path, required_label=0)
    anomalous_rows = load_rows(anomalous_path, required_label=1)
    ordered_rows = interleave_rows(normal_rows, anomalous_rows)
    payloads: list[str] = []
    for row in ordered_rows:
        normalized: dict[str, object] = {}
        for key, value in row.items():
            numeric = parse_numeric(value)
            normalized[key] = value if numeric is None and value.strip() else numeric
        payloads.append(json.dumps(normalized, separators=(",", ":"), sort_keys=True))
    return payloads


def build_batches(reservoir: list[str], data_flow_rate_ms: int, batch_interval_s: int, duration_s: int) -> list[list[str]]:
    total_messages = max(1, duration_s * 1000 // data_flow_rate_ms)
    window_count = max(1, duration_s // batch_interval_s)
    batches: list[list[str]] = [[] for _ in range(window_count)]
    for index in range(total_messages):
        event_time_ms = index * data_flow_rate_ms
        if event_time_ms >= duration_s * 1000:
            break
        batch_index = (event_time_ms // 1000) // batch_interval_s
        if batch_index < window_count:
            batches[batch_index].append(reservoir[index % len(reservoir)])
    return batches


def process_partition(iterator):
    work_rounds = env_int("BENCHMARK_WORK_ROUNDS", 10)
    processed = 0
    for payload in iterator:
        digest = payload.encode("utf-8")
        for _ in range(work_rounds):
            digest = hashlib.sha256(digest).digest()
        processed += 1
    yield processed


def collect_runtime_metrics(sc: SparkContext) -> tuple[float, float, float]:
    memory_bean = sc._jvm.java.lang.management.ManagementFactory.getMemoryMXBean()
    heap_bytes = float(memory_bean.getHeapMemoryUsage().getUsed())
    non_heap_bytes = float(memory_bean.getNonHeapMemoryUsage().getUsed())
    try:
        storage_bytes = float(sc._jsc.sc().env().blockManager().memoryStore().memoryUsed())
    except (Py4JError, AttributeError):
        # Spark 2.3.x on Windows can expose MemoryStore without a public memoryUsed()
        # bridge method in Py4J; keep benchmark runnable with a safe fallback.
        storage_bytes = 0.0
    return heap_bytes, non_heap_bytes, storage_bytes


def main() -> int:
    data_flow_rate_ms = env_int("BENCHMARK_DATA_FLOW_RATE_MS", 500)
    batch_interval_s = env_int("BENCHMARK_BATCH_INTERVAL_S", 10)
    duration_s = env_int("BENCHMARK_DURATION_S", 60)
    cores = env_int("BENCHMARK_CORES", 4)
    partitions = env_int("BENCHMARK_PARTITIONS", 1)
    output_json_text = os.getenv("BENCHMARK_OUTPUT_JSON", "").strip()
    output_json_path = Path(output_json_text) if output_json_text else None
    normal_path = env_path("STREAM_SOURCE_NORMAL_CSV_PATH", "Data/HRSS_normal_standard.csv")
    anomalous_path = env_path("STREAM_SOURCE_ANOMALOUS_CSV_PATH", "Data/HRSS_anomalous_standard.csv")

    conf = SparkConf().setAppName("SparkResourceBenchmarkPy")
    sc = SparkContext(conf=conf)

    peak_heap_bytes = 0.0
    peak_non_heap_bytes = 0.0
    peak_storage_bytes = 0.0

    try:
        reservoir = build_payloads(normal_path, anomalous_path)
        batches = build_batches(reservoir, data_flow_rate_ms, batch_interval_s, duration_s)

        total_processed = 0
        total_tasks = 0
        job_count = 0
        batch_times_ms: list[float] = []
        scheduling_delays_ms: list[float] = []
        actual_end_s = 0.0

        for batch_idx, batch in enumerate(batches):
            scheduled_start_s = batch_idx * float(batch_interval_s)
            if not batch:
                scheduling_delays_ms.append(max(0.0, (actual_end_s - scheduled_start_s) * 1000.0))
                continue

            rdd = sc.parallelize(batch, partitions).persist()
            total_tasks += rdd.getNumPartitions()

            start = time.perf_counter()
            counts = rdd.mapPartitions(process_partition).collect()
            processing_s = time.perf_counter() - start
            rdd.unpersist()

            job_count += 1
            scheduled_delay_s = max(0.0, actual_end_s - scheduled_start_s)
            actual_start_s = scheduled_start_s + scheduled_delay_s
            actual_end_s = actual_start_s + processing_s

            batch_times_ms.append(processing_s * 1000.0)
            scheduling_delays_ms.append(scheduled_delay_s * 1000.0)
            total_processed += sum(counts)

            heap_bytes, non_heap_bytes, storage_bytes = collect_runtime_metrics(sc)
            peak_heap_bytes = max(peak_heap_bytes, heap_bytes)
            peak_non_heap_bytes = max(peak_non_heap_bytes, non_heap_bytes)
            peak_storage_bytes = max(peak_storage_bytes, storage_bytes)

        heap_bytes, non_heap_bytes, storage_bytes = collect_runtime_metrics(sc)
        peak_heap_bytes = max(peak_heap_bytes, heap_bytes)
        peak_non_heap_bytes = max(peak_non_heap_bytes, non_heap_bytes)
        peak_storage_bytes = max(peak_storage_bytes, storage_bytes)

        processing_elapsed_s = sum(batch_times_ms) / 1000.0 if batch_times_ms else 0.0
        result = {
            "data_flow_rate_ms": data_flow_rate_ms,
            "batch_interval_s": batch_interval_s,
            "duration_s": duration_s,
            "cores": cores,
            "partitions": partitions,
            "messages": total_processed,
            "tasks": total_tasks,
            "batch_count": len(batches),
            "jobs": job_count,
            "elapsed_time_s": round(actual_end_s, 4),
            "task_time_s": round(processing_elapsed_s, 4),
            "avg_input_s": round((total_processed / duration_s) if duration_s else 0.0, 4),
            "avg_process_s": round((total_processed / processing_elapsed_s) if processing_elapsed_s else 0.0, 4),
            "mean_batch_ms": round((sum(batch_times_ms) / len(batch_times_ms)) if batch_times_ms else 0.0, 4),
            "mean_delay_ms": round((sum(scheduling_delays_ms) / len(scheduling_delays_ms)) if scheduling_delays_ms else 0.0, 4),
            "storage_memory_kib": round(peak_storage_bytes / 1024.0, 4),
            "peak_jvm_on_heap_mib": round(peak_heap_bytes / (1024.0 * 1024.0), 4),
            "peak_jvm_off_heap_mib": round(peak_non_heap_bytes / (1024.0 * 1024.0), 4),
            "peak_storage_on_heap_mib": round(peak_storage_bytes / (1024.0 * 1024.0), 4),
        }

        if output_json_path is not None:
            output_json_path.parent.mkdir(parents=True, exist_ok=True)
            output_json_path.write_text(json.dumps(result), encoding="utf-8")
        print(json.dumps(result))
        return 0
    finally:
        sc.stop()


if __name__ == "__main__":
    raise SystemExit(main())
