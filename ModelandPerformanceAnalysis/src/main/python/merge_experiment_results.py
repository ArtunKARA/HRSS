from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from effitrack_eval.runner import _build_summary_row


def _read_summary_rows(path: Path) -> List[Dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json_rows(root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in sorted(root.rglob("*.json")):
        if path.name == "tuned_runtime_config.json":
            continue
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict) or "dataset" not in payload or "model" not in payload:
            continue
        rows.append(_build_summary_row(payload))
    return rows


def _column_order(rows: Iterable[Dict[str, object]]) -> List[str]:
    preferred = [
        "dataset",
        "model",
        "family",
        "evaluation_strategy",
        "accuracy_mean",
        "accuracy_std",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "f1_mean",
        "f1_std",
        "baseline_f1_mean",
        "baseline_f1_std",
        "p_value",
        "significant",
        "direction",
        "learning_rate",
        "batch_size",
        "epochs",
        "optimizer",
        "loss_function",
        "class_weight_mode",
        "tuning_profile",
        "hardware",
        "result_source",
    ]
    seen = set()
    columns: List[str] = []
    materialized = list(rows)

    for column in preferred:
        for row in materialized:
            if column in row and column not in seen:
                columns.append(column)
                seen.add(column)
                break

    for row in materialized:
        for column in row.keys():
            if column not in seen:
                columns.append(column)
                seen.add(column)
    return columns


def _dedupe_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    merged: "OrderedDict[Tuple[str, str], Dict[str, object]]" = OrderedDict()
    for row in rows:
        key = (str(row.get("dataset", "")), str(row.get("model", "")))
        merged[key] = row
    return list(merged.values())


def _write_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = _column_order(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge experiment summary CSV and JSON outputs")
    parser.add_argument("--summary-csv", action="append", default=[], help="Existing summary CSV path")
    parser.add_argument("--json-root", action="append", default=[], help="Directory containing JSON experiment reports")
    parser.add_argument("--output", required=True, help="Merged summary CSV output path")
    args = parser.parse_args()

    rows: List[Dict[str, object]] = []

    for summary_path in args.summary_csv:
        path = Path(summary_path).resolve()
        for row in _read_summary_rows(path):
            payload = dict(row)
            payload["result_source"] = path.parent.name or path.name
            rows.append(payload)

    for json_root in args.json_root:
        root = Path(json_root).resolve()
        for row in _read_json_rows(root):
            payload = dict(row)
            payload["result_source"] = root.name
            rows.append(payload)

    merged_rows = _dedupe_rows(rows)
    merged_rows.sort(key=lambda item: (str(item.get("dataset", "")), str(item.get("family", "")), str(item.get("model", ""))))
    output_path = Path(args.output).resolve()
    _write_rows(output_path, merged_rows)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
