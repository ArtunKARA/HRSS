from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List


SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from effitrack_eval.config import DATASET_REGISTRY
from effitrack_eval.data import find_repo_root


DEFAULT_DATASETS = [
    "hrss_anomalous_optimized",
    "hrss_undersample_optimized",
    "hrss_smote_optimized",
    "hrss_anomalous_standard",
    "hrss_undersample_standard",
    "hrss_smote_standard",
]

DATASET_METADATA = {
    "hrss_anomalous_optimized": {
        "tr_display_name": "Optimized - Dengesiz",
        "en_display_name": "Optimized - Imbalanced",
        "tr_preparation_method": "Orijinal optimized veri, sinif orani korunur",
        "en_preparation_method": "Original optimized dataset, class ratio preserved",
        "tr_source_dataset": "HRSS_anomalous_optimized",
        "en_source_dataset": "HRSS_anomalous_optimized",
        "tr_description": "Optimized ana veri setinin dengesiz hali; normal ve anomali sinif oranlari korunur.",
        "en_description": "Original imbalanced optimized HRSS dataset.",
    },
    "hrss_undersample_optimized": {
        "tr_display_name": "Optimized - Undersample Dengeli",
        "en_display_name": "Optimized - Undersample Balanced",
        "tr_preparation_method": "Cogunluk sinifi downsampling ile azaltildi",
        "en_preparation_method": "Majority class reduced with downsampling",
        "tr_source_dataset": "HRSS_anomalous_optimized -> HRSS_undersample_optimized",
        "en_source_dataset": "HRSS_anomalous_optimized -> HRSS_undersample_optimized",
        "tr_description": "Optimized veri setinde cogunluk sinifi olan normal kayitlar azaltilarak sinif dengesi kurulmustur.",
        "en_description": "Downsampled balanced dataset derived from the optimized source.",
    },
    "hrss_smote_optimized": {
        "tr_display_name": "Optimized - SMOTE Dengeli",
        "en_display_name": "Optimized - SMOTE Balanced",
        "tr_preparation_method": "Azinlik sinifi SMOTE ile artirildi",
        "en_preparation_method": "Minority class increased with SMOTE",
        "tr_source_dataset": "HRSS_anomalous_optimized -> HRSS_SMOTE_optimized",
        "en_source_dataset": "HRSS_anomalous_optimized -> HRSS_SMOTE_optimized",
        "tr_description": "Optimized veri setinde azinlik sinifi olan anomali kayitlari SMOTE ile artirilip sinif dengesi kurulmustur.",
        "en_description": "SMOTE-balanced dataset derived from the optimized source.",
    },
    "hrss_anomalous_standard": {
        "tr_display_name": "Standard - Dengesiz",
        "en_display_name": "Standard - Imbalanced",
        "tr_preparation_method": "Orijinal standard veri, sinif orani korunur",
        "en_preparation_method": "Original standard dataset, class ratio preserved",
        "tr_source_dataset": "HRSS_anomalous_standard",
        "en_source_dataset": "HRSS_anomalous_standard",
        "tr_description": "Standard ana veri setinin dengesiz hali; normal ve anomali sinif oranlari korunur.",
        "en_description": "Original imbalanced standard HRSS dataset.",
    },
    "hrss_undersample_standard": {
        "tr_display_name": "Standard - Undersample Dengeli",
        "en_display_name": "Standard - Undersample Balanced",
        "tr_preparation_method": "Cogunluk sinifi downsampling ile azaltildi",
        "en_preparation_method": "Majority class reduced with downsampling",
        "tr_source_dataset": "HRSS_anomalous_standard -> HRSS_undersample_standard",
        "en_source_dataset": "HRSS_anomalous_standard -> HRSS_undersample_standard",
        "tr_description": "Standard veri setinde cogunluk sinifi olan normal kayitlar azaltilarak sinif dengesi kurulmustur.",
        "en_description": "Downsampled balanced dataset derived from the standard source.",
    },
    "hrss_smote_standard": {
        "tr_display_name": "Standard - SMOTE Dengeli",
        "en_display_name": "Standard - SMOTE Balanced",
        "tr_preparation_method": "Azinlik sinifi SMOTE ile artirildi",
        "en_preparation_method": "Minority class increased with SMOTE",
        "tr_source_dataset": "HRSS_anomalous_standard -> HRSS_SMOTE_standard",
        "en_source_dataset": "HRSS_anomalous_standard -> HRSS_SMOTE_standard",
        "tr_description": "Standard veri setinde azinlik sinifi olan anomali kayitlari SMOTE ile artirilip sinif dengesi kurulmustur.",
        "en_description": "SMOTE-balanced dataset derived from the standard source.",
    },
}

LOCALE_SETTINGS = {
    "tr": {
        "columns": {
            "dataset": "veri_seti_kodu",
            "display_name": "veri_seti_adi",
            "preparation_method": "hazirlama_yontemi",
            "source_dataset": "kaynak_veri",
            "total_rows": "toplam_satir",
            "normal_rows": "normal_satir",
            "anomaly_rows": "anomali_satir",
            "anomaly_ratio_pct": "anomali_orani_yuzde",
            "normal_to_anomaly_ratio": "normal_anomali_orani",
            "label_column": "etiket_kolonu",
            "relative_path": "dosya_yolu",
            "description": "aciklama",
        }
    },
    "en": {
        "columns": {
            "dataset": "dataset",
            "display_name": "display_name",
            "preparation_method": "preparation_method",
            "source_dataset": "source_dataset",
            "total_rows": "total_rows",
            "normal_rows": "normal_rows",
            "anomaly_rows": "anomaly_rows",
            "anomaly_ratio_pct": "anomaly_ratio_pct",
            "normal_to_anomaly_ratio": "normal_to_anomaly_ratio",
            "label_column": "label_column",
            "relative_path": "relative_path",
            "description": "description",
        }
    },
}

INTERNAL_COLUMNS = [
    "dataset",
    "display_name",
    "preparation_method",
    "source_dataset",
    "total_rows",
    "normal_rows",
    "anomaly_rows",
    "anomaly_ratio_pct",
    "normal_to_anomaly_ratio",
    "label_column",
    "relative_path",
    "description",
]


def _resolve_dataset_path(dataset_name: str) -> Path:
    repo_root = find_repo_root()
    relative_path = DATASET_REGISTRY[dataset_name].relative_path
    return (repo_root / relative_path).resolve()


def _count_labels(dataset_path: Path) -> Dict[str, float | int | str]:
    with dataset_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        total_rows = 0
        label_column = None
        label_counts: Dict[str, int] = {}

        for row in reader:
            if label_column is None:
                for candidate in ("Labels", "label", "Label", "class", "target", "y"):
                    if candidate in row:
                        label_column = candidate
                        break
            total_rows += 1
            if label_column is None:
                continue
            label_value = str(row[label_column]).strip()
            label_counts[label_value] = label_counts.get(label_value, 0) + 1

    normal_rows = int(label_counts.get("0", 0) + label_counts.get("0.0", 0))
    anomaly_rows = int(label_counts.get("1", 0) + label_counts.get("1.0", 0))
    anomaly_ratio_pct = (anomaly_rows / total_rows * 100.0) if total_rows else 0.0
    normal_to_anomaly_ratio = (normal_rows / anomaly_rows) if anomaly_rows else 0.0

    return {
        "total_rows": total_rows,
        "normal_rows": normal_rows,
        "anomaly_rows": anomaly_rows,
        "anomaly_ratio_pct": anomaly_ratio_pct,
        "normal_to_anomaly_ratio": normal_to_anomaly_ratio,
        "label_column": label_column or "N/A",
    }


def _build_rows(dataset_names: List[str], locale: str) -> List[Dict[str, str]]:
    repo_root = find_repo_root()
    rows: List[Dict[str, str]] = []
    display_key = "{}_display_name".format(locale)
    method_key = "{}_preparation_method".format(locale)
    source_key = "{}_source_dataset".format(locale)
    description_key = "{}_description".format(locale)

    for dataset_name in dataset_names:
        dataset_spec = DATASET_REGISTRY[dataset_name]
        dataset_path = _resolve_dataset_path(dataset_name)
        counts = _count_labels(dataset_path)
        metadata = DATASET_METADATA.get(dataset_name, {})

        rows.append(
            {
                "dataset": dataset_name,
                "display_name": metadata.get(display_key, dataset_name),
                "preparation_method": metadata.get(method_key, dataset_spec.description),
                "source_dataset": metadata.get(source_key, dataset_spec.name),
                "total_rows": str(int(counts["total_rows"])),
                "normal_rows": str(int(counts["normal_rows"])),
                "anomaly_rows": str(int(counts["anomaly_rows"])),
                "anomaly_ratio_pct": "{:.2f}".format(float(counts["anomaly_ratio_pct"])),
                "normal_to_anomaly_ratio": "{:.2f}:1".format(float(counts["normal_to_anomaly_ratio"])),
                "label_column": str(counts["label_column"]),
                "relative_path": str(dataset_path.relative_to(repo_root)),
                "description": metadata.get(description_key, dataset_spec.description),
            }
        )

    return rows


def _write_csv(path: Path, rows: List[Dict[str, str]], locale: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = LOCALE_SETTINGS[locale]["columns"]
    header = [columns[column] for column in INTERNAL_COLUMNS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow([row[column] for column in INTERNAL_COLUMNS])


def _write_markdown(path: Path, rows: List[Dict[str, str]], locale: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = LOCALE_SETTINGS[locale]["columns"]
    header = [columns[column] for column in INTERNAL_COLUMNS]
    separator = ["---"] * len(header)
    lines = [
        "| {} |".format(" | ".join(header)),
        "| {} |".format(" | ".join(separator)),
    ]
    for row in rows:
        values = [row[column].replace("\n", " ") for column in INTERNAL_COLUMNS]
        lines.append("| {} |".format(" | ".join(values)))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export selected real dataset overview tables")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=DEFAULT_DATASETS,
        help="Dataset names to include in the exported tables",
    )
    parser.add_argument(
        "--locale",
        choices=sorted(LOCALE_SETTINGS.keys()),
        default="tr",
        help="Output language for column names and display labels",
    )
    parser.add_argument("--csv-output", help="Target CSV path")
    parser.add_argument("--markdown-output", help="Target Markdown path")
    args = parser.parse_args()

    if not args.csv_output and not args.markdown_output:
        raise SystemExit("At least one of --csv-output or --markdown-output must be provided.")

    rows = _build_rows(args.datasets, locale=args.locale)

    if args.csv_output:
        _write_csv(Path(args.csv_output).resolve(), rows, locale=args.locale)
        print(Path(args.csv_output).resolve())
    if args.markdown_output:
        _write_markdown(Path(args.markdown_output).resolve(), rows, locale=args.locale)
        print(Path(args.markdown_output).resolve())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
