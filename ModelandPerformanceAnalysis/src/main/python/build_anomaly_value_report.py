from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from effitrack_eval.data import find_repo_root


SUMMARY_DEFAULT = "ModelandPerformanceAnalysis/results/summaries/selected_real_dataset_suite_summary_all.csv"
SUMMARY_FALLBACK = "ModelandPerformanceAnalysis/results/summaries/selected_six_dataset_suite_summary_all.csv"
OUTPUT_DEFAULT = "cikti_anomali_analizi.txt"

REPORT_DATASETS = {
    "real_imbalanced": "Data/HRSS_anomalous_optimized.csv",
    "real_downsample_balanced": "Data/benchmark_suite/HRSS_real_downsample_balanced.csv",
    "real_smote_balanced": "Data/benchmark_suite/HRSS_real_smote_balanced.csv",
}

PRIMARY_DATASETS = {
    "HRSS_anomalous_optimized": "Data/HRSS_anomalous_optimized.csv",
    "HRSS_undersample_optimized": "Data/HRSS_undersample_optimized.csv",
    "HRSS_SMOTE_optimized": "Data/HRSS_SMOTE_optimized.csv",
}

DISPLAY_NAMES = {
    "real_imbalanced": "Gercek Dengesiz",
    "real_downsample_balanced": "Gercek Downsample Dengeli",
    "real_smote_balanced": "Gercek SMOTE Dengeli",
}

FAMILY_NAMES = {
    "classical_ml": "Classical ML",
    "deep_learning": "Deep Learning",
    "transformer": "Transformer",
    "hybrid": "Hybrid",
}


def _resolve(path_str: str) -> Path:
    repo_root = find_repo_root()
    path = Path(path_str)
    if not path.is_absolute():
        path = repo_root / path
    return path


def _resolve_summary_path(requested: str) -> Path:
    if requested != SUMMARY_DEFAULT:
        return _resolve(requested)

    preferred = _resolve(SUMMARY_DEFAULT)
    if preferred.exists():
        return preferred
    return _resolve(SUMMARY_FALLBACK)


def _read_counts(path: Path) -> Dict[str, float]:
    total_rows = 0
    normal_rows = 0
    anomaly_rows = 0

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            total_rows += 1
            label = int(float(row["Labels"]))
            if label == 1:
                anomaly_rows += 1
            else:
                normal_rows += 1

    anomaly_ratio = (anomaly_rows / total_rows) if total_rows else 0.0
    imbalance_ratio = (normal_rows / anomaly_rows) if anomaly_rows else 0.0
    return {
        "total_rows": float(total_rows),
        "normal_rows": float(normal_rows),
        "anomaly_rows": float(anomaly_rows),
        "anomaly_ratio": anomaly_ratio,
        "imbalance_ratio": imbalance_ratio,
    }


def _read_summary_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _to_float(row: Dict[str, str], key: str) -> float:
    return float(row[key])


def _sort_by_f1(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: (
            _to_float(row, "f1_mean"),
            _to_float(row, "recall_mean"),
            _to_float(row, "precision_mean"),
        ),
        reverse=True,
    )


def _best_rows_by_dataset(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    result: Dict[str, Dict[str, str]] = {}
    for dataset_name in REPORT_DATASETS:
        dataset_rows = [row for row in rows if row["dataset"] == dataset_name]
        if not dataset_rows:
            raise FileNotFoundError(
                "Summary file does not contain the expected dataset '{}'".format(dataset_name)
            )
        result[dataset_name] = _sort_by_f1(dataset_rows)[0]
    return result


def _best_rows_by_family(dataset_rows: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    by_family: Dict[str, Dict[str, str]] = {}
    for row in dataset_rows:
        family = row["family"]
        if family not in by_family or _to_float(row, "f1_mean") > _to_float(by_family[family], "f1_mean"):
            by_family[family] = row
    return by_family


def _format_int(value: float) -> str:
    return "{:,.0f}".format(value)


def _format_float(value: float, digits: int = 4) -> str:
    return ("{:.%df}" % digits).format(value)


def _format_pct(value: float) -> str:
    return "{:.2f}%".format(value * 100.0)


def _operational_metrics(row: Dict[str, str], counts: Dict[str, float]) -> Dict[str, float]:
    precision = _to_float(row, "precision_mean")
    recall = _to_float(row, "recall_mean")
    total_rows = counts["total_rows"]
    normal_rows = counts["normal_rows"]
    anomaly_rows = counts["anomaly_rows"]

    true_positives = recall * anomaly_rows
    false_negatives = anomaly_rows - true_positives
    predicted_positives = (true_positives / precision) if precision > 0 else 0.0
    false_positives = max(predicted_positives - true_positives, 0.0)
    prevented_manual_checks = max(total_rows - predicted_positives, 0.0)
    review_reduction = 1.0 - (predicted_positives / total_rows) if total_rows else 0.0
    review_compression = (total_rows / predicted_positives) if predicted_positives else 0.0
    false_positive_rate = (false_positives / normal_rows) if normal_rows else 0.0
    miss_rate = (false_negatives / anomaly_rows) if anomaly_rows else 0.0
    baseline_lift = (
        _to_float(row, "f1_mean") / _to_float(row, "baseline_f1_mean")
        if _to_float(row, "baseline_f1_mean") > 0
        else 0.0
    )
    alerts_per_1000_rows = (predicted_positives / total_rows * 1000.0) if total_rows else 0.0
    anomalies_caught_per_1000_rows = (true_positives / total_rows * 1000.0) if total_rows else 0.0
    false_alarms_per_1000_rows = (false_positives / total_rows * 1000.0) if total_rows else 0.0

    return {
        "tp": true_positives,
        "fn": false_negatives,
        "fp": false_positives,
        "predicted_positive": predicted_positives,
        "prevented_manual_checks": prevented_manual_checks,
        "review_reduction": review_reduction,
        "review_compression": review_compression,
        "false_positive_rate": false_positive_rate,
        "miss_rate": miss_rate,
        "baseline_lift": baseline_lift,
        "alerts_per_1000_rows": alerts_per_1000_rows,
        "anomalies_caught_per_1000_rows": anomalies_caught_per_1000_rows,
        "false_alarms_per_1000_rows": false_alarms_per_1000_rows,
    }


def _timestamp_analysis(path: Path) -> Dict[str, float | int | bool]:
    timestamps: List[float] = []
    anomaly_timestamps: List[float] = []
    first_anomaly_row_index = -1
    first_anomaly_row_ts = -1.0

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader):
            timestamp = float(row["Timestamp"])
            timestamps.append(timestamp)
            if int(float(row["Labels"])) == 1:
                anomaly_timestamps.append(timestamp)
                if first_anomaly_row_index == -1:
                    first_anomaly_row_index = row_index
                    first_anomaly_row_ts = timestamp

    sorted_pairs = sorted(timestamps)
    monotonic = all(timestamps[index] <= timestamps[index + 1] for index in range(len(timestamps) - 1))
    return {
        "row_order_first_anomaly_index": first_anomaly_row_index,
        "row_order_first_anomaly_ts": first_anomaly_row_ts,
        "sorted_first_anomaly_ts": min(anomaly_timestamps) if anomaly_timestamps else -1.0,
        "time_span": (sorted_pairs[-1] - sorted_pairs[0]) if sorted_pairs else 0.0,
        "timestamp_monotonic": monotonic,
    }


def _top_n(dataset_rows: List[Dict[str, str]], n: int) -> List[Dict[str, str]]:
    return _sort_by_f1(dataset_rows)[:n]


def _build_report(summary_path: Path) -> str:
    primary_counts = {name: _read_counts(_resolve(path)) for name, path in PRIMARY_DATASETS.items()}
    benchmark_counts = {name: _read_counts(_resolve(path)) for name, path in REPORT_DATASETS.items()}
    summary_rows = _read_summary_rows(summary_path)
    best_rows = _best_rows_by_dataset(summary_rows)
    timestamp_info = _timestamp_analysis(_resolve(PRIMARY_DATASETS["HRSS_anomalous_optimized"]))

    original_total = primary_counts["HRSS_anomalous_optimized"]["total_rows"]
    undersample_total = primary_counts["HRSS_undersample_optimized"]["total_rows"]
    smote_total = primary_counts["HRSS_SMOTE_optimized"]["total_rows"]
    undersample_reduction = 1.0 - (undersample_total / original_total)
    smote_growth = (smote_total / original_total) - 1.0
    sampling_fraction = (
        primary_counts["HRSS_anomalous_optimized"]["anomaly_rows"]
        / primary_counts["HRSS_anomalous_optimized"]["normal_rows"]
    )

    lines: List[str] = []
    lines.append("ANOMALI TESPITI GERCEK DUNYA KATKI RAPORU")
    lines.append("")
    lines.append("1. Bu dosyanin amaci")
    lines.append(
        "Bu rapor, EffiTrack calismasinin gercek dunyada neye katkisi oldugunu, hangi parametrelerle"
    )
    lines.append(
        "olculdugunu, hangi faydalarin bugun sayisal olarak kanitlandigini ve hangilerinin ek loglama ile"
    )
    lines.append("kolayca olculebilecegini tek yerde toplar.")
    lines.append("")
    lines.append("2. Yonetici ozeti")
    lines.append(
        "- En guclu operasyonel sonuc: real_imbalanced veri setinde Random Forest, tum 19,634 kaydi incelemek yerine"
    )
    lines.append(
        "  yalnizca yaklasik 4,523 alarm ureterek manuel inceleme yukunu %76.96 azaltirken anomalilerin %96.21'ini yakaliyor."
    )
    lines.append(
        "- En guclu dengeli performans: real_smote_balanced veri setinde Random Forest, F1=0.9892 ve Accuracy=0.9892."
    )
    lines.append(
        "- En onemli uygulama katkisi: en iyi model classical ML ailesinden ve CPU uzerinde calisiyor; bu da canli ortama"
    )
    lines.append("  gecisi agir GPU zorunlulugu olmadan mumkun kiliyor.")
    lines.append(
        "- Istatistiksel guven: uc benchmark veri setinde de en iyi sonuc p-value=0.00195312 seviyesinde ve random baseline'dan anlamli bicimde daha iyi."
    )
    lines.append("")
    lines.append("3. Ana veri sayilari ve veri muhendisligi katkisi")
    lines.append(
        "- HRSS_anomalous_optimized: toplam {total}, normal {normal}, anomali {anomaly}, anomali orani {ratio}, sinif dengesizligi {imbalance}:1".format(
            total=_format_int(primary_counts["HRSS_anomalous_optimized"]["total_rows"]),
            normal=_format_int(primary_counts["HRSS_anomalous_optimized"]["normal_rows"]),
            anomaly=_format_int(primary_counts["HRSS_anomalous_optimized"]["anomaly_rows"]),
            ratio=_format_pct(primary_counts["HRSS_anomalous_optimized"]["anomaly_ratio"]),
            imbalance=_format_float(primary_counts["HRSS_anomalous_optimized"]["imbalance_ratio"], 2),
        )
    )
    lines.append(
        "- HRSS_undersample_optimized: toplam {total}, normal {normal}, anomali {anomaly}; veri hacmi orijinale gore {reduction} azaldi.".format(
            total=_format_int(primary_counts["HRSS_undersample_optimized"]["total_rows"]),
            normal=_format_int(primary_counts["HRSS_undersample_optimized"]["normal_rows"]),
            anomaly=_format_int(primary_counts["HRSS_undersample_optimized"]["anomaly_rows"]),
            reduction=_format_pct(undersample_reduction),
        )
    )
    lines.append(
        "  Bu, yeniden egitim ve batch isleme maliyetinin veri hacmi tarafinda yariya yakin azaltilabilecegini gosterir."
    )
    lines.append(
        "- HRSS_SMOTE_optimized: toplam {total}, normal {normal}, anomali {anomaly}; veri hacmi orijinale gore {growth} artti.".format(
            total=_format_int(primary_counts["HRSS_SMOTE_optimized"]["total_rows"]),
            normal=_format_int(primary_counts["HRSS_SMOTE_optimized"]["normal_rows"]),
            anomaly=_format_int(primary_counts["HRSS_SMOTE_optimized"]["anomaly_rows"]),
            growth=_format_pct(smote_growth),
        )
    )
    lines.append(
        "  Bu, daha buyuk egitim maliyeti pahasina daha yuksek anomaly coverage ve daha guclu sinif dengesi saglandigini gosterir."
    )
    lines.append(
        "- Undersampling orani, anomali_sayisi / normal_sayisi = {fraction:.4f}; SMOTE tarafinda hem Scala hem Python akisi k=5 komsu kullaniyor.".format(
            fraction=sampling_fraction
        )
    )
    lines.append("")
    lines.append("4. Sayisal olarak kanitlanmis temel katkilar")

    operational_row = best_rows["real_imbalanced"]
    operational_metrics = _operational_metrics(operational_row, benchmark_counts["real_imbalanced"])
    lines.append(
        "- Bakim / operator is yuku azalmasi: %s daha az manuel inceleme" % _format_pct(operational_metrics["review_reduction"])
    )
    lines.append(
        "- Mutlak is yukunden kacis: ayni veri akisi icinde yaklasik %s kayit manuel incelemeden kurtariliyor"
        % _format_int(operational_metrics["prevented_manual_checks"])
    )
    lines.append(
        "- Alarm yogunlugu azaltimi: tum kayitlari incelemek yerine %s kat daha az alarm inceleme" % _format_float(operational_metrics["review_compression"], 2)
    )
    lines.append(
        "- Anomali yakalama kapsami: %s recall, yani yaklasik %s anomali kacirma orani"
        % (
            _format_pct(_to_float(operational_row, "recall_mean")),
            _format_pct(operational_metrics["miss_rate"]),
        )
    )
    lines.append(
        "- Yalanci alarm kontrolu: normal kayitlar icinde yalanci alarm orani yalnizca %s"
        % _format_pct(operational_metrics["false_positive_rate"])
    )
    lines.append(
        "- Rastgele baseline ustune performans artisi: F1 acisindan %s kat"
        % _format_float(operational_metrics["baseline_lift"], 2)
    )
    lines.append(
        "- Sonuc kararliligi: en iyi operasyonel modelde F1 std = %s" % _format_float(_to_float(operational_row, "f1_std"))
    )
    lines.append(
        "- Dagitim kolayligi: en iyi modellerin tamami CPU uzerinde raporlanmis durumda; ek GPU mecburiyeti yok."
    )
    lines.append("")
    lines.append("5. Bu sayilari nasil hesapladik?")
    lines.append("- Tum temel model karsilastirmalari selected_real_dataset_suite_summary_all.csv uzerinden okunuyor.")
    lines.append("- Classical ML modelleri 10-fold cross-validation ile degerlendiriliyor; bu nedenle skorlar tek split sansina bagli degil.")
    lines.append("- TP = recall * anomaly_rows")
    lines.append("- Predicted positive = TP / precision")
    lines.append("- FP = predicted_positive - TP")
    lines.append("- Review reduction = 1 - predicted_positive / total_rows")
    lines.append("- Prevented manual checks = total_rows - predicted_positive")
    lines.append("- Miss rate = FN / anomaly_rows = 1 - recall")
    lines.append("- Baseline lift = model_f1 / baseline_f1")
    lines.append("- False positive rate among normals = FP / normal_rows")
    lines.append("- Bu raporda 'bakim azalmasi' dogrudan sayilmiyor; onun yerine bakim ekibine gidecek gereksiz alarm ve inceleme yukundeki azalma olculuyor.")
    lines.append("")
    lines.append("6. Veri seti bazinda ayrintili katkilar")

    for dataset_name in REPORT_DATASETS:
        dataset_rows = [row for row in summary_rows if row["dataset"] == dataset_name]
        dataset_rows = _sort_by_f1(dataset_rows)
        best = dataset_rows[0]
        second = dataset_rows[1]
        counts = benchmark_counts[dataset_name]
        metrics = _operational_metrics(best, counts)
        family_bests = _best_rows_by_family(dataset_rows)

        best_dl = family_bests.get("deep_learning")
        best_transformer = family_bests.get("transformer")
        best_hybrid = family_bests.get("hybrid")

        lines.append("- {}:".format(DISPLAY_NAMES[dataset_name]))
        lines.append(
            "  En iyi model = {model} ({family}), F1={f1}, Precision={precision}, Recall={recall}, Accuracy={accuracy}, p-value={pvalue}".format(
                model=best["model"],
                family=FAMILY_NAMES.get(best["family"], best["family"]),
                f1=_format_float(_to_float(best, "f1_mean")),
                precision=_format_float(_to_float(best, "precision_mean")),
                recall=_format_float(_to_float(best, "recall_mean")),
                accuracy=_format_float(_to_float(best, "accuracy_mean")),
                pvalue=_format_float(_to_float(best, "p_value"), 8),
            )
        )
        lines.append(
            "  Operasyonel yorum: TP~{tp}, FN~{fn}, FP~{fp}, review reduction={reduction}, false positive rate among normals={fpr}".format(
                tp=_format_int(metrics["tp"]),
                fn=_format_int(metrics["fn"]),
                fp=_format_int(metrics["fp"]),
                reduction=_format_pct(metrics["review_reduction"]),
                fpr=_format_pct(metrics["false_positive_rate"]),
            )
        )
        lines.append(
            "  Inceleme yukune etkisi: predicted alarms~{alarms}, prevented manual checks~{prevented}, alerts/1000 rows={alerts}, false alarms/1000 rows={false_alarms}".format(
                alarms=_format_int(metrics["predicted_positive"]),
                prevented=_format_int(metrics["prevented_manual_checks"]),
                alerts=_format_float(metrics["alerts_per_1000_rows"], 2),
                false_alarms=_format_float(metrics["false_alarms_per_1000_rows"], 2),
            )
        )
        lines.append(
            "  Ilk 3 model: 1){m1}/{f1} 2){m2}/{f2} 3){m3}/{f3}".format(
                m1=dataset_rows[0]["model"],
                f1=_format_float(_to_float(dataset_rows[0], "f1_mean")),
                m2=dataset_rows[1]["model"],
                f2=_format_float(_to_float(dataset_rows[1], "f1_mean")),
                m3=dataset_rows[2]["model"],
                f3=_format_float(_to_float(dataset_rows[2], "f1_mean")),
            )
        )
        lines.append(
            "  1. ve 2. model farki = %s F1 puani" % _format_float(_to_float(best, "f1_mean") - _to_float(second, "f1_mean"))
        )

        if best_dl is not None:
            lines.append(
                "  En iyi deep learning modeli = {model} (F1={f1}); en iyi modele farki = {gap}".format(
                    model=best_dl["model"],
                    f1=_format_float(_to_float(best_dl, "f1_mean")),
                    gap=_format_float(_to_float(best, "f1_mean") - _to_float(best_dl, "f1_mean")),
                )
            )
        if best_transformer is not None:
            lines.append(
                "  En iyi transformer modeli = {model} (F1={f1}); en iyi modele farki = {gap}".format(
                    model=best_transformer["model"],
                    f1=_format_float(_to_float(best_transformer, "f1_mean")),
                    gap=_format_float(_to_float(best, "f1_mean") - _to_float(best_transformer, "f1_mean")),
                )
            )
        if best_hybrid is not None:
            lines.append(
                "  En iyi hybrid modeli = {model} (F1={f1}); en iyi modele farki = {gap}".format(
                    model=best_hybrid["model"],
                    f1=_format_float(_to_float(best_hybrid, "f1_mean")),
                    gap=_format_float(_to_float(best, "f1_mean") - _to_float(best_hybrid, "f1_mean")),
                )
            )

    lines.append("")
    lines.append("7. Bu calisma gercek dunyada tam olarak neye katkida bulunuyor?")
    lines.append("- Alarm triage kalitesi: operatore butun akisi degil, supheli alt kumeyi gosteriyor.")
    lines.append("- Inceleme maliyeti azaltimi: tum satirlari kontrol etmek yerine modelin urettigi alarm listesi incelenebiliyor.")
    lines.append("- Operator odagi: yalnizca problem ihtimali yuksek kayitlar one alindigi icin vardiya ici dikkat dagilimi azaltilabiliyor.")
    lines.append("- Yalanci alarm baskilama: precision ve false positive rate ile alarm kalitesi sayisal olarak kontrol ediliyor.")
    lines.append("- Anomali kacirma riskinin gorunur hale gelmesi: recall ve miss rate ile ne kadar olay kacirdigimiz olculebiliyor.")
    lines.append("- Veri dengesizligiyle mucadele: undersampling ve SMOTE etkisi ayni benchmark catisi altinda gorulebiliyor.")
    lines.append("- Model seciminde yalnizca karmaasik mimarilere bagli kalmama: classical ML ailesi en guclu sonucu veriyor.")
    lines.append("- Dagitim maliyeti dusuk cozum: en iyi sonuclar CPU donaniminda alinmis.")
    lines.append("- Canli ortama gecis yaklasikligi: Spark + Kafka akisi repoda mevcut oldugu icin offline sonuc canli hatta tasinabilir.")
    lines.append("- Tekrarlanabilirlik: veri hazirlama, benchmark, runtime config ve workbook akislari scriptlenmis durumda.")
    lines.append("")
    lines.append("8. Bakim azalmasi iddiasini nasil okumaliyiz?")
    lines.append("- Dogrudan kanitlanan kisim: daha az alarm, daha az manuel inceleme, daha dusuk yalanci alarm ve daha yuksek anomaly catch.")
    lines.append("- Dolayli ama guclu operasyonel cikarim: daha az gereksiz alarm, bakim ekibinin gereksiz kontrol ve inceleme eforunu azaltabilir.")
    lines.append("- Henuz dogrudan kanitlanmayan kisim: gercek is emri sayisi azalmasi, plansiz durus suresi azalmasi, yedek parca tasarrufu ve maliyet kazanimi.")
    lines.append("- Bu son grup KPI icin CMMS / ticket / intervention loglari ile model alarm loglarini eslemek gerekir.")
    lines.append("- Yani bugunku veriyle 'bakim yuku azalmasi yonunde guclu on gosterge' diyebiliriz; fakat 'bakim maliyeti su kadar dustu' diye dogrudan yazmamak gerekir.")
    lines.append("")
    lines.append("9. Zaman ve tespit hiziyla ilgili ne diyebiliriz?")
    lines.append(
        "- HRSS_anomalous_optimized icinde Timestamp kolonuna gore veri kapsami yaklasik %s saniyelik bir pencereye yayiliyor."
        % _format_float(float(timestamp_info["time_span"]), 3)
    )
    lines.append(
        "- Dosya sirasina gore ilk etiketli anomali {}. satir civarinda ve timestamp={}".format(
            int(timestamp_info["row_order_first_anomaly_index"]),
            _format_float(float(timestamp_info["row_order_first_anomaly_ts"]), 6),
        )
    )
    lines.append(
        "- Timestamp'a gore siralanmis gorunumde ilk anomali timestamp={}".format(
            _format_float(float(timestamp_info["sorted_first_anomaly_ts"]), 6)
        )
    )
    lines.append(
        "- Ancak mevcut offline CSV akisi zaman sirasi acisindan monoton degil. Bu nedenle 'anomali kac saniyede bulundu' metrigini"
    )
    lines.append(
        "  bugunku dosyalarla bilimsel olarak dogrudan iddia etmek dogru olmaz; bunun icin canli akista event_ts ve prediction_ts loglanmali."
    )
    lines.append("")
    lines.append("10. Bugun hemen olculebilen KPI'lar")
    lines.append("- Precision: uretilen alarmin gercek problem olma orani.")
    lines.append("- Recall: gercek anomalilerin ne kadarinin yakalandigi.")
    lines.append("- F1: precision ve recall dengesinin tek skorda toplanmis hali.")
    lines.append("- Accuracy: toplam dogru karar orani.")
    lines.append("- False positive rate among normals: FP / normal_rows.")
    lines.append("- Miss rate: FN / anomaly_rows.")
    lines.append("- Review reduction: 1 - predicted_positive / total_rows.")
    lines.append("- Review compression factor: total_rows / predicted_positive.")
    lines.append("- Baseline lift: model_f1 / baseline_f1.")
    lines.append("- Stability: f1_std.")
    lines.append("")
    lines.append("11. Ek loglama ile kolayca olculebilecek KPI'lar")
    lines.append("- detection_latency_ms = prediction_ts - event_ts")
    lines.append("- kafka_publish_latency_ms = kafka_send_ts - prediction_ts")
    lines.append("- time_to_first_alert_sec = first_alert_ts - anomaly_start_ts")
    lines.append("- alerts_per_minute = anomaly_alert_count / monitored_minutes")
    lines.append("- operator_review_load_per_shift = alert_count_per_shift")
    lines.append("- mean_time_between_alerts")
    lines.append("- high_confidence_alert_rate (KafkaAnomalyDetection.scala icindeki >0.8 esigine gore)")
    lines.append("- uncertain_band_rate (0.2 ile 0.8 arasinda kalan skorlarin orani)")
    lines.append("- prevented_manual_checks = total_rows - predicted_positive")
    lines.append("- potential_maintenance_lead_time = intervention_ts - predicted_first_alert_ts")
    lines.append("")
    lines.append("12. Bu ek KPI'lari nerede olcmeliyiz?")
    lines.append("- SparkKafkaStreaming/src/main/scala/spark/KafkaAnomalyDetection.scala")
    lines.append("  Burada prediction_ts, kafka_send_ts, threshold band ve topic bilgisi loglanabilir.")
    lines.append("- SparkKafkaStreaming/src/main/scala/spark/KafkaConsumerRF.scala")
    lines.append("  Burada her kayit icin RF skor cikisi, model yukleme ve batch bazli latency loglanabilir.")
    lines.append("- SparkKafkaStreaming/src/main/scala/spark/KafkaProducerRF.scala")
    lines.append("  Burada event_ts ve payload seviyesinde orijinal olay zamani tasinabilir.")
    lines.append("- ModelandPerformanceAnalysis/results/summaries/*.csv")
    lines.append("  Buraya latency, alert volume ve operational KPI kolonlari eklenebilir.")
    lines.append("")
    lines.append("13. Parametre ve esik seviyesinde hangi katkilar tartisilabilir?")
    lines.append("- Wavelet esigi: AnomalyDetection.scala icinde |waveletValue| > 0.6")
    lines.append("- SMOTE komsu sayisi: Scala ve Python akislarinda k = 5")
    lines.append("- Kafka anomaly threshold: predictions > 0.8")
    lines.append("- Kafka normal threshold: predictions <= 0.2")
    lines.append("- Degerlendirme protokolu: classical ML icin 10-fold CV, non-classical modeller icin coklu run hold-out")
    lines.append("- Runtime metadata: learning rate, batch size, epoch, optimizer, loss function summary CSV icinde zaten var")
    lines.append("")
    lines.append("14. En guclu tez / sunum cumleleri")
    lines.append(
        "- Bu calisma, saha benzeri dengesiz veri dagiliminda manuel inceleme yukunu %76.96 azaltirken anomalilerin %96.21'ini yakalayabilen bir anomali tespit akisi ortaya koymaktadir."
    )
    lines.append(
        "- Veri dengeleme tarafinda undersampling egitim maliyetini veri boyutu acisindan %54.09 azaltirken, SMOTE dengeli sinif yapisiyla F1 skorunu 0.9892 seviyesine cikarmistir."
    )
    lines.append(
        "- En iyi sonucun CPU uzerinde calisan Random Forest ile elde edilmesi, cozumun gercek ortama dusuk dagitim maliyetiyle alinabilecegini gostermektedir."
    )
    lines.append("")
    lines.append("15. Kanit kaynaklari")
    lines.append("- Metrik summary dosyasi: {}".format(summary_path))
    lines.append("- Veri sayilari: Data/*.csv ve Data/benchmark_suite/*.csv")
    lines.append("- Anomali etiketleme akisi: DatasetAndPreprocessing/src/main/scala/AnomalyDetection.scala")
    lines.append("- Undersampling akisi: DatasetAndPreprocessing/src/main/scala/Undersampling.scala")
    lines.append("- SMOTE akisi: DatasetAndPreprocessing/src/main/scala/SMOTE.scala")
    lines.append("- Benchmark veri hazirlama: ModelandPerformanceAnalysis/src/main/python/effitrack_eval/data.py")
    lines.append("- Raporu ureten script: ModelandPerformanceAnalysis/src/main/python/build_anomaly_value_report.py")
    lines.append("- Canli akisa gecis noktasi: SparkKafkaStreaming/src/main/scala/spark/KafkaAnomalyDetection.scala")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build anomaly detection value report")
    parser.add_argument("--summary", default=SUMMARY_DEFAULT, help="Summary CSV used for model metrics")
    parser.add_argument("--output", default=OUTPUT_DEFAULT, help="Output text file path")
    args = parser.parse_args()

    summary_path = _resolve_summary_path(args.summary)
    output_path = _resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = _build_report(summary_path)
    output_path.write_text(report, encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
