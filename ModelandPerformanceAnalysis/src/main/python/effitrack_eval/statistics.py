from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def _rank_absolute_differences(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    ranks = np.zeros(len(values), dtype=np.float64)

    index = 0
    while index < len(sorted_values):
        end = index
        while end + 1 < len(sorted_values) and np.isclose(sorted_values[end + 1], sorted_values[index]):
            end += 1
        average_rank = (index + 1 + end + 1) / 2.0
        ranks[order[index : end + 1]] = average_rank
        index = end + 1

    return ranks


def exact_wilcoxon_signed_rank(model_scores: List[float], baseline_scores: List[float]) -> Dict[str, Any]:
    x = np.asarray(model_scores, dtype=np.float64)
    y = np.asarray(baseline_scores, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("Model scores and baseline scores must have the same shape.")

    differences = x - y
    non_zero_mask = ~np.isclose(differences, 0.0)
    differences = differences[non_zero_mask]

    if differences.size == 0:
        return {
            "test": "exact_wilcoxon_signed_rank",
            "n": 0,
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "direction": "no_difference",
            "mean_difference": 0.0,
            "median_difference": 0.0,
        }

    ranks = _rank_absolute_differences(np.abs(differences))
    w_plus = float(ranks[differences > 0].sum())
    w_minus = float(ranks[differences < 0].sum())
    statistic = float(min(w_plus, w_minus))

    total_rank_sum = float(ranks.sum())
    sampled_statistics: List[float] = []

    for mask in range(1 << len(ranks)):
        plus_sum = 0.0
        for index, rank in enumerate(ranks):
            if mask & (1 << index):
                plus_sum += float(rank)
        minus_sum = total_rank_sum - plus_sum
        sampled_statistics.append(min(plus_sum, minus_sum))

    null_distribution = np.asarray(sampled_statistics, dtype=np.float64)
    p_value = float(np.mean(null_distribution <= statistic + 1e-12))
    mean_difference = float(differences.mean())
    median_difference = float(np.median(differences))

    if mean_difference > 0:
        direction = "better_than_random"
    elif mean_difference < 0:
        direction = "worse_than_random"
    else:
        direction = "no_difference"

    return {
        "test": "exact_wilcoxon_signed_rank",
        "n": int(len(differences)),
        "statistic": statistic,
        "w_plus": w_plus,
        "w_minus": w_minus,
        "p_value": p_value,
        "significant": bool(p_value < 0.05),
        "direction": direction,
        "mean_difference": mean_difference,
        "median_difference": median_difference,
    }


def build_random_baseline(y_true: np.ndarray, positive_rate: float, seed: int) -> np.ndarray:
    clipped_rate = min(max(float(positive_rate), 0.0), 1.0)
    rng = np.random.default_rng(seed)
    return rng.binomial(1, clipped_rate, size=len(y_true)).astype(int)
