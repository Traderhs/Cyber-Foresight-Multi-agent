from __future__ import annotations

import math
import random
import statistics
from collections import Counter
from typing import Iterable, Sequence


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    left, right = set(a), set(b)
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def pairwise_mean_jaccard(sets: Sequence[Iterable[str]]) -> float | None:
    values: list[float] = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            values.append(jaccard(sets[i], sets[j]))
    return statistics.mean(values) if values else None


def agreement_rate(values: Sequence[object]) -> float | None:
    if not values:
        return None
    counts = Counter(values)
    return max(counts.values()) / len(values)


def exact_pairwise_agreement(values: Sequence[object]) -> float | None:
    if len(values) < 2:
        return None
    matches = 0
    total = 0
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            total += 1
            matches += int(values[i] == values[j])
    return matches / total if total else None


def population_variance(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("variance requires at least one value")
    mean = statistics.mean(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def average_ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        average = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = average
        i = j
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mx, my = statistics.mean(x), statistics.mean(y)
    sx = math.sqrt(sum((v - mx) ** 2 for v in x))
    sy = math.sqrt(sum((v - my) ** 2 for v in y))
    if sx == 0 or sy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True)) / (sx * sy)


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    return pearson(average_ranks(x), average_ranks(y))


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    concordant = discordant = ties_x = ties_y = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dx = (x[i] > x[j]) - (x[i] < x[j])
            dy = (y[i] > y[j]) - (y[i] < y[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + ties_x) * (concordant + discordant + ties_y)
    )
    if denominator == 0:
        return None
    return (concordant - discordant) / denominator


def weighted_kappa(
    left: Sequence[int],
    right: Sequence[int],
    *,
    categories: Sequence[int] | None = None,
) -> float | None:
    if len(left) != len(right) or not left:
        return None
    cats = list(categories or sorted(set(left) | set(right)))
    if len(cats) < 2:
        return 1.0
    index = {value: i for i, value in enumerate(cats)}
    n = len(left)
    matrix = [[0.0 for _ in cats] for _ in cats]
    row = [0.0 for _ in cats]
    col = [0.0 for _ in cats]
    for a, b in zip(left, right, strict=True):
        if a not in index or b not in index:
            raise ValueError("value not present in weighted-kappa categories")
        i, j = index[a], index[b]
        matrix[i][j] += 1.0 / n
        row[i] += 1.0 / n
        col[j] += 1.0 / n
    max_distance = max(1, len(cats) - 1)
    observed = expected = 0.0
    for i in range(len(cats)):
        for j in range(len(cats)):
            weight = ((i - j) / max_distance) ** 2
            observed += weight * matrix[i][j]
            expected += weight * row[i] * col[j]
    if expected == 0:
        return 1.0 if observed == 0 else None
    return 1.0 - observed / expected


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    seed: int = 20260916,
    repetitions: int = 2000,
    alpha: float = 0.05,
) -> tuple[float, float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(repetitions):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(statistics.mean(sample))
    means.sort()
    low_idx = max(0, int((alpha / 2) * repetitions))
    high_idx = min(repetitions - 1, int((1 - alpha / 2) * repetitions) - 1)
    return means[low_idx], means[high_idx]


def paired_cohen_dz(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    differences = [a - b for a, b in zip(left, right, strict=True)]
    sd = statistics.stdev(differences)
    if sd == 0:
        return 0.0
    return statistics.mean(differences) / sd


def median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def mean(values: Sequence[float]) -> float | None:
    return statistics.mean(values) if values else None
