"""Benchmark: how many known errors does a reviewer catch for a given review budget?"""
from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np

from .checks import ISSUE_TYPES, Issue
from .coco import Dataset
from .geometry import iou_matrix

BUDGETS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)


def catch_curve(order: list[int], errors_per_image: Counter, total_errors: int) -> list[float]:
    """Fraction of all errors sitting in the first b% of files, for each budget b."""
    n = len(order)
    cum = np.cumsum([errors_per_image.get(i, 0) for i in order])
    return [float(cum[max(1, round(b * n)) - 1] / total_errors) for b in BUDGETS]


def effort_to_catch(curve_order: list[int], errors_per_image: Counter, total: int, target: float) -> float:
    cum = np.cumsum([errors_per_image.get(i, 0) for i in curve_order]) / total
    return float((np.argmax(cum >= target) + 1) / len(curve_order))


def _match_issue(err: dict, issues: list[Issue]) -> tuple[bool, bool]:
    """(located, located with the right issue type)."""
    if not issues:
        return False, False
    ious = iou_matrix(np.array([err["bbox"]]), np.array([i.bbox for i in issues]))[0]
    hits = [it for it, v in zip(issues, ious) if v >= 0.5]
    return bool(hits), any(h.type == err["type"] for h in hits)


def benchmark(noisy: Dataset, key: list[dict], rows: list[dict], issues: list[Issue],
              annotator_of: dict[int, int], seed: int = 0) -> dict:
    errs_per_img = Counter(e["image_id"] for e in key)
    total = len(key)
    ids = list(noisy.images)

    reviewlens_order = [r["image_id"] for r in rows]
    busiest_order = sorted(ids, key=lambda i: -len(noisy.anns_by_image.get(i, [])))
    rng = random.Random(seed)
    rand_curves, rand_effort = [], []
    for _ in range(200):
        o = ids[:]
        rng.shuffle(o)
        rand_curves.append(catch_curve(o, errs_per_img, total))
        rand_effort.append(effort_to_catch(o, errs_per_img, total, 0.8))

    curves = {
        "ReviewLens ranking": catch_curve(reviewlens_order, errs_per_img, total),
        "Random order (status quo)": list(np.mean(rand_curves, axis=0)),
        "Most labels first": catch_curve(busiest_order, errs_per_img, total),
    }
    effort80 = {
        "ReviewLens ranking": effort_to_catch(reviewlens_order, errs_per_img, total, 0.8),
        "Random order (status quo)": float(np.mean(rand_effort)),
        "Most labels first": effort_to_catch(busiest_order, errs_per_img, total, 0.8),
    }

    # object-level: was the exact error pointed at?
    by_img: dict[int, list[Issue]] = defaultdict(list)
    for it in issues:
        by_img[it.image_id].append(it)
    per_type = {t: {"injected": 0, "located": 0, "typed": 0} for t in ISSUE_TYPES}
    for e in key:
        loc, typed = _match_issue(e, by_img[e["image_id"]])
        per_type[e["type"]]["injected"] += 1
        per_type[e["type"]]["located"] += loc
        per_type[e["type"]]["typed"] += typed

    # precision of the top-N issues (non-hits may still be genuine errors in the source data)
    ranked = sorted(issues, key=lambda i: -i.score)
    key_by_img = defaultdict(list)
    for e in key:
        key_by_img[e["image_id"]].append(e)

    def is_hit(it: Issue) -> bool:
        es = key_by_img.get(it.image_id, [])
        if not es:
            return False
        return bool((iou_matrix(np.array([it.bbox]), np.array([e["bbox"] for e in es]))[0] >= 0.5).any())

    hit_flags = [is_hit(it) for it in ranked]
    precision_at = {n: float(np.mean(hit_flags[:n])) for n in (25, 50, 100, 200) if n <= len(ranked)}

    # annotator attribution: estimated vs true errors per image, per annotator
    est = defaultdict(float)
    true = defaultdict(int)
    n_img = Counter(annotator_of.values())
    for r in rows:
        est[annotator_of[r["image_id"]]] += r["expected_errors"]
    for e in key:
        true[e["annotator"]] += 1
    annotators = [
        {"annotator": f"annotator_{a + 1}", "images": n_img[a],
         "true_errors_per_100": 100 * true[a] / n_img[a], "flag_score_per_100": 100 * est[a] / n_img[a]}
        for a in sorted(n_img)
    ]

    return {
        "n_images": len(ids),
        "n_labels": noisy.n_annotations,
        "n_injected": total,
        "n_images_with_errors": len(errs_per_img),
        "budgets": list(BUDGETS),
        "curves": curves,
        "effort_for_80pct": effort80,
        "per_type": per_type,
        "issue_precision_at": precision_at,
        "n_issues": len(issues),
        "annotators": annotators,
        "injected_by_type": dict(Counter(e["type"] for e in key)),
    }
