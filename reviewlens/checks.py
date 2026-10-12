"""Label-issue detection: compare human annotations against model predictions.

Every flag gets a score in [0, 1] that is *calibrated on the dataset itself*:

* "the model sees an object nobody labelled" is weighted by how often the model's
  predictions of that class, at that confidence, actually agree with the human
  labels elsewhere in the same dataset (empirical precision);
* "a labelled object the model can't find" is weighted by how often the model
  finds objects of that class and size elsewhere (empirical recall).

So a model that is weak on a class (domain shift, rare class, tiny objects)
automatically produces quieter flags for that class instead of flooding the
review queue. No training, no extra labels needed.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np

from .coco import Dataset
from .geometry import iou_matrix, size_bucket

ISSUE_TYPES = ("missing_object", "wrong_class", "loose_box", "spurious_box", "duplicate_box")

# Thresholds. Kept in one place so they are easy to tune per customer.
MATCH_IOU = 0.5            # human box and model box describe the same object
MISSING_MIN_SCORE = 0.35   # only trust fairly confident model boxes as "missing" evidence
MISSING_MAX_IOU = 0.3      # model box overlaps no human box (any class) beyond this
WRONG_CLASS_IOU = 0.6
WRONG_CLASS_MIN_SCORE = 0.35
LOOSE_MIN_SCORE = 0.5
LOOSE_IOU_RANGE = (0.2, 0.62)
SPURIOUS_ANY_IOU = 0.3     # no model box of any class overlaps the human box this much
SPURIOUS_MIN_SCORE = 0.05
DUPLICATE_IOU = 0.85
SCORE_BUCKETS = (0.35, 0.5, 0.7, 0.85, 1.01)
PRIOR_STRENGTH = 10.0      # pseudo-counts for shrinking per-class rates to the global rate


@dataclass
class Issue:
    image_id: int
    type: str
    score: float
    bbox: list[float]
    category_id: int
    ann_id: int | None = None
    suggested_category_id: int | None = None
    model_score: float | None = None
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _bucket(score: float) -> int:
    for i, edge in enumerate(SCORE_BUCKETS[1:]):
        if score < edge:
            return i
    return len(SCORE_BUCKETS) - 2


def _arrays(items: list[dict], with_score: bool = False):
    boxes = np.array([it["bbox"] for it in items], dtype=float).reshape(-1, 4)
    cats = np.array([it["category_id"] for it in items], dtype=int)
    if with_score:
        return boxes, cats, np.array([it["score"] for it in items], dtype=float)
    return boxes, cats


def _human_anns(ds: Dataset, img_id: int) -> tuple[list[dict], list[dict]]:
    anns = ds.anns_by_image.get(img_id, [])
    return [a for a in anns if not a.get("iscrowd")], [a for a in anns if a.get("iscrowd")]


class Calibration:
    """Empirical model precision (per class x score bucket) and recall (per class x size)."""

    def __init__(self) -> None:
        self.prec_hits = defaultdict(float)
        self.prec_total = defaultdict(float)
        self.rec_hits = defaultdict(float)
        self.rec_total = defaultdict(float)

    def fit(self, ds: Dataset, preds: dict[int, list[dict]]) -> Calibration:
        for img_id in ds.images:
            gts, crowd = _human_anns(ds, img_id)
            pr = [p for p in preds.get(img_id, []) if p["score"] >= SCORE_BUCKETS[0]]
            g_boxes, g_cats = _arrays(gts)
            p_boxes, p_cats, p_scores = _arrays(pr, with_score=True)
            iou = iou_matrix(p_boxes, g_boxes)
            same = (p_cats[:, None] == g_cats[None, :]) if len(pr) and len(gts) else np.zeros_like(iou, bool)
            crowd_iou = iou_matrix(p_boxes, _arrays(crowd)[0]) if crowd else np.zeros((len(pr), 0))
            for i, p in enumerate(pr):
                if crowd_iou.shape[1] and crowd_iou[i].max() > 0.1:
                    continue  # predictions inside crowd regions say nothing
                hit = float(((iou[i] >= MATCH_IOU) & same[i]).any()) if len(gts) else 0.0
                key = (p["category_id"], _bucket(p["score"]))
                self.prec_hits[key] += hit
                self.prec_total[key] += 1
                self.prec_hits[("*", key[1])] += hit
                self.prec_total[("*", key[1])] += 1
            # recall: is each human box found by a same-class prediction (score >= 0.25)?
            strong = p_scores >= 0.25 if len(pr) else np.zeros(0, bool)
            for j, g in enumerate(gts):
                found = float(((iou[:, j] >= MATCH_IOU) & same[:, j] & strong).any()) if len(pr) else 0.0
                sb = size_bucket(g["bbox"])
                self.rec_hits[(g["category_id"], sb)] += found
                self.rec_total[(g["category_id"], sb)] += 1
                self.rec_hits[("*", sb)] += found
                self.rec_total[("*", sb)] += 1
        return self

    @staticmethod
    def _shrunk(hits, total, key, global_key, default):
        g_rate = hits[global_key] / total[global_key] if total[global_key] else default
        return (hits[key] + PRIOR_STRENGTH * g_rate) / (total[key] + PRIOR_STRENGTH)

    def precision(self, cat_id: int, score: float) -> float:
        b = _bucket(score)
        return self._shrunk(self.prec_hits, self.prec_total, (cat_id, b), ("*", b), 0.5)

    def recall(self, cat_id: int, bbox) -> float:
        sb = size_bucket(bbox)
        return self._shrunk(self.rec_hits, self.rec_total, (cat_id, sb), ("*", sb), 0.5)


def find_issues(ds: Dataset, preds: dict[int, list[dict]], calib: Calibration | None = None) -> list[Issue]:
    calib = calib or Calibration().fit(ds, preds)
    issues: list[Issue] = []
    for img_id in ds.images:
        issues.extend(_image_issues(ds, img_id, preds.get(img_id, []), calib))
    return issues


def _image_issues(ds: Dataset, img_id: int, pr: list[dict], calib: Calibration) -> list[Issue]:
    out: list[Issue] = []
    gts, crowd = _human_anns(ds, img_id)
    pr = sorted((p for p in pr if p["score"] >= SPURIOUS_MIN_SCORE), key=lambda p: -p["score"])
    g_boxes, g_cats = _arrays(gts)
    p_boxes, p_cats, p_scores = _arrays(pr, with_score=True)
    iou = iou_matrix(p_boxes, g_boxes)  # (P, G)
    same = (p_cats[:, None] == g_cats[None, :]) if len(pr) and len(gts) else np.zeros_like(iou, bool)
    crowd_boxes = _arrays(crowd)[0]
    name = ds.cat_name

    # --- duplicate human boxes (model-free) -------------------------------
    gg = iou_matrix(g_boxes, g_boxes)
    dup_of: set[int] = set()
    for a in range(len(gts)):
        for b in range(a + 1, len(gts)):
            if g_cats[a] == g_cats[b] and gg[a, b] >= DUPLICATE_IOU and b not in dup_of:
                dup_of.add(b)
                out.append(Issue(img_id, "duplicate_box", 0.9, gts[b]["bbox"], int(g_cats[b]),
                                 ann_id=gts[b].get("id"),
                                 detail=f"two '{name(g_cats[b])}' boxes overlap {gg[a, b]:.0%}"))

    # --- missing objects: confident model box that no human box covers ----
    claimed = []  # class-agnostic NMS over the flags we raise
    for i, p in enumerate(pr):
        if p["score"] < MISSING_MIN_SCORE:
            break
        if len(gts) and iou[i].max() > MISSING_MAX_IOU:
            continue
        if len(crowd_boxes) and iou_matrix(p_boxes[i:i + 1], crowd_boxes).max() > 0.1:
            continue
        if claimed and iou_matrix(p_boxes[i:i + 1], np.array(claimed)).max() > 0.5:
            continue
        claimed.append(p_boxes[i])
        s = calib.precision(p["category_id"], p["score"])
        out.append(Issue(img_id, "missing_object", s, list(map(float, p["bbox"])), int(p["category_id"]),
                         model_score=float(p["score"]),
                         detail=f"model sees '{name(p['category_id'])}' ({p['score']:.2f}) with no label"))

    for j, g in enumerate(gts):
        if j in dup_of:
            continue
        col = iou[:, j] if len(pr) else np.zeros(0)
        same_col = same[:, j] if len(pr) else np.zeros(0, bool)
        same_match = (col >= MATCH_IOU) & same_col
        best_same = float(p_scores[same_match].max()) if same_match.any() else 0.0

        # --- wrong class: a confident other-class box sits on this label ----
        other = (col >= WRONG_CLASS_IOU) & ~same_col & (p_scores >= WRONG_CLASS_MIN_SCORE)
        if other.any() and best_same < 0.25:
            k = int(np.argmax(np.where(other, p_scores, -1)))
            s = calib.precision(int(p_cats[k]), float(p_scores[k])) * (1.0 - best_same)
            out.append(Issue(img_id, "wrong_class", s, g["bbox"], int(g_cats[j]), ann_id=g.get("id"),
                             suggested_category_id=int(p_cats[k]), model_score=float(p_scores[k]),
                             detail=f"labelled '{name(g_cats[j])}', model says '{name(p_cats[k])}' "
                                    f"({p_scores[k]:.2f})"))
            continue

        # --- spurious: no model box of ANY class anywhere near it -----------
        near_any = (col >= SPURIOUS_ANY_IOU).any() if len(pr) else False
        if not near_any:
            r = calib.recall(int(g_cats[j]), g["bbox"])
            out.append(Issue(img_id, "spurious_box", 0.9 * r, g["bbox"], int(g_cats[j]), ann_id=g.get("id"),
                             detail=f"no model box near this '{name(g_cats[j])}' "
                                    f"(model usually finds {r:.0%} of these)"))
            continue

        # --- loose box: same object, but the human box is badly fitted -------
        cand = same_col & (p_scores >= LOOSE_MIN_SCORE) & (col >= LOOSE_IOU_RANGE[0])
        if cand.any():
            k = int(np.argmax(np.where(cand, col, -1)))
            # the best-fitting confident model box is still a poor fit, and it
            # does not belong better to another human box
            if col[k] < LOOSE_IOU_RANGE[1] and iou[k].argmax() == j:
                tight = (LOOSE_IOU_RANGE[1] - col[k]) / (LOOSE_IOU_RANGE[1] - LOOSE_IOU_RANGE[0])
                s = calib.precision(int(p_cats[k]), float(p_scores[k])) * (0.5 + 0.5 * tight)
                out.append(Issue(img_id, "loose_box", s, g["bbox"], int(g_cats[j]), ann_id=g.get("id"),
                                 model_score=float(p_scores[k]),
                                 detail=f"'{name(g_cats[j])}' box only overlaps model box by {col[k]:.0%}"))
    return out


def rank_files(ds: Dataset, issues: list[Issue]) -> list[dict]:
    """Per-file risk = noisy-OR of issue scores (P(at least one real error))."""
    by_img: dict[int, list[Issue]] = defaultdict(list)
    for it in issues:
        by_img[it.image_id].append(it)
    rows = []
    for img_id, img in ds.images.items():
        its = sorted(by_img.get(img_id, []), key=lambda x: -x.score)
        risk = 1.0 - float(np.prod([1.0 - min(i.score, 0.999) for i in its])) if its else 0.0
        rows.append({
            "image_id": img_id,
            "file_name": img.get("file_name"),
            "risk": risk,
            "expected_errors": float(sum(i.score for i in its)),
            "n_issues": len(its),
            "n_labels": len(ds.anns_by_image.get(img_id, [])),
            "top_issue": its[0].type if its else "",
            "issues": its,
        })
    rows.sort(key=lambda r: (-r["risk"], -r["expected_errors"]))
    return rows
