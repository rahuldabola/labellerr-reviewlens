"""Inject realistic annotator mistakes into a clean COCO dataset, with an answer key.

Used only for benchmarking: it lets us measure how many *known* errors a
reviewer would catch for a given review budget.

The simulation splits images across annotators (as a Labellerr project would)
and makes one annotator careless (3x error rate), because in practice errors
cluster by person and by batch rather than being spread uniformly.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass

from .coco import Dataset

BASE_RATES = {  # per human box, except spurious which is per image
    "missing_object": 0.05,
    "wrong_class": 0.03,
    "loose_box": 0.03,
    "duplicate_box": 0.01,
    "spurious_box": 0.04,
}


@dataclass
class NoiseConfig:
    n_annotators: int = 6
    careless_annotator: int = 0
    careless_multiplier: float = 3.0
    rate_scale: float = 1.0
    seed: int = 7


def _confusable(ds: Dataset, cat_id: int, rng: random.Random) -> int:
    sup = ds.categories[cat_id].get("supercategory")
    pool = [c for c in ds.categories if c != cat_id and ds.categories[c].get("supercategory") == sup]
    if not pool:
        pool = [c for c in ds.categories if c != cat_id]
    return rng.choice(pool)


def _loosen(bbox, img, rng: random.Random) -> list[float]:
    x, y, w, h = bbox
    if rng.random() < 0.5:  # sloppy, oversized box
        f = rng.uniform(1.35, 1.7)
        nw, nh = w * f, h * f
        nx, ny = x - (nw - w) * rng.uniform(0.2, 0.8), y - (nh - h) * rng.uniform(0.2, 0.8)
    else:  # box dragged off the object
        nw, nh = w, h
        nx = x + rng.choice([-1, 1]) * w * rng.uniform(0.25, 0.4)
        ny = y + rng.choice([-1, 1]) * h * rng.uniform(0.25, 0.4)
    W, H = img["width"], img["height"]
    nx, ny = max(0.0, nx), max(0.0, ny)
    nw, nh = min(nw, W - nx), min(nh, H - ny)
    return [round(nx, 2), round(ny, 2), round(nw, 2), round(nh, 2)]


def inject(clean: Dataset, cfg: NoiseConfig | None = None) -> tuple[Dataset, list[dict], dict[int, int]]:
    """Returns (noisy dataset, answer key, image_id -> annotator)."""
    cfg = cfg or NoiseConfig()
    rng = random.Random(cfg.seed)
    noisy = copy.deepcopy(clean)
    next_id = max((a["id"] for anns in clean.anns_by_image.values() for a in anns), default=0) + 1
    all_boxes = [a for anns in clean.anns_by_image.values() for a in anns if not a.get("iscrowd")]
    key: list[dict] = []
    annotator_of: dict[int, int] = {}

    for n, img_id in enumerate(sorted(noisy.images)):
        annot = n % cfg.n_annotators
        annotator_of[img_id] = annot
        mult = cfg.rate_scale * (cfg.careless_multiplier if annot == cfg.careless_annotator else 1.0)
        rate = {k: v * mult for k, v in BASE_RATES.items()}
        img = noisy.images[img_id]
        kept = []
        for ann in noisy.anns_by_image[img_id]:
            if ann.get("iscrowd"):
                kept.append(ann)
                continue
            r = rng.random()
            edges = [rate["missing_object"], rate["wrong_class"], rate["loose_box"], rate["duplicate_box"]]
            if r < edges[0]:
                key.append({"image_id": img_id, "type": "missing_object", "bbox": ann["bbox"],
                            "category_id": ann["category_id"], "ann_id": None, "annotator": annot})
                continue  # annotator forgot this object
            r -= edges[0]
            if r < edges[1]:
                ann["category_id"] = _confusable(noisy, ann["category_id"], rng)
                key.append({"image_id": img_id, "type": "wrong_class", "bbox": ann["bbox"],
                            "category_id": ann["category_id"], "ann_id": ann["id"], "annotator": annot})
            elif r - edges[1] < edges[2]:
                ann["bbox"] = _loosen(ann["bbox"], img, rng)
                ann["area"] = ann["bbox"][2] * ann["bbox"][3]
                key.append({"image_id": img_id, "type": "loose_box", "bbox": ann["bbox"],
                            "category_id": ann["category_id"], "ann_id": ann["id"], "annotator": annot})
            elif r - edges[1] - edges[2] < edges[3]:
                dup = dict(ann, id=next_id, bbox=[v + rng.uniform(-1.5, 1.5) for v in ann["bbox"]])
                next_id += 1
                kept.append(dup)
                key.append({"image_id": img_id, "type": "duplicate_box", "bbox": dup["bbox"],
                            "category_id": dup["category_id"], "ann_id": dup["id"], "annotator": annot})
            kept.append(ann)
        if rng.random() < rate["spurious_box"] and all_boxes:
            # a box pasted where there is nothing (e.g. a mis-click or a box from the wrong image)
            src = rng.choice(all_boxes)
            w = min(src["bbox"][2], img["width"] * 0.5)
            h = min(src["bbox"][3], img["height"] * 0.5)
            w, h = max(w, 24.0), max(h, 24.0)
            x = rng.uniform(0, max(1.0, img["width"] - w))
            y = rng.uniform(0, max(1.0, img["height"] - h))
            box = [round(x, 2), round(y, 2), round(w, 2), round(h, 2)]
            spur = {"id": next_id, "image_id": img_id, "category_id": src["category_id"], "bbox": box,
                    "area": w * h, "iscrowd": 0}
            next_id += 1
            kept.append(spur)
            key.append({"image_id": img_id, "type": "spurious_box", "bbox": box,
                        "category_id": src["category_id"], "ann_id": spur["id"], "annotator": annot})
        noisy.anns_by_image[img_id] = kept
    return noisy, key, annotator_of
