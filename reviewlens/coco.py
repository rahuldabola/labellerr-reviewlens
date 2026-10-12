"""Minimal COCO loading / saving. Works on a Labellerr `coco_json` export as-is."""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Dataset:
    images: dict[int, dict]
    categories: dict[int, dict]
    anns_by_image: dict[int, list[dict]] = field(default_factory=dict)

    @classmethod
    def from_coco(cls, coco: dict) -> Dataset:
        by_img: dict[int, list[dict]] = defaultdict(list)
        for ann in coco.get("annotations", []):
            if "bbox" in ann and len(ann["bbox"]) == 4:
                by_img[ann["image_id"]].append(ann)
        images = {img["id"]: img for img in coco["images"]}
        for img_id in images:
            by_img.setdefault(img_id, [])
        return cls(
            images=images,
            categories={c["id"]: c for c in coco["categories"]},
            anns_by_image=dict(by_img),
        )

    @classmethod
    def load(cls, path: str | Path) -> Dataset:
        with open(path, encoding="utf-8") as f:
            return cls.from_coco(json.load(f))

    def to_coco(self) -> dict:
        return {
            "images": list(self.images.values()),
            "categories": list(self.categories.values()),
            "annotations": [a for anns in self.anns_by_image.values() for a in anns],
        }

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_coco(), f)

    def subset(self, image_ids) -> Dataset:
        ids = set(image_ids)
        return Dataset(
            images={i: self.images[i] for i in ids},
            categories=dict(self.categories),
            anns_by_image={i: list(self.anns_by_image.get(i, [])) for i in ids},
        )

    def cat_name(self, cat_id: int) -> str:
        c = self.categories.get(cat_id)
        return c["name"] if c else str(cat_id)

    @property
    def n_annotations(self) -> int:
        return sum(len(v) for v in self.anns_by_image.values())


def load_predictions(path: str | Path) -> dict[int, list[dict]]:
    """Load model predictions in COCO results format:
    [{"image_id", "category_id", "bbox": [x,y,w,h], "score"}, ...]
    """
    with open(path, encoding="utf-8") as f:
        preds = json.load(f)
    by_img: dict[int, list[dict]] = defaultdict(list)
    for p in preds:
        by_img[p["image_id"]].append(p)
    return dict(by_img)
