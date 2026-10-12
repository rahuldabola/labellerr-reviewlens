"""Run a detector over the project's images and write COCO-results predictions.

Any model works as long as it writes `[{image_id, category_id, bbox, score}]`.
This wrapper uses an Ultralytics YOLO checkpoint and maps its class names onto
the project's category names (case-insensitive), so a customer's own
fine-tuned checkpoint can be dropped in with `--weights`.
"""
from __future__ import annotations

import json
from pathlib import Path

from .coco import Dataset


def predict(ds: Dataset, images_dir: str | Path, weights: str = "yolo11m.pt", conf: float = 0.05,
            out_path: str | Path | None = None, batch: int = 8) -> list[dict]:
    from ultralytics import YOLO  # heavy import, keep it local

    model = YOLO(weights)
    by_name = {c["name"].lower(): cid for cid, c in ds.categories.items()}
    cls_map = {i: by_name.get(n.lower()) for i, n in model.names.items()}
    unmapped = sorted(n for i, n in model.names.items() if cls_map[i] is None)
    if unmapped:
        print(f"[predict] {len(unmapped)} model classes not in project, ignored: {unmapped[:8]}...")

    images_dir = Path(images_dir)
    items = [(img_id, images_dir / img["file_name"]) for img_id, img in ds.images.items()]
    items = [(i, p) for i, p in items if p.exists()]
    results: list[dict] = []
    for start in range(0, len(items), batch):
        chunk = items[start:start + batch]
        outs = model.predict([str(p) for _, p in chunk], conf=conf, verbose=False, max_det=300)
        for (img_id, _), r in zip(chunk, outs):
            for (x1, y1, x2, y2), c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                cat = cls_map.get(int(c))
                if cat is None:
                    continue
                results.append({"image_id": img_id, "category_id": cat, "score": round(float(s), 4),
                                "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]})
        print(f"[predict] {min(start + batch, len(items))}/{len(items)} images", end="\r")
    print()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f)
    return results
