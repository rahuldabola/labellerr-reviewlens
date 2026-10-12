"""Download a reproducible random subset of COCO val2017 (images + clean labels).

usage: python -m scripts.prepare_coco_subset --n 600
needs data/annotations/instances_val2017.json
(from http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
"""
from __future__ import annotations

import argparse
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from reviewlens.coco import Dataset

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "data" / "coco_subset"))
    args = ap.parse_args()

    full = Dataset.load(ROOT / "data" / "annotations" / "instances_val2017.json")
    ids = sorted(i for i, anns in full.anns_by_image.items() if anns)  # images with at least one label
    chosen = random.Random(args.seed).sample(ids, args.n)
    sub = full.subset(chosen)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    sub.save(out / "clean.json")

    def fetch(img: dict) -> None:
        dest = out / "images" / img["file_name"]
        if dest.exists():
            return
        r = requests.get(img["coco_url"], timeout=60)
        r.raise_for_status()
        dest.write_bytes(r.content)

    with ThreadPoolExecutor(16) as pool:
        list(pool.map(fetch, sub.images.values()))
    print(f"{len(sub.images)} images, {sub.n_annotations} labels -> {out}")


if __name__ == "__main__":
    main()
