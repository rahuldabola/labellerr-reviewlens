"""Robustness sweep: error rate x random seed. Prints a markdown table.

usage: python -m scripts.sweep --clean data/coco_subset/clean.json --preds data/coco_subset/preds_yolo11m.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from reviewlens.checks import find_issues, rank_files
from reviewlens.coco import Dataset, load_predictions
from reviewlens.evaluate import benchmark
from reviewlens.noise import NoiseConfig, inject


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--scales", default="0.25,0.5,1.0")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out")
    args = ap.parse_args()
    clean = Dataset.load(args.clean)
    preds = load_predictions(args.preds)
    names = ["ReviewLens ranking", "Random order (status quo)", "Most labels first"]
    out = []
    print("| label error rate | files with errors | caught @20% files: ReviewLens / random / most-labels | "
          "files to review for 80%: ReviewLens / random / most-labels | top-100 flag precision |")
    print("|---|---|---|---|---|")
    for scale in map(float, args.scales.split(",")):
        res = []
        for seed in range(args.seeds):
            noisy, key, annot = inject(clean, NoiseConfig(seed=100 + seed, rate_scale=scale))
            issues = find_issues(noisy, preds)
            b = benchmark(noisy, key, rank_files(noisy, issues), issues, annot, seed=seed)
            i20 = b["budgets"].index(0.2)
            res.append({
                "err_rate": b["n_injected"] / clean.n_annotations,
                "files_with_errors": b["n_images_with_errors"] / b["n_images"],
                **{f"c20_{n}": b["curves"][n][i20] for n in names},
                **{f"e80_{n}": b["effort_for_80pct"][n] for n in names},
                "p100": b["issue_precision_at"].get(100, float("nan")),
            })
        m = {k: float(np.mean([r[k] for r in res])) for k in res[0]}
        sd = {k: float(np.std([r[k] for r in res])) for k in res[0]}
        out.append({"scale": scale, "mean": m, "std": sd})
        def pct(k, m=m):
            return f"{100 * m[k]:.0f}%"

        print(f"| {pct('err_rate')} | {pct('files_with_errors')} | "
              f"**{pct('c20_' + names[0])}** ±{100 * sd['c20_' + names[0]]:.0f} / {pct('c20_' + names[1])} / {pct('c20_' + names[2])} | "
              f"**{pct('e80_' + names[0])}** / {pct('e80_' + names[1])} / {pct('e80_' + names[2])} | {pct('p100')} |")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
