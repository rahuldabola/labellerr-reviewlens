"""ReviewLens command line.

  python -m reviewlens.cli triage    --labels export.json --images imgs/ [--preds p.json | --weights best.pt] --out out/
  python -m reviewlens.cli benchmark --clean clean.json  --images imgs/ --preds p.json --out out/
  python -m reviewlens.cli pull      --project <id> --out export.json            (Labellerr API keys in env)
  python -m reviewlens.cli push      --project <id> --queue out/queue.csv --top 0.2 --reviewer lead@x.com [--apply]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from . import labellerr_io
from .checks import find_issues, rank_files
from .coco import Dataset, load_predictions


def _preds(args, ds: Dataset) -> dict:
    if args.preds and Path(args.preds).exists():
        return load_predictions(args.preds)
    from .predict import predict
    out = Path(args.out) / "predictions.json"
    predict(ds, args.images, weights=args.weights, out_path=out)
    return load_predictions(out)


def _write_outputs(out: Path, ds: Dataset, rows: list[dict], issues) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "queue.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "file_name", "image_id", "risk", "expected_errors", "n_issues", "n_labels", "top_issue"])
        for k, r in enumerate(rows):
            w.writerow([k + 1, r["file_name"], r["image_id"], f'{r["risk"]:.4f}', f'{r["expected_errors"]:.3f}',
                        r["n_issues"], r["n_labels"], r["top_issue"]])
    with open(out / "issues.json", "w", encoding="utf-8") as f:
        json.dump([i.to_dict() for i in sorted(issues, key=lambda i: -i.score)], f, indent=1)
    labellerr_io.write_suggestions_coco(ds, issues, out / "suggested_fixes_coco.json")


def cmd_triage(args) -> None:
    from .report import render
    ds = Dataset.load(args.labels)
    issues = find_issues(ds, _preds(args, ds))
    rows = rank_files(ds, issues)
    out = Path(args.out)
    _write_outputs(out, ds, rows, issues)
    render(out / "report.html", ds, rows, args.images, title="ReviewLens — review queue")
    print(f"{len(issues)} flags on {sum(r['n_issues'] > 0 for r in rows)}/{len(rows)} files -> {out}")


def cmd_benchmark(args) -> None:
    from .evaluate import benchmark
    from .noise import NoiseConfig, inject
    from .report import render

    clean = Dataset.load(args.clean)
    noisy, key, annot = inject(clean, NoiseConfig(seed=args.seed, rate_scale=args.rate_scale))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    noisy.save(out / "noisy_labels.json")
    with open(out / "answer_key.json", "w", encoding="utf-8") as f:
        json.dump(key, f)
    preds = _preds(args, clean)
    issues = find_issues(noisy, preds)
    rows = rank_files(noisy, issues)
    bench = benchmark(noisy, key, rows, issues, annot, seed=args.seed)
    _write_outputs(out, noisy, rows, issues)
    with open(out / "benchmark.json", "w", encoding="utf-8") as f:
        json.dump(bench, f, indent=1)
    notes = Path(args.notes).read_text(encoding="utf-8") if args.notes else ""
    render(out / "report.html", noisy, rows, args.images, bench=bench, truth=key, notes_html=notes,
           title="ReviewLens — review the right files first")
    i20 = bench["budgets"].index(0.2)
    print(json.dumps({k: round(v[i20], 3) for k, v in bench["curves"].items()}, indent=1))
    print("effort for 80%:", {k: round(v, 3) for k, v in bench["effort_for_80pct"].items()})
    print("per type:", bench["per_type"])
    print("precision@:", bench["issue_precision_at"])


def cmd_pull(args) -> None:
    print(labellerr_io.pull_export(args.project, args.out))


def cmd_push(args) -> None:
    with open(args.queue, encoding="utf-8") as f:
        rows = [{"file_name": r["file_name"], "risk": float(r["risk"])} for r in csv.DictReader(f)]
    plan = labellerr_io.push_review_queue(args.project, rows, args.top, args.reviewer, dry_run=not args.apply)
    print(json.dumps({k: v for k, v in plan.items() if k != "files"}, indent=1, default=str))
    print("first files:", plan["files"][:10])


def main() -> None:
    ap = argparse.ArgumentParser(prog="reviewlens")
    sub = ap.add_subparsers(required=True)
    for name, fn in (("triage", cmd_triage), ("benchmark", cmd_benchmark)):
        p = sub.add_parser(name)
        p.add_argument("--images", required=True)
        p.add_argument("--preds")
        p.add_argument("--weights", default="yolo11m.pt")
        p.add_argument("--out", required=True)
        p.set_defaults(fn=fn)
        if name == "triage":
            p.add_argument("--labels", required=True)
        else:
            p.add_argument("--clean", required=True)
            p.add_argument("--seed", type=int, default=7)
            p.add_argument("--rate-scale", type=float, default=1.0)
            p.add_argument("--notes", help="HTML snippet inserted into the report")
    p = sub.add_parser("pull")
    p.add_argument("--project", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_pull)
    p = sub.add_parser("push")
    p.add_argument("--project", required=True)
    p.add_argument("--queue", required=True)
    p.add_argument("--top", type=float, default=0.2)
    p.add_argument("--reviewer")
    p.add_argument("--apply", action="store_true", help="actually call bulk_assign_files (default: dry run)")
    p.set_defaults(fn=cmd_push)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
