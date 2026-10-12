"""End-to-end test against a real Labellerr workspace (free Researcher plan is enough).

  1. upload N benchmark images as a dataset
  2. create a bounding-box template (one question per class) + project
  3. upload the *noisy* labels as pre-annotations (stand-in for annotator work)
  4. pull them back with create_local_export(coco_json)       <- ReviewLens `pull`
  5. triage with cached YOLO predictions                       <- ReviewLens core
  6. resolve file_ids and (optionally) bulk-assign top files   <- ReviewLens `push`

Credentials come from env or a .env file in the repo root (never committed):
  LABELLERR_API_KEY, LABELLERR_API_SECRET, LABELLERR_CLIENT_ID, LABELLERR_EMAIL

usage: python -m scripts.live_roundtrip --n 30 [--apply]
Writes a log to docs/live_roundtrip.md.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

from reviewlens import labellerr_io
from reviewlens.checks import find_issues, rank_files
from reviewlens.coco import Dataset, load_predictions
from reviewlens.noise import NoiseConfig, inject

ROOT = Path(__file__).resolve().parents[1]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
LOG: list[str] = []


def log(msg: str) -> None:
    line = f"- `{time.strftime('%H:%M:%S')}` {msg}"
    print(line)
    LOG.append(line)


def load_env() -> None:
    env = ROOT / ".env"
    if env.exists():
        for raw in env.read_text(encoding="utf-8").splitlines():
            if "=" in raw and not raw.lstrip().startswith("#"):
                k, v = raw.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))
    missing = [k for k in ("LABELLERR_API_KEY", "LABELLERR_API_SECRET", "LABELLERR_CLIENT_ID") if not os.environ.get(k)]
    if missing:
        raise SystemExit(f"missing {missing}: put them in {env}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--apply", action="store_true", help="really bulk-assign the top files to LABELLERR_EMAIL")
    args = ap.parse_args()
    load_env()

    from labellerr.client import LabellerrClient
    from labellerr.core.annotation_templates import create_template
    from labellerr.core.datasets import LabellerrDataset, create_dataset_from_local
    from labellerr.core.projects import create_project
    from labellerr.core.schemas import DatasetConfig
    from labellerr.core.schemas.annotation_templates import (
        AnnotationQuestion,
        CreateTemplateParams,
        QuestionType,
    )
    from labellerr.core.schemas.projects import CreateProjectParams, RotationConfig

    client = LabellerrClient(os.environ["LABELLERR_API_KEY"], os.environ["LABELLERR_API_SECRET"],
                             os.environ["LABELLERR_CLIENT_ID"])

    # --- local data: N images with injected annotator mistakes -------------------------
    clean = Dataset.load(ROOT / "data/coco_subset/clean.json")
    noisy_all, key, _ = inject(clean, NoiseConfig(seed=7, rate_scale=0.5))
    ids = sorted(noisy_all.images)[: args.n]
    noisy = noisy_all.subset(ids)
    used = sorted({a["category_id"] for anns in noisy.anns_by_image.values() for a in anns})
    noisy.categories = {c: noisy.categories[c] for c in used}
    tmp = Path(tempfile.mkdtemp(prefix="reviewlens_"))
    for i in ids:
        shutil.copy(ROOT / "data/coco_subset/images" / noisy.images[i]["file_name"], tmp)
    preann = tmp.parent / f"{tmp.name}_preann.json"
    noisy.save(preann)
    n_err = sum(e["image_id"] in set(ids) for e in key)
    log(f"local: {len(ids)} images, {noisy.n_annotations} labels ({n_err} injected mistakes), {len(used)} classes")

    # --- 1-2. dataset, template, project ----------------------------------------------
    stamp = time.strftime("%m%d-%H%M")
    ds = create_dataset_from_local(client, DatasetConfig(dataset_name=f"reviewlens-{stamp}", data_type="image"),
                                   folder_to_upload=str(tmp))
    ds.status(interval=3.0, timeout=600)
    ds = LabellerrDataset(client=client, dataset_id=ds.dataset_id)
    log(f"dataset `{ds.dataset_id}` ready, files_count={ds.files_count}")

    questions = [AnnotationQuestion(question_number=k + 1, question=noisy.categories[c]["name"],
                                    question_type=QuestionType.bounding_box, required=False,
                                    color=COLORS[k % len(COLORS)])
                 for k, c in enumerate(used)]
    tpl = create_template(client, CreateTemplateParams(template_name=f"reviewlens-{stamp}", data_type="image",
                                                       questions=questions))
    log(f"template `{tpl.annotation_template_id}` with {len(questions)} BoundingBox questions")

    project = create_project(client, CreateProjectParams(
        project_name=f"ReviewLens live test {stamp}", data_type="image",
        rotations=RotationConfig(annotation_rotation_count=1, review_rotation_count=1, client_review_rotation_count=1),
        created_by=os.environ.get("LABELLERR_EMAIL")), datasets=[ds], annotation_template=tpl)
    project.status()
    log(f"project `{project.project_id}` created")

    # --- 3. annotator labels (pre-annotations) ----------------------------------------
    res = project.upload_preannotations(annotation_format="coco_json", annotation_file=str(preann))
    log(f"upload_preannotations -> {json.dumps(res.get('response', res), default=str)[:300]}")

    # --- 4. pull --------------------------------------------------------------------
    statuses = ["None", "assigned", "review", "r_assigned", "accepted", "rejected", "client_review", "cr_assigned"]
    export_path = tmp.parent / f"{tmp.name}_export.json"
    labellerr_io.pull_export(project.project_id, export_path, statuses=statuses)
    exported = Dataset.load(export_path)
    log(f"pull: export has {len(exported.images)} images, {exported.n_annotations} labels "
        f"(uploaded {noisy.n_annotations})")

    # --- 5. triage (predictions re-keyed by file name: export ids differ from COCO ids)
    by_name = {img["file_name"]: i for i, img in exported.images.items()}
    src_name = {i: img["file_name"] for i, img in clean.images.items()}
    name_to_cat = {c["name"]: cid for cid, c in exported.categories.items()}
    preds = {}
    for img_id, ps in load_predictions(ROOT / "data/coco_subset/preds_yolo11m.json").items():
        tgt = by_name.get(src_name.get(img_id))
        if tgt is None:
            continue
        preds[tgt] = [dict(p, image_id=tgt, category_id=name_to_cat[clean.categories[p["category_id"]]["name"]])
                      for p in ps if clean.categories[p["category_id"]]["name"] in name_to_cat]
    rows = rank_files(exported, find_issues(exported, preds))
    log(f"triage: {sum(r['n_issues'] > 0 for r in rows)}/{len(rows)} files flagged; top 3: "
        + ", ".join(f"{r['file_name']} ({r['expected_errors']:.1f})" for r in rows[:3]))

    # --- 6. push --------------------------------------------------------------------
    mapping = labellerr_io.file_ids_by_name(project.project_id, statuses=statuses)
    log(f"list_files resolved {sum(r['file_name'] in mapping for r in rows)}/{len(rows)} file names to file_ids")
    if args.apply:
        plan = labellerr_io.push_review_queue(project.project_id, rows, 0.2, os.environ.get("LABELLERR_EMAIL"),
                                              dry_run=False, statuses=statuses)
        log(f"bulk_assign_files({plan['n_files']} files) -> {json.dumps(plan.get('response'), default=str)[:300]}")

    out = ROOT / "docs/live_roundtrip.md"
    out.write_text("# Live Labellerr round trip\n\n" + "\n".join(LOG) + "\n", encoding="utf-8")
    print(f"log -> {out}")


if __name__ == "__main__":
    main()
