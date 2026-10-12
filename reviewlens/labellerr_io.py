"""Labellerr round trip: pull a COCO export of files awaiting review, push back a
prioritised review queue.

Written against the official SDK (github.com/Labellerr/SDKPython, v1.3.x):
  * LabellerrProject.create_local_export(CreateExportParams)  -> Export.status() -> download_url
  * LabellerrProject.list_files(search_queries)                -> file_name -> file_id
  * LabellerrProject.bulk_assign_files(file_ids, new_status, assign_to)

NOTE: not yet exercised against a live workspace (needs API keys). The parts
whose response shape I could not confirm from the SDK source are marked
`ASSUMPTION` and fail loudly rather than guessing silently.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import requests

REVIEW_STATUSES = ["review", "r_assigned"]


def _project(project_id: str):
    from labellerr.client import LabellerrClient
    from labellerr.core.projects import LabellerrProject

    client = LabellerrClient(os.environ["LABELLERR_API_KEY"], os.environ["LABELLERR_API_SECRET"],
                             os.environ["LABELLERR_CLIENT_ID"])
    return LabellerrProject(client=client, project_id=project_id)


def pull_export(project_id: str, out_path: str | Path, statuses=REVIEW_STATUSES) -> Path:
    """Export files that are waiting for review as COCO JSON."""
    from labellerr.core.schemas.exports import CreateExportParams

    project = _project(project_id)
    export = project.create_local_export(CreateExportParams(
        export_name="reviewlens-pull", export_description="ReviewLens triage input",
        export_format="coco_json", statuses=list(statuses)))
    result = export.status(interval=3.0, timeout=600)
    item = next(i for i in result["response"] if i.get("report_id") == export.report_id)
    url = item["download_url"]
    url = url.get("url", url) if isinstance(url, dict) else url  # ASSUMPTION: signed URL or {"url": ...}
    out_path = Path(out_path)
    out_path.write_bytes(requests.get(url, timeout=120).content)
    return out_path


def file_ids_by_name(project_id: str, statuses=REVIEW_STATUSES, page: int = 500) -> dict[str, str]:
    project = _project(project_id)
    query = [{"op": "OR", "id": "file_status", "values": [{"p": "in", "v": list(statuses)}]}]
    mapping, cursor = {}, None
    while True:
        res = project.list_files(search_queries=query, size=page, next_search_after=cursor).get("response", {})
        files = res.get("files") or res.get("data") or []  # ASSUMPTION: list key
        for f in files:
            name = f.get("file_name") or f.get("file_metadata", {}).get("file_name")
            if name and f.get("file_id"):
                mapping[name] = f["file_id"]
        cursor = res.get("next_search_after")
        if not files or not cursor:
            return mapping


def push_review_queue(project_id: str, ranked_rows: list[dict], top_fraction: float,
                      senior_reviewer: str | None, dry_run: bool = True) -> dict:
    """Route the riskiest files to a (senior) reviewer first.

    Low-risk files are left exactly where they are, so a team can still spot-check
    them, or a manager can choose to sample them at a lower rate.
    """
    n = max(1, round(top_fraction * len(ranked_rows)))
    top = ranked_rows[:n]
    plan = {"n_files": n, "assign_to": senior_reviewer, "new_status": "r_assigned",
            "files": [r["file_name"] for r in top]}
    if dry_run:
        return plan
    ids = file_ids_by_name(project_id)
    missing = [r["file_name"] for r in top if r["file_name"] not in ids]
    if missing:
        raise RuntimeError(f"{len(missing)} ranked files not found in project, e.g. {missing[:3]}")
    project = _project(project_id)
    plan["response"] = project.bulk_assign_files([ids[r["file_name"]] for r in top], "r_assigned",
                                                 assign_to=senior_reviewer)
    return plan


def write_suggestions_coco(ds, issues, path: str | Path) -> Path:
    """Suggested fixes (missing objects, class corrections) as a COCO file.

    Deliberately NOT uploaded automatically: on files that already carry human
    labels I do not know whether `upload_preannotations` merges or overwrites.
    That is the first question I would ask the platform team.
    """
    anns = []
    for k, it in enumerate(issues):
        if it.type == "missing_object":
            cat = it.category_id
        elif it.type == "wrong_class":
            cat = it.suggested_category_id
        else:
            continue
        anns.append({"id": k + 1, "image_id": it.image_id, "category_id": cat, "bbox": it.bbox,
                     "area": it.bbox[2] * it.bbox[3], "iscrowd": 0,
                     "conf_bucket": "high" if it.score >= 0.8 else "medium" if it.score >= 0.5 else "low"})
    used = {a["image_id"] for a in anns}
    coco = {"images": [ds.images[i] for i in used], "categories": list(ds.categories.values()),
            "annotations": anns}
    Path(path).write_text(json.dumps(coco), encoding="utf-8")
    return Path(path)
