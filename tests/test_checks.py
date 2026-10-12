import numpy as np

from reviewlens.checks import Calibration, find_issues, rank_files
from reviewlens.coco import Dataset
from reviewlens.geometry import iou_matrix
from reviewlens.noise import NoiseConfig, inject

CATS = [{"id": 1, "name": "dog", "supercategory": "animal"}, {"id": 2, "name": "cat", "supercategory": "animal"},
        {"id": 3, "name": "car", "supercategory": "vehicle"}]


def make(images_anns: dict[int, list[tuple[int, list[float]]]]) -> Dataset:
    coco = {"images": [], "annotations": [], "categories": CATS}
    aid = 1
    for img_id, anns in images_anns.items():
        coco["images"].append({"id": img_id, "file_name": f"{img_id}.jpg", "width": 640, "height": 480})
        for cat, box in anns:
            coco["annotations"].append({"id": aid, "image_id": img_id, "category_id": cat, "bbox": box, "iscrowd": 0})
            aid += 1
    return Dataset.from_coco(coco)


def pred(img_id, cat, box, score):
    return {"image_id": img_id, "category_id": cat, "bbox": box, "score": score}


# 20 "clean" images where model and humans agree, so calibration learns the model is reliable
AGREE = {i: [(1, [10, 10, 100, 100]), (3, [300, 200, 120, 80])] for i in range(100, 120)}
AGREE_PREDS = {i: [pred(i, 1, [12, 11, 98, 99], 0.9), pred(i, 3, [301, 199, 119, 82], 0.88)] for i in AGREE}


def run(extra_anns, extra_preds):
    ds = make({**AGREE, **extra_anns})
    preds = {**AGREE_PREDS, **extra_preds}
    issues = find_issues(ds, preds)
    return ds, [i for i in issues if i.image_id == 1]


def test_iou_basic():
    m = iou_matrix(np.array([[0, 0, 10, 10]]), np.array([[0, 0, 10, 10], [5, 0, 10, 10], [50, 50, 5, 5]]))
    assert np.allclose(m[0], [1.0, 50 / 150, 0.0])


def test_clean_image_has_no_flags():
    _, iss = run({1: [(1, [10, 10, 100, 100])]}, {1: [pred(1, 1, [11, 10, 99, 101], 0.92)]})
    assert iss == []


def test_missing_object_flagged_with_calibrated_score():
    _, iss = run({1: [(1, [10, 10, 100, 100])]},
                 {1: [pred(1, 1, [11, 10, 99, 101], 0.92), pred(1, 3, [400, 300, 100, 80], 0.9)]})
    assert [i.type for i in iss] == ["missing_object"]
    assert iss[0].score > 0.8  # model's car predictions at 0.9 always agreed with humans elsewhere


def test_wrong_class_suggests_model_class():
    _, iss = run({1: [(2, [10, 10, 100, 100])]}, {1: [pred(1, 1, [11, 10, 99, 101], 0.92)]})
    assert [i.type for i in iss] == ["wrong_class"]
    assert iss[0].suggested_category_id == 1


def test_spurious_and_duplicate():
    _, iss = run({1: [(1, [10, 10, 100, 100]), (1, [10.5, 10, 100, 100.5]), (3, [500, 50, 60, 60])]},
                 {1: [pred(1, 1, [11, 10, 99, 101], 0.92)]})
    assert sorted(i.type for i in iss) == ["duplicate_box", "spurious_box"]


def test_loose_box():
    _, iss = run({1: [(1, [10, 10, 160, 160])]}, {1: [pred(1, 1, [10, 10, 100, 100], 0.9)]})
    assert [i.type for i in iss] == ["loose_box"]


def test_rank_puts_flagged_file_first():
    ds, _ = run({1: [(2, [10, 10, 100, 100])]}, {1: [pred(1, 1, [11, 10, 99, 101], 0.92)]})
    rows = rank_files(ds, find_issues(ds, {**AGREE_PREDS, 1: [pred(1, 1, [11, 10, 99, 101], 0.92)]}))
    assert rows[0]["image_id"] == 1 and rows[0]["risk"] > 0.5 and rows[-1]["risk"] == 0


def test_weak_class_gets_quieter_flags():
    # model is unreliable on "cat": its cat predictions never match human labels
    anns = {i: [(2, [10, 10, 100, 100])] for i in range(200, 220)}
    preds = {i: [pred(i, 2, [400, 300, 80, 80], 0.9)] for i in anns}
    ds = make(anns)
    cal = Calibration().fit(ds, preds)
    assert cal.precision(2, 0.9) < 0.3


def test_noise_injection_is_reproducible_and_logged():
    ds = make({i: [(1, [10, 10, 100, 100]), (3, [300, 200, 120, 80])] for i in range(300)})
    n1, key1, _ = inject(ds, NoiseConfig(seed=1))
    _n2, key2, _ = inject(ds, NoiseConfig(seed=1))
    assert key1 == key2 and len(key1) > 0
    assert ds.n_annotations == 600  # original untouched
    dropped = sum(e["type"] == "missing_object" for e in key1)
    added = sum(e["type"] in ("duplicate_box", "spurious_box") for e in key1)
    assert n1.n_annotations == 600 - dropped + added
