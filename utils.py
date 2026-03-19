"""utils.py — Label mapping, VISOR annotation helpers, mask builders, transforms."""

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── Action label mapping ──────────────────────────────────────────────────────

STIR_KW = {
    'stir', 'mix', 'whisk', 'beat', 'blend', 'fold',
    'knead', 'toss', 'swirl', 'rotate', 'turn', 'combine',
}
ADD_KW = {
    'add', 'pour', 'sprinkle', 'put', 'place', 'drop', 'squeeze',
    'insert', 'throw', 'tip', 'shake', 'scatter', 'spread', 'dip',
    'coat', 'fill', 'load', 'transfer', 'empty', 'spoon', 'ladle',
    'scoop', 'slice', 'chop', 'grate', 'peel', 'cut', 'dice',
    'pinch', 'drizzle', 'dollop',
}
ACTION_CLASSES = {0: "None", 1: "Stirring", 2: "Adding ingredients"}


def map_action_label(verb: str) -> int:
    text = str(verb).lower()
    for kw in STIR_KW:
        if kw in text:
            return 1
    for kw in ADD_KW:
        if kw in text:
            return 2
    return 0


# ── VISOR contact annotation helpers ─────────────────────────────────────────

def classify_contact(value) -> str:
    """Normalise a raw in_contact_object value to one of 5 categories."""
    if value is None:
        return "missing"
    s = str(value).strip().lower()
    if s in ("", "none", "nan"):
        return "missing"
    if s == "hand-not-in-contact":
        return "not_in_contact"
    if s == "none-of-the-above":
        return "none_of_above"
    if s == "inconclusive":
        return "inconclusive"
    return "valid_id"


def extract_contact_info(frame_ann: dict) -> dict:
    """
    Walk all hand/glove annotations in a frame and return:
      contacted_ids  : set of str annotation IDs hands are touching
      obj_loss_valid : bool — whether object query loss should be active
    """
    contacted_ids = set()
    hand_cats     = []

    for ann in frame_ann.get("annotations", []):
        name = ann.get("name", "").lower()
        if "hand" not in name and "glove" not in name:
            continue
        raw = ann.get("in_contact_object", None)
        cat = classify_contact(raw)
        hand_cats.append(cat)
        if cat == "valid_id":
            contacted_ids.add(str(raw).strip())

    if not hand_cats:
        return {"contacted_ids": set(), "obj_loss_valid": False}

    any_valid     = any(c == "valid_id"      for c in hand_cats)
    any_free      = any(c == "not_in_contact" for c in hand_cats)
    any_ambiguous = any(c in ("none_of_above", "inconclusive", "missing")
                        for c in hand_cats)

    if any_valid:
        obj_loss_valid = True
    elif any_free and not any_ambiguous:
        obj_loss_valid = True
    else:
        obj_loss_valid = False

    return {"contacted_ids": contacted_ids, "obj_loss_valid": obj_loss_valid}


def extract_all_polys(frame_ann: dict) -> list:
    """Extract all polygon records from a frame annotation, keyed by ann_id."""
    polys = []
    for ann in frame_ann.get("annotations", []):
        ann_id     = str(ann.get("id", ""))
        class_name = ann.get("name", "unknown")
        for seg in ann.get("segments", []):
            if isinstance(seg, list) and seg:
                polys.append({
                    "vertices":   seg,
                    "ann_id":     ann_id,
                    "class_name": class_name,
                })
    return polys


def polygons_to_mask(polys: list, shape: tuple) -> dict:
    """Rasterise all polygons into binary masks keyed by annotation ID."""
    h, w = shape[:2]
    masks_by_ann: dict = {}
    for pd_ in polys:
        if not isinstance(pd_, dict):
            continue
        poly       = pd_.get("vertices", [])
        ann_id     = str(pd_.get("ann_id", ""))
        class_name = pd_.get("class_name", "unknown")
        if not (isinstance(poly, list) and poly):
            continue
        if ann_id not in masks_by_ann:
            masks_by_ann[ann_id] = {
                "mask":       np.zeros((h, w), dtype=np.uint8),
                "class_name": class_name,
            }
        try:
            pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
            if pts.size > 0:
                cv2.fillPoly(masks_by_ann[ann_id]["mask"], [pts], color=255)
        except Exception:
            continue
    return masks_by_ann


# ── Mask builders ─────────────────────────────────────────────────────────────

def build_binary_hand_mask(masks_by_ann: dict, h: int, w: int) -> np.ndarray:
    """Union of all hand and glove annotation masks."""
    hand_mask = np.zeros((h, w), dtype=np.uint8)
    for _, mdata in masks_by_ann.items():
        name = mdata["class_name"].lower()
        if "hand" in name or "glove" in name:
            hand_mask = np.maximum(hand_mask, mdata["mask"])
    return (hand_mask > 0).astype(np.uint8)


def build_contacted_object_mask(
    masks_by_ann:  dict,
    contacted_ids: set,
    h: int,
    w: int,
) -> np.ndarray:
    """Union of masks whose annotation ID is in contacted_ids."""
    obj_mask = np.zeros((h, w), dtype=np.uint8)
    for ann_id, mdata in masks_by_ann.items():
        if ann_id in contacted_ids:
            obj_mask = np.maximum(obj_mask, mdata["mask"])
    return (obj_mask > 0).astype(np.uint8)


# ── Albumentations transforms ─────────────────────────────────────────────────

def get_transforms(img_size: int, augment: bool) -> A.Compose:
    """
    Applies identical spatial transforms to the image and both masks.
    mask  = hand mask (primary)
    mask2 = object mask (additional target)
    """
    extra = {"mask2": "mask"}
    ops   = [A.Resize(img_size, img_size)]
    if augment:
        ops += [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(var_limit=(10, 50), p=0.2),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.2),
        ]
    ops += [
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
    return A.Compose(ops, additional_targets=extra)
