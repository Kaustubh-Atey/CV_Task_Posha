"""dataset.py — VisorDataset and TrainReadyDataset."""

import json
import os

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import (
    map_action_label,
    extract_all_polys,
    polygons_to_mask,
    extract_contact_info,
    build_binary_hand_mask,
    build_contacted_object_mask,
    get_transforms,
)


class VisorDataset(Dataset):
    """
    Indexes VISOR into (frame_path, ann_id, video_id, verb) records.
    __getitem__ returns raw BGR image, per-annotation masks, contact info, verb.
    """

    def __init__(self, csv_path: str, rgb_frames_path: str, annotations_path: str):
        self.annotations: dict = {}
        self.data: list        = []

        df = pd.read_csv(csv_path)
        for _, row in tqdm(df.iterrows(), desc="Indexing dataset", total=len(df)):
            verb           = row["verb"]
            participant_id = row["participant_id"]
            video_id       = row["video_id"]

            if video_id not in self.annotations:
                ann_path = os.path.join(annotations_path, f"{video_id}.json")
                if not os.path.exists(ann_path):
                    continue
                with open(ann_path) as f:
                    self.annotations[video_id] = json.load(f)

            frame_dir = os.path.join(rgb_frames_path, participant_id, video_id)
            if not os.path.exists(frame_dir):
                continue

            start_frame = int(row["start_frame"])
            stop_frame  = int(row["stop_frame"])
            wanted = {
                f"{video_id}_frame_{i:010}.jpg"
                for i in range(start_frame + 1, stop_frame)
            }
            existing = set(os.listdir(frame_dir)) & wanted
            if not existing:
                continue

            ann_file    = self.annotations[video_id]
            name_to_idx = {
                ann_file["video_annotations"][i]["image"]["name"]: i
                for i in range(len(ann_file["video_annotations"]))
            }
            for frame in existing:
                if frame in name_to_idx:
                    self.data.append({
                        "frame_path": os.path.join(frame_dir, frame),
                        "ann_id":     name_to_idx[frame],
                        "video_id":   video_id,
                        "verb":       str(verb),
                    })

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        item      = self.data[idx]
        img       = cv2.imread(item["frame_path"])
        if img is None:
            raise ValueError(f"Cannot load: {item['frame_path']}")
        ann_file  = self.annotations[item["video_id"]]
        frame_ann = ann_file["video_annotations"][item["ann_id"]]
        polys     = extract_all_polys(frame_ann)
        masks     = polygons_to_mask(polys, img.shape)
        contact   = extract_contact_info(frame_ann)
        return img, masks, contact, item["verb"]


class TrainReadyDataset(Dataset):
    """
    Wraps VisorDataset. Applies transforms and returns tensors.

    Returns per sample:
      img_t          : (3, H, W)  float32
      hand_mask_t    : (H, W)     long ∈ {0,1}
      obj_mask_t     : (H, W)     long ∈ {0,1}
      obj_loss_valid : ()         bool tensor
      cls_label      : ()         long
    """

    def __init__(self, dataset: VisorDataset, img_size: int = 512, augment: bool = True):
        self.dataset   = dataset
        self.transform = get_transforms(img_size, augment)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        img, masks_by_ann, contact, verb = self.dataset[idx]

        img_rgb   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w      = img.shape[:2]

        hand_mask = build_binary_hand_mask(masks_by_ann, h, w)
        obj_mask  = build_contacted_object_mask(
            masks_by_ann, contact["contacted_ids"], h, w
        )
        cls_label = map_action_label(verb)

        out         = self.transform(image=img_rgb, mask=hand_mask, mask2=obj_mask)
        img_t       = out["image"].float()
        hand_mask_t = out["mask"].long()
        obj_mask_t  = out["mask2"].long()

        return (
            img_t,
            hand_mask_t,
            obj_mask_t,
            torch.tensor(contact["obj_loss_valid"], dtype=torch.bool),
            torch.tensor(cls_label, dtype=torch.long),
        )
