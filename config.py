"""config.py — Training configuration dataclass."""

import torch
from dataclasses import dataclass, field


@dataclass
class Config:
    # ── Paths ──────────────────────────────────────────────────────────────
    csv_path_train:         str   = "/home/kaustubh/Projects/ARP/Data/EPIC_100_train_subset.csv"
    csv_path_val:           str   = "/home/kaustubh/Projects/ARP/Data/EPIC_100_val_subset.csv"
    rgb_frames_path_train:  str   = "/home/kaustubh/Projects/ARP/visor/VISOR_Data/GroundTruth-SparseAnnotations/rgb_frames/train"
    rgb_frames_path_val:    str   = "/home/kaustubh/Projects/ARP/visor/VISOR_Data/GroundTruth-SparseAnnotations/rgb_frames/val"
    annotations_path_train: str   = "/home/kaustubh/Projects/ARP/visor/VISOR_Data/GroundTruth-SparseAnnotations/annotations/train"
    annotations_path_val:   str   = "/home/kaustubh/Projects/ARP/visor/VISOR_Data/GroundTruth-SparseAnnotations/annotations/val"
    checkpoint_dir:         str   = "./checkpoints_swin_2query"

    # ── Model ──────────────────────────────────────────────────────────────
    encoder_name:           str   = "microsoft/swin-base-patch4-window7-224-in22k"
    img_size:               int   = 512
    decoder_dim:            int   = 256
    num_action_classes:     int   = 3
    mask_decoder_layers:    int   = 3
    mask_decoder_heads:     int   = 8
    mask_decoder_dropout:   float = 0.0
    num_mask_queries:       int   = 2

    # ── Training ───────────────────────────────────────────────────────────
    epochs:                 int   = 30
    batch_size:             int   = 64
    num_workers:            int   = 4
    freeze_encoder_epochs:  int   = 3

    # ── Optimizer ──────────────────────────────────────────────────────────
    lr:                     float = 6e-5
    encoder_lr_mult:        float = 0.1
    weight_decay:           float = 0.01
    warmup_ratio:           float = 0.05
    grad_clip:              float = 1.0

    # ── Loss ───────────────────────────────────────────────────────────────
    hand_seg_loss_weight:   float = 1.0
    obj_seg_loss_weight:    float = 0.8
    cls_loss_weight:        float = 0.5
    aux_loss_weight:        float = 0.4
    bce_pos_weight:         float = 5.0

    # ── Misc ───────────────────────────────────────────────────────────────
    seed:       int   = 42
    save_every: int   = 5
    use_amp:    bool  = True
    device: str = field(
        default_factory=lambda: "cuda:0" if torch.cuda.is_available() else "cpu"
    )
