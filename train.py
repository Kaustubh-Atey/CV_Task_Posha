"""train.py — Training and validation loops, main entry point."""

import csv
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

from config import Config
from dataset import VisorDataset, TrainReadyDataset
from model import HandSegFormer
from losses import (
    SegmentationLoss,
    compute_total_loss,
    seg_metrics,
    cls_accuracy,
    build_optimizer,
)
from utils import ACTION_CLASSES, map_action_label

warnings.filterwarnings("ignore")


def train_one_epoch(model, loader, optimizer, scheduler, seg_crit, cls_crit,
                    scaler, cfg: Config) -> dict:
    model.train()
    totals = dict(loss=0.0, hand_seg=0.0, obj_seg=0.0, cls=0.0,
                  hand_iou=0.0, hand_dice=0.0, obj_iou=0.0, acc=0.0)
    n = len(loader)

    for imgs, hand_masks, obj_masks, obj_valid, labels in tqdm(
        loader, desc="  train", leave=False
    ):
        imgs       = imgs.to(cfg.device)
        hand_masks = hand_masks.to(cfg.device)
        obj_masks  = obj_masks.to(cfg.device)
        obj_valid  = obj_valid.to(cfg.device)
        labels     = labels.to(cfg.device)

        optimizer.zero_grad()
        with autocast(enabled=cfg.use_amp):
            hand_logits, obj_logits, cls_logits, aux_hand, aux_obj = model(imgs)
            loss, hand_loss, obj_loss, cls_loss = compute_total_loss(
                hand_logits, obj_logits, cls_logits,
                aux_hand, aux_obj,
                hand_masks, obj_masks, obj_valid,
                labels, seg_crit, cls_crit, cfg,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        hand_iou, hand_dice = seg_metrics(hand_logits, hand_masks)
        obj_iou, _          = seg_metrics(obj_logits,  obj_masks)
        acc                 = cls_accuracy(cls_logits, labels)

        totals["loss"]      += loss.item()
        totals["hand_seg"]  += hand_loss.item()
        totals["obj_seg"]   += obj_loss.item()
        totals["cls"]       += cls_loss.item()
        totals["hand_iou"]  += hand_iou
        totals["hand_dice"] += hand_dice
        totals["obj_iou"]   += obj_iou
        totals["acc"]       += acc

    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def validate(model, loader, seg_crit, cls_crit, cfg: Config) -> dict:
    model.eval()
    totals = dict(loss=0.0, hand_seg=0.0, obj_seg=0.0, cls=0.0,
                  hand_iou=0.0, hand_dice=0.0, obj_iou=0.0, acc=0.0)
    n = len(loader)

    for imgs, hand_masks, obj_masks, obj_valid, labels in tqdm(
        loader, desc="  val  ", leave=False
    ):
        imgs       = imgs.to(cfg.device)
        hand_masks = hand_masks.to(cfg.device)
        obj_masks  = obj_masks.to(cfg.device)
        obj_valid  = obj_valid.to(cfg.device)
        labels     = labels.to(cfg.device)

        with autocast(enabled=cfg.use_amp):
            hand_logits, obj_logits, cls_logits, _, _ = model(imgs)
            hand_loss = seg_crit(hand_logits, hand_masks)
            obj_loss  = torch.tensor(0.0, device=cfg.device)
            if obj_valid.any():
                obj_loss = seg_crit(obj_logits[obj_valid], obj_masks[obj_valid])
            cls_loss = cls_crit(cls_logits, labels)
            loss     = (cfg.hand_seg_loss_weight * hand_loss
                        + cfg.obj_seg_loss_weight * obj_loss
                        + cfg.cls_loss_weight * cls_loss)

        hand_iou, hand_dice = seg_metrics(hand_logits, hand_masks)
        obj_iou, _          = seg_metrics(obj_logits,  obj_masks)
        acc                 = cls_accuracy(cls_logits, labels)

        totals["loss"]      += loss.item()
        totals["hand_seg"]  += hand_loss.item()
        totals["obj_seg"]   += obj_loss.item()
        totals["cls"]       += cls_loss.item()
        totals["hand_iou"]  += hand_iou
        totals["hand_dice"] += hand_dice
        totals["obj_iou"]   += obj_iou
        totals["acc"]       += acc

    return {k: v / n for k, v in totals.items()}


def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    print(f"Device: {cfg.device}  |  AMP: {cfg.use_amp}")

    print("\n[1/5] Loading VISOR dataset...")
    train_raw = VisorDataset(cfg.csv_path_train, cfg.rgb_frames_path_train,
                              cfg.annotations_path_train)
    val_raw   = VisorDataset(cfg.csv_path_val,   cfg.rgb_frames_path_val,
                              cfg.annotations_path_val)
    print(f"      Train: {len(train_raw)}  |  Val: {len(val_raw)}")

    print("\n[2/5] Computing class weights...")
    counts = [0] * cfg.num_action_classes
    for item in train_raw.data:
        counts[map_action_label(item["verb"])] += 1
    counts = [max(c, 1) for c in counts]
    total  = sum(counts)
    cls_w  = torch.tensor(
        [total / (cfg.num_action_classes * c) for c in counts], dtype=torch.float32
    )
    print("      Distribution:", {ACTION_CLASSES[i]: counts[i] for i in range(3)})
    print("      Weights      :", cls_w.tolist())

    print("\n[3/5] Building data loaders...")
    train_ds = TrainReadyDataset(train_raw, cfg.img_size, augment=True)
    val_ds   = TrainReadyDataset(val_raw,   cfg.img_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True)

    print(f"\n[4/5] Initialising HandSegFormer ({cfg.encoder_name})...")
    model = HandSegFormer(
        encoder_name         = cfg.encoder_name,
        decoder_dim          = cfg.decoder_dim,
        num_action_classes   = cfg.num_action_classes,
        mask_decoder_layers  = cfg.mask_decoder_layers,
        mask_decoder_heads   = cfg.mask_decoder_heads,
        mask_decoder_dropout = cfg.mask_decoder_dropout,
        num_mask_queries     = cfg.num_mask_queries,
    ).to(cfg.device)

    model.freeze_encoder()
    print(f"      Encoder FROZEN for first {cfg.freeze_encoder_epochs} epochs.")

    seg_crit  = SegmentationLoss(pos_weight=cfg.bce_pos_weight)
    cls_crit  = nn.CrossEntropyLoss(weight=cls_w.to(cfg.device))
    scaler    = GradScaler(enabled=cfg.use_amp)
    optimizer = build_optimizer(model, cfg, encoder_frozen=True)

    total_steps  = len(train_loader) * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"\n[5/5] Training for {cfg.epochs} epochs...\n")
    best_iou = 0.0
    log_rows: list = []

    for epoch in range(1, cfg.epochs + 1):

        if epoch == cfg.freeze_encoder_epochs + 1:
            print(f"\n  ── Epoch {epoch}: UNFREEZING encoder ──")
            model.unfreeze_encoder()
            remaining = len(train_loader) * (cfg.epochs - epoch + 1)
            optimizer = build_optimizer(model, cfg, encoder_frozen=False)
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, int(remaining * cfg.warmup_ratio), remaining
            )

        tr = train_one_epoch(model, train_loader, optimizer, scheduler,
                              seg_crit, cls_crit, scaler, cfg)
        vl = validate(model, val_loader, seg_crit, cls_crit, cfg)

        print(
            f"  Ep {epoch:02d}/{cfg.epochs}"
            f"  | TrLoss {tr['loss']:.4f}"
            f"  HandIoU {tr['hand_iou']:.4f}  ObjIoU {tr['obj_iou']:.4f}"
            f"  Acc {tr['acc']:.4f}"
            f"  || ValLoss {vl['loss']:.4f}"
            f"  HandIoU {vl['hand_iou']:.4f}  ObjIoU {vl['obj_iou']:.4f}"
            f"  Acc {vl['acc']:.4f}"
        )
        log_rows.append({"epoch": epoch,
                          **{f"tr_{k}": v for k, v in tr.items()},
                          **{f"vl_{k}": v for k, v in vl.items()}})

        if vl["hand_iou"] > best_iou:
            best_iou = vl["hand_iou"]
            torch.save({
                "epoch":        epoch,
                "model":        model.state_dict(),
                "val_hand_iou": best_iou,
                "val_obj_iou":  vl["obj_iou"],
                "val_acc":      vl["acc"],
                "cfg":          cfg,
            }, os.path.join(cfg.checkpoint_dir, "best.pth"))
            print(f"    ✓ Best saved  (HandIoU={best_iou:.4f}  "
                  f"ObjIoU={vl['obj_iou']:.4f}  Acc={vl['acc']:.4f})")

        if epoch % cfg.save_every == 0:
            torch.save({"epoch": epoch, "model": model.state_dict()},
                       os.path.join(cfg.checkpoint_dir, f"epoch_{epoch:02d}.pth"))

    log_path = os.path.join(cfg.checkpoint_dir, "train_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"\nTraining complete.  Best Val Hand IoU: {best_iou:.4f}")
    print(f"Log → {log_path}")


if __name__ == "__main__":
    main()
