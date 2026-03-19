"""losses.py — Loss functions, metrics, optimizer builder."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from model import HandSegFormer


# ── Loss functions ────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        prob  = torch.sigmoid(logits).squeeze(1)
        tgt   = targets.float()
        inter = (prob * tgt).sum(dim=(1, 2))
        denom = prob.sum(dim=(1, 2)) + tgt.sum(dim=(1, 2))
        return 1 - ((2 * inter + self.smooth) / (denom + self.smooth)).mean()


class SegmentationLoss(nn.Module):
    """
    Dice + BCE with pos_weight.
    Downsamples targets to match logit resolution if needed (aux losses at stride-4).
    """
    def __init__(self, pos_weight: float = 5.0):
        super().__init__()
        self.dice       = DiceLoss()
        self.pos_weight = pos_weight
        self._pw        = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        Hl, Wl = logits.shape[2:]
        Ht, Wt = targets.shape[1:]
        if (Hl, Wl) != (Ht, Wt):
            targets = F.interpolate(
                targets.float().unsqueeze(1), size=(Hl, Wl), mode="nearest"
            ).squeeze(1).long()
        if self._pw is None or self._pw.device != logits.device:
            self._pw = torch.tensor([self.pos_weight], device=logits.device)
        bce = F.binary_cross_entropy_with_logits(
            logits.squeeze(1), targets.float(), pos_weight=self._pw
        )
        return bce + self.dice(logits, targets)


def compute_total_loss(
    hand_logits, obj_logits, cls_logits,
    aux_hand, aux_obj,
    hand_masks, obj_masks, obj_loss_valid,
    labels, seg_crit, cls_crit, cfg: Config,
):
    hand_loss     = seg_crit(hand_logits, hand_masks)
    aux_hand_loss = torch.tensor(0.0, device=hand_logits.device)
    if aux_hand:
        for ah in aux_hand:
            aux_hand_loss = aux_hand_loss + seg_crit(ah, hand_masks)
        aux_hand_loss = aux_hand_loss / len(aux_hand)

    obj_loss     = torch.tensor(0.0, device=obj_logits.device)
    aux_obj_loss = torch.tensor(0.0, device=obj_logits.device)
    if obj_loss_valid.any():
        v_log  = obj_logits[obj_loss_valid]
        v_mask = obj_masks[obj_loss_valid]
        obj_loss = seg_crit(v_log, v_mask)
        if aux_obj:
            for ao in aux_obj:
                aux_obj_loss = aux_obj_loss + seg_crit(ao[obj_loss_valid], v_mask)
            aux_obj_loss = aux_obj_loss / len(aux_obj)

    cls_loss = cls_crit(cls_logits, labels)
    total = (
        cfg.hand_seg_loss_weight * hand_loss
        + cfg.obj_seg_loss_weight * obj_loss
        + cfg.cls_loss_weight     * cls_loss
        + cfg.aux_loss_weight     * (aux_hand_loss + aux_obj_loss)
    )
    return total, hand_loss, obj_loss, cls_loss


# ── Metrics ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def seg_metrics(logits: torch.Tensor, targets: torch.Tensor):
    """IoU and Dice for (B, 1, H, W) logits vs (B, H, W) targets."""
    preds = (torch.sigmoid(logits.squeeze(1)) > 0.5).long()
    tgts  = targets.long()
    inter = (preds & tgts).sum(dim=(1, 2)).float()
    union = (preds | tgts).sum(dim=(1, 2)).float()
    denom = (preds + tgts).sum(dim=(1, 2)).float()
    iou   = ((inter + 1e-6) / (union + 1e-6)).mean().item()
    dice  = ((2 * inter + 1e-6) / (denom + 1e-6)).mean().item()
    return iou, dice


@torch.no_grad()
def cls_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == targets).float().mean().item()


# ── Optimizer ─────────────────────────────────────────────────────────────────

def build_optimizer(model: HandSegFormer, cfg: Config, encoder_frozen: bool):
    if encoder_frozen:
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.lr, weight_decay=cfg.weight_decay,
        )
    return torch.optim.AdamW(
        [
            {"params": list(model.encoder.parameters()),
             "lr": cfg.lr * cfg.encoder_lr_mult},
            {"params": [p for n, p in model.named_parameters() if "encoder" not in n],
             "lr": cfg.lr},
        ],
        weight_decay=cfg.weight_decay,
    )
