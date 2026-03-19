"""inference.py — Two-query HandSegFormer inference on a test video."""

import argparse
import collections
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torchvision import transforms

from config import Config
from model import HandSegFormer
from utils import ACTION_CLASSES


# ── Pre / post-processing ─────────────────────────────────────────────────────

_NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def preprocess_frame(bgr: np.ndarray, img_size: int) -> torch.Tensor:
    rgb     = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    t       = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    return _NORM(t).unsqueeze(0)


def logit_to_mask(logit: torch.Tensor, orig_h: int, orig_w: int,
                  threshold: float = 0.5) -> np.ndarray:
    prob      = torch.sigmoid(logit[0, 0]).cpu().numpy()
    prob_full = cv2.resize(prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return (prob_full >= threshold).astype(np.uint8) * 255


def mask_pixel_fraction(mask: np.ndarray) -> float:
    return float(mask.sum()) / max(mask.size, 1) / 255.0


# ── Overlay rendering ─────────────────────────────────────────────────────────

HAND_COLOR   = (0, 220, 220)
LABEL_COLORS = {
    "None":               (180, 180, 180),
    "Stirring":           (50,  220,  50),
    "Adding ingredients": (50,  50,  255),
}


def draw_hand_overlay(frame: np.ndarray, mask: np.ndarray,
                      alpha: float = 0.5) -> np.ndarray:
    out      = frame.copy()
    mb       = mask > 0
    if not mb.any():
        return out
    coloured = np.zeros_like(frame, dtype=np.uint8)
    coloured[mb] = HAND_COLOR
    out[mb] = cv2.addWeighted(
        coloured[mb].reshape(-1, 3), alpha,
        frame[mb].reshape(-1, 3),   1 - alpha, 0,
    ).reshape(-1, 3)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, HAND_COLOR, thickness=2)
    return out


def draw_action_banner(frame: np.ndarray, label: str,
                       confidence: float) -> np.ndarray:
    out  = frame.copy()
    text = f"Action: {label} ({confidence*100:.1f}%)"
    lc   = LABEL_COLORS.get(label, (200, 200, 200))

    font       = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.85
    thickness  = 2
    pad        = 10

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    ov = out.copy()
    cv2.rectangle(ov, (0, 0), (tw + 2 * pad, th + baseline + 2 * pad), (18, 18, 18), -1)
    cv2.addWeighted(ov, 0.60, out, 0.40, 0, out)
    cv2.putText(out, text, (pad, pad + th), font, font_scale, lc, thickness, cv2.LINE_AA)
    return out


# ── Temporal smoother ─────────────────────────────────────────────────────────

class TemporalSmoother:
    def __init__(self, num_classes: int, window: int = 15):
        self.buffer: collections.deque = collections.deque(maxlen=window)

    def update(self, probs: np.ndarray) -> tuple:
        self.buffer.append(probs)
        avg = np.mean(self.buffer, axis=0)
        cls = int(np.argmax(avg))
        return cls, float(avg[cls])

    def override_to_none(self) -> tuple:
        return 0, 1.0


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: str):
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg   = ckpt.get("cfg", Config())
    model = HandSegFormer(
        encoder_name         = getattr(cfg, "encoder_name",         "microsoft/swin-base-patch4-window7-224-in22k"),
        decoder_dim          = getattr(cfg, "decoder_dim",          256),
        num_action_classes   = getattr(cfg, "num_action_classes",   3),
        mask_decoder_layers  = getattr(cfg, "mask_decoder_layers",  3),
        mask_decoder_heads   = getattr(cfg, "mask_decoder_heads",   8),
        mask_decoder_dropout = getattr(cfg, "mask_decoder_dropout", 0.0),
        num_mask_queries     = getattr(cfg, "num_mask_queries",     2),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    print(f"  Epoch {ckpt.get('epoch', '?')}  |"
          f"  Val HandIoU {ckpt.get('val_hand_iou', 0):.4f}")
    return model, cfg


# ── Inference pipeline ────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model, video_path, output_path, cfg,
    hand_threshold=0.5, obj_threshold=0.5, mask_alpha=0.5,
    smooth_window=15, device="cuda", use_amp=True,
    force_none_if_no_hand=False, force_none_if_no_object=False,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps_orig = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video  : {orig_w}×{orig_h}  {fps_orig:.1f} fps  {n_frames} frames")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps_orig, (orig_w, orig_h))

    smoother  = TemporalSmoother(cfg.num_action_classes, smooth_window)
    fps_meter: collections.deque = collections.deque(maxlen=30)
    t_start   = time.perf_counter()
    frame_idx = 0

    print(f"Running inference → {output_path}\n")

    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        t0  = time.perf_counter()
        inp = preprocess_frame(bgr, cfg.img_size).to(device)

        with autocast(enabled=(use_amp and device == "cuda")):
            hand_logits, obj_logits, cls_logits, _, _ = model(inp)

        hand_mask = logit_to_mask(hand_logits, orig_h, orig_w, hand_threshold)
        obj_mask  = logit_to_mask(obj_logits,  orig_h, orig_w, obj_threshold)
        hand_pct  = mask_pixel_fraction(hand_mask)
        obj_pct   = mask_pixel_fraction(obj_mask)

        probs          = F.softmax(cls_logits[0], dim=0).cpu().numpy()
        pred_cls, conf = smoother.update(probs)

        if force_none_if_no_hand and hand_pct == 0.0:
            pred_cls, conf = smoother.override_to_none()
        elif force_none_if_no_object and obj_pct == 0.0 and pred_cls != 0:
            pred_cls, conf = smoother.override_to_none()

        label     = ACTION_CLASSES[pred_cls]
        out_frame = bgr.copy()
        if hand_pct > 0:
            out_frame = draw_hand_overlay(out_frame, hand_mask, mask_alpha)
        out_frame = draw_action_banner(out_frame, label, conf)

        fps_meter.append(1.0 / max(time.perf_counter() - t0, 1e-6))
        writer.write(out_frame)
        frame_idx += 1

        if frame_idx % 100 == 0:
            elapsed = time.perf_counter() - t_start
            print(f"  {frame_idx:>5}/{n_frames}  {np.mean(fps_meter):.1f} fps  "
                  f"[{elapsed:.0f}s elapsed]")

    cap.release()
    writer.release()
    total = time.perf_counter() - t_start
    print(f"\nDone. {frame_idx} frames in {total:.1f}s  "
          f"(avg {frame_idx / max(total, 1e-6):.1f} fps)")
    print(f"Output → {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",          required=True)
    p.add_argument("--video",               required=True)
    p.add_argument("--output",              default="output.mp4")
    p.add_argument("--img_size",            type=int,   default=512)
    p.add_argument("--hand_threshold",      type=float, default=0.5)
    p.add_argument("--obj_threshold",       type=float, default=0.5)
    p.add_argument("--mask_alpha",          type=float, default=0.5)
    p.add_argument("--smooth_window",       type=int,   default=15)
    p.add_argument("--device",              default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no_amp",              action="store_true")
    p.add_argument("--force_none_if_no_hand",   action="store_true")
    p.add_argument("--force_none_if_no_object", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args       = parse_args()
    model, cfg = load_model(args.checkpoint, args.device)
    cfg.img_size = args.img_size

    run_inference(
        model                   = model,
        video_path              = args.video,
        output_path             = args.output,
        cfg                     = cfg,
        hand_threshold          = args.hand_threshold,
        obj_threshold           = args.obj_threshold,
        mask_alpha              = args.mask_alpha,
        smooth_window           = args.smooth_window,
        device                  = args.device,
        use_amp                 = not args.no_amp,
        force_none_if_no_hand   = args.force_none_if_no_hand,
        force_none_if_no_object = args.force_none_if_no_object,
    )
