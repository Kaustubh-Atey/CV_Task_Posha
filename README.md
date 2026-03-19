
# Hand Segmentation and Action Recognition

Egocentric hand segmentation and cooking action classification using a two-query
Mask2Former-style transformer trained on the EPIC-KITCHENS VISOR dataset.

---

### 📄 A detailed report covering the experiment, model architecture, dataset, training procedure, results and limitations is available in [`Task_Report.pdf`](./Task_Report.pdf).

### 🎬 Output videos with hand mask overlays and predicted action labels for the provided test videos are in the [`results/`](./results) directory.

---


## Overview

The model jointly predicts:
- **Hand mask** — binary segmentation of visible hands (and gloves) in each frame
- **Contacted-object mask** — binary segmentation of the object the hand is touching
- **Action label** — one of `Stirring`, `Adding ingredients`, or `None`

The object mask is not displayed at inference but is used internally: the hand and
object query embeddings are concatenated before the classification head, giving the
model context about both *what the hand looks like* and *what it is interacting with*.

---

## Repository Structure

```
├── config.py          # Training configuration dataclass (all hyperparameters and paths)
├── utils.py           # Label mapping, VISOR annotation helpers, mask builders, augmentation
├── dataset.py         # VisorDataset (raw VISOR loader) and TrainReadyDataset (tensor wrapper)
├── model.py           # PixelDecoder, MaskDecoderLayer, TransformerMaskDecoder, HandSegFormer
├── losses.py          # DiceLoss, SegmentationLoss, compute_total_loss, metrics, optimizer
├── train.py           # Training and validation loops, main entry point
├── inference.py       # Video inference — overlays hand mask and action label
├── inference_seg_only.py  # Video inference — hand mask overlay only, no action label
├── evaluate.py        # Loads checkpoint, runs classification report + confusion matrix
└── check_contact_stats.py  # Utility to analyse VISOR contact annotation statistics
```

### Script descriptions

| Script | Purpose |
|---|---|
| `config.py` | Single source of truth for all paths, model architecture, and training hyperparameters. Edit this to change any setting. |
| `utils.py` | Shared utilities: action keyword mapping, VISOR polygon extraction, hand/object mask rasterisation, Albumentations transform pipeline. |
| `dataset.py` | `VisorDataset` indexes frames from EPIC-KITCHENS CSVs and VISOR JSONs. `TrainReadyDataset` wraps it to return augmented tensors with hand mask, object mask, contact validity flag, and action label. |
| `model.py` | Full model definition. `PixelDecoder` is an FPN over Swin encoder stages. `TransformerMaskDecoder` runs two learnable queries (hand + object) via masked cross-attention. `HandSegFormer` assembles the full pipeline. |
| `losses.py` | Dice + BCE segmentation loss, auxiliary decoder loss, total loss combiner, IoU/Dice/accuracy metrics, and AdamW optimizer builder. |
| `train.py` | Training and validation loops. Handles encoder freeze/unfreeze schedule, checkpoint saving, and CSV logging. Run this to train. |
| `inference.py` | Frame-by-frame inference on a video. Overlays the predicted hand mask (cyan) and displays the action label as a text banner. Object mask is decoded internally for action prediction but not shown. |

---

## Data

Download EPIC-KITCHENS-100 RGB frames and VISOR sparse annotations from the
[official VISOR repository](https://epic-kitchens.github.io/VISOR/).

Update the paths in `config.py`:

```python
csv_path_train         = "/path/to/EPIC_100_train.csv"
csv_path_val           = "/path/to/EPIC_100_val.csv"
rgb_frames_path_train  = "/path/to/rgb_frames/train"
rgb_frames_path_val    = "/path/to/rgb_frames/val"
annotations_path_train = "/path/to/annotations/train"
annotations_path_val   = "/path/to/annotations/val"
checkpoint_dir         = "./checkpoints"
```

---

## Training

```bash
python train.py
```

Training runs for 30 epochs. The encoder is frozen for the first 3 epochs, then unfrozen with a lower learning rate.
Checkpoints are saved to `checkpoint_dir` defined in `config.py`.
The best checkpoint (by validation Hand IoU) is saved as `best.pth`.

**Key hyperparameters in `config.py`:**

| Parameter | Default | Description |
|---|---|---|
| `img_size` | 512 | Input resolution |
| `batch_size` | 64 | Training batch size |
| `epochs` | 30 | Total training epochs |
| `lr` | 6e-5 | Decoder learning rate |
| `encoder_lr_mult` | 0.1 | Encoder LR multiplier after unfreezing |
| `freeze_encoder_epochs` | 3 | Epochs to keep encoder frozen |
| `mask_decoder_layers` | 3 | Transformer decoder depth |

---

## Inference

```bash
python inference.py \
    --checkpoint  checkpoints/best.pth \
    --video       path/to/video.mp4 \
    --output      path/to/output.mp4
```

**Optional flags:**

| Flag | Default | Description |
|---|---|---|
| `--hand_threshold` | 0.5 | Sigmoid threshold for hand mask |
| `--obj_threshold` | 0.5 | Sigmoid threshold for object mask |
| `--mask_alpha` | 0.5 | Mask overlay transparency |
| `--smooth_window` | 15 | Frames to average for action label smoothing |
| `--force_none_if_no_hand` | off | Override action to None when no hand is detected |
| `--force_none_if_no_object` | off | Override Stirring/Adding to None when no object is detected |
| `--no_amp` | off | Disable mixed-precision inference |

---

## Test Video Results

Inference was run on the two provided test videos using the best checkpoint.
Output videos with hand mask overlays and predicted action labels are saved in
the `results/` directory.

| Input Video | Output |
|---|---|
| `/videos/Kitchen Training 101_ Proper Use of Gloves.mp4` | `results/video1_output.mp4` |
| `/videos/One-Pot Chicken Fajita Pasta.mp4` | `results/video2_output.mp4` |

Each output video shows:
- **Mask overlay** — predicted hand mask per frame
- **Text banner** — predicted action label with confidence score (`Action: <label> (<score>%)`)
