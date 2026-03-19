"""model.py — PixelDecoder, MaskDecoderLayer, TransformerMaskDecoder, HandSegFormer."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SwinModel


class PixelDecoder(nn.Module):
    """FPN over the 4 Swin encoder stages with GroupNorm lateral connections."""

    def __init__(self, hidden_sizes: list, out_dim: int):
        super().__init__()
        self.lateral = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, out_dim, kernel_size=1, bias=False),
                nn.GroupNorm(32, out_dim),
            )
            for c in hidden_sizes
        ])
        self.smooth = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(32, out_dim),
                nn.ReLU(inplace=True),
            )
            for _ in hidden_sizes
        ])

    def forward(self, feats: tuple) -> tuple:
        laterals = [lat(f) for lat, f in zip(self.lateral, feats)]
        fpn      = [None] * len(feats)
        fpn[-1]  = laterals[-1]
        for i in range(len(feats) - 2, -1, -1):
            fpn[i] = laterals[i] + F.interpolate(
                fpn[i + 1], size=laterals[i].shape[2:],
                mode="bilinear", align_corners=False,
            )
        fpn = [sm(f) for sm, f in zip(self.smooth, fpn)]
        return fpn[0], fpn   # pixel_feat (finest), all levels


class MaskDecoderLayer(nn.Module):
    """
    One Mask2Former-style decoder layer:
      1. Masked cross-attention  (query attends inside its predicted mask)
      2. Self-attention          (hand ↔ object queries interact)
      3. FFN with pre-LayerNorm
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads  = num_heads
        self.norm_cross = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_self  = nn.LayerNorm(dim)
        self.self_attn  = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_ffn   = nn.LayerNorm(dim)
        self.ffn        = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout),
        )

    def forward(self, queries: torch.Tensor, pixel_feat: torch.Tensor,
                prev_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            region  = (prev_mask.sigmoid() > 0.5)
            any_pos = region.any(dim=-1, keepdim=True)
            region  = torch.where(any_pos, region, torch.ones_like(region))
            bias    = (~region).float() * -1e4
            B, Q, S = bias.shape
            bias    = bias.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            bias    = bias.reshape(B * self.num_heads, Q, S)

        q_out, _ = self.cross_attn(
            self.norm_cross(queries), pixel_feat, pixel_feat, attn_mask=bias
        )
        queries  = queries + q_out
        q_norm   = self.norm_self(queries)
        q_out, _ = self.self_attn(q_norm, q_norm, q_norm)
        queries  = queries + q_out
        queries  = queries + self.ffn(self.norm_ffn(queries))
        return queries


class TransformerMaskDecoder(nn.Module):
    """
    Stack of MaskDecoderLayer with intermediate mask predictions.

    Returns:
      final_logit  : (B, Q, H', W')
      aux_logits   : list of (B, Q, H', W') — one per intermediate layer
      final_queries: (B, Q, D) — for classification head
    """

    def __init__(self, dim: int, num_heads: int, num_layers: int,
                 num_queries: int = 2, dropout: float = 0.0):
        super().__init__()
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, dim)
        nn.init.trunc_normal_(self.query_embed.weight, std=0.02)
        self.layers     = nn.ModuleList([
            MaskDecoderLayer(dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.mask_proj  = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])
        self.query_norm = nn.LayerNorm(dim)

    def forward(self, pixel_feat: torch.Tensor, fpn_levels: list) -> tuple:
        B, D, Hp, Wp = pixel_feat.shape
        S       = Hp * Wp
        pf_flat = pixel_feat.flatten(2).permute(0, 2, 1)
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        prev_mask = torch.zeros(B, self.num_queries, S, device=pixel_feat.device)

        aux_logits = []
        for layer, proj in zip(self.layers, self.mask_proj):
            queries   = layer(queries, pf_flat, prev_mask)
            qp        = proj(queries)
            logit     = torch.einsum('bqd,bds->bqs', qp, pixel_feat.flatten(2))
            logit     = logit.view(B, self.num_queries, Hp, Wp)
            prev_mask = logit.flatten(2)
            aux_logits.append(logit)

        queries     = self.query_norm(queries)
        final_proj  = self.mask_proj[-1](queries)
        final_logit = torch.einsum(
            'bqd,bds->bqs', final_proj, pixel_feat.flatten(2)
        ).view(B, self.num_queries, Hp, Wp)

        return final_logit, aux_logits[:-1], queries


class HandSegFormer(nn.Module):
    """
    Swin-B encoder + FPN PixelDecoder + TransformerMaskDecoder (2 queries).

    Query-1 → hand mask
    Query-2 → contacted-object mask
    Classification: MLP on concatenated query embeddings [q_hand | q_obj]
    """

    def __init__(
        self,
        encoder_name:         str   = "microsoft/swin-base-patch4-window7-224-in22k",
        decoder_dim:          int   = 256,
        num_action_classes:   int   = 3,
        mask_decoder_layers:  int   = 3,
        mask_decoder_heads:   int   = 8,
        mask_decoder_dropout: float = 0.0,
        num_mask_queries:     int   = 2,
    ):
        super().__init__()
        self.encoder = SwinModel.from_pretrained(
            encoder_name, output_hidden_states=True, add_pooling_layer=False
        )
        cfg_enc      = self.encoder.config
        hidden_sizes = [int(cfg_enc.embed_dim * (2 ** i))
                        for i in range(len(cfg_enc.depths))]
        self.enc_deep_dim = hidden_sizes[-1]
        self.decoder_dim  = decoder_dim

        self.pixel_decoder = PixelDecoder(hidden_sizes, decoder_dim)
        self.mask_decoder  = TransformerMaskDecoder(
            dim         = decoder_dim,
            num_heads   = mask_decoder_heads,
            num_layers  = mask_decoder_layers,
            num_queries = num_mask_queries,
            dropout     = mask_decoder_dropout,
        )
        self.cls_head = nn.Sequential(
            nn.Linear(decoder_dim * num_mask_queries, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_action_classes),
        )

    def forward(self, pixel_values: torch.Tensor):
        B, _, H, W = pixel_values.shape
        enc_out    = self.encoder(pixel_values, output_hidden_states=True)
        all_feats  = enc_out.reshaped_hidden_states  # tuple of 5, NCHW
        feats      = all_feats[:4]                   # [128, 256, 512, 1024] for FPN

        pixel_feat, fpn_levels        = self.pixel_decoder(feats)
        final_logit, aux_logits, qs   = self.mask_decoder(pixel_feat, fpn_levels)

        hand_logits = F.interpolate(
            final_logit[:, 0:1], size=(H, W), mode="bilinear", align_corners=False
        )
        obj_logits = F.interpolate(
            final_logit[:, 1:2], size=(H, W), mode="bilinear", align_corners=False
        )
        cls_logits = self.cls_head(qs.reshape(B, -1))

        aux_hand = [a[:, 0:1] for a in aux_logits]
        aux_obj  = [a[:, 1:2] for a in aux_logits]

        return hand_logits, obj_logits, cls_logits, aux_hand, aux_obj

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True
