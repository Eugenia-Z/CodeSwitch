"""Model architectures for code-switching prediction.

Supported backbones
-------------------
xlmr  — XLM-RoBERTa (encoder-only, bidirectional by default).
         Causal masking is applied manually via a 4D extended mask so
         position t attends only to positions 0..t.
xglm  — XGLM / GPT-style (decoder-only, natively causal).
         Standard 2D padding mask is sufficient; internal layers handle
         causality automatically.

Both share the same dual-head structure:
  switch_head   — binary classification (no-switch / switch at t → predicts t+1)
  duration_head — 3-class burst-length (small / medium / large on switch events)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import XGLMModel, XLMRobertaModel

from .config import BACKBONE_MODEL_DEFAULTS, ModelConfig


# ── Causal mask utilities (XLM-R only) ───────────────────────────────────────

def _make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Upper-triangular causal mask in XLM-R's extended-attention format.

    Blocked positions are filled with -1e9 (effectively -inf after softmax).
    Shape: [1, 1, seq_len, seq_len].
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1) * (-1e9)
    return mask.unsqueeze(0).unsqueeze(0)


def _combine_pad_and_causal_mask(
    attention_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Merge 2D padding mask [B, L] with upper-triangular causal mask → [B, 1, L, L]."""
    B, L = attention_mask.shape
    pad_mask = (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2) * (-1e9)
    return pad_mask + _make_causal_mask(L, device)


# ── Shared head factory ───────────────────────────────────────────────────────

def _make_heads(
    d_model: int, dropout: float
) -> tuple[nn.Sequential, nn.Sequential]:
    def _head(out_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, out_dim),
        )
    return _head(2), _head(3)


def _log_param_counts(model: nn.Module) -> None:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model ready  |  total={total:,}  trainable={trainable:,}")


# ── XLM-R backbone (manual causal masking) ───────────────────────────────────

class CausalXLMRCodeSwitchPredictor(nn.Module):
    """XLM-R encoder with manual 4D causal masking and dual classification heads."""

    def __init__(
        self,
        model_name:     str   = "xlm-roberta-base",
        dropout:        float = 0.1,
        freeze_encoder: bool  = False,
    ) -> None:
        super().__init__()
        print(f"Loading {model_name} (CAUSAL — manual 4D mask)...")
        self.encoder = XLMRobertaModel.from_pretrained(model_name)
        d_model = self.encoder.config.hidden_size

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            print("  Encoder FROZEN")
        else:
            print("  Encoder UNFROZEN — full fine-tuning")

        self.switch_head, self.duration_head = _make_heads(d_model, dropout)
        _log_param_counts(self)

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if attention_mask is not None:
            ext_mask = _combine_pad_and_causal_mask(attention_mask, input_ids.device)
        else:
            B, L     = input_ids.shape
            ext_mask = _make_causal_mask(L, input_ids.device).expand(B, -1, -1, -1)

        out    = self.encoder(input_ids=input_ids, attention_mask=ext_mask, return_dict=True)
        hidden = out.last_hidden_state[:, :-1, :]
        return self.switch_head(hidden), self.duration_head(hidden)


# Backward-compat alias (existing checkpoints and imports still work)
XLMRCodeSwitchPredictor = CausalXLMRCodeSwitchPredictor


# ── XGLM backbone (native causal masking) ────────────────────────────────────

class XGLMCodeSwitchPredictor(nn.Module):
    """XGLM (GPT-backbone) with native causal masking and dual classification heads.

    XGLM is decoder-only: position t attends only to 0..t by construction.
    We pass only the standard 2D padding mask; causal masking is handled internally.
    """

    def __init__(
        self,
        model_name:     str   = "facebook/xglm-564M",
        dropout:        float = 0.1,
        freeze_encoder: bool  = False,
    ) -> None:
        super().__init__()
        print(f"Loading {model_name} (CAUSAL — GPT backbone)...")
        self.encoder = XGLMModel.from_pretrained(model_name)
        d_model = self.encoder.config.hidden_size

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            print("  Encoder FROZEN")
        else:
            print("  Encoder UNFROZEN — full fine-tuning")

        self.switch_head, self.duration_head = _make_heads(d_model, dropout)
        _log_param_counts(self)

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out    = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden = out.last_hidden_state[:, :-1, :]
        return self.switch_head(hidden), self.duration_head(hidden)


# ── Factory ───────────────────────────────────────────────────────────────────

def build_model(cfg: ModelConfig) -> nn.Module:
    """Instantiate the correct predictor from a ModelConfig."""
    model_name = cfg.model_name or BACKBONE_MODEL_DEFAULTS.get(cfg.backbone, "xlm-roberta-base")

    if cfg.backbone == "xglm":
        return XGLMCodeSwitchPredictor(
            model_name=model_name,
            dropout=cfg.dropout,
            freeze_encoder=cfg.freeze_encoder,
        )

    return CausalXLMRCodeSwitchPredictor(
        model_name=model_name,
        dropout=cfg.dropout,
        freeze_encoder=cfg.freeze_encoder,
    )
