"""XLM-R based code-switching predictor (switch + duration heads)."""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import XLMRobertaModel


class XLMRCodeSwitchPredictor(nn.Module):
    """
    Full-fine-tuning XLM-R encoder with two token-level classification heads:
      - switch_head:   binary (no-switch / switch) at position t → predicts t+1
      - duration_head: 3-class burst length (small / medium / large) on switch events
    """

    def __init__(
        self,
        model_name:     str   = "xlm-roberta-base",
        dropout:        float = 0.1,
        freeze_encoder: bool  = False,
    ) -> None:
        super().__init__()
        print(f"Loading {model_name}...")
        self.encoder = XLMRobertaModel.from_pretrained(model_name)
        d_model = self.encoder.config.hidden_size

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            print("  Encoder frozen — only heads are trainable")
        else:
            print("  Encoder UNFROZEN — full fine-tuning")

        def _head(out_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, out_dim),
            )

        self.switch_head   = _head(2)
        self.duration_head = _head(3)

        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"✓ Model ready  |  total={total:,}  trainable={trainable:,}")

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out    = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        # Shift: h[t] → predict position t+1; drop the last token's hidden state
        hidden = out.last_hidden_state[:, :-1, :]
        return self.switch_head(hidden), self.duration_head(hidden)
