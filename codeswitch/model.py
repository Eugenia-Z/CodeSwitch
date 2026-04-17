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

    Optional ``use_causal_at_eval``: training uses one full bidirectional pass; when
    ``eval()`` is active, forward builds hidden state at each t from an encoder run
    on the prefix ``0..t`` only, so position t cannot attend to t+1..L−1.
    """

    def __init__(
        self,
        model_name:          str   = "xlm-roberta-base",
        dropout:             float = 0.1,
        freeze_encoder:      bool  = False,
        use_causal_at_eval:  bool  = False,
    ) -> None:
        super().__init__()
        self.use_causal_at_eval = use_causal_at_eval
        print(f"Loading {model_name}...")
        self.encoder = XLMRobertaModel.from_pretrained(model_name)
        d_model = self.encoder.config.hidden_size

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            print("  Encoder frozen — only heads are trainable")
        else:
            print("  Encoder UNFROZEN — full fine-tuning")

        if use_causal_at_eval:
            print(
                "  Eval: causal prefix decoding (train stays bidirectional full sequence)"
            )

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

    def _forward_bidirectional(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out    = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        hidden = out.last_hidden_state[:, :-1, :]
        return self.switch_head(hidden), self.duration_head(hidden)

    def _forward_causal_prefix(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """For each t in 0..L-2, run encoder on tokens 0..t and take hidden[:, t, :]."""
        B, L = input_ids.shape
        if L < 2:
            d_model = self.encoder.config.hidden_size
            dtype   = self.encoder.embeddings.word_embeddings.weight.dtype
            empty   = input_ids.new_zeros(B, 0, d_model, dtype=dtype)
            return self.switch_head(empty), self.duration_head(empty)

        rows: list[torch.Tensor] = []
        for t in range(L - 1):
            end  = t + 1
            ids  = input_ids[:, :end]
            mask = attention_mask[:, :end] if attention_mask is not None else None
            out  = self.encoder(input_ids=ids, attention_mask=mask, return_dict=True)
            rows.append(out.last_hidden_state[:, t, :])
        hidden = torch.stack(rows, dim=1)
        return self.switch_head(hidden), self.duration_head(hidden)

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        causal = self.use_causal_at_eval and (not self.training)
        if causal:
            return self._forward_causal_prefix(input_ids, attention_mask)
        return self._forward_bidirectional(input_ids, attention_mask)
