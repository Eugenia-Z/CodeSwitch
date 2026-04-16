"""Training utilities: seed, class weights, single-epoch training loop."""
from __future__ import annotations
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TrainConfig


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(
    loader: DataLoader,
    num_sw_classes:  int = 2,
    num_dur_classes: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Count class frequencies and return inverse-frequency weights (min-normalized)."""
    sw_counts  = torch.zeros(num_sw_classes)
    dur_counts = torch.zeros(num_dur_classes)

    for batch in loader:
        y_sw  = batch["y_switch"].reshape(-1)
        y_dur = batch["y_duration"].reshape(-1)
        for c in range(num_sw_classes):
            sw_counts[c]  += (y_sw  == c).sum()
        for c in range(num_dur_classes):
            dur_counts[c] += (y_dur == c).sum()

    sw_w  = 1.0 / sw_counts;  sw_w  /= sw_w.min()
    dur_w = 1.0 / dur_counts; dur_w /= dur_w.min()

    print(f"  Switch  counts:  {sw_counts.long().tolist()}")
    print(f"  Switch  weights: {[f'{w:.4f}' for w in sw_w.tolist()]}")
    print(f"  Duration counts:  {dur_counts.long().tolist()}")
    print(f"  Duration weights: {[f'{w:.4f}' for w in dur_w.tolist()]}")

    return sw_w, dur_w


def train_epoch(
    model:         nn.Module,
    loader:        DataLoader,
    optimizer,
    device:        torch.device,
    sw_criterion:  nn.Module,
    dur_criterion: nn.Module,
    config:        TrainConfig,
) -> dict[str, float]:
    model.train()
    sw_losses:  list[float] = []
    dur_losses: list[float] = []

    for batch in tqdm(loader, desc="  train", leave=False):
        ids   = batch["input_ids"].to(device)
        mask  = batch["attention_mask"].to(device)
        y_sw  = batch["y_switch"].to(device)
        y_dur = batch["y_duration"].to(device)

        sw_logits, dur_logits = model(ids, mask)

        loss_sw  = sw_criterion(sw_logits.reshape(-1, 2), y_sw.reshape(-1))

        dur_flat  = y_dur.reshape(-1)
        dlog_flat = dur_logits.reshape(-1, 3)
        valid     = dur_flat >= 0
        loss_dur  = (
            dur_criterion(dlog_flat[valid], dur_flat[valid])
            if valid.any()
            else torch.tensor(0.0, device=device)
        )

        loss = loss_sw + config.lambda_dur * loss_dur

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        sw_losses.append(loss_sw.item())
        dur_losses.append(loss_dur.item())

    return {
        "loss_sw":  float(np.mean(sw_losses)),
        "loss_dur": float(np.mean(dur_losses)),
    }
