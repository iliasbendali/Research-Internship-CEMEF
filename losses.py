# losses.py
from __future__ import annotations
import torch
import torch.nn.functional as F
import numpy as np


@torch.no_grad()
def compute_pos_weight_from_loader(
    loader,
    num_classes: int = 17,
    max_pos_weight: float = 20.0,
    device: str | torch.device = "cpu",
    max_batches: int | None = None,
) -> torch.Tensor:
    pos_sum = torch.zeros(num_classes, dtype=torch.float64)
    neg_sum = torch.zeros(num_classes, dtype=torch.float64)

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        y = batch["y"]
        mask = batch.get("mask", None)

        y = y.to(torch.float64)
        if mask is None:
            m = torch.ones(y.shape[:2], dtype=torch.float64)
        else:
            m = mask.to(torch.float64)

        m = m.unsqueeze(-1)
        pos_sum += (y * m).sum(dim=(0, 1)).cpu()
        neg_sum += ((1.0 - y) * m).sum(dim=(0, 1)).cpu()

    eps = 1e-7
    ratio = (neg_sum / (pos_sum + eps)).to(torch.float32)
    pos_weight = torch.sqrt(ratio)
    pos_weight = torch.clamp(pos_weight, min=1.0, max=max_pos_weight)

    return pos_weight.to(device)



def masked_bce_with_logits_loss(
    logits: torch.Tensor,     # (B,L,C)
    targets: torch.Tensor,    # (B,L,C) in [0,1]
    mask: torch.Tensor | None = None,  # (B,L) in {0,1}
    pos_weight: torch.Tensor | None = None,  # (C,)
) -> torch.Tensor:
    """
    BCEWithLogitsLoss + mask padding.
    Retourne un scalaire.
    """
    # BCE element-wise: (B,L,C)
    loss_el = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=pos_weight,
        reduction="none",
    )

    if mask is None:
        return loss_el.mean()

    # mask: (B,L,1)
    m = mask.unsqueeze(-1).to(loss_el.dtype)
    loss_el = loss_el * m

    denom = m.sum() * logits.shape[-1]  # nb d'éléments valides
    denom = torch.clamp(denom, min=1.0)

    return loss_el.sum() / denom

import torch
import torch.nn.functional as F

def masked_asymmetric_focal_loss_with_logits(
    logits: torch.Tensor,     # (B,L,C)
    targets: torch.Tensor,    # (B,L,C) in [0,1]
    mask: torch.Tensor = None,# (B,L) in {0,1}
    gamma_pos: float = 2.0,
    gamma_neg: float = 4.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Asymmetric focal loss (multi-label) sur logits.
    - pousse plus fort sur les faux positifs via gamma_neg
    - garde du signal sur les positives via gamma_pos
    """
    prob = torch.sigmoid(logits)
    pt_pos = prob
    pt_neg = 1.0 - prob

    # BCE élémentaire
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    # facteur focal asymétrique
    w_pos = (1.0 - pt_pos).clamp(min=0.0) ** gamma_pos
    w_neg = (1.0 - pt_neg).clamp(min=0.0) ** gamma_neg
    focal_w = targets * w_pos + (1.0 - targets) * w_neg

    loss_el = focal_w * bce  # (B,L,C)

    if mask is None:
        return loss_el.mean()

    m = mask.unsqueeze(-1).to(loss_el.dtype)
    loss_el = loss_el * m
    denom = torch.clamp(m.sum() * logits.shape[-1], min=1.0)
    return loss_el.sum() / denom


import torch
import torch.nn.functional as F

def masked_focal_bce_with_logits(
    logits: torch.Tensor,          # (B,L,C)
    targets: torch.Tensor,         # (B,L,C) in [0,1]
    mask: torch.Tensor,            # (B,L)
    pos_weight: torch.Tensor | None = None,  # (C,)
    gamma: float = 2.0,
    reduction: str = "mean",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Focal loss binaire stable:
      FL = (1 - pt)^gamma * BCE_with_logits
    avec mask temporel + pos_weight par classe.

    targets peut être soft (gauss), c'est OK.
    """
    B, L, C = logits.shape
    mask3 = mask.unsqueeze(-1).float()  # (B,L,1)

    # BCE par élément (B,L,C)
    bce = F.binary_cross_entropy_with_logits(
        logits, targets,
        reduction="none",
        pos_weight=pos_weight
    )

    # pt = probabilité du label correct
    probs = torch.sigmoid(logits)
    pt = probs * targets + (1.0 - probs) * (1.0 - targets)  # (B,L,C)
    focal = (1.0 - pt).clamp(min=0.0, max=1.0).pow(gamma)

    loss = focal * bce
    loss = loss * mask3

    if reduction == "mean":
        denom = mask3.sum() * C
        return loss.sum() / (denom.clamp_min(eps))
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss
