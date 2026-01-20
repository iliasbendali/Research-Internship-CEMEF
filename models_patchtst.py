# models_patchtst.py
from __future__ import annotations
import torch
import torch.nn as nn
from transformers import PatchTSTConfig, PatchTSTModel


class PatchTSTSpotter(nn.Module):
    def __init__(
        self,
        num_input_channels: int = 512,
        d_model: int = 256,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 8,
        ffn_dim: int = 1024,
        patch_length: int = 1,
        patch_stride: int = 1,
        channel_attention: bool = True,  # important si tu veux mixer les 512 dims
        pooling_over_channels: str = "mean",  # "mean" ou "max"
    ):
        super().__init__()

        self.pooling_over_channels = pooling_over_channels

        config = PatchTSTConfig(
            num_input_channels=num_input_channels,
            context_length=1,          # placeholder, pas bloquant en pratique (tu passes des windows)
            patch_length=patch_length,
            patch_stride=patch_stride,
            d_model=d_model,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            ffn_dim=ffn_dim,
            channel_attention=channel_attention,
        )

        self.backbone = PatchTSTModel(config)
        self.head = nn.Linear(d_model, 17)

    def forward(self, past_values: torch.Tensor):
        """
        past_values: (B, T, 512)
        return: dict(logits=(B, T, 17))
        """
        out = self.backbone(past_values=past_values)

        h = out.last_hidden_state  # attention: peut être (B,C,T,d_model) selon implémentation :contentReference[oaicite:3]{index=3}

        # Cas 4D : (B, C, T, d_model) -> pool channels -> (B, T, d_model)
        if h.dim() == 4:
            if self.pooling_over_channels == "max":
                h = h.max(dim=1).values
            else:
                h = h.mean(dim=1)

        # Cas 3D : (B, T, d_model) direct
        if h.dim() != 3:
            raise RuntimeError(f"Unexpected hidden state shape: {tuple(h.shape)}")

        logits = self.head(h)  # (B, T, 17)
        return {"logits": logits}
