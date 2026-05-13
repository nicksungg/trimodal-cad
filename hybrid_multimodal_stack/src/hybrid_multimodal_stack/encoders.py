from typing import Callable, Optional

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel


class DinoV2Encoder(nn.Module):
    def __init__(self, model_name: str = "facebook/dinov2-base") -> None:
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=False)
        self.backbone = AutoModel.from_pretrained(model_name)

    @property
    def hidden_size(self) -> int:
        return int(self.backbone.config.hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=pixel_values, output_hidden_states=False)
        return outputs.last_hidden_state


class FrozenPointCloudEncoder(nn.Module):
    def __init__(
        self,
        encoder_builder: Callable[[], nn.Module],
        checkpoint_path: Optional[str] = None,
        strict: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder_builder()
        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self.encoder.load_state_dict(state, strict=strict)
        self.encoder.requires_grad_(False)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.encoder(points)


class IdentityPointCloudEncoder(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.dim() == 2:
            return points.unsqueeze(1)
        if points.dim() == 3:
            return points
        raise ValueError("Expected points shape [B, D] or [B, N, D]")


class RecodeFourierPointEncoder(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        frequencies = 2.0 ** torch.arange(8, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.projection = nn.Linear(51, hidden_size)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        x = points
        x = (x.unsqueeze(-1) * self.frequencies).view(*x.shape[:-1], -1)
        x = torch.cat((points, x.sin(), x.cos()), dim=-1)
        x = self.projection(x)
        return x


def build_recode_frozen_point_encoder(hidden_size: int, checkpoint_path: Optional[str]) -> nn.Module:
    encoder = RecodeFourierPointEncoder(hidden_size=hidden_size)
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        encoder.load_state_dict(state, strict=True)
    encoder.requires_grad_(False)
    return encoder
