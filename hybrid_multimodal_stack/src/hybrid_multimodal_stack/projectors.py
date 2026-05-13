import re
import torch
import torch.nn as nn


class IdentityProjector(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class MLPProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, projector_type: str = "mlp2x_gelu") -> None:
        super().__init__()
        if projector_type == "linear":
            self.net = nn.Linear(in_dim, out_dim)
            return

        if projector_type == "identity":
            if in_dim != out_dim:
                raise ValueError("identity projector requires in_dim == out_dim")
            self.net = IdentityProjector()
            return

        match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
        if match is None:
            raise ValueError(f"Unknown projector type: {projector_type}")

        depth = int(match.group(1))
        layers = [nn.Linear(in_dim, out_dim)]
        for _ in range(1, depth):
            layers.append(nn.GELU())
            layers.append(nn.Linear(out_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
