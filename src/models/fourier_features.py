import torch
from torch import Tensor, nn
from typing_extensions import override


class RandomFourierFeatures(nn.Module):
    def __init__(
        self,
        in_features: int,
        fourier_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int,
        guide_out_features: int,
        scale: float,
    ):
        super().__init__()
        self.B = nn.Parameter(
            torch.randn(in_features, fourier_features) * scale, requires_grad=False
        )

        self.first_layer = nn.Sequential(
            nn.Linear(fourier_features * 2, hidden_features), nn.ReLU()
        )

        layers: list[nn.Module] = []
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(nn.ReLU())
        self.middle_layers = nn.Sequential(*layers)

        self.last_layer = nn.Linear(hidden_features, out_features)

        self.guide_head = nn.Linear(hidden_features, guide_out_features)

    @override
    def forward(self, coords: Tensor) -> tuple[Tensor, Tensor]:
        x = 2 * torch.pi * coords @ self.B
        x = torch.hstack((torch.cos(x), torch.sin(x)))
        x = self.first_layer(x)
        x = self.middle_layers(x)
        return self.last_layer(x), self.guide_head(x)
