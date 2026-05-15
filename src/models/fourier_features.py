import torch
from torch import nn


class RandomFourierFeatures(nn.Module):
    def __init__(
        self,
        in_features,
        fourier_features,
        hidden_features,
        hidden_layers,
        out_features,
        guide_out_features,
        scale,
    ):
        super().__init__()
        self.B = nn.Parameter(
            torch.randn(in_features, fourier_features) * scale, requires_grad=False
        )

        self.first_layer = nn.Sequential(
            nn.Linear(fourier_features * 2, hidden_features), nn.ReLU()
        )

        layers = []
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(nn.ReLU())
        self.middle_layers = nn.Sequential(*layers)

        self.last_layer = nn.Linear(hidden_features, out_features)

        self.guide_head = nn.Linear(hidden_features, guide_out_features)

    def forward(self, x):
        x = 2 * torch.pi * x @ self.B
        x = torch.hstack((torch.cos(x), torch.sin(x)))
        x = self.first_layer(x)
        x = self.middle_layers(x)
        return self.last_layer(x), self.guide_head(x)
