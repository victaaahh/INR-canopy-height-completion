from torch import Tensor, nn
from typing_extensions import override


class ReLUMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int,
        guide_out_features: int,
    ):
        super().__init__()
        self.first_layer = nn.Sequential(
            nn.Linear(in_features, hidden_features), nn.ReLU()
        )

        layers: list[nn.Module] = []
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(nn.ReLU())

        self.middle_layers = nn.Sequential(*layers)

        self.last_layer = nn.Linear(hidden_features, out_features)

        self.guide_head = nn.Linear(hidden_features, guide_out_features)

    @override
    def forward(self, coords: Tensor):
        x = self.first_layer(coords)
        x = self.middle_layers(x)
        return self.last_layer(x), self.guide_head(x)
