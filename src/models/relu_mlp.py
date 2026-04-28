from torch import nn


class ReLUMLP(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features):
        super().__init__()
        self.first_layer = nn.Sequential(
            nn.Linear(in_features, hidden_features), nn.ReLU()
        )

        layers = []
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(nn.ReLU())

        self.middle_layers = nn.Sequential(*layers)

        self.last_layer = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.first_layer(x)
        x = self.middle_layers(x)
        return self.last_layer(x)
