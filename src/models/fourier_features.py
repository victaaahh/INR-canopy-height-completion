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
        self.middle_layer = nn.Sequential(*layers)

        self.last_layer = nn.Linear(hidden_features, out_features)

        self.guide_head = nn.Linear(hidden_features, guide_out_features)

        self.first_layers = [self.first_layer]
        self.middle_layers = [self.middle_layer]
        self.last_layers = [self.last_layer, self.guide_head]

    @override
    def forward(self, coords: Tensor) -> tuple[Tensor, Tensor]:
        x = 2 * torch.pi * coords @ self.B
        x = torch.hstack((torch.cos(x), torch.sin(x)))
        x = self.first_layer(x)
        x = self.middle_layer(x)
        return self.last_layer(x), self.guide_head(x)


class RandomFourierFeaturesTargetToGuide(nn.Module):
    def __init__(
        self,
        in_features: int,
        fourier_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int,
        guide_hidden_features: int,
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
        self.middle_layer = nn.Sequential(*layers)

        self.last_layer = nn.Linear(hidden_features, out_features)

        if guide_hidden_features > 0:
            self.guide_head = nn.Sequential(
                nn.Linear(out_features + 2, guide_hidden_features),
                nn.ReLU(),
                nn.Linear(guide_hidden_features, guide_out_features),
            )
        else:
            self.guide_head = nn.Linear(out_features + 2, guide_out_features)

        self.first_layers = [self.first_layer]
        self.middle_layers = [self.middle_layer]
        self.last_layers = [self.last_layer, self.guide_head]

    @override
    def forward(self, coords: Tensor) -> tuple[Tensor, Tensor]:
        x = 2 * torch.pi * coords @ self.B
        x = torch.hstack((torch.cos(x), torch.sin(x)))
        x = self.first_layer(x)
        x = self.middle_layer(x)
        x_target = self.last_layer(x)
        x_guide = self.guide_head(torch.hstack((coords, x_target)))
        return x_target, x_guide


class RandomFourierFeaturesTargetToGuideFF(nn.Module):
    def __init__(
        self,
        in_features: int,
        fourier_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int,
        guide_fourier_features: int,
        guide_hidden_features: int,
        guide_out_features: int,
        scale: float,
        guide_scale: float,
    ):
        super().__init__()
        self.B = nn.Parameter(
            torch.randn(in_features, fourier_features) * scale, requires_grad=False
        )
        self.B_guide = nn.Parameter(
            torch.randn(in_features, guide_fourier_features) * guide_scale,
            requires_grad=False,
        )

        self.first_layer = nn.Sequential(
            nn.Linear(fourier_features * 2, hidden_features), nn.ReLU()
        )

        layers: list[nn.Module] = []
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(nn.ReLU())
        self.middle_layer = nn.Sequential(*layers)

        self.last_layer = nn.Linear(hidden_features, out_features)

        if guide_hidden_features > 0:
            self.guide_head = nn.Sequential(
                nn.Linear(
                    out_features + guide_fourier_features * 2, guide_hidden_features
                ),
                nn.ReLU(),
                nn.Linear(guide_hidden_features, guide_out_features),
            )
        else:
            self.guide_head = nn.Linear(
                out_features + fourier_features * 2, guide_out_features
            )

        self.first_layers = [self.first_layer]
        self.middle_layers = [self.middle_layer]
        self.last_layers = [self.last_layer, self.guide_head]

    @override
    def forward(self, coords: Tensor) -> tuple[Tensor, Tensor]:
        ff_target = 2 * torch.pi * coords @ self.B
        ff_target = torch.hstack((torch.cos(ff_target), torch.sin(ff_target)))
        ff_guide = 2 * torch.pi * coords @ self.B_guide
        ff_guide = torch.hstack((torch.cos(ff_guide), torch.sin(ff_guide)))
        x = self.first_layer(ff_target)
        x = self.middle_layer(x)
        x_target = self.last_layer(x)
        x_guide = self.guide_head(torch.hstack((ff_guide, x_target)))
        return x_target, x_guide


class RandomFourierFeaturesTargetToGuideBigHead(nn.Module):
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
        self.middle_layer = nn.Sequential(*layers)

        self.last_layer = nn.Linear(hidden_features, out_features)

        self.guide_coord_encoder = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 2048),
        )
        self.guide_height_encoder = nn.Sequential(
            nn.Linear(out_features, 32),
            nn.ReLU(),
            nn.Linear(32, 2048),
        )
        self.guide_decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2048, 32),
            nn.ReLU(),
            nn.Linear(32, guide_out_features),
        )

        self.first_layers = [self.first_layer]
        self.middle_layers = [self.middle_layer]
        self.last_layers = [self.last_layer, self.guide_head]

    @override
    def forward(self, coords: Tensor) -> tuple[Tensor, Tensor]:
        ff_target = 2 * torch.pi * coords @ self.B
        ff_target = torch.hstack((torch.cos(ff_target), torch.sin(ff_target)))
        x = self.first_layer(ff_target)
        x = self.middle_layer(x)
        x_target = self.last_layer(x)
        guide_coords = self.guide_coord_encoder(coords)
        guide_height = self.guide_height_encoder(x_target)
        x_guide = self.guide_decoder(guide_coords + guide_height)
        return x_target, x_guide
