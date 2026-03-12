from typing import Optional

import torch
from torch.utils.data import Dataset


class SingleImage(Dataset):
    """Torch dataset with the coordinates and pixel values of a single image.
    The coordinates are in the range [-1, 1]. Top left corner is (-1, -1),
    the pixel to the right is (-1, -0.xxx) and so on"""

    def __init__(
        self,
        image: torch.Tensor,
        channel_dim: int,
        device: Optional[torch.device] = None,
    ):
        assert len(image.shape) == 3, "image should have channel, width and height dims"
        assert channel_dim in [0, 2], "invalid channel_dim"
        if channel_dim == 0:
            image = torch.permute(image, (1, 2, 0))
        assert image.shape[0] == image.shape[1], "image should be a square"

        coord_range = torch.linspace(-1, 1, image.shape[0])
        self.coordinates = torch.stack(
            torch.meshgrid(coord_range, coord_range, indexing="ij"), dim=-1
        ).reshape(-1, 2)

        self.pixel_values = torch.reshape(image, (-1, image.shape[2]))

        if device:
            self.coordinates = self.coordinates.to(device)
            self.pixel_values = self.pixel_values.to(device)

    def __len__(self) -> int:
        return self.coordinates.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (self.coordinates[index], self.pixel_values[index])
