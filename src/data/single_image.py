from typing import Optional

import torch
from torch.utils.data import TensorDataset


def single_image_dataset(
    image: torch.Tensor, channel_dim: int, device: Optional[torch.device] = None
):
    assert len(image.shape) == 3, "image should have channel, width and height dims"
    assert channel_dim in [0, 2], "invalid channel_dim"
    if channel_dim == 0:
        image = torch.permute(image, (1, 2, 0))
    assert image.shape[0] == image.shape[1], "image should be a square"

    coord_range = torch.linspace(-1, 1, image.shape[0])
    coordinates = torch.stack(
        torch.meshgrid(coord_range, coord_range, indexing="ij"), dim=-1
    ).reshape(-1, 2)

    pixel_values = torch.reshape(image, (-1, image.shape[2]))

    if device:
        coordinates = coordinates.to(device)
        pixel_values = pixel_values.to(device)

    return TensorDataset(coordinates, pixel_values)
