from typing import Optional

import torch
from torch.utils.data import Dataset


class CustomTensorDataset(Dataset):
    """This Dataset class misuses the __getitems__ API and should be used
    with an identity function as collate_fn in the DataLoader"""

    def __init__(self, *tensors: torch.Tensor):
        length = len(tensors[0])
        for t in tensors:
            assert len(t) == length, "tensors should have the same length"
        self.tensors = tensors

    def __len__(self) -> int:
        return len(self.tensors[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return tuple(tensor[index] for tensor in self.tensors)

    def __getitems__(self, indices: list[int]) -> tuple[torch.Tensor, ...]:
        return tuple(tensor[indices] for tensor in self.tensors)


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

    return CustomTensorDataset(coordinates, pixel_values)
