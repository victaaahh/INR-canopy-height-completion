from typing import Optional

import torch
from torch.utils.data import Dataset


class CustomTensorDataset(Dataset):
    """This Dataset class misuses the __getitems__ API and should be used
    with an identity function as collate_fn in the DataLoader"""

    def __init__(self, *tensors: torch.Tensor):
        assert len(tensors) > 0, "must supply at least one tensor"
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
    image: torch.Tensor,
    channel_dim: int,
    device: Optional[torch.device] = None,
) -> CustomTensorDataset:
    assert len(image.shape) == 3, "image should have channel, width and height dims"
    assert channel_dim in [0, 2], "invalid channel_dim"
    if channel_dim == 0:
        image = torch.permute(image, (1, 2, 0))
    assert image.shape[0] == image.shape[1], "image should be a square"
    image_width = image.shape[0]

    coord_range = torch.linspace(-1, 1, image_width)
    coordinates = torch.stack(
        torch.meshgrid(coord_range, coord_range, indexing="ij"), dim=-1
    ).reshape(-1, 2)

    pixel_values = image.reshape(-1, image.shape[2])

    if device:
        coordinates = coordinates.to(device)
        pixel_values = pixel_values.to(device)

    return CustomTensorDataset(coordinates, pixel_values)


def single_image_sparse_dataset(
    image: torch.Tensor,
    channel_dim: int,
    test_grid_skip: int,
    sparsity: float,
    device: Optional[torch.device] = None,
) -> tuple[CustomTensorDataset, CustomTensorDataset, torch.Tensor, torch.Tensor]:
    assert len(image.shape) == 3, "image should have channel, width and height dims"
    assert channel_dim in [0, 2], "invalid channel_dim"
    if channel_dim == 0:
        image = torch.permute(image, (1, 2, 0))
    assert image.shape[0] == image.shape[1], "image should be a square"
    image_width = image.shape[0]

    # Create coordinate data and reshape
    coord_range = torch.linspace(-1, 1, image_width)
    coordinates = torch.stack(
        torch.meshgrid(coord_range, coord_range, indexing="ij"), dim=-1
    ).reshape(-1, 2)

    pixel_values = image.reshape(-1, image.shape[2])
    nan_mask = pixel_values.isnan().sum(axis=1) > 0

    # Create divide into train and test data
    test_mask = torch.zeros((image_width, image_width)).to(torch.bool)
    for i in range(image_width):
        for j in range(i % test_grid_skip, image_width, test_grid_skip):
            test_mask[i, j] = True

    test_mask = test_mask.flatten()
    combined_mask_test = test_mask & (~nan_mask)
    combined_mask_train = (~test_mask) & (~nan_mask)

    test_coordinates = coordinates[combined_mask_test]
    test_pixel_values = pixel_values[combined_mask_test]
    train_coordinates = coordinates[combined_mask_train]
    train_pixel_values = pixel_values[combined_mask_train]

    # Sparsify the train data
    train_mask = torch.bernoulli(torch.ones(train_coordinates.shape[0]) * sparsity).to(
        torch.bool
    )
    train_coordinates = train_coordinates[train_mask]
    train_pixel_values = train_pixel_values[train_mask]

    if device:
        train_coordinates = train_coordinates.to(device)
        train_pixel_values = train_pixel_values.to(device)
        test_coordinates = test_coordinates.to(device)
        test_pixel_values = test_pixel_values.to(device)

    full_train_mask = torch.zeros(image_width * image_width).to(torch.bool)
    full_train_mask[torch.where(combined_mask_train)[0][train_mask]] = True

    return (
        CustomTensorDataset(train_coordinates, train_pixel_values),
        CustomTensorDataset(test_coordinates, test_pixel_values),
        full_train_mask.reshape(image_width, image_width),
        combined_mask_test.reshape(image_width, image_width),
    )
