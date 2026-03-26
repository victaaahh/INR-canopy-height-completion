from typing import Literal, Optional, Union

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from src.data import single_image_dataset
from src.utils import Tracker


class SparseImageINR:
    def __init__(
        self,
        target_image: torch.Tensor,
        normalize_image: bool,
        channel_dim: int,
        sparsity: float,
        test_grid_skip: int,
        model: Module,
        optimizer: Optimizer,
        criterion,
        batch_size: Union[int, Literal["full"]],
        shuffle: bool,
        device: torch.device,
        tracker: Optional[Tracker] = None,
    ):
        assert len(target_image.shape) == 3, (
            "image should have channel, width and height dims"
        )
        assert channel_dim in [0, 2], "invalid channel_dim"
        if channel_dim == 0:
            self._target_image = target_image.permute(1, 2, 0)
        else:
            self._target_image = target_image
        assert self._target_image.shape[0] == self._target_image.shape[1], (
            "image should be a square"
        )
        self._target_channels = self._target_image.shape[2]
        self._target_width = self._target_image.shape[1]

        if normalize_image:
            self._target_image /= self._target_image.max()

        self.channel_dim = channel_dim
        self.sparsity = sparsity
        self.test_grid_skip = test_grid_skip
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.tracker = tracker

        self.train_pixels, self.test_pixels, self.train_mask, self.test_mask = (
            single_image_dataset(
                self._target_image,
                channel_dim=2,
                test_grid_skip=test_grid_skip,
                sparsity=sparsity,
                device=device,
            )
        )

        if batch_size == "full":
            self.train_dataloader = [self.train_pixels.tensors]
        else:
            self.train_dataloader = DataLoader(
                self.train_pixels,
                batch_size,
                shuffle,
                collate_fn=lambda x: (
                    x
                ),  # CustomTensorDataset requires identity collate_fn
            )
        self.test_dataloader = [self.test_pixels.tensors]

        self._sparse_target = None
        self._target_eval_grid = None
        self._reconstructed_target = None

    @property
    def target_image(self) -> torch.Tensor:
        if self.channel_dim == 0:
            return self._target_image.permute(2, 0, 1)
        else:
            return self._target_image

    @property
    def reconstructed_target(self) -> torch.Tensor:
        if self._reconstructed_target is None:
            coord_range = torch.linspace(-1, 1, self._target_width)
            coordinates = (
                torch.stack(
                    torch.meshgrid(coord_range, coord_range, indexing="ij"), dim=-1
                )
                .reshape(-1, 2)
                .to(self.device)
            )
            reconstructed_target = self.model(coordinates).detach().cpu()

            self._reconstructed_target = reconstructed_target.reshape(
                self._target_width, self._target_width, self._target_channels
            )

        if self.channel_dim == 0:
            return self._reconstructed_target.permute(2, 0, 1)
        else:
            return self._reconstructed_target

    def fit(self, epochs: int) -> float:
        self._reconstructed_target = None
        train_loss = 0.0  # initialize variable to avoid "possibly unbound" warning
        for _ in tqdm(range(epochs)):
            self.model.train()
            train_loss = 0.0
            for input_, target in self.train_dataloader:
                self.optimizer.zero_grad()
                output = self.model(input_)
                loss = self.criterion(output, target)
                train_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            train_loss /= len(self.train_dataloader)

            if self.tracker is not None:
                test_loss = self.evaluate()
                self.tracker.log_metrics(
                    {"train_loss": train_loss, "test_loss": test_loss}
                )

        return train_loss

    def evaluate(self) -> float:
        self.model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for input_, target in self.test_dataloader:
                output = self.model(input_)
                loss = self.criterion(output, target)
                test_loss += loss.item()
            test_loss /= len(self.test_dataloader)
        return test_loss
