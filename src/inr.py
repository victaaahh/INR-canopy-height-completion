import logging
from typing import Callable, Literal, Optional, Union

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from src.data import single_image_dataset
from src.utils import Tracker

log = logging.getLogger(__name__)


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
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        metrics: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
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
            self._shift = self._target_image.min()
            self._scale = self._target_image.max() - self._shift
            self._target_image = (self._target_image - self._shift) / self._scale

        self.normalize_image = normalize_image
        self.channel_dim = channel_dim
        self.sparsity = sparsity
        self.test_grid_skip = test_grid_skip
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics = metrics
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
            log.warning(
                "Training metrics are probably computed slightly wrong, \
                 since it is assumed that all batches have same size, but \
                 the last one might not"
            )
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
        to_return = self._target_image

        if self.normalize_image:
            to_return = to_return * self._scale + self._shift

        if self.channel_dim == 0:
            to_return = to_return.permute(2, 0, 1)

        return to_return

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

        to_return = self._reconstructed_target

        if self.normalize_image:
            to_return = to_return * self._scale + self._shift

        if self.channel_dim == 0:
            to_return = to_return.permute(2, 0, 1)

        return to_return

    def fit(self, epochs: int) -> tuple[float, dict[str, float]]:
        self._reconstructed_target = None
        epoch_loss = 0.0  # initialize variables to avoid "possibly unbound" warning
        metrics_values = {name: 0.0 for name, _ in self.metrics.items()}
        for _ in tqdm(range(epochs)):
            self.model.train()
            epoch_loss = 0.0
            metrics_values = {name: 0.0 for name, _ in self.metrics.items()}
            for input_, target in self.train_dataloader:
                self.optimizer.zero_grad()
                output = self.model(input_)
                loss = self.criterion(output, target)
                epoch_loss += loss.item()
                for name, metric in self.metrics.items():
                    metrics_values[name] += metric(output, target).item()
                loss.backward()
                self.optimizer.step()
            epoch_loss /= len(self.train_dataloader)
            for name in self.metrics:
                metrics_values[name] /= len(self.test_dataloader)

            if self.tracker is not None:
                test_loss, eval_metrics = self.evaluate()
                self.tracker.log_metrics(
                    {
                        "train_loss": epoch_loss,
                        **{f"train_{name}": v for name, v in metrics_values.items()},
                        "test_loss": test_loss,
                        **{f"test_{name}": v for name, v in eval_metrics.items()},
                    }
                )

        return epoch_loss, metrics_values

    def evaluate(self) -> tuple[float, dict[str, float]]:
        self.model.eval()
        epoch_loss = 0.0
        metrics_values = {name: 0.0 for name, _ in self.metrics.items()}
        with torch.no_grad():
            for input_, target in self.test_dataloader:
                output = self.model(input_)
                loss = self.criterion(output, target)
                epoch_loss += loss.item()
                for name, metric in self.metrics.items():
                    metrics_values[name] += metric(output, target).item()
            epoch_loss /= len(self.test_dataloader)
            for name in self.metrics:
                metrics_values[name] /= len(self.test_dataloader)
        return epoch_loss, metrics_values
