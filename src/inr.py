import logging
from typing import Callable, Literal, Optional, Union

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from src.data import single_image_dataset
from src.utils.metrics import full_window_SSIM

log = logging.getLogger(__name__)


class SparseImageINR:
    def __init__(
        self,
        target_image: torch.Tensor,
        channel_dim: int,
        sparsity: float,
        test_grid_skip: int,
        model: Module,
        optimizer: Optimizer,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        metrics: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
        full_image_metrics: dict[
            str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ],
        batch_size: Union[int, Literal["full"]],
        shuffle: bool,
        device: torch.device,
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

        self._shift = self._target_image.min()
        self._scale = max(
            self._target_image.max() - self._shift, 0.1
        )  # avoid division by zero
        self._target_image = (self._target_image - self._shift) / self._scale

        self.channel_dim = channel_dim
        self.sparsity = sparsity
        self.test_grid_skip = test_grid_skip
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics = metrics
        self.full_image_metrics = full_image_metrics
        self.device = device

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
            assert not any(
                isinstance(metric_class, full_window_SSIM)
                for metric_class in self.metrics.values()
            ), 'full_window_SSIM is incompatible with batch_size different than "full"'
            log.warning(
                'Training metrics are probably computed slightly wrong, \
                 since it is assumed that all batches have same size, but \
                 the last one might not when batch_size != "full"'
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
    def _rescaled_target_image(self) -> torch.Tensor:
        return self._target_image * self._scale + self._shift

    @property
    def target_image(self) -> torch.Tensor:
        if self.channel_dim == 0:
            return self._rescaled_target_image.permute(2, 0, 1)
        else:
            return self._rescaled_target_image

    @property
    def _rescaled_reconstructed_target(self) -> torch.Tensor:
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
        return self._reconstructed_target * self._scale + self._shift

    @property
    def reconstructed_target(self) -> torch.Tensor:
        if self.channel_dim == 0:
            return self._rescaled_reconstructed_target.permute(2, 0, 1)
        else:
            return self._rescaled_reconstructed_target

    def fit(
        self, epochs: int, return_test_metrics: bool = False
    ) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
        Optional[torch.Tensor],
        Optional[dict[str, torch.Tensor]],
        dict[str, torch.Tensor],
    ]:
        self._reconstructed_target = None
        train_losses = torch.zeros(epochs)  # initialize vars to avoid warning
        train_metrics = {name: torch.zeros(epochs) for name in self.metrics}
        full_image_metrics = {
            name: torch.zeros(epochs) for name in self.full_image_metrics
        }
        if return_test_metrics:
            test_losses = torch.zeros(epochs)
            test_metrics = {
                name: torch.zeros(epochs) for name, _ in self.metrics.items()
            }
        else:
            test_losses = None
            test_metrics = None

        for i in tqdm(range(epochs)):
            self.model.train()
            for coords, pixels in self.train_dataloader:
                self.optimizer.zero_grad()
                output = self.model(coords)
                loss = self.criterion(output, pixels)
                train_losses[i] += loss.item()
                loss.backward()
                self.optimizer.step()

                rescaled_output = output * self._scale + self._shift
                rescaled_pixels = pixels * self._scale + self._shift
                for name, metric in self.metrics.items():
                    train_metrics[name][i] += metric(
                        rescaled_output, rescaled_pixels
                    ).item()

            train_losses[i] /= len(self.train_dataloader)
            for name in self.metrics:
                train_metrics[name][i] /= len(self.test_dataloader)

            for name, metric in self.full_image_metrics.items():
                full_image_metrics[name][i] = metric(
                    self._rescaled_reconstructed_target, self._rescaled_target_image
                )

            if test_losses is not None and test_metrics is not None:
                test_loss, eval_metrics, _ = self.evaluate()
                test_losses[i] = test_loss
                for name in self.metrics:
                    test_metrics[name][i] = eval_metrics[name]

        return (
            train_losses,
            train_metrics,
            test_losses,
            test_metrics,
            full_image_metrics,
        )

    def evaluate(self) -> tuple[float, dict[str, float], dict[str, float]]:
        self.model.eval()
        epoch_loss = 0.0
        metrics_values = {name: 0.0 for name, _ in self.metrics.items()}
        with torch.no_grad():
            for coords, pixels in self.test_dataloader:
                output = self.model(coords)
                loss = self.criterion(output, pixels)
                epoch_loss += loss.item()
                rescaled_output = output * self._scale + self._shift
                rescaled_pixels = pixels * self._scale + self._shift
                for name, metric in self.metrics.items():
                    metrics_values[name] += metric(
                        rescaled_output, rescaled_pixels
                    ).item()
            epoch_loss /= len(self.test_dataloader)
            for name in self.metrics:
                metrics_values[name] /= len(self.test_dataloader)

            full_image_metrics = {
                name: float(
                    metric(
                        self._rescaled_reconstructed_target, self._rescaled_target_image
                    )
                )
                for name, metric in self.full_image_metrics.items()
            }
        return epoch_loss, metrics_values, full_image_metrics
