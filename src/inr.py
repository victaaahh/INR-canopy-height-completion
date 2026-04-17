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
        coord_range = torch.linspace(-1, 1, self._target_width)
        coordinates = (
            torch.stack(torch.meshgrid(coord_range, coord_range, indexing="ij"), dim=-1)
            .reshape(-1, 2)
            .to(self.device)
        )
        reconstructed_target = self.model(coordinates).detach().cpu()

        reconstructed_target = reconstructed_target.reshape(
            self._target_width, self._target_width, self._target_channels
        )
        return reconstructed_target * self._scale + self._shift

    @property
    def reconstructed_target(self) -> torch.Tensor:
        if self.channel_dim == 0:
            return self._rescaled_reconstructed_target.permute(2, 0, 1)
        else:
            return self._rescaled_reconstructed_target

    def fit(
        self,
        epochs: int,
        return_test_metrics: bool,
        full_image_metrics_interval: int,
    ) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
        Optional[torch.Tensor],
        Optional[dict[str, torch.Tensor]],
        Optional[dict[str, torch.Tensor]],
    ]:
        all_train_loss = torch.zeros(epochs)  # initialize vars to avoid warning
        all_train_metrics = {name: torch.zeros(epochs) for name in self.metrics}
        if return_test_metrics:
            all_test_loss = torch.zeros(epochs)
            all_test_metrics = {name: torch.zeros(epochs) for name in self.metrics}
            full_image_metric_computations = epochs // full_image_metrics_interval + (
                epochs % full_image_metrics_interval > 0
            )
            all_full_image_metrics = {
                name: torch.zeros(full_image_metric_computations)
                for name in self.full_image_metrics
            }
        else:
            all_test_loss = None
            all_test_metrics = None
            all_full_image_metrics = None

        for i in tqdm(range(epochs)):
            self.model.train()
            for coords, pixels in self.train_dataloader:
                self.optimizer.zero_grad()
                output = self.model(coords)
                loss = self.criterion(output, pixels)
                all_train_loss[i] += loss.item()
                loss.backward()
                self.optimizer.step()

                rescaled_output = output * self._scale + self._shift
                rescaled_pixels = pixels * self._scale + self._shift
                for name, metric in self.metrics.items():
                    all_train_metrics[name][i] += metric(
                        rescaled_output, rescaled_pixels
                    ).item()

            all_train_loss[i] /= len(self.train_dataloader)
            for name in self.metrics:
                all_train_metrics[name][i] /= len(self.test_dataloader)

            if (
                all_test_loss is not None
                and all_test_metrics is not None
                and all_full_image_metrics is not None
            ):
                return_full_image_metrics = (
                    i + 1
                ) % full_image_metrics_interval == 0 or i == epochs - 1
                test_loss, test_metrics, full_image_metrics = self.evaluate(
                    return_full_image_metrics=return_full_image_metrics
                )
                if full_image_metrics is not None:
                    for name in self.full_image_metrics:
                        j = i // full_image_metrics_interval
                        all_full_image_metrics[name][j] = full_image_metrics[name]
                all_test_loss[i] = test_loss
                for name in self.metrics:
                    all_test_metrics[name][i] = test_metrics[name]

        return (
            all_train_loss,
            all_train_metrics,
            all_test_loss,
            all_test_metrics,
            all_full_image_metrics,
        )

    def evaluate(
        self, return_full_image_metrics
    ) -> tuple[float, dict[str, float], Optional[dict[str, float]]]:
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

            if return_full_image_metrics:
                reconstructed = self._rescaled_reconstructed_target.to(self.device)
                target_image = self._rescaled_target_image.to(self.device)
                full_image_metrics = {
                    name: float(metric(reconstructed, target_image))
                    for name, metric in self.full_image_metrics.items()
                }
            else:
                full_image_metrics = None
        return epoch_loss, metrics_values, full_image_metrics
