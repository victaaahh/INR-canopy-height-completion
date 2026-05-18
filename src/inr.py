import logging
from typing import Callable, Literal

import torch
from torch import Tensor, device
from torch.optim import Optimizer
from tqdm import tqdm

from src.data import single_image_dataset, single_image_sparse_dataset
from src.models.base_inr import INR

log = logging.getLogger(__name__)


class SparseImageINR:
    def __init__(
        self,
        target_image: Tensor,
        guide_image: Tensor,
        channel_dim: int,
        full_image_metrics_fill_na: float,
        target_normalization_shift: float
        | Literal["individual_min_max", "individual_std"],
        target_normalization_scale: float
        | Literal["individual_min_max", "individual_std"],
        guide_normalization_shift: float,
        guide_normalization_scale: float,
        sparsity: float,
        test_grid_skip: int,
        model: INR,
        target_optimizer: Optimizer | None,
        guide_optimizer: Optimizer | None,
        joint_optimizer: Optimizer | None,
        criterion: Callable[[Tensor, Tensor], Tensor],
        metrics: dict[str, Callable[[Tensor, Tensor], Tensor]],
        full_image_metrics: dict[str, Callable[[Tensor, Tensor], Tensor]],
        device: device,
    ):
        assert len(target_image.shape) == 3, (
            "image should have channel, width and height dims"
        )
        assert channel_dim in [0, 2], "invalid channel_dim"
        if channel_dim == 0:
            self._target_image = target_image.permute(1, 2, 0)
            self._guide_image = guide_image.permute(1, 2, 0)
        else:
            self._target_image = target_image
            self._guide_image = guide_image
        assert self._target_image.shape[0] == self._target_image.shape[1], (
            "target image should be a square"
        )
        assert self._guide_image.shape[0:2] == self._target_image.shape[0:2], (
            "guide image should have same height and width as target image"
        )
        self._target_channels = self._target_image.shape[2]
        self._target_width = self._target_image.shape[1]
        self._guide_channels = self._guide_image.shape[2]

        non_nan_values = self._target_image[~target_image.isnan()]
        if (
            target_normalization_scale == "individual_min_max"
            and target_normalization_shift == "individual_min_max"
        ):
            self._shift = non_nan_values.min()
            self._scale = max(
                non_nan_values.max() - self._shift, 0.1
            )  # avoid division by zero
        elif (
            target_normalization_scale == "individual_std"
            and target_normalization_shift == "individual_std"
        ):
            self._shift = non_nan_values.mean()
            self._scale = max(non_nan_values.std(), 0.1)
        else:
            assert isinstance(target_normalization_shift, float)
            assert isinstance(target_normalization_scale, float)
            self._shift = target_normalization_shift
            self._scale = target_normalization_scale

        self._guide_shift = guide_normalization_shift
        self._guide_scale = guide_normalization_scale

        self._target_image = (self._target_image - self._shift) / self._scale
        self._guide_image = (self._guide_image - self._guide_shift) / self._guide_scale

        self.channel_dim = channel_dim
        self.full_image_metrics_fill_na = full_image_metrics_fill_na
        self.sparsity = sparsity
        self.test_grid_skip = test_grid_skip
        self.model = model
        self.target_optimizer = target_optimizer
        self.guide_optimizer = guide_optimizer
        self.joint_optimizer = joint_optimizer
        self.criterion = criterion
        self.metrics = metrics
        self.full_image_metrics = full_image_metrics
        self.device = device

        self.train_pixels, self.test_pixels, self.train_mask, self.test_mask = (
            single_image_sparse_dataset(
                self._target_image,
                channel_dim=2,
                test_grid_skip=test_grid_skip,
                sparsity=sparsity,
                device=device,
            )
        )

        self.guide_pixels = single_image_dataset(
            self._guide_image,
            channel_dim=2,
            device=device,
        )

    @property
    def _rescaled_target_image(self) -> Tensor:
        return self._target_image * self._scale + self._shift

    @property
    def target_image(self) -> Tensor:
        if self.channel_dim == 0:
            return self._rescaled_target_image.permute(2, 0, 1)
        else:
            return self._rescaled_target_image

    @property
    def guide_image(self) -> Tensor:
        rescaled_guide_image = self._guide_image * self._guide_scale + self._guide_shift
        if self.channel_dim == 0:
            return rescaled_guide_image.permute(2, 0, 1)
        else:
            return rescaled_guide_image

    @property
    def _rescaled_reconstructed_images(self) -> tuple[Tensor, Tensor]:
        coord_range = torch.linspace(-1, 1, self._target_width)
        coordinates = (
            torch.stack(torch.meshgrid(coord_range, coord_range, indexing="ij"), dim=-1)
            .reshape(-1, 2)
            .to(self.device)
        )
        reconstructed_target, reconstructed_guide = self.model(coordinates)

        reconstructed_target = (
            reconstructed_target.reshape(
                self._target_width, self._target_width, self._target_channels
            )
            * self._scale
            + self._shift
        )
        reconstructed_guide = (
            reconstructed_guide.reshape(
                self._target_width, self._target_width, self._guide_channels
            )
            * self._guide_scale
            + self._guide_shift
        )
        return reconstructed_target, reconstructed_guide

    @property
    def reconstructed_images(self) -> tuple[Tensor, Tensor]:
        if self.channel_dim == 0:
            recon_target, recon_guide = self._rescaled_reconstructed_images
            return recon_target.permute(2, 0, 1), recon_guide.permute(2, 0, 1)
        else:
            return self._rescaled_reconstructed_images

    def fit_target_one_step(self, coords: Tensor, pixels: Tensor):
        assert self.target_optimizer is not None
        self.target_optimizer.zero_grad()
        output, _ = self.model(coords)
        loss = self.criterion(output, pixels)
        _ = loss.backward()
        self.target_optimizer.step()
        return output, loss

    def fit_guide_one_step(
        self,
        target_coords: Tensor,
        target_pixels: Tensor,
        guide_coords: Tensor,
        guide_pixels: Tensor,
    ):
        assert self.guide_optimizer is not None
        self.guide_optimizer.zero_grad()
        target_output, _ = self.model(target_coords)
        _, guide_output = self.model(guide_coords)
        guide_loss = self.criterion(guide_output, guide_pixels)
        target_loss = self.criterion(target_output, target_pixels)
        _ = guide_loss.backward()
        self.guide_optimizer.step()
        return target_output, target_loss

    def fit_jointly_one_step(
        self,
        target_coords: Tensor,
        target_pixels: Tensor,
        guide_coords: Tensor,
        guide_pixels: Tensor,
        target_loss_weight: float,
        guide_loss_weight: float,
    ):
        assert self.joint_optimizer is not None
        self.joint_optimizer.zero_grad()
        target_output, _ = self.model(target_coords)
        _, guide_output = self.model(guide_coords)
        target_loss = self.criterion(target_output, target_pixels)
        guide_loss = self.criterion(guide_output, guide_pixels)
        total_loss = target_loss_weight * target_loss + guide_loss_weight * guide_loss
        _ = total_loss.backward()
        self.joint_optimizer.step()
        return target_output, target_loss

    def fit(
        self,
        epochs: int,
        return_test_metrics: bool,
        full_image_metrics_interval: int,
        what: Literal["target", "guide", "jointly"],
        target_loss_weight: float,
        guide_loss_weight: float,
    ) -> tuple[
        Tensor,
        dict[str, Tensor],
        Tensor | None,
        dict[str, Tensor] | None,
        dict[str, Tensor] | None,
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

        coords, pixels = self.train_pixels.tensors
        guide_coords, guide_pixels = self.guide_pixels.tensors
        for i in tqdm(range(epochs)):
            _ = self.model.train()
            if what == "target":
                output, loss = self.fit_target_one_step(coords, pixels)
            elif what == "guide":
                output, loss = self.fit_guide_one_step(
                    coords, pixels, guide_coords, guide_pixels
                )
            elif what == "jointly":
                output, loss = self.fit_jointly_one_step(
                    coords,
                    pixels,
                    guide_coords,
                    guide_pixels,
                    target_loss_weight,
                    guide_loss_weight,
                )
            all_train_loss[i] = loss.item()
            rescaled_output = output * self._scale + self._shift
            rescaled_pixels = pixels * self._scale + self._shift
            for name, metric in self.metrics.items():
                all_train_metrics[name][i] = metric(
                    rescaled_output, rescaled_pixels
                ).item()

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
        self, return_full_image_metrics: bool
    ) -> tuple[float, dict[str, float], dict[str, float] | None]:
        _ = self.model.eval()
        epoch_loss = 0.0
        metrics_values = {name: 0.0 for name, _ in self.metrics.items()}
        coords, pixels = self.test_pixels.tensors
        with torch.no_grad():
            output, _ = self.model(coords)
            loss = self.criterion(output, pixels)
            epoch_loss = loss.item()
            rescaled_output = output * self._scale + self._shift
            rescaled_pixels = pixels * self._scale + self._shift
            for name, metric in self.metrics.items():
                metrics_values[name] = metric(rescaled_output, rescaled_pixels).item()

            if return_full_image_metrics:
                target_image = self._rescaled_target_image.clone().to(self.device)
                nan_inds = torch.where(target_image.isnan())
                target_image[nan_inds] = self.full_image_metrics_fill_na
                reconstructed = (
                    self._rescaled_reconstructed_images[0].clone().to(self.device)
                )
                reconstructed[nan_inds] = self.full_image_metrics_fill_na
                full_image_metrics = {
                    name: float(metric(reconstructed, target_image))
                    for name, metric in self.full_image_metrics.items()
                }
            else:
                full_image_metrics = None
        return epoch_loss, metrics_values, full_image_metrics
