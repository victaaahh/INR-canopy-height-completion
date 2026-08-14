import logging
from collections.abc import Callable
from typing import Literal

import torch
from torch import Tensor, device, nn
from torch.optim import Optimizer
from tqdm import tqdm

from src.data import single_image_dataset, single_image_sparse_dataset

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
        guide_normalization_shift: Tensor,
        guide_normalization_scale: Tensor,
        sparsity: float,
        test_grid_skip: int,
        model: nn.Module,
        target_optimizer: Optimizer,
        guide_optimizer: Optimizer,
        joint_optimizer: Optimizer,
        target_criterion: Callable[[Tensor, Tensor], Tensor],
        guide_criterion: Callable[[Tensor, Tensor], Tensor],
        target_metrics: dict[str, Callable[[Tensor, Tensor], Tensor]],
        guide_metrics: dict[str, Callable[[Tensor, Tensor], Tensor]],
        full_target_metrics: dict[str, Callable[[Tensor, Tensor], Tensor]],
        full_guide_metrics: dict[str, Callable[[Tensor, Tensor], Tensor]],
        device: device,
    ):
        # Target image handling
        if len(target_image.shape) != 3:
            raise ValueError("image should have channel, width and height dims")
        if channel_dim not in [0, 2]:
            raise ValueError("invalid channel_dim")
        if channel_dim == 0:
            self._target_image = target_image.permute(1, 2, 0)
        else:
            self._target_image = target_image
        if self._target_image.shape[0] != self._target_image.shape[1]:
            raise ValueError("target image should be a square")
        self._target_channels = self._target_image.shape[2]
        self._target_width = self._target_image.shape[1]

        non_nan_values = self._target_image[~self._target_image.isnan()]
        if (
            target_normalization_scale == "individual_min_max"
            and target_normalization_shift == "individual_min_max"
        ):
            self._target_shift = non_nan_values.min()
            self._target_scale = max(
                non_nan_values.max() - self._target_shift, 0.1
            )  # avoid division by zero
        elif (
            target_normalization_scale == "individual_std"
            and target_normalization_shift == "individual_std"
        ):
            self._target_shift = non_nan_values.mean()
            self._target_scale = max(non_nan_values.std(), 0.1)
        else:
            assert isinstance(target_normalization_shift, float)
            assert isinstance(target_normalization_scale, float)
            self._target_shift = target_normalization_shift
            self._target_scale = target_normalization_scale

        self._target_image = (
            self._target_image - self._target_shift
        ) / self._target_scale

        self.train_dataset, self.test_dataset, self.train_mask, self.test_mask = (
            single_image_sparse_dataset(
                self._target_image,
                channel_dim=2,
                test_grid_skip=test_grid_skip,
                sparsity=sparsity,
                device=device,
            )
        )

        # Guide image handling
        if channel_dim == 0:
            self._guide_image = guide_image.permute(1, 2, 0)
        else:
            self._guide_image = guide_image
        if self._guide_image.shape[0:2] != self._target_image.shape[0:2]:
            raise ValueError(
                "guide image should have same height and width as target image"
            )
        self._guide_channels = self._guide_image.shape[2]

        if guide_normalization_shift.shape != (
            self._guide_channels,
        ) or guide_normalization_scale.shape != (self._guide_channels,):
            raise ValueError("a scale and shift should be provided for each channel")

        self._guide_shift = guide_normalization_shift.to(device)
        self._guide_scale = guide_normalization_scale.to(device)

        self._guide_image = (
            self._guide_image - self._guide_shift.cpu()
        ) / self._guide_scale.cpu()
        self.guide_dataset = single_image_dataset(
            self._guide_image,
            channel_dim=2,
            device=device,
        )

        self.train_coords, self.train_pixels = self.train_dataset.tensors
        self.test_coords, self.test_pixels = self.test_dataset.tensors
        self.rescaled_train_pixels = (
            self.train_pixels * self._target_scale + self._target_shift
        )
        self.rescaled_test_pixels = (
            self.test_pixels * self._target_scale + self._target_shift
        )
        self.guide_coords, self.guide_pixels = self.guide_dataset.tensors
        self.rescaled_guide_pixels = (
            self.guide_pixels * self._guide_scale + self._guide_shift
        )

        # Other parameters
        self.channel_dim = channel_dim
        self.full_image_metrics_fill_na = full_image_metrics_fill_na
        self.sparsity = sparsity
        self.test_grid_skip = test_grid_skip
        self.model = model
        self.target_optimizer = target_optimizer
        self.guide_optimizer = guide_optimizer
        self.joint_optimizer = joint_optimizer
        self.target_criterion = target_criterion
        self.guide_criterion = guide_criterion
        self.target_metrics = target_metrics
        self.guide_metrics = guide_metrics
        self.full_target_metrics = full_target_metrics
        self.full_guide_metrics = full_guide_metrics
        self.device = device

    @property
    def _rescaled_target_image(self) -> Tensor:
        return self._target_image * self._target_scale + self._target_shift

    @property
    def target_image(self) -> Tensor:
        if self.channel_dim == 0:
            return self._rescaled_target_image.permute(2, 0, 1)
        else:
            return self._rescaled_target_image

    @property
    def _rescaled_guide_image(self) -> Tensor:
        return self._guide_image * self._guide_scale.cpu() + self._guide_shift.cpu()

    @property
    def guide_image(self) -> Tensor:
        if self.channel_dim == 0:
            return self._rescaled_guide_image.permute(2, 0, 1)
        else:
            return self._rescaled_guide_image

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
            * self._target_scale
            + self._target_shift
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
        recon_target, recon_guide = self._rescaled_reconstructed_images
        recon_target = recon_target.detach().cpu()
        recon_guide = recon_guide.detach().cpu()
        if self.channel_dim == 0:
            return recon_target.permute(2, 0, 1), recon_guide.permute(2, 0, 1)
        else:
            return recon_target, recon_guide

    def _fit_target_one_step(
        self, return_full_target_and_guide: bool
    ) -> tuple[float, Tensor, Tensor | None]:
        self.target_optimizer.zero_grad()
        if return_full_target_and_guide:
            full_target_output, guide_output = self.model(self.guide_coords)
            target_output = full_target_output[self.train_mask.flatten()]
            output_to_return = full_target_output
        else:
            target_output, _ = self.model(self.train_coords)
            guide_output = None
            output_to_return = target_output
        loss = self.target_criterion(target_output, self.train_pixels)
        _ = loss.backward()
        self.target_optimizer.step()
        return loss.item(), output_to_return, guide_output

    def _fit_guide_one_step(self) -> tuple[float, Tensor, Tensor]:
        self.guide_optimizer.zero_grad()
        full_target_output, guide_output = self.model(self.guide_coords)
        loss = self.guide_criterion(guide_output, self.guide_pixels)
        _ = loss.backward()
        self.guide_optimizer.step()
        return loss.item(), full_target_output, guide_output

    def _fit_jointly_one_step(
        self,
        target_loss_weight: float,
        guide_loss_weight: float,
    ) -> tuple[float, float, float, Tensor, Tensor]:
        self.joint_optimizer.zero_grad()
        full_target_output, guide_output = self.model(self.guide_coords)
        target_output = full_target_output[self.train_mask.flatten()]
        target_loss = self.target_criterion(target_output, self.train_pixels)
        guide_loss = self.guide_criterion(guide_output, self.guide_pixels)
        total_loss = target_loss_weight * target_loss + guide_loss_weight * guide_loss
        _ = total_loss.backward()
        self.joint_optimizer.step()
        return (
            total_loss.item(),
            target_loss.item(),
            guide_loss.item(),
            full_target_output,
            guide_output,
        )

    def fit(
        self,
        what: Literal["target", "guide", "jointly"],
        epochs: int,
        return_target_train_metrics: bool,
        return_target_test_metrics: bool,
        return_guide_metrics: bool,
        full_image_metrics_interval: int,
        target_loss_weight: float | None,
        guide_loss_weight: float | None,
    ) -> tuple[
        Tensor,
        Tensor | None,
        Tensor | None,
        dict[str, Tensor] | None,
        dict[str, Tensor] | None,
        dict[str, Tensor] | None,
        dict[str, Tensor] | None,
        dict[str, Tensor] | None,
    ]:
        if what == "jointly":
            all_target_loss = torch.zeros(epochs)
            all_guide_loss = torch.zeros(epochs)
            if target_loss_weight is None or guide_loss_weight is None:
                raise ValueError(
                    "You have to provide loss weights when fitting jointly"
                )
        else:
            all_target_loss = None
            all_guide_loss = None

        all_train_metrics = None
        all_test_metrics = None
        all_full_target_metrics = None
        all_guide_metrics = None
        all_full_guide_metrics = None

        all_loss = torch.zeros(epochs)

        if return_target_train_metrics:
            all_train_metrics = {
                name: torch.zeros(epochs) for name in self.target_metrics
            }

        if return_target_test_metrics:
            all_test_metrics = {
                name: torch.zeros(epochs) for name in self.target_metrics
            }
            full_image_metric_computations = epochs // full_image_metrics_interval + (
                epochs % full_image_metrics_interval > 0
            )
            all_full_target_metrics = {
                name: torch.zeros(full_image_metric_computations)
                for name in self.full_target_metrics
            }

        if return_guide_metrics:
            all_guide_metrics = {
                name: torch.zeros(epochs) for name in self.guide_metrics
            }
            full_image_metric_computations = epochs // full_image_metrics_interval + (
                epochs % full_image_metrics_interval > 0
            )
            all_full_guide_metrics = {
                name: torch.zeros(full_image_metric_computations)
                for name in self.full_guide_metrics
            }

        for i in tqdm(range(epochs)):
            _ = self.model.train()
            if what == "target":
                if return_guide_metrics:
                    loss, full_target_output, guide_output = self._fit_target_one_step(
                        return_full_target_and_guide=True
                    )
                    target_output = full_target_output[self.train_mask.flatten()]
                else:
                    loss, target_output, _ = self._fit_target_one_step(False)
                    full_target_output, guide_output = None, None
            elif what == "guide":
                loss, full_target_output, guide_output = self._fit_guide_one_step()
                target_output = full_target_output[self.train_mask.flatten()]
            elif what == "jointly":
                assert target_loss_weight is not None and guide_loss_weight is not None
                assert all_target_loss is not None and all_guide_loss is not None
                loss, target_loss, guide_loss, full_target_output, guide_output = (
                    self._fit_jointly_one_step(target_loss_weight, guide_loss_weight)
                )
                target_output = full_target_output[self.train_mask.flatten()]
                all_target_loss[i] = target_loss
                all_guide_loss[i] = guide_loss
            else:
                raise ValueError("Choose between target, guide or jointly")

            all_loss[i] = loss

            if return_target_train_metrics:
                assert all_train_metrics is not None
                rescaled_target_output = (
                    target_output * self._target_scale + self._target_shift
                )
                for name, metric in self.target_metrics.items():
                    all_train_metrics[name][i] = metric(
                        rescaled_target_output, self.rescaled_train_pixels
                    ).item()

            if return_target_test_metrics or return_guide_metrics:
                return_full_image_metrics = (
                    i + 1
                ) % full_image_metrics_interval == 0 or i == epochs - 1

                (
                    _,
                    test_metrics,
                    guide_metrics,
                    full_target_metrics,
                    full_guide_metrics,
                ) = self.evaluate(
                    False,
                    return_target_test_metrics,
                    return_guide_metrics,
                    return_full_image_metrics,
                    full_target_output,
                    guide_output,
                )

                if return_target_test_metrics:
                    assert all_test_metrics is not None
                    assert all_full_target_metrics is not None
                    assert test_metrics is not None
                    for name in self.target_metrics:
                        all_test_metrics[name][i] = test_metrics[name]

                    if return_full_image_metrics:
                        assert full_target_metrics is not None
                        for name in self.full_target_metrics:
                            j = i // full_image_metrics_interval
                            all_full_target_metrics[name][j] = full_target_metrics[name]

                if return_guide_metrics:
                    assert all_guide_metrics is not None
                    assert all_full_guide_metrics is not None
                    assert guide_metrics is not None
                    for name in self.guide_metrics:
                        all_guide_metrics[name][i] = guide_metrics[name]

                    if return_full_image_metrics:
                        assert full_guide_metrics is not None
                        for name in self.full_target_metrics:
                            j = i // full_image_metrics_interval
                            all_full_guide_metrics[name][j] = full_guide_metrics[name]

        return (
            all_loss,
            all_target_loss,
            all_guide_loss,
            all_train_metrics,
            all_test_metrics,
            all_full_target_metrics,
            all_guide_metrics,
            all_full_guide_metrics,
        )

    def evaluate(
        self,
        target_train: bool,
        target_test: bool,
        guide: bool,
        return_full_image_metrics: bool,
        full_target_output: Tensor | None,
        guide_output: Tensor | None,
    ) -> tuple[
        dict[str, float] | None,
        dict[str, float] | None,
        dict[str, float] | None,
        dict[str, float] | None,
        dict[str, float] | None,
    ]:
        target_train_metrics = None
        target_test_metrics = None
        guide_metrics = None
        full_target_metrics = None
        full_guide_metrics = None

        _ = self.model.eval()

        if return_full_image_metrics and not (target_test or guide):
            raise ValueError(
                "Full image metrics are considered a subset of target_test and guide"
            )

        with torch.no_grad():
            if full_target_output is not None and guide_output is not None:
                rescaled_guide_output = (
                    guide_output * self._guide_scale + self._guide_shift
                )
                target_test_output = full_target_output[self.test_mask.flatten()]
                target_train_output = full_target_output[self.train_mask.flatten()]
                rescaled_target_train_output = (
                    target_train_output * self._target_scale + self._target_shift
                )
                rescaled_target_test_output = (
                    target_test_output * self._target_scale + self._target_shift
                )
            elif target_train and not target_test and not guide:
                target_train_output, _ = self.model(self.train_coords)
                rescaled_target_train_output = (
                    target_train_output * self._target_scale + self._target_shift
                )
                rescaled_target_test_output = None
                rescaled_guide_output = None
            elif (
                target_test
                and not target_train
                and not guide
                and not return_full_image_metrics
            ):
                target_test_output, _ = self.model(self.test_coords)
                rescaled_target_test_output = (
                    target_test_output * self._target_scale + self._target_shift
                )
                rescaled_target_train_output = None
                rescaled_guide_output = None
            else:
                full_target_output, guide_output = self.model(self.guide_coords)
                assert full_target_output is not None
                assert guide_output is not None
                rescaled_guide_output = (
                    guide_output * self._guide_scale + self._guide_shift
                )
                target_test_output = full_target_output[self.test_mask.flatten()]
                target_train_output = full_target_output[self.train_mask.flatten()]
                rescaled_target_train_output = (
                    target_train_output * self._target_scale + self._target_shift
                )
                rescaled_target_test_output = (
                    target_test_output * self._target_scale + self._target_shift
                )

            if target_train:
                assert rescaled_target_train_output is not None
                target_train_metrics = {
                    name: metric(
                        rescaled_target_train_output, self.rescaled_train_pixels
                    ).item()
                    for name, metric in self.target_metrics.items()
                }

            if target_test:
                assert rescaled_target_test_output is not None
                target_test_metrics = {
                    name: metric(
                        rescaled_target_test_output, self.rescaled_test_pixels
                    ).item()
                    for name, metric in self.target_metrics.items()
                }

            if guide:
                assert rescaled_guide_output is not None
                guide_metrics = {
                    name: metric(
                        rescaled_guide_output, self.rescaled_guide_pixels
                    ).item()
                    for name, metric in self.guide_metrics.items()
                }

            if return_full_image_metrics:
                assert rescaled_guide_output is not None
                assert full_target_output is not None
                rescaled_full_target_output = (
                    full_target_output * self._target_scale + self._target_shift
                )
                target_image = self._rescaled_target_image.clone().to(self.device)
                nan_inds = torch.where(target_image.isnan())
                target_image[nan_inds] = self.full_image_metrics_fill_na

                guide_image = self._rescaled_guide_image.clone().to(self.device)

                reconstructed_target = rescaled_full_target_output.reshape(
                    self._target_width, self._target_width, self._target_channels
                )
                reconstructed_guide = rescaled_guide_output.reshape(
                    self._target_width, self._target_width, self._guide_channels
                )
                reconstructed_target = reconstructed_target.clone().to(self.device)
                reconstructed_target[nan_inds] = self.full_image_metrics_fill_na
                reconstructed_guide = reconstructed_guide.clone().to(self.device)

                if target_test:
                    full_target_metrics = {
                        name: metric(reconstructed_target, target_image).item()
                        for name, metric in self.full_target_metrics.items()
                    }

                if guide:
                    full_guide_metrics = {
                        name: metric(reconstructed_guide, guide_image).item()
                        for name, metric in self.full_guide_metrics.items()
                    }

        return (
            target_train_metrics,
            target_test_metrics,
            guide_metrics,
            full_target_metrics,
            full_guide_metrics,
        )
