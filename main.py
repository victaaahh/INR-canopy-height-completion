import logging
import os
from collections.abc import Callable

import hydra
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig
from muon import SingleDeviceMuonWithAuxAdam
from omegaconf.dictconfig import DictConfig
from torch import Tensor
from torchvision.transforms.v2 import CenterCrop

from src import SparseImageINR
from src.data import OpenCanopy

log = logging.getLogger(__name__)


def create_optimizer(cfg: DictConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    if "_target_" in cfg:
        optimizer = hydra.utils.instantiate(cfg)(model.parameters())
    elif "muon" in cfg:
        assert type(model.middle_layers) == torch.nn.Module
        assert type(model.first_layer) == torch.nn.Module
        assert type(model.last_layer) == torch.nn.Module
        assert type(model.guide_head) == torch.nn.Module
        middle_weights = [p for p in model.middle_layers.parameters() if p.ndim >= 2]
        middle_biases = [p for p in model.middle_layers.parameters() if p.ndim < 2]
        other_params = [
            *model.first_layer.parameters(),
            *model.last_layer.parameters(),
            *model.guide_head.parameters(),
        ]
        optimizer = SingleDeviceMuonWithAuxAdam(
            [
                dict(params=middle_weights, use_muon=True, **cfg.muon),
                dict(
                    params=middle_biases + other_params,
                    use_muon=False,
                    **cfg.adam,
                ),
            ]
        )
    else:
        raise ValueError("Unsupported optimizer config")

    return optimizer


def generate_results_dict(
    stage: str,
    cfg: DictConfig,
    target_metrics_dict: dict[str, Callable[[Tensor], Tensor]],
    full_target_metrics_dict: dict[str, Callable[[Tensor], Tensor]],
    guide_metrics_dict: dict[str, Callable[[Tensor], Tensor]],
    full_guide_metrics_dict: dict[str, Callable[[Tensor], Tensor]],
) -> dict[str, list[float | Tensor]]:
    results: dict[str, list[float | Tensor]] = {
        f"{stage}_train_loss": [],
        f"{stage}_all_train_loss": [],
        **{f"{stage}_target_train_{name}": [] for name in target_metrics_dict},
        **{f"{stage}_target_test_{name}": [] for name in target_metrics_dict},
        **{f"{stage}_target_full_{name}": [] for name in full_target_metrics_dict},
        **{f"{stage}_guide_{name}": [] for name in guide_metrics_dict},
        **{f"{stage}_guide_full_{name}": [] for name in full_guide_metrics_dict},
    }
    if cfg.save_running_target_metrics:
        results = results | {
            **{f"{stage}_all_target_train_{name}": [] for name in target_metrics_dict},
            **{f"{stage}_all_target_test_{name}": [] for name in target_metrics_dict},
            **{
                f"{stage}_all_target_full_{name}": []
                for name in full_target_metrics_dict
            },
        }
    if cfg.save_running_guide_metrics:
        results = results | {
            **{f"{stage}_all_guide_{name}": [] for name in guide_metrics_dict},
            **{
                f"{stage}_all_guide_full_{name}": [] for name in full_guide_metrics_dict
            },
        }

    if cfg.fit == "jointly":
        results = results | {
            f"{stage}_target_loss": [],
            f"{stage}_all_target_loss": [],
            f"{stage}_guide_loss": [],
            f"{stage}_all_guide_loss": [],
        }

    return results


def append_results(
    results_dict: dict[str, list[float | Tensor]],
    stage: str,
    all_loss: Tensor,
    all_target_loss: Tensor | None,
    all_guide_loss: Tensor | None,
    all_train_metrics: dict[str, Tensor] | None,
    all_test_metrics: dict[str, Tensor] | None,
    all_full_target_metrics: dict[str, Tensor] | None,
    all_guide_metrics: dict[str, Tensor] | None,
    all_full_guide_metrics: dict[str, Tensor] | None,
    final_target_train_metrics: dict[str, float],
    final_target_test_metrics: dict[str, float],
    final_guide_metrics: dict[str, float],
    final_full_target_metrics: dict[str, float],
    final_full_guide_metrics: dict[str, float],
):
    results_dict[f"{stage}_train_loss"].append(all_loss[-1].item())
    results_dict[f"{stage}_all_train_loss"].append(all_loss)
    if all_train_metrics is not None:
        for name, vals in all_train_metrics.items():
            results_dict[f"{stage}_all_target_train_{name}"].append(vals)

    if all_test_metrics is not None:
        for name, vals in all_test_metrics.items():
            results_dict[f"{stage}_all_target_test_{name}"].append(vals)

    if all_full_target_metrics is not None:
        for name, vals in all_full_target_metrics.items():
            results_dict[f"{stage}_all_target_full_{name}"].append(vals)

    if all_guide_metrics is not None:
        for name, vals in all_guide_metrics.items():
            results_dict[f"{stage}_all_guide_{name}"].append(vals)

    if all_full_guide_metrics is not None:
        for name, vals in all_full_guide_metrics.items():
            results_dict[f"{stage}_all_guide_full_{name}"].append(vals)

    for name, val in final_target_train_metrics.items():
        results_dict[f"{stage}_target_train_{name}"].append(val)

    for name, val in final_target_test_metrics.items():
        results_dict[f"{stage}_target_test_{name}"].append(val)

    for name, val in final_guide_metrics.items():
        results_dict[f"{stage}_guide_{name}"].append(val)

    for name, val in final_full_target_metrics.items():
        results_dict[f"{stage}_target_full_{name}"].append(val)

    for name, val in final_full_guide_metrics.items():
        results_dict[f"{stage}_guide_full_{name}"].append(val)

    if all_target_loss is not None:
        results_dict[f"{stage}_target_loss"].append(all_target_loss[-1].item())
        results_dict[f"{stage}_all_target_loss"].append(all_target_loss)

    if all_guide_loss is not None:
        results_dict[f"{stage}_guide_loss"].append(all_guide_loss[-1].item())
        results_dict[f"{stage}_all_guide_loss"].append(all_guide_loss)


def save_images(
    inr: SparseImageINR,
    stage: str,
    cfg: DictConfig,
    opencanopy_index: int,
    image_dir: str,
):
    image_filenames: dict[str, Tensor] = {}
    if cfg.save_target_images or cfg.save_guide_images:
        reconstructed_images = inr.reconstructed_images
    else:
        reconstructed_images = None

    if cfg.save_target_images:
        target_image = inr.target_image.permute((1, 2, 0))
        train_mask = inr.train_mask
        test_mask = inr.test_mask
        assert reconstructed_images is not None
        reconstructed_target = reconstructed_images[0].permute((1, 2, 0))
        image_filenames = image_filenames | {
            f"{opencanopy_index}_{stage}_target_image.pt": target_image,
            f"{opencanopy_index}_{stage}_train_mask.pt": train_mask,
            f"{opencanopy_index}_{stage}_test_mask.pt": test_mask,
            f"{opencanopy_index}_{stage}_reconstructed_target.pt": reconstructed_target,
        }

    if cfg.save_guide_images:
        assert reconstructed_images is not None
        reconstructed_guide = reconstructed_images[1].permute((1, 2, 0))
        guide_image = inr.guide_image.permute((1, 2, 0))
        image_filenames = image_filenames | {
            f"{opencanopy_index}_{stage}_guide_image.pt": guide_image,
            f"{opencanopy_index}_{stage}_reconstructed_guide.pt": reconstructed_guide,
        }

    for name, img in image_filenames.items():
        torch.save(img, os.path.join(image_dir, name))


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig):
    seed = cfg.seed
    _ = torch.manual_seed(seed)
    opencanopy = OpenCanopy(
        "datasets/canopy_height", split=cfg.data.split, nan_above=cfg.data.nan_above
    )
    centercrop = CenterCrop(512)
    device = torch.device(cfg.device)
    log.info(f"using device: {device}")
    target_criterion = hydra.utils.instantiate(cfg.target_criterion)
    guide_criterion = hydra.utils.instantiate(cfg.guide_criterion)

    image_dir = os.path.join(HydraConfig.get().runtime.output_dir, "images")
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)

    target_metrics_dict = {
        name: hydra.utils.instantiate(metric)
        for metric_entry in cfg.target_metrics
        for name, metric in metric_entry.items()
    } | {"rescaled_loss": hydra.utils.instantiate(cfg.target_criterion)}
    guide_metrics_dict = {
        name: hydra.utils.instantiate(metric)
        for metric_entry in cfg.guide_metrics
        for name, metric in metric_entry.items()
    } | {"rescaled_loss": hydra.utils.instantiate(cfg.guide_criterion)}
    full_target_metrics_dict = {
        name: hydra.utils.instantiate(metric)
        for metric_entry in cfg.full_target_metrics
        for name, metric in metric_entry.items()
    }
    full_guide_metrics_dict = {
        name: hydra.utils.instantiate(metric)
        for metric_entry in cfg.full_guide_metrics
        for name, metric in metric_entry.items()
    }
    results = {
        "opencanopy_idx": list(cfg.data.opencanopy_indices),
    }
    if cfg.stage1.fit is not None:
        results = results | generate_results_dict(
            "stage1",
            cfg.stage1,
            target_metrics_dict,
            full_target_metrics_dict,
            guide_metrics_dict,
            full_guide_metrics_dict,
        )
    if cfg.stage2.fit is not None:
        results = results | generate_results_dict(
            "stage2",
            cfg.stage2,
            target_metrics_dict,
            full_target_metrics_dict,
            guide_metrics_dict,
            full_guide_metrics_dict,
        )
    for i in cfg.data.opencanopy_indices:
        log.info(f"Fitting on index {i}")
        guide, target = opencanopy[i]
        guide, target = centercrop(guide), centercrop(target)
        zero_ratio = torch.sum(target == 0.0) / torch.numel(target)
        if zero_ratio > cfg.data.zero_ratio_cutoff:
            log.warning(
                f"Opencanopy index {i} has {zero_ratio * 100:.2f}% > "
                + f"{cfg.data.zero_ratio_cutoff * 100:.2f}% zeroes and will be skipped"
            )
            results["opencanopy_idx"].remove(i)
            continue
        model = hydra.utils.instantiate(cfg.model).to(device)
        target_optimizer = create_optimizer(cfg.target_optimizer, model)
        if cfg.guide_optimizer.get("share_with_target", False):
            guide_optimizer = target_optimizer
        else:
            guide_optimizer = create_optimizer(cfg.guide_optimizer, model)
        if cfg.joint_optimizer.get("share_with_target", False):
            joint_optimizer = target_optimizer
        elif cfg.joint_optimizer.get("share_with_guide", False):
            joint_optimizer = guide_optimizer
        else:
            joint_optimizer = create_optimizer(cfg.joint_optimizer, model)

        inr = SparseImageINR(
            target_image=target,
            guide_image=guide,
            channel_dim=0,
            full_image_metrics_fill_na=cfg.data.full_image_metrics_fill_na,
            target_normalization_shift=cfg.data.target_normalization_shift,
            target_normalization_scale=cfg.data.target_normalization_scale,
            guide_normalization_shift=cfg.data.guide_normalization_shift,
            guide_normalization_scale=cfg.data.guide_normalization_scale,
            sparsity=cfg.data.sparsity,
            test_grid_skip=cfg.data.test_grid_skip,
            model=model,
            target_optimizer=target_optimizer,
            guide_optimizer=guide_optimizer,
            joint_optimizer=joint_optimizer,
            target_criterion=target_criterion,
            guide_criterion=guide_criterion,
            target_metrics=target_metrics_dict,
            guide_metrics=guide_metrics_dict,
            full_target_metrics=full_target_metrics_dict,
            full_guide_metrics=full_guide_metrics_dict,
            device=device,
        )

        loss_and_metrics = inr.fit(
            what=cfg.stage1.fit,
            epochs=cfg.stage1.epochs,
            return_target_train_metrics=cfg.stage1.save_running_target_metrics,
            return_target_test_metrics=cfg.stage1.save_running_target_metrics,
            return_guide_metrics=cfg.stage1.save_running_guide_metrics,
            full_image_metrics_interval=cfg.full_image_metrics_interval,
            target_loss_weight=cfg.target_loss_weight,
            guide_loss_weight=cfg.guide_loss_weight,
        )

        if cfg.stage1.fit is not None:
            (
                final_target_train,
                final_target_test,
                final_guide,
                final_full_target,
                final_full_guide,
            ) = inr.evaluate(True, True, True, True, None, None)
            assert final_target_train is not None
            assert final_target_test is not None
            assert final_guide is not None
            assert final_full_target is not None
            assert final_full_guide is not None
            final_metrics = (
                final_target_train,
                final_target_test,
                final_guide,
                final_full_target,
                final_full_guide,
            )

            append_results(results, "stage1", *loss_and_metrics, *final_metrics)

            save_images(
                inr,
                "stage1",
                cfg.stage1,
                i,
                image_dir,
            )

        if cfg.stage2.fit is not None:
            loss_and_metrics = inr.fit(
                what=cfg.stage2.fit,
                epochs=cfg.stage2.epochs,
                return_target_train_metrics=cfg.stage2.save_running_target_metrics,
                return_target_test_metrics=cfg.stage2.save_running_target_metrics,
                return_guide_metrics=cfg.stage2.save_running_guide_metrics,
                full_image_metrics_interval=cfg.full_image_metrics_interval,
                target_loss_weight=cfg.target_loss_weight,
                guide_loss_weight=cfg.guide_loss_weight,
            )

            (
                final_target_train,
                final_target_test,
                final_guide,
                final_full_target,
                final_full_guide,
            ) = inr.evaluate(True, True, True, True, None, None)
            assert final_target_train is not None
            assert final_target_test is not None
            assert final_guide is not None
            assert final_full_target is not None
            assert final_full_guide is not None
            final_metrics = (
                final_target_train,
                final_target_test,
                final_guide,
                final_full_target,
                final_full_guide,
            )

            append_results(results, "stage2", *loss_and_metrics, *final_metrics)

            save_images(
                inr,
                "stage2",
                cfg.stage2,
                i,
                image_dir,
            )

    table = pd.DataFrame(results)
    table.to_pickle(
        os.path.join(
            HydraConfig.get().runtime.output_dir,
            "results.pickle",
        )
    )


if __name__ == "__main__":
    main()
