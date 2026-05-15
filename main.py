import logging
import os

import hydra
import pandas as pd
import torch
from muon import SingleDeviceMuonWithAuxAdam
from torchvision.transforms.v2 import CenterCrop

from src import SparseImageINR
from src.data import OpenCanopy

log = logging.getLogger(__name__)


def create_optimizer(cfg, model):
    if "_target_" in cfg:
        optimizer = hydra.utils.instantiate(cfg)(model.parameters())
    elif "muon" in cfg:
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


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg):
    opencanopy = OpenCanopy(
        "datasets/canopy_height", split="train", nan_above=cfg.data.nan_above
    )
    centercrop = CenterCrop(512)
    device = torch.device(cfg.device)
    log.info(f"using device: {device}")
    criterion = hydra.utils.instantiate(cfg.criterion)
    metrics_dict = {
        name: hydra.utils.instantiate(metric)
        for metric_entry in cfg.metrics
        for name, metric in metric_entry.items()
    } | {"rescaled_loss": hydra.utils.instantiate(cfg.criterion)}
    full_image_metrics_dict = {
        name: hydra.utils.instantiate(metric)
        for metric_entry in cfg.full_image_metrics
        for name, metric in metric_entry.items()
    }
    results = {
        "opencanopy_idx": list(cfg.data.opencanopy_indices),
        "train_loss": [],
        **{f"train_{name}": [] for name in metrics_dict},
        "test_loss": [],
        **{f"test_{name}": [] for name in metrics_dict},
        **{f"full_image_{name}": [] for name in full_image_metrics_dict},
    }
    if cfg.save_running_metrics:
        results = results | {
            "all_train_loss": [],
            **{f"all_train_{name}": [] for name in metrics_dict},
            "all_test_loss": [],
            **{f"all_test_{name}": [] for name in metrics_dict},
            **{f"all_full_image_{name}": [] for name in full_image_metrics_dict},
        }
    if cfg.save_canopy_height_maps:
        results = results | {
            "target_image": [],
            "train_mask": [],
            "reconstructed_target": [],
            "test_mask": [],
        }
    if cfg.save_guide_images:
        results = results | {"guide_image": [], "reconstructed_guide": []}
    for i in cfg.data.opencanopy_indices:
        guide, target = opencanopy[i]
        target = centercrop(target)
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
            criterion=criterion,
            metrics=metrics_dict,
            full_image_metrics=full_image_metrics_dict,
            device=device,
        )
        (
            all_train_loss,
            all_train_metrics,
            all_test_loss,
            all_test_metrics,
            all_full_image_metrics,
        ) = inr.fit(
            cfg.epochs,
            return_test_metrics=cfg.save_running_metrics,
            full_image_metrics_interval=cfg.full_image_metrics_interval,
            what=cfg.what_to_fit,
            target_loss_weight=cfg.target_loss_weight,
            guide_loss_weight=cfg.guide_loss_weight,
        )
        if (
            all_test_loss is not None
            and all_test_metrics is not None
            and all_full_image_metrics is not None
        ):  # no reason to re-compute if we already have the final results
            test_loss = all_test_loss[-1]
            test_metrics = {name: vals[-1] for name, vals in all_test_metrics.items()}
            full_image_metrics = {
                name: vals[-1] for name, vals in all_full_image_metrics.items()
            }
        else:
            test_loss, test_metrics, full_image_metrics = inr.evaluate(
                return_full_image_metrics=True
            )
            assert full_image_metrics is not None

        results["train_loss"].append(all_train_loss[-1])
        results["test_loss"].append(test_loss)
        for name, vals in all_train_metrics.items():
            results[f"train_{name}"].append(vals[-1])
        for name, val in test_metrics.items():
            results[f"test_{name}"].append(val)
        for name, val in full_image_metrics.items():
            results[f"full_image_{name}"].append(val)

        if (
            all_test_loss is not None
            and all_test_metrics is not None
            and all_full_image_metrics is not None
        ):  # i.e save_running_metrics
            results["all_train_loss"].append(all_train_loss)
            results["all_test_loss"].append(all_test_loss)
            for name, vals in all_train_metrics.items():
                results[f"all_train_{name}"].append(vals)
            for name, vals in all_test_metrics.items():
                results[f"all_test_{name}"].append(vals)
            for name, vals in all_full_image_metrics.items():
                results[f"all_full_image_{name}"].append(vals)

        if (
            "target_image" in results
            and "train_mask" in results
            and "reconstructed_target" in results
            and "test_mask" in results
        ):
            target_image = inr.target_image.permute((1, 2, 0))
            train_mask = inr.train_mask
            reconstructed_images = inr.reconstructed_images
            reconstructed_target = reconstructed_images[0].permute((1, 2, 0))
            test_mask = inr.test_mask
            results["target_image"].append(target_image)
            results["train_mask"].append(train_mask)
            results["reconstructed_target"].append(reconstructed_target)
            results["test_mask"].append(test_mask)
        else:
            reconstructed_images = None

        if "guide_image" in results and "reconstructed_guide" in results:
            if reconstructed_images is not None:
                reconstructed_guide = reconstructed_images[1].permute((1, 2, 0))
            else:
                reconstructed_guide = inr.reconstructed_images[1].permute((1, 2, 0))
            guide_image = inr.guide_image.permute((1, 2, 0))
            results["guide_image"].append(guide_image)
            results["reconstructed_guide"].append(reconstructed_guide)

    table = pd.DataFrame(results)
    table.to_pickle(
        os.path.join(
            hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
            "results.pickle",
        )
    )
    return table["test_loss"].mean()  # return value used by optuna


if __name__ == "__main__":
    main()
