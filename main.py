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


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg):
    opencanopy = OpenCanopy(
        "datasets/canopy_height", split=cfg.data.split, nan_above=cfg.data.nan_above
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
    for i in cfg.data.opencanopy_indices:
        log.info(f"Fitting on index {i}")
        _, target = opencanopy[i]
        target = centercrop(target)
        zero_ratio = torch.sum(target == 0.0) / torch.numel(target)
        if zero_ratio > cfg.data.zero_ratio_cutoff:
            log.warning(
                f"Opencanopy index {i} has {zero_ratio * 100:.2f}% > "
                + f"{cfg.data.zero_ratio_cutoff * 100:.2f}% zeroes and will be skipped"
            )
            results["opencanopy_idx"].remove(i)
            continue
        model = hydra.utils.instantiate(cfg.model).to(device)
        if "_target_" in cfg.optimizer:
            optimizer = hydra.utils.instantiate(cfg.optimizer)(model.parameters())
        elif "muon" in cfg.optimizer:
            middle_weights = [
                p for p in model.middle_layers.parameters() if p.ndim >= 2
            ]
            middle_biases = [p for p in model.middle_layers.parameters() if p.ndim < 2]
            other_params = [
                *model.first_layer.parameters(),
                *model.last_layer.parameters(),
            ]
            optimizer = SingleDeviceMuonWithAuxAdam(
                [
                    dict(params=middle_weights, use_muon=True, **cfg.optimizer.muon),
                    dict(
                        params=middle_biases + other_params,
                        use_muon=False,
                        **cfg.optimizer.adam,
                    ),
                ]
            )
        else:
            raise ValueError("Unsupported optimizer config")

        inr = SparseImageINR(
            target_image=target,
            channel_dim=0,
            full_image_metrics_fill_na=cfg.data.full_image_metrics_fill_na,
            normalization_shift=cfg.data.normalization_shift,
            normalization_scale=cfg.data.normalization_scale,
            sparsity=cfg.data.sparsity,
            test_grid_skip=cfg.data.test_grid_skip,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            metrics=metrics_dict,
            full_image_metrics=full_image_metrics_dict,
            batch_size=cfg.dataloader.batch_size,
            shuffle=cfg.dataloader.shuffle,
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

        if cfg.save_images:
            target_image = inr.target_image.permute((1, 2, 0))
            train_mask = inr.train_mask
            reconstructed_target = inr.reconstructed_target.permute((1, 2, 0))
            test_mask = inr.test_mask
            image_dir = os.path.join(
                hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "images"
            )
            if not os.path.exists(image_dir):
                os.mkdir(image_dir)
            image_filenames = {
                f"{i}_target_image.pt": target_image,
                f"{i}_train_mask.pt": train_mask,
                f"{i}_reconstructed_target.pt": reconstructed_target,
                f"{i}_test_mask.pt": test_mask,
            }
            for name, img in image_filenames.items():
                torch.save(img, os.path.join(image_dir, name))

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
