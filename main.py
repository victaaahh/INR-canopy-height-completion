import logging
import os

import hydra
import pandas as pd
import torch
from torchvision.transforms.v2 import CenterCrop

from src import SparseImageINR
from src.data import OpenCanopy

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg):
    opencanopy = OpenCanopy("datasets/canopy_height", split="train")
    centercrop = CenterCrop(512)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
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
    if cfg.save_images:
        results = results | {
            "target_image": [],
            "train_mask": [],
            "reconstructed_target": [],
            "test_mask": [],
        }
    for i in cfg.data.opencanopy_indices:
        _, target = opencanopy[i]
        target = centercrop(target)
        model = hydra.utils.instantiate(cfg.model).to(device)
        optimizer = hydra.utils.instantiate(cfg.optimizer)(model.parameters())

        inr = SparseImageINR(
            target_image=target,
            channel_dim=0,
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
        ) = inr.fit(cfg.epochs, return_test_metrics=cfg.save_running_metrics)
        if all_test_loss is not None and all_test_metrics is not None:
            test_loss, test_metrics = (
                all_test_loss[-1],
                {name: vals[-1] for name, vals in all_test_metrics.items()},
            )
        else:
            test_loss, test_metrics, _ = inr.evaluate()

        results["train_loss"].append(all_train_loss[-1])
        results["test_loss"].append(test_loss)
        for name, vals in all_train_metrics.items():
            results[f"train_{name}"].append(vals[-1])
        for name, val in test_metrics.items():
            results[f"test_{name}"].append(val)
        for name, vals in all_full_image_metrics.items():
            results[f"full_image_{name}"].append(vals[-1])

        if all_test_metrics is not None:
            results["all_train_loss"].append(all_train_loss)
            results["all_test_loss"].append(all_test_loss)
            for name, vals in all_train_metrics.items():
                results[f"all_train_{name}"].append(vals)
            for name, vals in all_test_metrics.items():
                results[f"all_test_{name}"].append(vals)
            for name, vals in all_full_image_metrics.items():
                results[f"all_full_image_{name}"].append(vals)

        if "target_image" in results:
            target_image = inr.target_image.permute((1, 2, 0))
            train_mask = inr.train_mask
            reconstructed_target = inr.reconstructed_target.permute((1, 2, 0))
            test_mask = inr.test_mask
            results["target_image"].append(target_image)
            results["train_mask"].append(train_mask)
            results["reconstructed_target"].append(reconstructed_target)
            results["test_mask"].append(test_mask)

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
