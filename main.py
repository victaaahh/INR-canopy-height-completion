import logging

import hydra
import pandas as pd
import torch
from omegaconf import OmegaConf

from src import SparseImageINR
from src.data import OpenCanopy
from src.utils import Tracker
from src.utils.visualize import plot_height_map

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg):
    if not cfg.tracker.per_image:
        global_tracker = Tracker(
            project=cfg.tracker.project,
            group=cfg.tracker.group,
            name=cfg.tracker.name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    else:
        global_tracker = None
    opencanopy = OpenCanopy("datasets/canopy_height", split="train")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    log.info(f"using device: {device}")
    criterion = hydra.utils.instantiate(cfg.criterion)
    metrics_dict = {
        name: hydra.utils.instantiate(metric)
        for metric_entry in cfg.metrics
        for name, metric in metric_entry.items()
    }
    results = {
        "opencanopy_idx": list(cfg.data.opencanopy_indices),
        "train_losses": [],
        **{f"train_{name}": [] for name, _ in metrics_dict.items()},
        "test_losses": [],
        **{f"test_{name}": [] for name, _ in metrics_dict.items()},
    }
    for i in cfg.data.opencanopy_indices:
        _, target = opencanopy[i]
        model = hydra.utils.instantiate(cfg.model).to(device)
        optimizer = hydra.utils.instantiate(cfg.optimizer)(model.parameters())
        if cfg.tracker.per_image:
            tracker = Tracker(
                project=cfg.tracker.project,
                group=cfg.tracker.group,
                name=cfg.tracker.name + f"_{i}",
                config=OmegaConf.to_container(cfg, resolve=True),
            )
        else:
            tracker = None

        inr = SparseImageINR(
            target_image=target,
            normalize_image=cfg.data.normalize_image,
            channel_dim=0,
            sparsity=cfg.data.sparsity,
            test_grid_skip=cfg.data.test_grid_skip,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            metrics=metrics_dict,
            batch_size=cfg.dataloader.batch_size,
            shuffle=cfg.dataloader.shuffle,
            device=device,
            tracker=tracker,
        )
        train_loss, train_metrics = inr.fit(cfg.epochs)
        test_loss, test_metrics = inr.evaluate()
        results["train_losses"].append(train_loss)
        results["test_losses"].append(test_loss)
        for name, val in train_metrics.items():
            results[f"train_{name}"].append(val)
        for name, val in test_metrics.items():
            results[f"test_{name}"].append(val)

        if cfg.tracker.log_images:
            assert tracker is not None, "Can't log images when per_image=False"
            original = inr.target_image.permute((1, 2, 0))
            sparse_mask = inr.train_mask
            reconstructed = inr.reconstructed_target.permute((1, 2, 0))
            eval_grid_mask = inr.test_mask
            tracker.log_images(
                {
                    "images": [
                        plot_height_map(original, "original"),
                        plot_height_map(original, "sparse", mask=sparse_mask),
                        plot_height_map(reconstructed, "reconstructed"),
                        plot_height_map(original, "eval grid", mask=eval_grid_mask),
                    ]
                }
            )

        if tracker is not None:
            tracker.finish()

    if global_tracker is not None:
        table = pd.DataFrame(results)
        global_tracker.log_tables({"results": table})
        summary = {
            f"mean_{name}": torch.tensor(vals).mean() for name, vals in results.items()
        }
        global_tracker.summarize(summary)
        global_tracker.finish()
        return table["test_losses"].mean()


if __name__ == "__main__":
    main()
