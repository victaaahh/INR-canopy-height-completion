import logging

import hydra
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src import train
from src.data import OpenCanopy, single_image_dataset
from src.utils import Tracker

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg):
    tracker = Tracker(
        project=cfg.tracker.project,
        group=cfg.tracker.group,
        name=cfg.tracker.name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = hydra.utils.instantiate(cfg.model).to(device)
    optimizer = hydra.utils.instantiate(cfg.optimizer)(model.parameters())
    criterion = hydra.utils.instantiate(cfg.criterion)
    opencanopy = OpenCanopy("datasets/canopy_height", split="train")
    train_data, test_data = single_image_dataset(
        opencanopy[cfg.data.opencanopy_index][1],
        channel_dim=0,
        test_grid_skip=cfg.data.test_grid_skip,
        sparsity=cfg.data.sparsity,
        device=device,
    )
    if cfg.data.dataloader.batch_size == "full":
        train_dataloader = [train_data.tensors]
    else:
        train_dataloader = DataLoader(
            train_data,
            cfg.data.dataloader.batch_size,
            cfg.data.dataloader.shuffle,
            collate_fn=lambda x: x,  # CustomTensorDataset requires identity collate_fn
        )
    test_dataloader = [test_data.tensors]
    log.info(f"using device: {device}")
    train(
        epochs=cfg.epochs,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        tracker=tracker,
        device=device,
    )
    tracker.finish()


if __name__ == "__main__":
    main()
