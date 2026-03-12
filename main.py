import logging

import hydra
import torch
from torch.utils.data import DataLoader

from src import train
from src.data import OpenCanopy, single_image_dataset

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = hydra.utils.instantiate(cfg.model).to(device)
    optimizer = hydra.utils.instantiate(cfg.optimizer)(model.parameters())
    criterion = hydra.utils.instantiate(cfg.criterion)
    opencanopy = OpenCanopy("datasets/canopy_height", split="train")
    dataset = single_image_dataset(
        opencanopy[cfg.opencanopy_index][1], channel_dim=0, device=device
    )
    dataloader = DataLoader(
        dataset,
        cfg.dataloader.batch_size,
        cfg.dataloader.shuffle,
        collate_fn=lambda x: x,  # CustomTensorDataset requires identity collate_fn
    )
    log.info(f"using device: {device}")
    train(
        epochs=cfg.epochs,
        dataloader=dataloader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )


if __name__ == "__main__":
    main()
