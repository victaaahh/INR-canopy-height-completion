from typing import Union

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from src.utils import Tracker


def train(
    epochs: int,
    train_dataloader: Union[list[tuple[torch.Tensor, ...]], DataLoader],
    test_dataloader: Union[list[tuple[torch.Tensor, ...]], DataLoader],
    model: Module,
    criterion,
    optimizer: Optimizer,
    tracker: Tracker,
    device: torch.device,
):
    for i in tqdm(range(epochs)):
        model.train()
        train_loss = 0.0
        for input, target in train_dataloader:
            input = input.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, target)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for input, target in test_dataloader:
                input = input.to(device)
                target = target.to(device)
                output = model(input)
                loss = criterion(output, target)
                test_loss += loss.item()
        tracker.log_metrics({"train_loss": train_loss, "test_loss": test_loss})
