import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm


def train(
    epochs: int,
    dataloader: DataLoader,
    model: Module,
    criterion,
    optimizer: Optimizer,
    device: torch.device,
):
    for i in tqdm(range(epochs)):
        for input, target in dataloader:
            input = input.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            # scheduler step?
            # logging
