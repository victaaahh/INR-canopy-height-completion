from typing import Optional

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes


def mask_to_alpha_image(
    mask: torch.Tensor, red: float, green: float, blue: float
) -> torch.Tensor:
    """Given a mask, return an image (array) that visualizes the mask by being
    transparent where the mask is True and colored where the mask is false"""
    red_channel = torch.ones(mask.shape) * red
    green_channel = torch.ones(mask.shape) * green
    blue_channel = torch.ones(mask.shape) * blue
    return torch.stack((red_channel, green_channel, blue_channel, mask), dim=-1)


def plot_height_map(
    image: torch.Tensor, title: str, mask: Optional[torch.Tensor] = None
) -> Axes:
    plt.figure()
    plt.imshow(image, cmap="gray")
    plt.colorbar()
    if mask is not None:
        plt.imshow(mask_to_alpha_image(mask, red=1.0, green=0.0, blue=0.0))
    plt.title(title)
    return plt.gca()
