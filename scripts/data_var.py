import sys
sys.path.append("..")

import torch
from torchvision.transforms.v2 import CenterCrop

from src.data import OpenCanopy
import numpy as np

import logging
logging.basicConfig(level=logging.INFO)

log = logging.getLogger(__name__)

log.info(f"Starting...")

opencanopy = OpenCanopy("../datasets/canopy_height", split="train", nan_above=float("inf"))

sample = np.loadtxt("../../data_management/10000_samples.csv")

centercrop = CenterCrop(512)

n_bins = 6000
old_means = np.load("data_dist_results/means.npy")
mean = np.mean(old_means)

mean_squared_differences = np.zeros(len(sample))
mean_squares = np.zeros(len(sample))
means = np.zeros(len(sample))

for i, idx in enumerate(sample):
    if i % 500 == 0:
        log.info(f"Index: {i}")
        
    _, target = opencanopy[idx]
    target = centercrop(target)
    
    mean_squared_differences[i] = torch.mean((target - mean) * (target - mean)).item()
    mean_squares[i] = torch.mean(target*target).item()
    means[i] = torch.mean(target).item()

np.save("data_var_results/means.npy", means)
np.save("data_var_results/mean_squared_differences.npy", mean_squared_differences)
np.save("data_var_results/mean_squares.npy", mean_squares)
