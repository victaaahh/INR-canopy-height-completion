import os
import sys

sys.path.append("..")

import logging

import numpy as np
import torch
from torchvision.transforms.v2 import CenterCrop

from src.data import OpenCanopy

logging.basicConfig(level=logging.INFO)

log = logging.getLogger(__name__)

log.info(f"Starting...")

opencanopy = OpenCanopy(
    "../datasets/canopy_height", split="train", nan_above=float("inf")
)

sample = np.loadtxt("../notebooks/data_management/10000_samples.csv")

centercrop = CenterCrop(512)

n_bins = 256

all_hist = np.zeros((len(sample), 4, n_bins))

max_vals = np.zeros((len(sample), 4))
min_vals = np.zeros((len(sample), 4))
median_vals = np.zeros((len(sample), 4))
means = np.zeros((len(sample), 4))
mean_squares = np.zeros((len(sample), 4))

for i, idx in enumerate(sample):
    if i % 500 == 0:
        log.info(f"Index: {i}")

    sat, _ = opencanopy[idx]
    sat = centercrop(sat)

    for j, channel in enumerate(sat):
        hist, bin_edges = np.histogram(channel, range=(0.0, float(n_bins)), bins=n_bins)

        all_hist[i, j] = hist

        max_vals[i, j] = channel.max()
        min_vals[i, j] = channel.min()
        median_vals[i, j] = channel.median()
        means[i, j] = torch.mean(channel).item()
        mean_squares[i, j] = torch.mean(channel * channel).item()

if not os.path.exists("data_dist_results"):
    os.mkdir("data_dist_results")
np.save("data_dist_sat_results/all_hist.npy", all_hist)
np.save("data_dist_sat_results/max_vals.npy", max_vals)
np.save("data_dist_sat_results/min_vals.npy", min_vals)
np.save("data_dist_sat_results/median_vals.npy", median_vals)
np.save("data_dist_sat_results/means.npy", means)
np.save("data_dist_sat_results/mean_squares.npy", mean_squares)
