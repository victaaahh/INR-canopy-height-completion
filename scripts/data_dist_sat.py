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

all_hist = np.zeros((len(sample), n_bins))

max_vals = np.zeros(len(sample))
min_vals = np.zeros(len(sample))
median_vals = np.zeros(len(sample))
means = np.zeros(len(sample))
mean_squares = np.zeros(len(sample))

for i, idx in enumerate(sample):
    if i % 500 == 0:
        log.info(f"Index: {i}")

    _, target = opencanopy[idx]
    target = centercrop(target)

    hist, bin_edges = np.histogram(target, range=(0.0, float(n_bins)), bins=n_bins)

    all_hist[i] = hist

    max_vals[i] = target.max()
    min_vals[i] = target.min()
    median_vals[i] = target.median()
    means[i] = torch.mean(target).item()
    mean_squares[i] = torch.mean(target * target).item()

if not os.path.exists("data_dist_results"):
    os.mkdir("data_dist_results")
np.save("data_dist_sat_results/all_hist.npy", all_hist)
np.save("data_dist_sat_results/max_vals.npy", max_vals)
np.save("data_dist_sat_results/min_vals.npy", min_vals)
np.save("data_dist_sat_results/median_vals.npy", median_vals)
np.save("data_dist_sat_results/means.npy", means)
np.save("data_dist_sat_results/mean_squares.npy", mean_squares)
