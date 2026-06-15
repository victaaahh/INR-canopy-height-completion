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

global_hist = np.zeros(n_bins)

max_heights = np.zeros(len(sample))
median_heights = np.zeros(len(sample))
means = np.zeros(len(sample))
vars_ = np.zeros(len(sample))

pixels_larger_than_600 = 0
pixels_larger_than_6000 = 0

for i, idx in enumerate(sample):
    if i % 500 == 0:
        log.info(f"Index: {i}")
        
    _, target = opencanopy[idx]
    target = centercrop(target)
    
    hist, bin_edges = np.histogram(target, range=(0.0, float(n_bins)), bins=n_bins)
    
    
    global_hist += hist
    pixels_larger_than_600 += torch.sum(target>600).item()
    pixels_larger_than_6000 += torch.sum(target>6000).item()
    
    max_heights[i] = target.max()
    median_heights[i] = target.median()
    means[i] = torch.mean(target).item()
    vars_[i] = torch.var(target).item()

np.save("data_dist_results/global_hist.npy", global_hist)
np.save("data_dist_results/max_heights.npy", max_heights)
np.save("data_dist_results/median_heights.npy", median_heights)
np.save("data_dist_results/means.npy", means)
np.save("data_dist_results/vars_.npy", vars_)
np.save("data_dist_results/pixels_larger_than_600.npy", pixels_larger_than_600)
np.save("data_dist_results/pixels_larger_than_6000.npy", pixels_larger_than_6000)
