import sys

sys.path.append("..")

import logging
import os

import pandas as pd
import torch
from scipy.interpolate import (
    CloughTocher2DInterpolator,
    LinearNDInterpolator,
    NearestNDInterpolator,
)
from torchvision.transforms.v2 import CenterCrop

from src.data import OpenCanopy, single_image_sparse_dataset
from src.utils.metrics import ME, R2, RMSE, SSIM

logging.basicConfig(level=logging.INFO)

log = logging.getLogger(__name__)

log.info(f"Starting...")

_ = torch.manual_seed(8182)

opencanopy = OpenCanopy(
    "../datasets/canopy_height",
    split="test",
    nan_above=600,
)
opencanopy_indices = [
    9021,
    8507,
    5653,
    8567,
    10542,
    258,
    4994,
    7569,
    7052,
    6139,
    12145,
    11187,
    13529,
    3420,
    4843,
    11003,
    346,
    10260,
    509,
    1304,
    11221,
    2811,
    10881,
    2087,
    4797,
    5003,
    7751,
    11298,
    10227,
    2928,
    4387,
    4982,
    11922,
    11652,
    6406,
    1986,
    7825,
    10882,
    11340,
    2786,
    8549,
    3979,
    6230,
    6058,
    13570,
    4283,
    1579,
    6596,
    7934,
    3472,
    3924,
    9213,
    10024,
    1192,
    532,
    13033,
    4685,
    3019,
    7003,
    18,
    13637,
    9060,
    5943,
    2153,
    7837,
    158,
    1690,
    8530,
    7319,
    9659,
    11373,
    3633,
    870,
    2702,
    12252,
    11601,
    7313,
    743,
    12670,
    9254,
    10630,
    7072,
    11105,
    12066,
    10420,
    12071,
    1263,
    11252,
    937,
    7362,
    5796,
    2163,
    3352,
    9719,
    13339,
    6044,
    9214,
    639,
    5590,
    5710,
]

full_image_metrics_fill_na = 0.0
centercrop = CenterCrop(512)
sparsities = [1.0, 0.5, 0.1, 0.02]
coord_range = torch.linspace(-1, 1, 512)
all_coords = torch.stack(
    torch.meshgrid(coord_range, coord_range, indexing="ij"), dim=-1
).reshape(-1, 2)

rmse = RMSE()
me = ME()
r2 = R2()
ssim = SSIM(
    data_range=600,
    k_1=0.01,
    k_2=0.03,
    gaussian_kernel=True,
    sigma=1.5,
    kernel_size=11,
    device="cpu",
)

interpolation_methods = {
    "nearest": NearestNDInterpolator,
    "linear": LinearNDInterpolator,
    "cubic": CloughTocher2DInterpolator,
}

results: dict[str, list[int | float | str | torch.Tensor]] = {
    "opencanopy_idx": [],
    "method": [],
    "data.sparsity": [],
    "target_test_RMSE": [],
    "target_test_ME": [],
    "target_test_R2": [],
    "target_full_SSIM": [],
    "reconstructed_target": [],
}

for i in opencanopy_indices:
    log.info(f"Index: {i}")
    _, target = opencanopy[i]
    target = centercrop(target)
    target = target.permute(1, 2, 0)
    target_nan_mask = target.isnan()

    for sparsity in sparsities:
        train_dataset, test_dataset, train_mask, test_mask = (
            single_image_sparse_dataset(
                target,
                channel_dim=2,
                test_grid_skip=2,
                sparsity=sparsity,
            )
        )

        train_points, train_values = train_dataset.tensors
        test_points, test_values = test_dataset.tensors

        for name, class_ in interpolation_methods.items():
            results["opencanopy_idx"].append(i)
            results["method"].append(name)
            results["data.sparsity"].append(sparsity)
            interp = class_(train_points, train_values)

            test_pred = torch.tensor(interp(test_points))
            test_nan_mask = test_pred.isnan()
            test_pred_no_nan = test_pred[~test_nan_mask]
            test_values_no_nan = test_values[~test_nan_mask]

            test_rmse = rmse(test_pred_no_nan, test_values_no_nan)
            test_me = me(test_pred_no_nan, test_values_no_nan)
            test_r2 = r2(test_pred_no_nan, test_values_no_nan)

            results["target_test_RMSE"].append(test_rmse)
            results["target_test_ME"].append(test_me)
            results["target_test_R2"].append(test_r2)

            all_pred = torch.tensor(interp(all_coords))
            reconstruction = all_pred.reshape(512, 512, 1)
            reconstruction_nan_mask = reconstruction.isnan()

            reconstruction_no_nan = reconstruction.clone()
            reconstruction_no_nan[target_nan_mask | reconstruction_nan_mask] = (
                full_image_metrics_fill_na
            )

            target_no_nan = target.clone()
            target_no_nan[target_nan_mask | reconstruction_nan_mask] = (
                full_image_metrics_fill_na
            )

            full_ssim = ssim(reconstruction_no_nan, target_no_nan)

            results["target_full_SSIM"].append(full_ssim)

            results["reconstructed_target"].append(reconstruction)


table = pd.DataFrame(results)
if not os.path.exists("results"):
    os.mkdir("results")
table.to_pickle(
    "results/interpolation_baselines_results.pickle",
)
