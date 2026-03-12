import os

import geopandas as gpd
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


class OpenCanopy(Dataset):
    def __init__(self, data_dir, split):
        self.data_dir = data_dir
        geometries = gpd.read_file(os.path.join(data_dir, "geometries.geojson"))
        if split in ["test", "train", "val", "buffer"]:
            self.split = split
        else:
            raise ValueError("split should be 'test', 'train', 'val' or 'buffer")
        self.geometries = geometries[geometries["split"] == split]

    def __len__(self):
        return len(self.geometries)

    def __getitem__(self, idx):
        row = self.geometries.iloc[[idx]]
        year = row["lidar_year"].iloc[0]
        geometry = row["geometry"]
        with rasterio.open(os.path.join(self.data_dir, f"{year}/spot.vrt")) as vrt:
            geometry_proj = geometry.to_crs(vrt.crs).iloc[0]
            window = vrt.window(*geometry_proj.bounds)
            satelite_img = np.stack(
                [
                    vrt.read(band, window=window, out_dtype=np.float32)
                    for band in vrt.indexes
                ],
                axis=0,
            )
        with rasterio.open(os.path.join(self.data_dir, f"{year}/lidar.vrt")) as vrt:
            geometry_proj = geometry.to_crs(vrt.crs).iloc[0]
            window = vrt.window(*geometry_proj.bounds)
            canopy_top_height = torch.tensor(
                vrt.read(1, window=window, out_dtype=np.float32)
            )

        return torch.tensor(satelite_img), canopy_top_height[None, :, :]
