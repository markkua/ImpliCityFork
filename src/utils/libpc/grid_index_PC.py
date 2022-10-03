import os
from typing import List

import torch
from tqdm import tqdm
import numpy as np
import laspy
import pandas as pd
import logging


class GridIndexPointCloud:
    """
        index grid: i-axis(row) aligns to x, j-axis(col) aligns to y
        (0, 0)  -----------------------------------------------------------------------> [y], [axis=1]
        |       0       |       1       |       ...     |       grid_dimension[1]-1
        |____________________________________________________________________________
        |                       ...
        |       ...     | n = j * grid_dimension[1] + i |   ...
        |
        |
        [x], [axis=0]
    """

    def __init__(self, pts: np.ndarray):
        self.pts: np.ndarray = pts  # point cloud  [x, y, z, ...]
        self.index_dict = {}  # {grid_index: [index of points in this grid])

        self.grid_size = None  # (x, y) in meter
        self._grid_dimension = None  # (x, y)
        self._p_min = None
        self._p_max = None
        self._grid_num = -1
        self._index_created = False

    def create_index_grid_2d(self, grid_size):
        """

        Args:
            grid_size: array(size_x, size_y)

        Returns:

        """
        logging.debug("Creating index grid")
        self.grid_size = grid_size

        if not len(self.pts) > 0:
            logging.warning("No point in the GridIndexPointCloud")
            return

        self._p_min = self.pts.min(axis=0)
        self._p_max = self.pts.max(axis=0)
        self._grid_dimension = np.ceil((self._p_max - self._p_min)[:2] / self.grid_size).astype(int)
        self._grid_num = self._grid_dimension[0] * self._grid_dimension[1]
        self.index_dict = {k: [] for k in range(self._grid_num)}

        point_grid_index = self._get_index_num(self.pts)

        df = pd.DataFrame({'grid_index': point_grid_index})
        del point_grid_index
        group_by = df.groupby('grid_index')

        self.index_dict.update({k: np.array(v.index).astype(int) for k, v in group_by})
        del df

        self._index_created = True
        logging.debug(f"Index grid created, dimension = {self._grid_dimension}")
        return

    def crop_2d_index(self, p_min, p_max):
        if not self._index_created:
            logging.warning(f"Query fail! Create index grid first")
            return None
        # calculate indices of grids that touch the cropping ROI
        p_min = np.array(p_min[:2])
        p_max = np.array(p_max[:2])
        corners = np.array([p_min, p_max]).reshape((2, 2))
        corner_idx = self._get_index_num(corners)

        col = corner_idx % self._grid_dimension[1]
        row = np.floor(corner_idx / self._grid_dimension[1]).astype(int)

        cols = np.arange(col[0], col[1]+1)
        rows = np.arange(row[0], row[1]+1)

        _xv, _yv = np.meshgrid(rows, cols)
        grid_idx_ls = _xv.reshape(-1) * self._grid_dimension[1] + _yv.reshape(-1)

        grid_idx_ls = [x for x in grid_idx_ls if 0 <= x < self._grid_num]

        if 0 == len(grid_idx_ls):
            logging.warning(f"Empty point cloud cropping: p_min={p_min}, p_max={p_max}")
            return np.empty()

        # get all point index in these grids
        point_idx_ls = [self.index_dict[idx] for idx in grid_idx_ls]
        point_idx_ls = np.concatenate(point_idx_ls).reshape(-1)
        points_in_grid = self.pts[point_idx_ls]  # points located in these grids

        # do the cropping
        out_idx = np.where((points_in_grid[:, 0] > p_min[0]) & (points_in_grid[:, 0] < p_max[0]) &
                           (points_in_grid[:, 1] > p_min[1]) & (points_in_grid[:, 1] < p_max[1]))[0]

        return point_idx_ls[out_idx].copy()

    def crop_2d(self, p_min, p_max):
        idx = self.crop_2d_index(p_min, p_max)
        return self.pts[idx].copy(), idx

    def _get_index_num(self, pts):
        point_grid_index = np.floor((pts[:, 0] - self._p_min[0]) / self.grid_size[0]) * self._grid_dimension[1]\
                           + np.floor((pts[:, 1] - self._p_min[1]) / self.grid_size[1])
        point_grid_index = point_grid_index.astype(int)
        return point_grid_index


if __name__ == '__main__':
    # Unit Test
    # os.chdir('/scratch/bingxin/project/ImpliCity_Regress/ImpliCityFork')
    # input_pc_folder = "data/source_data/ZUR1/Point_Clouds"
    # input_pc_paths: List = [os.path.join(input_pc_folder, _path) for _path in os.listdir(input_pc_folder)]
    #
    # from src.utils.libpc import load_pc

    # # load demo data
    # merged_pts: np.ndarray = np.empty((0, 3))
    # for _full_path in tqdm(input_pc_paths, desc="Loading point clouds"):
    #     # _temp_points = load_pc(_full_path)
    #     _temp_points = load_pc(_full_path)
    #
    #     merged_pts = np.append(merged_pts, _temp_points, 0)
    # p_min = merged_pts.min(axis=0)
    # p_max = merged_pts.max(axis=0)
    p_min = np.array([4.6321000e+05, 5.2481500e+06, 3.8360692e+02])
    p_max = np.array([4.65450000e+05, 5.24994000e+06, 9.36144507e+02])

    # pseudo point cloud
    num_points = 100000000  # 1e8
    merged_pts = np.random.rand(num_points * 3).reshape(-1, 3) * (p_max - p_min) + p_min

    print(f"merged_pts: {merged_pts.shape}")
    print(f"min: {merged_pts.min(axis=0)}")
    print(f"max: {merged_pts.max(axis=0)}")

    # instance
    grid_idx_pc = GridIndexPointCloud(merged_pts)
    # grid_idx_pc.create_index_grid_2d((32, 32))  # 64 iters/ses
    grid_idx_pc.create_index_grid_2d((16, 16))  # 92 iters/sec
    # grid_idx_pc.create_index_grid_2d((8, 8))  # 110 iters/sec
    # grid_idx_pc.create_index_grid_2d((4, 4))  # 120 iters/sec
    # grid_idx_pc.create_index_grid_2d((2, 2))  # 120 iters/sec


    print(f"grid dimension: {grid_idx_pc._grid_dimension}")

    # test cropping
    crop_p_min = p_min + np.array([5., 5., 0.])
    crop_p_max = crop_p_min + np.array([64, 64, 0.])

    print(f"cropping from {crop_p_min} to {crop_p_max}")

    cropped_pc, _ = grid_idx_pc.crop_2d(crop_p_min, crop_p_max)

    print(f"cropped shape: {cropped_pc.shape}")
    print(f"cropped min: {cropped_pc.min(axis=0)}")
    print(f"cropped max: {cropped_pc.max(axis=0)}")

    # result looping through all points (simple stupid method)
    gt_pc_idx = np.where((merged_pts[:, 0] > crop_p_min[0]) & (merged_pts[:, 0] < crop_p_max[0]) &
                         (merged_pts[:, 1] > crop_p_min[1]) & (merged_pts[:, 1] < crop_p_max[1]))[0]

    gt_pc = merged_pts[gt_pc_idx]

    print(f"gt_pc shape: {gt_pc.shape}")
    print(f"gt_pc min: {gt_pc.min(axis=0)}")
    print(f"gt_pc max: {gt_pc.max(axis=0)}")


    # test speed
    for i in tqdm(range(10000), desc="index"):
        cropped_pc = grid_idx_pc.crop_2d(crop_p_min, crop_p_max)

    # for i in tqdm(range(100), desc="looping"):
    #     gt_pc_idx = np.where((merged_pts[:, 0] > crop_p_min[0]) & (merged_pts[:, 0] < crop_p_max[0]) &
    #                          (merged_pts[:, 1] > crop_p_min[1]) & (merged_pts[:, 1] < crop_p_max[1]))[0]
    #     gt_pc = merged_pts[gt_pc_idx]








