#%% md
# quick implementation of grid index for searching

#%%
import os
from typing import List
from tqdm import tqdm
import numpy as np
import laspy
import pandas as pd

#%%
os.chdir('/scratch/bingxin/project/ImpliCity_Regress/ImpliCityFork')
input_pc_folder = "data/source_data/ZUR1/Point_Clouds"
input_pc_paths: List = [os.path.join(input_pc_folder, _path) for _path in os.listdir(input_pc_folder)]


def load_las_as_numpy(las_path: str) -> np.ndarray:
    """
    Load .las point cloud and convert into numpy array
    This one is slow, because laspy returns a list of tuple, which can't be directly transformed into numpy array
    Args:
        las_path: full path to las file

    Returns:

    """
    with laspy.open(las_path) as f:
        _las = f.read()
    x = np.array(_las.x).reshape((-1, 1))
    y = np.array(_las.y).reshape((-1, 1))
    z = np.array(_las.z).reshape((-1, 1))
    points = np.concatenate([x, y, z], 1)
    # points = _las.points.array
    # points = np.asarray(points.tolist())[:, 0:3].astype(np.float)
    return points


merged_pts: np.ndarray = np.empty((0, 3))
for _full_path in tqdm(input_pc_paths, desc="Loading point clouds"):
    # _temp_points = load_pc(_full_path)
    _temp_points = load_las_as_numpy(_full_path)

    merged_pts = np.append(merged_pts, _temp_points, 0)
#%%
p_min = merged_pts.min(axis=0)
p_max = merged_pts.max(axis=0)
#%%
grid_size = np.array([32, 32])  # x, y
grid_dimension = np.ceil((p_max - p_min)[:2] / grid_size).astype(int)

#%%
# for index grid, p_min -> (0, 0), p_min -> (grid_dimension[0] - 1, grid_dimension[1] - 1)

# get grid extent
grid_num = grid_dimension[0] * grid_dimension[1]
# _x = np.linspace(p_min[0], p_max[0], grid_dimension[0]+1)
# _y = np.linspace(p_min[1], p_max[1], grid_dimension[1]+1)
# vx, vy = np.meshgrid(_x, _y)
# vx = np.expand_dims(vx, axis=2)
# vy = np.expand_dims(vy, axis=2)
#
# grid_extent = np.concatenate([
#     vx[:-1, :-1],
#     vy[:-1, :-1],
#     vx[1:, 1:],
#     vy[1:, 1:]
# ], axis=2).reshape((-1, 4))  # [xmin, ymin, xmax, ymax]

# del _x, _y, vx, vy
#%%
# assign grid index to each point
point_grid_index = np.floor((merged_pts[:, 1] - p_min[1]) / grid_size[1]) * grid_dimension[0] + np.floor((merged_pts[:, 0] - p_min[0]) / grid_size[0])
point_grid_index = point_grid_index.astype(int)

#%%
df = pd.DataFrame({'grid_index': point_grid_index})
del point_grid_index

#%%
index_dict = {k: v.index.tolist() for k, v in df.groupby('grid_index')}


#%%
from tqdm import tqdm

for i in tqdm(index_dict.keys()):
    a = merged_pts[index_dict[i]]
