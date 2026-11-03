# utils.py
import numpy as np
from typing import Tuple


def convert_centers_to_voxel_mask_grid(
        particle_centers: np.ndarray,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        z_range: Tuple[float, float],
        grid_shape: Tuple[int, int, int] = (64, 128, 128)
) -> np.ndarray:
    """
    将 (N, 3) 格式的粒子中心物理坐标转换为一个三维体素掩码网格。
    每个包含一个或多个粒子中心的体素的值被设置为 1，其他为 0。

    Args:
        particle_centers (np.ndarray): 形状为 (N, 3) 的粒子中心坐标数组 [x, y, z] (in mm)。
        x_range, y_range, z_range: 物理空间范围。
        grid_shape: 目标体素网格的形状 (Depth, Height, Width)。

    Returns:
        np.ndarray: 一个形状为 (D, H, W) 的 NumPy 数组，值为 0 或 1。
    """
    # --- 1. 参数解包 ---
    D, H, W = grid_shape
    min_x, max_x = x_range
    min_y, max_y = y_range
    min_z, max_z = z_range

    # --- 2. 初始化一个全零的体素网格 ---
    voxel_grid = np.zeros(grid_shape, dtype=np.uint8)  # 使用 uint8 节省空间

    # --- 3. 处理没有粒子的情况 ---
    if particle_centers.shape[0] == 0:
        return voxel_grid

    # --- 4. 坐标转换：一次性将所有物理坐标映射到体素索引 ---
    x_indices = np.floor((particle_centers[:, 0] - min_x) / (max_x - min_x) * W).astype(int)
    y_indices = np.floor((particle_centers[:, 1] - min_y) / (max_y - min_y) * H).astype(int)
    z_indices = np.floor((particle_centers[:, 2] - min_z) / (max_z - min_z) * D).astype(int)

    # --- 5. 过滤掉超出网格范围的粒子索引 ---
    valid_mask = (x_indices >= 0) & (x_indices < W) & \
                 (y_indices >= 0) & (y_indices < H) & \
                 (z_indices >= 0) & (z_indices < D)

    valid_indices = np.vstack((z_indices[valid_mask],
                               y_indices[valid_mask],
                               x_indices[valid_mask]))

    # --- 6. 移除重复的体素索引 ---
    # 因为多个粒子可能落入同一个体素，我们只需要标记一次
    unique_indices = np.unique(valid_indices, axis=1)

    # --- 7. 将对应位置的体素值设置为 1 ---
    # 使用高级索引直接赋值
    voxel_grid[unique_indices[0], unique_indices[1], unique_indices[2]] = 1

    return voxel_grid