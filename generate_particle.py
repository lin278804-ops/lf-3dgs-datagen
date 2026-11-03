# generate_3dpiv.py (修改后，一体化版本)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple

def generate_particle_point_clouds(
        ppm_concentration: float,
        num_microlenses_total: int,
        x_spatial_range: Tuple[float, float],
        y_spatial_range: Tuple[float, float],
        z_spatial_range: Tuple[float, float],
        particle_diameter_um: float,
        points_per_particle: int,
        particle_diameters_um=None,
        return_diameters: bool = False,
):
    """
    在指定三维空间内生成随机粒子，并立即将它们扩展为服从高斯分布的点光源云。

    Args:
        ppm_concentration (float): 期望的 PPM 值。
        num_microlenses_total (int): 总微透镜数量。
        x_spatial_range / y_spatial_range / z_spatial_range: 物理空间范围 (mm)。
        particle_diameter_um (float): 统一物理直径 (µm)；当 particle_diameters_um
            未提供时对所有粒子生效（与历史行为一致）。
        points_per_particle (int): 每个粒子扩展出的点光源数量。
        particle_diameters_um: 可选，形状 (N,) 的逐粒子直径 (µm)。提供时忽略
            统一直径，且 N 覆盖由 ppm 算出的粒子数。
        return_diameters: 若 True，额外返回形状 (N,) 的实际直径数组。

    Returns:
        point_clouds (N*P, 3), particle_centers (N, 3) [, diameters_um (N,)]
    """
    if particle_diameters_um is not None:
        diams = np.asarray(particle_diameters_um, dtype=np.float64).reshape(-1)
        num_particles = int(diams.size)
    else:
        num_particles = int(ppm_concentration * num_microlenses_total)
        diams = np.full(num_particles, float(particle_diameter_um), dtype=np.float64)

    print(f"[Point Cloud Generator] PPM 浓度: {ppm_concentration} particles/microlens")
    print(f"[Point Cloud Generator] 总微透镜数量: {num_microlenses_total}")
    print(f"[Point Cloud Generator] 正在生成 {num_particles} 个粒子中心...")
    if num_particles > 0 and np.allclose(diams, diams[0]):
        print(f"[Point Cloud Generator] 粒子直径: {diams[0]:.1f} µm")
    else:
        print(f"[Point Cloud Generator] 粒子直径: polydisperse "
              f"[{diams.min():.1f}, {diams.max():.1f}] µm")
    print(f"[Point Cloud Generator] 每个粒子扩展为 {points_per_particle} 个点光源")

    if num_particles <= 0:
        print("[Point Cloud Generator] 警告: 粒子生成数量为0。")
        empty = np.empty((0, 3), dtype=np.float64)
        empty_d = np.empty((0,), dtype=np.float64)
        return (empty, empty, empty_d) if return_diameters else (empty, empty)

    x_centers = np.random.uniform(x_spatial_range[0], x_spatial_range[1], num_particles)
    y_centers = np.random.uniform(y_spatial_range[0], y_spatial_range[1], num_particles)
    z_centers = np.random.uniform(z_spatial_range[0], z_spatial_range[1], num_particles)
    particle_centers = np.vstack((x_centers, y_centers, z_centers)).T

    total_points = num_particles * points_per_particle
    print(f"[Point Cloud Generator] 总计生成 {total_points} 个点光源。")

    # σ = r/3 = d/(2000*3) mm；支持逐粒子直径
    sigma = (diams / 2000.0) / 3.0  # (N,)
    random_offsets = np.random.randn(num_particles, points_per_particle, 3)
    scaled_offsets = random_offsets * sigma[:, None, None]
    point_clouds_3d = particle_centers[:, np.newaxis, :] + scaled_offsets
    point_clouds_flat = point_clouds_3d.reshape(total_points, 3)

    if return_diameters:
        return point_clouds_flat, particle_centers, diams.astype(np.float64)
    return point_clouds_flat, particle_centers

# test_generator.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 确保 generate_3dpiv.py 与此脚本在同一文件夹下


# --- 主测试脚本 ---
if __name__ == "__main__":
    print("--- 运行 'generate_particle_point_clouds' 函数的完整测试脚本 ---")
    plt.rcParams['font.sans-serif'] = ['SimHei']

    plt.rcParams['axes.unicode_minus'] = False
    # --- 1. 定义测试参数 ---
    # 为了可视化效果清晰，我们使用较少的粒子
    PPM = 0.01
    TOTAL_MICROLENSES = 500
    EXPECTED_PARTICLES = int(PPM * TOTAL_MICROLENSES) # 预期生成 5 个粒子

    # 物理空间范围
    X_RANGE = (-5.0, 5.0)
    Y_RANGE = (-5.0, 5.0)
    Z_RANGE = (-10.0, 10.0)

    # 粒子模型参数
    PARTICLE_DIAMETER_UM = 80.0  # 模拟80微米的粒子
    POINTS_PER_PARTICLE = 300    # 每个粒子扩展为300个点光源

    # --- 2. 主功能调用与验证 ---
    print("\n--- [主测试] 正在调用函数生成点云... ---")
    try:
        # 调用函数
        point_clouds, particle_centers = generate_particle_point_clouds(
            ppm_concentration=PPM,
            num_microlenses_total=TOTAL_MICROLENSES,
            x_spatial_range=X_RANGE,
            y_spatial_range=Y_RANGE,
            z_spatial_range=Z_RANGE,
            particle_diameter_um=PARTICLE_DIAMETER_UM,
            points_per_particle=POINTS_PER_PARTICLE
        )

        print("\n--- [主测试] 函数执行完毕，开始验证输出... ---")

        # 验证粒子中心数组的形状
        expected_centers_shape = (EXPECTED_PARTICLES, 3)
        print(f"验证粒子中心形状: 期望 {expected_centers_shape}, 得到 {particle_centers.shape}")
        assert particle_centers.shape == expected_centers_shape, "粒子中心形状不匹配！"
        print(" -> 粒子中心形状正确。")

        # 验证点云数组的形状
        expected_points_shape = (EXPECTED_PARTICLES * POINTS_PER_PARTICLE, 3)
        print(f"验证点云形状: 期望 {expected_points_shape}, 得到 {point_clouds.shape}")
        assert point_clouds.shape == expected_points_shape, "点云形状不匹配！"
        print(" -> 点云形状正确。")
        print("\n[主测试] 所有形状验证通过！")

    except Exception as e:
        print(f"\n[主测试失败] 函数调用或验证时发生错误: {e}")
        # 在失败时创建空数组以避免后续代码崩溃
        point_clouds, particle_centers = np.array([]), np.array([])

    # --- 3. 可视化验证 ---
    if particle_centers.shape[0] > 0:
        print("\n--- [可视化验证] 正在生成3D散点图... ---")
        print("说明：红色 'X' 代表粒子中心，蓝色小点代表其周围的点光源云。")

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制点光源云
        ax.scatter(point_clouds[:, 0], point_clouds[:, 1], point_clouds[:, 2],
                   s=1,  # 点光源用很小的点表示
                   c='blue',
                   alpha=0.3, # 使用透明度以显示云的密度
                   label=f'{point_clouds.shape[0]} 点光源')

        # 绘制粒子中心
        ax.scatter(particle_centers[:, 0], particle_centers[:, 1], particle_centers[:, 2],
                   s=80, # 粒子中心用大一点的标记
                   c='red',
                   marker='X', # 使用不同的标记
                   depthshade=False, # 确保标记颜色不会因深度而变暗
                   label=f'{particle_centers.shape[0]} 粒子中心')

        ax.set_title(f"可视化验证 (粒子直径: {PARTICLE_DIAMETER_UM}µm, {POINTS_PER_PARTICLE}点/粒子)")
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        
        # 设置坐标轴范围以匹配输入
        ax.set_xlim(X_RANGE)
        ax.set_ylim(Y_RANGE)
        ax.set_zlim(Z_RANGE)
        
        ax.legend()
        plt.show()

    # --- 4. 边缘情况测试 ---
    print("\n--- [边缘情况测试] 测试 PPM = 0 的情况... ---")
    try:
        empty_clouds, empty_centers = generate_particle_point_clouds(
            ppm_concentration=0,
            num_microlenses_total=TOTAL_MICROLENSES,
            x_spatial_range=X_RANGE,
            y_spatial_range=Y_RANGE,
            z_spatial_range=Z_RANGE,
            particle_diameter_um=PARTICLE_DIAMETER_UM,
            points_per_particle=POINTS_PER_PARTICLE
        )
        
        print(f"PPM=0时，返回的点云形状: {empty_clouds.shape}, 粒子中心形状: {empty_centers.shape}")
        assert empty_clouds.shape == (0, 3), "PPM=0时，点云形状应为(0,3)"
        assert empty_centers.shape == (0, 3), "PPM=0时，粒子中心形状应为(0,3)"
        print(" -> 边缘情况验证通过。")

    except Exception as e:
        print(f"\n[边缘情况测试失败] 测试PPM=0时发生错误: {e}")


    print("\n--- 所有测试完成 ---")