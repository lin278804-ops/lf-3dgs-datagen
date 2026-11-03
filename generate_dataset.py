# -*- coding: utf-8 -*-
import os
import numpy as np
from tqdm import tqdm
import cv2
import time
import json
from multiprocessing import Pool, cpu_count
import itertools
import random

# --- 导入自定义模块 ---
try:
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)

    from simulate_lightfield import LightFieldCamera, generate_sai_array
    from generate_particle import generate_particle_point_clouds
    from utils import convert_centers_to_voxel_mask_grid
except ImportError as e:
    print(f"错误: 无法导入必要的模块 - {e}")
    exit()

# 注：本文件曾被意外删除，2026-06-11 由 __pycache__/generate_dataset.cpython-310.pyc
# 字节码逐行重建（diagnostics/_dis_gendata.py），含 2026-06-10 的 RNG 修复
# （per-sample np.random.seed，修复 multiprocessing fork 共享随机状态导致的
# 粒子场重复 bug）。


def _parse_number_list(env_value: str, cast=float):
    """
    解析形如 "0.2,0.05,0.01" 或 "64,256,256" 的环境变量列表。
    """
    items = [s.strip() for s in str(env_value).split(",") if s.strip() != ""]
    return [cast(x) for x in items]


def _maybe_override_from_env():
    """
    从环境变量覆盖部分参数（不改变默认行为，只有设置了 env 才生效）。
    """
    overrides = {}
    if os.environ.get("PARTICLE_DIAMETER_LIST_UM"):
        overrides["PARTICLE_DIAMETER_LIST_UM"] = _parse_number_list(
            os.environ["PARTICLE_DIAMETER_LIST_UM"], float)
    if os.environ.get("PPM_CONCENTRATION_LIST"):
        overrides["PPM_CONCENTRATION_LIST"] = _parse_number_list(
            os.environ["PPM_CONCENTRATION_LIST"], float)
    if os.environ.get("POINTS_PER_PARTICLE"):
        overrides["POINTS_PER_PARTICLE"] = int(float(os.environ["POINTS_PER_PARTICLE"]))
    if os.environ.get("RAYS_PER_POINT"):
        overrides["RAYS_PER_POINT"] = int(float(os.environ["RAYS_PER_POINT"]))
    if os.environ.get("TARGET_GRID_SHAPE"):
        shape = tuple(_parse_number_list(os.environ["TARGET_GRID_SHAPE"], int))
        if len(shape) != 3:
            raise ValueError(f"TARGET_GRID_SHAPE 必须为 3 个整数，例如 '64,256,256'，实际为: {shape}")
        overrides["TARGET_GRID_SHAPE"] = shape
    if os.environ.get("MICROLENS_APERTURE_MM"):
        overrides["MICROLENS_APERTURE_MM"] = float(os.environ["MICROLENS_APERTURE_MM"])
    if os.environ.get("NUM_WORKERS"):
        overrides["NUM_WORKERS"] = int(os.environ["NUM_WORKERS"])
    return overrides


def generate_single_sample(args):
    """
    生成并保存一个数据样本。此函数设计为可被多进程调用。
    """
    sample_idx, diameter_um, concentration, config, camera_params_dict = args

    # --- RNG 修复（2026-06-10）：multiprocessing fork 会让所有 worker 共享
    # 相同的 numpy 随机状态，导致大量样本粒子场完全相同。
    # 这里按 sample_idx 显式播种，保证每个样本可复现且互不相同。 ---
    sample_seed = (int(config.get("GLOBAL_SEED", 20260610)) + int(sample_idx)) % 2147483648
    np.random.seed(sample_seed)

    lf_camera = LightFieldCamera(**camera_params_dict)

    OUTPUT_DIR = config["OUTPUT_DIR"]
    POINTS_PER_PARTICLE = config["POINTS_PER_PARTICLE"]
    RAYS_PER_POINT = config["RAYS_PER_POINT"]
    X_RANGE, Y_RANGE, Z_RANGE = config["X_RANGE"], config["Y_RANGE"], config["Z_RANGE"]
    TARGET_GRID_SHAPE = tuple(config["TARGET_GRID_SHAPE"])
    H_res, W_res = lf_camera.num_mla_y, lf_camera.num_mla_x
    V_res, U_res = lf_camera.pixels_per_microlens_y, lf_camera.pixels_per_microlens_x

    # 1. 生成粒子点云
    point_clouds, particle_centers = generate_particle_point_clouds(
        ppm_concentration=concentration,
        num_microlenses_total=H_res * W_res,
        x_spatial_range=X_RANGE, y_spatial_range=Y_RANGE, z_spatial_range=Z_RANGE,
        particle_diameter_um=diameter_um,
        points_per_particle=POINTS_PER_PARTICLE,
    )

    # 2. 光线追迹生成原始光场图
    raw_lf_image = lf_camera.generate_light_field_image(
        point_clouds,
        num_rays_per_particle=RAYS_PER_POINT,
    )

    # 3. 提取 SAI 阵列
    sai_4d = generate_sai_array(raw_lf_image, lf_camera)
    sai_stack = sai_4d.reshape(V_res * U_res, H_res, W_res).astype(np.float32)

    # 4. 真值体素网格
    particle_gt_grid = convert_centers_to_voxel_mask_grid(
        particle_centers=particle_centers,
        x_range=X_RANGE, y_range=Y_RANGE, z_range=Z_RANGE,
        grid_shape=TARGET_GRID_SHAPE,
    )

    # 5. 保存
    conc_str = f"{concentration:.3f}".replace(".", "p")
    output_filename_base = f"sample_{sample_idx:05d}_diam{int(diameter_um)}um_conc{conc_str}"

    output_npz_path = os.path.join(OUTPUT_DIR, f"{output_filename_base}.npz")
    np.savez_compressed(
        output_npz_path,
        sai_stack=sai_stack,
        particle_grid_gt=particle_gt_grid,
        particle_centers=particle_centers.astype(np.float32),
        particle_diameter_um=np.float32(diameter_um),
        ppm_concentration=np.float32(concentration),
        x_range=np.array(X_RANGE, dtype=np.float32),
        y_range=np.array(Y_RANGE, dtype=np.float32),
        z_range=np.array(Z_RANGE, dtype=np.float32),
        grid_shape=np.array(TARGET_GRID_SHAPE, dtype=np.int32),
        rng_seed=np.int64(sample_seed),
    )

    return True


def save_config_to_file(config_dict, directory):
    filepath = os.path.join(directory, "config.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("--- 数据集生成配置 ---\n\n")
        formatted_config = json.dumps(config_dict, indent=4, ensure_ascii=False)
        f.write(formatted_config)
    print(f"配置文件已保存至: {filepath}")


if __name__ == "__main__":
    # --- 1. 数据集与并行化参数 ---
    NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", "9999"))
    OUTPUT_DIR = os.environ.get(
        "OUTPUT_DIR",
        "/mnt/data/voxel_sim/simu_ltc/output/lightfield_dataset3/",
    )

    print(f"可用CPU核心数: {cpu_count()}")
    NUM_WORKERS = max(1, cpu_count() // 2)

    # --- 2. 物理模型参数（可被环境变量覆盖）---
    PARTICLE_DIAMETER_LIST_UM = [20.0, 50.0, 80.0]
    PPM_CONCENTRATION_LIST = [0.025, 0.05, 0.075]

    POINTS_PER_PARTICLE = 50
    RAYS_PER_POINT = 1000

    camera_params = {
        "main_lens_focal_length_mm": 100.0, "main_lens_aperture_diameter_mm": 25.0,
        "object_distance_mm": 200.0, "mla_focal_length_mm": 0.8,
        "num_mla_x": 232, "num_mla_y": 186, "microlens_aperture_mm": 0.099,
        "ccd_resolution_x": 5120, "ccd_resolution_y": 4096, "ccd_pixel_pitch_mm": 0.0045,
    }

    X_RANGE = (-10, 10)
    Y_RANGE = (-8.0, 8.0)
    Z_RANGE = (-5.0, 5.0)
    TARGET_GRID_SHAPE = (64, 128, 128)

    # --- 3. 环境变量覆盖 ---
    overrides = _maybe_override_from_env()
    PARTICLE_DIAMETER_LIST_UM = overrides.get("PARTICLE_DIAMETER_LIST_UM", PARTICLE_DIAMETER_LIST_UM)
    PPM_CONCENTRATION_LIST = overrides.get("PPM_CONCENTRATION_LIST", PPM_CONCENTRATION_LIST)
    POINTS_PER_PARTICLE = overrides.get("POINTS_PER_PARTICLE", POINTS_PER_PARTICLE)
    RAYS_PER_POINT = overrides.get("RAYS_PER_POINT", RAYS_PER_POINT)
    TARGET_GRID_SHAPE = overrides.get("TARGET_GRID_SHAPE", TARGET_GRID_SHAPE)
    if "MICROLENS_APERTURE_MM" in overrides:
        camera_params["microlens_aperture_mm"] = overrides["MICROLENS_APERTURE_MM"]
        print(f"覆盖 microlens_aperture_mm={camera_params['microlens_aperture_mm']}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 4. 生成计划 ---
    all_combinations = list(itertools.product(PARTICLE_DIAMETER_LIST_UM, PPM_CONCENTRATION_LIST))
    num_combinations = len(all_combinations)

    samples_per_combo = NUM_SAMPLES // num_combinations
    total_samples_to_generate = samples_per_combo * num_combinations
    print(f"发现 {num_combinations} 种 (粒径, 浓度) 组合。")
    print(f"每个组合将生成 {samples_per_combo} 个样本，总计 {total_samples_to_generate} 个。")

    generation_plan = all_combinations * samples_per_combo

    random.shuffle(generation_plan)

    GLOBAL_SEED = int(os.environ.get("GLOBAL_SEED", "20260610"))

    config_for_workers = {
        "OUTPUT_DIR": OUTPUT_DIR,
        "POINTS_PER_PARTICLE": POINTS_PER_PARTICLE,
        "RAYS_PER_POINT": RAYS_PER_POINT,
        "X_RANGE": X_RANGE, "Y_RANGE": Y_RANGE, "Z_RANGE": Z_RANGE,
        "TARGET_GRID_SHAPE": TARGET_GRID_SHAPE,
        "GLOBAL_SEED": GLOBAL_SEED,
    }

    full_config_data = {
        "输出目录": OUTPUT_DIR,
        "数据集参数": {
            "请求样本总数": NUM_SAMPLES,
            "实际生成总数": total_samples_to_generate,
            "组合总数": num_combinations,
            "每个组合的样本数": samples_per_combo,
            "全局随机种子": GLOBAL_SEED,
            "播种方式": "per-sample np.random.seed(GLOBAL_SEED + sample_idx)",
        },
        "粒子模型参数": {
            "粒径列表 (µm)": PARTICLE_DIAMETER_LIST_UM,
            "PPM浓度列表": PPM_CONCENTRATION_LIST,
            "每个粒子的点光源数": POINTS_PER_PARTICLE,
            "每个点光源的光线数": RAYS_PER_POINT,
        },
        "物理空间参数 (mm)": {
            "X范围": X_RANGE, "Y范围": Y_RANGE, "Z范围 (深度)": Z_RANGE,
        },
        "真值体素网格参数": {"形状 (D, H, W)": TARGET_GRID_SHAPE},
        "相机参数": camera_params,
    }
    save_config_to_file(full_config_data, OUTPUT_DIR)
    if "NUM_WORKERS" in overrides:
        NUM_WORKERS = max(1, int(overrides["NUM_WORKERS"]))
    print(f"将使用 {NUM_WORKERS} 个CPU核心进行并行生成。")

    # --- 5. 并行生成 ---
    tasks = []
    for i, (diameter_um, concentration) in enumerate(generation_plan):
        tasks.append((i, diameter_um, concentration, config_for_workers, camera_params))

    print(f"\n 开始并行生成 {total_samples_to_generate} 个数据样本")
    start_time = time.time()

    with Pool(processes=NUM_WORKERS) as pool:
        results = list(tqdm(pool.imap_unordered(generate_single_sample, tasks),
                            total=len(tasks), desc="生成数据集样本"))

    end_time = time.time()

    print("\n数据集生成完毕！")
    print(f"总计生成了 {len(results)} / {total_samples_to_generate} 个样本。")
    print(f"总耗时: {end_time - start_time:.2f} 秒。")
    print(f"数据保存在: '{OUTPUT_DIR}'")
