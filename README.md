# LF-3DGS 合成数据生成

光场粒子数据集的生成器：在体积里放粒子，把每个粒子扩成高斯点云，再按非聚焦光场相机做光线追迹，得到子孔径图像和粒子真值。

这是 [lf-3dgs-project](https://github.com/lin278804-ops/lf-3dgs-project) 所用训练数据的生成端。本仓库不包含已经生成的数据集。

## 流程

```text
ppm、粒径、空间范围
        │
        ▼
generate_particle_point_clouds     粒子中心 + 高斯点云（σ = 半径 / 3）
        │
        ▼
LightFieldCamera.generate_light_field_image
        默认每条光线权重 5，再按整帧最大值归一化到 0–255
        │
        ▼
generate_sai_array                 (视角行, 视角列, 透镜行, 透镜列)
        │
        ▼
.npz：sai_stack、particle_centers、体素掩码
```

正式相机和 dataset9 相同：

| 项 | 值 |
|---|---|
| 微透镜 | 186 × 232，孔径 0.099 mm |
| CCD | 4096 × 5120，像素 0.0045 mm（每透镜 22×22） |
| 主镜 | 焦距 100 mm，孔径 25 mm，物距 200 mm |
| 微透镜焦距 | 0.8 mm |
| 体积 | x ±10 mm，y ±8 mm，z ±5 mm |
| 真值网格 | 64 × 128 × 128 |
| 每个粒子的点 | 50 |
| 每个点的光线 | 1000 |
| 默认粒径 | 20 / 50 / 80 µm |
| 默认浓度 | 0.025 / 0.05 / 0.075（粒子数 = ppm × 透镜数） |

每个样本用 `GLOBAL_SEED + sample_idx` 播种，避免多进程里粒子场重复。

## 环境

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Smoke check 只用 NumPy 和 Matplotlib。批量生成还需要 OpenCV 和 tqdm。

## Smoke check

不生成完整数据集。用 6×6 透镜的小相机追少量光线，大约几秒。

```bash
python scripts/smoke_check.py
```

通过时打印 `SMOKE_OK`。它检查粒子数量、点云落在粒子附近、图像归一化到 255、子孔径形状、同一随机种子可复现、提高光线权重会增加能量、粒子中心写入体素网格，以及逐粒子直径会改变点云散布。

## 批量生成

在仓库根目录执行。下面这个例子只生成 9 个样本（3 个粒径 × 3 个浓度），写到 `output/smoke_set/`。正式规模把 `NUM_SAMPLES` 调大。

```bash
NUM_SAMPLES=9 NUM_WORKERS=2 OUTPUT_DIR=output/smoke_set python generate_dataset.py
```

可用环境变量（不设置则用上表默认值）：

| 变量 | 含义 |
|---|---|
| `NUM_SAMPLES` | 计划样本数，按组合均分 |
| `OUTPUT_DIR` | 输出目录 |
| `NUM_WORKERS` | 进程数 |
| `GLOBAL_SEED` | 默认 `20260610` |
| `PARTICLE_DIAMETER_LIST_UM` | 逗号分隔，例如 `50` |
| `PPM_CONCENTRATION_LIST` | 逗号分隔，例如 `0.025` |
| `POINTS_PER_PARTICLE` | 每个粒子的点数 |
| `RAYS_PER_POINT` | 每个点的光线数 |
| `TARGET_GRID_SHAPE` | `D,H,W`，例如 `64,128,128` |
| `MICROLENS_APERTURE_MM` | 微透镜孔径 |

单个 `.npz` 里有 `sai_stack`（视角数 × 透镜行 × 透镜列）、`particle_centers`、`particle_grid_gt`、粒径、浓度、空间范围和该样本的 `rng_seed`。

默认路径按帧最大值归一化，小粒子在图上不会更暗。需要强度正比于粒径平方时，调用 `generate_light_field_image(..., point_weights=..., normalize=False)`，不要用批量脚本的默认分支。
