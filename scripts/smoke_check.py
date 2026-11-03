"""CPU smoke check for the synthetic light-field generator.

Uses a tiny microlens array so ray tracing finishes in seconds. It checks the
same functions the full dataset uses:

1. Particle count follows ppm * microlens count, and the point cloud is
   centred on those particles.
2. Ray tracing returns a uint8 image whose maximum is 255.
3. Sub-aperture images tile the CCD at one view per microlens pixel.
4. The same seed reproduces the particle field.
5. Turning off per-frame max normalisation and raising a point weight
   increases deposited energy.
6. Particle centres land in the voxel occupancy grid.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from generate_particle import generate_particle_point_clouds
from simulate_lightfield import LightFieldCamera, generate_sai_array
from utils import convert_centers_to_voxel_mask_grid

# 0.099 / 0.0045 = 22 pixels per microlens, same ratio as the full camera.
PITCH = 0.099
PIXEL = 0.0045
N_MLA = 6
PX = int(round(PITCH / PIXEL))
X_RANGE = (-10.0, 10.0)
Y_RANGE = (-8.0, 8.0)
Z_RANGE = (-5.0, 5.0)


def tiny_camera() -> LightFieldCamera:
    return LightFieldCamera(
        main_lens_focal_length_mm=100.0,
        main_lens_aperture_diameter_mm=25.0,
        object_distance_mm=200.0,
        mla_focal_length_mm=0.8,
        num_mla_x=N_MLA,
        num_mla_y=N_MLA,
        microlens_aperture_mm=PITCH,
        ccd_resolution_x=N_MLA * PX,
        ccd_resolution_y=N_MLA * PX,
        ccd_pixel_pitch_mm=PIXEL,
    )


def check_particles() -> tuple:
    np.random.seed(0)
    ppm = 0.2
    n_mla = N_MLA * N_MLA
    clouds, centers = generate_particle_point_clouds(
        ppm_concentration=ppm,
        num_microlenses_total=n_mla,
        x_spatial_range=X_RANGE,
        y_spatial_range=Y_RANGE,
        z_spatial_range=Z_RANGE,
        particle_diameter_um=50.0,
        points_per_particle=8,
    )
    n = int(ppm * n_mla)
    assert centers.shape == (n, 3), centers.shape
    assert clouds.shape == (n * 8, 3), clouds.shape
    # Each particle's points stay near its centre (σ = r/3).
    grouped = clouds.reshape(n, 8, 3)
    spread = np.linalg.norm(grouped - centers[:, None, :], axis=-1).max()
    radius_mm = 50.0 / 2000.0
    assert spread < 5.0 * radius_mm, spread
    print("  particles ok", centers.shape, "max spread mm", round(float(spread), 4))
    return clouds, centers


def field_clouds(seed: int, ppm: float, n_mla: int, points: int) -> np.ndarray:
    """Particles inside the tiny camera's object field (MLA image is ~0.6 mm)."""
    np.random.seed(seed)
    clouds, _ = generate_particle_point_clouds(
        ppm_concentration=ppm,
        num_microlenses_total=n_mla,
        x_spatial_range=(-0.15, 0.15),
        y_spatial_range=(-0.15, 0.15),
        z_spatial_range=(-0.5, 0.5),
        particle_diameter_um=80.0,
        points_per_particle=points,
    )
    return clouds


def check_raytrace() -> None:
    clouds = field_clouds(2, 0.2, N_MLA * N_MLA, 4)
    cam = tiny_camera()
    assert cam.pixels_per_microlens_x == PX
    image = cam.generate_light_field_image(clouds, num_rays_per_particle=30)
    assert image.dtype == np.uint8, image.dtype
    assert image.shape == (N_MLA * PX, N_MLA * PX), image.shape
    assert int(image.max()) == 255, int(image.max())
    sai = generate_sai_array(image, cam)
    assert sai.shape == (PX, PX, N_MLA, N_MLA), sai.shape
    assert int(sai.max()) > 0
    print("  raytrace ok", image.shape, "sai", sai.shape, "max", int(image.max()))


def check_seed() -> None:
    def once(seed: int) -> np.ndarray:
        np.random.seed(seed)
        _, centers = generate_particle_point_clouds(
            ppm_concentration=0.1,
            num_microlenses_total=100,
            x_spatial_range=X_RANGE,
            y_spatial_range=Y_RANGE,
            z_spatial_range=Z_RANGE,
            particle_diameter_um=50.0,
            points_per_particle=2,
        )
        return centers

    a, b, c = once(7), once(7), once(8)
    assert np.allclose(a, b)
    assert not np.allclose(a, c)
    print("  seed ok", a.shape)


def check_weights() -> None:
    clouds = field_clouds(3, 0.15, N_MLA * N_MLA, 3)
    cam = tiny_camera()
    np.random.seed(11)
    base = cam.generate_light_field_image(
        clouds, num_rays_per_particle=40, normalize=False,
    )
    np.random.seed(11)
    weighted = cam.generate_light_field_image(
        clouds,
        num_rays_per_particle=40,
        point_weights=np.full(len(clouds), 20.0, dtype=np.float32),
        normalize=False,
    )
    assert base.dtype == np.float32 and weighted.dtype == np.float32
    assert float(weighted.sum()) > float(base.sum()) * 2.0
    print("  weights ok", round(float(base.sum()), 1), "->", round(float(weighted.sum()), 1))


def check_voxels(centers: np.ndarray) -> None:
    grid = convert_centers_to_voxel_mask_grid(
        centers, X_RANGE, Y_RANGE, Z_RANGE, grid_shape=(16, 16, 16),
    )
    assert grid.shape == (16, 16, 16)
    assert grid.dtype == np.uint8
    assert 1 <= int(grid.sum()) <= len(centers)
    print("  voxels ok", int(grid.sum()), "occupied /", len(centers))


def check_per_particle_diameter() -> None:
    diams = np.array([10.0, 100.0])
    clouds, centers, out = generate_particle_point_clouds(
        ppm_concentration=0.0,
        num_microlenses_total=1,
        x_spatial_range=X_RANGE,
        y_spatial_range=Y_RANGE,
        z_spatial_range=Z_RANGE,
        particle_diameter_um=50.0,
        points_per_particle=20,
        particle_diameters_um=diams,
        return_diameters=True,
    )
    assert centers.shape == (2, 3)
    assert np.allclose(out, diams)
    spread = np.linalg.norm(clouds.reshape(2, 20, 3) - centers[:, None, :], axis=-1)
    assert spread[1].mean() > spread[0].mean() * 3.0
    print("  per-particle diameter ok", round(float(spread[0].mean()), 4), round(float(spread[1].mean()), 4))


def main() -> None:
    print("LF synthetic generator smoke check")
    clouds, centers = check_particles()
    check_raytrace()
    check_seed()
    check_weights()
    check_voxels(centers)
    check_per_particle_diameter()
    print("SMOKE_OK")


if __name__ == "__main__":
    main()
