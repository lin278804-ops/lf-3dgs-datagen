import numpy as np
import matplotlib.pyplot as plt
# 假设 generate_3dpiv.py 文件与此脚本在同一目录下
# from generate_patrticle import generate_random_particles
import math
import matplotlib
from matplotlib.widgets import Slider # <-- 新增：导入滑块工具

class LightFieldCamera:
    # ... (您的 LightFieldCamera 类的代码保持完全不变)
    """
    模拟非聚焦型光场相机的光线追迹。
    *** 已更新：接口允许直接输入微透镜的数量 (num_mla_x, num_mla_y)。***
    """

    def __init__(self,
                 main_lens_focal_length_mm=100.0,
                 main_lens_aperture_diameter_mm=25.0,
                 object_distance_mm=200.0,
                 mla_focal_length_mm=0.8,
                 num_mla_x=200,  # <-- 修改点：直接输入微透镜水平数量
                 num_mla_y=200,  # <-- 修改点：直接输入微透镜竖直数量
                 microlens_aperture_mm=0.1,
                 ccd_resolution_x=2352,
                 ccd_resolution_y=1768,
                 ccd_pixel_pitch_mm=0.0055):

        # --- 基础参数设置 ---
        self.f_main = main_lens_focal_length_mm
        self.D_main = main_lens_aperture_diameter_mm
        self.u_main = object_distance_mm
        self.v_main = 1 / (1 / self.f_main - 1 / self.u_main)

        self.f_mla = mla_focal_length_mm
        self.microlens_pitch = microlens_aperture_mm
        self.s_mla_ccd = self.f_mla

        self.ccd_res_x = ccd_resolution_x
        self.ccd_res_y = ccd_resolution_y
        self.ccd_pixel_pitch = ccd_pixel_pitch_mm

        # --- 新增：参数校验与自洽性计算模块 ---

        print("--- 初始化相机参数校验 ---")

        # 1. 硬性约束检查：微透镜尺寸必须是像素尺寸的整数倍
        pixels_per_microlens = self.microlens_pitch / self.ccd_pixel_pitch
        # 使用浮点数容差来判断是否为整数
        if abs(pixels_per_microlens - round(pixels_per_microlens)) > 1e-9:
            raise ValueError(
                f"致命错误：微透镜孔径 ({self.microlens_pitch}mm) "
                f"不是像素尺寸 ({self.ccd_pixel_pitch}mm) 的整数倍。 "
                f"计算比值为: {pixels_per_microlens:.4f}。请调整参数后重试。"
            )
        self.pixels_per_microlens_x = int(round(pixels_per_microlens))
        self.pixels_per_microlens_y = int(round(pixels_per_microlens))
        print(f"参数检查通过：每个微透镜覆盖 {self.pixels_per_microlens_x}x{self.pixels_per_microlens_y} 个像素。")

        # 2. 从CCD参数计算最大可容纳的微透镜数量
        num_mla_x_on_ccd = self.ccd_res_x // self.pixels_per_microlens_x
        num_mla_y_on_ccd = self.ccd_res_y // self.pixels_per_microlens_y
        print(f"根据CCD尺寸，最多可容纳 {num_mla_x_on_ccd}x{num_mla_y_on_ccd} 个微透镜。")

        # 3. 获取用户直接输入的期望微透镜数量
        print(f"用户请求使用 {num_mla_x}x{num_mla_y} 的微透镜阵列。")

        # 4. 决策：确定最终用于光线追迹的微透镜数量 (取 "用户输入" 和 "CCD容量" 中的较小者)
        self.num_mla_x = min(num_mla_x, num_mla_x_on_ccd)
        self.num_mla_y = min(num_mla_y, num_mla_y_on_ccd)

        if num_mla_x > num_mla_x_on_ccd or num_mla_y > num_mla_y_on_ccd:
            print("\n!!! 警告: 您设定的微透镜阵列数量大于CCD传感器能容纳的范围。")
            print(f"!!! 将自动采用CCD所能支持的最大有效微透镜数量进行模拟。\n")

        print(f"--- 最终决定：光线追迹将使用 {self.num_mla_x}x{self.num_mla_y} 的微透镜阵列。---\n")

        # 5. 基于最终确定的微透镜数量，计算有效的、用于光追的MLA物理尺寸
        self.effective_mla_width = self.num_mla_x * self.microlens_pitch
        self.effective_mla_height = self.num_mla_y * self.microlens_pitch

        # CCD 靶面尺寸 (mm) - 仅为信息展示
        self.ccd_width = self.ccd_res_x * self.ccd_pixel_pitch
        self.ccd_height = self.ccd_res_y * self.ccd_pixel_pitch
        print(f"CCD 靶面尺寸: {self.ccd_width:.2f} mm x {self.ccd_height:.2f} mm")
        print(f"有效MLA模拟尺寸: {self.effective_mla_width:.2f} mm x {self.effective_mla_height:.2f} mm")

        # 6. 使用修正后的有效尺寸，生成正确的微透镜中心坐标
        self.mla_centers = self._generate_mla_centers()

    def _generate_mla_centers(self):
        """
        生成所有微透镜的中心坐标 (mx, my)，以有效MLA中心为 (0,0)。
        """
        centers = []
        # 使用有效的MLA尺寸来计算起始位置
        start_x = -self.effective_mla_width / 2 + self.microlens_pitch / 2
        start_y = -self.effective_mla_height / 2 + self.microlens_pitch / 2

        # 循环使用最终确定的 num_mla_x 和 num_mla_y
        for row_idx in range(self.num_mla_y):
            for col_idx in range(self.num_mla_x):
                mx = start_x + col_idx * self.microlens_pitch
                my = start_y + row_idx * self.microlens_pitch
                centers.append((mx, my))
        return np.array(centers)

    def _propagate_free_space(self, ray_coords, Zt):
        x_prime = ray_coords[:, 0] + Zt * ray_coords[:, 2]
        y_prime = ray_coords[:, 1] + Zt * ray_coords[:, 3]
        tan_theta_prime = ray_coords[:, 2]
        tan_phi_prime = ray_coords[:, 3]
        return np.vstack((x_prime, y_prime, tan_theta_prime, tan_phi_prime)).T

    def _trace_main_lens(self, ray_coords):
        x_prime = ray_coords[:, 0]
        y_prime = ray_coords[:, 1]
        tan_theta_prime = - (1 / self.f_main) * ray_coords[:, 0] + ray_coords[:, 2]
        tan_phi_prime = - (1 / self.f_main) * ray_coords[:, 1] + ray_coords[:, 3]
        return np.vstack((x_prime, y_prime, tan_theta_prime, tan_phi_prime)).T

    def _trace_microlens(self, ray_coords, ml_center_x, ml_center_y):
        x_prime = ray_coords[:, 0]
        y_prime = ray_coords[:, 1]
        tan_theta_prime = - (1 / self.f_mla) * ray_coords[:, 0] + ray_coords[:, 2] + (ml_center_x / self.f_mla)
        tan_phi_prime = - (1 / self.f_mla) * ray_coords[:, 1] + ray_coords[:, 3] + (ml_center_y / self.f_mla)
        return np.vstack((x_prime, y_prime, tan_theta_prime, tan_phi_prime)).T

    def _map_to_ccd_pixels(self, ray_x4, ray_y4):
        x_pixel_units = ray_x4 / self.ccd_pixel_pitch
        y_pixel_units = ray_y4 / self.ccd_pixel_pitch
        up = np.round(x_pixel_units + self.ccd_res_x / 2)
        vp = np.round(y_pixel_units + self.ccd_res_y / 2)
        up = np.clip(up, 0, self.ccd_res_x - 1).astype(int)
        vp = np.clip(vp, 0, self.ccd_res_y - 1).astype(int)
        return up, vp

    def generate_light_field_image(
        self,
        particles_3d,
        num_rays_per_particle=1000,
        point_weights=None,
        normalize: bool = True,
        return_counts: bool = False,
    ):
        """Ray-trace point sources onto the CCD.

        Args:
            particles_3d: (N, 3) point-source positions (mm).
            num_rays_per_particle: rays sampled per point.
            point_weights: optional (N,) per-point ray deposit weight.
                Default is a constant 5.0 for every point (historical behaviour).
            normalize: if True (default), scale the image so max == 255 and
                return uint8. If False, return float32 counts (or weighted
                counts) without normalisation.
            return_counts: if True, also return the un-normalised float32
                image. When normalize=False this equals the primary return.
        """
        light_field_image = np.zeros((self.ccd_res_y, self.ccd_res_x), dtype=np.float32)
        Z_obj_main_lens = self.u_main
        Z_main_lens_mla = self.v_main
        Z_mla_ccd = self.f_mla
        main_lens_aperture_radius = self.D_main / 2

        n_pts = len(particles_3d)
        if point_weights is None:
            weights = np.full(n_pts, 5.0, dtype=np.float32)
        else:
            weights = np.asarray(point_weights, dtype=np.float32).reshape(-1)
            if weights.size != n_pts:
                raise ValueError(
                    f"point_weights length {weights.size} != n_points {n_pts}"
                )

        for i, particle_pos in enumerate(particles_3d):
            p_x, p_y, p_z = particle_pos
            w = float(weights[i])

            # 主镜头光线落点坐标计算
            theta_aperture = np.random.uniform(0, 2 * np.pi, num_rays_per_particle)

            r_aperture = np.sqrt(np.random.uniform(0, main_lens_aperture_radius ** 2, num_rays_per_particle))
            x_at_main_lens = r_aperture * np.cos(theta_aperture)
            y_at_main_lens = r_aperture * np.sin(theta_aperture)

            # 计算主镜头处的光线矩阵
            dist_to_lens = Z_obj_main_lens - p_z
            tan_theta = (x_at_main_lens - p_x) / dist_to_lens
            tan_phi = (y_at_main_lens - p_y) / dist_to_lens
            current_rays = np.vstack((x_at_main_lens, y_at_main_lens, tan_theta, tan_phi)).T

            # 主镜头光线偏折+传播至微透镜阵列
            current_rays = self._trace_main_lens(current_rays)
            current_rays = self._propagate_free_space(current_rays, Z_main_lens_mla)

            mla_x_coords = current_rays[:, 0]
            mla_y_coords = current_rays[:, 1]

            # 使用与_generate_mla_centers一致的有效尺寸来定义坐标系
            start_x = -self.effective_mla_width / 2
            start_y = -self.effective_mla_height / 2

            ml_col_indices = np.floor((mla_x_coords - start_x) / self.microlens_pitch).astype(int)
            ml_row_indices = np.floor((mla_y_coords - start_y) / self.microlens_pitch).astype(int)

            valid_mla_mask = (ml_col_indices >= 0) & (ml_col_indices < self.num_mla_x) & \
                             (ml_row_indices >= 0) & (ml_row_indices < self.num_mla_y)
            valid_rays_mla = current_rays[valid_mla_mask]
            valid_ml_col_indices = ml_col_indices[valid_mla_mask]
            valid_ml_row_indices = ml_row_indices[valid_mla_mask]

            if len(valid_rays_mla) == 0:
                continue

            mla_flat_indices = valid_ml_row_indices * self.num_mla_x + valid_ml_col_indices

            if np.max(mla_flat_indices) >= len(self.mla_centers):
                valid_index_mask = mla_flat_indices < len(self.mla_centers)
                valid_rays_mla = valid_rays_mla[valid_index_mask]
                mla_flat_indices = mla_flat_indices[valid_index_mask]
                if len(valid_rays_mla) == 0:
                    continue

            ml_centers_for_rays = self.mla_centers[mla_flat_indices]

            current_rays_after_mla = self._trace_microlens(
                valid_rays_mla,
                ml_centers_for_rays[:, 0],
                ml_centers_for_rays[:, 1]
            )

            final_rays = self._propagate_free_space(current_rays_after_mla, Z_mla_ccd)
            ray_x4 = final_rays[:, 0]
            ray_y4 = final_rays[:, 1]

            up_pixels, vp_pixels = self._map_to_ccd_pixels(ray_x4, ray_y4)
            np.add.at(light_field_image, (vp_pixels, up_pixels), w)

        counts = light_field_image
        if normalize:
            max_val = counts.max()
            if max_val > 0:
                out = (counts / max_val) * 255.0
            else:
                out = counts.copy()
            out = out.astype(np.uint8)
        else:
            out = counts

        if return_counts:
            return out, counts
        return out


# -----------------------------------------------------------------------------
# 光场数据处理与可视化函数
# -----------------------------------------------------------------------------

def generate_sai_array(light_field_image, camera_instance):
    px_per_mla_y = camera_instance.pixels_per_microlens_y
    px_per_mla_x = camera_instance.pixels_per_microlens_x

    num_mla_y = camera_instance.num_mla_y
    num_mla_x = camera_instance.num_mla_x

    print(f"\n--- 正在生成 {px_per_mla_x * px_per_mla_y} 个子光圈图像(SAI) ---")
    valid_height = num_mla_y * px_per_mla_y
    valid_width = num_mla_x * px_per_mla_x
    total_height, total_width = light_field_image.shape
    start_row = (total_height - valid_height) // 2
    start_col = (total_width - valid_width) // 2
    end_row = start_row + valid_height
    end_col = start_col + valid_width
    cropped_lf_image = light_field_image[start_row:end_row, start_col:end_col]

    print(f"原始光场图像尺寸: {light_field_image.shape}")
    print(f"为保证尺寸一致性，居中裁剪至: {cropped_lf_image.shape}")

    # 创建一个4D数组来存储所有的SAI，形状为 (s, t, u, v)
    sai_array = np.zeros((px_per_mla_y, px_per_mla_x, num_mla_y, num_mla_x), dtype=np.uint8)

    for view_y in range(px_per_mla_y): # s
        for view_x in range(px_per_mla_x): # t
            sai = cropped_lf_image[view_y::px_per_mla_y, view_x::px_per_mla_x]
            sai_array[view_y, view_x, :, :] = sai

    print(f"SAI 4D数组生成完毕, 形状 (s,t,u,v): {sai_array.shape}")
    return sai_array


def show_sai_grid(sai_array, num_views_to_show=5):

    px_per_mla_y, px_per_mla_x, _, _ = sai_array.shape
    fig, axes = plt.subplots(num_views_to_show, num_views_to_show,
                             figsize=(num_views_to_show * 2, num_views_to_show * 2))
    fig.suptitle(f'子光圈图像 (SAI) 网格视图', fontsize=16)
    view_indices_y = np.linspace(0, px_per_mla_y - 1, num_views_to_show, dtype=int)
    view_indices_x = np.linspace(0, px_per_mla_x - 1, num_views_to_show, dtype=int)

    for i, view_y in enumerate(view_indices_y):
        for j, view_x in enumerate(view_indices_x):
            sub_aperture_image = sai_array[view_y, view_x, :, :]
            ax = axes[i, j]
            ax.imshow(sub_aperture_image, cmap='gray')
            ax.set_title(f'视角: (t={view_x}, s={view_y})')
            ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def generate_and_show_epi_stack(sai_4d_array):
    """
    接收4D SAI数组(s,t,u,v)，转置为EPI堆栈，并提供交互式滑块来查看任意EPI。
    """
    print("\n--- 正在生成 EPI 堆栈并启动交互式查看器 ---")
    
    # 原始SAI堆栈形状为 (s, t, u, v)
    s_max, t_max, u_max, v_max = [dim - 1 for dim in sai_4d_array.shape]

    # --- 1. 转置为水平EPI堆栈 (u, s, v, t) ---
    # 原始轴: s(0), t(1), u(2), v(3) -> 目标轴: u(2), s(0), v(3), t(1)
    horizontal_epi_stack = sai_4d_array.transpose(2, 0, 3, 1)
    print(f"水平EPI堆栈形状 (u,s,v,t): {horizontal_epi_stack.shape}")

    # --- 2. 转置为垂直EPI堆栈 (v, t, u, s) ---
    # 原始轴: s(0), t(1), u(2), v(3) -> 目标轴: v(3), t(1), u(2), s(0)
    vertical_epi_stack = sai_4d_array.transpose(3, 1, 2, 0)
    print(f"垂直EPI堆栈形状 (v,t,u,s): {vertical_epi_stack.shape}")
    
    # --- 3. 创建交互式绘图 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plt.subplots_adjust(bottom=0.25) # 为滑块留出空间
    fig.suptitle('交互式EPI堆栈查看器', fontsize=16)

    # 初始显示中心的EPI
    u_init = u_max // 2
    s_init = s_max // 2
    v_init = v_max // 2
    t_init = t_max // 2

    # 显示初始的水平EPI (v,t)
    # 注意：imshow会把第一个维度当成Y轴，所以我们需要转置
    h_epi_img = horizontal_epi_stack[u_init, s_init, :, :].T
    im_h = axes[0].imshow(h_epi_img, cmap='gray', aspect='auto')
    axes[0].set_title(f'水平 EPI @ (u={u_init}, s={s_init})')
    axes[0].set_xlabel('空间轴 v (微透镜列)')
    axes[0].set_ylabel('角度轴 t (像素列)')

    # 显示初始的垂直EPI (u,s)
    v_epi_img = vertical_epi_stack[v_init, t_init, :, :].T
    im_v = axes[1].imshow(v_epi_img, cmap='gray', aspect='auto')
    axes[1].set_title(f'垂直 EPI @ (v={v_init}, t={t_init})')
    axes[1].set_xlabel('空间轴 u (微透镜行)')
    axes[1].set_ylabel('角度轴 s (像素行)')
    
    # --- 4. 创建滑块 ---
    # 水平EPI的滑块
    ax_slider_u = plt.axes([0.15, 0.15, 0.3, 0.03])
    ax_slider_s = plt.axes([0.15, 0.1, 0.3, 0.03])
    slider_u = Slider(ax_slider_u, '选择 u (行)', 0, u_max, valinit=u_init, valstep=1)
    slider_s = Slider(ax_slider_s, '选择 s (视角)', 0, s_max, valinit=s_init, valstep=1)

    # 垂直EPI的滑块
    ax_slider_v = plt.axes([0.6, 0.15, 0.3, 0.03])
    ax_slider_t = plt.axes([0.6, 0.1, 0.3, 0.03])
    slider_v = Slider(ax_slider_v, '选择 v (列)', 0, v_max, valinit=v_init, valstep=1)
    slider_t = Slider(ax_slider_t, '选择 t (视角)', 0, t_max, valinit=t_init, valstep=1)

    # --- 5. 定义滑块更新函数 ---
    def update(val):
        # 更新水平EPI
        u = int(slider_u.val)
        s = int(slider_s.val)
        new_h_epi = horizontal_epi_stack[u, s, :, :].T
        im_h.set_data(new_h_epi)
        axes[0].set_title(f'水平 EPI @ (u={u}, s={s})')

        # 更新垂直EPI
        v = int(slider_v.val)
        t = int(slider_t.val)
        new_v_epi = vertical_epi_stack[v, t, :, :].T
        im_v.set_data(new_v_epi)
        axes[1].set_title(f'垂直 EPI @ (v={v}, t={t})')

        fig.canvas.draw_idle()

    # 绑定更新函数
    slider_u.on_changed(update)
    slider_s.on_changed(update)
    slider_v.on_changed(update)
    slider_t.on_changed(update)

    plt.show()
