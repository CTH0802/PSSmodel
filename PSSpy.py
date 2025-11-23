import sys
import os
import numpy as np
import scipy.constants as spc
import astropy.constants as astroc
from scipy.ndimage import distance_transform_edt
from functools import lru_cache
from numba import njit, prange

# ===================================================================
# 1. 基礎工具 (Utilities)
# ===================================================================
M_SUN_KG = 1.98847e30  # kg

def gaussian(x, mu, sig):
    return np.exp(-np.power((x - mu) / sig, 2.0) / 2)

def time_to_deg(ra_time):
    """將 'HH:MM:SS.SSS' 轉成度數"""
    h, m, s = [float(i) for i in ra_time.split(':')]
    return (h + m/60 + s/3600) * 15 # RA 1小時 = 15度

def pixel_to_arcsec(xpix, ypix, xcenpix, ycenpix, im):
    """
    (註：此函數在您提供的腳本中未被使用，但作為基礎工具保留)
    """
    delx, dely = im.delx*3600., im.dely*3600.
    x_arcsec = (xpix - xcenpix) * delx
    y_arcsec = (ypix - ycenpix) * dely
    return x_arcsec, y_arcsec

def spherical_coords(x,y):
    ''' Converts cartesian coordinates (x,y) to polar coordinates.'''
    r = np.sqrt((x**2)+(y**2)) # total distance from center
    theta = np.arctan2(y,x)  # Angle wrt x-axis, in radians
    return(r,theta)

def radius_sn(singal_noise, mim_singal=3, r_0=1.2):
    """
    (註：此函數是 grow_region 的輔助函數)
    """
    radius = r_0 * (1 - np.exp(mim_singal - singal_noise))
    return np.max(radius, 0)

def spheres(max_radius):
    """
    預先建立半徑從1到max_radius的所有球形座標。
    (註：此函數是 grow_region 的輔助函數)
    """
    sphere_dict = {}
    for r in range(1, max_radius + 1):
        x, y, z = np.meshgrid(np.arange(-r, r + 1),
                              np.arange(-r, r + 1),
                              np.arange(-r, r + 1))
        mask = (x**2 + y**2 + z**2 <= r**2) & (x**2 + y**2 + z**2 > 0)
        sphere_dict[r] = (x[mask], y[mask], z[mask])
    return sphere_dict

def velocity_to_channel_index(v_lsr, header, nz=None):
    """
    將 LSR 速度 (km/s) 轉成 cube 的 channel index (0-based).
    streamercom_v 是相對 V_sys 的速度，所以先加回 Local_Standard_Velocity
    為了避免循環 import，這裡直接定義 velocity_to_channel_index
    ----
    v_lsr : float or array-like
        LSR velocity [km/s]，可以是 scalar 或陣列。
    header : astropy.io.fits.Header
        含有 CRVAL3 / CRPIX3 / CDELT3 的 header。
    nz : int, optional
        頻道數 (cube.shape[0])，若提供會將 index clip 在 [0, nz-1]。
    ----
    根據 FITS WCS：
        v = CRVAL3 + (i + 1 - CRPIX3) * CDELT3
    反解得到 0-based index：
        i = (v - CRVAL3) / CDELT3 + CRPIX3 - 1
    """
    CRVAL3 = float(header["CRVAL3"])
    CRPIX3 = float(header["CRPIX3"])
    CDELT3 = float(header["CDELT3"])
    v_pix = (v_lsr - CRVAL3) / CDELT3 + CRPIX3 - 1.0  # 0-based index
    if nz is not None:
        v_pix = np.clip(v_pix, 0, nz - 1)
    return v_pix

# ===================================================================
# 2. 核心物理模型 (Physics Core)
# ===================================================================

def Omega_ref(radius_ref_au, Mass_star):
    """
    (註：此函數是 PSS_model 的輔助函數)
    """
    r_m = radius_ref_au * spc.astronomical_unit
    M_kg = Mass_star * M_SUN_KG # 改用 SciPy 的 M⊙
    return np.sqrt(spc.G * M_kg / r_m) / r_m

def PSS_model(Theta_zero, Phi_zero, Inclination, T_Myr, omega, 
              solar_mass, radius_in_au=1.5e3, radius_out_au=1e4, resolution=200,
              scale='log', log_power=1.5):
    """
    產生 3D 的流線模型 (PSS model)。
    (註：依賴 Omega_ref)
    """

    # ... (函數內部程式碼保持不變) ...
    
    theta_value = Theta_zero
    
    #######Unit conversion#######
    T_s = T_Myr * 1e6 * spc.year #Time (s)
    radius_in_m = radius_in_au * spc.astronomical_unit #Streamer edge (m)
    radius_out_m = radius_out_au * spc.astronomical_unit #Streamer edge (m)

    # --------------------------------------------------------------------
    # 根據選擇的 scale 來產生 streamline_radius
    # --------------------------------------------------------------------
    if scale == 'linear':
        streamline_radius = np.linspace(radius_in_m, radius_out_m, resolution)
    elif scale == 'log':
        s = np.linspace(0.0, 1.0, resolution)
        s_transformed = s ** log_power
        log_r_in = np.log10(radius_in_m)
        log_r_out = np.log10(radius_out_m)
        log_r = log_r_in + (log_r_out - log_r_in) * s_transformed
        streamline_radius = 10 ** log_r
    else:
        raise ValueError(f"無效的 scale: '{scale}'。請選擇 'linear' 或 'log'。")
    # --------------------------------------------------------------------
    
    c_s = 200 #m/s
    
    r_s = c_s * T_s #m
    V_infall = - np.sqrt(2 * spc.G * solar_mass * M_SUN_KG / streamline_radius) - 3.3 * c_s #m/s
    alpha = -1/3   #-1/3 ~ 1
    Omega_s = Omega_ref(r_s / spc.astronomical_unit, solar_mass) * omega #Keplerian velocity (radian)
    Omega = Omega_s * ((streamline_radius / r_s) ** (-2) + (streamline_radius / r_s) ** (alpha - 1))
    phi_value = Phi_zero + T_s * Omega_s * np.sqrt(2 * c_s ** 3 * T_s / (spc.G * solar_mass * M_SUN_KG)) * ((streamline_radius / r_s) ** (-1/2) + (streamline_radius / r_s) ** (alpha)) 
    # m_0 = spc.G * solar_mass * M_SUN_KG / c_s ** 2 / r_s
    # Omega = np.sqrt(m_0) / T_s * Omega_s * ((streamline_radius / r_s) ** (-2) + (streamline_radius / r_s) ** (alpha - 1))
    # phi_value = Phi_zero + 3 * Omega_s * np.sqrt(m_0) * ((streamline_radius / r_s) ** (-1/2) + (streamline_radius / r_s) ** (alpha)) 
    velocity_r = streamline_radius * Omega * np.sin(theta_value)

    ######Streamer coordinate######
    x = streamline_radius * np.sin(theta_value) * np.cos(phi_value)
    y = streamline_radius * np.sin(theta_value) * np.sin(phi_value)
    z = streamline_radius * np.cos(theta_value)
    ######Rotated by X-axis Streamer coordinate#######
    x_rotate = x                                                    # x
    y_rotate = y * np.cos(Inclination) + z * np.sin(Inclination)
    z_rotate = - y * np.sin(Inclination) + z * np.cos(Inclination)  # z
    ######r Velocity######
    u_r = V_infall * np.sin(theta_value) * np.cos(phi_value)
    v_r = V_infall * np.sin(theta_value) * np.sin(phi_value)
    w_r = V_infall * abs(np.cos(theta_value))
    ######Velocity vector######
    u = - velocity_r * np.sin(phi_value)
    v = velocity_r * np.cos(phi_value)
    w = np.zeros_like(velocity_r)

    u += u_r
    v += v_r
    w += w_r
    ######Rotated by X-axis Velocity######
    u_rotate = u
    v_rotate = v * np.cos(Inclination) + w * np.sin(Inclination) # v_y
    w_rotate = - v * np.sin(Inclination) + w * np.cos(Inclination)

    ######Unit conversions######
    x_rotate /= spc.astronomical_unit
    y_rotate /= spc.astronomical_unit
    z_rotate /= spc.astronomical_unit

    u_rotate /= 1e3
    v_rotate /= 1e3
    w_rotate /= 1e3

    return x_rotate, y_rotate, z_rotate, u_rotate, v_rotate, w_rotate #final 刪掉只保留x, z, v

def arrow_line(times, arrow_resolution, interval_of_arrows, x, y, z, u, v, w, x_rotate, y_rotate, z_rotate, u_rotate, v_rotate, w_rotate):
    """
    (註：此函數在您提供的腳本中未被使用，但作為 PSS_model 的繪圖輔助工具保留)
    """
    ######Streamer coordinate for Arrow######
    x_arrowline = x + u * times
    y_arrowline = y + v * times
    z_arrowline = z + w * times
    ######Rotated by X-axis Streamer coordinate for Arrow######
    x_arrowline_rotate = x_rotate + u_rotate * times
    y_arrowline_rotate = y_rotate + v_rotate * times
    z_arrowline_rotate = z_rotate + w_rotate * times
    ######Create new array######
    x_arrowline_interval = np.array([])
    y_arrowline_interval = np.array([])
    z_arrowline_interval = np.array([])
    x_arrowline_rotate_interval = np.array([])
    y_arrowline_rotate_interval = np.array([])
    z_arrowline_rotate_interval = np.array([])
    u_arrow = np.array([])
    v_arrow = np.array([])
    w_arrow = np.array([])
    u_arrow_rotate = np.array([])
    v_arrow_rotate = np.array([])
    w_arrow_rotate = np.array([])
    for i in range(arrow_resolution):
        ######Streamer coordinate for Arrow######
        x_arrowline_interval = np.append(x_arrowline_interval, x_arrowline[interval_of_arrows * i])
        y_arrowline_interval = np.append(y_arrowline_interval, y_arrowline[interval_of_arrows * i])
        z_arrowline_interval = np.append(z_arrowline_interval, z_arrowline[interval_of_arrows * i])
        
        u_arrow = np.append(u_arrow, u[interval_of_arrows * i])
        v_arrow = np.append(v_arrow, v[interval_of_arrows * i])
        w_arrow = np.append(w_arrow, w[interval_of_arrows * i])
        
        ######Rotated by X-axis Streamer coordinate for Arrow######
        x_arrowline_rotate_interval = np.append(x_arrowline_rotate_interval, x_arrowline_rotate [interval_of_arrows * i])
        y_arrowline_rotate_interval = np.append(y_arrowline_rotate_interval, y_arrowline_rotate [interval_of_arrows * i])
        z_arrowline_rotate_interval = np.append(z_arrowline_rotate_interval, z_arrowline_rotate [interval_of_arrows * i])
        
        u_arrow_rotate = np.append(u_arrow_rotate, u_rotate [interval_of_arrows * i])
        v_arrow_rotate = np.append(v_arrow_rotate, v_rotate [interval_of_arrows * i])
        w_arrow_rotate = np.append(w_arrow_rotate, w_rotate [interval_of_arrows * i])
    return x_arrowline_interval, y_arrowline_interval, z_arrowline_interval, u_arrow, v_arrow, w_arrow, x_arrowline_rotate_interval, y_arrowline_rotate_interval, z_arrowline_rotate_interval, u_arrow_rotate, v_arrow_rotate, w_arrow_rotate

# ===================================================================
# 3. 數據處理與遮罩 (Data Processing & Masking)
# ===================================================================

def circular_mask(shape, center, radius):
    """
    建立一個圓形 mask。
    
    Parameters:
    - shape: (ny, nx) 影像大小
    - center: (y, x)，圓心位置（像素座標）
    - radius: 半徑（pixel）

    Returns:
    - mask: 2D boolean array，圓形區域為 True，其餘為 False
    """
    Y, X = np.ogrid[:shape[0], :shape[1]]
    dist_from_center = np.sqrt((X - center[1])**2 + (Y - center[0])**2)
    mask = dist_from_center >= radius
    return mask

def grow_region(data, init_points, sigma_value, r_0=4, sigma_thresh=3, max_iter=1000):
    """
    依據初始點與像素強度，圈出擴展區域。
    (註：依賴 radius_sn 和 spheres)
    
    data: 3D array, 觀測影像資料
    init_points: list of (z, y, x), 初始點
    sigma_thresh: 門檻，像素強度需高於 mean + sigma_thresh * std
    return: 3D boolean mask of selected region
    """
    data = data / sigma_value
    region_mask = np.zeros_like(data, dtype=bool)
    depth, height, width = data.shape
    threshold = sigma_thresh

    to_check = set(init_points)
    visited = set(init_points)
    sphere_dict = spheres(max_radius=r_0)
    # for pt in init_points:
    #     print(f"Init {pt}: {data[pt[0], pt[1]]:.2f} > {threshold:.2f}?")

    for _ in range(max_iter):
        new_points = set()
        for z, y, x in to_check:
            if (0 <= z < depth) and (0 <= y < height) and (0 <= x < width):
                if data[z, y, x] > threshold and not region_mask[z, y, x]:
                    region_mask[z, y, x] = True
                    
                    # print(f"Accepted: x={x}, y={y}, value={data[x, y]:.2f}")
                    radius = radius_sn(data[z, y, x], mim_singal=sigma_thresh, r_0=4)
                    int_radius = int(np.ceil(radius))
                    
                    if int_radius not in sphere_dict:
                        continue  # 如果超過預設最大半徑就跳過
                    # print(f"Sigma level {sigma_level}, checking {radius}x{radius} region")
                    dz_array, dy_array, dx_array = sphere_dict[int_radius]
                    
                    for i in range(len(dx_array)):
                        nz, ny, nx = z + dz_array[i], y + dy_array[i], x + dx_array[i]
                        if (0 <= nz < depth) and (0 <= ny < width) and (0 <= nx < height):
                            if (nz, ny, nx) not in visited:
                                new_points.add((nz, ny, nx))
                                visited.add((nz, ny, nx))
        if not new_points:
            break
        to_check = new_points

    return region_mask

def get_bounding_box(x_pix, z_pix, v_pix, buffer, v_buffer, cube_shape):
    """
    根據模型線的像素座標，自動計算一個帶緩衝的邊界框。
    """
    if len(x_pix) == 0:
        return None
        
    nv, nz, nx = cube_shape
    
    x_min, x_max = np.min(x_pix), np.max(x_pix)
    z_min, z_max = np.min(z_pix), np.max(z_pix)
    v_min, v_max = np.min(v_pix), np.max(v_pix)
    
    # x 軸邊界 (使用 max/min 來確保不超出 [0, nx-1])
    x_min_bound = max(0, x_min - buffer)
    x_max_bound = min(nx - 1, x_max + buffer)
    
    # z 軸邊界 (使用 max/min 來確保不超出 [0, nz-1])
    z_min_bound = max(0, z_min - buffer)
    z_max_bound = min(nz - 1, z_max + buffer)
    
    # v 軸邊界 (使用 max/min 來確保不超出 [0, nv-1])
    v_min_bound = max(0, v_min - v_buffer)
    v_max_bound = min(nv - 1, v_max + v_buffer)
    
    bound = ([v_min_bound, v_max_bound],
            [z_min_bound, z_max_bound],
            [x_min_bound, x_max_bound])
    return bound


# ===================================================================
# 4. 誤差/距離計算 (Error / Distance Functions)
# ===================================================================

@njit
def nearest_weighted_distance_numba(x, z, v, model_x, model_z, model_v, v_weight):
    """
    numba 版本：回傳距離平方 d^2 和最近點 index
    """
    dmin = 1e30 
    idx_min = -1
    n = model_x.shape[0]

    for j in range(n):
        dx = x - model_x[j]
        dz = z - model_z[j]
        dv = v - model_v[j]
        dist_sq = dx*dx + dz*dz + v_weight * dv*dv

        if dist_sq < dmin:
            dmin = dist_sq
            idx_min = j

    return dmin, idx_min

def error_function(params, streamercom_x, streamercom_z, streamercom_v,
                   weight_v, T_Myr, omega, Inclination, solar_mass, scale, log_power):
    """
    計算 PSS_model 與數據點的誤差，使用最近鄰匹配來尋找最佳對應點。

    這裡同時：
    - 使用統一的加權距離 sqrt( dx^2 + dz^2 + weight_v * dv^2 ) 作為主誤差 (「AU-metric」)，
      供 grid / MCMC 最小化。
    - 紀錄分開的 RMS(position) 與 RMS(velocity)，方便之後轉換與診斷：
        error_function.last_pos_rmse      (in AU)
        error_function.last_vel_rmse      (in km/s)
        error_function.last_eq_vel_rmse   (equivalent km/s from total AU-metric)
    """

    Theta_zero, Phi_zero = params
    num_points = len(streamercom_x)
    if num_points == 0:
        return np.inf

    # --- unified metric using nearest neighbor ---
    d_list = []
    for i in range(num_points):
        radius_in_au = max(1.0, np.sqrt(streamercom_x[i] ** 2 + streamercom_z[i] ** 2) * 0.5)
        radius_out_au = radius_in_au * 30.0
        x_model, y_model, z_model, u_model, v_model, w_model = PSS_model(
            Theta_zero, Phi_zero, Inclination, T_Myr, omega,
            solar_mass, radius_in_au, radius_out_au,
            resolution=100, scale=scale, log_power=log_power
        )
        d_min, _ = nearest_weighted_distance_numba(
            streamercom_x[i],
            streamercom_z[i],
            streamercom_v[i],
            x_model,
            z_model,
            v_model,
            weight_v,
        )
        d_list.append(d_min)

    # RMS of distances
    d_arr = np.asarray(d_list, dtype=float)
    total_error = np.sqrt(np.mean(d_arr))

    return total_error

@lru_cache(maxsize=None)
def build_ball_offsets(max_r):
    """
    建立「距離球殼」的 offsets：
      - offsets: (N, 3)，每列是 (dv, dz, dx)
      - shell_k: (N,)，每個 offset 對應的殼層 index (1..max_r)

    定義：
      dist = sqrt(dv^2 + dz^2 + dx^2)
      1 <= ceil(dist) <= max_r 的點會被保留
    """
    coords = np.arange(-max_r, max_r + 1)
    dv, dz, dx = np.meshgrid(coords, coords, coords, indexing="ij")
    dist = np.sqrt(dv*dv + dz*dz + dx*dx)

    # 只保留 dist <= max_r 的點
    mask = dist <= max_r
    dv_sel = dv[mask]
    dz_sel = dz[mask]
    dx_sel = dx[mask]
    dist_sel = dist[mask]

    # 殼層 index：ceil(dist)，中心點 dist=0 → 設成 1 層
    shell_k = np.ceil(dist_sel).astype(np.int16)
    shell_k[dist_sel == 0.0] = 1

    offsets = np.column_stack([dv_sel, dz_sel, dx_sel]).astype(np.int16)
    return offsets, shell_k

@njit(parallel=True)
def stamp_shell_cube_numba(shell_cube, v_line, z_line, x_line,
                           dv_off, dz_off, dx_off, shell_k):
    nv, nz, nx = shell_cube.shape
    Npts = len(v_line)
    Noff = len(shell_k)

    for i in prange(Npts):   # ← 平行化這層 loop
        v0 = v_line[i]
        z0 = z_line[i]
        x0 = x_line[i]

        for j in range(Noff):   # ← 這層也可以平行化，但要注意寫入衝突
            vv = v0 + dv_off[j]
            zz = z0 + dz_off[j]
            xx = x0 + dx_off[j]

            if (0 <= vv < nv) and (0 <= zz < nz) and (0 <= xx < nx):
                k = shell_k[j]
                cur = shell_cube[vv, zz, xx]
                if cur < 0 or cur > k:
                    shell_cube[vv, zz, xx] = k

    return shell_cube

def apply_shell_ball_to_line(cube_shape, v_line, z_line, x_line, max_r):
    shell_cube = np.full(cube_shape, -1, dtype=np.int16)

    offsets, shell_k = build_ball_offsets(max_r)
    dv_off = offsets[:, 0]
    dz_off = offsets[:, 1]
    dx_off = offsets[:, 2]

    # 使用 numba JIT 加速 stamping
    shell_cube = stamp_shell_cube_numba(shell_cube,
                                        v_line, z_line, x_line,
                                        dv_off, dz_off, dx_off,
                                        shell_k)

    return shell_cube

# ===================================================================
def shell_error_from_cube(
    data_cube,
    Theta_zero, Phi_zero, Inclination, T_Myr, omega,
    pa_rad, dx_au, header, Local_Standard_Velocity,
    max_dist_value,
    M_star, radius_in_au, radius_out_au,
    scale, log_power,
):
    """
    使用 build_shell_distance_cube_sequential 所建立的「殼層距離」cube，
    搭配對應的 data cube，計算「以資料強度加權的殼層平均距離」作為誤差指標。

    search_bound : None 或 ([v_min,v_max],[z_min,z_max],[x_min,x_max])
        若非 None，僅在此子區域內建立 / 使用殼層距離。
    """
    cube_shape = data_cube.shape
    nv, nz, nx = cube_shape

    if not np.any(np.isfinite(data_cube) & (data_cube != 0.0)):
        return np.inf

    # 1) 建立 model line (物理單位：AU & km/s)
    x_m, y_m, z_m, u_m, v_m, w_m = PSS_model(
        Theta_zero, Phi_zero, Inclination, T_Myr, omega,
        M_star,
        radius_in_au, radius_out_au,
        resolution=100,
        scale=scale,
        log_power=log_power,
    )

    # 2) 轉換為像素 / 頻道座標
    im_cy = float(header["CRPIX2"]) - 1.0  # 0-based
    im_cx = float(header["CRPIX1"]) - 1.0

    x_rot = x_m * np.cos(pa_rad) - z_m * np.sin(pa_rad)
    z_rot = x_m * np.sin(pa_rad) + z_m * np.cos(pa_rad)

    x_pix_line = np.round(x_rot / dx_au + im_cx).astype(int)
    z_pix_line = np.round(z_rot / dx_au + im_cy).astype(int)

    if Local_Standard_Velocity is None:
        Local_Standard_Velocity = float(header.get("LSRVEL", 0.0))
    v_lsr = v_m + Local_Standard_Velocity
    v_pix_line = velocity_to_channel_index(v_lsr, header, nz=nv)
    v_pix_line = np.round(v_pix_line).astype(int)

    # 3) 先裁掉落在 cube 之外的 model 點
    valid_model = (
        (x_pix_line >= 0) & (x_pix_line < nx) &
        (z_pix_line >= 0) & (z_pix_line < nz) &
        (v_pix_line >= 0) & (v_pix_line < nv)
    )
    x_pix_line = x_pix_line[valid_model]
    z_pix_line = z_pix_line[valid_model]
    v_pix_line = v_pix_line[valid_model]

    if x_pix_line.size == 0:
        return np.inf

    # 4) 由 model line 建立殼層距離 cube（只在 bound 裡 stamp）
    shell_cube = apply_shell_ball_to_line(
        cube_shape,
        v_pix_line,   # 正確
        z_pix_line,   # 正確
        x_pix_line,   # 正確
        max_dist_value,
    )
    
    if shell_cube.shape != data_cube.shape:
        raise ValueError("shell_cube 與 data_cube 形狀必須一致")

    valid = (
        (shell_cube >= 1) &
        (shell_cube <= max_dist_value) &
        (data_cube > 0)
    )

    s = shell_cube[valid].astype(float)
    w = data_cube[valid].astype(float)

    if w.sum() == 0:
        return np.inf

    weighted_mean_shell = np.sum(s * w) / np.sum(w)
    return float(weighted_mean_shell)

@njit(parallel=True)
def fill_distance_cube_core(distance_cube,
                            model_x, model_z, model_v,
                            v_weight, max_dist_value,
                            v_min, v_max, z_min, z_max, x_min, x_max):
    """
    在給定的邊界內，對每一個 voxel 計算最近模型點的距離，
    結果寫到 distance_cube 裡。
    """
    for vv in prange(v_min, v_max + 1):
        for zz in range(z_min, z_max + 1):
            for xx in range(x_min, x_max + 1):
                d2, _ = nearest_weighted_distance_numba(
                    float(xx), float(zz), float(vv),
                    model_x, model_z, model_v,
                    v_weight,
                )
                d = np.sqrt(d2)
                if d <= max_dist_value:
                    distance_cube[vv, zz, xx] = d
                # 否則保持原本的 -1

def grow_distance_cube_bounded(
    cube_shape,
    Theta_zero, Phi_zero, Inclination, T_Myr, omega,
    pa_rad, dx_au, header, Local_Standard_Velocity,
    v_weight, max_dist_value,
    M_star, radius_in_au, radius_out_au,
    scale, log_power,
    bound=None,
):
    """
    使用與 nearest_weighted_distance 相同的加權距離：
        d^2 = dx^2 + dz^2 + v_weight * dv^2

    在 (v, z, x) 的子區域內，對每一個 voxel 找出距離模型線最近的點，
    把「根號後的距離」存成 distance_cube：
        distance_cube[v,z,x] = d_min  (若 d_min <= max_dist_value，否則設為 -1)

    參數中的座標一律以 pixel / channel 為單位。
    """

    nv, nz, nx = cube_shape
    distance_cube = np.full(cube_shape, -1.0, dtype=np.float32)

    # ---------- 1. 產生模型流線 (物理單位：AU, km/s) ----------
    x_m, y_m, z_m, u_m, v_m, w_m = PSS_model(
        Theta_zero, Phi_zero, Inclination, T_Myr, omega,
        M_star,
        radius_in_au, radius_out_au,
        resolution=100,
        scale=scale,
        log_power=log_power,
    )

    # ---------- 2. 轉成 cube 的 pixel / channel 座標 ----------
    im_center_y = float(header["CRPIX2"]) - 1.0  # 0-based
    im_center_x = float(header["CRPIX1"]) - 1.0

    # 旋轉到 image frame
    x_rot = x_m * np.cos(pa_rad) - z_m * np.sin(pa_rad)
    z_rot = x_m * np.sin(pa_rad) + z_m * np.cos(pa_rad)

    # 轉成像素座標 (x_pix, z_pix)
    x_pix = x_rot / dx_au + im_center_x   # 先保留為 float
    z_pix = z_rot / dx_au + im_center_y

    # 頻道座標 v_pix
    v_LSR = v_m + Local_Standard_Velocity
    v_pix = velocity_to_channel_index(v_LSR, header, nz=nv)   # float

    # ---------- 3. 只留下落在 cube 內的 model 點 ----------
    valid_mask = (
        (x_pix >= 0) & (x_pix < nx) &
        (z_pix >= 0) & (z_pix < nz) &
        (v_pix >= 0) & (v_pix < nv)
    )
    x_pix = x_pix[valid_mask]
    z_pix = z_pix[valid_mask]
    v_pix = v_pix[valid_mask]

    if x_pix.size == 0:
        # 沒有任何模型點落在 cube 內，直接回傳全 -1
        return distance_cube

    # 方便後面計算，轉成 numpy array（float64）
    model_x = np.asarray(x_pix, dtype=float)
    model_z = np.asarray(z_pix, dtype=float)
    model_v = np.asarray(v_pix, dtype=float)

    # ---------- 4. 決定實際要掃描的 (v,z,x) 邊界 ----------
    if bound is not None:
        v_bound, z_bound, x_bound = bound
        v_min, v_max = int(v_bound[0]), int(v_bound[1])
        z_min, z_max = int(z_bound[0]), int(z_bound[1])
        x_min, x_max = int(x_bound[0]), int(x_bound[1])
    else:
        v_min, z_min, x_min = 0, 0, 0
        v_max, z_max, x_max = nv - 1, nz - 1, nx - 1

    # 再跟 model 本身的位置 ± max_dist_value 取交集，避免掃太大
    pad = int(np.ceil(max_dist_value))
    v_min_model = int(np.floor(model_v.min())) - pad
    v_max_model = int(np.ceil(model_v.max())) + pad
    z_min_model = int(np.floor(model_z.min())) - pad
    z_max_model = int(np.ceil(model_z.max())) + pad
    x_min_model = int(np.floor(model_x.min())) - pad
    x_max_model = int(np.ceil(model_x.max())) + pad

    v_min = max(0,          max(v_min, v_min_model))
    z_min = max(0,          max(z_min, z_min_model))
    x_min = max(0,          max(x_min, x_min_model))
    v_max = min(nv - 1,     min(v_max, v_max_model))
    z_max = min(nz - 1,     min(z_max, z_max_model))
    x_max = min(nx - 1,     min(x_max, x_max_model))

    if (v_min > v_max) or (z_min > z_max) or (x_min > x_max):
        # 邊界完全沒有相交，回傳全 -1
        return distance_cube

    fill_distance_cube_core(
        distance_cube,
        model_x, model_z, model_v,
        float(v_weight), float(max_dist_value),
        int(v_min), int(v_max),
        int(z_min), int(z_max),
        int(x_min), int(x_max),
    )
    return distance_cube


# ===================================================================
# 5. MCMC 框架 (MCMC Framework)
# ===================================================================
def log_likelihood_shell(
    params,
    data_cube,
    pa_rad,
    dx_au,
    header,
    Local_Standard_Velocity,
    max_dist_value,
    M_star,
    radius_in_au,
    radius_out_au,
    scale,
    log_power,
):
    """
    基於「殼層距離 cube」的 log-likelihood（實驗性）

    search_bound 若非 None，格式為 ([v_min,v_max],[z_min,z_max],[x_min,x_max])，
    僅在該子區域內 stamp 殼層並計算殼層誤差。
    """
    Theta_zero, Phi_zero, Inclination, T_Myr, omega = params

    if data_cube is None:
        return -np.inf
    data_arr = np.asarray(data_cube, dtype=float)
    if not np.any(np.isfinite(data_arr) & (data_arr != 0.0)):
        return -np.inf

    E = shell_error_from_cube(
        data_arr,
        Theta_zero, Phi_zero, Inclination, T_Myr, omega,
        pa_rad, dx_au, header, Local_Standard_Velocity,
        max_dist_value,
        M_star, radius_in_au, radius_out_au,
        scale, log_power,
    )
    logL = -np.log10(E)
    return logL

def log_prior_shell(params, prior_ranges):
    """
    計算對數先驗值。
    """
    Theta0, Phi0, Incl, T, Omega = params
    
    # 檢查參數是否在先驗範圍內
    if not (
        prior_ranges["Theta zero"][0] < Theta0 < prior_ranges["Theta zero"][1]
        and in_phi_range(Phi0, *prior_ranges["Phi zero"])
        and prior_ranges["Inclination"][0] < Incl < prior_ranges["Inclination"][1]
        and prior_ranges["Time"][0] < T < prior_ranges["Time"][1]
        and prior_ranges["Omega"][0] < Omega < prior_ranges["Omega"][1]
    ):
        return -np.inf # 如果超出範圍，對數先驗為負無窮
    
    return 0.0 # 均勻先驗，對數值為 0

def log_posterior_shell(
    params,
    data_cube,
    priors,
    pa_rad,
    dx_au,
    header,
    Local_Standard_Velocity,
    max_dist_value,
    M_star,
    radius_in_au,
    radius_out_au,
    scale,
    log_power,
):
    """
    殼層距離版的 posterior：
      log P(θ | data) = log Prior(θ) + log L_shell(data | θ)
    """
    # 1. 先驗
    lp = log_prior_shell(params, priors)
    if not np.isfinite(lp):
        return -np.inf

    # 2. likelihood
    logL = log_likelihood_shell(
        params,
        data_cube,
        pa_rad,
        dx_au,
        header,
        Local_Standard_Velocity,
        max_dist_value,
        M_star,
        radius_in_au,
        radius_out_au,
        scale,
        log_power,
    )
    if not np.isfinite(logL):
        return -np.inf

    return lp + logL

def log_likelihood_distance(params, data_cube, search_bound,
                            pa_rad, dx_au, header, Local_Standard_Velocity,
                            v_weight_for_cube, max_dist_value,
                            M_star, radius_in_au, radius_out_au,
                            scale, log_power):
    """
    計算對數似然值，使用 distance_cube * data_cube 的誤差模型。
    
    這裡不再手動計算 model 的像素座標，而是把 5 個參數與
    WCS / 幾何資訊全部丟給 grow_distance_cube_bounded，
    由它負責：
      PSS_model → 物理座標 → (x_pix, z_pix, v_pix) → distance cube。
    """
    
    # 從 data_cube 獲取形狀
    cube_shape = data_cube.shape
    
    Theta_best, Phi_best, Inclination_best, T_best, Omega_best = params
    # 直接呼叫 grow_distance_cube_bounded 建立距離立方體
    distance_cube = grow_distance_cube_bounded(
        cube_shape,
        Theta_best, Phi_best, Inclination_best, T_best, Omega_best,
        pa_rad, dx_au, header, Local_Standard_Velocity,
        v_weight_for_cube, max_dist_value,
        M_star, radius_in_au, radius_out_au,
        scale, log_power,
        bound=search_bound,
    )
    
    valid_mask = distance_cube >= 0
    numerator = np.nansum(distance_cube[valid_mask] * data_cube[valid_mask])
    denominator = np.nansum(data_cube[valid_mask])
    if denominator == 0 or np.isnan(numerator):
        return -np.inf     
    normalized_error = np.sqrt(numerator / denominator)
    return -np.log10(normalized_error)

def in_phi_range(phi, phi_min, phi_max):
    """
    Handle periodic prior for Phi in [0, 2π):
    若區間沒有跨 0 度：直接判斷 phi_min <= phi <= phi_max
    若區間跨越 0 度：例如 (350°, 10°)，則接受 phi >= phi_min 或 phi <= phi_max
    """
    # 正規化到 [0, 2π)
    two_pi = 2.0 * np.pi
    phi = phi % two_pi
    phi_min = phi_min % two_pi
    phi_max = phi_max % two_pi

    if phi_min <= phi_max:
        return (phi_min <= phi) and (phi <= phi_max)
    else:
        # wrap-around case
        return (phi >= phi_min) or (phi <= phi_max)

def log_prior_distance(params, prior_ranges):
    """
    計算對數先驗值。
    """
    Theta0, Phi0, Incl, T, Omega = params
    
    # 檢查參數是否在先驗範圍內
    if not (
        prior_ranges["Theta zero"][0] < Theta0 < prior_ranges["Theta zero"][1]
        and in_phi_range(Phi0, *prior_ranges["Phi zero"])
        and prior_ranges["Inclination"][0] < Incl < prior_ranges["Inclination"][1]
        and prior_ranges["Time"][0] < T < prior_ranges["Time"][1]
        and prior_ranges["Omega"][0] < Omega < prior_ranges["Omega"][1]
    ):
        return -np.inf # 如果超出範圍，對數先驗為負無窮
    
    return 0.0 # 均勻先驗，對數值為 0

def log_posterior_distance(params, data_cube, search_bound, parameter_prior_ranges,
                           pa_rad, dx_au, header, Local_Standard_Velocity,
                           v_weight_for_cube, max_dist_value,
                           M_star, radius_in_au, radius_out_au,
                           scale, log_power):
    """
    計算對數後驗值。
    (註：依賴 log_prior_distance 和 log_likelihood_distance)
    """
    lp = log_prior_distance(params, parameter_prior_ranges)
    if not np.isfinite(lp):
        return -np.inf

    ll = log_likelihood_distance(
        params,
        data_cube,
        search_bound,
        pa_rad, dx_au, header, Local_Standard_Velocity,
        v_weight_for_cube, max_dist_value,
        M_star, radius_in_au, radius_out_au,
        scale, log_power,
    )
    return lp + ll

# ===================================================================
# 1. MCMC 先驗 (Prior)
# ===================================================================
def log_prior_fast(params, prior_ranges):
    """
    計算對數先驗值。
    (這與您舊的 log_prior 函數相同，只檢查 5 個參數是否在範圍內)
    """
    Theta0, Phi0, Incl, T, Omega = params
    
    # 檢查參數是否在先驗範圍內
    # 與這裡的 "Theta zero", "Phi zero" 等字串完全匹配。
    if not (
        prior_ranges["Theta zero"][0] < Theta0 < prior_ranges["Theta zero"][1]
        and in_phi_range(Phi0, *prior_ranges["Phi zero"])
        and prior_ranges["Inclination"][0] < Incl < prior_ranges["Inclination"][1]
        and prior_ranges["Time"][0] < T < prior_ranges["Time"][1]
        and prior_ranges["Omega"][0] < Omega < prior_ranges["Omega"][1]
    ):
        return -np.inf # 如果超出範圍，對數先驗為負無窮
    
    return 0.0 # 均勻先驗，對數值為 0

# ===================================================================
# 2. MCMC 似然 (Likelihood) - [核心函數]
# ===================================================================
def log_likelihood_fast(params, streamercom_x, streamercom_z, streamercom_v, 
                        weight_v, solar_mass, scale, log_power):
    """
    這是一個「快速」的對數似然函數。
    它只擬合 11 個質心點，而不是整個 3D 立方體。
    
    參數:
    - params (list/array): MCMC 傳入的 5 個自由參數 
                           [Theta0, Phi0, Incl, T_Myr, Omega]
    - (其他...): 您的觀測數據和固定參數
    """
    
    # 1. 解包 MCMC 傳入的 5 個參數
    Theta_zero, Phi_zero, Inclination, T_Myr, omega = params
    
    # 2. 準備 pss.error_function 需要的參數
    #    (注意：您的 error_function 的第一個參數 'params' 只需要 2 個值)
    params_for_error_func = [Theta_zero, Phi_zero]
    
    # 3. 呼叫您 *快速* 且 *有效* 的 error_function
    #    (它會回傳 RMSE)
    rmse_error = error_function(
        params_for_error_func,
        streamercom_x, 
        streamercom_z, 
        streamercom_v, 
        weight_v, 
        T_Myr,          # T_Myr 作為獨立參數傳入
        omega,          # omega 作為獨立參數傳入
        Inclination,    # Inclination 作為獨立參數傳入
        solar_mass,
        scale,
        log_power
    )
    
    # 4. 將 RMSE 轉換為 Log Likelihood
    #    我們使用 -log10(Error)，這與您慢速方法的邏輯一致
    if rmse_error <= 0 or np.isnan(rmse_error):
        return -np.inf # 避免 log10(0) 或 log10(nan)
        
    return -np.log10(rmse_error)

# ===================================================================
# 3. MCMC 後驗 (Posterior)
# ===================================================================
def log_posterior_fast(params, prior_ranges, streamercom_x, streamercom_z, streamercom_v, 
                       weight_v, solar_mass, scale, log_power):
    """
    這是 MCMC 採樣器 (sampler) 真正會呼叫的函數。
    它結合了 Prior 和 Likelihood。
    """
    
    # 1. 檢查先驗 (Priors)
    lp = log_prior_fast(params, prior_ranges)
    if not np.isfinite(lp):
        return -np.inf
    
    # 2. 計算快速的似然 (Likelihood)
    ll = log_likelihood_fast(params, streamercom_x, streamercom_z, streamercom_v, 
                             weight_v, solar_mass, scale, log_power)
    
    # 3. 回傳總和
    return lp + ll

# ===================================================================
# 6. MCMC 結果分析 (MCMC Analysis)
# ===================================================================

def report_and_get_best_params(samples, confidence_level):
    """
    根據 MCMC 樣本和指定的信賴水準，計算、格式化輸出最佳參數和不確定性。
    
    參數:
    - samples (np.ndarray): MCMC 採樣的扁平化結果 (flat chain)。
    - confidence_level (float): 所需的信賴水Z準百分比 (例如 68.3, 95.0, 99.7)。
    
    回傳:
    - tuple: 最佳擬合參數 (Theta, Phi, T, Inclination, Omega) 的弧度值。
    """
    
    # 參數標籤 (必須與 samples 的列順序一致)
    labels = ["Theta0", "Phi0", "Inclination", "T", "Omega"]
    
    # 1. 計算所需的百分位數 (兩端對稱)
    if confidence_level >= 100 or confidence_level <= 0:
        raise ValueError("信賴水準必須在 (0, 100) 之間。")
        
    tail_percent = (100.0 - confidence_level) / 2.0
    
    # 計算下限、中位數 (最佳值)、上限所需的百分位數
    percentiles = [tail_percent, 50.0, 100.0 - tail_percent]
    
    print(f"\n--- MCMC 最佳擬合與不確定性 ({confidence_level:.1f}% 信賴區間) ---")
    
    # 用於儲存最佳參數的弧度值 (可以直接傳給 PSS_model)
    best_fit_params_for_model = np.zeros(samples.shape[1])
    
    for i, label in enumerate(labels):
        # 計算所需的百分位數
        mcmc = np.percentile(samples[:, i], percentiles)
        
        # 最佳值 (50th 百分位數)
        best_value = mcmc[1]
        
        # 誤差計算: q[0] 是下限誤差，q[1] 是上限誤差
        q = np.diff(mcmc)
        lower_error = q[0]
        upper_error = q[1]
        
        # 3. 格式化輸出邏輯
        
        if label in ["Theta0", "Phi0", "Inclination"]:
            # 儲存弧度值供模型使用
            best_fit_params_for_model[i] = best_value 
            
            # 轉換為度數供輸出報告
            best = np.rad2deg(best_value)
            lower = np.rad2deg(lower_error)
            upper = np.rad2deg(upper_error)
            unit = "deg"
            
            # 輸出格式: 最佳值 (10.4f), 誤差 (.4e)
            output_str = f"{label:<12s}: {best:10.4f} (+{upper:.4e} / -{lower:.4e}) {unit} ({confidence_level:.1f}%)"
            
        elif label == "T":
            # 儲存原始值供模型使用
            best_fit_params_for_model[i] = best_value
            
            # 時間參數 (最佳值 & 誤差都用科學記號, .4e)
            best = best_value
            lower = lower_error
            upper = upper_error
            unit = "Myr"
            
            output_str = f"{label:<12s}: {best:.4e} (+{upper:.4e} / -{lower:.4e}) {unit} ({confidence_level:.1f}%)"

        else: # Omega
            # 儲存原始值供模型使用
            best_fit_params_for_model[i] = best_value
            
            # Omega 參數 (最佳值: 10.4f, 誤差: .4e)
            best = best_value
            lower = lower_error
            upper = upper_error
            unit = ""
            
            output_str = f"{label:<12s}: {best:10.4f} (+{upper:.4e} / -{lower:.4e}) {unit} ({confidence_level:.1f}%)"

        print(output_str)

    # 回傳弧度形式的最佳參數
    return tuple(best_fit_params_for_model)

"""
# ===================================================================
# 殼層距離 cube 以及誤差函數
# ===================================================================

def build_shell_distance_cube(
    cube_shape,
    Theta_zero, Phi_zero, Inclination, T_Myr, omega,
    pa_rad, dx_au, header, Local_Standard_Velocity,
    v_weight, max_dist_value,
    M_star, radius_in_au, radius_out_au,
    scale, log_power,
    bound=None,
):
    # 1. 準備殼層立方體
    nv, nz, nx = cube_shape
    shell_cube = np.full(cube_shape, -1.0, dtype=np.float32)

    # 2. 產生模型流線 (物理單位；AU & km/s)
    x_m, y_m, z_m, u_m, v_m, w_m = PSS_model(
        Theta_zero, Phi_zero, Inclination, T_Myr, omega,
        M_star,
        radius_in_au, radius_out_au,
        resolution=200,
        scale=scale,
        log_power=log_power,
    )

    # 3. 轉換為像素 / 頻道座標
    im_center_y = float(header["CRPIX2"]) - 1.0  # 0-based
    im_center_x = float(header["CRPIX1"]) - 1.0
    CDELT3 = float(header["CDELT3"])
    CRVAL3 = float(header["CRVAL3"])
    CRPIX3 = float(header["CRPIX3"])

    # 若呼叫端沒有特別給，從 header 讀取 LSRVEL（若沒有就維持原值）
    if Local_Standard_Velocity is None:
        Local_Standard_Velocity = float(header.get("LSRVEL", 0.0))

    # 旋轉到 image frame
    x_rot = x_m * np.cos(pa_rad) - z_m * np.sin(pa_rad)
    z_rot = x_m * np.sin(pa_rad) + z_m * np.cos(pa_rad)

    # 轉成像素座標 (x_pix, z_pix)
    x_pix = np.round(x_rot / dx_au + im_center_x).astype(int)
    z_pix = np.round(z_rot / dx_au + im_center_y).astype(int)

    # 轉成頻道座標 v_pix（依照 FITS WCS: v = CRVAL3 + (i + 1 - CRPIX3) * CDELT3）
    v_LSR = v_m + Local_Standard_Velocity
    v_pix = np.round((v_LSR - CRVAL3) / CDELT3 + CRPIX3 - 1.0).astype(int)

    # 4. 過濾掉落在 cube 之外的模型點
    valid_mask = (
        (x_pix >= 0) & (x_pix < nx) &
        (z_pix >= 0) & (z_pix < nz) &
        (v_pix >= 0) & (v_pix < nv)
    )
    x_pix = x_pix[valid_mask]
    z_pix = z_pix[valid_mask]
    v_pix = v_pix[valid_mask]

    if x_pix.size == 0:
        # 沒有任何模型點落在 cube 內，直接回傳全 -1
        return shell_cube

    # 5. 決定計算區域的邊界
    if bound is not None:
        v_bound, z_bound, x_bound = bound
        v_min, v_max = int(v_bound[0]), int(v_bound[1])
        z_min, z_max = int(z_bound[0]), int(z_bound[1])
        x_min, x_max = int(x_bound[0]), int(x_bound[1])
    else:
        v_min, z_min, x_min = 0, 0, 0
        v_max, z_max, x_max = nv - 1, nz - 1, nx - 1

    # 保險起見，再把邊界裁切到合法的索引範圍
    v_min = max(0, v_min)
    z_min = max(0, z_min)
    x_min = max(0, x_min)
    v_max = min(nv - 1, v_max)
    z_max = min(nz - 1, z_max)
    x_max = min(nx - 1, x_max)

    # 6. 建立子立方體遮罩，對模型線位置做 EDT
    sub_shape = (v_max - v_min + 1,
                 z_max - z_min + 1,
                 x_max - x_min + 1)
    # True = background, False = model-line seeds
    sub_mask = np.ones(sub_shape, dtype=bool)

    # 將模型點轉為子立方體內的相對座標，標記為 False
    for vv, zz, xx in zip(v_pix, z_pix, x_pix):
        if (v_min <= vv <= v_max and
            z_min <= zz <= z_max and
            x_min <= xx <= x_max):
            sub_mask[vv - v_min, zz - z_min, xx - x_min] = False

    # 若整個子區域裡沒有任何 False（即沒有模型點），直接回傳全 -1
    if np.all(sub_mask):
        return shell_cube

    # 7. 使用 EDT 計算到最近模型點的距離
    #    sampling[0] 對應頻道軸，帶入 sqrt(v_weight) 以實現
    #    d = sqrt(dx^2 + dz^2 + v_weight * dv^2)
    if v_weight > 0.0:
        v_scale = np.sqrt(v_weight)
    else:
        # v_weight = 0 時，速度軸不應影響距離，給一個很小的尺度避免數值問題
        v_scale = 1e-3

    sub_dist = distance_transform_edt(
        sub_mask,
        sampling=(v_scale, 1.0, 1.0)
    ).astype(np.float32)

    # 8. 將距離 d 轉成殼層編號 k，超過 max_dist_value 的改回 -1
    sub_shell = np.full_like(sub_dist, -1.0, dtype=np.float32)
    valid = sub_dist >= 0.0
    if np.any(valid):
        d_valid = sub_dist[valid]
        k = np.ceil(d_valid).astype(np.int32)
        # 把 0 距離也視為第一層殼
        k[k < 1] = 1
        # 上限不超過 max_dist_value
        k[k > int(max_dist_value)] = int(max_dist_value)
        sub_shell[valid] = k.astype(np.float32)

    # 9. 寫回到完整殼層立方體
    shell_cube[v_min:v_max + 1,
               z_min:z_max + 1,
               x_min:x_max + 1] = sub_shell

    return shell_cube"""