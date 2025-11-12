import sys
import os
import numpy as np
import scipy.constants as spc
import astropy.constants as astroc


# ===================================================================
# 1. 基礎工具 (Utilities)
# =Date: 2020-01-20
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
    alpha = -1/3
    Omega_s = Omega_ref(r_s / spc.astronomical_unit, solar_mass) * omega #Keplerian velocity (radian)
    Omega = Omega_s * ((streamline_radius / r_s) ** (-2) + (streamline_radius / r_s) ** (alpha - 1))
    phi_value = Phi_zero + T_s * Omega_s * np.sqrt(2 * c_s ** 3 * T_s / (spc.G * solar_mass * M_SUN_KG)) * ((streamline_radius / r_s) ** (-1/2) + (streamline_radius / r_s) ** (alpha)) #np.sqrt(2 * c_s ** 3 * T_s / (spc.G * solar_mass * 2e30)) * velocity_r = streamline_radius * Omega * np.sin(theta_value)
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

def error_function(params, streamercom_x, streamercom_z, streamercom_v, 
                   weight_v, T_Myr, omega, Inclination, solar_mass, scale, log_power):
    """
    計算 PSS_model 與數據點的誤差，使用最近鄰匹配來尋找最佳對應點。
    這裡同時：
    - 使用加權距離 sqrt( pos^2 + w_v * vel^2 ) 作為主誤差 (AU-metric)，供 grid / MCMC 最小化。
    - 紀錄分開的 RMS(position) 與 RMS(velocity)，方便之後轉換與診斷：
        error_function.last_pos_rmse      (in AU)
        error_function.last_vel_rmse      (in km/s)
        error_function.last_eq_vel_rmse   (equivalent km/s from total AU-metric)
    """
    
    Theta_zero, Phi_zero = params
    num_points = len(streamercom_x)
    if num_points == 0:
        return np.inf

    # 累積「對應點」的平方誤差，用來計算 RMS
    pos_sq_sum = 0.0
    vel_sq_sum = 0.0
    total_sq_sum = 0.0

    for i in range(num_points):
        # 以該點半徑決定當下要畫的 streamline 範圍，避免太短或太長
        radius_in_au = max(1.0, np.sqrt(streamercom_x[i] ** 2 + streamercom_z[i] ** 2) * 0.5)
        radius_out_au = radius_in_au * 30.0

        # 計算 PSS_model 曲線 (x,z: AU; v: km/s)
        x_model, y_model, z_model, u_model, v_model, w_model = PSS_model(
            Theta_zero, Phi_zero, Inclination, T_Myr, omega,
            solar_mass, radius_in_au, radius_out_au,
            resolution=100, scale=scale, log_power=log_power
        )

        # 對這個觀測點，計算它對所有模型點的加權距離
        dx = streamercom_x[i] - x_model
        dz = streamercom_z[i] - z_model
        dv = streamercom_v[i] - v_model

        # 加權平方距離 (全部轉成 AU^2 度量)
        distances_sq = dx**2 + dz**2 + weight_v * dv**2

        # 找最近的模型點
        j = np.argmin(distances_sq)

        # 分別記錄這個最佳對應點的 pos / vel / total 誤差
        pos_sq = dx[j]**2 + dz[j]**2          # AU^2
        vel_sq = dv[j]**2                     # (km/s)^2
        total_sq = distances_sq[j]            # AU^2 (含權重)

        pos_sq_sum += pos_sq
        vel_sq_sum += vel_sq
        total_sq_sum += total_sq

    # 計算 RMS
    mean_pos_sq = pos_sq_sum / num_points
    mean_vel_sq = vel_sq_sum / num_points
    mean_total_sq = total_sq_sum / num_points

    # 主誤差：給 grid / MCMC 用的「AU-metric」
    total_error = np.sqrt(mean_total_sq)

    # 紀錄診斷資訊（不改變回傳介面）
    error_function.last_pos_rmse = np.sqrt(mean_pos_sq)           # in AU
    error_function.last_vel_rmse = np.sqrt(mean_vel_sq)           # in km/s
    # 如果 weight_v > 0，換算等效 km/s；否則給 NaN
    if weight_v > 0:
        error_function.last_eq_vel_rmse = total_error / np.sqrt(weight_v)
    else:
        error_function.last_eq_vel_rmse = np.nan

    return total_error

def grow_distance_cube_bounded(cube_shape, model_line_coords, max_dist_value, v_weight, bound=None):
    """
    在指定的邊界內，計算每個像素與模型線的最短「加權」距離。
    (註：此函數用於「立方體擬合」)
    """
    nv, nz, nx = cube_shape

    if bound:
        v_bound, z_bound, x_bound = bound
        v_min, v_max = v_bound
        z_min, z_max = z_bound
        x_min, x_max = x_bound
    else:
        v_min, z_min, x_min = 0, 0, 0
        v_max, z_max, x_max = nv - 1, nz - 1, nx - 1

    # 確保索引是整數
    v_min, v_max = int(v_min), int(v_max)
    z_min, z_max = int(z_min), int(z_max)
    x_min, x_max = int(x_min), int(x_max)

    # 創建子立方體
    sub_distance_cube = np.full(
        (v_max - v_min + 1, 
         z_max - z_min + 1, 
         x_max - x_min + 1), 
        np.inf, 
        dtype=np.float32  # 使用 float32
    )
    v_grid, z_grid, x_grid = np.indices(sub_distance_cube.shape)
    
    # 篩選出在邊界內部的模型點，並轉換為子立方體的相對座標
    relative_model_coords = []
    for v, z, x in model_line_coords:
        if v_min <= v <= v_max and z_min <= z <= z_max and x_min <= x <= x_max:
            relative_model_coords.append((v - v_min, z - z_min, x - x_min))
    
    if not relative_model_coords:
        # 如果邊界內沒有模型點，返回一個全為 -1.0 的陣列
        return np.full(cube_shape, -1.0, dtype=np.float32)

    # 遍歷模型上的每個點
    for v_line, z_line, x_line in relative_model_coords:
        # --- 核心修改：使用加權距離 ---
        dist_sq = (
            (z_grid - z_line)**2 +
            (x_grid - x_line)**2 +
            v_weight * (v_grid - v_line)**2  # <--- 速度維度被加權
        )        
        
        # 更新子立方體中的最短距離
        sub_distance_cube = np.minimum(sub_distance_cube, dist_sq)

    # --- 類型修正 ---
    # 將超過最大距離的點標記為 -1.0 (而不是 np.nan，以避免 astype 錯誤)
    sub_distance_cube[sub_distance_cube > max_dist_value ** 2] = -1.0
    
    # 創建完整的距離立方體，默認為 -1.0
    full_distance_cube = np.full(cube_shape, -1.0, dtype=np.float32)
    
    # 將計算好的子立方體放回完整立方體的正確位置
    full_distance_cube[v_min:v_max+1, z_min:z_max+1, x_min:x_max+1] = sub_distance_cube

    return full_distance_cube

# ===================================================================
# 5. MCMC 框架 (MCMC Framework)
# (註：此區塊依賴 PSS_model 和 grow_distance_cube_bounded)
# ===================================================================

def log_likelihood(params, data_cube, search_bound, pa_rad, AU_per_pixel, im_center, dv, v_lastch_vel, v_lastch_num, v0, v_weight_for_cube, max_dist_value, M_star, radius_in_au, radius_out_au):
    """
    計算對數似然值，使用 distance_cube * data_cube 的誤差模型。
    
    參數：
    - params: 模型的參數 (Theta, Phi, T, Inclination, Omega)
    - data_cube: 觀測數據立方體 (V, Z, X)
    - search_bound: 裁剪後的邊界 [ [v_min, v_max], [z_min, z_max], [x_min, x_max] ]
    - 坐標轉換參數: AU_per_pixel, im_center, dv, v_lastch_num
    """
    
    # 從 data_cube 獲取形狀
    cube_shape = data_cube.shape
    
    Theta_best, Phi_best, Inclination_best, T_best, Omega_best = params

    # 1. 根據新的參數，重新生成模型線的物理座標
    # 假設 PSS_model 輸出 x, y, z 是 AU 單位，v 是 km/s 單位
    x_model, y_model, z_model, u_model, v_model, w_model = PSS_model(Theta_best, Phi_best, Inclination_best, T_best, Omega_best, M_star, radius_in_au, radius_out_au, 100, scale='log')
    
    # 2. 將物理座標轉換為整數像素座標
    x_pix_rotated = x_model / AU_per_pixel
    z_pix_rotated = z_model / AU_per_pixel
    x_pix_int = np.round(x_pix_rotated * np.cos(pa_rad) - z_pix_rotated * np.sin(pa_rad) + im_center[1]).astype(int)
    z_pix_int = np.round(x_pix_rotated * np.sin(pa_rad) + z_pix_rotated * np.cos(pa_rad) + im_center[0]).astype(int)
    v_pix_int = np.round(v_lastch_num - (v_model - v_lastch_vel + v0) / dv).astype(int)

    
    # 3. 生成新的距離立方體
    model_line_coords = zip(v_pix_int, z_pix_int, x_pix_int)
    distance_cube = grow_distance_cube_bounded(
        cube_shape, 
        model_line_coords, 
        max_dist_value, 
        v_weight_for_cube,
        bound=search_bound
    )
    
    # 4. 計算誤差：Error = sum(Distance * CubeData) / sum(CubeData)
    
    # 篩選掉距離為 < 0 的點 (超出範圍或無效)
    valid_mask = distance_cube >= 0
    
    # 計算分子：sum(Distance * CubeData)
    # 使用 distance_cube[valid_mask] 和 data_cube[valid_mask] 確保只對有效區域計算
    numerator = np.nansum(distance_cube[valid_mask] * data_cube[valid_mask])
    
    # 計算分母：sum(CubeData)
    denominator = np.nansum(data_cube[valid_mask])
    
    # 確保分母不為零
    if denominator == 0 or np.isnan(numerator):
        return -np.inf 
        
    # 計算最終誤差 (要最小化的量)
    normalized_error = np.sqrt(numerator / denominator)
    
    # 5. 轉換為 Log Likelihood
    return - np.log10(normalized_error)

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

def log_prior(params, prior_ranges):
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

def log_posterior(params, data_cube, search_bound, parameter_prior_ranges, pa_rad, AU_per_pixel, im_center, 
                  dv, v_lastch_vel, v_lastch_num, v0, v_weight_for_cube, max_dist_value, 
                  M_star, radius_in_au, radius_out_au):
    """
    計算對數後驗值。
    (註：依賴 log_prior 和 log_likelihood)
    """
    lp = log_prior(params, parameter_prior_ranges)
    if not np.isfinite(lp):
        return -np.inf
    
    # 傳入 log_likelihood 需要的參數
    ll = log_likelihood(params, data_cube, search_bound, pa_rad, AU_per_pixel, im_center, 
                        dv, v_lastch_vel, v_lastch_num, v0, v_weight_for_cube, max_dist_value, 
                        M_star, radius_in_au, radius_out_au)
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