import sys
import os
import numpy as np
import scipy.constants as spc

# def pixel_scale_AU(distance_pc, pc_to_AU, pixel_scale_arcsec, arcsec_per_degree):
#     # 計算 1 pixel 的 AU 長度
#     pixel_scale_AU = (distance_pc * 2 * np.pi * pc_to_AU * pixel_scale_arcsec) / (360 * arcsec_per_degree)
#     return pixel_scale_AU

def time_to_deg(ra_time):
    """將 'HH:MM:SS.SSS' 轉成度數"""
    h, m, s = [float(i) for i in ra_time.split(':')]
    return (h + m/60 + s/3600) * 15 # RA 1小時 = 15度

def calc_ra_arcsec(ra_start, ra_end, dec_deg, distance_pc, n_pixels):
    """
    計算 RA 方向的範圍（單位：arcsec，四捨五入後）
    並轉換成每個 pixel 的 AU 大小。

    參數：
    - ra_start, ra_end: RA起點和終點 (字串, 'HH:MM:SS.SSS'格式)
    - dec_deg: DEC赤緯，單位是度
    - distance_pc: 距離（單位pc）
    - n_pixels: 圖片的pixels數量

    輸出：
    - arcsec_range: 四捨五入後的總arcsec範圍
    - AU_per_pixel: 每個pixel代表多少AU
    """

    # 1. 轉成度
    ra_start_deg = time_to_deg(ra_start)
    ra_end_deg = time_to_deg(ra_end)
    
    # 2. RA範圍，單位 degree
    delta_ra_deg = ra_end_deg - ra_start_deg
    
    # 3. 轉 arcsec 並考慮 cos(DEC)
    dec_rad = np.deg2rad(dec_deg)
    delta_ra_arcsec = delta_ra_deg * 3600 * np.cos(dec_rad)

    # 4. 四捨五入
    arcsec_range = int(np.round(delta_ra_arcsec))

    # 5. 換成 AU
    AU_per_arcsec = 2 * np.pi * distance_pc * 206265 / 360 / 3600 # 1pc = 206265 arcsec
    total_AU = arcsec_range * AU_per_arcsec
    AU_per_pixel = total_AU / n_pixels

    return arcsec_range, AU_per_pixel

def pixel_to_arcsec(xpix, ypix, xcenpix, ycenpix, im):
    delx, dely = im.delx*3600., im.dely*3600.
    x_arcsec = (xpix - xcenpix) * delx
    y_arcsec = (ypix - ycenpix) * dely
    return x_arcsec, y_arcsec

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

def radius_sn(singal_noise, mim_singal=3, r_0=1.2):
    radius = r_0 * (1 - np.exp(mim_singal - singal_noise))
    return np.max(radius, 0)

def spheres(max_radius):
    """預先建立半徑從1到max_radius的所有球形座標"""
    sphere_dict = {}
    for r in range(1, max_radius + 1):
        x, y, z = np.meshgrid(np.arange(-r, r + 1),
                              np.arange(-r, r + 1),
                              np.arange(-r, r + 1))
        mask = (x**2 + y**2 + z**2 <= r**2) & (x**2 + y**2 + z**2 > 0)
        sphere_dict[r] = (x[mask], y[mask], z[mask])
    return sphere_dict

def grow_region(data, init_points, sigma_value, r_0=4, sigma_thresh=3, max_iter=1000):
    """
    依據初始點與像素強度，圈出擴展區域。
    
    data: 3D array, 觀測影像資料
    init_points: list of (x, y, z), 初始點
    sigma_thresh: 門檻，像素強度需高於 mean + sigma_thresh * std
    return: 3D boolean mask of selected region
    """
    data = data / sigma_value
    region_mask = np.zeros_like(data, dtype=bool)
    height, width, depth = data.shape
    threshold = sigma_thresh

    to_check = set(init_points)
    visited = set(init_points)
    sphere_dict = spheres(max_radius=r_0)
    # for pt in init_points:
    #     print(f"Init {pt}: {data[pt[0], pt[1]]:.2f} > {threshold:.2f}?")

    for _ in range(max_iter):
        new_points = set()
        for x, y, z in to_check:
            if (0 <= y < width) and (0 <= x < height) and (0 <= z < depth):
                if data[x, y, z] > threshold and not region_mask[x, y, z]:
                    region_mask[x, y, z] = True
                    
                    # print(f"Accepted: x={x}, y={y}, value={data[x, y]:.2f}")
                    radius = radius_sn(data[x, y, z], mim_singal=sigma_thresh, r_0=4)
                    int_radius = int(np.ceil(radius))
                    
                    if int_radius not in sphere_dict:
                        continue  # 如果超過預設最大半徑就跳過
                    # print(f"Sigma level {sigma_level}, checking {radius}x{radius} region")
                    dx_array, dy_array, dz_array = sphere_dict[int_radius]
                    
                    for i in range(len(dx_array)):
                        nx, ny, nz = x + dx_array[i], y + dy_array[i], z + dz_array[i]
                        if (0 <= ny < width) and (0 <= nx < height) and (0 <= nz < depth):
                            if (nx, ny, nz) not in visited:
                                new_points.add((nx, ny, nz))
                                visited.add((nx, ny, nz))
        if not new_points:
            break
        to_check = new_points

    return region_mask


def Omega_ref(radius_ref_au, Mass_star):
    Omega_ref = np.sqrt(spc.G * Mass_star * 2e30 / (radius_ref_au * 1.49e11)) / (radius_ref_au * 1.49e11) #Keplerian velocity (radian) 1e-13
    return Omega_ref

def x_func(Rx, A, k, R_0):
    x_cos = A * np.cos(k * (Rx + R_0))
    return x_cos

def z_func(Rz, B, k, R_0, C):
    z_sin = B * np.sin(k * (Rz + R_0)) + C
    return z_sin

def spherical_coords(x,y):
    ''' Converts cartesian coordinates (x,y) to polar coordinates.'''
    r = np.sqrt((x**2)+(y**2)) # total distance from center
    theta = np.arctan2(y,x)  # Angle wrt x-axis, in radians
    return(r,theta)

def gaussian(x, mu, sig):
    return np.exp(-np.power((x - mu) / sig, 2.0) / 2)

def A_const(R_0, k, params):
    X, Z, R_i = params[0], params[1], params[2]
    A = np.sum(X * np.cos(k * (R_i + R_0))) / np.sum(np.cos(k * (R_i + R_0)) ** 2)
    return A

def B_const(R_0, k, params):
    X, Z, R_i = params[0], params[1], params[2]
    N = len(X)
    B = (N * np.sum(Z * np.sin(k * (R_i + R_0))) - np.sum(Z) * np.sum(np.sin(k * (R_i + R_0)))) / (N * np.sum(np.sin(k * (R_i + R_0)) ** 2) - np.sum(np.sin(k * (R_i + R_0))) ** 2)
    return B

def C_const(R_0, k, params):
    X, Z, R_i = params[0], params[1], params[2]
    N = len(X)
    C = (np.sum(Z) * np.sum(np.sin(k * (R_i + R_0)) ** 2) - np.sum(Z * np.sin(k * (R_i + R_0))) * np.sum(np.sin(k * (R_i + R_0)))) / (N * np.sum(np.sin(k * (R_i + R_0)) ** 2) - np.sum(np.sin(k * (R_i + R_0))) ** 2)
    return C

def chi_square(R_0, k, params):
    X, Z, R_i = params[0], params[1], params[2]
    A = A_const(R_0, k, params)
    B = B_const(R_0, k, params)
    C = C_const(R_0, k, params)
        
    return np.sum((X - A * np.cos(k * (R_i + R_0))) ** 2 + (Z - B * np.sin(k * (R_i + R_0)) - C) ** 2)

def f_kR(R_0, k, params):
    
    X, Z, R_i = params[0], params[1], params[2]
    
    A = A_const(R_0, k, params)
    B = B_const(R_0, k, params)
    C = C_const(R_0, k, params)

    F_func = A * np.sum(X * R_i * np.sin(k * (R_i + R_0))) - A ** 2 * np.sum(R_i * np.cos(k * (R_i + R_0)) * np.sin(k * (R_i + R_0))) - B * np.sum(Z * R_i * np.cos(k * (R_i + R_0))) + B ** 2 * np.sum(R_i * np.sin(k * (R_i + R_0)) * np.cos(k * (R_i + R_0))) + B * C * np.sum(R_i * np.cos(k * (R_i + R_0)))
    return F_func

def g_kR(R_0, k, params):
    
    X, Z, R_i = params[0], params[1], params[2]
    
    A = A_const(R_0, k, params)
    B = B_const(R_0, k, params)
    C = C_const(R_0, k, params)
    
    G_func = A * np.sum(X * np.sin(k * (R_i + R_0))) - A ** 2 * np.sum(np.cos(k * (R_i + R_0)) * np.sin(k * (R_i + R_0))) - B * np.sum(Z * np.cos(k * (R_i + R_0))) + B ** 2 * np.sum(np.sin(k * (R_i + R_0)) * np.cos(k * (R_i + R_0))) + B * C * np.sum(np.cos(k * (R_i + R_0)))
    return G_func

def calc_params(R_root2, K_root2, params, radius_ref_au):
    A = A_const(R_root2, K_root2, params)
    B = B_const(R_root2, K_root2, params)
    C = C_const(R_root2, K_root2, params)
    if A < 0:
        A = -A
        B = -B
        C = -C
        R_root2 += np.pi / K_root2
    radius_ref_m = radius_ref_au * spc.astronomical_unit
    
    inclin = np.arcsin(-B / A)
    phi_par = R_root2 * K_root2
    theta_par = np.arctan2(A, C / np.cos(inclin))
    omega_ref = (A / radius_ref_m / np.sin(theta_par)) ** (4/3) / np.cos(inclin)
    t = K_root2 * (np.cos(inclin)) ** (1/4) / omega_ref ** (3/4)
    
    return A, B, C, inclin, phi_par, theta_par, omega_ref, t

    
def arrow_line(times, arrow_resolution, interval_of_arrows, x, y, z, u, v, w, x_rotate, y_rotate, z_rotate, u_rotate, v_rotate, w_rotate):
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

def PSS_model(Theta_zero, Phi_zero, Inclination, T_Myr, omega, 
              solar_mass, radius_in_au=1.5e3, radius_out_au=1e4, resolution=200):

    # Omega = gM/r**3
    # T_Myr = free-fall time

    # #######Parameters#######
    # Theta_zero = spc.pi/3
    # Phi_zero = spc.pi*6/7
    # Inclination = -spc.pi/3 #The angle of streamer rotated.

    # #######Model parameters#######
    # T_Myr = 1 #Time (Myr)
    # Omega_ref = 3e-13 #Keplerian velocity (radian) 1e-13
    # radius_ref_au = 500 #Disk edge (au)

    # #######Variables#######
    # radius_edge_au = 3000 #Streamer edge (au)
    theta_value = Theta_zero

    #######Unit conversion#######
    T_s = T_Myr * 1e6 * spc.year #Time (s)
    # Omega_ref = Omega_ref(radius_ref_pixel, distance_pc, pc_to_AU, pixel_scale_arcsec, arcsec_per_degree)
    radius_in_m = radius_in_au * spc.astronomical_unit #Streamer edge (m)
    radius_out_m = radius_out_au * spc.astronomical_unit #Streamer edge (m)

    #######Resolution parameters######
    # arrow_resolution = 20
    # interval_of_arrows = int(resolution / arrow_resolution)
    c_s = 200 #m/s
    
    streamline_radius = np.linspace(radius_in_m, radius_out_m, resolution)
    # omega_ref = Omega_ref(radius_ref_au, solar_mass) #Keplerian velocity (radian)
    # phi_value = Phi_zero + T_s ** (-1/2) * omega * omega_ref * (streamline_radius / radius_ref_m) ** (-1/2) * (radius_ref_m / c_s) ** (3/2)
    # velocity_r = radius_ref_m * omega * omega_ref * np.sin(theta_value) * (streamline_radius / radius_ref_m) ** (-1)
    # V_infall = - np.sqrt(spc.G * solar_mass * 2e30 / streamline_radius)
    
    r_s = c_s * T_s #m
    V_infall = - np.sqrt(2 * spc.G * solar_mass * 2e30 / streamline_radius) - 3.3 * c_s #m/s
    # phi_value = Phi_zero + T_s * Omega_ref * ((streamline_radius / r_s) ** (-1/2) + (streamline_radius / r_s) ** (-1/3) - 2)
    # velocity_r = r_s * Omega_ref * np.sin(theta_value) * ((streamline_radius / r_s) ** (-1) + (streamline_radius / r_s) ** (-1/3))
    alpha = -1/3
    Omega_s = Omega_ref(r_s / 1.49e11, solar_mass) * omega #Keplerian velocity (radian)
    Omega = Omega_s * ((streamline_radius / r_s) ** (-2) + (streamline_radius / r_s) ** (alpha - 1))
    phi_value = Phi_zero + T_s * Omega_s * np.sqrt(2 * c_s ** 3 * T_s / (spc.G * solar_mass * 2e30)) * ((streamline_radius / r_s) ** (-1/2) + (streamline_radius / r_s) ** (alpha)) #np.sqrt(2 * c_s ** 3 * T_s / (spc.G * solar_mass * 2e30)) * 
    velocity_r = streamline_radius * Omega * np.sin(theta_value)

    # V_infall = -660 #3.3c_s (m/s)

    # ######Origin coordinate#######
    # x_orin = streamline_radius * np.sin(theta_value) * np.cos(Phi_zero)
    # y_orin = streamline_radius * np.sin(theta_value) * np.sin(Phi_zero)
    # z_orin = streamline_radius * np.cos(theta_value)
    # ######Rotated by X-axis origin coordinate######
    # y_orin_rotate = y_orin * np.cos(Inclination) + z_orin * np.sin(Inclination)
    # z_orin_rotate = - y_orin * np.sin(Inclination) + z_orin * np.cos(Inclination)
    
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

    # print(r=np.sqrt(x_rotate**2+y_rotate**2+z_rotate**2))
    # u_rotate = - x_rotate
    u += u_r
    v += v_r
    w += w_r
    ######Rotated by X-axis Velocity######
    u_rotate = u
    v_rotate = v * np.cos(Inclination) + w * np.sin(Inclination) # v_y
    w_rotate = - v * np.sin(Inclination) + w * np.cos(Inclination)

    # times = 5e9 #length
    # ######Streamer coordinate for Arrow######
    # x_arrowline = x + u * times
    # y_arrowline = y + v * times
    # z_arrowline = z + w * times
    # ######Rotated by X-axis Streamer coordinate for Arrow######
    # x_arrowline_rotate = x_rotate + u_rotate * times
    # y_arrowline_rotate = y_rotate + v_rotate * times
    # z_arrowline_rotate = z_rotate + w_rotate * times

    # ######Draw arrow line######
    # x_arrowline_interval, y_arrowline_interval, z_arrowline_interval, u_arrow, v_arrow, w_arrow, x_arrowline_rotate_interval, y_arrowline_rotate_interval, z_arrowline_rotate_interval, u_arrow_rotate, v_arrow_rotate, w_arrow_rotate = arrow_line(times, arrow_resolution, interval_of_arrows, x, y, z, u, v, w, x_rotate, y_rotate, z_rotate, u_rotate, v_rotate, w_rotate)

    # ######Unit conversions######
    #     ######m->au######
    # x_orin /= spc.astronomical_unit
    # y_orin /= spc.astronomical_unit
    # z_orin /= spc.astronomical_unit

    # y_orin_rotate /= spc.astronomical_unit
    # z_orin_rotate /= spc.astronomical_unit

    x_rotate /= spc.astronomical_unit
    y_rotate /= spc.astronomical_unit
    z_rotate /= spc.astronomical_unit

    # x_arrowline_rotate /= spc.astronomical_unit
    # y_arrowline_rotate /= spc.astronomical_unit
    # z_arrowline_rotate /= spc.astronomical_unit

    # x_arrowline_rotate_interval /= spc.astronomical_unit
    # y_arrowline_rotate_interval /= spc.astronomical_unit
    # z_arrowline_rotate_interval /= spc.astronomical_unit
        ######m->km######
    # velocity_vectors /= 1e3
    u_rotate /= 1e3
    v_rotate /= 1e3
    w_rotate /= 1e3
    # u_arrow_rotate /= 1e3
    # v_arrow_rotate /= 1e3
    # w_arrow_rotate /= 1e3
    #     ######radian to degrees######
    # Theta_zero_deg = Theta_zero * 180/spc.pi
    # Phi_zero_deg = Phi_zero * 180/spc.pi
    # Inclination_deg = Inclination * 180/spc.pi    


    # ######Plot ratated streamer######
    # fig = go.Figure()
    # # fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0],)) #origin streamer line
    # fig.add_trace(go.Scatter3d(x=x_orin, y=y_orin_rotate, z=z_orin_rotate, mode='lines', name='T=0, Rotated')) #origin streamer line
    # fig.add_trace(go.Scatter3d(x=x_rotate, y=y_rotate, z=z_rotate, mode='lines', name='T=63', marker=dict(color='grey'))) #streamer line
    # for i in range(arrow_resolution):
    #     fig.add_trace(go.Scatter3d(x=[x_rotate[interval_of_arrows * i], x_arrowline_rotate[interval_of_arrows * i]], 
    #                             y=[y_rotate[interval_of_arrows * i], y_arrowline_rotate[interval_of_arrows * i]], 
    #                             z=[z_rotate[interval_of_arrows * i], z_arrowline_rotate[interval_of_arrows * i]], mode='lines', 
    #                             marker=dict(color='rgb(100,100,100)'), line=dict(width=3))) #origin streamer line
    #     # print(u[interval_of_arrows * i], v[interval_of_arrows * i], w[interval_of_arrows * i])
    # fig.add_trace(go.Cone(x=x_arrowline_rotate_interval, y=y_arrowline_rotate_interval, z=z_arrowline_rotate_interval, 
    #                         u=u_arrow_rotate, v=v_arrow_rotate, w=w_arrow_rotate, name='Streamer line', sizemode="scaled", 
    #                         sizeref=0.3, cmin=np.min(velocity_vectors), cmax=np.max(velocity_vectors), colorscale="Portland",)) #arrow
    # fig.update_layout(showlegend=False)
    # # fig.add_trace(go.Cone(x=x, y=y_rotate, z=z_rotate, u=u, v=v, w=w, sizemode="scaled", sizeref=3, colorscale="Portland", cmin=np.min(velocity_vectors), cmax=np.max(velocity_vectors))) #arrow
    # # fig.update_layout(scene = dict( 
    # #                             xaxis = dict(nticks=4, range=x_axis_range,),
    # #                             yaxis = dict(nticks=4, range=y_axis_range,),
    # #                             zaxis = dict(nticks=4, range=z_axis_range,),
    # #                             aspectmode='manual',
    # #                             aspectratio=dict(x=1, y=1, z=1)
    # #                             )
    # #                 )
    # fig.show()
    # fig.write_html(f"/Users/thchuang/Documents/Code/MS_project/streamerimages/html/{round(Theta_zero_deg, 2)}_{round(Phi_zero_deg, 2)}_{round(Inclination_deg, 2)}.html")

    ######Plot X-Z-Vy######
    # plt.scatter(x_rotate[::-1], z_rotate[::-1], s=500, c=v_rotate[::-1], cmap="coolwarm")
    # plt.colorbar(label='Vy (km/s)')
    # plt.scatter(0,0)
    # plt.xlabel('X (au)')
    # plt.ylabel('Z (au)')
    # plt.xlim(-1.5e14,1.5e14)
    # plt.ylim(-1.5e14,1.5e14)
    # plt.gca().set_aspect('equal')
    # plt.savefig(f"/Users/thchuang/Documents/Code/MS_project/streamerimages/pngs/{round(Theta_zero_deg, 2)}_{round(Phi_zero_deg, 2)}_{round(Inclination_deg, 2)}.png")
    # plt.close()
    # print(u_r,v_r,w_r)
    return x_rotate, y_rotate, z_rotate, u_rotate, v_rotate, w_rotate #final 刪掉只保留x, z, v

def error_function(params, streamercom_x, streamercom_z, streamercom_v, 
                   weight_v, T_Myr, omega, Inclination, solar_mass):
    """
    計算 PSS_model 與數據點的誤差，使用最近鄰匹配來尋找最佳對應點。

    params: PSS_model 的參數
    streamercom_x, streamercom_y, streamercom_v: 數據中的 11 個元素 (觀測值)

    返回:
    - error: 總誤差
    """
    
    Theta_zero, Phi_zero = params
    
    total_error = 0
    num_points = len(streamercom_x)  # 應該是 11

    for i in range(num_points):
        radius_in_au = np.sqrt(streamercom_x[i] ** 2 + streamercom_z[i] ** 2)
        radius_out_au = radius_in_au * 3
        # 計算 PSS_model 曲線
        x_model, y_model, z_model, u_rotate, v_rotate, w_rotate = PSS_model(
            Theta_zero, Phi_zero, Inclination, T_Myr, omega, 
            solar_mass, radius_in_au, radius_out_au, resolution=20)

        # 轉換為 NumPy 陣列
        x_model, z_model, v_model = np.array(x_model), np.array(z_model), np.array(v_rotate)
        
        # 計算此數據點與這段曲線所有點的距離
        distances = (streamercom_x[i] - x_model) ** 2 + \
                    (streamercom_z[i] - z_model) ** 2 + \
                    weight_v * (streamercom_v[i] - v_model) ** 2
        # print(np.shape(distances), np.shape(x_model))
        # 找到最近的 PSS_model 點
        nearest_index = np.argmin(distances)
        
        # 計算此點的誤差
        error = distances[nearest_index]  # 已經是平方誤差
        total_error += error
    total_error = np.sqrt(total_error / num_points)
    return total_error

"""MCMC"""
def model_prediction(params, M_star, radius_in_au, radius_out_au, resolution):
    # 拆解參數（依你模型而定）
    Theta0, Phi0, Incl, T, Omega = params
    x_model, y_model, z_model, u_model, v_model, w_model = PSS_model(Theta0, Phi0, Incl, T, Omega, M_star, radius_in_au, radius_out_au, resolution)
    return x_model, z_model, v_model

def log_likelihood(params, x_data, z_data, v_data, M_star, radius_in_au, radius_out_au, resolution):
    """    
    logL 公式註解:
    假設觀測數據誤差服從獨立同分佈的高斯分佈 (Gaussian Distribution)。
    則對於每個數據點 i (包含 x, z, v 座標):
    P(data_i | model_i, sigma) = (1 / (sigma * sqrt(2*pi))) * exp(-(data_i - model_i)^2 / (2 * sigma^2))

    總概似度 (Likelihood) 是所有數據點機率的乘積：
    L = Product(P(data_i | model_i, sigma))

    取自然對數 (log-likelihood)：
    logL = Sum(log(P(data_i | model_i, sigma)))
    logL = Sum(log(1 / (sigma * sqrt(2*pi))) - (data_i - model_i)^2 / (2 * sigma^2))
    logL = Sum(-(data_i - model_i)^2 / (2 * sigma^2)) - N * log(sigma * sqrt(2*pi))

    其中 N 是數據點的總數。
    在 MCMC (或最大概似估計) 中，我們通常只關心與模型參數有關的部分，
    常數項 - N * log(sigma * sqrt(2*pi)) 不會影響後驗機率的峰值位置，因此可以省略。

    最終，logL 可簡化為：
    logL = -0.5 * ( Sum(residual_x^2 / sigma_pos^2) + Sum(residual_z^2 / sigma_pos^2) + Sum(residual_v^2 / sigma_v^2) )

    程式碼中的 chi2 = Sum(residual_x^2 + residual_z^2) / sigma_pos^2 + Sum(residual_v^2) / sigma_v^2
    所以 `return -0.5 * chi2` 正是這個簡化後的 logL。
    """
    x_model, y_model, z_model, u_model, v_model, w_model = model_prediction(params, M_star, radius_in_au, radius_out_au, resolution)
    # 定義誤差（假設獨立同分布 Gaussian）
    sigma_pos = 1.0 # 空間誤差
    sigma_v   = 0.1   # 速度誤差
    residual_x = x_data - x_model
    residual_z = z_data - z_model
    residual_v = v_data - v_model
    chi2 = np.sum(residual_x**2 + residual_z**2) / sigma_pos**2 + np.sum(residual_v**2) / sigma_v**2
    return -0.5 * chi2

def log_prior(params, prior_ranges):
    Theta0, Phi0, Incl, T, Omega = params # 這裡只拆解 MCMC 要採樣的參數
    # 從傳入的 prior_ranges 字典中獲取每個參數的範圍
    min_Theta0, max_Theta0 = prior_ranges['Theta0']
    min_Phi0, max_Phi0 = prior_ranges['Phi0']
    min_Incl, max_Incl = prior_ranges['Incl']
    min_T, max_T = prior_ranges['T']
    min_Omega, max_Omega = prior_ranges['Omega']
    if not (min_Theta0 < Theta0 < max_Theta0 and min_Phi0 < Phi0 < max_Phi0 and min_Incl < Incl < max_Incl and min_T < T < max_T and min_Omega < Omega < max_Omega):
        return -np.inf
    return 0.0  # Uniform prior (log(1) = 0)

def log_posterior(params, x_data, z_data, v_data):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, x_data, z_data, v_data)
