"""
Per-emb-2 streamer grid fitting (moment map only version)
Two-stage grid search: coarse + refined
"""

# ============================================================
# 1. Import modules
# ============================================================
import sys, os, time
import numpy as np
import scipy.constants as spc
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.collections import LineCollection
from matplotlib.colors import PowerNorm
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from tqdm import tqdm
from spectral_cube import SpectralCube
import emcee 
import PSSpy as pss
import corner 


RUN_REFINE = False
RUN_MCMC_GRID = False
USE_CACHED_FIT = True
FIT_CACHE = "Per-emb-2_fit_gird_results.npz"
# FIT_CACHE = "Per-emb-2_fit_results.npz" #mcmc result

pa_deg = 50
pa_rad = np.deg2rad(pa_deg)
Local_Standard_Velocity = 7.05  # km/s
distance_pc = 300
M_SUN_KG = 1.98847e30
radius_ref_au = 1e3
M_star = 3.2
scale = "log"
log_power = 1.5

# ===== 新增圖片輸出資料夾變數 =====
PLOT_DIR = "Per-emb-2_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# ============================================================
# 3. 讀入 moment maps
# ============================================================
f_mom0 = "Per-emb-2-HC3N_10-9_TdV.fits"
f_mom1 = "Per-emb-2-HC3N_10-9_fit_Vc.fits"

# 直接用 astropy 讀取 2D moment maps
data_mom0, header = fits.getdata(f_mom0, header=True)
data_mom1 = fits.getdata(f_mom1)

dx_au = (abs(header['CDELT1']) + abs(header['CDELT2'])) / 2 * 3600.0 * distance_pc
dv = 0.1

n_pixels = data_mom0.shape[0]
im_center = (header['CRPIX1'] - 1, header['CRPIX2'] - 1)

# 座標格點
x = np.arange(n_pixels) - im_center[0]
z = np.arange(n_pixels) - im_center[1]
xx, zz = np.meshgrid(x, z)
r, theta = pss.spherical_coords(xx, zz)

# ============================================================
# 4. 計算 streamer 質心點
# ============================================================
N_elements = 11
pars = np.linspace(18, 50, N_elements+1)

x_means = np.zeros(N_elements)
z_means = np.zeros(N_elements)
v_means = np.zeros(N_elements)
xzstd = np.zeros(N_elements)
x_array_list = []
z_array_list = []
v_array_list = []
weights_list = []

# --- 幾何加權平均 ---
for i in range(N_elements):
    dinds = (r > pars[i]) & (r <= pars[i+1])
    x_means[i] = np.average(xx[dinds], weights=data_mom0[dinds])
    z_means[i] = np.average(zz[dinds], weights=data_mom0[dinds])
    xzstd[i]   = np.sqrt(np.average((xx[dinds]-x_means[i])**2 + (zz[dinds]-z_means[i])**2,
                                    weights=data_mom0[dinds]))

r_means, theta_means = pss.spherical_coords(x_means, z_means)
theta_r = interp1d(r_means, theta_means, fill_value="extrapolate")
std_r   = interp1d(r_means, xzstd, fill_value="extrapolate")

mom1_vel = np.ma.masked_invalid(data_mom1)

# --- 速度加權平均 ---
for i in range(N_elements):
    r_ref = (pars[i]+pars[i+1])/2
    theta_ref = theta_r(r_ref)
    std_ref = std_r(r_ref) / r_ref
    delta_theta = np.pi - np.abs(np.pi - np.abs(theta - theta_ref))
    weights = data_mom0 * pss.gaussian(delta_theta, 0, std_ref)
    dinds = (r>pars[i]) & (r<=pars[i+1])  # Identify points in given distance range
    dinds_v = (r>pars[i]) & (r<=pars[i+1]) & np.isfinite(data_mom1)
    x_means[i] = np.average(xx[dinds], weights=weights[dinds])
    z_means[i] = np.average(zz[dinds], weights=weights[dinds])
    v_means[i] = np.average(mom1_vel[dinds_v], weights=weights[dinds_v])
    # 存儲每次迴圈的值
    x_array_list.append(xx[dinds])
    z_array_list.append(zz[dinds])
    v_array_list.append(mom1_vel[dinds_v])
    weights_list.append(weights[dinds]/np.max(weights[dinds]))

# 單位轉換
x_means_AU = (x_means * np.cos(pa_rad) + z_means * np.sin(pa_rad)) * dx_au 
z_means_AU = (-x_means * np.sin(pa_rad) + z_means * np.cos(pa_rad)) * dx_au 
v_means_LS_km = v_means - Local_Standard_Velocity 
x_array = np.array(x_array_list, dtype=object)
z_array = np.array(z_array_list, dtype=object)
v_array = np.array(v_array_list, dtype=object)
weights_array = np.array(weights_list, dtype=object)
print(f"[Extracted] {np.sum(np.isfinite(x_means_AU))} valid centroids")

# ============================================================
# 4b. 視覺化：moment1 上標出抽出的質心點並存檔（WCS-based extent, Per-emb-50 style）
# ============================================================

# WCS 2D, image shape and reference
w2d = WCS(header).sub(['longitude', 'latitude'])
nx = header['NAXIS1']
ny = header['NAXIS2']
ra0 = header['CRVAL1']
dec0 = header['CRVAL2']
bottom_left = w2d.pixel_to_world(0, 0)
top_right = w2d.pixel_to_world(nx-1, ny-1)
x1 = (bottom_left.ra.deg - ra0) * 3600.0
x2 = (top_right.ra.deg - ra0) * 3600.0
y1 = (bottom_left.dec.deg - dec0) * 3600.0
y2 = (top_right.dec.deg - dec0) * 3600.0
extent = (min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2))

# 質心點的角秒偏移（相對影像中心, consistent with extent frame）
# x_means, z_means are in pixel offset from center (n_pixels/2)
# dx_arcsec from header
dx_arcsec = abs(header['CDELT1']) * 3600.0
dy_arcsec = abs(header['CDELT2']) * 3600.0
streamer_ra_arcsec = (x_means) * dx_arcsec
streamer_dec_arcsec = (z_means) * dy_arcsec

fig, ax = plt.subplots(figsize=(7, 6))
vmin = Local_Standard_Velocity - 1.0
vmax = Local_Standard_Velocity + 1.0
im = ax.imshow(data_mom1, origin='lower', cmap='coolwarm',
               extent=extent, vmin=vmin, vmax=vmax)
# 標質心點（顏色依照 Vc, not LSR-subtracted, same vmin/vmax）
sc = ax.scatter(streamer_ra_arcsec, streamer_dec_arcsec,
                c=v_means, cmap='coolwarm', vmin=vmin, vmax=vmax,
                s=40, marker='o', edgecolors='black', linewidths=1.0,
                label='Streamer centroids')
# 標中心
ax.plot(0, 0, marker='+', color='white', markersize=14, markeredgewidth=2)
ax.set_xlabel('RA Offset (arcsec)')
ax.set_ylabel('Dec Offset (arcsec)')
ax.set_title('Per-emb-2 HC3N moment1 with centroids')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.04)
cbar = fig.colorbar(im, cax=cax)
cbar.set_label('Velocity (km/s)')
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'Per-emb-2_centroids_mom1.png'), dpi=200)
plt.close(fig)

def _compute_extent(header, im_center, ny, nx):
    dx_arcsec = header["CDELT1"] * 3600.0
    dz_arcsec = header["CDELT2"] * 3600.0
    ra_min = (0   - im_center[1]) * dx_arcsec
    ra_max = (nx  - im_center[1]) * dx_arcsec
    dec_min= (0   - im_center[0]) * dz_arcsec
    dec_max= (ny  - im_center[0]) * dz_arcsec
    return (ra_min, ra_max, dec_min, dec_max), dx_arcsec, dz_arcsec

def plot_streamer_on_mom0(theta_deg, phi_deg, inc_deg, T_Myr, omega,
                          header, pa_rad, dx_au, im_center,
                          mom0, label, outname,
                          cen_x_AU=None, cen_z_AU=None, cen_v_LS_km=None):
    """
    在 moment-0 圖上畫出 best-fit PSS 流線，並可選擇疊加質心點。
    這版跟 SCrA 保持一致：
      - 主圖方形 (RA/Dec 1:1)
      - colorbar 貼右側，不撐壞主圖
      - 質心點（若有提供）一起畫在圖上
    """

    # 角度轉成 rad
    theta = np.deg2rad(theta_deg)
    phi   = np.deg2rad(phi_deg)
    inc   = np.deg2rad(inc_deg)

    ny, nx = mom0.shape
    extent, dx_arcsec, dz_arcsec = _compute_extent(header, im_center, ny, nx)

    # ---------- PSS model ----------
    x_m, y_m, z_m, u_m, v_m, w_m = pss.PSS_model(
        theta, phi, inc, T_Myr, omega,
        M_star,
        radius_in_au=2e3,
        radius_out_au=7e3,       # 原本你用 7e3，可視需要再調
        resolution=200,
        scale=scale,
        log_power=log_power,
    )

    x_pix = x_m / dx_au
    z_pix = z_m / dx_au

    x_pix_rot = x_pix * np.cos(pa_rad) - z_pix * np.sin(pa_rad) + im_center[1]
    z_pix_rot = x_pix * np.sin(pa_rad) + z_pix * np.cos(pa_rad) + im_center[0]

    ra_off  = (x_pix_rot - im_center[1]) * dx_arcsec
    dec_off = (z_pix_rot - im_center[0]) * dz_arcsec

    # ---------- 建 model 線 segments ----------
    pts = np.column_stack([ra_off, dec_off])
    if pts.shape[0] < 2:
        print("[plot_streamer_on_mom1] model points too few, skip.")
        return
    segments = np.stack([pts[:-1], pts[1:]], axis=1)

    fig, ax = plt.subplots(figsize=(6.2, 6))
    norm = PowerNorm(gamma=0.5, vmin=0, vmax=np.nanmax(mom0))

    # 背景：moment-1 map
    im = ax.imshow(
        mom0,
        origin="lower",
        cmap='inferno',
        extent=extent,
        # norm=norm,
        vmin=0,
    )

    # colorbar 貼右側，小一點
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.04)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('(Jy/beam km/s)')

    # model 線黑色外框 + 內層依速度上色
    lc_edge = LineCollection(segments, colors="black", linewidth=8, zorder=2)
    ax.add_collection(lc_edge)

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    lc = LineCollection(
        segments,
        cmap="coolwarm",
        norm=norm,
        linewidth=6.5,
        zorder=3,
    )
    lc.set_array(v_m + Local_Standard_Velocity)
    ax.add_collection(lc)
    ax.add_collection(lc)
    ax.add_collection(lc)
    ax.add_collection(lc)

    # num_element = 5
    # xarray_arc, z_array_arc = x_array[num_element] * dx_arcsec, z_array[num_element] * dx_arcsec
    # weights_im = ax.scatter( xarray_arc, z_array_arc, c=weights_array[num_element], s=8, cmap='YlGn_r')
    # x_means_arc, z_means_arc = x_means * dx_arcsec, z_means * dx_arcsec
    # ax.plot(x_means_arc, z_means_arc, color='w', lw=3, zorder=4)
    # divider = make_axes_locatable(ax)
    # cax     = divider.append_axes('right', size='3%', pad=0.04)
    # cbar = fig.colorbar(weights_im, cax=cax)
    # cbar.set_label('weight value')
    
    # ---------- 疊加質心點 ----------
    if cen_x_AU is not None and cen_z_AU is not None:
        # 這裡改成「直接以像素座標」來畫質心點，不再從 AU 反推
        cen_x_pix = np.asarray(cen_x_AU)
        cen_z_pix = np.asarray(cen_z_AU)

        # 轉成與背景 extent 一致的 RA/Dec offset
        cen_ra  = (cen_x_pix - im_center[1]) * dx_arcsec
        cen_dec = (cen_z_pix - im_center[0]) * dz_arcsec

        if cen_v_LS_km is not None:
            cen_v = np.asarray(cen_v_LS_km) + Local_Standard_Velocity
            ax.scatter(
                cen_ra,
                cen_dec,
                c=cen_v,
                cmap="coolwarm",
                vmin=vmin,
                vmax=vmax,
                s=10,
                marker="o",
                edgecolors="black",
                linewidths=0.5,
                zorder=5,
                label="Streamer Centroids",
            )
        else:
            ax.scatter(
                cen_ra,
                cen_dec,
                facecolors="none",
                edgecolors="black",
                s=45,
                marker="o",
                zorder=5,
                label="Streamer Centroids",
            )

    # 中心位置
    ax.scatter(0, 0, c="b", s=60, marker="+", zorder=6)

    ax.set_xlabel("RA Offset (arcsec)")
    ax.set_ylabel("Dec Offset (arcsec)")
    ax.set_title(label)

    # 主圖保持方形比例
    ax.set_aspect("equal", adjustable="box")

    # --- 比例尺與方向標示 ---
    # 定義比例尺位置（以 arcsec 為單位）
    text_pos_x = extent[1] - 0.25 * (extent[1] - extent[0])
    text_pos_y = extent[2] + 0.2 * (extent[3] - extent[2])
    scale_length = 3000  # AU

    # 將 3000 AU 轉成 arcsec
    scale_length_arcsec = scale_length / (distance_pc)  # 1" ≈ 1 AU / distance(pc)

    # 定義比例尺線段 (RA offset 軸)
    scale_range_x = [text_pos_x, text_pos_x - scale_length_arcsec]
    scale_range_y = [text_pos_y - 0.2, text_pos_y - 0.2]

    # 繪製比例尺與文字
    ax.plot(scale_range_x, scale_range_y, color='w', lw=3, zorder=10)
    ax.text(
        text_pos_x - scale_length_arcsec / 2,
        text_pos_y - 5.0,
        f"{int(scale_length)} AU",
        ha='center', va='bottom',
        fontsize=14, family='Times New Roman', color='w'
    )

    # --- 加上方向箭頭 (NE arrow) ---
    # ax.quiver(
    #     -2, -2,  # 起點 (RA, Dec offset)
    #     -1, -1,  # 指向左下：代表 N 與 E 的方向
    #     color='lightgrey', scale=12, zorder=10
    # )

    ax.set_xlim(45, -45)
    ax.set_ylim(-45, 45)

    # --- Beam 標示 ---
    try:
        bmaj = header.get("BMAJ", None)
        bmin = header.get("BMIN", None)
        bpa  = header.get("BPA", 0.0)
        if bmaj and bmin:
            bmaj_arcsec = bmaj * 3600.0
            bmin_arcsec = bmin * 3600.0
            beam_x = extent[0] + 0.22 * (extent[1] - extent[0])
            beam_y = extent[2] + 0.2 * (extent[3] - extent[2])
            beam_angle = 90 - bpa   #把天文定義轉成 mpl 角度
            beam = Ellipse(
                (beam_x, beam_y),
                width=bmaj_arcsec,   # 長軸
                height=bmin_arcsec,  # 短軸
                angle=beam_angle,    
                facecolor='none',
                edgecolor='white', lw=1
            )
            ax.add_patch(beam)
            ax.text(
                beam_x, beam_y - 0.6 * bmaj_arcsec,
                f"{bmaj_arcsec:.2f}″ × {bmin_arcsec:.2f}″",
                color='white', fontsize=10, ha='center', va='top',
                bbox=dict(facecolor='black', alpha=0, lw=0)
            )
    except Exception as e:
        print(f"[Warning] Beam info not found or invalid: {e}")

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, outname), dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_streamer_on_mom1(theta_deg, phi_deg, inc_deg, T_Myr, omega,
                          header, pa_rad, dx_au, im_center,
                          mom1, label, outname,
                          cen_x_AU=None, cen_z_AU=None, cen_v_LS_km=None,
                          v_range=1.0):
    """
    在 moment-1 圖上畫出 best-fit PSS 流線，並可選擇疊加質心點。
    這版跟 SCrA 保持一致：
      - 主圖方形 (RA/Dec 1:1)
      - colorbar 貼右側，不撐壞主圖
      - 質心點（若有提供）一起畫在圖上
    """

    # 角度轉成 rad
    theta = np.deg2rad(theta_deg)
    phi   = np.deg2rad(phi_deg)
    inc   = np.deg2rad(inc_deg)

    ny, nx = mom1.shape
    extent, dx_arcsec, dz_arcsec = _compute_extent(header, im_center, ny, nx)

    # ---------- PSS model ----------
    x_m, y_m, z_m, u_m, v_m, w_m = pss.PSS_model(
        theta, phi, inc, T_Myr, omega,
        M_star,
        radius_in_au=2e3,
        radius_out_au=7e3,       # 原本你用 7e3，可視需要再調
        resolution=200,
        scale=scale,
        log_power=log_power,
    )

    x_pix = x_m / dx_au
    z_pix = z_m / dx_au

    x_pix_rot = x_pix * np.cos(pa_rad) - z_pix * np.sin(pa_rad) + im_center[1]
    z_pix_rot = x_pix * np.sin(pa_rad) + z_pix * np.cos(pa_rad) + im_center[0]

    ra_off  = (x_pix_rot - im_center[1]) * dx_arcsec
    dec_off = (z_pix_rot - im_center[0]) * dz_arcsec

    # ---------- 建 model 線 segments ----------
    pts = np.column_stack([ra_off, dec_off])
    if pts.shape[0] < 2:
        print("[plot_streamer_on_mom1] model points too few, skip.")
        return
    segments = np.stack([pts[:-1], pts[1:]], axis=1)

    # ---------- 顏色範圍 ----------
    vmin = Local_Standard_Velocity - v_range
    vmax = Local_Standard_Velocity + v_range

    fig, ax = plt.subplots(figsize=(6.2, 6))

    # 背景：moment-1 map
    im = ax.imshow(
        mom1,
        origin="lower",
        cmap="coolwarm",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
    )

    # colorbar 貼右側，小一點
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.04)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Velocity (km/s)")

    # model 線黑色外框 + 內層依速度上色
    lc_edge = LineCollection(segments, colors="black", linewidth=8, zorder=2)
    ax.add_collection(lc_edge)

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    lc = LineCollection(
        segments,
        cmap="coolwarm",
        norm=norm,
        linewidth=6.5,
        zorder=3,
    )
    lc.set_array(v_m + Local_Standard_Velocity)
    ax.add_collection(lc)
    ax.add_collection(lc)
    ax.add_collection(lc)
    ax.add_collection(lc)

    # num_element = 5
    # xarray_arc, z_array_arc = x_array[num_element] * dx_arcsec, z_array[num_element] * dx_arcsec
    # weights_im = ax.scatter( xarray_arc, z_array_arc, c=weights_array[num_element], s=8, cmap='YlGn_r')
    # x_means_arc, z_means_arc = x_means * dx_arcsec, z_means * dx_arcsec
    # ax.plot(x_means_arc, z_means_arc, color='k', lw=3, zorder=4)
    # divider = make_axes_locatable(ax)
    # cax     = divider.append_axes('right', size='3%', pad=0.04)
    # cbar = fig.colorbar(weights_im, cax=cax)
    # cbar.set_label('weight value')

    # ---------- 疊加質心點 ----------
    if cen_x_AU is not None and cen_z_AU is not None:
        # 這裡改成「直接以像素座標」來畫質心點，不再從 AU 反推
        cen_x_pix = np.asarray(cen_x_AU)
        cen_z_pix = np.asarray(cen_z_AU)

        # 轉成與背景 extent 一致的 RA/Dec offset
        cen_ra  = (cen_x_pix - im_center[1]) * dx_arcsec
        cen_dec = (cen_z_pix - im_center[0]) * dz_arcsec

        if cen_v_LS_km is not None:
            cen_v = np.asarray(cen_v_LS_km) + Local_Standard_Velocity
            ax.scatter(
                cen_ra,
                cen_dec,
                c=cen_v,
                cmap="coolwarm",
                vmin=vmin,
                vmax=vmax,
                s=10,
                marker="o",
                edgecolors="black",
                linewidths=0.5,
                zorder=5,
                label="Streamer Centroids",
            )
        else:
            ax.scatter(
                cen_ra,
                cen_dec,
                facecolors="none",
                edgecolors="black",
                s=45,
                marker="o",
                zorder=5,
                label="Streamer Centroids",
            )

    # 中心位置
    ax.scatter(0, 0, c="b", s=60, marker="+", zorder=6)

    ax.set_xlabel("RA Offset (arcsec)")
    ax.set_ylabel("Dec Offset (arcsec)")
    ax.set_title(label)

    # 主圖保持方形比例
    ax.set_aspect("equal", adjustable="box")
    
    # --- 比例尺與方向標示 ---
    # 定義比例尺位置（以 arcsec 為單位）
    text_pos_x = extent[1] - 0.25 * (extent[1] - extent[0])
    text_pos_y = extent[2] + 0.2 * (extent[3] - extent[2])
    scale_length = 3000  # AU

    # 將 3000 AU 轉成 arcsec
    scale_length_arcsec = scale_length / (distance_pc)  # 1" ≈ 1 AU / distance(pc)

    # 定義比例尺線段 (RA offset 軸)
    scale_range_x = [text_pos_x, text_pos_x - scale_length_arcsec]
    scale_range_y = [text_pos_y - 0.2, text_pos_y - 0.2]

    # 繪製比例尺與文字
    ax.plot(scale_range_x, scale_range_y, color='k', lw=3, zorder=10)
    ax.text(
        text_pos_x - scale_length_arcsec / 2,
        text_pos_y - 5.0,
        f"{int(scale_length)} AU",
        ha='center', va='bottom',
        fontsize=14, family='Times New Roman', color='k'
    )

    # # --- 加上方向箭頭 (NE arrow) ---
    # ax.quiver(
    #     -2, -2,  # 起點 (RA, Dec offset)
    #     -1, -1,  # 指向左下：代表 N 與 E 的方向
    #     color='grey', scale=12, zorder=10
    # )
    
    ax.set_xlim(45, -45)
    ax.set_ylim(-45, 45)

    # --- Beam 標示 ---
    try:
        bmaj = header.get("BMAJ", None)
        bmin = header.get("BMIN", None)
        bpa  = header.get("BPA", 0.0)
        if bmaj and bmin:
            bmaj_arcsec = bmaj * 3600.0
            bmin_arcsec = bmin * 3600.0
            beam_x = extent[0] + 0.22 * (extent[1] - extent[0])
            beam_y = extent[2] + 0.2 * (extent[3] - extent[2])
            beam_angle = 90 -bpa   #把天文定義轉成 mpl 角度
            beam = Ellipse(
                (beam_x, beam_y),
                width=bmaj_arcsec,   # 長軸
                height=bmin_arcsec,  # 短軸
                angle=beam_angle,    
                facecolor='none',
                edgecolor='k', lw=1
            )
            ax.add_patch(beam)
            ax.text(
                beam_x, beam_y - 0.55 * bmaj_arcsec,
                f"{bmaj_arcsec:.2f}″ × {bmin_arcsec:.2f}″",
                color='k', fontsize=10, ha='center', va='top'
            )
    except Exception as e:
        print(f"[Warning] Beam info not found or invalid: {e}")

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, outname), dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_r_theta_weights_from_output(x_array, z_array, weights_array,
                                     outname):
    """
    用 extract_streamer_centroids 回傳的
    x_array, z_array, weights_array
    畫出 (r, theta) 的權重分布。
    theta_offset_deg: 畫圖時在角度上加的偏移量（單位：deg）
                      例如 SCrA 想要 0 度在左邊，可以用 180.
    """
    all_r = []
    all_theta = []
    all_w = []

    N = len(x_array)
    for i in range(N):
        x_bin = x_array[i]
        z_bin = z_array[i]
        w_bin = weights_array[i]

        if x_bin.size == 0:
            continue

        r_bin, theta_bin = pss.spherical_coords(x_bin, z_bin)

        all_r.append(r_bin)
        all_theta.append(theta_bin)
        all_w.append(w_bin)

    # 串成一條長向量
    all_r = np.concatenate(all_r)
    theta_all = np.concatenate(all_theta)

    # --- 這裡加角度偏移 ---
    theta_all = theta_all + np.pi
    # wrap 回 [-pi, pi] 比較好看
    theta_all = (theta_all + np.pi) % (2 * np.pi) - np.pi

    all_theta_deg = np.rad2deg(theta_all)
    all_w = np.concatenate(all_w)

    mask = all_w > 0

    fig, ax = plt.subplots(figsize=(6, 4))
    sc = ax.scatter(all_r[mask], all_theta_deg[mask],
                    c=all_w[mask], s=5, cmap="inferno")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Directional weight")

    ax.set_xlabel("r (pixel)")
    ax.set_ylabel(r"$\theta$ (deg)")
    ax.set_title(r"Streamer weight in $(r,\theta)$ space")

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, outname), dpi=200, bbox_inches="tight")
    plt.close(fig)
    
def plot_z_v_imshow(
    best_theta, best_phi, best_incl, best_T, best_omega,
    z_means_AU, v_means_LS_km,
    xx, zz, mom0, mom1,
    pa_rad, dx_au,
    label, outname,
):
    """
    用 imshow 畫 z–v 圖（x 方向疊起來）
    mom0 intensity 作為強度，加總所有 x
    mom1 velocity 用在 v 軸
    """

    # ---- 將影像旋轉到 streamer 座標 (x_rot, z_rot) ----
    x_rot = xx * np.cos(pa_rad) + zz * np.sin(pa_rad)
    z_rot = -xx * np.sin(pa_rad) + zz * np.cos(pa_rad)

    # ---- 轉成 AU ----
    z_all_AU = z_rot * dx_au
    v_all = mom1.copy()

    # ---- 選出 valid pixel ----
    mask = np.isfinite(v_all) & np.isfinite(mom0) & (mom0 > 0)

    z_flat = z_all_AU[mask]
    v_flat = v_all[mask]
    I_flat = mom0[mask]

    # ---- 建立 z–v grid ----
    Nz = 200
    Nv = 200

    z_min, z_max = np.min(z_flat), np.max(z_flat)
    v_min, v_max = np.min(v_flat), np.max(v_flat)

    # 2D 權重圖
    H, z_edges, v_edges = np.histogram2d(
        z_flat, v_flat,
        bins=[Nz, Nv],
        range=[[z_min, z_max], [v_min, v_max]],
        weights=I_flat,
    )

    Zc = 0.5 * (z_edges[:-1] + z_edges[1:])
    Vc = 0.5 * (v_edges[:-1] + v_edges[1:])

    # ---- PSS model ----
    theta = np.deg2rad(best_theta)
    phi   = np.deg2rad(best_phi)
    inc   = np.deg2rad(best_incl)

    x_m, y_m, z_m, u_m, v_m, w_m = pss.PSS_model(
        theta, phi, inc, best_T, best_omega,
        M_star,
        radius_in_au=2e3,
        radius_out_au=7e3,
        resolution=200,
        scale=scale,
        log_power=log_power,
    )
    z_model = z_m
    v_model = v_m + Local_Standard_Velocity

    # ---- data centroids ----
    z_obs = z_means_AU
    v_obs = v_means_LS_km + Local_Standard_Velocity

    # ---- 畫圖 ----
    fig, ax = plt.subplots(figsize=(7, 5))

    im = ax.imshow(
        H.T,
        origin="lower",
        extent=(z_min, z_max, v_min, v_max),
        cmap="inferno",
        aspect="auto",
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\Sigma I_{\rm mom0}$  (Jy beam$^{-1}$ km s$^{-1}$)")

    # 疊上 model
    ax.plot(z_model, v_model, color="cyan", lw=2.5, label="PSS model")

    # 疊上質心
    ax.scatter(z_obs, v_obs, c="white", s=30, edgecolors="k", label="Centroids")

    ax.set_xlabel("Z offset (AU)")
    ax.set_ylabel("Velocity (km/s)")
    ax.set_title(label)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, outname), dpi=200)
    plt.close()
# ============================================================
# 4d. MCMC refinement around grid best-fit (fast likelihood)
# ============================================================
def run_mcmc_grid(
    start_params_rad,
    prior_ranges,
    x_means_AU,
    z_means_AU,
    v_means_LS_km,
    v_weight_phys,
    solar_mass,
    scale,
    log_power,
    nwalkers=64,
    nsteps=3000,
    burnin=1000,
    out_chain="Per-emb-2_mcmc_grid_chain.npz",
):
    """
    以 grid-search 的最佳解為起點，對 5D 參數跑一輪 MCMC (fast likelihood)。
    
    start_params_rad: array-like, [Theta0, Phi0, Incl, T_Myr, Omega] (Theta/Phi/Incl 用 rad)
    """
    ndim = 5
    start_params_rad = np.asarray(start_params_rad, dtype=float)

    # 初始 walkers：在最佳解附近加一點小亂數
    # 角度用 ~0.02 rad 的擾動，Time / Omega 用相對 5% 的擾動
    sigma_theta = 0.02
    sigma_phi   = 0.02
    sigma_incl  = 0.02
    sigma_T     = 0.05 * start_params_rad[3] if start_params_rad[3] > 0 else 0.01
    sigma_Omega = 0.05 * start_params_rad[4] if start_params_rad[4] > 0 else 0.02

    p0 = np.zeros((nwalkers, ndim), dtype=float)
    for i in range(nwalkers):
        trial = start_params_rad.copy()
        trial[0] += np.random.normal(scale=sigma_theta)
        trial[1] += np.random.normal(scale=sigma_phi)
        trial[2] += np.random.normal(scale=sigma_incl)
        trial[3] += np.random.normal(scale=sigma_T)
        trial[4] += np.random.normal(scale=sigma_Omega)
        p0[i] = trial

    # emcee sampler，直接用 PSSpy 裡的 log_posterior_fast
    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        pss.log_posterior_fast,
        args=(prior_ranges, x_means_AU, z_means_AU, v_means_LS_km,
              v_weight_phys, solar_mass, scale, log_power),
    )

    print(f"\n[MCMC_grid] Start sampling: nwalkers={nwalkers}, nsteps={nsteps}")
    _ = sampler.run_mcmc(p0, nsteps, progress=True)

    # 取掉前 burn-in 步
    flat_samples = sampler.get_chain(discard=burnin, thin=1, flat=True)
    print(f"[MCMC_grid] Flat samples shape = {flat_samples.shape}")

    # 用 PSSpy 的 helper 印出結果 & 拿回最佳參數 (rad)
    best_params_rad = pss.report_and_get_best_params(flat_samples, confidence_level=68.3)

    # 存檔 chain
    np.savez(
        out_chain,
        chain=flat_samples,
        start_params=start_params_rad,
        best_params=best_params_rad,
    )

    return best_params_rad, flat_samples

# ============================================================
# 5. 若已存在 cache 且選擇使用，載入最佳解並跳過 grid search
# ============================================================
if USE_CACHED_FIT and os.path.exists(FIT_CACHE):
    print(f"[Cache] Loading cached fit from {FIT_CACHE}, skip grid search.")
    cache = np.load(FIT_CACHE)
    best_theta = float(cache["best_theta"])   # deg
    best_phi   = float(cache["best_phi"])     # deg
    best_T     = float(cache["best_T"])       # Myr
    best_incl  = float(cache["best_incl"])    # deg
    best_omega = float(cache["best_omega"])

    if "v_weight_phys" in cache.files:
        v_weight_phys = float(cache["v_weight_phys"])
    if "r_ref_AU" in cache.files:
        r_ref_AU = float(cache["r_ref_AU"])
    if "M_0" in cache.files:
        M_0 = float(cache["M_0"])
    if "Mdot" in cache.files:
        M_dot = float(cache["Mdot"])
    if "pos_rmse_AU" in cache.files:
        pos_rmse_AU = float(cache["pos_rmse_AU"])
    if "vel_rmse_kms" in cache.files:
        vel_rmse_kms = float(cache["vel_rmse_kms"])
    if "eq_rmse_kms" in cache.files:
        eq_rmse_kms = float(cache["eq_rmse_kms"])

    print("\n==================== Cached Best-fit Parameters ====================")
    print(f"Theta        = {best_theta:.3f} deg")
    print(f"Phi          = {best_phi:.3f} deg")
    print(f"Time (T_Myr) = {best_T:.6f} Myr")
    print(f"Inclination  = {best_incl:.3f} deg")
    print(f"Omega        = {best_omega:.4f}")
    print("===================================================================")
    use_cached = True
else:
    use_cached = False

# ============================================================
# 6. 如果沒有 cache：跑 coarse(+refine) grid，得到 grid best-fit
# ============================================================
if not use_cached:
    v_weight_phys = (dx_au / dv) ** 2

    # --- Stage 1: coarse grid ---
    n_theta, n_phi, n_T_Myr, n_Incl, n_Omega = 10, 10, 10, 10, 10
    theta = np.linspace(0, np.pi, n_theta+1)[1:]
    phi   = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    T_Myr = np.linspace(1e-1, 5e-1, n_T_Myr)
    Incl  = np.linspace(-np.pi/2, np.pi/2, n_Incl+2)[1:-1]
    omega = np.linspace(0, 1, n_Omega+1)[1:]

    error = np.zeros((n_theta, n_phi, n_T_Myr, n_Incl, n_Omega))

    print("\n[Grid Search Stage 1] coarse grid running ...")
    total_iter = n_theta * n_phi * n_T_Myr * n_Incl * n_Omega
    with tqdm(total=total_iter, ncols=80) as pbar:
        for i_theta in range(n_theta):
            for i_phi in range(n_phi):
                for i_T in range(n_T_Myr):
                    for i_I in range(n_Incl):
                        for i_O in range(n_Omega):
                            error[i_theta, i_phi, i_T, i_I, i_O] = pss.error_function(
                                [theta[i_theta], phi[i_phi]],
                                x_means_AU, z_means_AU, v_means_LS_km,
                                v_weight_phys, T_Myr[i_T], omega[i_O], Incl[i_I],
                                M_star, scale='log', log_power=log_power)
                            pbar.update(1)

    min_idx = np.unravel_index(np.argmin(error), error.shape)
    min_theta, min_phi, min_T, min_I, min_O = min_idx

    print(f"→ Minimum found at indices {min_idx}")
    print(f"Theta={np.rad2deg(theta[min_theta]):.2f}°, Phi={np.rad2deg(phi[min_phi]):.2f}°, "
          f"T={T_Myr[min_T]:.5f} Myr, Incl={np.rad2deg(Incl[min_I]):.2f}°, Omega={omega[min_O]:.3f}")

    # ============================================================
    # 6. 第二階段 refinement grid search
    # ============================================================
    if RUN_REFINE:
        n_theta_r, n_phi_r, n_T_r, n_Incl_r, n_Omega_r = 15, 15, 15, 15, 15

        # ---------- 6a. 從 coarse grid 的低誤差點推 refined 範圍 ----------
        Theta_grid, Phi_grid, T_grid, Incl_grid, Omega_grid = np.meshgrid(
            theta, phi, T_Myr, Incl, omega, indexing='ij'
        )

        Theta_flat = Theta_grid.ravel()
        Phi_flat   = Phi_grid.ravel()
        T_flat     = T_grid.ravel()
        Incl_flat  = Incl_grid.ravel()
        Omega_flat = Omega_grid.ravel()
        err_flat   = error.ravel()

        best_val = error[min_idx]

        # 挑「夠好」的格點：先用 5%，不夠再 10%，再不夠就取前 1%
        frac = 0.05
        mask_good = err_flat <= best_val * (1.0 + frac)

        if mask_good.sum() < 50:
            frac = 0.10
            mask_good = err_flat <= best_val * (1.0 + frac)
        if mask_good.sum() < 50:
            k = max(50, int(0.01 * err_flat.size))
            idx_sort = np.argsort(err_flat)
            mask_good = np.zeros_like(err_flat, dtype=bool)
            mask_good[idx_sort[:k]] = True

        Theta_good = Theta_flat[mask_good]
        Phi_good   = Phi_flat[mask_good]
        T_good     = T_flat[mask_good]
        Incl_good  = Incl_flat[mask_good]
        Omega_good = Omega_flat[mask_good]

        def padded_range(arr, pad_frac=0.3, lo_abs=None, hi_abs=None):
            """取 5–95% 分位數，加一點 padding，並限制在物理解範圍。"""
            q5, q95 = np.percentile(arr, [5, 95])
            width = q95 - q5
            if width <= 0:
                width = np.std(arr) if np.std(arr) > 0 else 1e-3
                q5 = np.median(arr) - width
                q95 = np.median(arr) + width
            pad = pad_frac * width
            lo = q5 - pad
            hi = q95 + pad
            if lo_abs is not None:
                lo = max(lo, lo_abs)
            if hi_abs is not None:
                hi = min(hi, hi_abs)
            return lo, hi

        theta_lo, theta_hi = padded_range(Theta_good,
                                            lo_abs=0.0,
                                            hi_abs=np.pi)
        phi_lo, phi_hi     = padded_range(Phi_good,
                                            lo_abs=0.0,
                                            hi_abs=2.0*np.pi)
        T_lo, T_hi         = padded_range(T_good,
                                            lo_abs=np.min(T_Myr),
                                            hi_abs=np.max(T_Myr))
        inc_lo, inc_hi     = padded_range(Incl_good,
                                            lo_abs=-0.5*np.pi,
                                            hi_abs=0.5*np.pi)
        omega_lo, omega_hi = padded_range(Omega_good,
                                            lo_abs=0.0,
                                            hi_abs=1.0)

        print("\n[Grid Search Stage 2] refined ranges from low-error region:")
        print(f"Theta   : {np.rad2deg(theta_lo):6.2f}–{np.rad2deg(theta_hi):6.2f} deg")
        print(f"Phi     : {np.rad2deg(phi_lo):6.2f}–{np.rad2deg(phi_hi):6.2f} deg")
        print(f"Incl    : {np.rad2deg(inc_lo):6.2f}–{np.rad2deg(inc_hi):6.2f} deg")
        print(f"T_Myr   : {T_lo:.5f}–{T_hi:.5f} Myr")
        print(f"Omega   : {omega_lo:.3f}–{omega_hi:.3f}")

        # ---------- 6b. 在 refined 範圍上重做 grid ----------
        theta_r = np.linspace(theta_lo, theta_hi, n_theta_r)
        phi_r   = np.linspace(phi_lo,   phi_hi,   n_phi_r)
        T_Myr_r = np.linspace(T_lo,     T_hi,     n_T_r)
        Incl_r  = np.linspace(inc_lo,   inc_hi,   n_Incl_r)
        omega_r = np.linspace(omega_lo, omega_hi, n_Omega_r)

        error_refine = np.zeros((n_theta_r, n_phi_r, n_T_r, n_Incl_r, n_Omega_r))

        print("\n[Grid Search Stage 2] refinement grid running ...")
        total_iter_r = n_theta_r * n_phi_r * n_T_r * n_Incl_r * n_Omega_r
        with tqdm(total=total_iter_r, ncols=80) as pbar:
            for i_theta in range(n_theta_r):
                for i_phi in range(n_phi_r):
                    for i_T in range(n_T_r):
                        for i_I in range(n_Incl_r):
                            for i_O in range(n_Omega_r):
                                error_refine[i_theta, i_phi, i_T, i_I, i_O] = pss.error_function(
                                    [theta_r[i_theta], phi_r[i_phi]],
                                    x_means_AU, z_means_AU, v_means_LS_km,
                                    v_weight_phys,
                                    T_Myr_r[i_T], omega_r[i_O], Incl_r[i_I],
                                    M_star, scale='log', log_power=log_power
                                )
                                pbar.update(1)

        min_idx_r = np.unravel_index(np.argmin(error_refine), error_refine.shape)
        min_theta_r, min_phi_r, min_T_r, min_I_r, min_O_r = min_idx_r

        print(f"→ Refined minimum at indices {min_idx_r}")
    else:
        print("[Grid Search Stage 2] Skipped (RUN_REFINE=False)")
        theta_r, phi_r, T_Myr_r, Incl_r, omega_r = theta, phi, T_Myr, Incl, omega
        error_refine = error
        min_theta_r, min_phi_r, min_T_r, min_I_r, min_O_r = min_theta, min_phi, min_T, min_I, min_O
        min_idx_r = (min_theta_r, min_phi_r, min_T_r, min_I_r, min_O_r)
    # ============================================================
    # 7. 最終結果輸出
    # ============================================================
    best_theta = np.rad2deg(theta_r[min_theta_r])
    best_phi   = np.rad2deg(phi_r[min_phi_r])
    best_T     = T_Myr_r[min_T_r]
    best_incl  = np.rad2deg(Incl_r[min_I_r])
    best_omega = omega_r[min_O_r]
    best_error = error_refine[min_idx_r]

    r_ref_AU = 200 * best_T * 1e6 * spc.year / spc.astronomical_unit
    M_0 = M_star * M_SUN_KG * spc.G / (200**3 * best_T * 1e6 * spc.year)
    M_dot = M_star / (best_T * 1e6)

    print("\n==================== Final Best-fit Parameters ====================")
    print(f"Theta        = {best_theta:.3f} deg")
    print(f"Phi          = {best_phi:.3f} deg")
    print(f"Time (T_Myr) = {best_T:.6f} Myr")
    print(f"Inclination  = {best_incl:.3f} deg")
    print(f"Omega        = {best_omega:.4f}")
    print(f"r_ref        = {r_ref_AU:.3f} AU")
    print(f"M_0          = {M_0:.3e} (dimensionless)")
    print(f"Mdot         = {M_dot:.3e} M_sun/yr")
    print("===================================================================")

    # 儲存最佳解到 cache，之後調整畫圖時可直接載入
    np.savez(
        FIT_CACHE,
        best_theta=best_theta,
        best_phi=best_phi,
        best_T=best_T,
        best_incl=best_incl,
        best_omega=best_omega,
        v_weight_phys=v_weight_phys,
        pa_deg=pa_deg,
        # --- 物理與統計結果一併存入 ---
        r_ref_AU=r_ref_AU,
        M_0=M_0,
        Mdot=M_dot,
    )

    # ============================================================
    # 8. Optional visualization
    # ============================================================
    # plt.figure(figsize=(6,6))
    # plt.imshow(np.nanmin(error_refine, axis=(2,3,4)), origin='lower', cmap='viridis')
    # plt.title("Refined Grid Search: min(error) vs θ–φ")
    # plt.xlabel("φ index"); plt.ylabel("θ index")
    # plt.colorbar(label="Error (AU-weighted)")
    # plt.tight_layout()
    # plt.savefig(os.path.join(PLOT_DIR, "Per-emb-2_error_theta_phi.png"), dpi=180)
    # plt.close()

# ============================================================
#  MCMC 使用的參數先驗範圍 (rad / Myr / dimensionless)
# ============================================================
parameter_prior_ranges = {
    "Theta zero": (0.0, np.pi),                 # 0–180 deg
    "Phi zero":   (0.0, 2.0 * np.pi),           # 0–360 deg
    "Inclination":(-0.5 * np.pi, 0.5 * np.pi),  # -90–90 deg
    "Time":       (0.1, 0.5),                   # Myr
    "Omega":      (0.0, 1.0),                   # dimensionless
}
# ============================================================
# 7. 若開啟 RUN_MCMC_GRID：在「目前」best 上再 refine 一次
# ============================================================
if RUN_MCMC_GRID:
    start_params_rad = np.array([
        np.deg2rad(best_theta),
        np.deg2rad(best_phi),
        np.deg2rad(best_incl),
        best_T,
        best_omega,
    ])

    best_params_mcmc_rad, mcmc_samples = run_mcmc_grid(
        start_params_rad,
        parameter_prior_ranges,
        x_means_AU,
        z_means_AU,
        v_means_LS_km,
        v_weight_phys,
        M_star,
        scale,
        log_power,
        nwalkers=1000,
        nsteps=20000,
        burnin=100,
        out_chain="Per-emb-2_mcmc_grid_chain.npz",
    )

    best_theta_rad, best_phi_rad, best_incl_rad, best_T, best_omega = best_params_mcmc_rad
    best_theta = np.rad2deg(best_theta_rad)
    best_phi   = np.rad2deg(best_phi_rad)
    best_incl  = np.rad2deg(best_incl_rad)

    r_ref_AU = 200 * best_T * 1e6 * spc.year / spc.astronomical_unit
    M_0 = M_star * M_SUN_KG * spc.G / (200**3 * best_T * 1e6 * spc.year)
    M_dot = M_star / (best_T * 1e6)

    print("\n==================== MCMC-refined Best-fit Parameters ====================")
    print(f"Theta        = {best_theta:.3f} deg")
    print(f"Phi          = {best_phi:.3f} deg")
    print(f"Time (T_Myr) = {best_T:.6f} Myr")
    print(f"Inclination  = {best_incl:.3f} deg")
    print(f"Omega        = {best_omega:.4f}")
    print(f"r_ref        = {r_ref_AU:.3f} AU")
    print(f"M_0          = {M_0:.3e} (dimensionless)")
    print(f"Mdot         = {M_dot:.3e} M_sun/yr")
    print("===================================================================")
    samples_plot = mcmc_samples.copy()
    samples_plot[:, 0] = np.rad2deg(samples_plot[:, 0])  # Theta
    samples_plot[:, 1] = np.rad2deg(samples_plot[:, 1])  # Phi
    samples_plot[:, 2] = np.rad2deg(samples_plot[:, 2])  # Incl

    labels = [
        r"$\Theta_0$ (deg)",
        r"$\Phi_0$ (deg)",
        r"$i$ (deg)",
        r"$T$ (Myr)",
        r"$\omega$",
    ]

    fig = corner.corner(
        samples_plot,
        labels=labels,
        show_titles=True,
        title_fmt=".3f",
        quantiles=[0.16, 0.5, 0.84],
        truths=[best_theta, best_phi, best_incl, best_T, best_omega],
        smooth=1.0
    )
    fig.savefig(os.path.join(PLOT_DIR, "Per-emb-2_mcmc_corner.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    # 這邊直接用 MCMC 的結果存 cache（不要再改回 grid 那組）
    np.savez(
        FIT_CACHE,
        best_theta=best_theta,
        best_phi=best_phi,
        best_T=best_T,
        best_incl=best_incl,
        best_omega=best_omega,
        v_weight_phys=v_weight_phys,
        pa_deg=pa_deg,
        r_ref_AU=r_ref_AU,
        M_0=M_0,
        Mdot=M_dot,
    )

# (前面：cache / grid / refine / MCMC 統統跑完，best_theta 等都已經是「最後版本」)

print("\n[Visualization] Overlaying best-fit streamer model on moment1 map...")

plot_streamer_on_mom1(
    best_theta,
    best_phi,
    best_incl,
    best_T,
    best_omega,
    header,
    pa_rad,
    dx_au,
    im_center,
    data_mom1,
    label='Per-emb-2 ' + r'$\rm HC_3N$',
    outname=f"Per-emb-2_model_vs_mom1_overlay_{pa_deg}.png",
    cen_x_AU=im_center[0] + x_means,
    cen_z_AU=im_center[1] + z_means,
    cen_v_LS_km=v_means_LS_km,
    v_range=1.0,
)

plot_streamer_on_mom0(
    best_theta,
    best_phi,
    best_incl,
    best_T,
    best_omega,
    header,
    pa_rad,
    dx_au,
    im_center,
    data_mom0,
    label='Per-emb-2 ' + r'$\rm HC_3N$',
    outname=f"Per-emb-2_model_vs_mom0_overlay_{pa_deg}.png",
    cen_x_AU=im_center[0] + x_means,
    cen_z_AU=im_center[1] + z_means,
    cen_v_LS_km=v_means_LS_km,
)
print("\n[Visualization] Drawing Z–V diagram...")

plot_z_v_imshow(
    best_theta,
    best_phi,
    best_incl,
    best_T,
    best_omega,
    z_means_AU,
    v_means_LS_km,
    xx, zz,
    data_mom0, data_mom1,
    pa_rad, dx_au,
    label='Per-emb-2 HC3N (Z–V diagram)',
    outname="Per-emb-2_z_v_imshow.png",
)

plot_r_theta_weights_from_output(x_array, z_array, weights_array, outname="Per-emb-2_weights_cacheonly.png")
