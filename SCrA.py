# ============================================================
# S CrA streamer fitting script（整理版）
#
# 區塊結構：
#   1) 參數宣告 / Imports / 開關
#   2) 定義函數 (helpers & MCMC moves)
#   3) 資料前處理：mask streamer、平移至中心、抽質心
#   4) Grid fitting：用 11 點 error_function 找初始解
#   5) 用 grid 結果自動決定 MCMC 先驗範圍
#   6) MCMC_grid   ：11 點 + fast likelihood（選配）
#   7) MCMC_3D     ：(Theta, Phi, Incl) wide prior（選配）
#   8) MCMC_distance：distance_cube + log_posterior（cube，選配）
#   9) 多峰 refinement（選配）
# ============================================================

# ---------- 1. 參數宣告 / Imports / 開關 ----------
# 標準函式庫
import sys
import os
import warnings

# 額外工具
import cv2

# 第三方函式庫 (科學計算/優化)
import emcee
import numpy as np
import scipy.constants as spc

from scipy.interpolate import interp1d

# 天文學/數據處理函式庫
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from spectral_cube import SpectralCube

# 繪圖函式庫
import matplotlib as mpl
import matplotlib.pyplot as plt
import corner
from matplotlib.colors import PowerNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import LineCollection
from matplotlib.ticker import FuncFormatter, FormatStrFormatter

# For grid search progress
from tqdm.auto import tqdm
from itertools import product

# 後驗分析 / 工具
from scipy.signal import find_peaks

# 專案本地模組
import PSSpy as pss
from pss_grid_search import run_grid_search, compute_priors_from_grid

# --- 基本天文參數（S CrA） ---
Local_Standard_Velocity = 5.86  # km/s (Gupta 2024)
pa_deg = 0
pa_rad = np.deg2rad(pa_deg)
distance_pc = 160.0
M_SUN_KG = 1.98847e30
radius_ref_au = 280
M_star = 2

scale = "log"
log_power = 1.5

radius_in_au, radius_out_au = 2.8e2, 5e3
# 資料與輸出
cube_fname = "S_CrA_13CO_spw25_tav_jupyter_shifted.fits"
PLOT_DIR = "S-CrA_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# --- 分析開關 ---
RUN_GRID = False            # 5D grid search 找初始解（建議開）
RUN_MCMC_GRID = False       # 11 個質心點 fast likelihood
RUN_MCMC_3D = False         # 三參數 (Theta, Phi, Incl) 測試
RUN_MCMC_DISTANCE = False    # 使用 distance_cube 的 5D MCMC
RUN_MCMC_GRID_REFINE = False  # 自動對多峰 posterior 做局部 MCMC refinement

# ============================================================
# 2. 定義函數 (helpers & moves)
# ============================================================

def get_mcmc_moves(mode="explore"):
    """
    回傳 emcee moves。
    mode:
      - "explore": 偏探索 (DE + Snooker)
      - "refine" : 偏收斂 (StretchMove)
      - 其他     : 折衷

    Moves 說明：
      - StretchMove：affine-invariant，收斂穩定，適合 refine
      - DEMove：差分進化（Differential Evolution），跳躍性大，探索性強
      - DESnookerMove：特殊反射式 move，有助於跳出局部極值
    """
    if mode == "explore":
        # 以探索為主：DEMove 為主，Snooker 輔助，StretchMove 較少
        return [
            (emcee.moves.StretchMove(a=2.5), 0.3),   # affine-invariant，收斂穩定，適合 refine
            (emcee.moves.DEMove(),           0.6),   # 差分進化，跳躍性大，探索性強
            (emcee.moves.DESnookerMove(),    0.1),   # 反射式 move，有助於跳出局部極值
        ]
    elif mode == "refine":
        # 以收斂為主：StretchMove 為主，DEMove 輔助
        return [
            (emcee.moves.StretchMove(a=2.5), 0.8),   # affine-invariant，收斂穩定，適合 refine
            (emcee.moves.DEMove(),           0.1),   # 差分進化，跳躍性大，探索性強
            (emcee.moves.DESnookerMove(),    0.1),   # 反射式 move，有助於跳出局部極值
        ]
    else:
        # 折衷模式：各 move 均衡
        return [
            (emcee.moves.StretchMove(a=2.5), 0.4),   # affine-invariant，收斂穩定，適合 refine
            (emcee.moves.DEMove(),           0.4),   # 差分進化，跳躍性大，探索性強
            (emcee.moves.DESnookerMove(),    0.2),   # 反射式 move，有助於跳出局部極值
        ]

# --- S CrA streamer masking/centroid helper ---
def build_streamer_masked_cube_scra(subcube, header, rms_channel, im_center):
    """
    S CrA 專用：
    1) 先用大圓遮掉中心，搭配 grow_region 找出 streamer 區域
    2) 再加上兩個額外圓形遮罩清掉多餘 emission
    3) 對整個 streamer cube 做平移，使 streamer 盡量置中
    回傳：
      shifted_cube_data : 平移後的 streamer 資料立方體 (numpy array, NaN=無效)
      shifted_mom0      : 對應 moment0
      shifted_mom1      : 對應 moment1
    """
    ny, nx = subcube.shape[1], subcube.shape[2]

    # 以兩個亮區中點當作初始中心 (這裡沿用你之前使用的座標)
    center1 = (388, 393)  # (y, x)
    center2 = (369, 382)  # (y, x)
    new_center = (int((center1[0] + center2[0]) / 2),
                  int((center1[1] + center2[1]) / 2))
    
    # 1) 大圓遮中心 + grow_region 長出 streamer
    radius_center = 35
    mask2d = pss.circular_mask((ny, nx), new_center, radius_center)
    mask3d = np.repeat(mask2d[np.newaxis, :, :], subcube.shape[0], axis=0)
    masked_center_cube = subcube.with_mask(mask3d)
    maskcent_cube_data = masked_center_cube.filled_data[:].value

    # grow_region 找 streamer
    init_points = [
        (35, new_center[0], new_center[1]),
        (35, 355, 371),
        (35, 355, 340),
        (35, 369, 309),
        (35, 389, 279),
        (35, 463, 257),
    ]
    stream_mask = pss.grow_region(
        maskcent_cube_data,
        init_points,
        rms_channel,
        sigma_thresh=3.5,
        max_iter=1000,
    )

    masked_cube = masked_center_cube.with_mask(stream_mask)
    # 2) 額外兩個圓形遮罩，清掉雜訊/多餘結構
    ny, nx = masked_cube.shape[1], masked_cube.shape[2]

    # # mask 1
    # radius1 = 11
    # pos1 = [320, 412]  # (y, x)
    # m2d_1 = pss.circular_mask((ny, nx), pos1, radius1)
    # m3d_1 = np.repeat(m2d_1[np.newaxis, :, :], masked_cube.shape[0], axis=0)
    # masked_cube = masked_cube.with_mask(m3d_1)
    
    # # mask 2
    # radius2 = 30
    # pos2 = [335, 438]  # (y, x)
    # m2d_2 = pss.circular_mask((ny, nx), pos2, radius2)
    # m3d_2 = np.repeat(m2d_2[np.newaxis, :, :], masked_cube.shape[0], axis=0)
    # masked_cube = masked_cube.with_mask(m3d_2)

    # 3) 對 streamer cube 做平移，讓 new_center 對齊 im_center
    new_cube_data = masked_cube.with_fill_value(np.nan).filled_data[:].value

    nv = new_cube_data.shape[0]
    ty = im_center[0] - new_center[0]
    tx = im_center[1] - new_center[1]
    M = np.float32([[1, 0, tx], [0, 1, ty]])

    shifted_cube_data = np.full_like(new_cube_data, np.nan)
    for v_slice in range(nv):
        shifted_slice = cv2.warpAffine(
            new_cube_data[v_slice],
            M,
            (nx, ny),
            borderValue=np.nan,
        )
        shifted_cube_data[v_slice] = shifted_slice

    # 對平移後的 cube 算 moment
    shifted_mom0 = np.nanmean(shifted_cube_data, axis=0)
    # 使用帶權平均時要避開 NaN
    with np.errstate(invalid="ignore"):
        # 速度軸的實際 km/s 由外部計算，這裡先由 subcube 提供
        spec = subcube.spectral_axis.to(u.km/u.s).value
        spec2d = np.repeat(spec[:, None, None], ny, axis=1)
        spec2d = np.repeat(spec2d, nx, axis=2)
        shifted_mom1 = np.nansum(shifted_cube_data * spec2d, axis=0) / np.nansum(
            shifted_cube_data, axis=0
        )

    return shifted_cube_data, shifted_mom0, shifted_mom1


def extract_streamer_centroids(new_cube_data, spec_kms, header, pa_rad, dx_au):
    """
    從 masked cube 抽出 11 個 streamer 質心點。
    回傳: x_AU, z_AU, v_LSR_km
    """
    cube_shape = new_cube_data.shape
    im_center = (int(header["CRPIX2"]), int(header["CRPIX1"]))

    v, z, x = np.indices(cube_shape)
    x_rel = x - im_center[1]
    z_rel = z - im_center[0]
    r, theta = pss.spherical_coords(x_rel, z_rel)

    # 僅計算一次有效資料 mask，避免在每個 shell 重複呼叫 np.isfinite
    mask_valid = np.isfinite(new_cube_data) & (new_cube_data > 0)

    # 使用實際 streamer 標記點決定流線方向
    find_streamcom = np.array([
        [396, 396],
        [371, 355],
        [340, 355],
        [309, 369],
        [279, 389],
        [257, 463],
    ])

    # ✅ 減掉當初對齊中心 new_center
    center1 = (388, 393)
    center2 = (369, 382)
    new_center = (int((center1[0] + center2[0]) / 2),
                  int((center1[1] + center2[1]) / 2))

    find_streamcom -= np.array([new_center[0], new_center[1]])  # (y, x)
    find_x = find_streamcom[:, 1]
    find_y = find_streamcom[:, 0]
    find_r, find_theta = pss.spherical_coords(find_x, find_y)

    idx = np.argsort(find_r)
    find_r_sorted = find_r[idx]
    find_theta_sorted = find_theta[idx]

    find_streaml = interp1d(
        find_r_sorted,
        find_theta_sorted,
        fill_value=(find_theta_sorted[0], find_theta_sorted[-1]),
        bounds_error=False,
    )

    N = 15
    pars = np.linspace(40, 200, N + 1)

    x_means = np.zeros(N)
    z_means = np.zeros(N)
    v_means = np.zeros(N)
    xzstd   = np.zeros(N)

    # step 1: 幾何中心（以 flux 為權重，不使用方向壓制）
    for i in tqdm(range(N), desc="[SCrA] centroid step1 (geom)", ncols=80, leave=False):
        d = ((r > pars[i]) & (r <= pars[i+1]) & mask_valid)
        if np.any(d):
            w = new_cube_data[d].copy()
            w[~np.isfinite(w)] = 0.0
            if np.sum(w) <= 0:
                x_means[i] = np.nan
                z_means[i] = np.nan
                xzstd[i] = np.nan
            else:
                x_means[i] = np.average(x_rel[d], weights=w)
                z_means[i] = np.average(z_rel[d], weights=w)
                xzstd[i] = np.sqrt(np.average(
                    (x_rel[d] - x_means[i])**2 + (z_rel[d] - z_means[i])**2,
                    weights=w
                ))
        else:
            x_means[i] = np.nan
            z_means[i] = np.nan
            xzstd[i] = np.nan

    # step 2: r-theta, r-std 插值
    valid = np.isfinite(x_means)
    if np.sum(valid) < 2:
        raise RuntimeError("質心點太少，無法建立內插。")

    r_m, theta_m = pss.spherical_coords(x_means[valid], z_means[valid])
    theta_r = interp1d(r_m, theta_m,
                       fill_value=(theta_m[0], theta_m[-1]),
                       bounds_error=False)
    std_r = interp1d(r_m, xzstd[valid],
                     fill_value=(xzstd[0], xzstd[-1]),
                     bounds_error=False)

    # step 3: 高斯權重加上 v（可用 THETA_SIGMA_SCALE 縮窄方向）
    THETA_SIGMA_SCALE = 0.2  # 角度高斯寬度縮放 (<1 更窄, >1 更寬)

    x_means_ref = np.full(N, np.nan)
    z_means_ref = np.full(N, np.nan)
    v_means     = np.full(N, np.nan)

    for i in tqdm(range(N), desc="[SCrA] centroid step2 (vel)", ncols=80, leave=False):
        if not np.isfinite(x_means[i]):
            continue

        r_mid = 0.5 * (pars[i] + pars[i+1])
        theta_ref = theta_r(r_mid)
        sigma_base = std_r(r_mid) / max(r_mid, 1.0)
        sigma_theta = THETA_SIGMA_SCALE * sigma_base
        if not np.isfinite(sigma_theta) or sigma_theta <= 0:
            sigma_theta = np.deg2rad(10.0) * THETA_SIGMA_SCALE

        delta_theta = np.pi - np.abs(np.pi - np.abs(theta - theta_ref))
        gauss_w = np.exp(-0.5 * (delta_theta / sigma_theta)**2)

        d = (r > pars[i]) & (r <= pars[i+1]) & mask_valid
        if not np.any(d):
            continue

        w = new_cube_data[d] * gauss_w[d]
        w[~np.isfinite(w)] = 0.0
        if np.sum(w) <= 0:
            continue

        x_means_ref[i] = np.average(x_rel[d], weights=w)
        z_means_ref[i] = np.average(z_rel[d], weights=w)
        v_vals = spec_kms[v[d]]
        v_means[i] = np.average(v_vals, weights=w)

    # step 4: 轉物理單位
    streamer_v_km = v_means
    streamer_x_pix = x_means_ref
    streamer_z_pix = z_means_ref

    x_rot = streamer_x_pix * np.cos(pa_rad) + streamer_z_pix * np.sin(pa_rad)
    z_rot = -streamer_x_pix * np.sin(pa_rad) + streamer_z_pix * np.cos(pa_rad)

    streamer_x_AU = x_rot * dx_au
    streamer_z_AU = z_rot * dx_au
    streamer_v_LS = streamer_v_km - Local_Standard_Velocity

    print(f"[Extracted] {np.sum(np.isfinite(streamer_x_AU))} valid centroids")
    return streamer_x_AU, streamer_z_AU, streamer_v_LS

def plot_streamer_on_mom1(theta, phi, inc, T_Myr, omega,
                          header, pa_rad, dx_au, im_center,
                          mom1, label, outname,
                          v_range=1.0,
                          vmin=None, vmax=None,
                          radius_in_au=radius_in_au,
                          radius_out_au=radius_out_au,
                          scale='log', log_power=log_power,
                          cen_x_AU=None, cen_z_AU=None, cen_v_LS_km=None):
    """
    在 moment-1 圖上畫出一條 PSS 流線，並可選擇疊加質心點。

    Parameters
    ----------
    theta, phi, inc : float (radian)
        PSS_model 角度參數.
    T_Myr : float
        時間 (Myr)
    omega : float
        無因次角動量參數
    header : FITS header
    pa_rad : float
        影像座標的 position angle (radian)
    dx_au : float
        每個像素對應的 AU
    im_center : (y0, x0)
        影像中心像素座標
    mom1 : 2D array
        moment-1 速度場
    label : str
        圖片標題
    outname : str
        輸出檔名（存在全域 PLOT_DIR）
    cen_x_AU, cen_z_AU, cen_v_LS_km : array-like, optional
        抽出的 streamer 質心點 (已在 sky-frame; 單位 AU / km/s)。
        若提供，會一起畫在圖上。
    """

    # --- 基本尺寸 & 像素刻度 ---
    dx_arcsec = abs(header["CDELT1"]) * 3600.0
    dz_arcsec = abs(header["CDELT2"]) * 3600.0
    ny, nx = mom1.shape
    cy, cx = im_center  # (y, x)

    # --- 建立 PSS model ---
    x_m, y_m, z_m, u_m, v_m, w_m = pss.PSS_model(
        theta, phi, inc, T_Myr, omega,
        M_star,
        radius_in_au=radius_in_au,
        radius_out_au=radius_out_au,
        resolution=200,
        scale=scale,
        log_power=log_power,
    )

    # AU -> 像素
    x_pix = x_m / dx_au
    z_pix = z_m / dx_au

    # 旋轉到影像座標，再平移回以 im_center 為中心的座標
    x_pix_rot = x_pix * np.cos(pa_rad) - z_pix * np.sin(pa_rad) + cx
    z_pix_rot = x_pix * np.sin(pa_rad) + z_pix * np.cos(pa_rad) + cy

    # 轉成 RA/Dec offset（arcsec），以 im_center 為 (0,0)
    ra_off  = (x_pix_rot - cx) * dx_arcsec
    dec_off = (z_pix_rot - cy) * dz_arcsec

    # --- 決定影像的 extent（用真正的視場，不額外手動加負號）---
    ra_min_offset = (0   - cx) * dx_arcsec
    ra_max_offset = (nx - cx) * dx_arcsec
    dec_min_offset = (0   - cy) * dz_arcsec
    dec_max_offset = (ny - cy) * dz_arcsec

    extent = (
        min(ra_min_offset, ra_max_offset),
        max(ra_min_offset, ra_max_offset),
        min(dec_min_offset, dec_max_offset),
        max(dec_min_offset, dec_max_offset),
    )

    # --- 組成 LineCollection ---
    pts = np.column_stack([ra_off, dec_off])
    if pts.shape[0] < 2:
        print("[plot_streamer_on_mom1] model points too few, skip.")
        return
    segments = np.stack([pts[:-1], pts[1:]], axis=1)

    # --- 速度顏色範圍 ---
    if vmin is None or vmax is None:
        vmin = Local_Standard_Velocity - v_range
        vmax = Local_Standard_Velocity + v_range

    fig, ax = plt.subplots(figsize=(6.2, 6))

    # 背景 moment-1
    im = ax.imshow(
        mom1,
        origin="lower",
        cmap="coolwarm",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
    )

    # colorbar（固定在右側，不撐壞主圖）
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.04)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("(km/s)")

    # 黑色外框線（提升可見度）
    lc_edge = LineCollection(segments, colors="black", linewidth=4, zorder=2)
    ax.add_collection(lc_edge)

    # 依 model LOS 速度上色的主線
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    lc = LineCollection(
        segments,
        cmap="coolwarm",
        norm=norm,
        linewidth=2.5,
        zorder=3,
    )
    lc.set_array(v_m + Local_Standard_Velocity)
    ax.add_collection(lc)

    # --- 疊加質心點（如果有提供） ---
    if cen_x_AU is not None and cen_z_AU is not None:
        cen_x_AU = np.asarray(cen_x_AU)
        cen_z_AU = np.asarray(cen_z_AU)

        cen_x_pix = cen_x_AU / dx_au
        cen_z_pix = cen_z_AU / dx_au

        cen_ra = cen_x_pix * dx_arcsec
        cen_dec = cen_z_pix * dz_arcsec

        if cen_v_LS_km is not None:
            cen_v = np.asarray(cen_v_LS_km) + Local_Standard_Velocity
            ax.scatter(
                cen_ra,
                cen_dec,
                c=cen_v,
                cmap="coolwarm",
                vmin=vmin,
                vmax=vmax,
                s=45,
                marker="o",
                edgecolors="black",
                linewidths=1.0,
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
    ax.scatter(0, 0, c="w", s=60, marker="+", zorder=6)

    ax.set_xlabel("RA Offset (arcsec)")
    ax.set_ylabel("Dec Offset (arcsec)")
    ax.set_title(label)

    # 主圖保持方形比例：RA/DEC 1:1
    ax.set_aspect("equal", adjustable="box")

    # 如果你想限制顯示範圍，可以在這裡解開：
    # ax.set_xlim(-12.5, 12.5)
    # ax.set_ylim(-12.5, 12.5)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, outname), dpi=200, bbox_inches="tight")
    plt.close(fig)
    
def summarize_1d_posterior(samples, name, bins=40):
    """
    粗略判斷 1D 後驗分佈的形狀：
      - flat-ish   : 幾乎沒有明顯峰
      - unimodal   : 單峰
      - multimodal : 多峰/結構複雜

    加了「高度門檻」，只數真正明顯的峰，避免被 noise 誤導。
    """
    samples = np.asarray(samples)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        print(f"  {name:<12s}: no samples")
        return

    hist, _ = np.histogram(samples, bins=bins)
    if np.all(hist == 0):
        print(f"  {name:<12s}: flat-ish")
        return

    smooth = np.convolve(hist, [1, 2, 1], mode="same")

    # 只把超過一定高度的局部極大視為「峰」
    peak_mask = (smooth[1:-1] > smooth[:-2]) & (smooth[1:-1] > smooth[2:])
    peaks = smooth[1:-1][peak_mask]

    if peaks.size == 0:
        n_peaks = 0
    else:
        # 門檻：只算超過全域最大值 30% 的峰
        thr = 0.3 * np.max(peaks)
        n_peaks = int(np.sum(peaks >= thr))

    if n_peaks <= 0:
        shape = "flat-ish"
    elif n_peaks == 1:
        shape = "unimodal"
    else:
        shape = "multimodal"

    print(f"  {name:<12s}: {shape}")

def find_posterior_peaks_1d(samples, bins=80, prominence_frac=0.1):
    """
    找出 1D posterior 的主要峰。
    回傳 list[(center, half_width)]，單位與 samples 相同。
    - prominence_frac: 峰值相對高度門檻，避免把雜訊當峰。
    """
    samples = np.asarray(samples)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return []

    hist, edges = np.histogram(samples, bins=bins)
    if np.all(hist == 0):
        return []

    # 用 find_peaks 找出夠明顯的峰
    prominence = prominence_frac * np.max(hist)
    peaks, _ = find_peaks(hist, prominence=prominence)
    if peaks.size == 0:
        return []

    centers = 0.5 * (edges[peaks] + edges[peaks + 1])

    # 半寬：用幾個 bin 當局部 prior 範圍，避免太窄
    bin_width = edges[1] - edges[0]
    half_widths = np.full_like(centers, 3.0 * bin_width)

    return list(zip(centers, half_widths))
# ============================================================
# 3. Grid fitting：用 error_function 找初始解
# ============================================================

# 讀 cube & header
cube = SpectralCube.read(cube_fname)
header = fits.getheader(cube_fname)

im_center = (int(header["CRPIX2"]), int(header["CRPIX1"]))
dx_arcsec = abs(header["CDELT2"]) * 3600.0
dx_au = (abs(header["CDELT1"]) + abs(header["CDELT2"])) / 2 * 3600.0 * distance_pc
dv = abs(float(header["CDELT3"]))
v0 = header["CRVAL3"]

# 速度軸設定（依 S CrA 資料）
v_lastch_vel = 14.8636
v_lastch_num = 150

# velocity 子立方體 & moment
velocity_range = [2.4926, 14.8636] * u.km / u.s
subcube = cube.spectral_slab(velocity_range[0], velocity_range[1])
moment0 = subcube.moment(order=0).value
moment1 = subcube.moment(order=1).value
cube_shape = subcube.shape
spec_kms = subcube.spectral_axis.to(u.km/u.s).value  # 每個 channel 的實際速度

# 儲存原始 moment 圖
fits.PrimaryHDU(data=moment0, header=header).writeto(
    os.path.join("S_CrA_13CO_mom0.fits"), overwrite=True
)
hdu_mom1 = fits.PrimaryHDU(data=moment1, header=header)
hdu_mom1.header["BUNIT"] = "km/s"
hdu_mom1.writeto(
    os.path.join("S_CrA_13CO_mom1.fits"), overwrite=True
)
# 噪音
rms_channel = 0.026211100061251217

# 建立 streamer 專用 cube（遮罩 + 平移）
shifted_cube_data, str_mom0, str_mom1 = build_streamer_masked_cube_scra(
    subcube, header, rms_channel, im_center
)

# 儲存 streamer moment 圖
fits.PrimaryHDU(data=str_mom0, header=header).writeto(
    os.path.join("S_CrA_13CO_streamer_mom0.fits"), overwrite=True
)
fits.PrimaryHDU(data=str_mom1, header=header).writeto(
    os.path.join("S_CrA_13CO_streamer_mom1.fits"), overwrite=True
)

# 從 streamer cube 抽質心
try:
    streamercom_x_AU, streamercom_z_AU, streamercom_v_LS_km = extract_streamer_centroids(
        shifted_cube_data,
        spec_kms,
        header,
        pa_rad,
        dx_au,
    )
except Exception as e:
    print("抽取質心失敗，請檢查 build_streamer_masked_cube_scra 或資料品質：", e)
    raise

# ------------------------------------------------------------
# Quick check: overlay extracted centroids on streamer moment-1
# ------------------------------------------------------------
try:
    # Compute extent for the shifted streamer moment-1 map (str_mom1)
    dx_arcsec_1 = abs(header["CDELT1"]) * 3600.0
    dz_arcsec_1 = abs(header["CDELT2"]) * 3600.0
    rows_1, cols_1 = str_mom1.shape

    w2d_chk = WCS(header).sub(['longitude', 'latitude'])
    nx_chk = header['NAXIS1']
    ny_chk = header['NAXIS2']
    ra0_chk = header['CRVAL1']
    dec0_chk = header['CRVAL2']

    bottom_left_chk = w2d_chk.pixel_to_world(0, 0)
    top_right_chk = w2d_chk.pixel_to_world(nx_chk - 1, ny_chk - 1)

    x1_chk = (bottom_left_chk.ra.deg - ra0_chk) * 3600.0
    x2_chk = (top_right_chk.ra.deg - ra0_chk) * 3600.0
    y1_chk = (bottom_left_chk.dec.deg - dec0_chk) * 3600.0
    y2_chk = (top_right_chk.dec.deg - dec0_chk) * 3600.0

    extent_cent = (min(x1_chk, x2_chk), max(x1_chk, x2_chk),
                min(y1_chk, y2_chk), max(y1_chk, y2_chk))

    # --- Centroid 座標 ---
    dx_arcsec_1 = abs(header["CDELT1"]) * 3600.0
    dz_arcsec_1 = abs(header["CDELT2"]) * 3600.0
    streamer_x_pix = streamercom_x_AU / dx_au
    streamer_z_pix = streamercom_z_AU / dx_au
    streamercom_ra_arcsec  = streamer_x_pix * dx_arcsec_1
    streamercom_dec_arcsec = streamer_z_pix * dz_arcsec_1
    streamercom_v_km = streamercom_v_LS_km + Local_Standard_Velocity

    fig_chk, ax_chk = plt.subplots(figsize=(8.27, 8.27))
    cmap_chk = "coolwarm"
    vmin_chk = Local_Standard_Velocity - 1.0
    vmax_chk = Local_Standard_Velocity + 1.0

    im_chk = ax_chk.imshow(
        str_mom1,
        origin="lower",
        cmap=cmap_chk,
        extent=extent_cent,
        vmin=vmin_chk,
        vmax=vmax_chk,
    )

    divider_chk = make_axes_locatable(ax_chk)
    cax_chk = divider_chk.append_axes("right", size="3%", pad="1%")
    cbar_chk = fig_chk.colorbar(im_chk, cax=cax_chk)
    cbar_chk.set_label("(km/s)", fontsize=14)

    # (0,0) 是平移後的中心
    ax_chk.scatter(0, 0, c="w", s=100, marker="+", zorder=5)

    # Overlay centroids colored by velocity
    ax_chk.scatter(
        streamercom_ra_arcsec,
        streamercom_dec_arcsec,
        c=streamercom_v_km,
        cmap=cmap_chk,
        vmin=vmin_chk,
        vmax=vmax_chk,
        s=50,
        marker="o",
        edgecolors="black",
        linewidths=1,
        label="Streamer Centroids",
        zorder=6,
    )

    ax_chk.set_xlim(-12.5, 12.5)
    ax_chk.set_ylim(-12.5, 12.5)
    ax_chk.set_title("S CrA $^{13}$CO streamer centroids", fontsize=18)
    ax_chk.set_xlabel("RA Offset (arcsec)", fontsize=14)
    ax_chk.set_ylabel("DEC Offset (arcsec)", fontsize=14)
    ax_chk.legend(loc="upper right", fontsize=10)

    fig_chk.tight_layout()
    fig_chk.savefig(
        os.path.join(PLOT_DIR, "streamer_centroids_on_mom1.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig_chk)
except Exception as e:
    print("[plot] 無法繪製 streamer centroids 檢查圖：", e)

# 權重（以物理空間為主）
v_weight_pix = (dv / dx_arcsec) ** 2
v_weight_phys = (np.std(streamercom_x_AU) / np.std(streamercom_v_LS_km))**2
max_dist_value = 100.0

if RUN_GRID:
    best_params, grid, error = run_grid_search(
        streamercom_x_AU, streamercom_z_AU, streamercom_v_LS_km,
        v_weight_phys, M_star, scale, log_power, radius_ref_au
    )

    # 利用 coarse grid 的低誤差區自動產生先驗範圍（與 Per-emb-50 一致）
    parameter_prior_ranges = compute_priors_from_grid(
        error,
        grid,
        best_params["best_val"],
    )

    Theta_init = best_params["Theta"]
    Phi_init   = best_params["Phi"]
    Incl_init  = best_params["Incl"]
    T_init     = best_params["T"]
    Omega_init = best_params["Omega"]

    print("\n[MCMC priors from grid (data-driven)]")
    for name, (lo, hi) in parameter_prior_ranges.items():
        if name in ["Theta zero", "Phi zero", "Inclination"]:
            print(f"{name:<12s}: ({np.rad2deg(lo):6.2f}, {np.rad2deg(hi):6.2f}) deg")
        elif name == "Time":
            print(f"{name:<12s}: ({lo:.5f}, {hi:.5f}) Myr")
        else:
            print(f"{name:<12s}: ({lo:.3f}, {hi:.3f})")
else:
    print("[Grid fitting] Skipped (manual init used).")
# ============================================================
# 4. 先驗範圍 (共用給後面 MCMC)
# ============================================================

# parameter_prior_ranges = {
#     "Theta zero": (Theta_init - np.deg2rad(5), Theta_init + np.deg2rad(10)),
#     "Phi zero":   (Phi_init   - np.deg2rad(30), Phi_init   + np.deg2rad(30)),
#     "Inclination":(Incl_init  - np.deg2rad(10), Incl_init  + np.deg2rad(5)),
#     "Time":       (T_init * 1e-3, T_init * 1e-2),
#     "Omega":      (Omega_init * 1e-3, min(Omega_init * 2, 1.0)),
# }

# # 對 Phi 加 periodic prior（窄一點方便收斂）
# phi_prior_width_deg = 10.0
# phi_half = np.deg2rad(phi_prior_width_deg)
# phi_min = (Phi_init - phi_half) % (2*np.pi)
# phi_max = (Phi_init + phi_half) % (2*np.pi)
# parameter_prior_ranges["Phi zero"] = (phi_min, phi_max)

# print("\n[MCMC priors]")
# for name, (lo, hi) in parameter_prior_ranges.items():
#     if name in ["Theta zero", "Phi zero", "Inclination"]:
#         print(f"{name:<12s}: ({np.rad2deg(lo):6.2f}, {np.rad2deg(hi):6.2f}) deg")
#     elif name == "Time":
#         print(f"{name:<12s}: ({lo:.5f}, {hi:.5f}) Myr")
#     else:
#         print(f"{name:<12s}: ({lo:.3f}, {hi:.3f})")

# ============================================================
# 5. MCMC_grid：fast likelihood（選配）
# ============================================================

if RUN_MCMC_GRID:
    print("\n[MCMC_grid] 使用 log_posterior_fast（11 質心點）")
    ndim = 5
    nwalkers = 20
    nsteps = 15000

    labels_5d = ["Theta zero", "Phi zero", "Inclination", "Time", "Omega"]

    # initial walkers：在先驗範圍內均勻
    # initial walkers：在 grid best-fit 附近打一顆高斯球
    p0 = np.zeros((nwalkers, ndim))

    center_vals = [Theta_init, Phi_init, Incl_init, T_init, Omega_init]
    # 角度用幾度，Time / Omega 用相對比例，避免太散
    sigma_vals  = [
        np.deg2rad(5.0),    # Theta
        np.deg2rad(8.0),    # Phi
        np.deg2rad(8.0),    # Incl
        0.05 * T_init,      # Time
        0.10 * Omega_init,  # Omega
    ]

    for j, key in enumerate(labels_5d):
        lo, hi = parameter_prior_ranges[key]
        proposal = center_vals[j] + sigma_vals[j] * np.random.randn(nwalkers)
        # Phi 是週期的，不過我們 prior 已經是窄區間，clip 即可
        proposal = np.clip(proposal, lo, hi)
        p0[:, j] = proposal

    moves = get_mcmc_moves(mode="refine")

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim,
        pss.log_posterior_fast,
        args=(
            parameter_prior_ranges,
            streamercom_x_AU,
            streamercom_z_AU,
            streamercom_v_LS_km,
            v_weight_phys,
            M_star,
            scale,
            log_power,
        ),
        moves=moves,
    )
    sampler.run_mcmc(p0, nsteps, progress=True)

    # 自動 burn-in / thin
    try:
        tau = sampler.get_autocorr_time()  # 不用 tol=0 這麼硬
        burnin = int(2 * np.nanmax(tau))
        thin   = max(1, int(0.1 * np.nanmin(tau)))  # 原本 0.5 改成 0.1
        print(f"... tau={tau}, burn-in={burnin}, thin={thin}")
        n_eff_steps = (sampler.iteration - burnin) // thin
        ess = (nwalkers * n_eff_steps) / (2.0 * tau)
        print(f"... ESS ≈ {ess}")
    except Exception as e:
        print("[MCMC_grid] tau 估計失敗，用預設。", e)
        burnin, thin = 1000, 50
        n_eff_steps = (sampler.iteration - burnin) // thin
        ess = np.full(ndim, nwalkers * max(n_eff_steps, 1))
        print(f"[MCMC_grid] 粗略 ESS ≈ {ess}")

    flat = sampler.get_chain(discard=burnin, thin=thin, flat=True)

    # --- 對 Phi 做 unwrap，避免 0/2π 斷裂 ---
    phi_ref = Phi_init
    phi_samples = flat[:, 1]
    phi_wrapped = ((phi_samples - phi_ref + np.pi) % (2*np.pi)) - np.pi + phi_ref

    flat_wrapped = flat.copy()
    flat_wrapped[:, 1] = phi_wrapped

    # --- 用 unwrap 後的樣本算統計量 ---
    q16, q50, q84 = np.percentile(flat_wrapped, [16, 50, 84], axis=0)
    Theta_med, Phi_med, Incl_med, T_med, Omega_med = q50

    print("\n[MCMC_grid] 參數的中位數與68%區間：")
    for i, name in enumerate(labels_5d):
        lo, md, hi = q16[i], q50[i], q84[i]
        if i in [0, 1, 2]:
            lo, md, hi = np.rad2deg([lo, md, hi])
            unit = "deg"
        elif name == "Time":
            unit = "Myr"
        else:
            unit = ""
        print(f"{name:12s}: {md:.6f} (+{hi-md:.6f}/-{md-lo:.6f}) {unit}")
    
    print("\n[MCMC_grid] 1D posterior 形狀判斷：")
    for i, name in enumerate(labels_5d):
        summarize_1d_posterior(flat_wrapped[:, i], name)

    # --- Corner plot 也要用 flat_wrapped ---
    # --- Corner plot：unwrap 後再轉成度數 ---
    samples_plot = flat_wrapped.copy()

    # 先把三個角度轉成度
    for idx in [0, 1, 2]:
        samples_plot[:, idx] = np.rad2deg(samples_plot[:, idx])

    labels_plot = ["Theta zero (°)", "Phi zero (°)", "Inclination (°)",
                   "Time (Myr)", "Omega"]

    # 用 16–84% 定義每一維的顯示範圍，避免被少數離群點撐爆
    q16_plot, q50_plot, q84_plot = np.percentile(samples_plot, [16, 50, 84], axis=0)
    ranges = []
    for i, label in enumerate(labels_plot):
        lo, md, hi = q16_plot[i], q50_plot[i], q84_plot[i]
        width = hi - lo
        # fallback: use min/max if width is zero or negative
        if width <= 0:
            data = samples_plot[:, i]
            lo_span, hi_span = np.nanmin(data), np.nanmax(data)
            if np.isfinite(lo_span) and np.isfinite(hi_span) and (hi_span - lo_span) > 0:
                width = hi_span - lo_span
                md = 0.5 * (hi_span + lo_span)
            else:
                width = 1e-3
        # Unified: all parameters, including Theta/Inclination, use centered range logic
        lo_range = md - 1.5 * width
        hi_range = md + 1.5 * width

        # Do NOT clamp for any parameter; all are centered (except Phi, which already has no clamp)
        ranges.append((lo_range, hi_range))
        
    fig = corner.corner(
        samples_plot,
        labels=labels_plot,
        range=ranges,
        show_titles=True,
        title_fmt=".2f",
        plot_datapoints=False,
        fill_contours=True,
        smooth=1.0,
    )

    fig.savefig(os.path.join(PLOT_DIR, "corner_mcmc_grid.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- 用中位數解畫 moment-1 上的線 ---
    plot_streamer_on_mom1(
        Theta_med, Phi_med, Incl_med, T_med, Omega_med,
        header, pa_rad, dx_au, im_center,
        str_mom1,
        label="MCMC_grid median streamline",
        outname="mcmc_grid_median_mom1.png",
        cen_x_AU=streamercom_x_AU,
        cen_z_AU=streamercom_z_AU,
        cen_v_LS_km=streamercom_v_LS_km,
    )
    # ============================================================
    # 5b. 基於 MCMC_grid 的多峰參數，自動做局部 refinement
    # ============================================================

    if RUN_MCMC_GRID_REFINE:
        print("\n[MCMC_grid_refine] 檢查多峰參數並進行局部 MCMC ...")

        # 1) 偵測哪些參數呈現多峰（用 unwrap 後的 flat_wrapped）
        multi_params = {}
        for i, name in enumerate(labels_5d):
            peaks = find_posterior_peaks_1d(
                flat_wrapped[:, i],
                bins=80,
                prominence_frac=0.12
            )
            if len(peaks) > 1:
                multi_params[name] = peaks

        if not multi_params:
            print("[MCMC_grid_refine] 未偵測到明顯多峰，跳過局部 refinement。")
        else:
            print("[MCMC_grid_refine] 偵測到多峰參數：")
            for pname, peaks in multi_params.items():
                if pname in ["Theta zero", "Phi zero", "Inclination"]:
                    centers_deg = [np.rad2deg(c) for c, w in peaks]
                    print(f"  {pname}: {len(peaks)} peaks at {centers_deg} deg")
                else:
                    centers = [c for c, w in peaks]
                    print(f"  {pname}: {len(peaks)} peaks at {centers}")

            # 2) 對每個(參數, 峰)做一個局部 MCMC，先驗只縮那一維
            for pname, peaks in multi_params.items():
                p_index = labels_5d.index(pname)

                for k, (center, half_width) in enumerate(peaks, start=1):
                    print(f"\n[MCMC_grid_refine] {pname} 峰 {k}: center={center:.4g}, half_width={half_width:.4g}")

                    # 建局部 priors：從原本 parameter_prior_ranges 複製
                    local_priors = dict(parameter_prior_ranges)

                    if pname == "Phi zero":
                        # Phi 使用週期邏輯，範圍交給 in_phi_range 處理
                        two_pi = 2.0 * np.pi
                        lo = (center - half_width) % two_pi
                        hi = (center + half_width) % two_pi
                        local_priors["Phi zero"] = (lo, hi)
                    elif pname in ["Theta zero", "Inclination"]:
                        lo = max(center - half_width,
                                 parameter_prior_ranges[pname][0])
                        hi = min(center + half_width,
                                 parameter_prior_ranges[pname][1])
                        local_priors[pname] = (lo, hi)
                    else:
                        # Time / Omega
                        lo = max(center - half_width,
                                 parameter_prior_ranges[pname][0])
                        hi = min(center + half_width,
                                 parameter_prior_ranges[pname][1])
                        local_priors[pname] = (lo, hi)

                    # 3) 初始化局部 walkers：確保在局部 prior 內有夠多「線性獨立」的點
                    ndim_ref = 5
                    nwalkers_ref = 24
                    nsteps_ref = 4000

                    p0_ref = np.zeros((nwalkers_ref, ndim_ref))
                    center_vals = [Theta_med, Phi_med, Incl_med, T_med, Omega_med]
                    sigma_vals = [
                        np.deg2rad(3.0),   # Theta
                        np.deg2rad(5.0),   # Phi
                        np.deg2rad(5.0),   # Incl
                        0.03 * T_med,      # Time
                        0.05 * Omega_med,  # Omega
                    ]

                    for iw in range(nwalkers_ref):
                        trial = np.zeros(ndim_ref)
                        for j, key in enumerate(labels_5d):
                            lo_j, hi_j = local_priors[key]

                            if j == p_index:
                                # 對正在 refinement 的參數，直接在局部 prior 內做均勻抽樣
                                val = lo_j + (hi_j - lo_j) * np.random.rand()
                            else:
                                # 其他參數在全域中位數附近加高斯擾動，再限制在局部 prior 範圍
                                val = center_vals[j] + sigma_vals[j] * np.random.randn()
                                if key == "Phi zero":
                                    # wrap 到 [0, 2π)
                                    val = val % (2.0 * np.pi)
                                val = np.clip(val, lo_j, hi_j)

                            trial[j] = val
                        p0_ref[iw, :] = trial
                        
                    # 確保 walkers 並非完全重合（避免 condition number 太大）
                    # 若發現所有 walker 幾乎一樣，再加一點微小雜訊
                    if np.linalg.matrix_rank(p0_ref) < ndim_ref:
                        eps = 1e-4
                        p0_ref += eps * np.random.randn(*p0_ref.shape)

                    sampler_ref = emcee.EnsembleSampler(
                        nwalkers_ref, ndim_ref,
                        pss.log_posterior_fast,
                        args=(
                            local_priors,
                            streamercom_x_AU,
                            streamercom_z_AU,
                            streamercom_v_LS_km,
                            v_weight_phys,
                            M_star,
                            scale,
                            log_power,
                        ),
                        moves=get_mcmc_moves(mode="refine"),
                    )

                    sampler_ref.run_mcmc(p0_ref, nsteps_ref, progress=True)

                    # 4) 簡單 burn-in / thinning（這裡用穩定保守值就好）
                    burn_ref = int(0.3 * nsteps_ref)
                    thin_ref = 10
                    flat_ref = sampler_ref.get_chain(
                        discard=burn_ref,
                        thin=thin_ref,
                        flat=True
                    )

                    # Phi unwrap
                    flat_ref_wrapped = flat_ref.copy()
                    phi_ref0 = Phi_init
                    phi_s = flat_ref[:, 1]
                    phi_unwrap = ((phi_s - phi_ref0 + np.pi) % (2.0 * np.pi)) - np.pi + phi_ref0
                    flat_ref_wrapped[:, 1] = phi_unwrap

                    # 中位數解
                    q16r, q50r, q84r = np.percentile(flat_ref_wrapped, [16, 50, 84], axis=0)
                    th_r, ph_r, inc_r, T_r, omg_r = q50r

                    print(f"[MCMC_grid_refine] {pname} 峰 {k} 中位數：")
                    print(f"  Theta = {np.rad2deg(th_r):6.2f} deg")
                    print(f"  Phi   = {np.rad2deg(ph_r):6.2f} deg")
                    print(f"  Incl  = {np.rad2deg(inc_r):6.2f} deg")
                    print(f"  T     = {T_r:.5f} Myr")
                    print(f"  Omega = {omg_r:.4f}")

                    # 5) Corner plot（角度轉度）
                    samples_ref_plot = flat_ref_wrapped.copy()
                    for idx in [0, 1, 2]:
                        samples_ref_plot[:, idx] = np.rad2deg(samples_ref_plot[:, idx])

                    labels_ref = ["Theta zero (°)", "Phi zero (°)", "Inclination (°)",
                                  "Time (Myr)", "Omega"]

                    # 動態檢查每個參數範圍（用16–84%+padding，與主corner一致）
                    ranges_ref = []
                    q16_ref, q50_ref, q84_ref = np.percentile(samples_ref_plot, [16, 50, 84], axis=0)
                    for i, label in enumerate(labels_ref):
                        lo, md, hi = q16_ref[i], q50_ref[i], q84_ref[i]
                        width = hi - lo
                        if width <= 0:
                            # fallback: use min/max or small default
                            data = samples_ref_plot[:, i]
                            lo_span, hi_span = np.nanmin(data), np.nanmax(data)
                            if np.isfinite(lo_span) and np.isfinite(hi_span) and (hi_span - lo_span) > 0:
                                width = hi_span - lo_span
                                md = 0.5 * (hi_span + lo_span)
                            else:
                                width = 1e-3
                        # Unified: all parameters, including Theta/Inclination, use centered range logic
                        lo_range = md - 1.5 * width
                        hi_range = md + 1.5 * width
                        # Do NOT clamp for any parameter; all are centered (except Phi, which already has no clamp)
                        ranges_ref.append((lo_range, hi_range))

                    fig_ref = corner.corner(
                        samples_ref_plot,
                        labels=labels_ref,
                        show_titles=True,
                        title_fmt=".2f",
                        plot_datapoints=False,
                        smooth=1.0,
                        fill_contours=True,
                        range=ranges_ref,
                    )

                    corner_name = f"corner_mcmc_grid_refine_{pname.replace(' ', '_')}_peak{k}.png"
                    fig_ref.savefig(
                        os.path.join(PLOT_DIR, corner_name),
                        dpi=200,
                        bbox_inches="tight"
                    )
                    plt.close(fig_ref)

                    # 6) 用此局部中位數模型做 centroid RMSE（方便之後挑最好的一組）
                    rmse_ref = pss.error_function(
                        [th_r, ph_r],
                        streamercom_x_AU,
                        streamercom_z_AU,
                        streamercom_v_LS_km,
                        v_weight_phys,
                        T_r, omg_r, inc_r,
                        M_star,
                        scale,
                        log_power,
                    )
                    print(f"[MCMC_grid_refine] {pname} 峰 {k} centroid RMSE ≈ {rmse_ref:.4f}")

# ============================================================
# 6. MCMC_3D：三參數（選配）
# ============================================================

if RUN_MCMC_3D:
    print("\n[MCMC_3D] wide prior on (Theta, Phi, Incl)")

    def log_prior_3d(params, prior_ranges_3d):
        th, ph, inc = params
        if not (
            prior_ranges_3d["Theta zero"][0] <= th <= prior_ranges_3d["Theta zero"][1]
            and prior_ranges_3d["Phi zero"][0] <= ph <= prior_ranges_3d["Phi zero"][1]
            and prior_ranges_3d["Inclination"][0] <= inc <= prior_ranges_3d["Inclination"][1]
        ):
            return -np.inf
        return 0.0

    def log_post_3d(params, prior_ranges_3d,
                    x_d, z_d, v_d, wv, Ms, T_fix, omg_fix, scale, log_power):
        lp = log_prior_3d(params, prior_ranges_3d)
        if not np.isfinite(lp):
            return -np.inf
        th, ph, inc = params
        rmse = pss.error_function(
            [th, ph],
            x_d, z_d, v_d,
            wv, T_fix, omg_fix, inc,
            Ms, scale, log_power,
        )
        if rmse <= 0 or np.isnan(rmse):
            return -np.inf
        return lp - np.log10(rmse)

    prior_3d = {
        "Theta zero": (0.0, 0.5*np.pi),
        "Phi zero":   (0.0, 2.0*np.pi),
        "Inclination":(-0.5*np.pi, 0.5*np.pi),
    }

    ndim3, nwalkers3, nsteps3 = 3, 48, 8000
    T_fix, omg_fix = T_init, Omega_init

    init3 = np.array([Theta_init, Phi_init, Incl_init])
    p0_3d = np.zeros((nwalkers3, ndim3))
    jitter = np.deg2rad([20.0, 40.0, 30.0])

    for i, key in enumerate(["Theta zero", "Phi zero", "Inclination"]):
        lo, hi = prior_3d[key]
        p = init3[i] + jitter[i] * np.random.randn(nwalkers3)
        p0_3d[:, i] = np.clip(p, lo, hi)

    moves3 = [
        (emcee.moves.StretchMove(a=2.5), 0.2),
        (emcee.moves.DEMove(),           0.8),
    ]

    sampler_3d = emcee.EnsembleSampler(
        nwalkers3, ndim3,
        log_post_3d,
        args=(
            prior_3d,
            streamercom_x_AU,
            streamercom_z_AU,
            streamercom_v_LS_km,
            v_weight_phys,
            M_star,
            T_fix, omg_fix,
            scale, log_power,
        ),
        moves=moves3,
    )
    sampler_3d.run_mcmc(p0_3d, nsteps3, progress=True)

    try:
        tau3 = sampler_3d.get_autocorr_time(tol=0)
        burn3 = int(2*np.max(tau3))
        thin3 = max(1, int(0.5*np.min(tau3)))
        print(f"[MCMC_3D] tau={tau3}, burn-in={burn3}, thin={thin3}")
    except Exception as e:
        print("[MCMC_3D] tau 估計失敗，用預設。", e)
        burn3, thin3 = 1000, 20

    flat3 = sampler_3d.get_chain(discard=burn3, thin=thin3, flat=True)
    samples3 = flat3.copy()
    samples3[:, :3] = np.rad2deg(samples3[:, :3])

    fig3 = corner.corner(samples3,
                         labels=["Theta zero (°)", "Phi zero (°)", "Inclination (°)"],
                         show_titles=True, title_fmt=".2f")
    fig3.savefig(os.path.join(PLOT_DIR, "corner_mcmc_3d.png"), dpi=180)
    plt.close(fig3)

# ============================================================
# 7. MCMC_distance：distance_cube + log_posterior
# ============================================================

if RUN_MCMC_DISTANCE:
    print("\n[MCMC_distance] 使用 distance_cube-based log_posterior")

    data_cube = shifted_cube_data
    cube_shape = data_cube.shape
    v_weight_for_cube = v_weight_pix

    # 用 grid 最佳解建 model 線 → search_bound
    x_m0, y_m0, z_m0, u_m0, v_m0, w_m0 = pss.PSS_model(
        Theta_init, Phi_init, Incl_init, T_init, Omega_init,
        M_star,
        radius_in_au=radius_in_au,
        radius_out_au=radius_out_au,
        resolution=80,
        scale=scale,
        log_power=log_power,
    )
    dx_arcsec = abs(header["CDELT1"]) * 3600.0
    dz_arcsec = abs(header["CDELT2"]) * 3600.0
    x_pix = x_m0 / dx_au
    z_pix = z_m0 / dx_au
    ra_off  = x_pix * dx_arcsec
    dec_off = z_pix * dz_arcsec
    x_best_pix_int = np.round(
        x_pix * np.cos(pa_rad) - z_pix * np.sin(pa_rad) + im_center[1]
    ).astype(int)
    z_best_pix_int = np.round(
        x_pix * np.sin(pa_rad) + z_pix * np.cos(pa_rad) + im_center[0]
    ).astype(int)
    v_best_pix_int = np.round(
        v_lastch_num - (v_m0 - v_lastch_vel + Local_Standard_Velocity) / dv
    ).astype(int)

    search_bound = pss.get_bounding_box(
        x_best_pix_int, z_best_pix_int, v_best_pix_int,
        buffer=6, v_buffer=3, cube_shape=cube_shape
    )
    print("[MCMC_distance] search_bound:", search_bound)

    ndim, nwalkers, nsteps = 5, 20, 1000
    labels_5d = ["Theta zero", "Phi zero", "Inclination", "Time", "Omega"]

    p0 = np.zeros((nwalkers, ndim))
    print("[MCMC_distance] 初始化 walkers (緊貼 grid 解)...")

    center_vals = [Theta_init, Phi_init, Incl_init, T_init, Omega_init]
    sigma_vals  = [
        np.deg2rad(5.0),    # Theta
        np.deg2rad(8.0),    # Phi
        np.deg2rad(8.0),    # Incl
        0.05 * T_init,      # Time
        0.10 * Omega_init,  # Omega
    ]

    for j, key in enumerate(labels_5d):
        lo, hi = parameter_prior_ranges[key]
        proposal = center_vals[j] + sigma_vals[j] * np.random.randn(nwalkers)
        proposal = np.clip(proposal, lo, hi)
        p0[:, j] = proposal

    moves = get_mcmc_moves(mode="refine")

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim,
        pss.log_posterior,
        args=(
            data_cube,
            search_bound,
            parameter_prior_ranges,
            pa_rad,
            dx_au,
            im_center,
            dv,
            v_lastch_vel,
            v_lastch_num,
            v0,
            v_weight_for_cube,
            max_dist_value,
            M_star,
            radius_in_au,
            radius_out_au,
        ),
        moves=moves,
    )
    sampler.run_mcmc(p0, nsteps, progress=True)

    # 自動 burn-in / thin
    try:
        tau = sampler.get_autocorr_time()  # 不用 tol=0 這麼硬
        burnin = int(2 * np.nanmax(tau))
        thin   = max(1, int(0.1 * np.nanmin(tau)))  # 原本 0.5 改成 0.1
        print(f"... tau={tau}, burn-in={burnin}, thin={thin}")
        n_eff_steps = (sampler.iteration - burnin) // thin
        ess = (nwalkers * n_eff_steps) / (2.0 * tau)
        print(f"... ESS ≈ {ess}")
    except Exception as e:
        print("[MCMC_distance] tau 估計失敗，用預設。", e)
        burnin = int(0.2*nsteps)
        thin   = 20
        n_eff_steps = (sampler.iteration - burnin) // thin
        ess = np.full(ndim, nwalkers * max(n_eff_steps, 1))
        print(f"[MCMC_distance] 粗略 ESS ≈ {ess}")

    flat = sampler.get_chain(discard=burnin, thin=thin, flat=True)
    flat_lnprob = sampler.get_log_prob(discard=burnin, thin=thin, flat=True)
    print("[MCMC_distance] flat_samples:", flat.shape)

    # --- Phi 的角度展開修正 ---
    angle_idx = [0, 1, 2]
    phi_samples = flat[:, 1]
    phi_ref = Phi_init
    phi_wrapped = ((phi_samples - phi_ref + np.pi) % (2*np.pi)) - np.pi + phi_ref
    flat_wrapped = flat.copy()
    flat_wrapped[:, 1] = phi_wrapped

    q16, q50, q84 = np.percentile(flat_wrapped, [16, 50, 84], axis=0)

    print("\n[MCMC_distance] 參數中位數與 68% 區間：")
    for i, name in enumerate(labels_5d):
        lo, md, hi = q16[i], q50[i], q84[i]
        if i in angle_idx:
            lo, md, hi = np.rad2deg([lo, md, hi])
            unit = "deg"
        elif name == "Time":
            unit = "Myr"
        else:
            unit = ""
        print(f"{name:12s}: {md:.6f} (+{hi-md:.6f}/-{md-lo:.6f}) {unit}")

    print("\n[MCMC_distance] 1D posterior 形狀判斷：")
    for i, name in enumerate(labels_5d):
        summarize_1d_posterior(flat_wrapped[:, i], name)

    # Corner plot（角度轉度）
    # --- Corner plot：unwrap 後再轉成度數 ---
    # Corner plot：unwrap 後再轉度，並限制顯示範圍
    samples_plot = flat_wrapped.copy()
    for idx in angle_idx:
        samples_plot[:, idx] = np.rad2deg(samples_plot[:, idx])

    labels_plot = ["Theta zero (°)", "Phi zero (°)", "Inclination (°)",
                   "Time (Myr)", "Omega"]
    
    # Compute q16/q50/q84 for samples_plot (posterior-centered)
    q16_plot, q50_plot, q84_plot = np.percentile(samples_plot, [16, 50, 84], axis=0)
    
    ranges = []
    for i, label in enumerate(labels_plot):
        lo, md, hi = q16_plot[i], q50_plot[i], q84_plot[i]
        width = hi - lo
        if width <= 0:
            # fallback: use min/max or small default
            data = samples_plot[:, i]
            lo_span, hi_span = np.nanmin(data), np.nanmax(data)
            if np.isfinite(lo_span) and np.isfinite(hi_span) and (hi_span - lo_span) > 0:
                width = hi_span - lo_span
                md = 0.5 * (hi_span + lo_span)
            else:
                width = 1e-3
        # Unified: all parameters, including Theta/Inclination, use centered range logic
        lo_range = md - 1.5 * width
        hi_range = md + 1.5 * width
        # Do NOT clamp for any parameter; all are centered (except Phi, which already has no clamp)
        ranges.append((lo_range, hi_range))
        
    fig = corner.corner(
        samples_plot,
        labels=labels_plot,
        range=ranges,
        show_titles=True,
        title_fmt=".2f",
        plot_datapoints=False,
        fill_contours=True,
        smooth=1.0,
    )

    fig.savefig(os.path.join(PLOT_DIR, "corner_mcmc_distance.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- 用中位數解畫 v-r 圖 ---
    Theta_med, Phi_med, Incl_med, T_med, Omega_med = q50

    x_m, y_m, z_m, u_m, v_m, w_m = pss.PSS_model(
        Theta_med, Phi_med, Incl_med, T_med, Omega_med,
        M_star,
        radius_in_au=radius_in_au,
        radius_out_au=radius_out_au,
        resolution=200,
        scale=scale,
        log_power=log_power,
    )

    r_model = np.sqrt(x_m**2 + z_m**2)
    r_data = np.sqrt(streamercom_x_AU**2 + streamercom_z_AU**2)

    plt.figure(figsize=(6, 5))
    plt.scatter(r_model, v_m, s=5, alpha=0.3, label="model")
    plt.scatter(r_data, streamercom_v_LS_km,
                c="r", s=30, marker="x", label="centroids")
    plt.xlabel("r (AU)")
    plt.ylabel("v (km/s)")
    plt.legend()
    plt.title("Cube MCMC median model vs data")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "cube_mcmc_median_model_vs_data.png"), dpi=180)
    plt.close()
    
    plot_streamer_on_mom1(
        Theta_med, Phi_med, Incl_med, T_med, Omega_med,
        header, pa_rad, dx_au, im_center,
        str_mom1,
        label="MCMC_distance median streamline",
        outname="mcmc_distance_median_mom1.png",
        cen_x_AU=streamercom_x_AU,
        cen_z_AU=streamercom_z_AU,
        cen_v_LS_km=streamercom_v_LS_km
        )
print("\n全部段落執行完成。")