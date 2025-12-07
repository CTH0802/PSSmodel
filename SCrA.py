# ============================================================
# S CrA streamer fitting script（整理版）
#
# 區塊結構：
#   1) 參數宣告 / Imports / 開關
#   2) 定義函數 (helpers & MCMC moves)
#   3) 資料前處理：mask streamer、平移至中心、抽質心
#   4) Grid fitting：用 32 點 error_function 找初始解
#   5) 用 grid 結果自動決定 MCMC 先驗範圍
#   6) MCMC_grid   ：32 點 + fast likelihood（選配）
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
from multiprocessing import Pool
from emcee.autocorr import AutocorrError
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
pa_deg = 0.0
pa_rad = np.deg2rad(pa_deg)
distance_pc = 160.0
M_SUN_KG = 1.98847e30
radius_ref_au = 280
M_star = 2

scale = "log"
log_power = 1.5

radius_in_au, radius_out_au = 4e2, 1.5e3
# 資料與輸出
cube_fname = "S_CrA_13CO_spw25_tav_jupyter_shifted.fits"
CACHE_DIR = "SCrA_results/cache"
os.makedirs(CACHE_DIR, exist_ok=True)
PLOT_DIR = "SCrA_results/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

CACHE_PATH_GRID = os.path.join(CACHE_DIR, "SCrA_grid_results.npz")
CACHE_PATH_MCMC_GRID = os.path.join(CACHE_DIR, "SCrA_mcmc_grid_results.npz")
CACHE_PATH_MCMC_SHELL = os.path.join(CACHE_DIR, "SCrA_mcmc_shell_results.npz")
CACHE_PATH_FINAL = os.path.join(CACHE_DIR, "SCrA_fit_results_final.npz")

USE_CACHE_SOURCE = "grid"

# --- 分析開關 ---
# RUN_GRID = True               # 5D grid search 找初始解
# RUN_MCMC_GRID = True          # 32 個質心點 fast likelihood
# RUN_MCMC_SHELL = True         # distance_cube MCMC
# RUN_FROM_CACHE_ONLY = False   # True: 僅讀 cache 畫圖，完全不重跑

RUN_GRID = False               # 5D grid search 找初始解
RUN_MCMC_GRID = False          # 32 個質心點 fast likelihood
RUN_MCMC_SHELL = False         # distance_cube MCMC
RUN_FROM_CACHE_ONLY = True   # True: 僅讀 cache 畫圖，完全不重跑

# USE_EDT_ERROR_FOR_GRID = False
# RUN_MCMC_GRID_REFINE = False  # MCMC_grid 多峰局部 refinement
# RUN_MCMC_3D = False           # (Theta, Phi, Incl) 測試

def _resolve_cache_path(source: str) -> str:
    s = (source or "").lower()
    if s == "final":
        return CACHE_PATH_FINAL
    elif s == "mcmc_shell":
        return CACHE_PATH_MCMC_SHELL
    elif s == "mcmc_grid":
        return CACHE_PATH_MCMC_GRID
    elif s == "grid":
        return CACHE_PATH_GRID
    return CACHE_PATH_FINAL

def _extract_params_from_cache(c: dict, source: str):
    s = (source or "").lower()
    def _try(a,b,c1,d,e):
        try:
            return (float(c[a]), float(c[b]), float(c[c1]), float(c[d]), float(c[e]))
        except Exception:
            return None
    if s == "mcmc_shell":
        order = [
            ("mcmc_shell_median_Theta","mcmc_shell_median_Phi","mcmc_shell_median_Incl","mcmc_shell_median_T","mcmc_shell_median_Omega"),
            ("mcmc_grid_median_Theta","mcmc_grid_median_Phi","mcmc_grid_median_Incl","mcmc_grid_median_T","mcmc_grid_median_Omega"),
            ("grid_best_Theta","grid_best_Phi","grid_best_Incl","grid_best_T","grid_best_Omega"),
            ("best_Theta","best_Phi","best_Incl","best_T","best_Omega"),
        ]
    elif s == "mcmc_grid":
        order = [
            ("mcmc_grid_median_Theta","mcmc_grid_median_Phi","mcmc_grid_median_Incl","mcmc_grid_median_T","mcmc_grid_median_Omega"),
            ("grid_best_Theta","grid_best_Phi","grid_best_Incl","grid_best_T","grid_best_Omega"),
            ("best_Theta","best_Phi","best_Incl","best_T","best_Omega"),
        ]
    elif s == "grid":
        order = [
            ("grid_best_Theta","grid_best_Phi","grid_best_Incl","grid_best_T","grid_best_Omega"),
            ("mcmc_grid_median_Theta","mcmc_grid_median_Phi","mcmc_grid_median_Incl","mcmc_grid_median_T","mcmc_grid_median_Omega"),
            ("best_Theta","best_Phi","best_Incl","best_T","best_Omega"),
        ]
    else:
        order = [
            ("best_Theta","best_Phi","best_Incl","best_T","best_Omega"),
            ("mcmc_distance_median_Theta","mcmc_distance_median_Phi","mcmc_distance_median_Incl","mcmc_distance_median_T","mcmc_distance_median_Omega"),
            ("mcmc_grid_median_Theta","mcmc_grid_median_Phi","mcmc_grid_median_Incl","mcmc_grid_median_T","mcmc_grid_median_Omega"),
            ("grid_best_Theta","grid_best_Phi","grid_best_Incl","grid_best_T","grid_best_Omega"),
        ]
    for keys in order:
        out = _try(*keys)
        if out: return out
    raise KeyError("Cache missing parameters.")


# 統一 cache 容器（本次執行逐步填入）
cache = {
    "target": "SCrA",
    "cube_fname": cube_fname,
    "distance_pc": distance_pc,
    "M_star": M_star,
    "pa_deg": pa_deg,
    "Local_Standard_Velocity": Local_Standard_Velocity,
    "radius_in_au": radius_in_au,
    "radius_out_au": radius_out_au,
}


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
def build_streamer_masked_cube_scra(subcube, rms_channel, im_center):
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


def extract_streamer_centroids(new_cube_data, header, pa_rad, dx_au):
    """
    從 masked S CrA cube 抽出 streamer 質心點，並且：
      - 在 image frame 中，以 protostar (CRPIX) 為原點定義 (x_rel, z_rel)
      - 再用 -PA 旋轉到 model frame，輸出 streamer_x_AU, streamer_z_AU
    這樣的 frame 流程與 Per-emb-50 的版本一致。

    回傳:
      streamer_x_AU  : model frame x (AU)
      streamer_z_AU  : model frame z (AU)
      streamer_v_LS  : LOS velocity (km/s, 相對 LSR)
      x_array, z_array, v_array, weights_array : 每個 shell 中 voxel 的紀錄（list of arrays）
      x_means_ref, z_means_ref, v_means        : image frame 下的質心 (pixel, km/s)
    """

    cube_shape = new_cube_data.shape  # (nv, ny, nx)
    im_center = (int(header["CRPIX2"]), int(header["CRPIX1"]))  # (y, x)

    # 建立 voxel 座標
    v, z, x = np.indices(cube_shape)
    x_rel = x - im_center[1]
    z_rel = z - im_center[0]
    r, theta = pss.spherical_coords(x_rel, z_rel)
    
    stream_points_abs = np.array([
    [396, 396],
    [355, 371],
    [355, 340],
    [369, 309],
    [389, 279],
    [463, 257],
    ])
    stream_points = stream_points_abs - np.array([im_center[1], im_center[0]])
    find_r, find_theta = pss.spherical_coords(stream_points[:, 0],
                                              stream_points[:, 1])
    find_streaml = interp1d(
        find_r,
        find_theta,
        fill_value=(find_theta[0], find_theta[-1]),
        bounds_error=False,
    )
    
    N = 32
    pars = np.linspace(40, 200, N + 1)

    x_means = np.zeros(N)
    z_means = np.zeros(N)
    v_means = np.zeros(N)   # 先暫存 channel index
    xzstd   = np.zeros(N)

    x_array_list = []
    z_array_list = []
    v_array_list = []
    weights_list = []
    
    for i in tqdm(range(N), desc="[SCrA] centroid step1 (pos.)", ncols=80, leave=False):
        r_mid = 0.5 * (pars[i] + pars[i+1])
        theta0 = find_streaml(r_mid)

        # 沿著 streamer 方向的 cos(angle) 當作權重
        weight_theta = (x_rel * np.cos(theta0) + z_rel * np.sin(theta0)) / r
        weight_theta[~np.isfinite(weight_theta)] = 0.0
        weight_theta[weight_theta < 0.9] = 0.0

        shell = (r > pars[i]) & (r <= pars[i+1]) & (new_cube_data > 0) & np.isfinite(new_cube_data)
        if np.sum(shell) > 0:
            w = new_cube_data[shell] * weight_theta[shell]
            if np.sum(w) <= 0:
                x_means[i] = z_means[i] = xzstd[i] = np.nan
            else:
                x_means[i] = np.average(x_rel[shell], weights=w)
                z_means[i] = np.average(z_rel[shell], weights=w)
                xzstd[i] = np.sqrt(np.average(
                    (x_rel[shell] - x_means[i])**2 +
                    (z_rel[shell] - z_means[i])**2,
                    weights=w
                ))
        else:
            x_means[i] = z_means[i] = xzstd[i] = np.nan

    # ------------------------------------------------------------
    # Step 2: 用 xzstd 決定 angular Gaussian，重新算位置+速度
    # ------------------------------------------------------------
    valid = np.isfinite(x_means) & np.isfinite(z_means) & np.isfinite(xzstd)
    if np.sum(valid) < 2:
        raise RuntimeError("質心點太少，無法建立內插。")

    r_m, theta_m = pss.spherical_coords(x_means[valid], z_means[valid])
    theta_r = interp1d(
        r_m, theta_m,
        fill_value=(theta_m[0], theta_m[-1]),
        bounds_error=False,
    )
    std_r = interp1d(
        r_m, xzstd[valid],
        fill_value=(xzstd[0], xzstd[-1]),
        bounds_error=False,
    )
    
    x_means_ref = np.zeros(N)
    z_means_ref = np.zeros(N)
    
    for i in tqdm(range(N), desc="[SCrA] centroid step2 (vel.)", ncols=80, leave=False):
        r_mid = 0.5 * (pars[i] + pars[i+1])

        if not np.isfinite(x_means[i]):
            x_means_ref[i] = z_means_ref[i] = v_means[i] = np.nan
            x_array_list.append(np.array([]))
            z_array_list.append(np.array([]))
            v_array_list.append(np.array([]))
            weights_list.append(np.array([]))
            continue

        theta_ref = theta_r(r_mid)
        std_ref   = std_r(r_mid) / max(r_mid, 1.0)

        # 角距離 + Gaussian
        delta_theta = np.pi - np.abs(np.pi - np.abs(theta - theta_ref))
        weights = new_cube_data * pss.gaussian(delta_theta, 0, std_ref)

        d = (r > pars[i]) & (r <= pars[i+1]) & (new_cube_data > 0) & (np.isfinite(new_cube_data))
        if np.sum(d) > 0 and np.sum(weights[d]) > 0:
            x_means_ref[i] = np.average(x_rel[d], weights=weights[d])
            z_means_ref[i] = np.average(z_rel[d], weights=weights[d])
            v_means[i]     = np.average(v[d],    weights=weights[d])
        else:
            x_means_ref[i] = z_means_ref[i] = v_means[i] = np.nan

        # 存這個 shell 裡所有 voxel 的資訊（之後畫圖用）
        x_array_list.append(x_rel[d])
        z_array_list.append(z_rel[d])
        v_array_list.append(v[d])
        weights_list.append(weights[d] / np.nanmax(weights[d]))


    # ------------------------------------------------------------
    # Step 3: 轉成 model frame + AU + km/s
    # ------------------------------------------------------------
    x_img_pix = x_means_ref
    z_img_pix = z_means_ref

    x_model_pix = x_img_pix * np.cos(pa_rad) + z_img_pix * np.sin(pa_rad)
    z_model_pix = -x_img_pix * np.sin(pa_rad) + z_img_pix * np.cos(pa_rad)

    streamer_x_AU = x_model_pix * dx_au
    streamer_z_AU = z_model_pix * dx_au

    dv     = float(header["CDELT3"])
    v0     = float(header["CRVAL3"])
    crpix3 = float(header["CRPIX3"])

    streamer_v_km = v0 + (v_means + 1 - crpix3) * dv
    streamer_v_LS = streamer_v_km - Local_Standard_Velocity

    print(f"[Extracted] {np.sum(np.isfinite(streamer_x_AU))} valid centroids (3D cube)")

    x_array = np.array(x_array_list, dtype=object)
    z_array = np.array(z_array_list, dtype=object)
    v_array = np.array(v_array_list, dtype=object)
    weights_array = np.array(weights_list, dtype=object)

    return (
        streamer_x_AU,
        streamer_z_AU,
        streamer_v_LS,
        x_array,
        z_array,
        v_array,
        weights_array,
        x_means_ref,
        z_means_ref,
        v_means,
    )

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

def summarize_1d_posterior(samples, name, bins=30):
    samples = np.asarray(samples)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        print(f"  {name:<12s}: no samples")
        return

    hist, _ = np.histogram(samples, bins=bins)
    if np.all(hist == 0):
        print(f"  {name:<12s}: flat-ish")
        return

    smooth = np.convolve(hist, [1, 4, 6, 4, 1], mode="same")
    peak_mask = (smooth[1:-1] > smooth[:-2]) & (smooth[1:-1] > smooth[2:])
    peaks = smooth[1:-1][peak_mask]
    if peaks.size == 0:
        n_peaks = 0
    else:
        thr = 0.5 * np.max(peaks)
        n_peaks = int(np.sum(peaks >= thr))

    if n_peaks <= 0:
        shape = "flat-ish"
    elif n_peaks == 1:
        shape = "unimodal"
    else:
        shape = "multimodal"
    print(f"  {name:<12s}: {shape}")

def _compute_extent(header, im_center, ny, nx):
    dx_arcsec = header["CDELT1"] * 3600.0
    dz_arcsec = header["CDELT2"] * 3600.0
    ra_min = (0   - im_center[1]) * dx_arcsec
    ra_max = (nx  - im_center[1]) * dx_arcsec
    dec_min= (0   - im_center[0]) * dz_arcsec
    dec_max= (ny  - im_center[0]) * dz_arcsec
    return (ra_min, ra_max, dec_min, dec_max), dx_arcsec, dz_arcsec

def plot_streamer_on_mom0(theta, phi, inc, T_Myr, omega,
                          header, pa_rad, dx_au, im_center,
                          mom0, label, outname,
                          v_range=1.0,
                          vmin=None, vmax=None,
                          radius_in_au=radius_in_au,
                          radius_out_au=radius_out_au,
                          scale='log', log_power=log_power,
                          cen_x_pix=None, cen_z_pix=None, cen_v_LS_km=None):

    # --- 基本尺寸 & 像素刻度 ---
    dx_arcsec = abs(header["CDELT1"]) * 3600.0
    dz_arcsec = abs(header["CDELT2"]) * 3600.0
    ny, nx = mom0.shape
    extent, dx_arcsec, dz_arcsec = _compute_extent(header, im_center, ny, nx)

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

    x_pix_rot = x_pix * np.cos(pa_rad) - z_pix * np.sin(pa_rad) + im_center[1]
    z_pix_rot = x_pix * np.sin(pa_rad) + z_pix * np.cos(pa_rad) + im_center[0]

    ra_off  = (x_pix_rot - im_center[1]) * dx_arcsec
    dec_off = (z_pix_rot - im_center[0]) * dz_arcsec

    # --- 組成 LineCollection ---
    pts = np.column_stack([ra_off, dec_off])
    if pts.shape[0] < 2:
        print("[plot_streamer_on_mom0] model points too few, skip.")
        return
    segments = np.stack([pts[:-1], pts[1:]], axis=1)

    # --- 速度顏色範圍 ---
    if vmin is None or vmax is None:
        vmin = Local_Standard_Velocity - v_range
        vmax = Local_Standard_Velocity + v_range

    fig, ax = plt.subplots(figsize=(6.2, 6))
    norm = PowerNorm(gamma=0.5,
                     vmin=np.nanmin(mom0),
                     vmax=np.nanmax(mom0))

    im = ax.imshow(
        mom0,
        origin="lower",
        cmap="inferno",
        extent=extent,
        norm=norm,
        # vmin=np.nanmin(mom0),
        # vmax=np.nanmax(mom0)
    )

    # colorbar（固定在右側，不撐壞主圖）
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.04)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("(Jy/beam km/s)")

    # # 黑色外框線（提升可見度）
    # lc_edge = LineCollection(segments, colors="black", linewidth=4, zorder=2)
    # ax.add_collection(lc_edge)

    # # 依 model LOS 速度上色的主線
    # norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    # lc = LineCollection(
    #     segments,
    #     cmap="coolwarm",
    #     norm=norm,
    #     linewidth=2.5,
    #     zorder=3,
    # )
    # lc.set_array(v_m + Local_Standard_Velocity)
    # ax.add_collection(lc)

    # num_element = 8
    # xarray_arc, z_array_arc = x_array[num_element] * dx_arcsec, z_array[num_element] * dx_arcsec
    # weights_im = ax.scatter( xarray_arc, z_array_arc, c=weights_array[num_element], s=8, cmap='YlGn_r')
    # x_means_arc, z_means_arc = x_means * dx_arcsec, z_means * dx_arcsec
    # ax.plot(x_means_arc, z_means_arc, color='k', lw=3, zorder=4)
    # divider = make_axes_locatable(ax)
    # cax     = divider.append_axes('right', size='3%', pad=0.04)
    # cbar = fig.colorbar(weights_im, cax=cax)
    # cbar.set_label('weight value')
    

    # --- 疊加質心點（如果有提供，輸入單位：pixel） ---
    if cen_x_pix is not None and cen_z_pix is not None:
        cen_x_pix = np.asarray(cen_x_pix)
        cen_z_pix = np.asarray(cen_z_pix)

        # 轉成 RA/Dec offset
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
    ax.scatter(0, 0, c="C3", s=60, marker="+", zorder=6)

    ax.set_xlabel("RA Offset (arcsec)")
    ax.set_ylabel("Dec Offset (arcsec)")
    ax.set_title(label)
    ax.set_xlim(12.5, -12.5)
    ax.set_ylim(-12.5, 12.5)
    
    # 主圖保持方形比例：RA/DEC 1:1
    ax.set_aspect("equal", adjustable="box")
    # --- 比例尺與方向標示 ---
    # 定義比例尺位置（以 arcsec 為單位）
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    text_pos_x = x1 + 0.25 * (x0 - x1)
    text_pos_y = y0 + 0.15 * (y1 - y0)
    scale_length = 800  # AU

    # 將 3000 AU 轉成 arcsec
    scale_length_arcsec = scale_length / (distance_pc)  # 1" ≈ 1 AU / distance(pc)

    # 定義比例尺線段 (RA offset 軸)
    scale_range_x = [text_pos_x, text_pos_x - scale_length_arcsec]
    scale_range_y = [text_pos_y - 0.2, text_pos_y - 0.2]

    # 繪製比例尺與文字
    ax.plot(scale_range_x, scale_range_y, color='k', lw=3, zorder=10)
    ax.text(
        text_pos_x - scale_length_arcsec / 2,
        text_pos_y - 2.0,
        f"{int(scale_length)} AU",
        ha='center', va='bottom',
        fontsize=14, family='Times New Roman', color='k'
    )

    # # --- 加上方向箭頭 (NE arrow) ---
    # ax.quiver(
    #     0.4,  0.4 * np.tan(np.deg2rad(10)),
    #     1.4, -1.4 * np.tan(np.deg2rad(10)),
    #     color='grey', scale=12, zorder=10
    # )
    # beam（若有）
    try:
        bmaj = header.get("BMAJ", None)
        bmin = header.get("BMIN", None)
        bpa  = header.get("BPA", 0.0)
        if bmaj and bmin:
            bmaj_arcsec = bmaj * 3600.0
            bmin_arcsec = bmin * 3600.0
            beam_x = x0 + 0.15 * (x1 - x0)
            beam_y = y0 + 0.15 * (y1 - y0)
            beam = mpl.patches.Ellipse(
                (beam_x, beam_y),
                width=bmin_arcsec,
                height=bmaj_arcsec,
                angle=bpa,
                facecolor="none",
                edgecolor="k",
                lw=1.2,
                zorder=15,
            )
            ax.add_patch(beam)
            ax.text(
                beam_x, beam_y - 0.55 * bmaj_arcsec,
                f"{bmaj_arcsec:.2f}″ × {bmin_arcsec:.2f}″",
                color='k', fontsize=10, ha='center', va='top'
            )
    except Exception as e:
        print(f"[mom0] beam draw failed: {e}")

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, outname), dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_streamer_on_mom1(theta, phi, inc, T_Myr, omega,
                          header, pa_rad, dx_au, im_center,
                          mom1, label, outname,
                          v_range=1.0,
                          vmin=None, vmax=None,
                          radius_in_au=radius_in_au,
                          radius_out_au=radius_out_au,
                          scale='log', log_power=log_power,
                          cen_x_pix=None, cen_z_pix=None, cen_v_LS_km=None):

    # --- 基本尺寸 & 像素刻度 ---
    dx_arcsec = abs(header["CDELT1"]) * 3600.0
    dz_arcsec = abs(header["CDELT2"]) * 3600.0
    ny, nx = mom1.shape
    extent, dx_arcsec, dz_arcsec = _compute_extent(header, im_center, ny, nx)

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

    x_pix_rot = x_pix * np.cos(pa_rad) - z_pix * np.sin(pa_rad) + im_center[1]
    z_pix_rot = x_pix * np.sin(pa_rad) + z_pix * np.cos(pa_rad) + im_center[0]

    ra_off  = (x_pix_rot - im_center[1]) * dx_arcsec
    dec_off = (z_pix_rot - im_center[0]) * dz_arcsec

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
    cbar.set_label("Velocity (km/s)")

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

    # num_element = 8
    # xarray_arc, z_array_arc = x_array[num_element] * dx_arcsec, z_array[num_element] * dx_arcsec
    # weights_im = ax.scatter( xarray_arc, z_array_arc, c=weights_array[num_element], s=8, cmap='YlGn_r')
    # x_means_arc, z_means_arc = x_means * dx_arcsec, z_means * dx_arcsec
    # ax.plot(x_means_arc, z_means_arc, color='k', lw=3, zorder=4)
    # divider = make_axes_locatable(ax)
    # cax     = divider.append_axes('right', size='3%', pad=0.04)
    # cbar = fig.colorbar(weights_im, cax=cax)
    # cbar.set_label('weight value')
    

    # --- 疊加質心點（輸入 cen_x_pix, cen_z_pix） ---
    if cen_x_pix is not None and cen_z_pix is not None:
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
                s=20,
                marker="o",
                edgecolors="black",
                linewidths=0.7,
                zorder=5,
                label="Streamer Centroids",
            )
        else:
            ax.scatter(
                cen_ra,
                cen_dec,
                facecolors="none",
                edgecolors="black",
                s=20,
                marker="o",
                zorder=5,
                label="Streamer Centroids",
            )

    # 中心位置
    ax.scatter(0, 0, c="C3", s=60, marker="+", zorder=6)

    ax.set_xlabel("RA Offset (arcsec)")
    ax.set_ylabel("Dec Offset (arcsec)")
    ax.set_title(label)
    ax.set_xlim(12.5, -12.5)
    ax.set_ylim(-12.5, 12.5)
    
    # 主圖保持方形比例：RA/DEC 1:1
    ax.set_aspect("equal", adjustable="box")
    # --- 比例尺與方向標示 ---
    # 定義比例尺位置（以 arcsec 為單位）
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    text_pos_x = x1 + 0.25 * (x0 - x1)
    text_pos_y = y0 + 0.15 * (y1 - y0)
    scale_length = 800  # AU

    # 將 3000 AU 轉成 arcsec
    scale_length_arcsec = scale_length / (distance_pc)  # 1" ≈ 1 AU / distance(pc)

    # 定義比例尺線段 (RA offset 軸)
    scale_range_x = [text_pos_x, text_pos_x - scale_length_arcsec]
    scale_range_y = [text_pos_y - 0.2, text_pos_y - 0.2]

    # 繪製比例尺與文字
    ax.plot(scale_range_x, scale_range_y, color='k', lw=3, zorder=10)
    ax.text(
        text_pos_x - scale_length_arcsec / 2,
        text_pos_y - 2.0,
        f"{int(scale_length)} AU",
        ha='center', va='bottom',
        fontsize=14, family='Times New Roman', color='k'
    )

    # # --- 加上方向箭頭 (NE arrow) ---
    # ax.quiver(
    #     0.4,  0.4 * np.tan(np.deg2rad(10)),
    #     1.4, -1.4 * np.tan(np.deg2rad(10)),
    #     color='grey', scale=12, zorder=10
    # )
    # beam（若有）
    try:
        bmaj = header.get("BMAJ", None)
        bmin = header.get("BMIN", None)
        bpa  = header.get("BPA", 0.0)
        if bmaj and bmin:
            bmaj_arcsec = bmaj * 3600.0
            bmin_arcsec = bmin * 3600.0
            beam_x = x0 + 0.15 * (x1 - x0)
            beam_y = y0 + 0.15 * (y1 - y0)
            beam = mpl.patches.Ellipse(
                (beam_x, beam_y),
                width=bmin_arcsec,
                height=bmaj_arcsec,
                angle=bpa,
                facecolor="none",
                edgecolor="k",
                lw=1.2,
                zorder=15,
            )
            ax.add_patch(beam)
            ax.text(
                beam_x, beam_y - 0.55 * bmaj_arcsec,
                f"{bmaj_arcsec:.2f}″ × {bmin_arcsec:.2f}″",
                color='k', fontsize=10, ha='center', va='top'
            )
    except Exception as e:
        print(f"[mom0] beam draw failed: {e}")

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, outname), dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_z_v_diagram_from_cube(theta_deg, phi_deg, inc_deg, T_Myr, omega,
                               new_cube_data, header, pa_rad, dx_au,
                               z_means_pix, streamer_v_LS_km,
                               outname,
                               label="S CrA z-v with data"):
    """
    由 masked data cube 建立 z–v 圖（**沿影像 x 軸堆疊**）：

      x: image-frame z (AU) ，以 protostar 為 0 （不再用 streamer 垂直方向）
      y: V_LSR (km/s)
      背景: new_cube_data 沿 x 軸 (axis=2) 平均後的強度
      藍線: PSS_model 轉到 **image frame** 後的 z–v
      黑點: streamer 質心（若提供 x,z 會轉到 image frame；否則直接使用傳入的 z）
    """

    if new_cube_data is None:
        print("[z–v] new_cube_data is None, skip.")
        return

    # ---------- 1) velocity 軸 ----------
    nz, ny, nx = new_cube_data.shape
    CRVAL3 = float(header["CRVAL3"])
    CRPIX3 = float(header["CRPIX3"])
    CDELT3 = float(header["CDELT3"])
    v_axis = CRVAL3 + (np.arange(nz) + 1 - CRPIX3) * CDELT3  # km/s, LSR
    vmin = np.nanmin(v_axis)
    vmax = np.nanmax(v_axis)

    # ---------- 2) image-frame z (AU) ----------
    # 注意：這裡不做 PA 旋轉，直接用影像的 y (row) 當作 z_img，原點在 protostar
    im_cy = float(header["CRPIX2"]) - 1.0  # FITS -> 0-based
    y_idx = np.arange(ny)
    z_img_pix = y_idx - im_cy
    z_img_AU = z_img_pix * dx_au

    # ---------- 3) 沿 x 軸平均，得到 pv(v,z_img) ----------
    # 直接對每個速度切片做 column-wise 平均
    pv = np.nanmean(new_cube_data, axis=2)  # shape = (nz, ny)
    if CDELT3 < 0:
        pv = pv[::-1]
        v_axis = v_axis[::-1]
    # ---------- 4) 繪圖 ----------
    fig, ax = plt.subplots(figsize=(7, 4))
    # 用百分位控制動態範圍，避免 outliers
    vmin_img = np.nanpercentile(pv, 5)
    vmax_img = np.nanpercentile(pv, 100)

    img = ax.imshow(
        pv,
        origin="lower",
        cmap="inferno",
        extent=[z_img_AU[0], z_img_AU[-1], vmin, vmax],
        aspect="auto",
        vmin=vmin_img,
        vmax=vmax_img,
    )
    # ---------- Add colorbar ----------
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cbar = fig.colorbar(img, cax=cax)
    cbar.set_label("Averaged Intensity", fontsize=10)

    # ---------- Add contour ----------
    try:
        levels = np.linspace(vmin_img, vmax_img, 5)
        ax.contour(
            z_img_AU,
            v_axis,
            pv,
            levels=levels,
            colors="white",
            linewidths=0.6,
            alpha=0.7
        )
    except Exception as e:
        print(f"[z–v] contour failed: {e}")
        
    # ---------- 5) 疊上 PSS_model（先轉成 image frame 的 z_img） ----------
    theta = np.deg2rad(theta_deg)
    phi   = np.deg2rad(phi_deg)
    inc   = np.deg2rad(inc_deg)

    x_m, y_m, z_m, u_m, v_m, w_m = pss.PSS_model(
        theta, phi, inc,
        T_Myr, omega,
        M_star,
        radius_in_au=radius_in_au,
        radius_out_au=radius_out_au,
        resolution=400,
        scale=scale,
        log_power=log_power,
    )
    # 將 model 的 (x_m, z_m) 旋轉到影像座標的 y 軸（image-frame z）
    # image-frame z = x*sin(PA) + z*cos(PA)
    z_model_img_AU = x_m * np.sin(pa_rad) + z_m * np.cos(pa_rad)

    ax.plot(
        z_model_img_AU,
        v_m + Local_Standard_Velocity,
        color="tab:blue",
        lw=2.0,
        label="Model (image-frame)",
        zorder=3,
    )

    # ---------- 6) 疊上質心 ----------
    if z_means_pix is not None and streamer_v_LS_km is not None:
        # z_means_pix 已經是相對 protostar 的影像座標像素位移，直接乘上 dx_au 變成 AU
        z_cent_img_AU = np.asarray(z_means_pix) * dx_au
        v_cent = np.asarray(streamer_v_LS_km) + Local_Standard_Velocity
        good = np.isfinite(z_cent_img_AU) & np.isfinite(v_cent)
        ax.scatter(
            z_cent_img_AU[good],
            v_cent[good],
            c="tab:blue",
            s=30,
            edgecolors="k",
            linewidths=0.6,
            label="Centroids",
            zorder=4,
        )

    # ---------- 7) 裝飾 ----------
    ax.set_xlabel("z (AU, image frame)")
    ax.set_ylabel("Velocity (km/s, LSR)")
    ax.set_title(label)
    ax.set_ylim(4, 8)
    ax.set_xlim(1200, -800)
    leg = ax.legend(frameon=False, fontsize=9)
    for txt in leg.get_texts():
        txt.set_color("black")
    ax.grid(alpha=0.2)

    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, outname), dpi=200)
    plt.close(fig)
    print(f"[z–v] Saved {outname}")
    
def run_quick_mode_scra():
    """
    Quick Mode for S CrA:
      - 從 cache (grid / mcmc_grid / mcmc_shell / final) 找 best-fit
      - 盡量載入 streamer cube、moment maps
      - 畫：
          1) z–v 圖
          2) moment-1 overlay
          3) moment-0 overlay
      - 印出物理參數
      - 結束程式
    """
    print("[Quick Mode] RUN_FROM_CACHE_ONLY=True → 僅讀取 cache 並繪圖")

    try:
        # 1) 讀取 cache
        cache_path_to_use = _resolve_cache_path(USE_CACHE_SOURCE)
        c = np.load(cache_path_to_use, allow_pickle=True)
        print(f"[cache] Loaded cache ({USE_CACHE_SOURCE}): {cache_path_to_use}")

        Theta_best, Phi_best, Incl_best, T_best, Omega_best = _extract_params_from_cache(
            c, USE_CACHE_SOURCE
        )
        Theta_best_deg = np.rad2deg(Theta_best)
        Phi_best_deg   = np.rad2deg(Phi_best)
        Incl_best_deg  = np.rad2deg(Incl_best)
        rms_channel = 0.026211100061251217

        # 2) 讀取 S CrA streamer 專用 moment map
        try:
            str_mom0 = fits.getdata("S_CrA_13CO_streamer_mom0.fits")
            str_mom1 = fits.getdata("S_CrA_13CO_streamer_mom1.fits")
            mom0 = fits.getdata("S_CrA_13CO_mom0.fits")
            mom1 = np.where(fits.getdata("S_CrA_13CO_mom0.fits") > 3 * rms_channel, fits.getdata("S_CrA_13CO_mom1.fits"), np.nan)
            header   = fits.getheader("S_CrA_13CO_streamer_mom1.fits")
        except Exception:
            cube = SpectralCube.read(cube_fname)
            header = fits.getheader(cube_fname)
            vrange = [2.4926, 14.8636] * u.km/u.s
            subcube = cube.spectral_slab(vrange[0], vrange[1])
            str_mom0 = subcube.moment(order=0).value
            str_mom1 = subcube.moment(order=1).value

        # 基本轉換
        im_center = (int(header["CRPIX2"]), int(header["CRPIX1"]))
        dx_arcsec = abs(header["CDELT2"]) * 3600.0
        dx_au     = dx_arcsec * distance_pc  # 160 pc for SCrA
        dv        = abs(float(header["CDELT3"]))

        # 3) 盡量讀取 streamer cube（若沒有，z–v 圖會自動跳過）
        new_cube_data = None
        try:
            new_cube_data = fits.getdata("S_CrA_13CO_streamer_cube.fits")
            print("[cache] Loaded streamer cube from FITS")
        except Exception as e:
            print(f"[cache] No streamer FITS ({e}), z–v diagram may be skipped.")

        # 4) 從 cache 抓 streamer 質心（若有）
        cen_x_pix = cen_z_pix = cen_v_LS = None

        if ("streamercom_x_AU" in c) and ("streamercom_z_AU" in c):
            sx = c["streamercom_x_AU"]
            sz = c["streamercom_z_AU"]

            x_rot = sx / dx_au
            z_rot = sz / dx_au
            cen_x_pix = x_rot*np.cos(pa_rad) - z_rot*np.sin(pa_rad) + im_center[1]
            cen_z_pix = x_rot*np.sin(pa_rad) + z_rot*np.cos(pa_rad) + im_center[0]

            streamercom_x_AU = sx
            streamercom_z_AU = sz
            streamercom_v_LS = c.get("streamercom_v_LS_km", None)

            if streamercom_v_LS is not None:
                cen_v_LS = streamercom_v_LS

                x_array = c["x_array"]
                z_array = c["z_array"]
                v_array = c["v_array"]
                weights_array = c["weights_array"]
                x_means = c["x_means"]
                z_means = c["z_means"]
                v_means = c["v_means"]

        # 5) z–v 圖（若 new_cube_data 存在）
        plot_z_v_diagram_from_cube(
            theta_deg=Theta_best_deg,
            phi_deg=Phi_best_deg,
            inc_deg=Incl_best_deg,
            T_Myr=T_best,
            omega=Omega_best,
            new_cube_data=new_cube_data,
            header=header,
            pa_rad=pa_rad,
            dx_au=dx_au,
            z_means_pix=z_means,
            streamer_v_LS_km=cen_v_LS,
            outname="SCrA_z_v_data_overlay.png",
            label="S CrA $^{13}$CO z–v"
        )

        plot_streamer_on_mom1(
            Theta_best, Phi_best, Incl_best,
            float(T_best), float(Omega_best),
            header, pa_rad, dx_au, im_center,
            str_mom1,
            label="S CrA $^{13}$CO",
            outname="SCrA_mom1_cacheonly.png",
            cen_x_pix=cen_x_pix,
            cen_z_pix=cen_z_pix,
            cen_v_LS_km=cen_v_LS,
            # vmax=np.nanmax(mom1),
            # vmin=np.nanmin(mom1)
        )

        plot_streamer_on_mom0(
            Theta_best, Phi_best, Incl_best,
            float(T_best), float(Omega_best),
            header, pa_rad, dx_au, im_center,
            str_mom0,
            label="S CrA $^{13}$CO",
            outname="SCrA_mom0_cacheonly.png",
            cen_x_pix=cen_x_pix,
            cen_z_pix=cen_z_pix,
            cen_v_LS_km=cen_v_LS,
        )
        
        plot_r_theta_weights_from_output(x_array, z_array, weights_array, outname="SCrA_weights_cacheonly.png")

        # 8) Print physical values
        r_ref_AU = 280 * T_best * 1e6 * spc.year / spc.astronomical_unit  # radius_ref_au=280

        M_0 = M_star * M_SUN_KG * spc.G / (280.0**3 * T_best * 1e6 * spc.year)
        M_dot = M_star / (T_best * 1e6)

        print("\n==================== Parameters (S CrA) ====================")
        print(f"Theta        = {Theta_best_deg:.3f} deg")
        print(f"Phi          = {Phi_best_deg:.3f} deg")
        print(f"Inclination  = {Incl_best_deg:.3f} deg")
        print(f"Time (T_Myr) = {T_best:.6f} Myr")
        print(f"Omega        = {Omega_best:.4f}")
        print(f"r_ref        = {r_ref_AU:.2f} AU")
        print(f"M_0          = {M_0:.3e}")
        print(f"Mdot         = {M_dot:.3e} M_sun/yr")
        print("============================================================")
        print("[Quick Mode] 繪圖完成，程式結束。")
        sys.exit(0)

    except Exception as e:
        print(f"[Quick Mode] 失敗，改跑完整流程: {e}")
        return
# ============================================================
# 3. 資料準備：讀 cube + 建 mask + 抽質心
# ============================================================
def prepare_data():
    global cube, header, im_center, dx_arcsec, dx_au, dv, v0
    global v_lastch_vel, v_lastch_num, subcube, moment0, moment1, cube_shape, spec_kms
    global rms_channel, shifted_cube_data, str_mom0, str_mom1
    global streamercom_x_AU, streamercom_z_AU, streamercom_v_LS_km
    global x_array, z_array, v_array, weights_array, x_means, z_means, v_means
    global v_weight_pix, v_weight_phys, max_dist_value
    # 讀 cube & header
    cube = SpectralCube.read(cube_fname)
    header = fits.getheader(cube_fname)

    im_center = (int(header["CRPIX2"]), int(header["CRPIX1"]))
    dx_arcsec = abs(header["CDELT2"]) * 3600.0
    # 這裡沿用原本 S CrA 寫法：用 CDELT1, CDELT2 的平均換算 AU
    dx_au = (abs(header["CDELT1"]) + abs(header["CDELT2"])) / 2 * 3600.0 * distance_pc
    dv = abs(float(header["CDELT3"]))
    v0 = header["CRVAL3"]

    # 速度軸 / 子立方體設定（依原本 S CrA 設定）
    v_lastch_vel = 14.8636
    v_lastch_num = 150
    velocity_range = [2.4926, 14.8636] * u.km / u.s

    subcube = cube.spectral_slab(velocity_range[0], velocity_range[1])
    moment0 = subcube.moment(order=0).value
    moment1 = subcube.moment(order=1).value
    cube_shape = subcube.shape
    spec_kms = subcube.spectral_axis.to(u.km / u.s).value  # 每個 channel 的實際速度

    # 儲存原始 moment 圖
    fits.PrimaryHDU(data=moment0, header=header).writeto(
        os.path.join("S_CrA_13CO_mom0.fits"), overwrite=True
    )
    print("[mask] Saved S_CrA_13CO_mom0.fits")
    hdu_mom1 = fits.PrimaryHDU(data=moment1, header=header)
    hdu_mom1.header["BUNIT"] = "km/s"
    hdu_mom1.writeto(
        os.path.join("S_CrA_13CO_mom1.fits"), overwrite=True
    )
    print("[mask] Saved S_CrA_13CO_mom1.fits")
    # 噪音（沿用你原本的數值）
    rms_channel = 0.026211100061251217
    
    # 建立 streamer 專用 cube（遮罩 + 平移）
    shifted_cube_data, str_mom0, str_mom1 = build_streamer_masked_cube_scra(
        subcube, rms_channel, im_center
    )

    # 儲存 streamer cube（給 Quick Mode 用）
    fits.PrimaryHDU(data=shifted_cube_data, header=header).writeto(
        os.path.join("S_CrA_13CO_streamer_cube.fits"), overwrite=True
    )
    print("[mask] Saved S_CrA_13CO_streamer_cube.fits")

    # 儲存 streamer moment 圖
    fits.PrimaryHDU(data=str_mom0, header=header).writeto(
        os.path.join("S_CrA_13CO_streamer_mom0.fits"), overwrite=True
    )
    print("[mask] Saved S_CrA_13CO_streamer_mom0.fits")

    fits.PrimaryHDU(data=str_mom1, header=header).writeto(
        os.path.join("S_CrA_13CO_streamer_mom1.fits"), overwrite=True
    )
    print("[mask] Saved S_CrA_13CO_streamer_mom1.fits")

     # 從 streamer cube 抽質心
    try:
        (streamercom_x_AU,
         streamercom_z_AU,
         streamercom_v_LS_km,
         x_array,
         z_array,
         v_array,
         weights_array,
         x_means,
         z_means,
         v_means) = extract_streamer_centroids(
            shifted_cube_data,
            header,
            pa_rad,
            dx_au,
        )
    except Exception as e:
        print("抽取質心失敗，請檢查 build_streamer_masked_cube_scra 或資料品質：", e)
        raise
    # 權重（以物理空間為主）
    v_weight_pix = (dv / dx_arcsec) ** 2
    v_weight_phys = (dx_au / dv)**2
    max_dist_value = 100.0
    # 把關鍵結果也塞進 cache，方便後續或 quick mode 使用
    cache.update({
        "streamercom_x_AU": streamercom_x_AU,
        "streamercom_z_AU": streamercom_z_AU,
        "streamercom_v_LS_km": streamercom_v_LS_km,
        # 這四個是「list of arrays」，強制轉成 object array
        "x_array": np.array(x_array, dtype=object),
        "z_array": np.array(z_array, dtype=object),
        "v_array": np.array(v_array, dtype=object),
        "weights_array": np.array(weights_array, dtype=object),
        # 下面這些就是一般 1D array，沒問題
        "x_means": x_means,
        "z_means": z_means,
        "v_means": v_means,
        "v_weight_pix": float(v_weight_pix),
        "v_weight_phys": float(v_weight_phys),
        "max_dist_value": float(max_dist_value),
    })

# --- Grid fitting logic moved into function ---
def run_grid():
    if RUN_GRID:
        best_params, grid, error = run_grid_search(
            streamercom_x_AU, streamercom_z_AU, streamercom_v_LS_km,
            v_weight_phys, M_star, scale, log_power, radius_ref_au,
            n_grid=10,
            T_factor_range=(65.0, 150.0),
            verbose=True,
        )
        # 利用 coarse grid 的低誤差區自動產生先驗範圍（與 Per-emb-50 一致）
        global parameter_prior_ranges, Theta_init, Phi_init, Incl_init, T_init, Omega_init
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

        print("\n[MCMC priors from grid]")
        for name, (lo, hi) in parameter_prior_ranges.items():
            if name in ["Theta zero", "Phi zero", "Inclination"]:
                print(f"{name:<12s}: ({np.rad2deg(lo):6.2f}, {np.rad2deg(hi):6.2f}) deg")
            elif name == "Time":
                print(f"{name:<12s}: ({lo:.5f}, {hi:.5f}) Myr")
            else:
                print(f"{name:<12s}: ({lo:.3f}, {hi:.3f})")
                
        cache.update({
            "grid_best_Theta": float(Theta_init),
            "grid_best_Phi":   float(Phi_init),
            "grid_best_Incl":  float(Incl_init),
            "grid_best_T":     float(T_init),
            "grid_best_Omega": float(Omega_init),
            "grid_best_error": float(best_params["best_val"]),
        })
        for key, (lo, hi) in parameter_prior_ranges.items():
            cache[f"prior_{key}_lo"] = float(lo)
            cache[f"prior_{key}_hi"] = float(hi)
        np.savez(CACHE_PATH_GRID, **cache)
        print(f"[cache] Saved grid search results to {CACHE_PATH_GRID}")
    else:
        print("[Grid fitting] Skipped (manual init used).")


# Helper function: sample initial walkers from priors
def _sample_initial_walkers_from_priors(prior_ranges, n_walkers, seed=42):
    """
    從先驗範圍裡均勻抽樣 initial walkers。
    """
    rng = np.random.default_rng(seed)
    p0 = np.zeros((n_walkers, 5), dtype=float)

    th_lo, th_hi = prior_ranges["Theta zero"]
    ph_lo, ph_hi = prior_ranges["Phi zero"]
    inc_lo, inc_hi = prior_ranges["Inclination"]
    t_lo, t_hi = prior_ranges["Time"]
    om_lo, om_hi = prior_ranges["Omega"]

    p0[:, 0] = rng.uniform(th_lo, th_hi, n_walkers)
    p0[:, 1] = rng.uniform(ph_lo, ph_hi, n_walkers)
    p0[:, 2] = rng.uniform(inc_lo, inc_hi, n_walkers)
    p0[:, 3] = rng.uniform(t_lo, t_hi, n_walkers)
    p0[:, 4] = rng.uniform(om_lo, om_hi, n_walkers)

    return p0

# ============================================================
# 5. MCMC_grid：fast likelihood（選配）
# ============================================================
def run_mcmc_grid():
    if not RUN_MCMC_GRID:
        print("[MCMC_grid] Skipped (RUN_MCMC_GRID = False)")
        return
    print("\n[MCMC_grid] start (15 質心 fast likelihood)")

    cache.get("grid_used", False)
    # --- Use MCMC_grid medians as center ---
    Theta_center = cache["grid_best_Theta"]
    Phi_center   = cache["grid_best_Phi"]
    Incl_center  = cache["grid_best_Incl"]
    T_center     = cache["grid_best_T"]
    Omega_center = cache["grid_best_Omega"]

    center_vals = [Theta_center, Phi_center, Incl_center, T_center, Omega_center]
    
    ndim = 5
    nwalkers = 20
    nsteps = 20000
    labels_5d = ["Theta zero", "Phi zero", "Inclination", "Time", "Omega"]
    p0 = np.zeros((nwalkers, ndim))

    sigma_vals  = [
        np.deg2rad(5.0),
        np.deg2rad(18.0),
        np.deg2rad(9.0),
        0.05 * (parameter_prior_ranges["Time"][1] - parameter_prior_ranges["Time"][0]),
        0.05 * (parameter_prior_ranges["Omega"][1] - parameter_prior_ranges["Omega"][0]),
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
        tau = sampler.get_autocorr_time(quiet=True)
        if (not np.all(np.isfinite(tau))) or (np.any(tau <= 0)):
            raise RuntimeError(f"tau invalid: {tau}")
        burnin = int(2 * np.nanmax(tau))
        thin   = max(1, int(0.1 * np.nanmin(tau)))
        print(f"[MCMC_grid] tau: {tau}, burnin={burnin}, thin={thin}")
    except Exception as e:
        print("[MCMC_grid] tau failed, use default.", e)
        burnin, thin = 1000, 50

    flat = sampler.get_chain(discard=burnin, thin=thin, flat=True)

    # --- 對 Phi 做 unwrap，避免 0/2π 斷裂 ---
    phi_ref = Phi_center
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


    samples_plot = flat_wrapped.copy()
    for idx in [0, 1, 2]:
        samples_plot[:, idx] = np.rad2deg(samples_plot[:, idx])
    labels_plot = [
    r"$\Theta_0$ (deg)",
    r"$\Phi_0$ (deg)",
    r"$i$ (deg)",
    r"$T$ (Myr)",
    r"$\omega$",
    ]
    q16p, q50p, q84p = np.percentile(samples_plot, [16, 50, 84], axis=0)
    ranges = []
    for i in range(len(labels_plot)):
        lo, md, hi = q16p[i], q50p[i], q84p[i]
        width = hi - lo if hi > lo else 1e-3
        ranges.append((md - 2*width, md + 2*width))

    fig = corner.corner(samples_plot,
                        labels=labels_plot,
                        range=ranges,
                        show_titles=True,
                        title_fmt=".3f",
                        quantiles=[0.16, 0.5, 0.84],
                        truths=[np.rad2deg(Theta_med), np.rad2deg(Phi_med), np.rad2deg(Incl_med), T_med, Omega_med],
                        smooth=1)

    fig.savefig(os.path.join(PLOT_DIR, "corner_mcmc_grid.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    cache.update({
        "mcmc_grid_used": True,
        "mcmc_grid_median_Theta": float(Theta_med),
        "mcmc_grid_median_Phi":   float(Phi_med),
        "mcmc_grid_median_Incl":  float(Incl_med),
        "mcmc_grid_median_T":     float(T_med),
        "mcmc_grid_median_Omega": float(Omega_med),
    })
    np.savez(CACHE_PATH_MCMC_GRID, **cache)
    print(f"[cache] Saved MCMC grid results to {CACHE_PATH_MCMC_GRID}")
    # ---- 寫入 FINAL cache ----
    cache.update({
        "best_Theta": float(Theta_med),
        "best_Phi":   float(Phi_med),
        "best_Incl":  float(Incl_med),
        "best_T":     float(T_med),
        "best_Omega": float(Omega_med),
        "best_source": "mcmc_grid",   # optional
    })

    np.savez(CACHE_PATH_FINAL, **cache)
    print(f"[cache] Saved FINAL best-fit to {CACHE_PATH_FINAL}")

# ============================================================
# 5. 殼層版 MCMC：使用 log_posterior_shell（最簡潔版本）
# ============================================================
def run_mcmc_shell():
    if not RUN_MCMC_SHELL:
        print("[MCMC_shell] Skipped (RUN_MCMC_SHELL = False)")
        return

    print("\n[MCMC_shell] start (distance-shell likelihood)")

    labels_5d = ["Theta zero", "Phi zero", "Inclination", "Time", "Omega"]

    cache.get("grid_used", False)
    Theta_center = cache["grid_best_Theta"]
    Phi_center   = cache["grid_best_Phi"]
    Incl_center  = cache["grid_best_Incl"]
    T_center     = cache["grid_best_T"]
    Omega_center = cache["grid_best_Omega"]
    center_vals = [Theta_center, Phi_center, Incl_center, T_center, Omega_center]

    ndim = 5
    nwalkers, nsteps = 20, 8000

    # 跟 Per-emb-50 一樣的 sigma 設定（單位：rad, Myr, dimensionless）
    sigmas = [
        np.deg2rad(5.0),   # Theta
        np.deg2rad(18.0),  # Phi
        np.deg2rad(9.0),   # Incl
        0.05 * (parameter_prior_ranges["Time"][1] - parameter_prior_ranges["Time"][0]),
        0.05 * (parameter_prior_ranges["Omega"][1] - parameter_prior_ranges["Omega"][0]),
    ]

    p0 = np.zeros((nwalkers, ndim))
    for j, key in enumerate(labels_5d):
        lo, hi = parameter_prior_ranges[key]
        prop = center_vals[j] + sigmas[j] * np.random.randn(nwalkers)
        # priors 是有限區間 → 用 clip 壓回範圍內
        prop = np.clip(prop, lo, hi)
        p0[:, j] = prop
    
    down_factor = 5
    down_factor_v = 3
    shifted_cube_data_ds = shifted_cube_data[::down_factor_v, ::down_factor, ::down_factor]
    print(f"[mask] shifted_cube_data downsampled: {shifted_cube_data.shape} → {shifted_cube_data_ds.shape}")
    
    max_dist_value = 50.0
    # 在 run_mcmc_shell() 外面或一開始
    DATA_BBOX = pss.compute_data_bbox(
            shifted_cube_data_ds,
            max_r=max_dist_value,
            extra_margin=5,
    )
    print("[bbox] DATA_BBOX =", DATA_BBOX)
    
    log_args = (
        shifted_cube_data_ds,
        parameter_prior_ranges,
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
        DATA_BBOX
    )

    moves = get_mcmc_moves(mode="refine")

    print(f"[MCMC_shell] nwalkers = {nwalkers}, nsteps = {nsteps}")
    with Pool(processes=8) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            pss.log_posterior_shell,
            args=log_args,
            pool=pool,
            moves=moves,
        )
        sampler.run_mcmc(p0, nsteps, progress=True)

    af = sampler.acceptance_fraction
    print(f"[MCMC_shell] acceptance fraction: mean={np.mean(af):.3f}, "
          f"min={np.min(af):.3f}, max={np.max(af):.3f}")

    try:
        tau = sampler.get_autocorr_time(quiet=True)
        if (not np.all(np.isfinite(tau))) or (np.any(tau <= 0)):
            raise RuntimeError(f"tau invalid: {tau}")
        burnin = int(2 * np.nanmax(tau))
        thin   = max(1, int(0.1 * np.nanmin(tau)))
        print(f"[MCMC_shell] tau: {tau}, burnin={burnin}, thin={thin}")
    except Exception as e:
        print("[MCMC_shell] tau 估計失敗，用預設。", e)
        burnin, thin = 30, 5

    chain = sampler.get_chain()
    print("chain shape:", chain.shape)  # (nsteps, nwalkers, ndim)
    print("mean acceptance:", np.mean(sampler.acceptance_fraction))

    lp = sampler.get_log_prob(flat=True)
    print("non-finite log_prob fraction =", np.mean(~np.isfinite(lp)))

    flat = sampler.get_chain(discard=burnin, thin=thin, flat=True)

    # -----------------------------
    # 7) unwrap Phi，避免 0/2π 斷裂
    # -----------------------------
    phi_samples = flat[:, 1]
    phi_wrapped = ((phi_samples - Phi_center + np.pi) % (2*np.pi)) - np.pi + Phi_center
    flat_wrapped = flat.copy()
    flat_wrapped[:, 1] = phi_wrapped

    # -----------------------------
    # 8) 統計量：median ±68% & 形狀判斷
    # -----------------------------
    q16, q50, q84 = np.percentile(flat_wrapped, [16, 50, 84], axis=0)
    Theta_med, Phi_med, Incl_med, T_med, Omega_med = q50

    print("\n[MCMC_shell] median ±68%:")
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

    print("\n[MCMC_shell] 1D posterior 形狀判斷：")
    for i, name in enumerate(labels_5d):
        summarize_1d_posterior(flat_wrapped[:, i], name)

    # -----------------------------
    # 9) corner plot（角度轉度）
    # -----------------------------
    samples_plot = flat_wrapped.copy()
    for idx in [0, 1, 2]:
        samples_plot[:, idx] = np.rad2deg(samples_plot[:, idx])
    labels_plot = [
    r"$\Theta_0$ (deg)",
    r"$\Phi_0$ (deg)",
    r"$i$ (deg)",
    r"$T$ (Myr)",
    r"$\omega$",
    ]
    q16p, q50p, q84p = np.percentile(samples_plot, [16, 50, 84], axis=0)
    ranges = []
    for i in range(len(labels_plot)):
        lo, md, hi = q16p[i], q50p[i], q84p[i]
        width = hi - lo if hi > lo else 1e-3
        ranges.append((md - 2*width, md + 2*width))

    fig = corner.corner(samples_plot,
                        labels=labels_plot,
                        range=ranges,
                        show_titles=True,
                        title_fmt=".3f",
                        quantiles=[0.16, 0.5, 0.84],
                        truths=[np.rad2deg(Theta_med), np.rad2deg(Phi_med), np.rad2deg(Incl_med), T_med, Omega_med],
                        smooth=1)
    fig.savefig(os.path.join(PLOT_DIR, "corner_mcmc_shell.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)

    # -----------------------------
    # 10) 寫入 cache（MCMC_shell + FINAL）
    # -----------------------------
    cache.update({
        "mcmc_shell_used": True,
        "mcmc_shell_median_Theta": float(Theta_med),
        "mcmc_shell_median_Phi":   float(Phi_med),
        "mcmc_shell_median_Incl":  float(Incl_med),
        "mcmc_shell_median_T":     float(T_med),
        "mcmc_shell_median_Omega": float(Omega_med),
        "mcmc_shell_acceptance_mean": float(np.mean(af)),
        "mcmc_shell_acceptance_min":  float(np.min(af)),
        "mcmc_shell_acceptance_max":  float(np.max(af)),
        "mcmc_shell_burnin": int(burnin),
        "mcmc_shell_thin":   int(thin),
        "mcmc_shell_nwalkers": int(nwalkers),
        "mcmc_shell_nsteps":   int(nsteps),
    })
    try:
        cache["mcmc_shell_tau"] = np.asarray(tau)
    except NameError:
        pass

    np.savez(CACHE_PATH_MCMC_SHELL, **cache)
    print(f"[cache] Saved MCMC shell results to {CACHE_PATH_MCMC_SHELL}")

    # ---- 更新 FINAL best-fit ----
    cache.update({
        "best_Theta": float(Theta_med),
        "best_Phi":   float(Phi_med),
        "best_Incl":  float(Incl_med),
        "best_T":     float(T_med),
        "best_Omega": float(Omega_med),
        "best_source": "mcmc_shell",
    })
    np.savez(CACHE_PATH_FINAL, **cache)
    print(f"[cache] Saved FINAL best-fit to {CACHE_PATH_FINAL}")


# ============================================================
# 6b. Overlay best-fit parameters on streamer moment map (from cache)
# ============================================================
def run_final_best_fit_and_overlay():
    """
    從 cache 抓出最終 best-fit 參數，畫在 S CrA streamer 的
      - z–v 圖
      - moment-1 overlay
      - moment-0 overlay

    目前預設從 USE_CACHE_SOURCE 對應的 cache 檔案讀取參數：
      - "mcmc_shell" → CACHE_PATH_MCMC_SHELL
      - "mcmc_grid"  → CACHE_PATH_MCMC_GRID
      - "grid"       → CACHE_PATH_GRID
      - "final"      → CACHE_PATH_FINAL
    """
    try:
        cache_path_to_use = _resolve_cache_path(USE_CACHE_SOURCE)
        c = np.load(cache_path_to_use, allow_pickle=True)
        print(f"[cache] Loaded for overlay ({USE_CACHE_SOURCE}): {cache_path_to_use}")

        # 盡量從 cache 中抽出一組 (Theta, Phi, Incl, T, Omega)
        Theta_best, Phi_best, Incl_best, T_best, Omega_best = _extract_params_from_cache(
            c, USE_CACHE_SOURCE
        )
        Theta_best_deg = np.rad2deg(Theta_best)
        Phi_best_deg   = np.rad2deg(Phi_best)
        Incl_best_deg  = np.rad2deg(Incl_best)

        # streamer 專用 moment maps
        try:
            str_mom0 = fits.getdata("S_CrA_13CO_streamer_mom0.fits")
            str_mom1 = fits.getdata("S_CrA_13CO_streamer_mom1.fits")
            header   = fits.getheader("S_CrA_13CO_streamer_mom1.fits")
        except Exception as e:
            print(f"[overlay] Failed to load streamer moment maps from FITS: {e}")
            return

        im_center = (int(header["CRPIX2"]), int(header["CRPIX1"]))
        dx_arcsec = abs(header["CDELT2"]) * 3600.0
        dx_au     = dx_arcsec * distance_pc

        # 讀取 streamer cube（如果有的話就畫 z–v）
        new_cube_data = None
        try:
            new_cube_data = fits.getdata("S_CrA_13CO_streamer_cube.fits")
            print("[overlay] Loaded streamer cube for z–v diagram")
        except Exception as e:
            print(f"[overlay] No streamer cube FITS ({e}), z–v diagram will be skipped.")

        # 從 cache 抓質心（若有），轉成 pixel 座標
        cen_x_pix = cen_z_pix = cen_v_LS_km = None
        if ("streamercom_x_AU" in c) and ("streamercom_z_AU" in c):
            sx_AU = c["streamercom_x_AU"]
            sz_AU = c["streamercom_z_AU"]

            # model frame (AU) → 影像 pixel
            x_rot = sx_AU / dx_au
            z_rot = sz_AU / dx_au
            cen_x_pix = x_rot * np.cos(pa_rad) - z_rot * np.sin(pa_rad) + im_center[1]
            cen_z_pix = x_rot * np.sin(pa_rad) + z_rot * np.cos(pa_rad) + im_center[0]

            if "streamercom_v_LS_km" in c:
                cen_v_LS_km = c["streamercom_v_LS_km"]

        # 1) z–v 圖（若有 cube）
        if new_cube_data is not None:
            plot_z_v_diagram_from_cube(
                theta_deg=Theta_best_deg,
                phi_deg=Phi_best_deg,
                inc_deg=Incl_best_deg,
                T_Myr=float(T_best),
                omega=float(Omega_best),
                new_cube_data=new_cube_data,
                header=header,
                pa_rad=pa_rad,
                dx_au=dx_au,
                z_means_pix=None,          # 目前不疊 centroids；之後需要可以從 cache 裡抓
                streamer_v_LS_km=None,
                outname="SCrA_z_v_data_overlay_best.png",
                label="S CrA $^{13}$CO z–v (best-fit)",
            )

        # 2) moment-1 overlay（速度場）
        plot_streamer_on_mom1(
            Theta_best, Phi_best, Incl_best,
            float(T_best), float(Omega_best),
            header, pa_rad, dx_au, im_center,
            str_mom1,
            label="S CrA $^{13}$CO moment1 (best-fit)",
            outname="SCrA_13CO_model_vs_mom1_overlay_best.png",
            cen_x_pix=cen_x_pix,
            cen_z_pix=cen_z_pix,
            cen_v_LS_km=cen_v_LS_km,
        )

        # 3) moment-0 overlay（強度）
        plot_streamer_on_mom0(
            Theta_best, Phi_best, Incl_best,
            float(T_best), float(Omega_best),
            header, pa_rad, dx_au, im_center,
            str_mom0,
            label="S CrA $^{13}$CO moment0 (best-fit)",
            outname="SCrA_13CO_model_vs_mom0_overlay_best.png",
            cen_x_pix=cen_x_pix,
            cen_z_pix=cen_z_pix,
            cen_v_LS_km=cen_v_LS_km,
        )

        print("[overlay] Generated best-fit z–v, moment0, and moment1 overlay plots from cache.")

    except Exception as e:
        print(f"[overlay] Failed to generate overlay from cache: {e}")
        
# ------------------- MAIN FUNCTION -------------------
def main():
    if RUN_FROM_CACHE_ONLY:
        run_quick_mode_scra()
    else:
        prepare_data()
        run_grid()
        run_mcmc_grid()
        run_mcmc_shell()
        run_final_best_fit_and_overlay()
    print("\n全部段落執行完成。")

if __name__ == "__main__":
    main()

# ============================================================
# 5b. 基於 MCMC_grid 的多峰參數，自動做局部 refinement
# ============================================================

#     if RUN_MCMC_GRID_REFINE:
#         print("\n[MCMC_grid_refine] 檢查多峰參數並進行局部 MCMC ...")

#         # 1) 偵測哪些參數呈現多峰（用 unwrap 後的 flat_wrapped）
#         multi_params = {}
#         for i, name in enumerate(labels_5d):
#             peaks = find_posterior_peaks_1d(
#                 flat_wrapped[:, i],
#                 bins=80,
#                 prominence_frac=0.12
#             )
#             if len(peaks) > 1:
#                 multi_params[name] = peaks

#         if not multi_params:
#             print("[MCMC_grid_refine] 未偵測到明顯多峰，跳過局部 refinement。")
#         else:
#             print("[MCMC_grid_refine] 偵測到多峰參數：")
#             for pname, peaks in multi_params.items():
#                 if pname in ["Theta zero", "Phi zero", "Inclination"]:
#                     centers_deg = [np.rad2deg(c) for c, w in peaks]
#                     print(f"  {pname}: {len(peaks)} peaks at {centers_deg} deg")
#                 else:
#                     centers = [c for c, w in peaks]
#                     print(f"  {pname}: {len(peaks)} peaks at {centers}")

#             # 2) 對每個(參數, 峰)做一個局部 MCMC，先驗只縮那一維
#             for pname, peaks in multi_params.items():
#                 p_index = labels_5d.index(pname)

#                 for k, (center, half_width) in enumerate(peaks, start=1):
#                     print(f"\n[MCMC_grid_refine] {pname} 峰 {k}: center={center:.4g}, half_width={half_width:.4g}")

#                     # 建局部 priors：從原本 parameter_prior_ranges 複製
#                     local_priors = dict(parameter_prior_ranges)

#                     if pname == "Phi zero":
#                         # Phi 使用週期邏輯，範圍交給 in_phi_range 處理
#                         two_pi = 2.0 * np.pi
#                         lo = (center - half_width) % two_pi
#                         hi = (center + half_width) % two_pi
#                         local_priors["Phi zero"] = (lo, hi)
#                     elif pname in ["Theta zero", "Inclination"]:
#                         lo = max(center - half_width,
#                                  parameter_prior_ranges[pname][0])
#                         hi = min(center + half_width,
#                                  parameter_prior_ranges[pname][1])
#                         local_priors[pname] = (lo, hi)
#                     else:
#                         # Time / Omega
#                         lo = max(center - half_width,
#                                  parameter_prior_ranges[pname][0])
#                         hi = min(center + half_width,
#                                  parameter_prior_ranges[pname][1])
#                         local_priors[pname] = (lo, hi)

#                     # 3) 初始化局部 walkers：確保在局部 prior 內有夠多「線性獨立」的點
#                     ndim_ref = 5
#                     nwalkers_ref = 24
#                     nsteps_ref = 4000

#                     p0_ref = np.zeros((nwalkers_ref, ndim_ref))
#                     center_vals = [Theta_med, Phi_med, Incl_med, T_med, Omega_med]
#                     sigma_vals = [
#                         np.deg2rad(3.0),   # Theta
#                         np.deg2rad(5.0),   # Phi
#                         np.deg2rad(5.0),   # Incl
#                         0.03 * T_med,      # Time
#                         0.05 * Omega_med,  # Omega
#                     ]

#                     for iw in range(nwalkers_ref):
#                         trial = np.zeros(ndim_ref)
#                         for j, key in enumerate(labels_5d):
#                             lo_j, hi_j = local_priors[key]

#                             if j == p_index:
#                                 # 對正在 refinement 的參數，直接在局部 prior 內做均勻抽樣
#                                 val = lo_j + (hi_j - lo_j) * np.random.rand()
#                             else:
#                                 # 其他參數在全域中位數附近加高斯擾動，再限制在局部 prior 範圍
#                                 val = center_vals[j] + sigma_vals[j] * np.random.randn()
#                                 if key == "Phi zero":
#                                     # wrap 到 [0, 2π)
#                                     val = val % (2.0 * np.pi)
#                                 val = np.clip(val, lo_j, hi_j)

#                             trial[j] = val
#                         p0_ref[iw, :] = trial
                        
#                     # 確保 walkers 並非完全重合（避免 condition number 太大）
#                     # 若發現所有 walker 幾乎一樣，再加一點微小雜訊
#                     if np.linalg.matrix_rank(p0_ref) < ndim_ref:
#                         eps = 1e-4
#                         p0_ref += eps * np.random.randn(*p0_ref.shape)

#                     sampler_ref = emcee.EnsembleSampler(
#                         nwalkers_ref, ndim_ref,
#                         pss.log_posterior_fast,
#                         args=(
#                             local_priors,
#                             streamercom_x_AU,
#                             streamercom_z_AU,
#                             streamercom_v_LS_km,
#                             v_weight_phys,
#                             M_star,
#                             scale,
#                             log_power,
#                         ),
#                         moves=get_mcmc_moves(mode="refine"),
#                     )

#                     sampler_ref.run_mcmc(p0_ref, nsteps_ref, progress=True)

#                     # 4) 簡單 burn-in / thinning（這裡用穩定保守值就好）
#                     burn_ref = int(0.3 * nsteps_ref)
#                     thin_ref = 10
#                     flat_ref = sampler_ref.get_chain(
#                         discard=burn_ref,
#                         thin=thin_ref,
#                         flat=True
#                     )

#                     # Phi unwrap
#                     flat_ref_wrapped = flat_ref.copy()
#                     phi_ref0 = Phi_init
#                     phi_s = flat_ref[:, 1]
#                     phi_unwrap = ((phi_s - phi_ref0 + np.pi) % (2.0 * np.pi)) - np.pi + phi_ref0
#                     flat_ref_wrapped[:, 1] = phi_unwrap

#                     # 中位數解
#                     q16r, q50r, q84r = np.percentile(flat_ref_wrapped, [16, 50, 84], axis=0)
#                     th_r, ph_r, inc_r, T_r, omg_r = q50r

#                     print(f"[MCMC_grid_refine] {pname} 峰 {k} 中位數：")
#                     print(f"  Theta = {np.rad2deg(th_r):6.2f} deg")
#                     print(f"  Phi   = {np.rad2deg(ph_r):6.2f} deg")
#                     print(f"  Incl  = {np.rad2deg(inc_r):6.2f} deg")
#                     print(f"  T     = {T_r:.5f} Myr")
#                     print(f"  Omega = {omg_r:.4f}")

#                     # 5) Corner plot（角度轉度）
#                     samples_ref_plot = flat_ref_wrapped.copy()
#                     for idx in [0, 1, 2]:
#                         samples_ref_plot[:, idx] = np.rad2deg(samples_ref_plot[:, idx])

#                     labels_ref = ["Theta zero (°)", "Phi zero (°)", "Inclination (°)",
#                                   "Time (Myr)", "Omega"]

#                     # 動態檢查每個參數範圍（用16–84%+padding，與主corner一致）
#                     ranges_ref = []
#                     q16_ref, q50_ref, q84_ref = np.percentile(samples_ref_plot, [16, 50, 84], axis=0)
#                     for i, label in enumerate(labels_ref):
#                         lo, md, hi = q16_ref[i], q50_ref[i], q84_ref[i]
#                         width = hi - lo
#                         if width <= 0:
#                             # fallback: use min/max or small default
#                             data = samples_ref_plot[:, i]
#                             lo_span, hi_span = np.nanmin(data), np.nanmax(data)
#                             if np.isfinite(lo_span) and np.isfinite(hi_span) and (hi_span - lo_span) > 0:
#                                 width = hi_span - lo_span
#                                 md = 0.5 * (hi_span + lo_span)
#                             else:
#                                 width = 1e-3
#                         # Unified: all parameters, including Theta/Inclination, use centered range logic
#                         lo_range = md - 1.5 * width
#                         hi_range = md + 1.5 * width
#                         # Do NOT clamp for any parameter; all are centered (except Phi, which already has no clamp)
#                         ranges_ref.append((lo_range, hi_range))

#                     fig_ref = corner.corner(
#                         samples_ref_plot,
#                         labels=labels_ref,
#                         show_titles=True,
#                         title_fmt=".2f",
#                         plot_datapoints=False,
#                         smooth=1.0,
#                         fill_contours=True,
#                         range=ranges_ref,
#                     )

#                     corner_name = f"corner_mcmc_grid_refine_{pname.replace(' ', '_')}_peak{k}.png"
#                     fig_ref.savefig(
#                         os.path.join(PLOT_DIR, corner_name),
#                         dpi=200,
#                         bbox_inches="tight"
#                     )
#                     plt.close(fig_ref)

#                     # 6) 用此局部中位數模型做 centroid RMSE（方便之後挑最好的一組）
#                     rmse_ref = pss.error_function(
#                         [th_r, ph_r],
#                         streamercom_x_AU,
#                         streamercom_z_AU,
#                         streamercom_v_LS_km,
#                         v_weight_phys,
#                         T_r, omg_r, inc_r,
#                         M_star,
#                         scale,
#                         log_power,
#                     )
#                     print(f"[MCMC_grid_refine] {pname} 峰 {k} centroid RMSE ≈ {rmse_ref:.4f}")

# # ============================================================
# # 6. MCMC_3D：三參數（選配）
# # ============================================================

# if RUN_MCMC_3D:
#     print("\n[MCMC_3D] wide prior on (Theta, Phi, Incl)")

#     def log_prior_3d(params, prior_ranges_3d):
#         th, ph, inc = params
#         if not (
#             prior_ranges_3d["Theta zero"][0] <= th <= prior_ranges_3d["Theta zero"][1]
#             and prior_ranges_3d["Phi zero"][0] <= ph <= prior_ranges_3d["Phi zero"][1]
#             and prior_ranges_3d["Inclination"][0] <= inc <= prior_ranges_3d["Inclination"][1]
#         ):
#             return -np.inf
#         return 0.0

#     def log_post_3d(params, prior_ranges_3d,
#                     x_d, z_d, v_d, wv, Ms, T_fix, omg_fix, scale, log_power):
#         lp = log_prior_3d(params, prior_ranges_3d)
#         if not np.isfinite(lp):
#             return -np.inf
#         th, ph, inc = params
#         rmse = pss.error_function(
#             [th, ph],
#             x_d, z_d, v_d,
#             wv, T_fix, omg_fix, inc,
#             Ms, scale, log_power,
#         )
#         if rmse <= 0 or np.isnan(rmse):
#             return -np.inf
#         return lp - np.log10(rmse)

#     prior_3d = {
#         "Theta zero": (0.0, 0.5*np.pi),
#         "Phi zero":   (0.0, 2.0*np.pi),
#         "Inclination":(-0.5*np.pi, 0.5*np.pi),
#     }

#     ndim3, nwalkers3, nsteps3 = 3, 48, 8000
#     T_fix, omg_fix = T_init, Omega_init

#     init3 = np.array([Theta_init, Phi_init, Incl_init])
#     p0_3d = np.zeros((nwalkers3, ndim3))
#     jitter = np.deg2rad([20.0, 40.0, 30.0])

#     for i, key in enumerate(["Theta zero", "Phi zero", "Inclination"]):
#         lo, hi = prior_3d[key]
#         p = init3[i] + jitter[i] * np.random.randn(nwalkers3)
#         p0_3d[:, i] = np.clip(p, lo, hi)

#     moves3 = [
#         (emcee.moves.StretchMove(a=2.5), 0.2),
#         (emcee.moves.DEMove(),           0.8),
#     ]

#     sampler_3d = emcee.EnsembleSampler(
#         nwalkers3, ndim3,
#         log_post_3d,
#         args=(
#             prior_3d,
#             streamercom_x_AU,
#             streamercom_z_AU,
#             streamercom_v_LS_km,
#             v_weight_phys,
#             M_star,
#             T_fix, omg_fix,
#             scale, log_power,
#         ),
#         moves=moves3,
#     )
#     sampler_3d.run_mcmc(p0_3d, nsteps3, progress=True)

#     try:
#         tau3 = sampler_3d.get_autocorr_time(tol=0)
#         burn3 = int(2*np.max(tau3))
#         thin3 = max(1, int(0.5*np.min(tau3)))
#         print(f"[MCMC_3D] tau={tau3}, burn-in={burn3}, thin={thin3}")
#     except Exception as e:
#         print("[MCMC_3D] tau 估計失敗，用預設。", e)
#         burn3, thin3 = 1000, 20

#     flat3 = sampler_3d.get_chain(discard=burn3, thin=thin3, flat=True)
#     samples3 = flat3.copy()
#     samples3[:, :3] = np.rad2deg(samples3[:, :3])

#     fig3 = corner.corner(samples3,
#                          labels=["Theta zero (°)", "Phi zero (°)", "Inclination (°)"],
#                          show_titles=True, title_fmt=".2f")
#     fig3.savefig(os.path.join(PLOT_DIR, "corner_mcmc_3d.png"), dpi=180)
#     plt.close(fig3)

# # ============================================================
# # 7. MCMC_distance：distance_cube + log_posterior
# # ============================================================

# if RUN_MCMC_DISTANCE:
#     print("\n[MCMC_distance] 使用 distance_cube-based log_posterior")

#     data_cube = shifted_cube_data
#     cube_shape = data_cube.shape
#     v_weight_for_cube = v_weight_pix

#     # 用 grid 最佳解建 model 線 → search_bound
#     x_m0, y_m0, z_m0, u_m0, v_m0, w_m0 = pss.PSS_model(
#         Theta_init, Phi_init, Incl_init, T_init, Omega_init,
#         M_star,
#         radius_in_au=radius_in_au,
#         radius_out_au=radius_out_au,
#         resolution=80,
#         scale=scale,
#         log_power=log_power,
#     )
#     dx_arcsec = abs(header["CDELT1"]) * 3600.0
#     dz_arcsec = abs(header["CDELT2"]) * 3600.0
#     x_pix = x_m0 / dx_au
#     z_pix = z_m0 / dx_au
#     ra_off  = x_pix * dx_arcsec
#     dec_off = z_pix * dz_arcsec
#     x_best_pix_int = np.round(
#         x_pix * np.cos(pa_rad) - z_pix * np.sin(pa_rad) + im_center[1]
#     ).astype(int)
#     z_best_pix_int = np.round(
#         x_pix * np.sin(pa_rad) + z_pix * np.cos(pa_rad) + im_center[0]
#     ).astype(int)
#     v_best_pix_int = np.round(
#         v_lastch_num - (v_m0 - v_lastch_vel + Local_Standard_Velocity) / dv
#     ).astype(int)

#     search_bound = pss.get_bounding_box(
#         x_best_pix_int, z_best_pix_int, v_best_pix_int,
#         buffer=6, v_buffer=3, cube_shape=cube_shape
#     )
#     print("[MCMC_distance] search_bound:", search_bound)

#     ndim, nwalkers, nsteps = 5, 20, 1000
#     labels_5d = ["Theta zero", "Phi zero", "Inclination", "Time", "Omega"]

#     p0 = np.zeros((nwalkers, ndim))
#     print("[MCMC_distance] 初始化 walkers (緊貼 grid 解)...")

#     center_vals = [Theta_init, Phi_init, Incl_init, T_init, Omega_init]
#     sigma_vals  = [
#         np.deg2rad(5.0),    # Theta
#         np.deg2rad(8.0),    # Phi
#         np.deg2rad(8.0),    # Incl
#         0.05 * T_init,      # Time
#         0.10 * Omega_init,  # Omega
#     ]

#     for j, key in enumerate(labels_5d):
#         lo, hi = parameter_prior_ranges[key]
#         proposal = center_vals[j] + sigma_vals[j] * np.random.randn(nwalkers)
#         proposal = np.clip(proposal, lo, hi)
#         p0[:, j] = proposal

#     moves = get_mcmc_moves(mode="refine")

#     sampler = emcee.EnsembleSampler(
#         nwalkers, ndim,
#         pss.log_posterior,
#         args=(
#             data_cube,
#             search_bound,
#             parameter_prior_ranges,
#             pa_rad,
#             dx_au,
#             im_center,
#             dv,
#             v_lastch_vel,
#             v_lastch_num,
#             v0,
#             v_weight_for_cube,
#             max_dist_value,
#             M_star,
#             radius_in_au,
#             radius_out_au,
#         ),
#         moves=moves,
#     )
#     sampler.run_mcmc(p0, nsteps, progress=True)

#     # 自動 burn-in / thin
#     try:
#         tau = sampler.get_autocorr_time()  # 不用 tol=0 這麼硬
#         burnin = int(2 * np.nanmax(tau))
#         thin   = max(1, int(0.1 * np.nanmin(tau)))  # 原本 0.5 改成 0.1
#         print(f"... tau={tau}, burn-in={burnin}, thin={thin}")
#         n_eff_steps = (sampler.iteration - burnin) // thin
#         ess = (nwalkers * n_eff_steps) / (2.0 * tau)
#         print(f"... ESS ≈ {ess}")
#     except Exception as e:
#         print("[MCMC_distance] tau 估計失敗，用預設。", e)
#         burnin = int(0.2*nsteps)
#         thin   = 20
#         n_eff_steps = (sampler.iteration - burnin) // thin
#         ess = np.full(ndim, nwalkers * max(n_eff_steps, 1))
#         print(f"[MCMC_distance] 粗略 ESS ≈ {ess}")

#     flat = sampler.get_chain(discard=burnin, thin=thin, flat=True)
#     flat_lnprob = sampler.get_log_prob(discard=burnin, thin=thin, flat=True)
#     print("[MCMC_distance] flat_samples:", flat.shape)

#     # --- Phi 的角度展開修正 ---
#     angle_idx = [0, 1, 2]
#     phi_samples = flat[:, 1]
#     phi_ref = Phi_init
#     phi_wrapped = ((phi_samples - phi_ref + np.pi) % (2*np.pi)) - np.pi + phi_ref
#     flat_wrapped = flat.copy()
#     flat_wrapped[:, 1] = phi_wrapped

#     q16, q50, q84 = np.percentile(flat_wrapped, [16, 50, 84], axis=0)

#     print("\n[MCMC_distance] 參數中位數與 68% 區間：")
#     for i, name in enumerate(labels_5d):
#         lo, md, hi = q16[i], q50[i], q84[i]
#         if i in angle_idx:
#             lo, md, hi = np.rad2deg([lo, md, hi])
#             unit = "deg"
#         elif name == "Time":
#             unit = "Myr"
#         else:
#             unit = ""
#         print(f"{name:12s}: {md:.6f} (+{hi-md:.6f}/-{md-lo:.6f}) {unit}")

#     print("\n[MCMC_distance] 1D posterior 形狀判斷：")
#     for i, name in enumerate(labels_5d):
#         summarize_1d_posterior(flat_wrapped[:, i], name)

#     # Corner plot（角度轉度）
#     # --- Corner plot：unwrap 後再轉成度數 ---
#     # Corner plot：unwrap 後再轉度，並限制顯示範圍
#     samples_plot = flat_wrapped.copy()
#     for idx in angle_idx:
#         samples_plot[:, idx] = np.rad2deg(samples_plot[:, idx])

#     labels_plot = ["Theta zero (°)", "Phi zero (°)", "Inclination (°)",
#                    "Time (Myr)", "Omega"]
    
#     # Compute q16/q50/q84 for samples_plot (posterior-centered)
#     q16_plot, q50_plot, q84_plot = np.percentile(samples_plot, [16, 50, 84], axis=0)
    
#     ranges = []
#     for i, label in enumerate(labels_plot):
#         lo, md, hi = q16_plot[i], q50_plot[i], q84_plot[i]
#         width = hi - lo
#         if width <= 0:
#             # fallback: use min/max or small default
#             data = samples_plot[:, i]
#             lo_span, hi_span = np.nanmin(data), np.nanmax(data)
#             if np.isfinite(lo_span) and np.isfinite(hi_span) and (hi_span - lo_span) > 0:
#                 width = hi_span - lo_span
#                 md = 0.5 * (hi_span + lo_span)
#             else:
#                 width = 1e-3
#         # Unified: all parameters, including Theta/Inclination, use centered range logic
#         lo_range = md - 1.5 * width
#         hi_range = md + 1.5 * width
#         # Do NOT clamp for any parameter; all are centered (except Phi, which already has no clamp)
#         ranges.append((lo_range, hi_range))
        
#     fig = corner.corner(
#         samples_plot,
#         labels=labels_plot,
#         range=ranges,
#         show_titles=True,
#         title_fmt=".2f",
#         plot_datapoints=False,
#         fill_contours=True,
#         smooth=1.0,
#     )

#     fig.savefig(os.path.join(PLOT_DIR, "corner_mcmc_distance.png"), dpi=200, bbox_inches="tight")
#     plt.close(fig)

#     # --- 用中位數解畫 v-r 圖 ---
#     Theta_med, Phi_med, Incl_med, T_med, Omega_med = q50

#     x_m, y_m, z_m, u_m, v_m, w_m = pss.PSS_model(
#         Theta_med, Phi_med, Incl_med, T_med, Omega_med,
#         M_star,
#         radius_in_au=radius_in_au,
#         radius_out_au=radius_out_au,
#         resolution=200,
#         scale=scale,
#         log_power=log_power,
#     )

#     r_model = np.sqrt(x_m**2 + z_m**2)
#     r_data = np.sqrt(streamercom_x_AU**2 + streamercom_z_AU**2)

#     plt.figure(figsize=(6, 5))
#     plt.scatter(r_model, v_m, s=5, alpha=0.3, label="model")
#     plt.scatter(r_data, streamercom_v_LS_km,
#                 c="r", s=30, marker="x", label="centroids")
#     plt.xlabel("r (AU)")
#     plt.ylabel("v (km/s)")
#     plt.legend()
#     plt.title("Cube MCMC median model vs data")
#     plt.tight_layout()
#     plt.savefig(os.path.join(PLOT_DIR, "cube_mcmc_median_model_vs_data.png"), dpi=180)
#     plt.close()
    
#     plot_streamer_on_mom1(
#         Theta_med, Phi_med, Incl_med, T_med, Omega_med,
#         header, pa_rad, dx_au, im_center,
#         str_mom1,
#         label="MCMC_distance median streamline",
#         outname="mcmc_distance_median_mom1.png",
#         cen_x_AU=streamercom_x_AU,
#         cen_z_AU=streamercom_z_AU,
#         cen_v_LS_km=streamercom_v_LS_km
#         )
