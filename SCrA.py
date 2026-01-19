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
from scipy.ndimage import gaussian_filter

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
from numba import njit

# 後驗分析 / 工具
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

# 專案本地模組
import PSSpy as pss
from pss_grid_search import run_grid_search, compute_priors_from_grid

# --- 基本天文參數（S CrA） ---
Local_Standard_Velocity = 5.86  # km/s (Gupta 2024)
pa_default = 0
pa_env = os.getenv("PA_OVERRIDE_DEG")

if pa_env is not None:
    pa_deg = float(pa_env)
    print(f"[CONFIG] PA overridden by environment: {pa_deg} deg")
else:
    pa_deg = pa_default
    print(f"[CONFIG] PA using default value: {pa_deg} deg")

pa_rad = np.deg2rad(pa_deg)
distance_pc = 160.0
M_SUN_KG = 1.98847e30
radius_ref_au = 240
M_star = 2

scale = "log"
log_power = 1.5

radius_in_au, radius_out_au = 3e2, 1.55e3
# 資料與輸出
cube_fname = "S_CrA_13CO_spw25_tav_jupyter_shifted.fits"
CACHE_DIR = "SCrA_results/cache"
os.makedirs(CACHE_DIR, exist_ok=True)
PLOT_DIR = os.getenv("PLOT_DIR", "SCrA_results/plots")
os.makedirs(PLOT_DIR, exist_ok=True)

CACHE_PATH_GRID = os.path.join(CACHE_DIR, "SCrA_grid_results.npz")
CACHE_PATH_MCMC_GRID = os.path.join(CACHE_DIR, "SCrA_mcmc_grid_results.npz")
CACHE_PATH_MCMC_SHELL = os.path.join(CACHE_DIR, "SCrA_mcmc_shell_results.npz")
CACHE_PATH_FINAL = os.path.join(CACHE_DIR, "SCrA_fit_results_final.npz")

USE_CACHE_SOURCE = "grid"
sample_from = "Median"

# ---------- corner 重畫模式 ----------
REBUILD_CORNER_ONLY = False   # True: 不跑資料、不跑MCMC，只從 cache 重畫 corner
REBUILD_WHICH = ("mcmc_grid", "mcmc_shell")  # 想重畫哪些：可改成只留其中一個"mcmc_grid", "mcmc_shell"

# --- 分析開關 ---
RUN_GRID = True               # 5D grid search 找初始解
RUN_MCMC_GRID = False          # 32 個質心點 fast likelihood
RUN_MCMC_SHELL = False         # distance_cube MCMC
RUN_FROM_CACHE_ONLY = False   # True: 僅讀 cache 畫圖，完全不重跑

# RUN_GRID = False               # 5D grid search 找初始解
# RUN_MCMC_GRID = False          # 32 個質心點 fast likelihood
# RUN_MCMC_SHELL = False         # distance_cube MCMC
# RUN_FROM_CACHE_ONLY = True   # True: 僅讀 cache 畫圖，完全不重跑

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

    def _try(a, b, c1, d, e):
        try:
            return (
                float(c[a]),
                float(c[b]),
                float(c[c1]),
                float(c[d]),
                float(c[e]),
            )
        except Exception:
            return None

    # -------------------------
    # MEDIAN
    # -------------------------
    if sample_from == "Median":
        if s == "mcmc_shell":
            order = [
                ("mcmc_shell_median_Theta","mcmc_shell_median_Phi","mcmc_shell_median_Incl","mcmc_shell_median_T","mcmc_shell_median_Omega"),
                ("mcmc_grid_median_Theta","mcmc_grid_median_Phi","mcmc_grid_median_Incl","mcmc_grid_median_T","mcmc_grid_median_Omega"),
                ("grid_best_Theta","grid_best_Phi","grid_best_Incl","grid_best_T","grid_best_Omega"),
                ("best_Theta_median","best_Phi_median","best_Incl_median","best_T_median","best_Omega_median"),
            ]

        elif s == "mcmc_grid":
            order = [
                ("mcmc_grid_median_Theta","mcmc_grid_median_Phi","mcmc_grid_median_Incl","mcmc_grid_median_T","mcmc_grid_median_Omega"),
                ("grid_best_Theta","grid_best_Phi","grid_best_Incl","grid_best_T","grid_best_Omega"),
            ]

        elif s == "grid":
            order = [
                ("grid_best_Theta","grid_best_Phi","grid_best_Incl","grid_best_T","grid_best_Omega"),
            ]

    # -------------------------
    # PEAK (replacing MAP)
    # -------------------------
    elif sample_from == "Peak":
        if s == "mcmc_shell":
            order = [
                ("mcmc_shell_peak2d_Theta","mcmc_shell_peak2d_Phi","mcmc_shell_peak2d_Incl","mcmc_shell_peak2d_T","mcmc_shell_peak2d_Omega"),
                ("mcmc_grid_peak2d_Theta","mcmc_grid_peak2d_Phi","mcmc_grid_peak2d_Incl","mcmc_grid_peak2d_T","mcmc_grid_peak2d_Omega"),
                ("grid_best_Theta","grid_best_Phi","grid_best_Incl","grid_best_T","grid_best_Omega"),
                ("best_Theta_peak2d","best_Phi_peak2d","best_Incl_peak2d","best_T_peak2d","best_Omega_peak2d"),
            ]

        elif s == "mcmc_grid":
            order = [
                ("mcmc_grid_peak2d_Theta","mcmc_grid_peak2d_Phi","mcmc_grid_peak2d_Incl","mcmc_grid_peak2d_T","mcmc_grid_peak2d_Omega"),
                ("grid_best_Theta","grid_best_Phi","grid_best_Incl","grid_best_T","grid_best_Omega"),
            ]

        elif s == "grid":
            order = [
                ("grid_best_Theta","grid_best_Phi","grid_best_Incl","grid_best_T","grid_best_Omega"),
            ]

        else:
            raise ValueError(f"Unknown source='{source}' for Peak sampling")

    else:
        raise ValueError("sample_from must be 'Median' or 'Peak'")

    # -------------------------
    # Try in order
    # -------------------------
    for keys in order:
        out = _try(*keys)
        if out is not None:
            return out

    raise KeyError("Cache missing requested parameters.")


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

def build_streamer_masked_cube(cube, header, rms_channel):
    """
    回傳:
      im_center (y, x),
      masked_cube,      # 手動 mask + grow_region 後的 SpectralCube
      new_cube_data     # np.ndarray, masked_cube 填 0 之後的資料
    """
    im_center = (int(round(header["CRPIX2"] - 1.0)), int(round(header["CRPIX1"] - 1.0)))
    ny, nx = cube.shape[1], cube.shape[2]

    # 以兩個亮區中點當作初始中心 (這裡沿用你之前使用的座標)
    center1 = (391, 395)  # (y, x)
    center2 = (368, 383)  # (y, x)
    new_center = (int((center1[0] + center2[0]) / 2),
                  int((center1[1] + center2[1]) / 2))

    # 1) 大圓遮中心 + grow_region 長出 streamer
    radius_center = 35
    mask2d = pss.circular_mask((ny, nx), new_center, radius_center)
    mask3d = np.repeat(mask2d[np.newaxis, :, :], cube.shape[0], axis=0)
    masked_center_cube = cube.with_mask(mask3d)
    maskcent_cube_data = masked_center_cube.filled_data[:].value
    
    # grow_region 找 streamer
    init_points = [
        (35, new_center[0], new_center[1]),
        (35, 396, 396),
        (35, 355, 371),
        (35, 351, 358),
        (35, 355, 340),
        (35, 361, 326),
        (35, 369, 309),
        (35, 378, 295),
        (35, 389, 279),
        (35, 413, 268),
        (35, 463, 257),
        (35, 484, 255),
        (35, 499, 255),
    ]

    stream_mask = pss.grow_region(
        maskcent_cube_data,
        init_points,
        rms_channel,
        sigma_thresh=3.5,
        max_iter=1000,
    )

    masked_cube = masked_center_cube.with_mask(stream_mask)
    # # 2) 額外兩個圓形遮罩，清掉雜訊/多餘結構
    # ny, nx = masked_cube.shape[1], masked_cube.shape[2]

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
    new_cube_data = masked_cube.with_fill_value(0).filled_data[:].value

    # nv = new_cube_data.shape[0]
    # ty = im_center[0] - new_center[0]
    # tx = im_center[1] - new_center[1]
    # M = np.float32([[1, 0, tx], [0, 1, ty]])

    # shifted_cube_data = np.full_like(new_cube_data, np.nan)
    # for v_slice in range(nv):
    #     shifted_slice = cv2.warpAffine(
    #         new_cube_data[v_slice],
    #         M,
    #         (nx, ny),
    #         borderValue=np.nan,
    #     )
    #     shifted_cube_data[v_slice] = shifted_slice

    # return shifted_cube_data, shifted_mom0, shifted_mom1

    return new_center, masked_cube, new_cube_data

def extract_streamer_centroids(new_cube_data, header, pa_rad, dx_au, center=None):
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
    if center == None:
        im_center = (int(round(header["CRPIX2"] - 1.0)), int(round(header["CRPIX1"] - 1.0)))
    else:
        im_center = center
    v, z, x = np.indices(cube_shape)
    x_rel = x - im_center[1]
    z_rel = z - im_center[0]
    r, theta = pss.spherical_coords(x_rel, z_rel)
    
    find_x = np.array([  0,  -25,  -38,  -56,  -70,  -87, -101, -117, -128, -139, -141, -141])
    find_y = np.array([  0, -41, -45, -41, -35, -27, -18,  -7,  17,  67,  88, 103])
    find_r, find_theta = pss.spherical_coords(find_x, find_y)
    find_streaml = interp1d(find_r, find_theta, fill_value="extrapolate")

    N = 25
    pars = np.linspace(40, 200, N + 1)

    x_means = np.full(N, np.nan)
    z_means = np.full(N, np.nan)
    v_means = np.full(N, np.nan)
    xzstd   = np.full(N, np.nan)
    x_array_list = []
    z_array_list = []
    v_array_list = []
    weights_list = []
    
    for i in tqdm(range(N), desc="[SCrA] centroid step1 (pos.)", ncols=80, leave=False):
        r_mid = 0.5 * (pars[i] + pars[i+1])
        theta0 = find_streaml(r_mid)
        with np.errstate(divide="ignore", invalid="ignore"):
            weight_theta = (x_rel * np.cos(theta0) + z_rel * np.sin(theta0)) / r
        weight_theta[r == 0] = 0
        weight_theta[weight_theta < 0] = 0
        d = (r > pars[i]) & (r <= pars[i+1]) & (new_cube_data > 0)    
        
        # n_vox = np.sum(d)
        # max_I = np.nanmax(new_cube_data[d]) if n_vox > 0 else np.nan
        # print(f"shell {i:2d}, r ~ {r_mid:6.1f} pix, N_vox = {n_vox:6d}, Imax = {max_I:6.3f}")
        
        if np.sum(d) > 0:
            w = new_cube_data[d]            # 只用 intensity 當權重
            if np.sum(w) <= 0:
                x_means[i] = z_means[i] = xzstd[i] = np.nan
            else:
                x_means[i] = np.average(x_rel[d], weights=w)
                z_means[i] = np.average(z_rel[d], weights=w)
                xzstd[i] = np.sqrt(np.average(
                    (x_rel[d] - x_means[i])**2 +
                    (z_rel[d] - z_means[i])**2,
                    weights=w,
                ))
        else:
            x_means[i] = z_means[i] = xzstd[i] = np.nan

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

    # step 3: 高斯方向權重 + velocity
    for i in tqdm(range(N), desc="[SCrA] centroid step2 (vel.)", ncols=80, leave=False):
        r_mid = 0.5 * (pars[i] + pars[i+1])
        if not np.isfinite(x_means[i]):
            x_means[i] = z_means[i] = v_means[i] = np.nan
            continue

        theta_ref = theta_r(r_mid)
        std_ref = std_r(r_mid) / max(r_mid, 1e-3)

        delta_theta = np.pi - np.abs(np.pi - np.abs(theta - theta_ref))
        weights = new_cube_data * pss.gaussian(delta_theta, 0, std_ref)

        d = (r > pars[i]) & (r <= pars[i+1]) & (new_cube_data > 0)
        if np.sum(d) > 0 and np.sum(weights[d]) > 0:
            x_means[i] = np.average(x_rel[d], weights=weights[d])
            z_means[i] = np.average(z_rel[d], weights=weights[d])
            v_means[i] = np.average(v[d],    weights=weights[d])

            x_array_list.append(x_rel[d])
            z_array_list.append(z_rel[d])
            v_array_list.append(v[d])
            weights_list.append(weights[d] / np.nanmax(weights[d]))
        else:
            x_means[i] = z_means[i] = v_means[i] = np.nan
            x_array_list.append(np.array([]))
            z_array_list.append(np.array([]))
            v_array_list.append(np.array([]))
            weights_list.append(np.array([]))

    x_rot = x_means * np.cos(pa_rad) + z_means * np.sin(pa_rad)
    z_rot = -x_means * np.sin(pa_rad) + z_means * np.cos(pa_rad)

    streamer_x_AU = x_rot * dx_au
    streamer_z_AU = z_rot * dx_au

    dv     = float(header["CDELT3"])
    v0     = float(header["CRVAL3"])
    crpix3 = float(header["CRPIX3"])

    streamer_v_km = v0 + (v_means + 1 - crpix3) * dv
    streamer_v_LS = streamer_v_km - Local_Standard_Velocity
    x_array = np.array(x_array_list, dtype=object)
    z_array = np.array(z_array_list, dtype=object)
    v_array = np.array(v_array_list, dtype=object)
    weights_array = np.array(weights_list, dtype=object)
    print(f"[Extracted] {np.sum(np.isfinite(streamer_x_AU))} valid centroids")
    # valid_idx = np.where(np.isfinite(streamer_x_AU))[0]
    # print("Valid shell indices:", valid_idx)
    # print("r range of valid shells (pixel):",
    #     0.5 * (pars[valid_idx] + pars[valid_idx + 1]))
    # print("x_means = ", x_means)
    # print("v_means = ", v_means)
    return streamer_x_AU, streamer_z_AU, streamer_v_LS, x_array, z_array, v_array, weights_array, x_means, z_means, v_means

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

def check_mcmc_convergence(sampler, nsteps=None, min_n_tau=50, require_tau=True):
    info = {}

    acc = np.mean(sampler.acceptance_fraction)
    info["acceptance"] = acc

    lp = sampler.get_log_prob()
    bad_frac = np.mean(~np.isfinite(lp))
    info["bad_frac"] = bad_frac

    # --- 基本 sanity check ---
    if acc < 0.15 or acc > 0.6:
        return False, {**info, "reason": "bad acceptance"}

    if bad_frac > 1e-3:
        return False, {**info, "reason": "too many -inf"}

    if nsteps is None:
        nsteps = sampler.get_chain().shape[0]

    # --- tau diagnostic ---
    try:
        tau = sampler.get_autocorr_time(quiet=True)
        info["tau"] = tau

        # tau sanity
        if (not np.all(np.isfinite(tau))) or np.any(tau <= 0):
            return False, {**info, "reason": "tau invalid"}

        n_tau = nsteps / np.max(tau)
        info["n_tau"] = n_tau

        if n_tau < min_n_tau:
            return False, {**info, "reason": f"chain too short: n_tau={n_tau:.1f} < {min_n_tau}"}

        return True, info

    except Exception as e:
        info["tau_error"] = str(e)
        if require_tau:
            # tau 算不出來通常代表 chain 還不夠長 or 太 noisy
            return False, {**info, "reason": "tau failed"}
        else:
            info["warning"] = "tau failed but acceptance/logprob look OK"
            return True, info

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

def corner_2d_peak(x, y, bins=50, smooth=1.0, xlim=None, ylim=None):
    """
    Mimic corner's 'smooth' peak on a 2D marginal:
      histogram2d (density) -> gaussian_filter(smooth) -> argmax -> bin center
    Returns
    -------
    (x_peak, y_peak), H_smooth, (ix, iy), (xcenters, ycenters)
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    if xlim is None:
        xlo, xhi = np.nanpercentile(x, [0, 100])
    else:
        xlo, xhi = xlim
    if ylim is None:
        ylo, yhi = np.nanpercentile(y, [0, 100])
    else:
        ylo, yhi = ylim

    H, xedges, yedges = np.histogram2d(
        x, y, bins=bins, range=[[xlo, xhi], [ylo, yhi]], density=True
    )
    Hs = gaussian_filter(H, smooth)

    ix, iy = np.unravel_index(np.argmax(Hs), Hs.shape)
    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])
    return (xcenters[ix], ycenters[iy]), Hs, (ix, iy), (xcenters, ycenters)

def kde_5d_peak(samples_plot, ranges, n_candidates=20000, bw_method="scott", seed=0):
    """
    從 samples_plot (畫圖單位) 建 5D KDE（smooth），並在候選點中找最大 density 的點當 peak。
    ranges: list of (min, max) for each dim, 用來限制候選點範圍
    """
    rng = np.random.default_rng(seed)

    s = np.asarray(samples_plot, float)
    ndim = s.shape[1]

    kde = gaussian_kde(s.T, bw_method=bw_method)

    def in_ranges(x):
        ok = np.ones(x.shape[0], dtype=bool)
        for k in range(ndim):
            ok &= (x[:, k] >= ranges[k][0]) & (x[:, k] <= ranges[k][1])
        return ok

    # 候選點先從 samples 抽（快且穩）
    n_take = min(n_candidates, s.shape[0])
    idx = rng.choice(s.shape[0], size=n_take, replace=False)
    cand = s[idx]
    cand = cand[in_ranges(cand)]
    if cand.shape[0] < max(2000, n_take // 10):
        # 如果 ranges 太窄導致候選點不足，退而求其次用全部樣本（仍然會被 KDE 平滑）
        cand = s

    dens = kde(cand.T)
    peak_plot = cand[np.argmax(dens)]
    return peak_plot

def peak_pm(samples_1d, peak, frac_side=0.34):
    """
    Compute peak-centered ±(34%) interval on each side:
      P(peak-err_lo <= x <= peak) = frac_side
      P(peak <= x <= peak+err_hi) = frac_side

    Returns
    -------
    err_lo, err_hi
    """
    s = np.asarray(samples_1d, float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return np.nan, np.nan

    left  = s[s <= peak]
    right = s[s >= peak]

    # 需要每一側至少有一些點，不然會爆
    if left.size < 5 or right.size < 5:
        return np.nan, np.nan

    # 左側：取 (1-frac_side) 分位數，會靠近 peak
    q_left  = np.quantile(left,  1.0 - frac_side)
    # 右側：取 frac_side 分位數，會靠近 peak
    q_right = np.quantile(right, frac_side)

    err_lo = peak - q_left
    err_hi = q_right - peak
    return float(err_lo), float(err_hi)

def draw_2d_interval_lines(
    axes, centers, lo, hi,
    center_color="C0",
    interval_color="k",
    lw_main=1.2, lw_side=1.0,
    alpha_main=0.95, alpha_side=0.75,
    ls_main="-", ls_side="--",
    zorder_center=20,
    zorder_interval=10,
):
    """
    在 corner 的 off-diagonal (i>j) 子圖畫區間線：
    centers/lo/hi 的單位要與 corner 畫圖一致（deg, Myr, ...）
    """
    # --- guard：立刻抓到你傳錯顏色 ---
    if not isinstance(center_color, str):
        raise TypeError(f"center_color must be str, got {type(center_color)}: {center_color!r}")
    if not isinstance(interval_color, str):
        raise TypeError(f"interval_color must be str, got {type(interval_color)}: {interval_color!r}")

    ndim = len(centers)
    for i in range(1, ndim):
        for j in range(i):
            ax = axes[i, j]

            # x（param j）
            ax.axvline(centers[j], color=center_color, lw=lw_main, ls=ls_main,
                       alpha=alpha_main, zorder=zorder_center)
            ax.axvline(lo[j],      color=interval_color, lw=lw_side, ls=ls_side,
                       alpha=alpha_side, zorder=zorder_interval)
            ax.axvline(hi[j],      color=interval_color, lw=lw_side, ls=ls_side,
                       alpha=alpha_side, zorder=zorder_interval)

            # y（param i）
            ax.axhline(centers[i], color=center_color, lw=lw_main, ls=ls_main,
                       alpha=alpha_main, zorder=zorder_center)
            ax.axhline(lo[i],      color=interval_color, lw=lw_side, ls=ls_side,
                       alpha=alpha_side, zorder=zorder_interval)
            ax.axhline(hi[i],      color=interval_color, lw=lw_side, ls=ls_side,
                       alpha=alpha_side, zorder=zorder_interval)

def rebuild_corner_from_cache(which="mcmc_grid", cache_source=None, out_tag="replot"):
    """
    只從 cache 讀 flat samples，重畫 corner plot（median + peak2d）。
    which: "mcmc_grid" or "mcmc_shell"
    cache_source: 用哪個 cache path(預設會用 _resolve_cache_path(USE_CACHE_SOURCE)
    out_tag: 輸出檔名後綴，避免覆蓋舊圖
    """
    if cache_source is None:
        cache_path = _resolve_cache_path(USE_CACHE_SOURCE)
    else:
        cache_path = cache_source

    c = np.load(cache_path, allow_pickle=True)
    print(f"[corner-replot] Loaded cache: {cache_path}")

    key_samples = f"{which}_flat_samples"
    if key_samples not in c:
        raise KeyError(
            f"Cache 缺少 `{key_samples}`，代表你當初沒有把 flat samples 存進去。"
            f"（解法：至少重跑一次 {which} 讓它寫入 cache)"
        )

    flat = c[key_samples]  # shape: (Nsamples, 5) in RAD units for first 3 params
    flat = np.asarray(flat)
    if flat.ndim != 2 or flat.shape[1] != 5:
        raise ValueError(f"{key_samples} 形狀不對：{flat.shape}")

    labels_5d = ["Theta zero", "Phi zero", "Inclination", "Time", "Omega"]

    # ---- 1) median ----
    q16, q50, q84 = np.percentile(flat, [16, 50, 84], axis=0)
    Theta_med, Phi_med, Incl_med, T_med, Omega_med = q50

    # ---- 2) corner plot 用的 samples（角度轉成 deg）----
    samples_plot = flat.copy()
    for idx in [0, 1, 2]:
        samples_plot[:, idx] = np.rad2deg(samples_plot[:, idx])

    labels_plot = [
        r"$\theta$ (deg)",
        r"$\phi_0$ (deg)",
        r"$i$ (deg)",
        r"$t_{\rm s}$ (Myr)",
        r"$\omega$",
    ]

    # ranges：沿用你原本邏輯（以 16-84 的寬度為基礎）
    q16p, q50p, q84p = np.percentile(samples_plot, [16, 50, 84], axis=0)
    ranges = []
    for i in range(len(labels_plot)):
        lo, md, hi = q16p[i], q50p[i], q84p[i]
        width = hi - lo if hi > lo else 1e-3
        ranges.append((md - 1.2 * width, md + 1.2 * width))

    # ---- 3) peak5d：smooth 5D (KDE) global peak ----
    # 仍沿用 cache 裡的 smooth 設定（如果沒有就用預設）
    smooth_corner = float(c.get(f"{which}_peak2d_smooth", 1.0))

    # KDE bandwidth：用 smooth_corner 去控制（越大越平滑）
    # 這裡給一個簡單映射：bw_scale = 1/sqrt(smooth)（可自行調）
    # 如果你想更直覺，也可以直接固定 "scott" / "silverman"
    bw_scale = 1.0 / np.sqrt(max(smooth_corner, 1e-6))

    peak_5d_plot = kde_5d_peak(
        samples_plot,
        ranges=ranges,
        n_candidates=20000,
        bw_method=bw_scale,   # 用 float 控制 KDE 平滑
        seed=0,
    )

    Theta_pk5d_deg, Phi_pk5d_deg, Incl_pk5d_deg, T_pk5d, Omega_pk5d = peak_5d_plot

    peak_5d_rad = peak_5d_plot.copy()
    for k in [0, 1, 2]:
        peak_5d_rad[k] = np.deg2rad(peak_5d_plot[k])

    # 這組是 rad 單位（給 peak_pm 用）
    Theta_pk5d, Phi_pk5d, Incl_pk5d, T_pk5d_rad, Omega_pk5d_val = peak_5d_rad

    print(f"[corner-replot] peak5d (KDE smooth) = {peak_5d_plot}")

    # ---- 4) median corner ----
    fig = corner.corner(
        samples_plot,
        labels=labels_plot,
        range=ranges,
        show_titles=True,
        plot_contours=True,
        title_fmt=".3f",
        quantiles=[0.16, 0.5, 0.84],
        truths=[np.rad2deg(Theta_med), np.rad2deg(Phi_med), np.rad2deg(Incl_med), T_med, Omega_med],
        smooth=1.0,
    )
    axes = np.array(fig.axes).reshape((5, 5))

    cent_med = q50p
    lo_med   = q16p
    hi_med   = q84p

    draw_2d_interval_lines(
        axes,
        centers=cent_med,   # q50p
        lo=lo_med,          # q16p
        hi=hi_med,          # q84p
        center_color="C0",
        interval_color="k",
        lw_main=1.3,
        lw_side=1.0,
        alpha_main=0.9,
        alpha_side=0.75,
        ls_main="-",
        ls_side="--",
    )
    out1 = os.path.join(PLOT_DIR, f"corner_{which}_median_{out_tag}.png")
    fig.savefig(out1, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[corner-replot] Saved: {out1}")

    # ---- 5) peak corner + title(±區間線) ----
    fig = corner.corner(
        samples_plot,
        labels=labels_plot,
        range=ranges,
        show_titles=False,
        plot_contours=True,
        title_fmt=".3f",
        truths=[Theta_pk5d_deg, Phi_pk5d_deg, Incl_pk5d_deg, T_pk5d, Omega_pk5d],
        smooth=smooth_corner,
    )
    axes = np.array(fig.axes).reshape((5, 5))

    # 你原本的 title 生成工具（簡化沿用）
    def clip_zero(x, atol=5e-13):
        x = np.asarray(x, dtype=float)
        x[np.isclose(x, 0.0, atol=atol)] = 0.0
        return x

    def fmt_pm(x, nd=3):
        if not np.isfinite(x):
            return "?"
        return f"{x:.{nd}f}"

    def sup(x, nd=3):
        return rf"^{{+{fmt_pm(x, nd)}}}"

    def sub(x, nd=3):
        if not np.isfinite(x):
            return rf"_{{{fmt_pm(x, nd)}}}"
        if x == 0.0:
            return rf"_{{{fmt_pm(x, nd)}}}"
        return rf"_{{-{fmt_pm(x, nd)}}}"

    # 這裡沿用你現行：peak-centered 的左右各取 frac_side=0.68（你原碼就是這樣）
    err_lo = np.zeros(5)
    err_hi = np.zeros(5)
    for k in range(5):
        err_lo[k], err_hi[k] = peak_pm(flat[:, k], peak_5d_rad[k], frac_side=0.68)

    err_lo = clip_zero(err_lo)
    err_hi = clip_zero(err_hi)

    err_lo_deg = err_lo.copy()
    err_hi_deg = err_hi.copy()
    for i in [0, 1, 2]:
        err_lo_deg[i] = np.rad2deg(err_lo[i])
        err_hi_deg[i] = np.rad2deg(err_hi[i])
    err_lo_deg = clip_zero(err_lo_deg)
    err_hi_deg = clip_zero(err_hi_deg)

    titles = [
        rf"$\theta_0\ (\mathrm{{deg}}) = {Theta_pk5d_deg:.3f}" + sup(err_hi_deg[0]) + sub(err_lo_deg[0]) + r"$",
        rf"$\phi_0\ (\mathrm{{deg}}) = {Phi_pk5d_deg:.3f}"   + sup(err_hi_deg[1]) + sub(err_lo_deg[1]) + r"$",
        rf"$i\ (\mathrm{{deg}}) = {Incl_pk5d_deg:.3f}"       + sup(err_hi_deg[2]) + sub(err_lo_deg[2]) + r"$",
        rf"$t_{{\rm s}}\ (\mathrm{{Myr}}) = {T_pk5d:.3f}"    + sup(err_hi[3])     + sub(err_lo[3])     + r"$",
        rf"$\omega = {Omega_pk5d:.3f}"                       + sup(err_hi[4])     + sub(err_lo[4])     + r"$",
    ]
    for k in range(5):
        axes[k, k].set_title(titles[k], fontsize=12)

    peak_plot = np.array([Theta_pk5d_deg, Phi_pk5d_deg, Incl_pk5d_deg, T_pk5d, Omega_pk5d], float)    
    err_lo_plot = err_lo.copy()
    err_hi_plot = err_hi.copy()
    for i in [0, 1, 2]:
        err_lo_plot[i] = np.rad2deg(err_lo_plot[i])
        err_hi_plot[i] = np.rad2deg(err_hi_plot[i])

    lo_plot = peak_plot - err_lo_plot
    hi_plot = peak_plot + err_hi_plot
    for i in range(5):
        ax = axes[i, i]
        ax.axvline(lo_plot[i], ls="--", lw=1.2, color="k", alpha=0.9)
        ax.axvline(hi_plot[i], ls="--", lw=1.2, color="k", alpha=0.9)
    draw_2d_interval_lines(
        axes,
        centers=peak_plot,
        lo=lo_plot,
        hi=hi_plot,
        center_color="C0",      # 你說 center 想維持藍色
        interval_color="k",
        lw_main=1.3,
        lw_side=1.0,
        alpha_main=0.9,
        alpha_side=0.75,
        ls_main="-",
        ls_side="--",
    )
    out2 = os.path.join(PLOT_DIR, f"corner_{which}_peak5d_{out_tag}.png")
    fig.savefig(out2, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[corner-replot] Saved: {out2}")

def _compute_extent(header, im_center, ny, nx):
    dx_arcsec = header["CDELT1"] * 3600.0
    dz_arcsec = header["CDELT2"] * 3600.0
    ra_min = (0   - im_center[1]) * dx_arcsec
    ra_max = (nx  - im_center[1]) * dx_arcsec
    dec_min= (0   - im_center[0]) * dz_arcsec
    dec_max= (ny  - im_center[0]) * dz_arcsec
    return (ra_min, ra_max, dec_min, dec_max), dx_arcsec, dz_arcsec

def plot_streamer_on_mom0(theta, phi, inc, T_Myr, omega,
                          header, pa_rad, dx_au, new_center,
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
    extent, dx_arcsec, dz_arcsec = _compute_extent(header, new_center, ny, nx)

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

    # AU -> pixel
    x_pix = x_m / dx_au
    z_pix = z_m / dx_au

    # rotate to image frame
    x_pix_rot = x_pix * np.cos(pa_rad) - z_pix * np.sin(pa_rad)
    z_pix_rot = x_pix * np.sin(pa_rad) + z_pix * np.cos(pa_rad)

    ra_off  = x_pix_rot * dx_arcsec
    dec_off = z_pix_rot * dz_arcsec

    pts = np.column_stack([ra_off, dec_off])
    if pts.shape[0] < 2:
        print("[plot_streamer_on_mom0] model points too few, skip.")
        return
    segments = np.stack([pts[:-1], pts[1:]], axis=1)

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

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.04)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("(Jy/beam km/s)")

    # model line
    lc_edge = LineCollection(segments, colors="black", linewidth=4, zorder=2)
    ax.add_collection(lc_edge)

    lc = LineCollection(
        segments,
        cmap="coolwarm",
        norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax),
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

    # centroids
    # if cen_x_pix is not None and cen_z_pix is not None:
    #     cen_ra  = (cen_x_pix - new_center[1]) * dx_arcsec
    #     cen_dec = (cen_z_pix - new_center[0]) * dz_arcsec

    #     if cen_v_LS_km is not None:
    #         ax.scatter(
    #             cen_ra, cen_dec,
    #             c=np.asarray(cen_v_LS_km) + Local_Standard_Velocity,
    #             cmap="coolwarm", vmin=vmin, vmax=vmax,
    #             s=10, edgecolors="black", linewidths=0.7,
    #             zorder=5,
    #         )
    #     else:
    #         ax.scatter(
    #             cen_ra, cen_dec,
    #             facecolors="none", edgecolors="black",
    #             s=10, zorder=5,
    #         )

    # 中心位置
    ax.scatter(0, 0, c="C0", s=60, marker="+", zorder=6)

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
    ax.plot(scale_range_x, scale_range_y, color='w', lw=3, zorder=10)
    ax.text(
        text_pos_x - scale_length_arcsec / 2,
        text_pos_y - 2.0,
        f"{int(scale_length)} AU",
        ha='center', va='bottom',
        fontsize=14, family='Times New Roman', color='w'
    )
    beam = mpl.patches.Ellipse(
        (0, 0),
        width=70 * dx_arcsec,
        height=70 * dx_arcsec,
        angle=0,
        facecolor="none",
        edgecolor="w",
        linestyle="--",
        lw=1.2,
        zorder=12,
    )
    ax.add_patch(beam)
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
                          header, pa_rad, dx_au, new_center,
                          mom1, label, outname,
                          v_range=1.0,
                          vmin=None, vmax=None,
                          radius_in_au=radius_in_au,
                          radius_out_au=radius_out_au,
                          scale='log', log_power=log_power,
                          cen_x_pix=None, cen_z_pix=None, cen_v_LS_km=None):

    dx_arcsec = abs(header["CDELT1"]) * 3600.0
    dz_arcsec = abs(header["CDELT2"]) * 3600.0
    ny, nx = mom1.shape
    extent, dx_arcsec, dz_arcsec = _compute_extent(header, new_center, ny, nx)

    x_m, y_m, z_m, u_m, v_m, w_m = pss.PSS_model(
        theta, phi, inc, T_Myr, omega,
        M_star,
        radius_in_au=radius_in_au,
        radius_out_au=radius_out_au,
        resolution=200,
        scale=scale,
        log_power=log_power,
    )

    x_pix = x_m / dx_au
    z_pix = z_m / dx_au

    x_pix_rot = x_pix * np.cos(pa_rad) - z_pix * np.sin(pa_rad)
    z_pix_rot = x_pix * np.sin(pa_rad) + z_pix * np.cos(pa_rad)

    ra_off  = x_pix_rot * dx_arcsec
    dec_off = z_pix_rot * dz_arcsec

    pts = np.column_stack([ra_off, dec_off])
    if pts.shape[0] < 2:
        print("[plot_streamer_on_mom1] model points too few, skip.")
        return
    segments = np.stack([pts[:-1], pts[1:]], axis=1)

    if vmin is None or vmax is None:
        vmin = Local_Standard_Velocity - v_range
        vmax = Local_Standard_Velocity + v_range

    fig, ax = plt.subplots(figsize=(6.2, 6))
    im = ax.imshow(
        mom1,
        origin="lower",
        cmap="coolwarm",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        # vmin=np.nanmin(mom1),
        # vmax=np.nanmax(mom1),
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.04)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Velocity (km/s)")
    
    lc_edge = LineCollection(segments, colors="black", linewidth=4, zorder=2)
    ax.add_collection(lc_edge)

    lc = LineCollection(
        segments,
        cmap="coolwarm",
        norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax),
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
    

    # if cen_x_pix is not None and cen_z_pix is not None:
    #     cen_ra  = (cen_x_pix - new_center[1]) * dx_arcsec
    #     cen_dec = (cen_z_pix - new_center[0]) * dz_arcsec

    #     if cen_v_LS_km is not None:
    #         ax.scatter(
    #             cen_ra, cen_dec,
    #             c=np.asarray(cen_v_LS_km) + Local_Standard_Velocity,
    #             cmap="coolwarm", vmin=vmin, vmax=vmax,
    #             s=10, edgecolors="black", linewidths=0.7,
    #             zorder=5,
    #         )
    #     else:
    #         ax.scatter(
    #             cen_ra, cen_dec,
    #             facecolors="none", edgecolors="black",
    #             s=10, zorder=5,
    #         )

    ax.scatter(0, 0, c="C0", s=70, marker="+", zorder=6)

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
    beam = mpl.patches.Ellipse(
        (0, 0),
        width=70 * dx_arcsec,
        height=70 * dx_arcsec,
        angle=0,
        facecolor="none",
        edgecolor="k",
        linestyle="--",
        lw=1.2,
        zorder=12,
    )
    ax.add_patch(beam)
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
    im_cy = float(header["CRPIX2"])  # FITS -> 0-based
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
        label="Model",
        zorder=3,
    )

    # ---------- 6) 疊上質心 ----------
    # if z_means_pix is not None and streamer_v_LS_km is not None:
    #     # z_means_pix 已經是相對 protostar 的影像座標像素位移，直接乘上 dx_au 變成 AU
    #     z_cent_img_AU = np.asarray(z_means_pix) * dx_au
    #     v_cent = np.asarray(streamer_v_LS_km) + Local_Standard_Velocity
    #     good = np.isfinite(z_cent_img_AU) & np.isfinite(v_cent)
    #     ax.scatter(
    #         z_cent_img_AU[good],
    #         v_cent[good],
    #         c="tab:blue",
    #         s=30,
    #         edgecolors="k",
    #         linewidths=0.6,
    #         label="Centroids",
    #         zorder=4,
    #     )

    # ---------- 7) 裝飾 ----------
    ax.set_xlabel("z (AU, image frame)")
    ax.set_ylabel("Velocity (km/s, LSR)")
    ax.set_title(label)
    ax.set_ylim(4, 8)
    ax.set_xlim(1200, -800)
    leg = ax.legend(frameon=False, fontsize=9)
    for txt in leg.get_texts():
        txt.set_color("white")
    ax.grid(alpha=0.2)

    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, outname), dpi=200)
    plt.close(fig)
    print(f"[z–v] Saved {outname}")
    
def run_quick_mode_scra():
    print("[Quick Mode] RUN_FROM_CACHE_ONLY=True → 僅讀取 cache 並繪圖")

    try:
        # --------------------------------------------------
        # 1) Load cache
        # --------------------------------------------------
        cache_path = _resolve_cache_path(USE_CACHE_SOURCE)
        c = np.load(cache_path, allow_pickle=True)
        print(f"[cache] Loaded cache ({USE_CACHE_SOURCE}): {cache_path}")

        Theta_best, Phi_best, Incl_best, T_best, Omega_best = \
            _extract_params_from_cache(c, USE_CACHE_SOURCE)

        Theta_best_deg = np.rad2deg(Theta_best)
        Phi_best_deg   = np.rad2deg(Phi_best)
        Incl_best_deg  = np.rad2deg(Incl_best)
        
        #mcmc_grid peak
        # Theta_best_deg = 89.614 
        # Phi_best_deg   = 22.610 
        # Incl_best_deg  = -81.219 
        # Theta_best     = np.deg2rad(Theta_best_deg) 
        # Phi_best       = np.deg2rad(Phi_best_deg)
        # Incl_best      = np.deg2rad(Incl_best_deg)
        # T_best         = 0.792 
        # Omega_best     = 0.217 
        
        # shell peak
        Theta_best_deg = 89.709
        Phi_best_deg   = 7.134 
        Incl_best_deg  = -81.591 
        Theta_best     = np.deg2rad(Theta_best_deg) 
        Phi_best       = np.deg2rad(Phi_best_deg)
        Incl_best      = np.deg2rad(Incl_best_deg)
        T_best         = 0.940 
        Omega_best     = 0.230 
        # ★ A-mode center: 必須從 cache 讀 ★
        if "new_center_yx" not in c:
            raise RuntimeError("Cache missing new_center_yx (A-mode requires this).")
        new_center = tuple(c["new_center_yx"])
        print(f"[A-mode] Using new_center (y,x) = {new_center}")

        # --------------------------------------------------
        # 2) Load moment maps
        # --------------------------------------------------
        rms_channel = 0.026211100061251217

        str_mom0 = fits.getdata("S_CrA_13CO_streamer_mom0.fits")
        str_mom1 = fits.getdata("S_CrA_13CO_streamer_mom1.fits")
        mom0     = fits.getdata("S_CrA_13CO_mom0.fits")
        mom1     = np.where(
            mom0 > 3 * rms_channel,
            fits.getdata("S_CrA_13CO_mom1.fits"),
            np.nan,
        )
        header = fits.getheader("S_CrA_13CO_streamer_mom1.fits")

        dx_arcsec = abs(header["CDELT1"]) * 3600.0
        dx_au     = dx_arcsec * distance_pc

        # --------------------------------------------------
        # 3) Load streamer cube (optional, for z–v)
        # --------------------------------------------------
        try:
            new_cube_data = fits.getdata("S_CrA_13CO_streamer_cube.fits")
            print("[cache] Loaded streamer cube")
        except Exception:
            new_cube_data = None
            print("[cache] No streamer cube → skip z–v diagram")

        # --------------------------------------------------
        # 4) Centroids (model-frame → image-frame pixel)
        # --------------------------------------------------
        cen_x_pix = cen_z_pix = cen_v_LS = None

        if "streamercom_x_AU" in c and "streamercom_z_AU" in c:
            sx = c["streamercom_x_AU"]
            sz = c["streamercom_z_AU"]

            x_rot = sx / dx_au
            z_rot = sz / dx_au

            cen_x_pix = (
                x_rot * np.cos(pa_rad) - z_rot * np.sin(pa_rad)
                + new_center[1]
            )
            cen_z_pix = (
                x_rot * np.sin(pa_rad) + z_rot * np.cos(pa_rad)
                + new_center[0]
            )

            cen_v_LS = c.get("streamercom_v_LS_km", None)

            x_array = c.get("x_array", None)
            z_array = c.get("z_array", None)
            weights_array = c.get("weights_array", None)

        # --------------------------------------------------
        # 5) z–v diagram
        # --------------------------------------------------
        if new_cube_data is not None and cen_v_LS is not None:
            z_means_pix = c["z_means"] if "z_means" in c else None
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
                z_means_pix=z_means_pix,
                streamer_v_LS_km=cen_v_LS,
                outname="SCrA_z_v_data_overlay.png",
                label="S CrA $^{13}$CO z–v",
            )

        # --------------------------------------------------
        # 6) moment overlays (A-mode!)
        # --------------------------------------------------
        plot_streamer_on_mom1(
            Theta_best, Phi_best, Incl_best,
            float(T_best), float(Omega_best),
            header, pa_rad, dx_au, new_center,
            str_mom1,
            label="S CrA $^{13}$CO",
            outname="SCrA_mom1_cacheonly.png",
            cen_x_pix=cen_x_pix,
            cen_z_pix=cen_z_pix,
            cen_v_LS_km=cen_v_LS,
        )

        plot_streamer_on_mom0(
            Theta_best, Phi_best, Incl_best,
            float(T_best), float(Omega_best),
            header, pa_rad, dx_au, new_center,
            str_mom0,
            label="S CrA $^{13}$CO",
            outname="SCrA_mom0_cacheonly.png",
            cen_x_pix=cen_x_pix,
            cen_z_pix=cen_z_pix,
            cen_v_LS_km=cen_v_LS,
        )

        if x_array is not None and weights_array is not None:
            plot_r_theta_weights_from_output(
                x_array, z_array, weights_array,
                outname="SCrA_weights_cacheonly.png"
            )

        # 8) Print physical values
        r_ref_AU = 200 * T_best * 1e6 * spc.year / spc.astronomical_unit  # radius_ref_au=200

        M_0 = M_star * M_SUN_KG * spc.G / (200.0**3 * T_best * 1e6 * spc.year)
        M_dot = M_star / (T_best * 1e6)

        print("\n==================== Parameters (SCrA) ====================")
        print(f"Theta        = {Theta_best_deg:.3f} deg")
        print(f"Phi          = {Phi_best_deg:.3f} deg")
        print(f"Inclination  = {Incl_best_deg:.3f} deg")
        print(f"Time (T_Myr) = {T_best:.6f} Myr")
        print(f"Omega        = {Omega_best:.4f}")
        print(f"r_ref        = {r_ref_AU:.2f} AU")
        print(f"M_0          = {M_0:.3e}")
        print(f"Mdot         = {M_dot:.3e} M_sun/yr")
        print(dx_au * 70)
        print("============================================================")
        print("[Quick Mode] Done.")
        sys.exit(0)

    except Exception as e:
        print(f"[Quick Mode] failed → fallback to full run: {e}")
# ============================================================
# 3. 資料準備：讀 cube + 建 mask + 抽質心
# ============================================================
def prepare_data():
    global cube, header, new_center, dx_arcsec, dx_au, dv
    global v_lastch_vel, v_lastch_num, subcube, moment0, moment1
    global rms_channel, new_cube_data, str_mom0, str_mom1
    global streamercom_x_AU, streamercom_z_AU, streamercom_v_LS_km
    global x_array, z_array, v_array, weights_array, x_means, z_means, v_means
    global v_weight_phys, max_dist_value

    # --------------------------------------------------
    # 1) Read cube & header
    # --------------------------------------------------
    cube = SpectralCube.read(cube_fname)
    header = fits.getheader(cube_fname)

    # --- Ensure spectral axis is in km/s (do NOT subtract Vsys) ---
    cube = cube.with_spectral_unit(
        u.km / u.s,
        velocity_convention="radio",
        rest_value=header["RESTFRQ"] * u.Hz,
    )

    im_center = (int(round(header["CRPIX2"] - 1.0)),
                int(round(header["CRPIX1"] - 1.0)))
    dx_arcsec = abs(header["CDELT1"]) * 3600.0
    dx_au     = dx_arcsec * distance_pc
    # REMOVE old dv, v0 from 2D header (do not use these anymore)
    # dv        = abs(float(header["CDELT3"]))
    # v0        = float(header["CRVAL3"])

    # --------------------------------------------------
    # 2) Subcube / moments
    # --------------------------------------------------
    v_lastch_vel = 14.8636
    v_lastch_num = 150
    velocity_range = [2.4926, 14.8636] * u.km / u.s

    subcube = cube.spectral_slab(velocity_range[0], velocity_range[1])

    # --- true velocity axis from subcube (km/s) ---
    spec_kms = subcube.spectral_axis.to_value(u.km / u.s)
    dv = float(spec_kms[1] - spec_kms[0])

    moment0 = subcube.moment(order=0).value
    moment1 = subcube.moment(order=1).value
    rms_channel = 0.026211100061251217

    # Save original moments (original header)
    fits.PrimaryHDU(data=moment0, header=header).writeto("S_CrA_13CO_mom0.fits", overwrite=True)
    print("[mask] Saved S_CrA_13CO_mom0.fits")

    hdu_mom1 = fits.PrimaryHDU(data=moment1, header=header)
    hdu_mom1.header["BUNIT"] = "km/s"
    hdu_mom1.writeto("S_CrA_13CO_mom1.fits", overwrite=True)
    print("[mask] Saved S_CrA_13CO_mom1.fits")

    # --------------------------------------------------
    # 3) Build streamer cube (masked) + define new_center
    # --------------------------------------------------
    new_center, masked_cube, new_cube_data = build_streamer_masked_cube(
        subcube, header, rms_channel
    )

    # --- Use 3D header from masked_cube (REQUIRED) ---
    header_stream = masked_cube.hdu.header.copy()
    # Keep A-mode center (CRPIX is 1-based in FITS)
    header_stream["CRPIX1"] = float(new_center[1]) + 1.0
    header_stream["CRPIX2"] = float(new_center[0]) + 1.0
    # Explicitly declare velocity unit
    header_stream["CUNIT3"] = "km/s"

    # Save streamer cube + streamer moments (use header_stream)
    try:
        fits.PrimaryHDU(data=new_cube_data, header=header_stream).writeto(
            "S_CrA_13CO_streamer_cube.fits", overwrite=True
        )
        print("[mask] Saved S_CrA_13CO_streamer_cube.fits")
    except Exception as e:
        print(f"[mask] Failed to save streamer cube FITS: {e}")

    str_mom0 = masked_cube.moment(order=0).value
    str_mom1 = masked_cube.moment(order=1).value

    fits.PrimaryHDU(data=str_mom0, header=header_stream).writeto(
        "S_CrA_13CO_streamer_mom0.fits", overwrite=True
    )
    print("[mask] Saved S_CrA_13CO_streamer_mom0.fits")

    fits.PrimaryHDU(data=str_mom1, header=header_stream).writeto(
        "S_CrA_13CO_streamer_mom1.fits", overwrite=True
    )
    print("[mask] Saved S_CrA_13CO_streamer_mom1.fits")

    # --------------------------------------------------
    # 4) Extract centroids (A-mode: center=new_center)
    # --------------------------------------------------
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
            new_cube_data,
            header_stream,   # ← 用 streamer header（CRPIX=new_center）
            pa_rad,
            dx_au,
            center=new_center
        )
    except Exception as e:
        print("抽取質心失敗，請檢查 build_streamer_masked_cube 或資料品質：", e)
        raise
    # print(streamercom_v_LS_km)
    # --------------------------------------------------
    # 5) Weights & cache
    # --------------------------------------------------
    v_weight_phys = (dx_au / dv) ** 2
    max_dist_value = 100.0

    cache.update({
        "streamercom_x_AU": streamercom_x_AU,
        "streamercom_z_AU": streamercom_z_AU,
        "streamercom_v_LS_km": streamercom_v_LS_km,

        "x_array": np.array(x_array, dtype=object),
        "z_array": np.array(z_array, dtype=object),
        "v_array": np.array(v_array, dtype=object),
        "weights_array": np.array(weights_array, dtype=object),

        "x_means": x_means,
        "z_means": z_means,
        "v_means": v_means,

        "v_weight_phys": float(v_weight_phys),
        "max_dist_value": float(max_dist_value),

        # centers
        "im_center_yx": np.array(im_center),
        "new_center_yx": np.array(new_center),
    })

    np.savez(CACHE_PATH_GRID, **cache)
    print(f"[cache] Saved prepare_data cache to {CACHE_PATH_GRID}")

# --- Grid fitting logic moved into function ---
def run_grid():
    global Theta_init, Phi_init, Incl_init, T_init, Omega_init
    global parameter_prior_ranges, sigma_like
    if RUN_GRID:
        best_params, grid, error = run_grid_search(
            streamercom_x_AU, streamercom_z_AU, streamercom_v_LS_km,
            v_weight_phys, M_star, scale, log_power, radius_ref_au,
            n_grid=10,
            phi_grid=(0, 2 * np.pi),
            T_factor_range=(2.253e-02, 1.079), #full range(1.01, 46.45)400, (65, 150), (4.464e-1, 2.053)800, (10.062, 462.7)240
            verbose=True,
        )

        sigma_like = None
        parameter_prior_ranges, sigma_like = compute_priors_from_grid(
            error, grid, best_params["best_val"], phi_range=(0, 2 * np.pi),
        ) #phi_range=(0, 2 * np.pi),

        Theta_init = best_params["Theta"] # rad
        Phi_init   = best_params["Phi"] # rad
        Incl_init  = best_params["Incl"] # rad
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
        # 8) Print physical values
        r_ref_AU = radius_ref_au * T_init * 1e6 * spc.year / spc.astronomical_unit  # radius_ref_au=280

        M_0 = M_star * M_SUN_KG * spc.G / (radius_ref_au**3 * T_init * 1e6 * spc.year)
        M_dot = M_star / (T_init * 1e6)

        print("\n==================== Parameters (S CrA) ====================")
        print(f"Theta        = {np.rad2deg(Theta_init):.3f} deg")
        print(f"Phi          = {np.rad2deg(Phi_init):.3f} deg")
        print(f"Inclination  = {np.rad2deg(Incl_init):.3f} deg")
        print(f"Time (T_Myr) = {T_init:.6f} Myr")
        print(f"Omega        = {Omega_init:.4f}")
        print(f"r_ref        = {r_ref_AU:.2f} AU")
        print(f"M_0          = {M_0:.3e}")
        print(f"Mdot         = {M_dot:.3e} M_sun/yr")
        print("============================================================")
        print("[Quick Mode] 繪圖完成，程式結束。")
    else:
        print("[Grid fitting] Skipped (manual init used).")

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
    print("Grid best:") 
    print(f"Theta = {np.rad2deg(Theta_center):.3f} deg")
    print(f"Phi   = {np.rad2deg(Phi_center):.3f} deg")
    print(f"Incl  = {np.rad2deg(Incl_center):.3f} deg")
    print(f"T     = {T_center:.6f} Myr")
    print(f"Omega = {Omega_center:.4f}")
    center_vals = [Theta_center, Phi_center, Incl_center, T_center, Omega_center]
    
    ndim = 5
    nwalkers = 20
    nsteps = 30000
    labels_5d = ["Theta zero", "Phi zero", "Inclination", "Time", "Omega"]
    p0 = np.zeros((nwalkers, ndim))

    sigma_vals  = [
        np.deg2rad(9.0),
        np.deg2rad(18.0),
        np.deg2rad(9.0),
        0.05 * (parameter_prior_ranges["Time"][1] - parameter_prior_ranges["Time"][0]),
        0.05 * (parameter_prior_ranges["Omega"][1] - parameter_prior_ranges["Omega"][0]),
    ]

    for j, key in enumerate(labels_5d):
        lo, hi = parameter_prior_ranges[key]
        proposal = center_vals[j] + sigma_vals[j] * np.random.randn(nwalkers)
        proposal = np.clip(proposal, lo, hi)
        p0[:, j] = proposal

    moves = get_mcmc_moves(mode="refine")
    # -----------------------------
    # Retry settings (全新重跑)
    # -----------------------------
    MAX_RETRY = 5
    success = False
    sampler = None
    conv_info = None
    sampler_try_last = None
    conv_info_last = None

    for attempt in range(MAX_RETRY + 1):
        print(f"\n[MCMC_grid] full-run attempt {attempt+1}/{MAX_RETRY+1}")

        # --- re-init p0 from scratch every attempt ---
        p0 = np.zeros((nwalkers, ndim))
        for j, key in enumerate(labels_5d):
            lo, hi = parameter_prior_ranges[key]
            prop = center_vals[j] + sigma_vals[j] * np.random.randn(nwalkers)
            prop = np.clip(prop, lo, hi)
            p0[:, j] = prop

        with Pool(processes=8) as pool:
            sampler_try = emcee.EnsembleSampler(
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
                    sigma_like,
                ),
                pool=pool,
                moves=moves,
            )
            sampler_try.run_mcmc(p0, nsteps, progress=True)

        ok, info = check_mcmc_convergence(sampler_try, nsteps=nsteps)
        print("[MCMC_grid] convergence check:", info)

        print("mean acceptance:", np.mean(sampler_try.acceptance_fraction))
        lp_chain = sampler_try.get_log_prob()
        print("non-finite log_prob fraction =", np.mean(~np.isfinite(lp_chain)))

        # 記住最後一次（不論好壞）
        sampler_try_last = sampler_try
        conv_info_last = info

        if ok:
            sampler = sampler_try
            conv_info = info
            success = True
            print("[MCMC_grid] ✓ Accepted this run")
            break
        else:
            print("[MCMC_grid] ✗ Re-run from scratch (bad chain)")

    # ✅ 最後處理：沒通過 gate 也不要結束，改用最後一次結果
    if not success:
        print("[MCMC_grid] ⚠ WARNING: did not pass convergence gate after retries.")
        print("[MCMC_grid] → Using the LAST run anyway (for downstream plots/results).")
        sampler = sampler_try_last
        conv_info = conv_info_last

    # 自動 burn-in / thin
    try:
        tau = sampler.get_autocorr_time(quiet=True)
        if (not np.all(np.isfinite(tau))) or (np.any(tau <= 0)):
            raise RuntimeError(f"tau invalid: {tau}")
        burnin = int(2 * np.nanmax(tau))
        thin   = max(1, int(1 * np.nanmin(tau)))
        print(f"[MCMC_grid] tau: {tau}, burnin={burnin}, thin={thin}")
    except Exception as e:
        print("[MCMC_grid] tau failed, use default.", e)
        burnin, thin = 100, 50

    chain = sampler.get_chain()
    print("chain shape:", chain.shape)  # (nsteps, nwalkers, ndim)
    print("mean acceptance:", np.mean(sampler.acceptance_fraction))

    lp_chain = sampler.get_log_prob()
    print("non-finite log_prob fraction =", np.mean(~np.isfinite(lp_chain)))
    print(parameter_prior_ranges)
    
    flat    = sampler.get_chain(discard=burnin, thin=thin, flat=True)
    lp_flat = sampler.get_log_prob(discard=burnin, thin=thin, flat=True)

    q16, q50, q84 = np.percentile(flat, [16, 50, 84], axis=0)
    Theta_med, Phi_med, Incl_med, T_med, Omega_med = q50
    
    print("\n[MCMC_grid] median ±68%:")
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

    print("\n[MCMC_grid] 1D posterior shape:")
    for i, name in enumerate(labels_5d):
        summarize_1d_posterior(flat[:, i], name)

    # corner plot（角度轉度）
    samples_plot = flat.copy()
    for idx in [0, 1, 2]:
        samples_plot[:, idx] = np.rad2deg(samples_plot[:, idx])
    labels_plot = [
    r"$\theta$ (deg)",
    r"$\phi_0$ (deg)",
    r"$i$ (deg)",
    r"$t_{\rm s}$ (Myr)",
    r"$\omega$",
    ]
    q16p, q50p, q84p = np.percentile(samples_plot, [16, 50, 84], axis=0)
    
    smooth_corner = 1.0   # must match corner.corner(..., smooth=1.0)
    bins_corner   = 50    # choose consistent bins; can tune to your sample size

    pair_peaks = {}  # (i,j) -> (peak_i, peak_j)

    ranges = []
    for i in range(len(labels_plot)):
        lo, md, hi = q16p[i], q50p[i], q84p[i]
        width = hi - lo if hi > lo else 1e-3
        ranges.append((md - 1.2*width, md + 1.2*width))

    peak_5d_plot = kde_5d_peak(
        samples_plot,
        ranges=ranges,
        n_candidates=20000,     # 可調：1e4~5e4 常用
        bw_method="scott",      # 可調："silverman" 或 float
        seed=0
    )

    Theta_pk5d_deg, Phi_pk5d_deg, Incl_pk5d_deg, T_pk5d, Omega_pk5d = peak_5d_plot

    peak_5d_rad = peak_5d_plot.copy()
    for k in [0, 1, 2]:
        peak_5d_rad[k] = np.deg2rad(peak_5d_plot[k])

    Theta_pk5d_rad, Phi_pk5d_rad, Incl_pk5d_rad, T_pk5d_val, Omega_pk5d_val = peak_5d_rad
    print("\n[MCMC_grid] 5D KDE-smoothed peak (global peak in smoothed 5D posterior):")
    print(f"Theta = {Theta_pk5d_deg:.3f} deg")
    print(f"Phi   = {Phi_pk5d_deg:.3f} deg")
    print(f"Incl  = {Incl_pk5d_deg:.3f} deg")
    print(f"T     = {T_pk5d:.6f} Myr")
    print(f"Omega = {Omega_pk5d:.4f}")

    fig = corner.corner(samples_plot,
                        labels=labels_plot,
                        range=ranges,
                        show_titles=True,
                        plot_contours=True,
                        title_fmt=".3f",
                        quantiles=[0.16, 0.5, 0.84],
                        truths=[np.rad2deg(Theta_med), np.rad2deg(Phi_med), np.rad2deg(Incl_med), T_med, Omega_med],
                        smooth=1.0)
    axes = np.array(fig.axes).reshape((ndim, ndim))

    # 用「畫圖單位」的 median 16/50/84：你前面已經算過 q16p, q50p, q84p
    cent_med = q50p
    lo_med   = q16p
    hi_med   = q84p

    draw_2d_interval_lines(
        axes,
        centers=cent_med,
        lo=lo_med,
        hi=hi_med,
        center_color="C0",      # median 中心線顏色
        interval_color="k",     # median 區間線顏色（想更清楚可改成 "w"）
        lw_main=1.3,
        lw_side=1.0,
        alpha_main=0.9,
        alpha_side=0.75,
        ls_main="-",
        ls_side="--",
    )
    fig.savefig(os.path.join(PLOT_DIR, "corner_mcmc_grid_median.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    
    fig = corner.corner(
        samples_plot,
        labels=labels_plot,
        range=ranges,
        show_titles=False,
        plot_contours=True,
        title_fmt=".3f",
        truths=[Theta_pk5d_deg, Phi_pk5d_deg, Incl_pk5d_deg, T_pk5d, Omega_pk5d],
        smooth=smooth_corner,
    )
    axes = np.array(fig.axes).reshape((ndim, ndim))

    def clip_zero(x, atol=5e-13):
        x = np.asarray(x, dtype=float)
        x[np.isclose(x, 0.0, atol=atol)] = 0.0
        return x

    def fmt_pm(x, nd=3):
        if not np.isfinite(x):
            return "?"
        return f"{x:.{nd}f}"

    def sup(x, nd=3):
        return rf"^{{+{fmt_pm(x, nd)}}}"

    def sub(x, nd=3):
        if not np.isfinite(x):
            return rf"_{{{fmt_pm(x, nd)}}}"
        if x == 0.0:
            return rf"_{{{fmt_pm(x, nd)}}}"
        return rf"_{{-{fmt_pm(x, nd)}}}"

    # ---- peak-centered ±34% each side ----
    err_lo = np.zeros(ndim)
    err_hi = np.zeros(ndim)

    for k in range(ndim):
        err_lo[k], err_hi[k] = peak_pm(flat[:, k], peak_5d_rad[k], frac_side=0.68)

    err_lo = clip_zero(err_lo)
    err_hi = clip_zero(err_hi)

    err_lo_deg = err_lo.copy()
    err_hi_deg = err_hi.copy()
    for i in [0, 1, 2]:
        err_lo_deg[i] = np.rad2deg(err_lo[i])
        err_hi_deg[i] = np.rad2deg(err_hi[i])
    err_lo_deg = clip_zero(err_lo_deg)
    err_hi_deg = clip_zero(err_hi_deg)

    titles = [
        rf"$\theta_0\ (\mathrm{{deg}}) = {Theta_pk5d_deg:.3f}" + sup(err_hi_deg[0]) + sub(err_lo_deg[0]) + r"$",
        rf"$\phi_0\ (\mathrm{{deg}}) = {Phi_pk5d_deg:.3f}"   + sup(err_hi_deg[1]) + sub(err_lo_deg[1]) + r"$",
        rf"$i\ (\mathrm{{deg}}) = {Incl_pk5d_deg:.3f}"       + sup(err_hi_deg[2]) + sub(err_lo_deg[2]) + r"$",
        rf"$t_{{\rm s}}\ (\mathrm{{Myr}}) = {T_pk5d:.3f}"    + sup(err_hi[3])     + sub(err_lo[3])     + r"$",
        rf"$\omega = {Omega_pk5d:.3f}"                       + sup(err_hi[4])     + sub(err_lo[4])     + r"$",
    ]
    for k in range(ndim):
        axes[k, k].set_title(titles[k], fontsize=12)
    peak_plot = np.array([Theta_pk5d_deg, Phi_pk5d_deg, Incl_pk5d_deg, T_pk5d, Omega_pk5d], float)    
    err_lo_plot = err_lo.copy()
    err_hi_plot = err_hi.copy()
    for i in [0, 1, 2]:
        err_lo_plot[i] = np.rad2deg(err_lo_plot[i])
        err_hi_plot[i] = np.rad2deg(err_hi_plot[i])

    lo_plot = peak_plot - err_lo_plot
    hi_plot = peak_plot + err_hi_plot

    # --- 畫線：對角線兩條線 + 2D 十字線 ---
    for i in range(ndim):
        ax = axes[i, i]
        ax.axvline(lo_plot[i], ls="--", lw=1.2, color="k", alpha=0.9)
        ax.axvline(hi_plot[i], ls="--", lw=1.2, color="k", alpha=0.9)
    draw_2d_interval_lines(
        axes,
        centers=peak_plot,   # 這裡要用「畫圖單位」：deg, Myr, ...
        lo=lo_plot,
        hi=hi_plot,
        center_color="C0",     # 中心線顏色（你說要保留藍色）
        interval_color="k",    # 區間線顏色
        lw_main=1.3,
        lw_side=1.0,
        alpha_main=0.9,
        alpha_side=0.75,
        ls_main="-",
        ls_side="--",
    )
    fig.savefig(os.path.join(PLOT_DIR, "corner_mcmc_grid_map.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    cache.update({
        "mcmc_grid_used": True,
        "mcmc_grid_median_Theta": float(Theta_med),
        "mcmc_grid_median_Phi":   float(Phi_med),
        "mcmc_grid_median_Incl":  float(Incl_med),
        "mcmc_grid_median_T":     float(T_med),
        "mcmc_grid_median_Omega": float(Omega_med),
    })

    cache.update({
        "mcmc_grid_peak2d_Theta": float(Theta_pk5d_rad),
        "mcmc_grid_peak2d_Phi": float(Phi_pk5d_rad),
        "mcmc_grid_peak2d_Incl": float(Incl_pk5d_rad),
        "mcmc_grid_peak2d_T": float(T_pk5d_val),
        "mcmc_grid_peak2d_Omega": float(Omega_pk5d_val),
        "mcmc_grid_peak2d_Theta_deg": float(Theta_pk5d_deg),
        "mcmc_grid_peak2d_Phi_deg": float(Phi_pk5d_deg),
        "mcmc_grid_peak2d_Incl_deg": float(Incl_pk5d_deg),

        "mcmc_grid_peak2d_smooth": float(smooth_corner),
        "mcmc_grid_flat_samples": flat,               
        "burnin": int(burnin),
        "thin": int(thin),
    })

    np.savez(CACHE_PATH_MCMC_GRID, **cache)
    print(f"[cache] Saved MCMC grid results to {CACHE_PATH_MCMC_GRID}")   

# ============================================================
# 5. 殼層版 MCMC：使用 log_posterior_shell（最簡潔版本）
# ============================================================
def run_mcmc_shell():
    if not RUN_MCMC_SHELL:
        print("[MCMC_shell] Skipped (RUN_MCMC_SHELL = False)")
        return

    print("\n[MCMC_shell] start (distance-shell likelihood)")

    labels_5d = ["Theta zero", "Phi zero", "Inclination", "Time", "Omega"]

    
    ndim = 5
    labels_5d = ["Theta zero", "Phi zero", "Inclination", "Time", "Omega"]
    nwalkers, nsteps = 32, 5000  

    if cache.get("mcmc_grid_used", False):
        Theta_center = cache["mcmc_grid_median_Theta"]
        Phi_center   = cache["mcmc_grid_median_Phi"]
        Incl_center  = cache["mcmc_grid_median_Incl"]
        T_center     = cache["mcmc_grid_median_T"]
        Omega_center = cache["mcmc_grid_median_Omega"]
        print("[MCMC_shell] init center from MCMC_grid medians")
    else:
        Theta_center = cache["grid_best_Theta"]
        Phi_center   = cache["grid_best_Phi"]
        Incl_center  = cache["grid_best_Incl"]
        T_center     = cache["grid_best_T"]
        Omega_center = cache["grid_best_Omega"]
        print("[MCMC_shell] init center from grid_best")
    # Theta_center = cache["grid_best_Theta"]
    # Phi_center   = cache["grid_best_Phi"]
    # Incl_center  = cache["grid_best_Incl"]
    # T_center     = cache["grid_best_T"]
    # Omega_center = cache["grid_best_Omega"]
    # print("[MCMC_shell] init center from grid_best")

    center_vals = [Theta_center, Phi_center, Incl_center, T_center, Omega_center]
    
    # 跟 Per-emb-50 一樣的 sigma 設定（單位：rad, Myr, dimensionless）
    sigmas = [
        np.deg2rad(9.0),   # Theta
        np.deg2rad(18.0),  # Phi
        np.deg2rad(9.0),   # Incl
        0.05 * (parameter_prior_ranges["Time"][1] - parameter_prior_ranges["Time"][0]),
        0.05 * (parameter_prior_ranges["Omega"][1] - parameter_prior_ranges["Omega"][0]),
    ]

    p0 = np.zeros((nwalkers, ndim))
    for j, key in enumerate(labels_5d):
        lo, hi = parameter_prior_ranges[key]
        prop = center_vals[j] + sigmas[j] * np.random.randn(nwalkers)
        prop = np.clip(prop, lo, hi)
        p0[:, j] = prop
    
    down_factor = 5
    down_factor_v = 3

    def bin3d_sum(cube, fv, fy, fx):
        """
        將 (nv, ny, nx) 的 cube 以 (fv, fy, fx) 分箱，每個 block 做加總。
        若維度不能整除，會先裁掉尾端多出來的部分，確保 reshape 合法。
        """
        nv, ny, nx = cube.shape
        nv2 = (nv // fv) * fv
        ny2 = (ny // fy) * fy
        nx2 = (nx // fx) * fx

        if (nv2, ny2, nx2) != (nv, ny, nx):
            cube = cube[:nv2, :ny2, :nx2]

        # reshape 成 block，再沿 block 維度加總
        # (nv2, ny2, nx2)
        # -> (nv2/fv, fv, ny2/fy, fy, nx2/fx, fx)
        cube_b = cube.reshape(nv2//fv, fv, ny2//fy, fy, nx2//fx, fx)
        cube_b = cube_b.sum(axis=(1, 3, 5))  # sum over fv, fy, fx
        return cube_b, (nv2, ny2, nx2)

    # --- 改成「bin sum」而不是 slicing downsample ---
    shifted_cube_data_ds, trimmed_shape = bin3d_sum(new_cube_data, down_factor_v, down_factor, down_factor)
    print(f"[mask] shifted_cube_data binned-sum: {new_cube_data.shape} -> trimmed {trimmed_shape} -> {shifted_cube_data_ds.shape}")

    # 空間尺度：一個新 pixel = 原本 down_factor 個 pixel 的寬度
    dx_au_ds = dx_au * down_factor

    # --- header_ds 必須用 streamer cube 的 3D header ---
    header3d = fits.getheader("S_CrA_13CO_streamer_cube.fits")
    header_ds = header3d.copy()

    # ----- spatial axis (x/y): CDELT scales, CRPIX shifts to keep WCS center consistent -----
    for ax, fac in [(1, down_factor), (2, down_factor)]:
        k_cdelt = f"CDELT{ax}"
        k_crpix = f"CRPIX{ax}"
        if k_cdelt in header_ds:
            header_ds[k_cdelt] = float(header_ds[k_cdelt]) * fac
        if k_crpix in header_ds:
            header_ds[k_crpix] = (float(header_ds[k_crpix]) - 1.0) / fac + 1.0

    # ----- spectral axis (v): 用「每個 bin 的中心 channel」當作新 channel 的世界座標 -----
    crval3 = float(header3d["CRVAL3"])
    cdelt3 = float(header3d["CDELT3"])
    crpix3 = float(header3d.get("CRPIX3", 1.0))

    # 注意：我們 bin 的是 new_cube_data（可能被裁切過），所以 nv_full 用 trimmed_shape[0]
    nv_full_trim = int(trimmed_shape[0])
    i_full = np.arange(nv_full_trim, dtype=float)
    v_full = crval3 + (i_full + 1.0 - crpix3) * cdelt3  # in header3d CUNIT3

    # 每 fv 個 channel 為一組：取「中心那個」(對 fv=3 就是第 2 個，也就是 offset=1)
    fv = down_factor_v
    center_offset = fv // 2  # fv=3 -> 1
    n_bin = nv_full_trim // fv
    v_full_reshape = v_full.reshape(n_bin, fv)
    v_bin_center = v_full_reshape[:, center_offset]

    header_ds["CRPIX3"] = 1.0
    header_ds["CRVAL3"] = float(v_bin_center[0])
    header_ds["CDELT3"] = float(cdelt3) * fv
    
    max_dist_value = 50
    print("min/max data =", np.nanmin(shifted_cube_data_ds), np.nanmax(shifted_cube_data_ds))
    print("frac>0 =", np.mean(shifted_cube_data_ds > 0))
    print("frac!=0 =", np.mean(shifted_cube_data_ds != 0))
    DATA_BBOX = pss.compute_data_bbox(
            shifted_cube_data_ds,
            max_r=max_dist_value,
            extra_margin=5,
    )
    print("[bbox] DATA_BBOX =", DATA_BBOX)
    E_center, Neff = pss.shell_error_from_cube(
        shifted_cube_data_ds,
        center_vals[0], center_vals[1], center_vals[2], center_vals[3], center_vals[4],
        pa_rad, dx_au_ds, header_ds, Local_Standard_Velocity,
        max_dist_value,
        M_star, radius_in_au, radius_out_au,
        scale, log_power,
        DATA_BBOX,
    )
    print("[MCMC_shell] reference shell error E_center =", E_center)
    print("[MCMC_shell] reference shell Neff =", Neff)
    SIGMA_LIKE_SHELL = 2 * E_center
    log_args = (
        shifted_cube_data_ds,
        parameter_prior_ranges,
        pa_rad,
        dx_au_ds,
        header_ds,
        Local_Standard_Velocity,
        max_dist_value,
        M_star,
        radius_in_au,
        radius_out_au,
        scale,
        log_power,
        DATA_BBOX,
        SIGMA_LIKE_SHELL,
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

    try:
        tau = sampler.get_autocorr_time(quiet=True)
        if (not np.all(np.isfinite(tau))) or (np.any(tau <= 0)):
            raise RuntimeError(f"tau invalid: {tau}")
        burnin = int(2 * np.nanmax(tau))
        thin   = max(1, int(0.1 * np.nanmin(tau)))
        print(f"[MCMC_shell] tau: {tau}, burnin={burnin}, thin={thin}")
    except Exception as e:
        print("[MCMC_shell] tau 估計失敗，用預設。", e)
        burnin, thin = 50, 25

    chain = sampler.get_chain()
    print("chain shape:", chain.shape)  # (nsteps, nwalkers, ndim)
    print("mean acceptance:", np.mean(sampler.acceptance_fraction))

    lp_chain = sampler.get_log_prob()
    print("non-finite log_prob fraction =", np.mean(~np.isfinite(lp_chain)))
    print(parameter_prior_ranges)
    flat    = sampler.get_chain(discard=burnin, thin=thin, flat=True)
    lp_flat = sampler.get_log_prob(discard=burnin, thin=thin, flat=True)

    q16, q50, q84 = np.percentile(flat, [16, 50, 84], axis=0)
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

    print("\n[MCMC_shell] 1D posterior shape:")
    for i, name in enumerate(labels_5d):
        summarize_1d_posterior(flat[:, i], name)

    # corner plot（角度轉度）
    samples_plot = flat.copy()
    for idx in [0, 1, 2]:
        samples_plot[:, idx] = np.rad2deg(samples_plot[:, idx])
    labels_plot = [
    r"$\theta$ (deg)",
    r"$\phi_0$ (deg)",
    r"$i$ (deg)",
    r"$t_{\rm s}$ (Myr)",
    r"$\omega$",
    ]
    q16p, q50p, q84p = np.percentile(samples_plot, [16, 50, 84], axis=0)
    
    smooth_corner = 1.0   # must match corner.corner(..., smooth=1.0)
    bins_corner   = 50    # choose consistent bins; can tune to your sample size

    pair_peaks = {}  # (i,j) -> (peak_i, peak_j)
    acc = [[] for _ in range(ndim)]  # acc[k] collects peak estimates for param k

    ranges = []
    for i in range(len(labels_plot)):
        lo, md, hi = q16p[i], q50p[i], q84p[i]
        width = hi - lo if hi > lo else 1e-3
        ranges.append((md - 1.2*width, md + 1.2*width))

    peak_5d_plot = kde_5d_peak(
        samples_plot,
        ranges=ranges,
        n_candidates=20000,     # 可調：1e4~5e4 常用
        bw_method="scott",      # 可調："silverman" 或 float
        seed=0
    )

    Theta_pk5d_deg, Phi_pk5d_deg, Incl_pk5d_deg, T_pk5d, Omega_pk5d = peak_5d_plot

    peak_5d_rad = peak_5d_plot.copy()
    for k in [0, 1, 2]:
        peak_5d_rad[k] = np.deg2rad(peak_5d_plot[k])

    Theta_pk5d_rad, Phi_pk5d_rad, Incl_pk5d_rad, T_pk5d_val, Omega_pk5d_val = peak_5d_rad
    print("\n[MCMC_shell] 5D KDE-smoothed peak (global peak in smoothed 5D posterior):")
    print(f"Theta = {Theta_pk5d_deg:.3f} deg")
    print(f"Phi   = {Phi_pk5d_deg:.3f} deg")
    print(f"Incl  = {Incl_pk5d_deg:.3f} deg")
    print(f"T     = {T_pk5d:.6f} Myr")
    print(f"Omega = {Omega_pk5d:.4f}")

    fig = corner.corner(samples_plot,
                        labels=labels_plot,
                        range=ranges,
                        show_titles=True,
                        plot_contours=True,
                        title_fmt=".3f",
                        quantiles=[0.16, 0.5, 0.84],
                        truths=[np.rad2deg(Theta_med), np.rad2deg(Phi_med), np.rad2deg(Incl_med), T_med, Omega_med],
                        smooth=1.0)
    axes = np.array(fig.axes).reshape((ndim, ndim))

    # 用「畫圖單位」的 median 16/50/84：你前面已經算過 q16p, q50p, q84p
    cent_med = q50p
    lo_med   = q16p
    hi_med   = q84p

    draw_2d_interval_lines(
        axes,
        centers=cent_med,
        lo=lo_med,
        hi=hi_med,
        center_color="C0",      # median 中心線顏色
        interval_color="k",     # median 區間線顏色（想更清楚可改成 "w"）
        lw_main=1.3,
        lw_side=1.0,
        alpha_main=0.9,
        alpha_side=0.75,
        ls_main="-",
        ls_side="--",
    )
    fig.savefig(os.path.join(PLOT_DIR, "corner_mcmc_shell_median.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    
    fig = corner.corner(
        samples_plot,
        labels=labels_plot,
        range=ranges,
        show_titles=False,
        plot_contours=True,
        title_fmt=".3f",
        truths=[Theta_pk5d_deg, Phi_pk5d_deg, Incl_pk5d_deg, T_pk5d, Omega_pk5d],
        smooth=smooth_corner,
    )
    axes = np.array(fig.axes).reshape((ndim, ndim))

    def clip_zero(x, atol=5e-13):
        x = np.asarray(x, dtype=float)
        x[np.isclose(x, 0.0, atol=atol)] = 0.0
        return x

    def fmt_pm(x, nd=3):
        if not np.isfinite(x):
            return "?"
        return f"{x:.{nd}f}"

    def sup(x, nd=3):
        return rf"^{{+{fmt_pm(x, nd)}}}"

    def sub(x, nd=3):
        if not np.isfinite(x):
            return rf"_{{{fmt_pm(x, nd)}}}"
        if x == 0.0:
            return rf"_{{{fmt_pm(x, nd)}}}"
        return rf"_{{-{fmt_pm(x, nd)}}}"

    # ---- peak-centered ±34% each side ----
    err_lo = np.zeros(ndim)
    err_hi = np.zeros(ndim)

    for k in range(ndim):
        err_lo[k], err_hi[k] = peak_pm(flat[:, k], peak_5d_rad[k], frac_side=0.68)

    err_lo = clip_zero(err_lo)
    err_hi = clip_zero(err_hi)

    err_lo_deg = err_lo.copy()
    err_hi_deg = err_hi.copy()
    for i in [0, 1, 2]:
        err_lo_deg[i] = np.rad2deg(err_lo[i])
        err_hi_deg[i] = np.rad2deg(err_hi[i])
    err_lo_deg = clip_zero(err_lo_deg)
    err_hi_deg = clip_zero(err_hi_deg)

    titles = [
        rf"$\theta_0\ (\mathrm{{deg}}) = {Theta_pk5d_deg:.3f}" + sup(err_hi_deg[0]) + sub(err_lo_deg[0]) + r"$",
        rf"$\phi_0\ (\mathrm{{deg}}) = {Phi_pk5d_deg:.3f}"   + sup(err_hi_deg[1]) + sub(err_lo_deg[1]) + r"$",
        rf"$i\ (\mathrm{{deg}}) = {Incl_pk5d_deg:.3f}"       + sup(err_hi_deg[2]) + sub(err_lo_deg[2]) + r"$",
        rf"$t_{{\rm s}}\ (\mathrm{{Myr}}) = {T_pk5d:.3f}"    + sup(err_hi[3])     + sub(err_lo[3])     + r"$",
        rf"$\omega = {Omega_pk5d:.3f}"                       + sup(err_hi[4])     + sub(err_lo[4])     + r"$",
    ]
    for k in range(ndim):
        axes[k, k].set_title(titles[k], fontsize=12)
    peak_plot = np.array([Theta_pk5d_deg, Phi_pk5d_deg, Incl_pk5d_deg, T_pk5d, Omega_pk5d], float)    
    err_lo_plot = err_lo.copy()
    err_hi_plot = err_hi.copy()
    for i in [0, 1, 2]:
        err_lo_plot[i] = np.rad2deg(err_lo_plot[i])
        err_hi_plot[i] = np.rad2deg(err_hi_plot[i])

    lo_plot = peak_plot - err_lo_plot
    hi_plot = peak_plot + err_hi_plot

    # --- 畫線：對角線兩條線 + 2D 十字線 ---
    for i in range(ndim):
        ax = axes[i, i]
        ax.axvline(lo_plot[i], ls="--", lw=1.2, color="k", alpha=0.9)
        ax.axvline(hi_plot[i], ls="--", lw=1.2, color="k", alpha=0.9)
    draw_2d_interval_lines(
        axes,
        centers=peak_plot,   # 這裡要用「畫圖單位」：deg, Myr, ...
        lo=lo_plot,
        hi=hi_plot,
        center_color="C0",     # 中心線顏色（你說要保留藍色）
        interval_color="k",    # 區間線顏色
        lw_main=1.3,
        lw_side=1.0,
        alpha_main=0.9,
        alpha_side=0.75,
        ls_main="-",
        ls_side="--",
    )
    fig.savefig(os.path.join(PLOT_DIR, "corner_mcmc_shell_map.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    
    # ---- 寫入 SHELL cache：保留 median，同時改存 peak2d（取代 map）----
    cache.update({
        "mcmc_shell_used": True,

        # --- Median (rad) ---
        "mcmc_shell_median_Theta": float(Theta_med),
        "mcmc_shell_median_Phi":   float(Phi_med),
        "mcmc_shell_median_Incl":  float(Incl_med),
        "mcmc_shell_median_T":     float(T_med),
        "mcmc_shell_median_Omega": float(Omega_med),
    })
    
    cache.update({
        # --- Peak2D (rad) ---
        "mcmc_shell_peak2d_Theta": float(Theta_pk5d_rad),
        "mcmc_shell_peak2d_Phi":   float(Phi_pk5d_rad),
        "mcmc_shell_peak2d_Incl":  float(Incl_pk5d_rad),
        "mcmc_shell_peak2d_T":     float(T_pk5d_val),
        "mcmc_shell_peak2d_Omega": float(Omega_pk5d_val),
        # --- Peak2D (deg, optional but recommended) ---
        "mcmc_shell_peak2d_Theta_deg": float(Theta_pk5d_deg),
        "mcmc_shell_peak2d_Phi_deg":   float(Phi_pk5d_deg),
        "mcmc_shell_peak2d_Incl_deg":  float(Incl_pk5d_deg),

        # --- Diagnostics (optional but nice to keep consistent with shell) ---
        "mcmc_shell_peak2d_bins":   int(bins_corner),
        "mcmc_shell_peak2d_smooth": float(smooth_corner),
        "mcmc_shell_peak2d_pair_peaks": np.array(
            [(i, j, pair_peaks[(i, j)][0], pair_peaks[(i, j)][1])
            for (i, j) in sorted(pair_peaks.keys())],
            dtype=float
        ),

        # --- Samples ---
        "mcmc_shell_flat_samples": flat,
        "burnin": int(burnin),
        "thin":   int(thin),
    })

    np.savez(CACHE_PATH_MCMC_SHELL, **cache)
    print(f"[cache] Saved MCMC shell results to {CACHE_PATH_MCMC_SHELL}")
    # ---- 寫入 FINAL cache：median + peak2d（取代 map）----
    cache.update({
        "best_Theta_median": float(Theta_med),
        "best_Phi_median":   float(Phi_med),
        "best_Incl_median":  float(Incl_med),
        "best_T_median":     float(T_med),
        "best_Omega_median": float(Omega_med),
    })

    cache.update({
        # 用 peak2d 當作你要的「best」
        "best_Theta_peak2d": float(Theta_pk5d_rad),
        "best_Phi_peak2d":   float(Phi_pk5d_rad),
        "best_Incl_peak2d":  float(Incl_pk5d_rad),
        "best_T_peak2d":     float(T_pk5d_val),
        "best_Omega_peak2d": float(Omega_pk5d_val),
        # 角度的 deg 版（方便畫圖/報告直接讀）
        "best_Theta_peak2d_deg": float(Theta_pk5d_deg),
        "best_Phi_peak2d_deg":   float(Phi_pk5d_deg),
        "best_Incl_peak2d_deg":  float(Incl_pk5d_deg),
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

        # ---- 1) best-fit params ----
        Theta_best, Phi_best, Incl_best, T_best, Omega_best = _extract_params_from_cache(
            c, USE_CACHE_SOURCE
        )
        Theta_best_deg = np.rad2deg(Theta_best)
        Phi_best_deg   = np.rad2deg(Phi_best)
        Incl_best_deg  = np.rad2deg(Incl_best)

        # ---- 2) load streamer moment maps + header (prefer streamer header) ----
        try:
            str_mom0 = fits.getdata("S_CrA_13CO_streamer_mom0.fits")
            str_mom1 = fits.getdata("S_CrA_13CO_streamer_mom1.fits")
            header   = fits.getheader("S_CrA_13CO_streamer_mom1.fits")
        except Exception as e:
            print(f"[overlay] Failed to load streamer moment maps from FITS: {e}")
            return

        # ---- 3) choose center: ALWAYS use new_center (A-mode) ----
        if "new_center_yx" in c:
            new_center = tuple(np.array(c["new_center_yx"]).astype(int))
        else:
            # fallback: if cache doesn't have it, use header CRPIX as last resort
            new_center = (int(header["CRPIX2"]), int(header["CRPIX1"]))
            print("[overlay][WARN] cache missing new_center_yx → fallback to header CRPIX")

        # (optional) for debug
        print(f"[overlay] using new_center(y,x) = {new_center}")

        # ---- 4) scales ----
        dx_arcsec = abs(header["CDELT1"]) * 3600.0
        dx_au     = dx_arcsec * distance_pc

        # ---- 5) load streamer cube for z–v (optional) ----
        new_cube_data = None
        try:
            new_cube_data = fits.getdata("S_CrA_13CO_streamer_cube.fits")
            print("[overlay] Loaded streamer cube for z–v diagram")
        except Exception as e:
            print(f"[overlay] No streamer cube FITS ({e}), z–v diagram will be skipped.")

        # ---- 6) centroids from cache → pixel (A-mode center = new_center) ----
        cen_x_pix = cen_z_pix = cen_v_LS_km = None
        if ("streamercom_x_AU" in c) and ("streamercom_z_AU" in c):
            sx_AU = c["streamercom_x_AU"]
            sz_AU = c["streamercom_z_AU"]

            x_rot = sx_AU / dx_au
            z_rot = sz_AU / dx_au

            # model frame (x_rot,z_rot) -> image pixel using new_center
            cen_x_pix = x_rot * np.cos(pa_rad) - z_rot * np.sin(pa_rad) + new_center[1]
            cen_z_pix = x_rot * np.sin(pa_rad) + z_rot * np.cos(pa_rad) + new_center[0]

            if "streamercom_v_LS_km" in c:
                cen_v_LS_km = c["streamercom_v_LS_km"]

        # ---- 7) z–v plot (use header + new_center convention) ----
        if new_cube_data is not None:
            # 如果你想疊 centroids，就把 cache 的 z_means 傳進去
            z_means_pix = c["z_means"] if "z_means" in c else None
            plot_z_v_diagram_from_cube(
                theta_deg=Theta_best_deg,
                phi_deg=Phi_best_deg,
                inc_deg=Incl_best_deg,
                T_Myr=float(T_best),
                omega=float(Omega_best),
                new_cube_data=new_cube_data,
                header=header,          # streamer header（CRPIX 應該也等於 new_center）
                pa_rad=pa_rad,
                dx_au=dx_au,
                z_means_pix=z_means_pix,
                streamer_v_LS_km=cen_v_LS_km,
                outname=f"SCrA_z_v_data_overlay_best_{pa_deg}.png",
                label="S CrA $^{13}$CO z–v (best-fit)",
            )

        # ---- 8) moment-1 overlay ----
        plot_streamer_on_mom1(
            Theta_best, Phi_best, Incl_best,
            float(T_best), float(Omega_best),
            header, pa_rad, dx_au, new_center,   # <<< use new_center
            str_mom1,
            label="S CrA $^{13}$CO moment1 (best-fit)",
            outname=f"SCrA_13CO_model_vs_mom1_overlay_best_{pa_deg}.png",
            cen_x_pix=cen_x_pix,
            cen_z_pix=cen_z_pix,
            cen_v_LS_km=cen_v_LS_km,
        )

        # ---- 9) moment-0 overlay ----
        plot_streamer_on_mom0(
            Theta_best, Phi_best, Incl_best,
            float(T_best), float(Omega_best),
            header, pa_rad, dx_au, new_center,   # <<< use new_center
            str_mom0,
            label="S CrA $^{13}$CO moment0 (best-fit)",
            outname=f"SCrA_13CO_model_vs_mom0_overlay_best_{pa_deg}.png",
            cen_x_pix=cen_x_pix,
            cen_z_pix=cen_z_pix,
            cen_v_LS_km=cen_v_LS_km,
        )

        print("[overlay] Generated best-fit z–v, moment0, and moment1 overlay plots from cache.")

    except Exception as e:
        print(f"[overlay] Failed to generate overlay from cache: {e}")
        
# ------------------- MAIN FUNCTION -------------------
def main():
    # ---- NEW: corner-only mode ----
    if REBUILD_CORNER_ONLY:
        print("[corner-only] Rebuild corner plots from cache, no rerun.")
        for w in REBUILD_WHICH:
            rebuild_corner_from_cache(which=w, out_tag="cacheonly")
        sys.exit(0)

    if RUN_FROM_CACHE_ONLY:
        run_quick_mode_scra()
    else:
        prepare_data()
        run_grid()
    if RUN_MCMC_GRID:
        run_mcmc_grid()
    if RUN_MCMC_SHELL:
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
