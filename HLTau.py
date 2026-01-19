# ============================================================
# HL Tau streamer fitting script（整理版）
#
# 區塊結構：
#   1) 參數宣告 / Imports / 開關
#   2) 定義函數 (helpers & MCMC moves)
#   3) 資料前處理：mask streamer、平移至中心、抽質心
#   4) Grid fitting：用 29 點 error_function 找初始解
#   5) 用 grid 結果自動決定 MCMC 先驗範圍
#   6) MCMC_grid   ：29 點 + fast likelihood（選配）
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

# 後驗分析 / 工具
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

# 專案本地模組
import PSSpy as pss
from pss_grid_search import run_grid_search, compute_priors_from_grid

# --- 基本天文參數（HL Tau） ---
Local_Standard_Velocity = 7.14  # km/s (Gupta 2024)
pa_deg = 20.0
pa_rad = np.deg2rad(pa_deg)
distance_pc = 147.0
M_SUN_KG = 1.98847e30
radius_ref_au = 300
M_star = 2.1

scale = "log"
log_power = 1.5

radius_in_au, radius_out_au = 3e1, 3.6e2
# 資料與輸出
cube_fname = "HLTau_HCOp32.fits"
CACHE_DIR = "HLTau_results/cache"
os.makedirs(CACHE_DIR, exist_ok=True)
PLOT_DIR = "HLTau_results/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

CACHE_PATH_GRID = os.path.join(CACHE_DIR, "HLTau_grid_results.npz")
CACHE_PATH_MCMC_GRID = os.path.join(CACHE_DIR, "HLTau_mcmc_grid_results.npz")
CACHE_PATH_MCMC_SHELL = os.path.join(CACHE_DIR, "HLTau_mcmc_shell_results.npz")
CACHE_PATH_FINAL = os.path.join(CACHE_DIR, "HLTau_fit_results_final.npz")

USE_CACHE_SOURCE = "mcmc_grid"
sample_from = "Median"

# ---------- corner 重畫模式 ----------
REBUILD_CORNER_ONLY = False   # True: 不跑資料、不跑MCMC，只從 cache 重畫 corner
REBUILD_WHICH = ("mcmc_grid", "mcmc_shell")  # 想重畫哪些：可改成只留其中一個

# --- 分析開關 ---
# RUN_GRID = True               # 5D grid search 找初始解
# RUN_MCMC_GRID = True          # 29 個質心點 fast likelihood
# RUN_MCMC_SHELL = True         # distance_cube MCMC
# RUN_FROM_CACHE_ONLY = False   # True: 僅讀 cache 畫圖，完全不重跑

RUN_GRID = False               # 5D grid search 找初始解
RUN_MCMC_GRID = False          # 29 個質心點 fast likelihood
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
    "target": "HLTau",
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
            (emcee.moves.DEMove(),           0.3),   # 差分進化，跳躍性大，探索性強
            (emcee.moves.DESnookerMove(),    0.3),   # 反射式 move，有助於跳出局部極值
        ]

# --- HL Tau streamer masking/centroid helper ---
def build_streamer_masked_cube_hltau(cube, header, rms_channel):
    """
    HL Tau 專用：
    只使用 grow_region 從 cube 中找出 streamer 區域，回傳：

      im_center       : (y, x)，以 CRPIX2, CRPIX1 為中心
      masked_cube     : 已套用 streamer mask 的 SpectralCube
      new_cube_data   : numpy.ndarray，將 masked_cube 填成 NaN 之後的資料
    """
    # 以 FITS header 定義影像中心
    im_center = (int(round(header["CRPIX2"] - 1.0)),
                int(round(header["CRPIX1"] - 1.0)))
    cube_data = cube.filled_data[:].value

    init_points = [
        (110, im_center[0], im_center[1]), # (274 - 164) origin channel
        (110, 220, 196),
        (110, 226, 207),
        (110, 227, 226),
        (110, 227, 251),
        (110, 207, 267),
        (110, 185, 277),
        (110, 161, 289),
        (110, 139, 298),
        (110, 94,  305),
    ]

    stream_mask = pss.grow_region(
        cube_data,
        init_points,
        rms_channel,
        sigma_thresh=4.0,
        max_iter=1000,
    )

    # 套用 mask 到 SpectralCube
    masked_cube = cube.with_mask(stream_mask)
    new_cube_data = masked_cube.with_fill_value(0).filled_data[:].value
    return im_center, masked_cube, new_cube_data

def extract_streamer_centroids(new_cube_data, header, pa_rad, dx_au, spec_kms):
    """
    HL Tau streamer centroid extraction (3D cube version, same style as Per-emb-50).

    回傳:
      streamer_x_AU, streamer_z_AU, streamer_v_LS_km
      x_array, z_array, v_array, weights_array : 每個 shell 中 voxel 的紀錄（list of arrays）
      x_means_ref, z_means_ref, v_means        : image frame 下的質心 (pixel, channel)
    """
    cube_shape = new_cube_data.shape   # (nv, ny, nx)
    im_center = (int(round(header["CRPIX2"] - 1.0)),
                int(round(header["CRPIX1"] - 1.0)))
    # 建立 voxel 座標
    v, z, x = np.indices(cube_shape)
    x_rel = x - im_center[1]
    z_rel = z - im_center[0]
    r, theta = pss.spherical_coords(x_rel, z_rel)

    # ------------------------------------------------------------
    # 手動指定 streamer 大略通過的點（跟你 2D 版一致）
    # ------------------------------------------------------------
    stream_points_abs = np.array([
        [201, 201],
        [196, 220],
        [207, 226],
        [226, 227],
        [251, 227],
        [267, 207],
        [277, 185],
        [289, 161],
        [298, 139],
        [305,  94],
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

    # ------------------------------------------------------------
    # Step 1: 幾何中心（方向權重 + 半徑 shell）
    # ------------------------------------------------------------
    N = 25
    pars = np.linspace(10, 155, N + 1)   # HL Tau 用的半徑範圍（pixel）

    x_means = np.zeros(N)
    z_means = np.zeros(N)
    v_means = np.zeros(N)   # 先暫存 channel index
    xzstd   = np.zeros(N)

    x_array_list = []
    z_array_list = []
    v_array_list = []
    weights_list = []

    for i in tqdm(range(N), desc="[HLTau] centroid step1 (pos.)", ncols=80, leave=False):
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

    for i in tqdm(range(N), desc="[HLTau] centroid step2 (vel.)", ncols=80, leave=False):
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

    streamer_v_km = np.full_like(v_means, np.nan, dtype=float)
    valid_v = np.isfinite(v_means)

    streamer_v_km[valid_v] = spec_kms[v_means[valid_v].astype(int)]
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

def plot_r_theta_weights_from_output(x_array, z_array, weights_array, outname):
    """
    用 extract_streamer_centroids 回傳的
    x_array, z_array, weights_array
    畫出 (r, theta) 的權重分布。
    """
    all_r = []
    all_theta = []
    all_w = []

    N = len(x_array)  # 應該是 29 個 bin
    for i in range(N):
        x_bin = x_array[i]
        z_bin = z_array[i]
        w_bin = weights_array[i]

        if x_bin.size == 0:
            continue

        # 用你原本的 spherical_coords 算每個 pixel 的 (r, theta)
        r_bin, theta_bin = pss.spherical_coords(x_bin, z_bin)

        all_r.append(r_bin)
        all_theta.append(theta_bin)
        all_w.append(w_bin)

    # 串成一條長向量

    all_r = np.concatenate(all_r)
    all_theta = np.rad2deg(np.concatenate(all_theta))
    all_w = np.concatenate(all_w)
    mask = all_w > 0
    fig, ax = plt.subplots(figsize=(6, 4))
    sc = ax.scatter(all_r[mask], all_theta[mask], c=all_w[mask], s=5, cmap="inferno")
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
                          header, pa_rad, dx_au, im_center,
                          mom0, label, outname,
                          vmin=None, vmax=None,
                          radius_in_au=radius_in_au,
                          radius_out_au=radius_out_au,
                          scale='log', log_power=log_power,
                          cen_x_pix=None, cen_z_pix=None, cen_v_LS_km=None):

    # --- 基本尺寸 & 像素刻度 ---
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
    vmin = -1.6
    vmax = 15

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

    # 黑色外框線（提升可見度）
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
    ax.scatter(0, 0, c="w", s=60, marker="+", zorder=6)

    ax.set_xlabel("RA Offset (arcsec)")
    ax.set_ylabel("Dec Offset (arcsec)")
    ax.set_title(label)
    ax.set_xlim(2.2, -2.2)
    ax.set_ylim(-2.2, 2.2)
    
    # 主圖保持方形比例：RA/DEC 1:1
    ax.set_aspect("equal", adjustable="box")
    # --- 比例尺與方向標示 ---
    # 定義比例尺位置（以 arcsec 為單位）
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    text_pos_x = x1 + 0.25 * (x0 - x1)
    text_pos_y = y0 + 0.15 * (y1 - y0)
    scale_length = 147  # AU

    # 將 3000 AU 轉成 arcsec
    scale_length_arcsec = scale_length / (distance_pc)  # 1" ≈ 1 AU / distance(pc)

    # 定義比例尺線段 (RA offset 軸)
    scale_range_x = [text_pos_x, text_pos_x - scale_length_arcsec]
    scale_range_y = [text_pos_y - 0.2, text_pos_y - 0.2]

    # 繪製比例尺與文字
    ax.plot(scale_range_x, scale_range_y, color='k', lw=3, zorder=10)
    ax.text(
        text_pos_x - scale_length_arcsec / 2,
        text_pos_y - 0.5,
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
                          vmin=None, vmax=None,
                          radius_in_au=radius_in_au,
                          radius_out_au=radius_out_au,
                          scale='log', log_power=log_power,
                          cen_x_pix=None, cen_z_pix=None, cen_v_LS_km=None):

    # --- 基本尺寸 & 像素刻度 ---
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
    vmin = -1.6
    vmax = 15

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
    ax.set_xlim(2.2, -2.2)
    ax.set_ylim(-2.2, 2.2)
    
    # 主圖保持方形比例：RA/DEC 1:1
    ax.set_aspect("equal", adjustable="box")
    # --- 比例尺與方向標示 ---
    # 定義比例尺位置（以 arcsec 為單位）
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    text_pos_x = x1 + 0.25 * (x0 - x1)
    text_pos_y = y0 + 0.15 * (y1 - y0)
    scale_length = 147  # AU

    # 將 3000 AU 轉成 arcsec
    scale_length_arcsec = scale_length / (distance_pc)  # 1" ≈ 1 AU / distance(pc)

    # 定義比例尺線段 (RA offset 軸)
    scale_range_x = [text_pos_x, text_pos_x - scale_length_arcsec]
    scale_range_y = [text_pos_y - 0.2, text_pos_y - 0.2]

    # 繪製比例尺與文字
    ax.plot(scale_range_x, scale_range_y, color='k', lw=3, zorder=10)
    ax.text(
        text_pos_x - scale_length_arcsec / 2,
        text_pos_y - 0.5,
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
                               label="HL Tau z-v with data"):
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
        label="Model",
        zorder=3,
    )

    # # ---------- 6) 疊上質心 ----------
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
    ax.set_ylim(-1.6, 15)
    ax.set_xlim(300, -300)
    leg = ax.legend(frameon=False, fontsize=9)
    for txt in leg.get_texts():
        txt.set_color("white")
    ax.grid(alpha=0.2)

    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, outname), dpi=200)
    plt.close(fig)
    print(f"[z–v] Saved {outname}")
    
def run_quick_mode_hltau():
    """
    Quick Mode for HL Tau:
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
        rms_channel = 2.509251078668e-3
        
        #mcmc_grid peak
        # Theta_best_deg = 67.591 # 67.591, 70.225
        # Phi_best_deg   = -159.246 # -159.246, -164.650
        # Incl_best_deg  = -87.712 # -87.712, -83.926
        # Theta_best     = np.deg2rad(Theta_best_deg) 
        # Phi_best       = np.deg2rad(Phi_best_deg)
        # Incl_best      = np.deg2rad(Incl_best_deg)
        # T_best         = 0.736 # 0.736, 0.791
        # Omega_best     = 0.092 # 0.092, 0.092
        
        # shell peak
        Theta_best_deg = 70.225 # 67.591, 70.225
        Phi_best_deg   = -164.650 # -159.246, -164.650
        Incl_best_deg  = -83.926 # -87.712, -83.926
        Theta_best     = np.deg2rad(Theta_best_deg) 
        Phi_best       = np.deg2rad(Phi_best_deg)
        Incl_best      = np.deg2rad(Incl_best_deg)
        T_best         = 0.791 # 0.736, 0.791
        Omega_best     = 0.092 # 0.092, 0.092
        
        # 2) 讀取 HL Tau streamer 專用 moment map
        try:
            str_mom0 = fits.getdata("HL_Tau_HCOp32_streamer_mom0.fits")
            str_mom1 = fits.getdata("HL_Tau_HCOp32_streamer_mom1.fits")
            mom0 = fits.getdata("HL_Tau_HCOp32_mom0.fits")
            mom1 = np.where(fits.getdata("HL_Tau_HCOp32_mom0.fits") > 3 * rms_channel, fits.getdata("HL_Tau_HCOp32_mom1.fits"), np.nan)
            header = fits.getheader(cube_fname)  # overlay 用
            header_kms = fits.getheader("HL_Tau_HCOp32_streamer_cube.fits")  # z–v 用
        except Exception:
            cube = SpectralCube.read(cube_fname)
            header = fits.getheader(cube_fname)
            vrange = [2.4926, 14.8636] * u.km/u.s
            subcube = cube.spectral_slab(vrange[0], vrange[1])
            str_mom0 = subcube.moment(order=0).value
            str_mom1 = subcube.moment(order=1).value

        # 基本轉換
        im_center = (int(round(header["CRPIX2"] - 1.0)),
                    int(round(header["CRPIX1"] - 1.0)))
        dx_arcsec = abs(header["CDELT1"]) * 3600.0
        dx_au     = dx_arcsec * distance_pc  # 160 pc for HLTau

        # 3) 盡量讀取 streamer cube（若沒有，z–v 圖會自動跳過）
        new_cube_data = None
        try:
            new_cube_data = fits.getdata("HL_Tau_HCOp32_streamer_cube.fits")
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
            header=header_kms,
            pa_rad=pa_rad,
            dx_au=dx_au,
            z_means_pix=z_means,
            streamer_v_LS_km=cen_v_LS,
            outname="HLTau_z_v_data_overlay.png",
            label="HL Tau HCO$^{+}$ z–v"
        )

        plot_streamer_on_mom1(
            Theta_best, Phi_best, Incl_best,
            float(T_best), float(Omega_best),
            header, pa_rad, dx_au, im_center,
            str_mom1,
            label="HL Tau HCO$^{+}$",
            outname="HLTau_mom1_cacheonly.png",
            cen_x_pix=cen_x_pix,
            cen_z_pix=cen_z_pix,
            cen_v_LS_km=cen_v_LS,
            vmin=np.nanmin(mom1),
            vmax=np.nanmax(mom1)
        )

        plot_streamer_on_mom0(
            Theta_best, Phi_best, Incl_best,
            float(T_best), float(Omega_best),
            header, pa_rad, dx_au, im_center,
            str_mom0,
            label="HL Tau HCO$^{+}$",
            outname="HLTau_mom0_cacheonly.png",
            cen_x_pix=cen_x_pix,
            cen_z_pix=cen_z_pix,
            cen_v_LS_km=cen_v_LS,
        )

        plot_r_theta_weights_from_output(x_array, z_array, weights_array, outname="HLTau_weights_cacheonly.png")

        # 8) Print physical values
        r_ref_AU = 200 * T_best * 1e6 * spc.year / spc.astronomical_unit  # radius_ref_au=200

        M_0 = M_star * M_SUN_KG * spc.G / (200.0**3 * T_best * 1e6 * spc.year)
        M_dot = M_star / (T_best * 1e6)

        print("\n==================== Parameters (HL Tau) ====================")
        print(f"Theta        = {Theta_best_deg:.3f} deg")
        print(f"Phi          = {Phi_best_deg:.3f} deg")
        print(f"Inclination  = {Incl_best_deg:.3f} deg")
        print(f"Time (T_Myr) = {T_best:.6f} Myr")
        print(f"Omega        = {Omega_best:.4f}")
        print(f"r_ref        = {r_ref_AU:.2f} AU")
        print(f"M_0          = {M_0:.3e}")
        print(f"Mdot         = {M_dot:.3e} M_sun/yr")
        print("============================================================")
        print("[Quick Mode] Done.")
        # T_factor_range=(5.4, 245.0)        
        # Omega_ref = pss.Omega_ref(radius_ref_au, M_star)
        # P_half_Myr = np.pi / Omega_ref / 1e6 / spc.year
        # T_range = [T_factor_range[0]*P_half_Myr, T_factor_range[1]*P_half_Myr]
        # T_grid     = np.logspace(np.log10(T_range[0]), np.log10(T_range[1]), 10)

        # M_0 = M_star * M_SUN_KG * spc.G / (280.0**3 * T_grid[5] * 1e6 * spc.year)
        # M_0_max = M_star * M_SUN_KG * spc.G / (280.0**3 * T_grid[0] * 1e6 * spc.year)
        # M_0_min = M_star * M_SUN_KG * spc.G / (280.0**3 * T_grid[9] * 1e6 * spc.year)

        # print("\n==================== Parameters (HL Tau) ====================")
        # print(f"M_0          = {M_0:.3e}")
        # print(f"M_0 range = ({M_0_min:.3e}, {M_0_max:.3e})")
        # print("===============================================================")
        sys.exit(0)

    except Exception as e:
        print(f"[Quick Mode] 失敗，改跑完整流程: {e}")
        return
# ============================================================
# 3. 資料準備：讀 cube + 建 mask + 抽質心
# ============================================================
def prepare_data():
    global cube, header, im_center, dx_arcsec, dx_au, dv
    global subcube, moment0, moment1, cube_shape, spec_kms
    global rms_channel, new_cube_data, str_mom0, str_mom1
    global streamercom_x_AU, streamercom_z_AU, streamercom_v_LS_km
    global x_array, z_array, v_array, weights_array, x_means, z_means, v_means
    global v_weight_pix, v_weight_phys, max_dist_value

    # --------------------------------------------------
    # 1) Read cube & 2D header
    # --------------------------------------------------
    cube = SpectralCube.read(cube_fname)
    header = fits.getheader(cube_fname)   # 只給 2D 用

    # velocity unit → km/s（SpectralCube 會同步更新 WCS）
    cube = cube.with_spectral_unit(
        u.km/u.s,
        velocity_convention="radio",
        rest_value=header["RESTFRQ"] * u.Hz,
    )

    # 2D geometry
    im_center = (int(round(header["CRPIX2"] - 1.0)),
                int(round(header["CRPIX1"] - 1.0)))
    dx_arcsec = abs(header["CDELT1"]) * 3600.0
    dx_au     = dx_arcsec * distance_pc

    # --------------------------------------------------
    # 2) Spectral slab (THIS defines the real 3D WCS)
    # --------------------------------------------------
    subcube = cube.spectral_slab(-1.6*u.km/u.s, 15*u.km/u.s)
    cube_shape = subcube.shape

    spec_kms = subcube.spectral_axis.to_value(u.km/u.s)   # <-- slab 後的真實速度軸
    dv = float(spec_kms[1] - spec_kms[0])

    # --------------------------------------------------
    # 3) Moment maps (2D, 用原 header)
    # --------------------------------------------------
    moment0 = subcube.moment(order=0).value
    moment1 = subcube.moment(order=1).value

    fits.PrimaryHDU(moment0, header).writeto("HL_Tau_HCOp32_mom0.fits", overwrite=True)
    hdu1 = fits.PrimaryHDU(moment1, header)
    hdu1.header["BUNIT"] = "km/s"
    hdu1.writeto("HL_Tau_HCOp32_mom1.fits", overwrite=True)

    rms_channel = 2.509251078668e-3

    # --------------------------------------------------
    # 4) Keep ONLY redshifted channels (still a SpectralCube)
    # --------------------------------------------------
    red_mask_v = (spec_kms > Local_Standard_Velocity)     # (nv,)
    red_subcube = subcube.with_mask(red_mask_v[:, None, None])

    # --------------------------------------------------
    # 5) Build streamer cube (IMPORTANT)
    # --------------------------------------------------
    im_center, masked_cube, new_cube_data = build_streamer_masked_cube_hltau(
        red_subcube, header, rms_channel
    )

    # 3D header MUST come from masked_cube
    header3d = masked_cube.hdu.header.copy()
    header3d["CUNIT3"] = "km/s"

    fits.PrimaryHDU(new_cube_data, header3d).writeto(
        "HL_Tau_HCOp32_streamer_cube.fits", overwrite=True
    )

    # streamer moments
    str_mom0 = masked_cube.moment(order=0).value
    str_mom1 = masked_cube.moment(order=1).value

    fits.PrimaryHDU(str_mom0, header).writeto(
        "HL_Tau_HCOp32_streamer_mom0.fits", overwrite=True
    )
    fits.PrimaryHDU(str_mom1, header).writeto(
        "HL_Tau_HCOp32_streamer_mom1.fits", overwrite=True
    )

    # --------------------------------------------------
    # 5) Extract centroids (spec_kms 直接傳)
    # --------------------------------------------------
    (
        streamercom_x_AU,
        streamercom_z_AU,
        streamercom_v_LS_km,
        x_array,
        z_array,
        v_array,
        weights_array,
        x_means,
        z_means,
        v_means,
    ) = extract_streamer_centroids(
        new_cube_data,
        header,
        pa_rad,
        dx_au,
        spec_kms,
    )

    v_weight_pix  = (dv / dx_arcsec) ** 2
    v_weight_phys = (dx_au / dv) ** 2
    max_dist_value = 50.0
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
    # print(v_means)
    # print(streamercom_v_LS_km)
    
# --- Grid fitting logic moved into function ---
def run_grid():
    global Theta_init, Phi_init, Incl_init, T_init, Omega_init
    global parameter_prior_ranges, sigma_like
    if RUN_GRID:
        best_params, grid, error = run_grid_search(
            streamercom_x_AU, streamercom_z_AU, streamercom_v_LS_km,
            v_weight_phys, M_star, scale, log_power, radius_ref_au,
            n_grid=10,
            phi_grid=(-np.pi, np.pi),
            T_factor_range=(2.366e-02, 1.133), #5.4, 245.0
            verbose=True,
        )
        
        sigma_like = None
        parameter_prior_ranges, sigma_like = compute_priors_from_grid(
            error, grid, best_params["best_val"]
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
            "grid_used": True,
            "grid_best_Theta": float(Theta_init),
            "grid_best_Phi":   float(Phi_init),
            "grid_best_Incl":  float(Incl_init),
            "grid_best_T":     float(T_init),
            "grid_best_Omega": float(Omega_init),
            "grid_best_error": float(best_params["best_val"]),
            "sigma_like_fast": float(sigma_like)
        })
        for key, (lo, hi) in parameter_prior_ranges.items():
            cache[f"prior_{key}_lo"] = float(lo)
            cache[f"prior_{key}_hi"] = float(hi)
        np.savez(CACHE_PATH_GRID, **cache)
        print(f"[cache] Saved grid search results to {CACHE_PATH_GRID}")
        print("Grid results:") 
        print(f"Theta = {np.rad2deg(Theta_init):.3f} deg")
        print(f"Phi   = {np.rad2deg(Phi_init):.3f} deg")
        print(f"Incl  = {np.rad2deg(Incl_init):.3f} deg")
        print(f"T     = {T_init:.6f} Myr")
        print(f"Omega = {Omega_init:.4f}")
    else:
        print("[Grid fitting] Skipped (manual init used).")

# ============================================================
# 5. MCMC_grid：fast likelihood（選配）
# ============================================================
def run_mcmc_grid():
    if not RUN_MCMC_GRID:
        print("[MCMC_grid] Skipped (RUN_MCMC_GRID = False)")
        return
    print("\n[MCMC_grid] start 29 質心 fast likelihood)")

    cache.get("grid_used", True)
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

    p0 = np.zeros((nwalkers, ndim))
    for j, key in enumerate(labels_5d):
        lo, hi = parameter_prior_ranges[key]
        prop = center_vals[j] + sigma_vals[j] * np.random.randn(nwalkers)
        prop = np.clip(prop, lo, hi)
        p0[:, j] = prop

    moves = get_mcmc_moves(mode="refine")#
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

    Theta_pk5d, Phi_pk5d, Incl_pk5d, T_pk5d, Omega_pk5d = peak_5d_rad

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
        "mcmc_grid_peak2d_Theta": float(Theta_pk5d),
        "mcmc_grid_peak2d_Phi": float(Phi_pk5d),
        "mcmc_grid_peak2d_Incl": float(Incl_pk5d),
        "mcmc_grid_peak2d_T": float(T_pk5d),
        "mcmc_grid_peak2d_Omega": float(Omega_pk5d),
        "mcmc_grid_peak2d_Theta_deg": float(Theta_pk5d_deg),
        "mcmc_grid_peak2d_Phi_deg": float(Phi_pk5d_deg),
        "mcmc_grid_peak2d_Incl_deg": float(Incl_pk5d_deg),

        "mcmc_grid_peak2d_bins": int(bins_corner),
        "mcmc_grid_peak2d_smooth": float(smooth_corner),
        "mcmc_grid_peak2d_pair_peaks": np.array(
            [(i, j, pair_peaks[(i,j)][0], pair_peaks[(i,j)][1]) for (i,j) in sorted(pair_peaks.keys())],
            dtype=float
        ),
        "mcmc_grid_flat_samples": flat,               # raw
        "burnin": int(burnin),
        "thin": int(thin),
    })

    np.savez(CACHE_PATH_MCMC_GRID, **cache)
    print(f"[cache] Saved MCMC grid results to {CACHE_PATH_MCMC_GRID}")  
    M_0 = M_star * M_SUN_KG * spc.G / (200.0**3 * T_med * 1e6 * spc.year)

    print("\n==== Dimensionless mass (HLTau) ====")
    print(f"M_0     = {M_0:.3e}")
    print("====================================")


# ============================================================
# 5. 殼層版 MCMC：使用 log_posterior_shell（最簡潔版本）
# ============================================================
def run_mcmc_shell():
    if not RUN_MCMC_SHELL:
        print("[MCMC_shell] Skipped (RUN_MCMC_SHELL = False)")
        return

    print("\n[MCMC_shell] start (distance-shell likelihood)")
    
    ndim = 5
    labels_5d = ["Theta zero", "Phi zero", "Inclination", "Time", "Omega"]
    nwalkers, nsteps = 32, 4000  
    # cache_path_to_use = _resolve_cache_path(USE_CACHE_SOURCE)
    # cache = np.load(cache_path_to_use, allow_pickle=True)
    # print(f"[cache] Loaded cache ({USE_CACHE_SOURCE}): {cache_path_to_use}")

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
    sigma_vals  =  [
    np.deg2rad(9.0),
    np.deg2rad(18.0),
    np.deg2rad(9.0),
    0.05 * (parameter_prior_ranges["Time"][1] - parameter_prior_ranges["Time"][0]),
    0.05 * (parameter_prior_ranges["Omega"][1] - parameter_prior_ranges["Omega"][0]),
    ]
    
    p0 = np.zeros((nwalkers, ndim))
    for j, key in enumerate(labels_5d):
        lo, hi = parameter_prior_ranges[key]
        prop = center_vals[j] + sigma_vals[j] * np.random.randn(nwalkers)
        prop = np.clip(prop, lo, hi)
        p0[:, j] = prop
    
    down_factor = 3
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
    header3d = fits.getheader("HL_Tau_HCOp32_streamer_cube.fits")
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
        Theta_center, Phi_center, Incl_center, T_center, Omega_center,
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
        burnin, thin = 30, 5

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

    print("\n[MCMC_shell] 1D posterior 形狀判斷：")
    for i, name in enumerate(labels_5d):
        summarize_1d_posterior(flat[:, i], name)

    # -----------------------------
    # 9) corner plot（角度轉度）
    # -----------------------------
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
        
        # --- Diagnostics (optional but nice to keep consistent with grid) ---
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
    從 cache 抓出最終 best-fit 參數，畫在 HL Tau streamer 的
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
            str_mom0 = fits.getdata("HL_Tau_HCOp32_streamer_mom0.fits")
            str_mom1 = fits.getdata("HL_Tau_HCOp32_streamer_mom1.fits")
            header   = fits.getheader("HL_Tau_HCOp32_streamer_mom1.fits")
            header_kms = fits.getheader("HL_Tau_HCOp32_streamer_cube.fits")
        except Exception as e:
            print(f"[overlay] Failed to load streamer moment maps from FITS: {e}")
            return

        im_center = (int(round(header["CRPIX2"] - 1.0)),
                    int(round(header["CRPIX1"] - 1.0)))
        dx_arcsec = abs(header["CDELT1"]) * 3600.0
        dx_au     = dx_arcsec * distance_pc

        # 讀取 streamer cube（如果有的話就畫 z–v）
        new_cube_data = None
        try:
            new_cube_data = fits.getdata("HL_Tau_HCOp32_streamer_cube.fits")
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
                header=header_kms,
                pa_rad=pa_rad,
                dx_au=dx_au,
                z_means_pix=None,          # 目前不疊 centroids；之後需要可以從 cache 裡抓
                streamer_v_LS_km=None,
                outname="HLTau_z_v_data_overlay_best.png",
                label="HL Tau HCO$^{+}$ z–v (best-fit)",
            )

        # 2) moment-1 overlay（速度場）
        plot_streamer_on_mom1(
            Theta_best, Phi_best, Incl_best,
            float(T_best), float(Omega_best),
            header, pa_rad, dx_au, im_center,
            str_mom1,
            label="HL Tau HCO$^{+}$ moment1 (best-fit)",
            outname="HLTau_HCOp32_model_vs_mom1_overlay_best.png",
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
            label="HL Tau HCO$^{+}$ moment0 (best-fit)",
            outname="HLTau_HCOp32_model_vs_mom0_overlay_best.png",
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
        run_quick_mode_hltau()
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