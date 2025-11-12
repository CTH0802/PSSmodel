#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-emb-50 streamer fitting script（整理版）

流程總覽：
  1) 參數 / 開關 / Imports
  2) Helper functions（mask, centroids, plotting, MCMC moves）
  3) Quick Mode：RUN_FROM_CACHE_ONLY=True 時，直接讀 cache 畫圖後結束
  4) 正常流程：
      - 讀 cube + 建 streamer mask
      - 抽 streamer 質心
      - Grid search（選配）
      - MCMC_grid（選配）
      - MCMC_distance（選配）
      - 決定最終 best-fit 參數，寫入 cache
      - 用 cache best-fit + 質心畫 moment0/moment1 overlay
"""

# ============================================================
# 1. Imports / 基本參數 / 開關
# ============================================================

import os
import sys
import numpy as np
import scipy.constants as spc
import emcee

from scipy.interpolate import interp1d
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from spectral_cube import SpectralCube

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import PowerNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import corner
from tqdm.auto import tqdm
from scipy.signal import find_peaks

import PSSpy as pss
from pss_grid_search import run_grid_search, compute_priors_from_grid

# ---------- 基本物理與檔名設定 ----------
Local_Standard_Velocity = 7.5  # km/s
pa_deg = 170 + 90
pa_rad = np.deg2rad(pa_deg)

distance_pc = 300.0
M_SUN_KG = 1.98847e30
M_star = 2.58                 # M_sun
radius_ref_au = 240.0

scale = "log"
log_power = 1.5

radius_in_au, radius_out_au = 2e2, 3.8e3
cube_fname = "Per-emb-50_CD_l021l060_uvsub_H2CO_multi_small_fitcube.fits"

PLOT_DIR = "Per-emb-50_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

CACHE_PATH = os.path.join("Per-emb-50_fit_results.npz")

# ---------- 開關 ----------
RUN_GRID = False               # 5D grid search 找初始解
RUN_MCMC_GRID = False          # 11 個質心點 fast likelihood
RUN_MCMC_DISTANCE = False     # distance_cube MCMC
RUN_MCMC_GRID_REFINE = False  # MCMC_grid 多峰局部 refinement
RUN_MCMC_3D = False           # (Theta, Phi, Incl) 測試
RUN_FROM_CACHE_ONLY = True   # True: 僅讀 cache 畫圖，完全不重跑

# 統一 cache 容器（本次執行逐步填入）
cache = {
    "target": "Per-emb-50",
    "cube_fname": cube_fname,
    "distance_pc": distance_pc,
    "M_star": M_star,
    "pa_deg": pa_deg,
    "Local_Standard_Velocity": Local_Standard_Velocity,
    "radius_in_au": radius_in_au,
    "radius_out_au": radius_out_au,
}


# ============================================================
# 2. Helper functions
# ============================================================

def get_mcmc_moves(mode="explore"):
    if mode == "explore":
        return [
            (emcee.moves.StretchMove(a=2.5), 0.3),
            (emcee.moves.DEMove(),           0.6),
            (emcee.moves.DESnookerMove(),    0.1),
        ]
    elif mode == "refine":
        return [
            (emcee.moves.StretchMove(a=2.5), 0.8),
            (emcee.moves.DEMove(),           0.1),
            (emcee.moves.DESnookerMove(),    0.1),
        ]
    else:
        return [
            (emcee.moves.StretchMove(a=2.5), 0.4),
            (emcee.moves.DEMove(),           0.4),
            (emcee.moves.DESnookerMove(),    0.2),
        ]


def build_streamer_masked_cube(cube, header, rms_channel):
    """
    回傳:
      im_center (y, x),
      masked_center_cube,
      masked_cube,
      new_cube_data (np.ndarray, masked_cube 填 0)
    """
    im_center = (int(header["CRPIX2"]), int(header["CRPIX1"]))
    ny, nx = cube.shape[1], cube.shape[2]

    masked_center_cube = cube

    # 手動 mask 核心
    mask_specs = [
        (4, [108, 67]),
        (3, [121, 64]),
        (4, [103, 72]),
        (4.5, [114, 66]),
        (3, [94, 84]),
        (6.5, [99, 79]),
    ]
    for radius, pos in mask_specs:
        mask2d = pss.circular_mask((ny, nx), pos, radius)
        mask3d = np.repeat(mask2d[np.newaxis, :, :], cube.shape[0], axis=0)
        masked_center_cube = masked_center_cube.with_mask(mask3d)

    # grow_region 找 streamer
    init_points = [
        (34, im_center[0], im_center[1]),
        (34, 121, 50),
        (34, 115, 53),
        (34, 109, 55),
        (34, 105, 58),
        (34, 93, 66),
        (34, 71, 73),
    ]
    maskcent_cube_data = masked_center_cube.filled_data[:].value
    maskcent_stream_mask = pss.grow_region(
        maskcent_cube_data,
        init_points,
        rms_channel,
        sigma_thresh=3,
        max_iter=1000,
    )

    masked_cube = masked_center_cube.with_mask(maskcent_stream_mask)
    masked_cube = masked_cube.with_fill_value(0.0)
    new_cube_data = masked_cube.filled_data[:].value

    return im_center, masked_center_cube, masked_cube, new_cube_data


def extract_streamer_centroids(new_cube_data, header, pa_rad, dx_au,
                               v_lastch_vel, v_lastch_num):
    """
    從 masked cube 抽出 11 個 streamer 質心點。
    回傳:
      streamer_x_AU, streamer_z_AU, streamer_v_LS_km
    """
    cube_shape = new_cube_data.shape
    im_center = (int(header["CRPIX2"]), int(header["CRPIX1"]))

    v, z, x = np.indices(cube_shape)
    x_rel = x - im_center[1]
    z_rel = z - im_center[0]
    r, theta = pss.spherical_coords(x_rel, z_rel)

    # 預先指定大致流線方向（per-emb-50 已調過）
    find_x = np.array([0, 5, 8, 10, 13, 21, 28])
    find_y = np.array([0, -4, -10, -16, -20, -32, -54])
    find_r, find_theta = pss.spherical_coords(find_x, find_y)
    find_streaml = interp1d(find_r, find_theta, fill_value="extrapolate")

    N = 11
    pars = np.linspace(0.07, 67, N + 1)

    x_means = np.zeros(N)
    z_means = np.zeros(N)
    v_means = np.zeros(N)
    xzstd   = np.zeros(N)
    x_array_list = []
    z_array_list = []
    v_array_list = []
    weights_list = []

    # step 1: 幾何中心（加上方向權重）
    for i in range(N):
        r_mid = 0.5 * (pars[i] + pars[i+1])
        theta0 = find_streaml(r_mid)

        weight_theta = (x_rel * np.cos(theta0) + z_rel * np.sin(theta0)) / r
        weight_theta[r == 0] = 0
        weight_theta[weight_theta < 0.99] = 0

        d = (r > pars[i]) & (r <= pars[i+1]) & (new_cube_data > 0)
        if np.sum(d) > 0:
            w = new_cube_data[d] * weight_theta[d]
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
    for i in range(N):
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
        else:
            x_means[i] = z_means[i] = v_means[i] = np.nan
            # 存儲每次迴圈的值
        x_array_list.append(x_rel[d])
        z_array_list.append(z_rel[d])
        v_array_list.append(v[d])
        weights_list.append(weights[d]/np.max(weights[d]))

    # step 4: 轉物理單位 + 旋轉到 model frame
    streamer_x_pix = np.array(x_means)
    streamer_z_pix = np.array(z_means)
    streamer_v_pix = np.array(v_means)

    x_rot = streamer_x_pix * np.cos(pa_rad) + streamer_z_pix * np.sin(pa_rad)
    z_rot = -streamer_x_pix * np.sin(pa_rad) + streamer_z_pix * np.cos(pa_rad)

    streamer_x_AU = x_rot * dx_au
    streamer_z_AU = z_rot * dx_au

    dv = abs(float(header["CDELT3"]))  # km/s / channel
    streamer_v_km = v_lastch_vel + (v_lastch_num - streamer_v_pix) * dv
    streamer_v_LS = streamer_v_km - Local_Standard_Velocity
    x_array = np.array(x_array_list, dtype=object)
    z_array = np.array(z_array_list, dtype=object)
    v_array = np.array(v_array_list, dtype=object)
    weights_array = np.array(weights_list, dtype=object)
    print(f"[Centroids] 有效質心點數量 = {np.sum(np.isfinite(streamer_x_AU))}")
    return streamer_x_AU, streamer_z_AU, streamer_v_LS, x_array, z_array, v_array, weights_array, x_means, z_means, v_means


def summarize_1d_posterior(samples, name, bins=40):
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
    peak_mask = (smooth[1:-1] > smooth[:-2]) & (smooth[1:-1] > smooth[2:])
    peaks = smooth[1:-1][peak_mask]
    if peaks.size == 0:
        n_peaks = 0
    else:
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
    samples = np.asarray(samples)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return []

    hist, edges = np.histogram(samples, bins=bins)
    if np.all(hist == 0):
        return []

    prominence = prominence_frac * np.max(hist)
    peaks, _ = find_peaks(hist, prominence=prominence)
    if peaks.size == 0:
        return []

    centers = 0.5 * (edges[peaks] + edges[peaks + 1])
    bin_width = edges[1] - edges[0]
    half_widths = np.full_like(centers, 3.0 * bin_width)
    return list(zip(centers, half_widths))


# === Overlay helpers（與 Per-emb-2 統一風格） ===

def _compute_extent(header, im_center, ny, nx):
    dx_arcsec = abs(header["CDELT1"]) * 3600.0
    dz_arcsec = abs(header["CDELT2"]) * 3600.0
    ra_min = (0   - im_center[1]) * dx_arcsec
    ra_max = (nx  - im_center[1]) * dx_arcsec
    dec_min= (0   - im_center[0]) * dz_arcsec
    dec_max= (ny  - im_center[0]) * dz_arcsec
    return (min(ra_min, ra_max), max(ra_min, ra_max),
            min(dec_min, dec_max), max(dec_min, dec_max)), dx_arcsec, dz_arcsec


def plot_streamer_on_mom0(theta_deg, phi_deg, inc_deg, T_Myr, omega,
                          header, pa_rad, dx_au, im_center,
                          mom0, label, outname,
                          cen_x_pix=None, cen_z_pix=None, cen_v_LS_km=None,
                          radius_in_au=radius_in_au,
                          radius_out_au=radius_out_au):
    theta = np.deg2rad(theta_deg)
    phi   = np.deg2rad(phi_deg)
    inc   = np.deg2rad(inc_deg)

    ny, nx = mom0.shape
    extent, dx_arcsec, dz_arcsec = _compute_extent(header, im_center, ny, nx)

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

    x_pix_rot = x_pix * np.cos(pa_rad) - z_pix * np.sin(pa_rad) + im_center[1]
    z_pix_rot = x_pix * np.sin(pa_rad) + z_pix * np.cos(pa_rad) + im_center[0]

    ra_off  = (x_pix_rot - im_center[1]) * dx_arcsec
    dec_off = (z_pix_rot - im_center[0]) * dz_arcsec

    pts = np.column_stack([ra_off, dec_off])
    if pts.shape[0] < 2:
        print("[mom0] model points too few, skip.")
        return
    segments = np.stack([pts[:-1], pts[1:]], axis=1)

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
    )

    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="3%", pad=0.04)
    # cbar = fig.colorbar(im, cax=cax)
    # cbar.set_label("(Jy/beam km/s)")

    # lc_edge = LineCollection(segments, colors="black", linewidth=6, zorder=2)
    # ax.add_collection(lc_edge)

    # # 用 model v_m + LSR 當顏色（範圍依資料可調）
    # v_model_LSR = v_m + Local_Standard_Velocity
    # v_seg = 0.5 * (v_model_LSR[:-1] + v_model_LSR[1:])
    # norm_v = mpl.colors.Normalize(vmin=5.5, vmax=8.0)
    # lc = LineCollection(
    #     segments,
    #     cmap="coolwarm",
    #     norm=norm_v,
    #     linewidth=4.5,
    #     zorder=3,
    # )
    # lc.set_array(v_seg)
    # ax.add_collection(lc)
    
    num_element = 8
    xarray_arc, z_array_arc = x_array[num_element] * dx_arcsec, z_array[num_element] * dx_arcsec
    weights_im = ax.scatter( xarray_arc, z_array_arc, c=weights_array[num_element], s=8, cmap='YlGn_r')
    x_means_arc, z_means_arc = x_means * dx_arcsec, z_means * dx_arcsec
    ax.plot(x_means_arc, z_means_arc, color='w', lw=3, zorder=4)
    divider = make_axes_locatable(ax)
    cax     = divider.append_axes('right', size='3%', pad=0.04)
    cbar = fig.colorbar(weights_im, cax=cax)
    cbar.set_label('weight value')

    # # 質心點（若提供）
    # if cen_x_pix is not None and cen_z_pix is not None:
    #     cen_ra  = (cen_x_pix - im_center[1]) * dx_arcsec
    #     cen_dec = (cen_z_pix - im_center[0]) * dz_arcsec
    #     if cen_v_LS_km is not None:
    #         cen_v = cen_v_LS_km + Local_Standard_Velocity
    #         ax.scatter(
    #             cen_ra, cen_dec,
    #             c=cen_v,
    #             cmap="coolwarm",
    #             vmin=5.5, vmax=8.0,
    #             s=20,
    #             marker="o",
    #             edgecolors="black",
    #             linewidths=0.6,
    #             zorder=5,
    #             label="Centroids",
    #         )
    #     else:
    #         ax.scatter(
    #             cen_ra, cen_dec,
    #             facecolors="none",
    #             edgecolors="black",
    #             s=36,
    #             marker="o",
    #             zorder=5,
    #             label="Centroids",
    #         )

    ax.scatter(0, 0, c="r", s=50, marker="+", zorder=6)

    ax.set_xlabel("RA Offset (arcsec)")
    ax.set_ylabel("Dec Offset (arcsec)")
    ax.set_xlim(-4, 10.5)
    ax.set_ylim(-12, 2.5)
    ax.set_title(label)
    ax.set_aspect("equal", adjustable="box")
    # --- 比例尺與方向標示 ---
    # 定義比例尺位置（以 arcsec 為單位）
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    text_pos_x = x1 - 0.05 * (x1 - x0)
    text_pos_y = y0 + 0.15 * (y1 - y0)
    scale_length = 500  # AU

    # 將 3000 AU 轉成 arcsec
    scale_length_arcsec = scale_length / (distance_pc)  # 1" ≈ 1 AU / distance(pc)

    # 定義比例尺線段 (RA offset 軸)
    scale_range_x = [text_pos_x, text_pos_x - scale_length_arcsec]
    scale_range_y = [text_pos_y - 0.2, text_pos_y - 0.2]

    # 繪製比例尺與文字
    ax.plot(scale_range_x, scale_range_y, color='k', lw=3, zorder=10)
    ax.text(
        text_pos_x - scale_length_arcsec / 2,
        text_pos_y - 1.0,
        f"{int(scale_length)} AU",
        ha='center', va='bottom',
        fontsize=14, family='Times New Roman', color='k'
    )

    # --- 加上方向箭頭 (NE arrow) ---
    ax.quiver(
        0.4,  0.4 * np.tan(np.deg2rad(10)),
        1.4, -1.4 * np.tan(np.deg2rad(10)),
        color='lightgrey', scale=12, zorder=10
    )
    # beam（若有）
    try:
        bmaj = header.get("BMAJ", None)
        bmin = header.get("BMIN", None)
        bpa  = header.get("BPA", 0.0)
        if bmaj and bmin:
            bmaj_arcsec = bmaj * 3600.0
            bmin_arcsec = bmin * 3600.0
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
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


def plot_streamer_on_mom1(theta_deg, phi_deg, inc_deg, T_Myr, omega,
                          header, pa_rad, dx_au, im_center,
                          mom1, label, outname,
                          cen_x_pix=None, cen_z_pix=None, cen_v_LS_km=None,
                          v_range=1.0,
                          radius_in_au=radius_in_au,
                          radius_out_au=radius_out_au):

    theta = np.deg2rad(theta_deg)
    phi   = np.deg2rad(phi_deg)
    inc   = np.deg2rad(inc_deg)

    ny, nx = mom1.shape
    extent, dx_arcsec, dz_arcsec = _compute_extent(header, im_center, ny, nx)

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

    x_pix_rot = x_pix * np.cos(pa_rad) - z_pix * np.sin(pa_rad) + im_center[1]
    z_pix_rot = x_pix * np.sin(pa_rad) + z_pix * np.cos(pa_rad) + im_center[0]

    ra_off  = (x_pix_rot - im_center[1]) * dx_arcsec
    dec_off = (z_pix_rot - im_center[0]) * dz_arcsec

    pts = np.column_stack([ra_off, dec_off])
    if pts.shape[0] < 2:
        print("[mom1] model points too few, skip.")
        return
    segments = np.stack([pts[:-1], pts[1:]], axis=1)

    vmin = 5.5
    vmax = 8

    fig, ax = plt.subplots(figsize=(6.2, 6))
    im = ax.imshow(
        mom1,
        origin="lower",
        cmap="coolwarm",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.04)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Velocity (km/s)")

    # # model 線
    # lc_edge = LineCollection(segments, colors="black", linewidth=6, zorder=2)
    # ax.add_collection(lc_edge)

    # v_model_LSR = v_m + Local_Standard_Velocity
    # v_seg = 0.5 * (v_model_LSR[:-1] + v_model_LSR[1:])
    # norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    # lc = LineCollection(
    #     segments,
    #     cmap="coolwarm",
    #     norm=norm,
    #     linewidth=4.5,
    #     zorder=3,
    # )
    # lc.set_array(v_seg)
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
    
    # # 質心
    # if cen_x_pix is not None and cen_z_pix is not None:
    #     cen_ra  = (cen_x_pix - im_center[1]) * dx_arcsec
    #     cen_dec = (cen_z_pix - im_center[0]) * dz_arcsec
    #     if cen_v_LS_km is not None:
    #         cen_v = cen_v_LS_km + Local_Standard_Velocity
    #         ax.scatter(
    #             cen_ra, cen_dec,
    #             c=cen_v,
    #             cmap="coolwarm",
    #             vmin=vmin, vmax=vmax,
    #             s=20,
    #             marker="o",
    #             edgecolors="black",
    #             linewidths=0.6,
    #             zorder=5,
    #             label="Centroids",
    #         )
    #     else:
    #         ax.scatter(
    #             cen_ra, cen_dec,
    #             facecolors="none",
    #             edgecolors="black",
    #             s=36,
    #             marker="o",
    #             zorder=5,
    #             label="Centroids",
    #         )

    ax.scatter(0, 0, c="r", s=50, marker="+", zorder=6)

    ax.set_xlabel("RA Offset (arcsec)")
    ax.set_ylabel("Dec Offset (arcsec)")
    ax.set_xlim(-4, 10.5)
    ax.set_ylim(-12, 2.5)
    ax.set_title(label)
    ax.set_aspect("equal", adjustable="box")
    # --- 比例尺與方向標示 ---
    # 定義比例尺位置（以 arcsec 為單位）
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    text_pos_x = x1 - 0.05 * (x1 - x0)
    text_pos_y = y0 + 0.15 * (y1 - y0)
    scale_length = 500  # AU

    # 將 3000 AU 轉成 arcsec
    scale_length_arcsec = scale_length / (distance_pc)  # 1" ≈ 1 AU / distance(pc)

    # 定義比例尺線段 (RA offset 軸)
    scale_range_x = [text_pos_x, text_pos_x - scale_length_arcsec]
    scale_range_y = [text_pos_y - 0.2, text_pos_y - 0.2]

    # 繪製比例尺與文字
    ax.plot(scale_range_x, scale_range_y, color='k', lw=3, zorder=10)
    ax.text(
        text_pos_x - scale_length_arcsec / 2,
        text_pos_y - 1.0,
        f"{int(scale_length)} AU",
        ha='center', va='bottom',
        fontsize=14, family='Times New Roman', color='k'
    )

    # --- 加上方向箭頭 (NE arrow) ---
    ax.quiver(
        0.4,  0.4 * np.tan(np.deg2rad(10)),
        1.4, -1.4 * np.tan(np.deg2rad(10)),
        color='grey', scale=12, zorder=10
    )
    # beam（若有）
    try:
        bmaj = header.get("BMAJ", None)
        bmin = header.get("BMIN", None)
        bpa  = header.get("BPA", 0.0)
        if bmaj and bmin:
            bmaj_arcsec = bmaj * 3600.0
            bmin_arcsec = bmin * 3600.0
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
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
                               streamer_z_AU, streamer_v_LS_km,
                               outname,
                               label="Per-emb-50 H2CO z-v with data",
                               nbins_z=200):
    """
    由 masked data cube 直接建立 z–v 圖：

      x: model-frame z (AU)，以 protostar 為 0
      y: V_LSR (km/s)
      背景: new_cube_data 在「垂直於 streamer 方向」平均後的強度 (簡化實作)
      藍線: PSS_model z-v
      黑點: streamer 質心

    只需要 cube，無需額外 PV fits。
    """

    if new_cube_data is None:
        print("[z–v] new_cube_data is None, skip.")
        return

    # ---------- 1. 速度軸 ----------
    nz, ny, nx = new_cube_data.shape
    CRVAL3 = float(header["CRVAL3"])
    CRPIX3 = float(header["CRPIX3"])
    CDELT3 = float(header["CDELT3"])
    v_axis = CRVAL3 + (np.arange(nz) + 1 - CRPIX3) * CDELT3  # km/s, LSR

    vmin = np.nanmin(v_axis)
    vmax = np.nanmax(v_axis)

    # ---------- 2. 建立旋轉後的 z 座標 (AU) ----------
    # header CRPIX 是 1-based，要轉成 index-0
    im_cy = float(header["CRPIX2"]) - 1.0
    im_cx = float(header["CRPIX1"]) - 1.0

    y_idx = np.arange(ny)
    x_idx = np.arange(nx)
    xx, yy = np.meshgrid(x_idx, y_idx)  # xx: x, yy: y

    x_rel = xx - im_cx
    z_rel = yy - im_cy

    # 旋轉到 model frame
    x_rot = x_rel * np.cos(pa_rad) + z_rel * np.sin(pa_rad)
    z_rot = -x_rel * np.sin(pa_rad) + z_rel * np.cos(pa_rad)

    # z in AU（以 protostar 為 0）
    z_AU = z_rot * dx_au

    # ---------- 3. 沿垂直方向平均，得到 pv(v, z) ----------
    # 我們把每個 voxel 的 z_AU 丟進 bins，對應的強度加總，再除以權重
    z_min = np.nanmin(z_AU)
    z_max = np.nanmax(z_AU)
    z_bins = np.linspace(z_min, z_max, nbins_z + 1)
    z_centers = 0.5 * (z_bins[:-1] + z_bins[1:])

    pv = np.zeros((nz, nbins_z))
    pv[:] = np.nan

    # 展平成 1D，方便 histogram
    z_flat = z_AU.ravel()
    for k in range(nz):
        I = new_cube_data[k].ravel()
        m = np.isfinite(I) & (I > 0)

        if not np.any(m):
            continue

        # 對每個 z-bin 做加權平均
        # 用 histogram 計算總權重和加權和
        w_sum, _ = np.histogram(z_flat[m], bins=z_bins, weights=I[m])
        cnt, _   = np.histogram(z_flat[m], bins=z_bins)

        with np.errstate(invalid="ignore", divide="ignore"):
            pv_row = w_sum / cnt
        pv[k, :] = pv_row

    # ---------- 4. 繪圖 ----------
    fig, ax = plt.subplots(figsize=(7, 4))

    # 背景 PV（灰階）
    img = ax.imshow(
        pv,
        origin="lower",
        cmap="Greys_r",
        extent=[z_centers[0], z_centers[-1], vmin, vmax],
        aspect="auto",
        vmin=np.nanpercentile(pv, 5),
        vmax=np.nanpercentile(pv, 99),
    )

    # ---------- 5. 疊上 PSS_model 的 z–v ----------
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

    ax.plot(
        z_m,
        v_m + Local_Standard_Velocity,
        color="tab:blue",
        lw=2.0,
        label="Model",
        zorder=3,
    )

    # ---------- 6. 疊上質心 ----------
    if streamer_z_AU is not None and streamer_v_LS_km is not None:
        z_data = np.asarray(streamer_z_AU)
        v_data = np.asarray(streamer_v_LS_km) + Local_Standard_Velocity
        good = np.isfinite(z_data) & np.isfinite(v_data)
        ax.scatter(
            z_data[good],
            v_data[good],
            c="k",
            s=30,
            edgecolors="white",
            linewidths=0.6,
            label="Centroids",
            zorder=4,
        )

    # ---------- 7. 外觀 ----------
    ax.set_xlabel("z (AU)")
    ax.set_ylabel("Velocity (km/s, LSR)")
    ax.set_title(label)
    ax.set_ylim(vmin, vmax)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, outname), dpi=200)
    plt.close(fig)
    print(f"[z–v] Saved {outname}")
    
# ============================================================
# 3. Quick Mode：只用 cache 畫圖就結束
# ============================================================

if RUN_FROM_CACHE_ONLY:
    print("[Quick Mode] RUN_FROM_CACHE_ONLY=True → 僅讀取 cache 並繪圖")

    try:
        c = np.load(CACHE_PATH, allow_pickle=True)
        print(f"[cache] Loaded cache: {CACHE_PATH}")

        # 選擇最終 best-fit（新欄位，若不存在則 fallback）
        if "best_Theta" in c:
            Theta_best = c["best_Theta"]
            Phi_best   = c["best_Phi"]
            Incl_best  = c["best_Incl"]
            T_best     = c["best_T"]
            Omega_best = c["best_Omega"]
        else:
            # 舊版 fallback：優先順序 MCMC_grid > MCMC_distance > grid
            Theta_best = c.get(
                "mcmc_grid_median_Theta",
                c.get("mcmc_distance_median_Theta",
                      c["grid_best_Theta"])
            )
            Phi_best = c.get(
                "mcmc_grid_median_Phi",
                c.get("mcmc_distance_median_Phi",
                      c["grid_best_Phi"])
            )
            Incl_best = c.get(
                "mcmc_grid_median_Incl",
                c.get("mcmc_distance_median_Incl",
                      c["grid_best_Incl"])
            )
            T_best = c.get(
                "mcmc_grid_median_T",
                c.get("mcmc_distance_median_T",
                      c["grid_best_T"])
            )
            Omega_best = c.get(
                "mcmc_grid_median_Omega",
                c.get("mcmc_distance_median_Omega",
                      c["grid_best_Omega"])
            )

        Theta_best_deg = np.rad2deg(Theta_best)
        Phi_best_deg   = np.rad2deg(Phi_best)
        Incl_best_deg  = np.rad2deg(Incl_best)

        # 讀 streamer 專用 moment map；若不存在，用 cube 快速產生
        try:
            str_mom0 = fits.getdata("Per-emb-50_H2CO_streamer_mom0.fits")
            str_mom1 = fits.getdata("Per-emb-50_H2CO_streamer_mom1.fits")
            cube = fits.getdata("Per-emb-50_CD_l021l060_uvsub_H2CO_multi_small_fitcube.fits")
            header = fits.getheader("Per-emb-50_H2CO_streamer_mom1.fits")
        except Exception:
            cube = SpectralCube.read(cube_fname)
            header = fits.getheader(cube_fname)
            velocity_range = [10.0236, 4.0984] * u.km / u.s
            subcube = cube.spectral_slab(velocity_range[0], velocity_range[1])
            str_mom0 = subcube.moment(order=0).value
            str_mom1 = subcube.moment(order=1).value

        im_center = (int(header["CRPIX2"]), int(header["CRPIX1"]))
        dx_arcsec = abs(header["CDELT2"]) * 3600.0
        dx_au = dx_arcsec * distance_pc

        # 讀取 masked streamer cube（若有）
        try:
            new_cube_data = fits.getdata("Per-emb-50_H2CO_streamer_cube.fits")
            print("[cache] Loaded streamer cube from FITS")
        except Exception:
            print("[cache] No streamer cube found in FITS, skip loading new_cube_data")
            new_cube_data = None

        # 從 cache 抓質心（若有），轉成像素座標
        cen_x_pix = cen_z_pix = cen_v_LS = None
        if ("streamercom_x_AU" in c) and ("streamercom_z_AU" in c):
            sx = c["streamercom_x_AU"]
            sz = c["streamercom_z_AU"]
            x_rot = sx / dx_au
            z_rot = sz / dx_au
            cen_x_pix = x_rot * np.cos(pa_rad) - z_rot * np.sin(pa_rad) + im_center[1]
            cen_z_pix = x_rot * np.sin(pa_rad) + z_rot * np.cos(pa_rad) + im_center[0]
            # 這裡 streamercom_z_AU, streamercom_v_LS_km 若需要也從 c 取
            if "streamercom_z_AU" in c:
                streamercom_z_AU = c["streamercom_z_AU"]
            else:
                streamercom_z_AU = None
            if "streamercom_v_LS_km" in c:
                streamercom_v_LS_km = c["streamercom_v_LS_km"]
            else:
                streamercom_v_LS_km = None
            plot_z_v_diagram_from_cube(
                theta_deg=np.rad2deg(Theta_best),
                phi_deg=np.rad2deg(Phi_best),
                inc_deg=np.rad2deg(Incl_best),
                T_Myr=T_best,
                omega=Omega_best,
                new_cube_data=new_cube_data,
                header=header,
                pa_rad=pa_rad,
                dx_au=dx_au,
                streamer_z_AU=streamercom_z_AU,
                streamer_v_LS_km=streamercom_v_LS_km,
                outname="Per-emb-50_z_v_data_overlay.png",
            )
            if "streamercom_v_LS_km" in c:
                cen_v_LS = c["streamercom_v_LS_km"]
                x_array = c["x_array"]
                z_array = c["z_array"]
                v_array = c["v_array"]
                weights_array = c["weights_array"]
                x_means = c["x_means"]
                z_means = c["z_means"]
                v_means = c["v_means"]


        # 畫圖
        plot_streamer_on_mom1(
            Theta_best_deg, Phi_best_deg, Incl_best_deg,
            float(T_best), float(Omega_best),
            header, pa_rad, dx_au, im_center,
            str_mom1,
            label='Per-emb-50 '+r'$\rm H_2CO$',
            outname="Per-emb-50_mom1_cacheonly.png",
            cen_x_pix=cen_x_pix,
            cen_z_pix=cen_z_pix,
            cen_v_LS_km=cen_v_LS,
            v_range=1.0,
        )

        plot_streamer_on_mom0(
            Theta_best_deg, Phi_best_deg, Incl_best_deg,
            float(T_best), float(Omega_best),
            header, pa_rad, dx_au, im_center,
            str_mom0,
            label='Per-emb-50 '+r'$\rm H_2CO$',
            outname="Per-emb-50_mom0_cacheonly.png",
            cen_x_pix=cen_x_pix,
            cen_z_pix=cen_z_pix,
            cen_v_LS_km=cen_v_LS,
        )

        print("[Quick Mode] 完成 cache-based 圖片，結束程式。")
        sys.exit(0)

    except Exception as e:
        print(f"[Quick Mode] 失敗，改跑完整流程: {e}")
        # 繼續往下跑
# ============================================================
# 4. 正常流程：讀 cube + 建 mask + 質心
# ============================================================

cube = SpectralCube.read(cube_fname)
header = fits.getheader(cube_fname)

im_center = (int(header["CRPIX2"]), int(header["CRPIX1"]))
dx_arcsec = abs(header["CDELT2"]) * 3600.0
dx_au = dx_arcsec * distance_pc
dv = abs(float(header["CDELT3"]))
v0 = header["CRVAL3"]

# 根據你原本設定
v_lastch_vel = 4.0984
v_lastch_num = 70
velocity_range = [10.0236, 4.0984] * u.km / u.s

subcube = cube.spectral_slab(velocity_range[0], velocity_range[1])
moment0 = subcube.moment(order=0).value
moment1 = subcube.moment(order=1).value
rms_channel = 3.521605434804e-1

# 存一般 moment map
fits.PrimaryHDU(data=moment0, header=header).writeto(
    "Per-emb-50_H2CO_mom0.fits", overwrite=True
)
h1 = fits.PrimaryHDU(data=moment1.data, header=header)
h1.header["BUNIT"] = "km/s"
h1.writeto("Per-emb-50_H2CO_mom1.fits", overwrite=True)

# 做 streamer mask
im_center, masked_center_cube, masked_cube, new_cube_data = build_streamer_masked_cube(
    subcube, header, rms_channel
)

str_mom0 = masked_cube.moment(order=0).value
str_mom1 = masked_cube.moment(order=1).value
fits.PrimaryHDU(data=str_mom0, header=header).writeto(
    "Per-emb-50_H2CO_streamer_mom0.fits", overwrite=True
)
fits.PrimaryHDU(data=str_mom1, header=header).writeto(
    "Per-emb-50_H2CO_streamer_mom1.fits", overwrite=True
)

# 抽 streamer 質心
streamercom_x_AU, streamercom_z_AU, streamercom_v_LS_km, x_array, z_array, v_array, weights_array, x_means, z_means, v_means = extract_streamer_centroids(
    new_cube_data, header, pa_rad, dx_au, v_lastch_vel, v_lastch_num
)

cache.update({
    "streamercom_x_AU": streamercom_x_AU,
    "streamercom_z_AU": streamercom_z_AU,
    "streamercom_v_LS_km": streamercom_v_LS_km,
    "x_array": x_array,
    "z_array": z_array,
    "v_array": v_array,
    "weights_array": weights_array,
    "x_means": x_means, 
    "z_means": z_means, 
    "v_means": v_means
})

# 權重（error_function 裡面會用這個）
v_weight_phys = (dv / dx_au) ** 2

# ============================================================
# 5. Grid search
# ============================================================

if RUN_GRID:
    best_params, grid, error = run_grid_search(
        streamercom_x_AU,
        streamercom_z_AU,
        streamercom_v_LS_km,
        v_weight_phys,
        M_star,
        scale,
        log_power,
        radius_ref_au,
        n_grid=10,
        T_factor_range=(40.0, 50.0)
    )
    Theta_init = best_params["Theta"]
    Phi_init   = best_params["Phi"]
    Incl_init  = best_params["Incl"]
    T_init     = best_params["T"]
    Omega_init = best_params["Omega"]

    parameter_prior_ranges = compute_priors_from_grid(
        error, grid, best_params["best_val"]
    )

    print("\n[MCMC priors from grid]")
    for name, (lo, hi) in parameter_prior_ranges.items():
        if name in ["Theta zero", "Phi zero", "Inclination"]:
            print(f"{name:<12s}: {np.rad2deg(lo):6.2f}–{np.rad2deg(hi):6.2f} deg")
        elif name == "Time":
            print(f"{name:<12s}: {lo:.5f}–{hi:.5f} Myr")
        else:
            print(f"{name:<12s}: {lo:.3f}–{hi:.3f}")

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
else:
    print("[Grid] Skipped, using manual priors.")
    Theta_init = np.deg2rad(60.0)
    Phi_init   = np.deg2rad(20.0)
    Incl_init  = np.deg2rad(-30.0)
    T_init     = 0.05
    Omega_init = 0.2
    parameter_prior_ranges = {
        "Theta zero": (np.deg2rad(10.0),  np.deg2rad(89.0)),
        "Phi zero":   (0.0,               2.0 * np.pi),
        "Inclination":(np.deg2rad(-89.0), np.deg2rad(89.0)),
        "Time":       (0.01,              0.5),
        "Omega":      (0.0,               1.0),
    }
    cache.update({
        "grid_best_Theta": float(Theta_init),
        "grid_best_Phi":   float(Phi_init),
        "grid_best_Incl":  float(Incl_init),
        "grid_best_T":     float(T_init),
        "grid_best_Omega": float(Omega_init),
        "grid_best_error": np.nan,
    })
    for key, (lo, hi) in parameter_prior_ranges.items():
        cache[f"prior_{key}_lo"] = float(lo)
        cache[f"prior_{key}_hi"] = float(hi)

# ============================================================
# 6. MCMC_grid（選配, 用 11 質心）
# ============================================================

if RUN_MCMC_GRID:
    print("\n[MCMC_grid] start (11 質心 fast likelihood)")
    ndim = 5
    labels_5d = ["Theta zero", "Phi zero", "Inclination", "Time", "Omega"]
    nwalkers, nsteps = 20, 15000

    center_vals = [Theta_init, Phi_init, Incl_init, T_init, Omega_init]
    sigma_vals  = [
        np.deg2rad(5.0),
        np.deg2rad(8.0),
        np.deg2rad(8.0),
        0.05 * T_init,
        0.10 * Omega_init,
    ]

    p0 = np.zeros((nwalkers, ndim))
    for j, key in enumerate(labels_5d):
        lo, hi = parameter_prior_ranges[key]
        prop = center_vals[j] + sigma_vals[j] * np.random.randn(nwalkers)
        prop = np.clip(prop, lo, hi)
        p0[:, j] = prop

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

    try:
        tau = sampler.get_autocorr_time()
        burnin = int(2 * np.nanmax(tau))
        thin   = max(1, int(0.1 * np.nanmin(tau)))
    except Exception as e:
        print("[MCMC_grid] tau failed, use default.", e)
        burnin, thin = 1000, 50

    flat = sampler.get_chain(discard=burnin, thin=thin, flat=True)

    # unwrap Phi
    phi_samples = flat[:, 1]
    phi_ref = Phi_init
    phi_wrapped = ((phi_samples - phi_ref + np.pi) % (2*np.pi)) - np.pi + phi_ref
    flat_wrapped = flat.copy()
    flat_wrapped[:, 1] = phi_wrapped

    q16, q50, q84 = np.percentile(flat_wrapped, [16, 50, 84], axis=0)
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
        summarize_1d_posterior(flat_wrapped[:, i], name)

    # corner plot（角度轉度）
    samples_plot = flat_wrapped.copy()
    for idx in [0, 1, 2]:
        samples_plot[:, idx] = np.rad2deg(samples_plot[:, idx])
    labels_plot = ["Theta zero (°)", "Phi zero (°)", "Inclination (°)",
                   "Time (Myr)", "Omega"]
    q16p, q50p, q84p = np.percentile(samples_plot, [16, 50, 84], axis=0)
    ranges = []
    for i in range(len(labels_plot)):
        lo, md, hi = q16p[i], q50p[i], q84p[i]
        width = hi - lo if hi > lo else 1e-3
        ranges.append((md - 1.5*width, md + 1.5*width))

    fig = corner.corner(samples_plot,
                        labels=labels_plot,
                        range=ranges,
                        show_titles=True,
                        title_fmt=".2f",
                        plot_datapoints=False,
                        fill_contours=True,
                        smooth=1.0)
    fig.savefig(os.path.join(PLOT_DIR, "corner_mcmc_grid.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)

    cache.update({
        "mcmc_grid_used": True,
        "mcmc_grid_median_Theta": float(Theta_med),
        "mcmc_grid_median_Phi":   float(Phi_med),
        "mcmc_grid_median_Incl":  float(Incl_med),
        "mcmc_grid_median_T":     float(T_med),
        "mcmc_grid_median_Omega": float(Omega_med),
    })
else:
    cache["mcmc_grid_used"] = False

# ============================================================
# 7. MCMC_distance / MCMC_3D（如有開，邏輯同上，這裡略）
#    你原本的實作可以保留，重點是最後把 median 寫進 cache
# ============================================================

# （此處省略 RUN_MCMC_DISTANCE, RUN_MCMC_3D 的程式碼，你可以直接沿用，
#  只要在完結時把 median 寫入：
#   cache["mcmc_distance_median_Theta"] = ...
#  並標記 cache["mcmc_distance_used"] = True
#  即可被下面 best-fit 選擇邏輯使用。）


# ============================================================
# 8. 決定最終 best-fit + 計算 RMSE + 寫 cache
# ============================================================

# 先決定優先順序：distance > grid_mcmc > grid
Theta_best = Phi_best = Incl_best = T_best = Omega_best = None

if cache.get("mcmc_distance_used", False):
    Theta_best = cache["mcmc_distance_median_Theta"]
    Phi_best   = cache["mcmc_distance_median_Phi"]
    Incl_best  = cache["mcmc_distance_median_Incl"]
    T_best     = cache["mcmc_distance_median_T"]
    Omega_best = cache["mcmc_distance_median_Omega"]
elif cache.get("mcmc_grid_used", False):
    Theta_best = cache["mcmc_grid_median_Theta"]
    Phi_best   = cache["mcmc_grid_median_Phi"]
    Incl_best  = cache["mcmc_grid_median_Incl"]
    T_best     = cache["mcmc_grid_median_T"]
    Omega_best = cache["mcmc_grid_median_Omega"]
else:
    Theta_best = cache["grid_best_Theta"]
    Phi_best   = cache["grid_best_Phi"]
    Incl_best  = cache["grid_best_Incl"]
    T_best     = cache["grid_best_T"]
    Omega_best = cache["grid_best_Omega"]

# 用 best-fit 評估一次 error_function（會更新 last_* RMSE if implemented）
rmse_combo = pss.error_function(
    [Theta_best, Phi_best],
    streamercom_x_AU,
    streamercom_z_AU,
    streamercom_v_LS_km,
    v_weight_phys,
    T_best, Omega_best, Incl_best,
    M_star,
    scale,
    log_power,
)

pos_rmse = getattr(pss.error_function, "last_pos_rmse", np.nan)
vel_rmse = getattr(pss.error_function, "last_vel_rmse", np.nan)
eq_rmse  = getattr(pss.error_function, "last_eq_vel_rmse", np.nan)

r_ref_AU = 200 * T_best * 1e6 * spc.year / spc.astronomical_unit

# M_0, Mdot（用 best-fit 的 T）
M_0 = M_star * M_SUN_KG * spc.G / (200.0**3 * T_best * 1e6 * spc.year)
M_dot = M_star / (T_best * 1e6)  # [M_sun / yr]，假設全星質量在 T 內累積

cache.update({
    "best_Theta": float(Theta_best),
    "best_Phi":   float(Phi_best),
    "best_Incl":  float(Incl_best),
    "best_T":     float(T_best),
    "best_Omega": float(Omega_best),
    "r_ref_AU": float(r_ref_AU),  
    "best_pos_RMSE_AU":  float(pos_rmse),
    "best_vel_RMSE_kms": float(vel_rmse),
    "best_eq_RMSE_kms":  float(eq_rmse),
    "best_M0":           float(M_0),
    "best_Mdot_Msun_per_yr": float(M_dot),
})

print("\n==================== Final Best-fit (Per-emb-50) ====================")
print(f"Theta        = {np.rad2deg(Theta_best):.3f} deg")
print(f"Phi          = {np.rad2deg(Phi_best):.3f} deg")
print(f"Inclination  = {np.rad2deg(Incl_best):.3f} deg")
print(f"Time (T_Myr) = {T_best:.6f} Myr")
print(f"Omega        = {Omega_best:.4f}")
print(f"r_ref        = {r_ref_AU:.3f} AU")
print(f"Position RMSE: {pos_rmse:.4f} AU")
print(f"Velocity RMSE: {vel_rmse:.4f} km/s")
print(f"Combined RMSE: {eq_rmse:.4f} km/s-equivalent")
print(f"M_0          = {M_0:.3e} (dimensionless)")
print(f"Mdot         = {M_dot:.3e} M_sun/yr")
print("====================================================================")

# 寫 cache
try:
    np.savez(CACHE_PATH, **cache)
    print(f"[cache] Saved results to {CACHE_PATH}")
except Exception as e:
    print(f"[cache] Failed to save cache: {e}")

# ============================================================
# 9. 用 cache best-fit 畫 overlay（同一份邏輯，供檢查）
# ============================================================

try:
    c = np.load(CACHE_PATH)
    print(f"[cache] Loaded for overlay: {CACHE_PATH}")

    Theta_best = c["best_Theta"]
    Phi_best   = c["best_Phi"]
    Incl_best  = c["best_Incl"]
    T_best     = c["best_T"]
    Omega_best = c["best_Omega"]

    Theta_best_deg = np.rad2deg(Theta_best)
    Phi_best_deg   = np.rad2deg(Phi_best)
    Incl_best_deg  = np.rad2deg(Incl_best)

    # streamer moment maps
    str_mom0 = fits.getdata("Per-emb-50_H2CO_streamer_mom0.fits")
    str_mom1 = fits.getdata("Per-emb-50_H2CO_streamer_mom1.fits")
    header   = fits.getheader("Per-emb-50_H2CO_streamer_mom1.fits")

    im_center = (int(header["CRPIX2"]), int(header["CRPIX1"]))
    dx_arcsec = abs(header["CDELT2"]) * 3600.0
    dx_au = dx_arcsec * distance_pc

    cen_x_pix = cen_z_pix = cen_v_LS = None
    if ("streamercom_x_AU" in c) and ("streamercom_z_AU" in c):
        sx = c["streamercom_x_AU"]
        sz = c["streamercom_z_AU"]
        x_rot = sx / dx_au
        z_rot = sz / dx_au
        cen_x_pix = x_rot * np.cos(pa_rad) - z_rot * np.sin(pa_rad) + im_center[1]
        cen_z_pix = x_rot * np.sin(pa_rad) + z_rot * np.cos(pa_rad) + im_center[0]
        if "streamercom_v_LS_km" in c:
            cen_v_LS = c["streamercom_v_LS_km"]

    plot_streamer_on_mom1(
        Theta_best_deg, Phi_best_deg, Incl_best_deg,
        float(T_best), float(Omega_best),
        header, pa_rad, dx_au, im_center,
        str_mom1,
        label="Per-emb-50 H2CO moment1 (best-fit)",
        outname="Per-emb-50_model_vs_mom1_overlay_best.png",
        cen_x_pix=cen_x_pix,
        cen_z_pix=cen_z_pix,
        cen_v_LS_km=cen_v_LS,
        v_range=1.0,
    )

    plot_streamer_on_mom0(
        Theta_best_deg, Phi_best_deg, Incl_best_deg,
        float(T_best), float(Omega_best),
        header, pa_rad, dx_au, im_center,
        str_mom0,
        label="Per-emb-50 H2CO moment0 (best-fit)",
        outname="Per-emb-50_model_vs_mom0_overlay_best.png",
        cen_x_pix=cen_x_pix,
        cen_z_pix=cen_z_pix,
        cen_v_LS_km=cen_v_LS,
    )

    print("[overlay] Generated best-fit overlay plots.")

except Exception as e:
    print(f"[overlay] Failed to generate overlay from cache: {e}")