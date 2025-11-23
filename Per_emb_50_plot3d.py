#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
讀取 Per-emb-50 的 cache + cube，
用 PSS_model 建立最佳參數的 model 流線，
並使用 shell / distance_cube 來產生 5 種 3D 互動圖：

1) Model + shell
2) Model + distance cube
3) Model + shell error (normalized)
4) Model + distance error (normalized)
5) Model + original data cube

座標軸：RA offset (arcsec), DEC offset (arcsec), LOS velocity (km/s)
輸出格式：HTML (Plotly，互動式)
"""

import os
import numpy as np
import scipy.constants as spc
from astropy.io import fits
import plotly.graph_objects as go

from PSSpy import (
    PSS_model,
    apply_shell_ball_to_line,
    grow_distance_cube_bounded,
    velocity_to_channel_index
)
from Per_emb_50 import _extract_params_from_cache, _resolve_cache_path


# ================== 路徑與基本設定 ==================

CACHE_DIR = "Per50_results/cache"
PLOT_DIR  = "Per50_results/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

CUBE_FNAME = "Per-emb-50_H2CO_streamer_cube.fits"

USE_CACHE_SOURCE = "mcmc_shell"
cache_path = _resolve_cache_path(USE_CACHE_SOURCE)

# 距離、質量、PA 等物理參數
LSR_velocity  = 7.5  # km/s
DISTANCE_PC   = 300.0        
M_STAR_MSUN   = 2.58         
M_SUN_KG      = 1.98847e30
R_IN_AU       = 2e2
R_OUT_AU      = 3.8e3
RESOLUTION    = 200
SCALE         = "log"
LOG_POWER     = 1.5
PA_DEG        = 170 + 90
PA_RAD        = np.deg2rad(PA_DEG)

# shell / distance cube 的參數
MAX_SHELL_R   = 30.0            # 球殼最大距離 (在 voxel space)，你可以調整
MAX_DIST_VAL  = 30.0            # distance_cube 的最大距離

# ================== 工具函式 ==================

def load_best_params(cache_path):
    """從 cache 讀取 best-fit 參數（弧度 & T_Myr, omega）"""
    c = np.load(cache_path, allow_pickle=True)

    Theta, Phi, Incl, T_Myr, Omega = _extract_params_from_cache(c, USE_CACHE_SOURCE)
    r_ref_AU = 200 * T_Myr * 1e6 * spc.year / spc.astronomical_unit

    # M_0, Mdot（用 best-fit 的 T）
    M_0 = M_STAR_MSUN * M_SUN_KG * spc.G / (200.0**3 * T_Myr * 1e6 * spc.year)
    M_dot = M_STAR_MSUN / (T_Myr * 1e6)  # [M_sun / yr]，假設全星質量在 T 內累積
    print(f"[info] Using parameters from {USE_CACHE_SOURCE}:")
    print("\n==================== Parameters ====================")
    print(f"Theta        = {np.rad2deg(Theta):.3f} deg")
    print(f"Phi          = {np.rad2deg(Phi):.3f} deg")
    print(f"Inclination  = {np.rad2deg(Incl):.3f} deg")
    print(f"Time (T_Myr) = {T_Myr:.6f} Myr")
    print(f"Omega        = {Omega:.4f}")
    print(f"r_ref        = {r_ref_AU:.3f} AU")
    print(f"M_0          = {M_0:.3e} (dimensionless)")
    print(f"Mdot         = {M_dot:.3e} M_sun/yr")
    print("====================================================================")
    return dict(
    Theta=Theta,
    Phi=Phi,
    Incl=Incl,
    T=T_Myr,
    Omega=Omega,
    Theta_deg=np.rad2deg(Theta),
    Phi_deg=np.rad2deg(Phi),
    Incl_deg=np.rad2deg(Incl),
    )


def pixel_to_world_arrays(x_pix, z_pix, v_pix, header):
    """
    把 0-based 的 (x_pix, z_pix, v_pix) 轉成：
        ra_off (arcsec), dec_off (arcsec), v_lsr (km/s)

    這裡假設 CDELT1, CDELT2, CDELT3, CRVAL3, CRPIX(1,2,3) 的單位已經是
    deg, deg, km/s, km/s, pixel (fits 1-based)。
    """
    x_pix = np.asarray(x_pix, dtype=float)
    z_pix = np.asarray(z_pix, dtype=float)
    v_pix = np.asarray(v_pix, dtype=float)

    CRPIX1 = float(header["CRPIX1"])
    CRPIX2 = float(header["CRPIX2"])
    CRPIX3 = float(header["CRPIX3"])
    CDELT1 = float(header["CDELT1"])  # deg / pix
    CDELT2 = float(header["CDELT2"])  # deg / pix
    CDELT3 = float(header["CDELT3"])  # km/s per channel
    CRVAL3 = float(header["CRVAL3"])  # km/s at CRPIX3

    # pixel index (0-based) → fits pixel (1-based)
    x_fits = x_pix + 1.0
    z_fits = z_pix + 1.0
    v_fits = v_pix + 1.0

    # offset in pixel relative to reference pixel
    dx_pix = x_fits - CRPIX1
    dz_pix = z_fits - CRPIX2
    dv_pix = v_fits - CRPIX3

    # offset in sky-axes (deg) & velocity (km/s)
    ra_off_deg  = dx_pix * CDELT1
    dec_off_deg = dz_pix * CDELT2
    v_lsr       = dv_pix * CDELT3 + CRVAL3

    ra_off_arcsec  = ra_off_deg  * 3600.0
    dec_off_arcsec = dec_off_deg * 3600.0

    return ra_off_arcsec, dec_off_arcsec, v_lsr


def build_model_line_in_cube(params, header, pa_rad, dx_au, lsr_vel):
    """
    使用 PSS_model 建立 3D 流線，並轉成：
      - 像素座標 (x_pix, z_pix, v_pix, 0-based integer)
      - sky 座標 (ra_off, dec_off, v_lsr)
    """
    nv = int(header["NAXIS3"])
    nz = int(header["NAXIS2"])
    nx = int(header["NAXIS1"])

    im_cy = float(header["CRPIX2"]) - 1.0  # 0-based
    im_cx = float(header["CRPIX1"]) - 1.0

    # 1) PSS_model in physical space
    x_m, y_m, z_m, u_m, v_m, w_m = PSS_model(
        params["Theta"], params["Phi"], params["Incl"],
        params["T"], params["Omega"],
        M_STAR_MSUN,
        radius_in_au=R_IN_AU,
        radius_out_au=R_OUT_AU,
        resolution=RESOLUTION,
        scale=SCALE,
        log_power=LOG_POWER,
    )

    # 2) rotate to image frame
    x_rot = x_m * np.cos(pa_rad) - z_m * np.sin(pa_rad)
    z_rot = x_m * np.sin(pa_rad) + z_m * np.cos(pa_rad)

    # 3) to pixel index
    x_pix = x_rot / dx_au + im_cx
    z_pix = z_rot / dx_au + im_cy

    v_lsr_model = v_m + lsr_vel
    # 用和其他地方一致的函式轉換
    v_pix = velocity_to_channel_index(v_lsr_model, header, nz=nv)
    
    # 5) keep only those inside cube
    valid = (
        (x_pix >= 0) & (x_pix < nx) &
        (z_pix >= 0) & (z_pix < nz) &
        (v_pix >= 0) & (v_pix < nv)
    )
    x_pix_int = np.round(x_pix[valid]).astype(int)
    z_pix_int = np.round(z_pix[valid]).astype(int)
    v_pix_int = np.round(v_pix[valid]).astype(int)
    x_pix = x_pix[valid]
    z_pix = z_pix[valid]
    v_pix = v_pix[valid]
    # 6) convert pixel → RA/DEC/vel
    ra_off, dec_off, v_lsr = pixel_to_world_arrays(x_pix, z_pix, v_pix, header)

    return dict(
        x_pix=x_pix,
        z_pix=z_pix,
        v_pix=v_pix,
        x_pix_int=x_pix_int,
        z_pix_int=z_pix_int,
        v_pix_int=v_pix_int,
        ra_off=ra_off,
        dec_off=dec_off,
        v_lsr=v_lsr,
    )

# ================== 主流程：計算 cubes ==================

def prepare_cubes_and_model():
    """讀取 cube, header, cache，並建立：
       - model line in sky coords
       - shell_cube
       - distance_cube
       - data cube
    """
    # 1) read cube
    data_cube = fits.getdata(CUBE_FNAME)
    header    = fits.getheader(CUBE_FNAME)
    nv, nz, nx = data_cube.shape

    # 2) spatial scale: dx_au
    dx_arcsec = abs(header["CDELT2"]) * 3600.0
    dx_au     = dx_arcsec * DISTANCE_PC
    CDELT3    = float(header["CDELT3"])   # km/s per channel
    dv        = abs(CDELT3)
    v_weight_phys = (dv / dx_au)**2

    # 3) best-fit params
    params = load_best_params(cache_path)

    # 4) model line in cube & sky coords
    model_line = build_model_line_in_cube(
        params,
        header,
        PA_RAD,
        dx_au,
        LSR_velocity,
    )

    # 5) shell_cube from model line
    shell_cube = apply_shell_ball_to_line(
        data_cube.shape,
        model_line["v_pix_int"],
        model_line["z_pix_int"],
        model_line["x_pix_int"],
        MAX_SHELL_R,
    )

    # 6) distance_cube from model
    distance_cube = grow_distance_cube_bounded(
        data_cube.shape,
        params["Theta"], params["Phi"], params["Incl"],
        params["T"], params["Omega"],
        PA_RAD, dx_au, header,
        Local_Standard_Velocity=LSR_velocity,
        v_weight=v_weight_phys,
        max_dist_value=MAX_DIST_VAL,
        M_star=M_STAR_MSUN,
        radius_in_au=R_IN_AU,
        radius_out_au=R_OUT_AU,
        scale=SCALE,
        log_power=LOG_POWER,
        bound=None,
    )

    return dict(
        data_cube=data_cube,
        header=header,
        params=params,
        model_line=model_line,
        shell_cube=shell_cube,
        distance_cube=distance_cube,
    )

# ================== 建立各種 3D 圖 ==================

def make_fig_model_plus_shell(info):
    """model + shell (用 shell index 當顏色)"""
    shell_cube = info["shell_cube"]
    header     = info["header"]

    mask_shell = shell_cube >= 1
    v_s, z_s, x_s = np.where(mask_shell)
    shell_k = shell_cube[mask_shell]

    ra_s, dec_s, vlsr_s = pixel_to_world_arrays(x_s, z_s, v_s, header)

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=ra_s, y=dec_s, z=vlsr_s,
        mode='markers',
        name='Shell',
        marker=dict(
            size=1,
            color=shell_k,
            colorscale='inferno',
            opacity=0.3,
            colorbar=dict(
                title="Shell index",
                len=0.4,
                x=1.0,
            ),
        ),
    ))

    fig.add_trace(go.Scatter3d(
        x=info["model_line"]["ra_off"],
        y=info["model_line"]["dec_off"],
        z=info["model_line"]["v_lsr"],
        mode='markers',
        name='Best-fit model',
        marker=dict(
            size=3.5,
            symbol='circle',
            line=dict(width=0.5, color='grey'),
            color=info["model_line"]["v_lsr"],
            colorscale='RdBu',
            opacity=0.9,
        ),
    ))

    _update_layout_common(fig, "Model + Shell (RA offset, DEC offset, v_LSR)")
    return fig


def make_fig_model_plus_distance(info):
    """model + distance cube (distance 值當顏色)"""
    distance_cube = info["distance_cube"]
    header        = info["header"]

    mask_dist = distance_cube > 0
    v_d, z_d, x_d = np.where(mask_dist)
    distances = distance_cube[mask_dist]

    ra_d, dec_d, vlsr_d = pixel_to_world_arrays(x_d, z_d, v_d, header)

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=ra_d, y=dec_d, z=vlsr_d,
        mode='markers',
        name='Distance points',
        marker=dict(
            size=1,
            color=distances,
            colorscale='inferno_r',
            opacity=0.1,
            cmax=30.0,
            cmin=0.0,
            colorbar=dict(
                title="Distance",
                len=0.4,
                x=1.0,
            ),
        ),
    ))

    fig.add_trace(go.Scatter3d(
        x=info["model_line"]["ra_off"],
        y=info["model_line"]["dec_off"],
        z=info["model_line"]["v_lsr"],
        mode='markers',
        name='Best-fit model',
        marker=dict(
            size=3.5,
            symbol='circle',
            line=dict(width=0.5, color='grey'),
            color=info["model_line"]["v_lsr"],
            colorscale='RdBu',
            opacity=0.9,
        ),
    ))

    _update_layout_common(fig, "Model + Distance cube (RA offset, DEC offset, v_LSR)")
    return fig


def make_fig_model_plus_shell_error(info):
    """
    model + shell error：
    只畫有觀測訊號 (data_cube > 0) 且 被貼上殼層 (shell_cube > 0) 的 voxel，
    顏色用 shell index（殼層編號）當作 error 大小。
    """
    data_cube  = info["data_cube"]
    shell_cube = info["shell_cube"]
    header     = info["header"]

    # 只取有資料 & 有殼層的點
    mask = (data_cube > 0) & (shell_cube > 0)
    if not np.any(mask):
        print("[shell_error] No voxels with data>0 and shell>0, only plot model.")
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=info["model_line"]["ra_off"],
            y=info["model_line"]["dec_off"],
            z=info["model_line"]["v_lsr"],
            mode='markers',
            name='Best-fit model',
            marker=dict(
                size=3.5,
                symbol='circle',
                line=dict(width=0.5, color='grey'),
                color=info["model_line"]["v_lsr"],
                colorscale='RdBu',
                opacity=0.9,
            ),
        ))
        _update_layout_common(fig, "Model + Shell error (RA offset, DEC offset, v_LSR)")
        return fig

    v_e, z_e, x_e = np.where(mask)
    shell_k = shell_cube[mask].astype(float)

    ra_e, dec_e, vlsr_e = pixel_to_world_arrays(x_e, z_e, v_e, header)

    fig = go.Figure()

    # data voxels 的殼層誤差
    fig.add_trace(go.Scatter3d(
        x=ra_e, y=dec_e, z=vlsr_e,
        mode='markers',
        name='Shell error (data voxels)',
        marker=dict(
            size=2,
            color=shell_k,
            colorscale='inferno',
            opacity=0.6,
            cmin=1.0,
            cmax=float(np.nanmax(shell_k)),
            colorbar=dict(
                title="Shell index (error)",
                len=0.4,
                x=1.0,
            ),
        ),
    ))

    # 疊上 model 線
    fig.add_trace(go.Scatter3d(
        x=info["model_line"]["ra_off"],
        y=info["model_line"]["dec_off"],
        z=info["model_line"]["v_lsr"],
        mode='markers',
        name='Best-fit model',
        marker=dict(
            size=3.5,
            symbol='circle',
            line=dict(width=0.5, color='grey'),
            color=info["model_line"]["v_lsr"],
            colorscale='RdBu',
            opacity=0.9,
        ),
    ))

    _update_layout_common(fig, "Model + Shell error (RA offset, DEC offset, v_LSR)")
    return fig


def make_fig_model_plus_distance_error(info):
    """
    model + distance error：
    對有觀測訊號 (data_cube > 0) 的 voxel，
    顏色畫它在 distance_cube 裡的原始距離（到 model streamline 的誤差）。
    """
    data_cube     = info["data_cube"]
    distance_cube = info["distance_cube"]
    header        = info["header"]

    mask = (data_cube > 0) & (distance_cube > 0)
    if not np.any(mask):
        print("[distance_error] No voxels with data>0 and distance>0, only plot model.")
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=info["model_line"]["ra_off"],
            y=info["model_line"]["dec_off"],
            z=info["model_line"]["v_lsr"],
            mode='markers',
            name='Best-fit model',
            marker=dict(
                size=3.5,
                symbol='circle',
                line=dict(width=0.5, color='grey'),
                color=info["model_line"]["v_lsr"],
                colorscale='RdBu',
                opacity=0.9,
            ),
        ))
        _update_layout_common(fig, "Model + Distance error (RA offset, DEC offset, v_LSR)")
        return fig

    v_e, z_e, x_e = np.where(mask)
    err_vals = distance_cube[mask].astype(float)   # 就是你要的原始距離

    ra_e, dec_e, vlsr_e = pixel_to_world_arrays(x_e, z_e, v_e, header)

    fig = go.Figure()

    # data voxels 的距離誤差
    fig.add_trace(go.Scatter3d(
        x=ra_e, y=dec_e, z=vlsr_e,
        mode='markers',
        name='Distance error (data voxels)',
        marker=dict(
            size=2,
            color=err_vals,
            colorscale='inferno',
            opacity=0.6,
            cmin=0.0,
            cmax=float(np.nanmax(err_vals)),
            colorbar=dict(
                title="Distance error",
                len=0.4,
                x=1.0,
            ),
        ),
    ))

    # 疊上 model 線
    fig.add_trace(go.Scatter3d(
        x=info["model_line"]["ra_off"],
        y=info["model_line"]["dec_off"],
        z=info["model_line"]["v_lsr"],
        mode='markers',
        name='Best-fit model',
        marker=dict(
            size=3.5,
            symbol='circle',
            line=dict(width=0.5, color='grey'),
            color=info["model_line"]["v_lsr"],
            colorscale='RdBu',
            opacity=0.9,
        ),
    ))

    _update_layout_common(fig, "Model + Distance error (RA offset, DEC offset, v_LSR)")
    return fig


def make_fig_model_plus_data(info):
    """model + 原始 data cube (>0 的 voxel)"""
    data_cube = info["data_cube"]
    header    = info["header"]

    mask_data = data_cube > 0
    v_d, z_d, x_d = np.where(mask_data)
    intensity = data_cube[mask_data]

    ra_d, dec_d, vlsr_d = pixel_to_world_arrays(x_d, z_d, v_d, header)

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=ra_d, y=dec_d, z=vlsr_d,
        mode='markers',
        name='Original data',
        marker=dict(
            size=1,
            color=vlsr_d,
            colorscale='RdBu',
            opacity=0.5,
        ),
    ))

    fig.add_trace(go.Scatter3d(
        x=info["model_line"]["ra_off"],
        y=info["model_line"]["dec_off"],
        z=info["model_line"]["v_lsr"],
        mode='markers',
        name='Best-fit model',
        marker=dict(
            size=3.5,
            symbol='circle',
            line=dict(width=0.5, color='grey'),
            color=info["model_line"]["v_lsr"],
            colorscale='RdBu',
            opacity=0.9,
            colorbar=dict(
                title="v_LSR (km/s)",
                len=0.4,
                x=1.0,
            ),
        ),
    ))

    _update_layout_common(fig, "Model + Observations (RA offset, DEC offset, v_LSR)")
    return fig


def _update_layout_common(fig, title):
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='RA offset (arcsec)',
            yaxis_title='DEC offset (arcsec)',
            zaxis_title='v_LSR (km/s)',
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )


# ================== 主入口：輸出 5 張 HTML ==================

def main():
    info = prepare_cubes_and_model()

    figs = [
        ("Per50_model_shell.html",            make_fig_model_plus_shell(info)),
        ("Per50_model_distance.html",         make_fig_model_plus_distance(info)),
        ("Per50_model_shell_error.html",      make_fig_model_plus_shell_error(info)),
        ("Per50_model_distance_error.html",   make_fig_model_plus_distance_error(info)),
        ("Per50_model_data.html",             make_fig_model_plus_data(info)),
    ]

    for fname, fig in figs:
        outpath = os.path.join(PLOT_DIR, fname)
        fig.write_html(outpath, include_plotlyjs="cdn", full_html=True)
        print(f"[save] {outpath}")

    # 如果你只想一次看某一張，也可以：
    # figs[0][1].show()


if __name__ == "__main__":
    main()