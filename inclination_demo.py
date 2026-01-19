#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inclination demo:
Visualize how the best-fit streamer changes with inclination angle.
Automatically loads fit parameters from each target's cache (.npz).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.collections import LineCollection
import matplotlib as mpl
import PSSpy as pss

# ============================================================
# 1. 基本設定
# ============================================================
TARGETS = ["Per-emb-2", "Per-emb-50", "SCrA"]   # 可改成你要畫的列表
TARGET_NAME = "Per-emb-2"                       # ← 選一個要畫的目標
CACHE_DIR = "."                                 # 假設 cache 在目前目錄
GIF_NAME = f"inclination_{TARGET_NAME}.gif"

FRAMES = 8
FPS = 2
R_vis = 9e3  # 視覺化用縮放範圍 (AU) 3e3(-50), 1.3e4(-2)

# ============================================================
# 2. 嘗試載入 cache 檔案
# ============================================================
# cache_path = os.path.join(CACHE_DIR, f"{TARGET_NAME}_fit_results.npz")

# if not os.path.exists(cache_path):
#     raise FileNotFoundError(f"[Error] Cache file not found: {cache_path}")

# cache = np.load(cache_path)
# print(f"[Loaded] {cache_path}")

# 從 cache 中讀取參數
# Theta_deg = float(cache["best_theta"])
# Phi_deg   = float(cache["best_phi"])
# Incl_deg  = float(cache["best_incl"])
# T_Myr     = float(cache["best_T"])
# omega     = float(cache["best_omega"])
Theta_deg = float(76.3)
Phi_deg   = float(167.3)
Incl_deg  = float(-48)
T_Myr     = float(0.279)
omega     = float(0.569)
# M_star    = float(cache.get("M_star", 3.2))  # 若沒存 M_star 則預設 3.2
# r_in_au   = float(cache.get("r_in_au", 2e2))
# r_out_au  = float(cache.get("r_out_au", 7e3))
M_star    = float(3.2)  # 若沒存 M_star 則預設 3.2
r_in_au   = float(2e3)
r_out_au  = float(7e3)
print(f"θ={Theta_deg:.2f}°, φ={Phi_deg:.2f}°, i={Incl_deg:.2f}°, "
      f"T={T_Myr:.3f} Myr, ω={omega:.3f}, M*={M_star:.2f} M☉")

# ============================================================
# 3. 準備動畫設定
# ============================================================
Theta_best = np.deg2rad(Theta_deg)
Phi_best   = np.deg2rad(Phi_deg)
Incl_best  = np.deg2rad(Incl_deg)

incl_start = 0.0
incl_end   = Incl_best

fig = plt.figure(figsize=(6, 6), dpi=200)
ax = fig.add_subplot(111, projection='3d')
cmap = mpl.cm.viridis


def make_sphere(R):
    """畫背景球 (視覺用)"""
    phi = np.linspace(0, np.pi, 40)
    theta = np.linspace(0, 2 * np.pi, 80)
    phi, theta = np.meshgrid(phi, theta)
    x = R * np.sin(phi) * np.cos(theta)
    y = R * np.sin(phi) * np.sin(theta)
    z = R * np.cos(phi)
    return x, y, z


sphere_x, sphere_y, sphere_z = make_sphere(R_vis)

# ============================================================
# 4. 更新函式（每一幀）
# ============================================================
def update(frame):
    ax.clear()

    # 線性插值 inclination
    if FRAMES > 1:
        t = frame / (FRAMES - 1)
    else:
        t = 1.0
    inc_now = incl_start + t * (incl_end - incl_start)

    # 呼叫 PSS 模型
    x_m, y_m, z_m, u_m, v_m, w_m = pss.PSS_model(
        Theta_best,
        Phi_best,
        inc_now,
        T_Myr,
        omega,
        M_star,
        radius_in_au=r_in_au,
        radius_out_au=r_out_au,
        resolution=400,
        scale="log",
        log_power=1.5,
    )
    PA_deg = 0
    pa_rad = np.deg2rad(PA_deg)

    x0 = x_m.copy()
    z0 = z_m.copy()

    # 把 model frame 轉回 observation：用 +PA
    x_m =  x0*np.cos(pa_rad) - z0*np.sin(pa_rad)
    z_m =  x0*np.sin(pa_rad) + z0*np.cos(pa_rad)
    
    # --- 畫背景球 ---
    ax.plot_surface(
        sphere_x, sphere_y, sphere_z,
        color="lightblue", alpha=0.08,
        edgecolor="gray", linewidth=0.2, zorder=0,
    )

    # --- 座標軸 ---
    L = R_vis * 1.2
    ax.plot([-L, L], [0, 0], [0, 0], color='black', linewidth=1)
    ax.text(L * 1.05, 0, 0, 'X', fontsize=9)
    ax.plot([0, 0], [-L, L], [0, 0], color='black', linewidth=1)
    ax.text(0, L * 1.05, 0, 'Y', fontsize=9)
    ax.plot([0, 0], [0, 0], [-L, L], color='black', linewidth=1)
    ax.text(0, 0, L * 1.05, 'Z', fontsize=9)

    # --- 中心星 ---
    ax.scatter(0, 0, 0, s=20, c='grey', depthshade=True)

    # --- Streamer ---
    if len(x_m) > 1:
        norm = mpl.colors.Normalize(vmin=np.nanmin(v_m), vmax=np.nanmax(v_m))
        colors = cmap(norm(v_m[:-1]))
        for i in range(len(x_m) - 1):
            ax.plot3D(
                x_m[i:i+2], y_m[i:i+2], z_m[i:i+2],
                color='C3', linewidth=3
            )

    # --- 標註文字 ---
    # ax.text(
    #     -0.9 * L, -0.9 * L, 0.9 * L,
    #     f"{TARGET_NAME}\n"
    #     f"Inclination = {np.rad2deg(inc_now):.0f}°",
    #     fontsize=8,
    #     bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.7),
    # )

    # --- 圖形設定 ---
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(R_vis, -R_vis)
    ax.set_ylim(R_vis, -R_vis)
    ax.set_zlim(-R_vis, R_vis)
    ax.axis("off")
    ax.view_init(elev=0, azim=90)


# ============================================================
# 5. 執行動畫並輸出
# ============================================================
ani = animation.FuncAnimation(fig, update, frames=FRAMES, interval=800, blit=False)
ani.save(GIF_NAME, writer="pillow", fps=FPS)
print(f"[Saved] {GIF_NAME}")